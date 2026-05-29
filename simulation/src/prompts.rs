//! LLM prompt construction and response parsing for `voice_decision`.
//!
//! The LLM is asked, given an employee's local context and the Brinsfield
//! six-motive operational definitions, to return a JSON decision:
//!
//! ```json
//! {
//!   "decision": "voice" | "silence",
//!   "motives": {"ineffectual": .., "relational": .., "defensive": ..,
//!               "diffident": .., "disengaged": .., "deviant": ..},
//!   "rationale": "short reason"
//! }
//! ```
//!
//! When `decision = silence`, `motives` is a (soft) distribution over the six
//! Brinsfield motives. When `decision = voice`, `motives` is `null` / omitted
//! and the verdict carries a uniform vector (no motive is assigned to VOICE).
//! Parse failures fall back to `Silence + uniform` with `parse_failed = true`.
//!
//! Three prompt versions (`v1`/`v2`/`v3`) progressively elaborate the
//! operational definitions (pre-registration prompt-sensitivity analysis,
//! design §6).

use serde_json::Value;

use crate::motives::{MotiveLabel, MotiveVec6};
use crate::world::{Employee, Expression, SilenceWorld};

// --------------------------------------------------------------------------- //
// Brinsfield motive operational definitions (Table-3 abstracted)
// --------------------------------------------------------------------------- //

/// One-line operational definition per motive (used in every prompt version).
pub const MOTIVE_DEFINITIONS: [(&str, &str); 6] = [
    (
        "ineffectual",
        "speaking up would make no difference — nothing ever changes",
    ),
    (
        "relational",
        "staying silent to preserve relationships or spare others' feelings",
    ),
    (
        "defensive",
        "staying silent out of fear of negative personal consequences",
    ),
    (
        "diffident",
        "staying silent from a lack of confidence in oneself or one's view",
    ),
    (
        "disengaged",
        "staying silent because of detachment or withdrawal from the issue",
    ),
    (
        "deviant",
        "withholding to harm, sabotage, or gain at the organisation's expense",
    ),
];

// --------------------------------------------------------------------------- //
// Prompt construction
// --------------------------------------------------------------------------- //

/// Build the voice-decision prompt for `agent_id`, embedding the Brinsfield
/// operational definitions at the requested `prompt_version` (1/2/3).
pub fn build_silence_prompt(
    world: &SilenceWorld,
    agent_id: socsim_core::AgentId,
    prompt_version: u8,
) -> String {
    let emp = &world.employees[&agent_id];
    let team = &world.teams[emp.team];
    let rho = world.neighbour_silence_ratio(agent_id);
    let sigma = world.issue_salience;

    let context = format_context(emp, team.supervisor_openness, sigma, rho);
    let defs = motive_definition_block(prompt_version);

    format!(
        "You are an employee at work. An ethically or operationally important \
         issue has arisen and you must decide whether to SPEAK UP (voice) or \
         REMAIN SILENT.\n\n\
         Your inner state:\n\
         {context}\n\
         Around you, {neigh_pct:.0}% of colleagues are currently silent.\n\n\
         If you remain silent, your reason may be a blend of these six motives:\n\
         {defs}\n\n\
         Reply with a SINGLE JSON object on one line:\n\
         {{\"decision\": \"voice\" | \"silence\", \
            \"motives\": {{\"ineffectual\": p1, \"relational\": p2, \"defensive\": p3, \
                          \"diffident\": p4, \"disengaged\": p5, \"deviant\": p6}}, \
            \"rationale\": \"short reason\"}}\n\
         Rules: if decision = voice, set \"motives\" to null. If decision = silence, \
         the six motive numbers must be non-negative and sum to about 1. Output JSON only.",
        neigh_pct = rho * 100.0,
    )
}

fn motive_definition_block(version: u8) -> String {
    let mut out = String::new();
    for (name, def) in MOTIVE_DEFINITIONS {
        match version {
            // v1: bare name + short gloss.
            1 => out.push_str(&format!("  - {name}: {def}\n")),
            // v2: emphasise that defensive (fear) is only one of six.
            2 => out.push_str(&format!(
                "  - {name} — {def}. Consider all six independently; fear is just one.\n"
            )),
            // v3: add the Brinsfield framing that most silence is NOT fear-based.
            _ => out.push_str(&format!(
                "  - {name} — {def}. (Empirically, fear-based 'defensive' silence is the \
                 minority of cases; weigh the non-fear motives fairly.)\n"
            )),
        }
    }
    out
}

fn format_context(emp: &Employee, supervisor_openness: f64, sigma: f64, rho: f64) -> String {
    format!(
        "  fear of consequences      f = {f:.2}\n\
         \x20 psychological safety      ψ = {psi:.2}\n\
         \x20 implicit-voice theory     ι = {iota:.2}\n\
         \x20 neuroticism               n = {n:.2}\n\
         \x20 extraversion              e = {e:.2}\n\
         \x20 supervisor openness       u = {u:+.2}\n\
         \x20 issue salience            σ = {sigma:.2}\n\
         \x20 perceived peer silence    ρ = {rho:.2}\n",
        f = emp.fear,
        psi = emp.psych_safety,
        iota = emp.ivt_strength,
        n = emp.neuroticism,
        e = emp.extraversion,
        u = supervisor_openness,
        sigma = sigma,
        rho = rho,
    )
}

// --------------------------------------------------------------------------- //
// Response parsing
// --------------------------------------------------------------------------- //

/// Parsed voice-decision verdict.
#[derive(Debug, Clone, PartialEq)]
pub struct VoiceDecisionVerdict {
    pub expression: Expression,
    /// Six-motive distribution (uniform for VOICE).
    pub motive_vec: MotiveVec6,
    pub rationale: String,
    /// True if the response failed to parse and we fell back to the
    /// `Silence + uniform` default.
    pub parse_failed: bool,
}

/// Parse an LLM response into a verdict.
///
/// Lenient: extracts the first balanced `{...}` substring, accepts mixed-case
/// labels, and falls back to `Silence + uniform` (parse_failed = true).
pub fn parse_voice_decision(text: &str) -> VoiceDecisionVerdict {
    let fallback = VoiceDecisionVerdict {
        expression: Expression::Silence,
        motive_vec: MotiveVec6::uniform(),
        rationale: String::new(),
        parse_failed: true,
    };

    let json_str = match extract_json_object(text) {
        Some(s) => s,
        None => return fallback,
    };
    let val: Value = match serde_json::from_str(&json_str) {
        Ok(v) => v,
        Err(_) => return fallback,
    };

    let decision = val
        .get("decision")
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_ascii_lowercase())
        .unwrap_or_default();
    let rationale = val
        .get("rationale")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();

    let expression = match decision.as_str() {
        "voice" | "speak" | "speak_up" => Expression::Voice,
        "silence" | "silent" | "withhold" => Expression::Silence,
        _ => {
            return VoiceDecisionVerdict {
                expression: Expression::Silence,
                motive_vec: MotiveVec6::uniform(),
                rationale,
                parse_failed: true,
            }
        }
    };

    if expression == Expression::Voice {
        return VoiceDecisionVerdict {
            expression,
            motive_vec: MotiveVec6::uniform(),
            rationale,
            parse_failed: false,
        };
    }

    // SILENCE: parse the six-motive distribution.
    match parse_motives(val.get("motives")) {
        Some(mut mv) => {
            mv.normalize();
            VoiceDecisionVerdict {
                expression,
                motive_vec: mv,
                rationale,
                parse_failed: false,
            }
        }
        None => VoiceDecisionVerdict {
            expression,
            motive_vec: MotiveVec6::uniform(),
            rationale,
            parse_failed: true,
        },
    }
}

/// Parse a `"motives"` object into a (pre-normalisation) [`MotiveVec6`].
/// Accepts either a 6-key object or a 6-element array (canonical order).
fn parse_motives(v: Option<&Value>) -> Option<MotiveVec6> {
    let v = v?;
    if v.is_null() {
        return None;
    }
    if let Some(obj) = v.as_object() {
        let mut a = [0.0; 6];
        let mut found = false;
        for (k, val) in obj {
            if let Some(label) = MotiveLabel::parse(k) {
                if let Some(x) = val.as_f64() {
                    a[label.index()] = x.max(0.0);
                    found = true;
                }
            }
        }
        if found {
            return Some(MotiveVec6::from_array(a));
        }
        return None;
    }
    if let Some(arr) = v.as_array() {
        if arr.len() == 6 {
            let mut a = [0.0; 6];
            for (i, item) in arr.iter().enumerate() {
                a[i] = item.as_f64().unwrap_or(0.0).max(0.0);
            }
            return Some(MotiveVec6::from_array(a));
        }
    }
    None
}

/// Extract the first balanced `{...}` substring from `text`.
fn extract_json_object(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let start = bytes.iter().position(|&b| b == b'{')?;
    let mut depth = 0i32;
    let mut in_str = false;
    let mut esc = false;
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if in_str {
            if esc {
                esc = false;
            } else if b == b'\\' {
                esc = true;
            } else if b == b'"' {
                in_str = false;
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(text[start..=i].to_string());
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::Team;

    #[test]
    fn parses_canonical_voice() {
        let v =
            parse_voice_decision(r#"{"decision": "voice", "motives": null, "rationale": "ok"}"#);
        assert_eq!(v.expression, Expression::Voice);
        assert!(!v.parse_failed);
        // VOICE → uniform vector.
        for p in v.motive_vec.to_array() {
            assert!((p - 1.0 / 6.0).abs() < 1e-12);
        }
    }

    #[test]
    fn parses_silence_with_six_motives() {
        let v = parse_voice_decision(
            r#"{"decision":"silence","motives":{"ineffectual":0.5,"relational":0.1,
               "defensive":0.1,"diffident":0.1,"disengaged":0.1,"deviant":0.1},
               "rationale":"pointless"}"#,
        );
        assert_eq!(v.expression, Expression::Silence);
        assert!(!v.parse_failed);
        assert_eq!(v.motive_vec.primary(), MotiveLabel::Ineffectual);
        let s: f64 = v.motive_vec.to_array().iter().sum();
        assert!((s - 1.0).abs() < 1e-9);
    }

    #[test]
    fn tolerates_surrounding_text_and_array_motives() {
        let v = parse_voice_decision(
            r#"Sure: {"decision":"silence","motives":[0,0,1,0,0,0],"rationale":"fear"}. Done."#,
        );
        assert_eq!(v.motive_vec.primary(), MotiveLabel::Defensive);
    }

    #[test]
    fn unknown_decision_falls_back_uniform() {
        let v = parse_voice_decision(r#"{"decision":"???","motives":null}"#);
        assert!(v.parse_failed);
        assert_eq!(v.expression, Expression::Silence);
    }

    #[test]
    fn silence_without_motives_falls_back() {
        let v = parse_voice_decision(r#"{"decision":"silence","rationale":"x"}"#);
        assert!(v.parse_failed);
        for p in v.motive_vec.to_array() {
            assert!((p - 1.0 / 6.0).abs() < 1e-12);
        }
    }

    #[test]
    fn no_json_falls_back() {
        let v = parse_voice_decision("no json here");
        assert!(v.parse_failed);
    }

    #[test]
    fn prompt_versions_differ() {
        use socsim_core::{AgentId, SimClock, SimRng};
        use socsim_net::SocialNetwork;
        use std::collections::BTreeMap;
        let mut rng = SimRng::from_seed(0);
        let ids: Vec<AgentId> = (0..3).map(|i| AgentId(i as u64)).collect();
        let net = SocialNetwork::watts_strogatz(&ids, 2, 0.1, &mut rng);
        let mut emps = BTreeMap::new();
        for &id in &ids {
            emps.insert(id, Employee::neutral(0, 0, 0));
        }
        let w = SilenceWorld::new(SimClock::new(1), emps, vec![Team::default()], net);
        let p1 = build_silence_prompt(&w, AgentId(0), 1);
        let p3 = build_silence_prompt(&w, AgentId(0), 3);
        assert_ne!(p1, p3);
        assert!(p1.contains("defensive"));
    }
}
