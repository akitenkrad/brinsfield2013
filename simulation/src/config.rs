//! Simulation configuration for Brinsfield (2013).
//!
//! Holds all knobs surfaced by the `run` / `sweep` / `ablate` CLI: team /
//! network shape, the initial six-motive distribution, the sign-constrained
//! motive-softmax `β` matrix (the rule-mode ablation), retaliation / shock
//! parameters, the EMA motive learning rate, and the LLM settings used when
//! `decision_mode == Llm`.

use serde::Serialize;

use crate::motives::MotiveVec6;

// --------------------------------------------------------------------------- //
// DecisionMode — LLM vs 6/4/3-dim rule ablations
// --------------------------------------------------------------------------- //

/// Decision-mechanism selector. The driver wires **exactly one** decision
/// mechanism (mutually exclusive). The three `rule_*` modes are the ablation
/// axis: a 6-motive softmax, a 4-motive (Knoll) collapse, and a 3-motive
/// (Van Dyne) collapse — all bit-deterministic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionMode {
    /// `voice_decision` — LLM-driven six-motive assignment.
    Llm,
    /// `voice_decision_rule_6dim` — six-motive softmax ablation.
    Rule6dim,
    /// `voice_decision_rule_4dim` — collapse to Knoll's 4 forms.
    Rule4dim,
    /// `voice_decision_rule_3dim` — collapse to Van Dyne's 3 forms.
    Rule3dim,
}

impl DecisionMode {
    /// Stable lowercase label (used in CSV / JSON / directory names).
    pub fn label(&self) -> &'static str {
        match self {
            DecisionMode::Llm => "llm",
            DecisionMode::Rule6dim => "rule_6dim",
            DecisionMode::Rule4dim => "rule_4dim",
            DecisionMode::Rule3dim => "rule_3dim",
        }
    }

    /// Whether this mode reaches the LLM layer.
    pub fn is_llm(&self) -> bool {
        matches!(self, DecisionMode::Llm)
    }

    /// Effective motive dimensionality for the rule modes (6 for LLM, which
    /// always emits a full six-motive vector).
    pub fn n_dims(&self) -> usize {
        match self {
            DecisionMode::Llm | DecisionMode::Rule6dim => 6,
            DecisionMode::Rule4dim => 4,
            DecisionMode::Rule3dim => 3,
        }
    }
}

/// Parse a [`DecisionMode`] from a CLI string.
pub fn parse_decision_mode(s: &str) -> Result<DecisionMode, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "llm" | "ollama" | "openai" => Ok(DecisionMode::Llm),
        "rule_6dim" | "rule6dim" | "rule6" | "6dim" => Ok(DecisionMode::Rule6dim),
        "rule_4dim" | "rule4dim" | "rule4" | "4dim" => Ok(DecisionMode::Rule4dim),
        "rule_3dim" | "rule3dim" | "rule3" | "3dim" => Ok(DecisionMode::Rule3dim),
        _ => Err(format!(
            "invalid decision-mode: \"{s}\" (llm / rule_6dim / rule_4dim / rule_3dim)"
        )),
    }
}

// --------------------------------------------------------------------------- //
// LLM settings (re-exported from socsim-llm)
// --------------------------------------------------------------------------- //

pub use socsim_llm::LlmSettings;

// --------------------------------------------------------------------------- //
// MotiveInit — initial six-motive marginal distribution
// --------------------------------------------------------------------------- //

/// Initial six-motive distribution used to seed each employee's `motive_vec`.
/// Values are clamped + L1-normalised at use time. Defaults track the design
/// doc's `motive-init` example `(0.35, 0.20, 0.13, 0.13, 0.13, 0.06)` — an
/// ineffectual-dominant prior with a defensive share near the 12.65% anchor.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct MotiveInit {
    pub ineffectual: f64,
    pub relational: f64,
    pub defensive: f64,
    pub diffident: f64,
    pub disengaged: f64,
    pub deviant: f64,
}

impl Default for MotiveInit {
    fn default() -> Self {
        MotiveInit {
            ineffectual: 0.35,
            relational: 0.20,
            defensive: 0.13,
            diffident: 0.13,
            disengaged: 0.13,
            deviant: 0.06,
        }
    }
}

impl MotiveInit {
    /// Parse a comma-separated 6-tuple `"i,r,def,dif,dis,dev"`.
    pub fn parse(s: &str) -> Result<MotiveInit, String> {
        let parts: Vec<f64> = s
            .split([',', ' '])
            .filter(|t| !t.is_empty())
            .map(|t| t.trim().parse::<f64>())
            .collect::<Result<_, _>>()
            .map_err(|e| format!("invalid motive-init list \"{s}\": {e}"))?;
        if parts.len() != 6 {
            return Err(format!(
                "motive-init needs exactly 6 values (got {})",
                parts.len()
            ));
        }
        Ok(MotiveInit {
            ineffectual: parts[0],
            relational: parts[1],
            defensive: parts[2],
            diffident: parts[3],
            disengaged: parts[4],
            deviant: parts[5],
        })
    }

    /// Normalised six-array `(ineff, rel, def, dif, dis, dev)`; degenerate
    /// inputs fall back to the default.
    pub fn normalised(&self) -> [f64; 6] {
        let raw = [
            self.ineffectual.max(0.0),
            self.relational.max(0.0),
            self.defensive.max(0.0),
            self.diffident.max(0.0),
            self.disengaged.max(0.0),
            self.deviant.max(0.0),
        ];
        let s: f64 = raw.iter().sum();
        if s <= 0.0 {
            let d = Self::default();
            return [
                d.ineffectual,
                d.relational,
                d.defensive,
                d.diffident,
                d.disengaged,
                d.deviant,
            ];
        }
        let mut out = [0.0; 6];
        for i in 0..6 {
            out[i] = raw[i] / s;
        }
        out
    }

    /// As a (normalised) [`MotiveVec6`].
    pub fn to_motive_vec(&self) -> MotiveVec6 {
        MotiveVec6::from_array(self.normalised())
    }
}

// --------------------------------------------------------------------------- //
// BetaGroup — sign-constrained coefficients
// --------------------------------------------------------------------------- //

/// Sign-constrained `β` group for the rule-mode voice / motive decision.
///
/// `voice_logit` mixes the contextual features into a Bernoulli VOICE/SILENCE
/// probability. The six motive rows weight `(ψ, f, ι, ρ, σ, n, e)` into a
/// softmax over the Brinsfield motives *conditional on SILENCE*. Row signs
/// follow the design doc §4.3 (e.g. `β_{defensive,ψ} < 0`, `β_{deviant,n} > 0`,
/// `β_{diffident,n} > 0`, `β_{diffident,ρ} > 0` for the spiral of silence).
#[derive(Debug, Clone, Copy, Serialize)]
pub struct BetaGroup {
    // ── VOICE logit (Bernoulli) ─────────────────────────────────────────────
    pub voice_intercept: f64,
    /// Coefficient on ψ (psychological safety) for VOICE (positive).
    pub beta_psafety: f64,
    /// Coefficient on `u` (supervisor openness) for VOICE (positive).
    pub beta_supervisor: f64,
    /// Coefficient on σ (issue salience) for VOICE (positive).
    pub beta_salience: f64,
    /// Coefficient on `f` (fear) — negative for VOICE.
    pub beta_fear: f64,
    /// Coefficient on ι (implicit voice theory) — negative for VOICE.
    pub beta_ivt: f64,
    /// Coefficient on ρ (perceived peer silence) — negative for VOICE.
    pub beta_rho: f64,

    // ── motive softmax magnitude (shared scale for the six rows) ────────────
    /// Global gain on the motive-row logits (sweepable; default 1.0).
    pub motive_gain: f64,
}

impl Default for BetaGroup {
    fn default() -> Self {
        BetaGroup {
            voice_intercept: 0.0,
            beta_psafety: 1.2,
            beta_supervisor: 0.6,
            beta_salience: 0.4,
            beta_fear: 1.5,
            beta_ivt: 0.8,
            beta_rho: 1.0,
            motive_gain: 1.0,
        }
    }
}

// --------------------------------------------------------------------------- //
// NetworkKind
// --------------------------------------------------------------------------- //

/// Inter-employee social-network family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NetworkKind {
    /// Watts–Strogatz small-world (default — design §4.3).
    WattsStrogatz,
    /// Erdős–Rényi G(n,p) — sensitivity.
    ErdosRenyi,
    /// Barabási–Albert preferential attachment — sensitivity.
    BarabasiAlbert,
}

/// Parse a [`NetworkKind`] from a CLI string.
pub fn parse_network_kind(s: &str) -> Result<NetworkKind, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "ws" | "watts-strogatz" | "small-world" => Ok(NetworkKind::WattsStrogatz),
        "er" | "erdos-renyi" | "erdos_renyi" => Ok(NetworkKind::ErdosRenyi),
        "ba" | "barabasi-albert" | "scale-free" => Ok(NetworkKind::BarabasiAlbert),
        _ => Err(format!(
            "invalid network kind: \"{s}\" (watts-strogatz / erdos-renyi / barabasi-albert)"
        )),
    }
}

// --------------------------------------------------------------------------- //
// Config
// --------------------------------------------------------------------------- //

/// Configuration for a single run.
#[derive(Debug, Clone)]
pub struct Config {
    // ── organisation shape ─────────────────────────────────────────────────
    pub n_teams: usize,
    pub team_size: usize,
    pub n_levels: u8,

    // ── network ────────────────────────────────────────────────────────────
    pub network_kind: NetworkKind,
    /// `k` for Watts–Strogatz / `m` for Barabási–Albert.
    pub network_k: usize,
    /// β for Watts–Strogatz / p for Erdős–Rényi.
    pub network_beta: f64,
    /// Supervisor-openness homogeneity `η_sup ∈ [0,1]` (1 = identical supervisors).
    pub supervisor_homogeneity: f64,

    // ── decision-mode switch ───────────────────────────────────────────────
    pub decision_mode: DecisionMode,
    /// Prompt template version for the LLM mode (`v1`/`v2`/`v3`).
    pub prompt_version: u8,

    // ── initial motive distribution + β group (rule ablation) ──────────────
    pub motive_init: MotiveInit,
    pub beta: BetaGroup,
    /// EMA learning rate η for `motive_dynamics`.
    pub motive_learn_rate: f64,
    /// Psychological-safety learning rate (η in `psafety_update`).
    pub psafety_learn: f64,

    // ── retaliation / shocks ───────────────────────────────────────────────
    pub p_retaliate: f64,
    pub shock_t: Option<u64>,
    pub shock_magnitude: f64,

    // ── horizon / repeats ──────────────────────────────────────────────────
    pub t_max: u64,
    pub runs: usize,
    pub seed: u64,

    // ── LLM settings (used iff `decision_mode == Llm`) ─────────────────────
    pub llm: LlmSettings,

    // ── output ─────────────────────────────────────────────────────────────
    pub output_dir: String,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            n_teams: 5,
            team_size: 8,
            n_levels: 3,
            network_kind: NetworkKind::WattsStrogatz,
            network_k: 6,
            network_beta: 0.1,
            supervisor_homogeneity: 0.0,
            decision_mode: DecisionMode::Rule6dim,
            prompt_version: 1,
            motive_init: MotiveInit::default(),
            beta: BetaGroup::default(),
            motive_learn_rate: 0.10,
            psafety_learn: 0.05,
            p_retaliate: 0.05,
            shock_t: None,
            shock_magnitude: 0.3,
            t_max: 48,
            runs: 1,
            seed: 42,
            llm: LlmSettings::default(),
            output_dir: "results".to_string(),
        }
    }
}

impl Config {
    /// Total number of employees.
    pub fn n_employees(&self) -> usize {
        self.n_teams.saturating_mul(self.team_size)
    }
}

/// JSON representation of a `run`'s `config.json`.
#[derive(Serialize)]
pub struct RunConfigJson {
    pub command: &'static str,
    pub n_teams: usize,
    pub team_size: usize,
    pub n_levels: u8,
    pub n_employees: usize,
    pub network_kind: NetworkKind,
    pub network_k: usize,
    pub network_beta: f64,
    pub supervisor_homogeneity: f64,
    pub decision_mode: DecisionMode,
    pub prompt_version: u8,
    pub motive_init: MotiveInit,
    pub beta: BetaGroup,
    pub motive_learn_rate: f64,
    pub psafety_learn: f64,
    pub p_retaliate: f64,
    pub shock_t: Option<u64>,
    pub shock_magnitude: f64,
    pub t_max: u64,
    pub runs: usize,
    pub seed: u64,
    pub llm_temperature: f32,
    pub llm_seed: u64,
    pub llm_cache_path: Option<String>,
    pub output_dir: String,
}

impl Config {
    /// Build the `config.json` representation.
    pub fn to_run_config_json(&self) -> RunConfigJson {
        RunConfigJson {
            command: "run",
            n_teams: self.n_teams,
            team_size: self.team_size,
            n_levels: self.n_levels,
            n_employees: self.n_employees(),
            network_kind: self.network_kind,
            network_k: self.network_k,
            network_beta: self.network_beta,
            supervisor_homogeneity: self.supervisor_homogeneity,
            decision_mode: self.decision_mode,
            prompt_version: self.prompt_version,
            motive_init: self.motive_init,
            beta: self.beta,
            motive_learn_rate: self.motive_learn_rate,
            psafety_learn: self.psafety_learn,
            p_retaliate: self.p_retaliate,
            shock_t: self.shock_t,
            shock_magnitude: self.shock_magnitude,
            t_max: self.t_max,
            runs: self.runs,
            seed: self.seed,
            llm_temperature: self.llm.temperature,
            llm_seed: self.llm.seed,
            llm_cache_path: self.llm.cache_path.clone(),
            output_dir: self.output_dir.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn motive_init_normalises() {
        let p = MotiveInit::parse("1,1,1,1,1,1").unwrap();
        let n = p.normalised();
        assert!((n.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        for v in n {
            assert!((v - 1.0 / 6.0).abs() < 1e-12);
        }
    }

    #[test]
    fn motive_init_zero_falls_back() {
        let p = MotiveInit::parse("0,0,0,0,0,0").unwrap();
        let n = p.normalised();
        assert!(n.iter().sum::<f64>() > 0.0);
    }

    #[test]
    fn motive_init_wrong_arity_errors() {
        assert!(MotiveInit::parse("0.3,0.3,0.4").is_err());
    }

    #[test]
    fn parse_decision_mode_variants() {
        assert_eq!(parse_decision_mode("llm").unwrap(), DecisionMode::Llm);
        assert_eq!(
            parse_decision_mode("rule_6dim").unwrap(),
            DecisionMode::Rule6dim
        );
        assert_eq!(
            parse_decision_mode("rule4").unwrap(),
            DecisionMode::Rule4dim
        );
        assert_eq!(parse_decision_mode("3dim").unwrap(), DecisionMode::Rule3dim);
        assert!(parse_decision_mode("bogus").is_err());
    }

    #[test]
    fn decision_mode_dims() {
        assert_eq!(DecisionMode::Rule6dim.n_dims(), 6);
        assert_eq!(DecisionMode::Rule4dim.n_dims(), 4);
        assert_eq!(DecisionMode::Rule3dim.n_dims(), 3);
        assert_eq!(DecisionMode::Llm.n_dims(), 6);
    }

    #[test]
    fn parse_network_kind_variants() {
        assert_eq!(
            parse_network_kind("watts-strogatz").unwrap(),
            NetworkKind::WattsStrogatz
        );
        assert_eq!(parse_network_kind("ER").unwrap(), NetworkKind::ErdosRenyi);
        assert_eq!(
            parse_network_kind("ba").unwrap(),
            NetworkKind::BarabasiAlbert
        );
    }
}
