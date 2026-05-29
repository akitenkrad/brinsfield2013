//! Integration tests for the Brinsfield (2013) six-motive silence simulation.
//!
//! **No live LLM required.** Rule modes need no LLM; the LLM path is driven by
//! `socsim_llm::mock::ScriptedClient`. Tests cover rule-mode bit-determinism,
//! the 6/4/3-dim ablation invariants, and an LLM-mode end-to-end via a scripted
//! client with an in-memory cache.

use brinsfield_silence_simulation::config::{Config, DecisionMode, LlmSettings, MotiveInit};
use brinsfield_silence_simulation::llm::{wrap_client, SilenceClient};
use brinsfield_silence_simulation::simulation::{run_with_client, SimulationResult};

use socsim_llm::mock::ScriptedClient;
use socsim_llm::PromptCache;

fn small_cfg(mode: DecisionMode) -> Config {
    Config {
        n_teams: 3,
        team_size: 6,
        n_levels: 2,
        network_k: 4,
        t_max: 10,
        runs: 1,
        seed: 1234,
        decision_mode: mode,
        motive_init: MotiveInit::default(),
        llm: LlmSettings::default(),
        output_dir: "results".to_string(),
        ..Config::default()
    }
}

/// Scripted client emitting one of three canonical six-motive JSON responses.
fn scripted_client() -> SilenceClient {
    let backend = ScriptedClient::new("mock-model", |prompt: &str| match prompt.len() % 3 {
        0 => r#"{"decision":"voice","motives":null,"rationale":"speak"}"#.to_string(),
        1 => r#"{"decision":"silence","motives":{"ineffectual":0.5,"relational":0.1,
                    "defensive":0.13,"diffident":0.1,"disengaged":0.1,"deviant":0.07},
                    "rationale":"pointless"}"#
            .to_string(),
        _ => r#"{"decision":"silence","motives":{"ineffectual":0.2,"relational":0.2,
                    "defensive":0.4,"diffident":0.1,"disengaged":0.05,"deviant":0.05},
                    "rationale":"fear"}"#
            .to_string(),
    });
    wrap_client(backend, PromptCache::in_memory())
}

// --------------------------------------------------------------------------- //
// Rule-mode determinism + invariants
// --------------------------------------------------------------------------- //

#[test]
fn rule6_smoke_run() {
    let r: SimulationResult = run_with_client(&small_cfg(DecisionMode::Rule6dim), None).unwrap();
    assert!(!r.metrics_rows.is_empty());
    assert_eq!(r.metadata.total(), 0, "rule mode makes 0 LLM calls");
    for row in &r.metrics_rows {
        let s = row.motive_mix_ineffectual
            + row.motive_mix_relational
            + row.motive_mix_defensive
            + row.motive_mix_diffident
            + row.motive_mix_disengaged
            + row.motive_mix_deviant;
        assert!(s.abs() < 1e-9 || (s - 1.0).abs() < 1e-9);
        assert!((0.0..=1.0).contains(&row.silence_rate));
        assert!((0.0..=1.0).contains(&row.climate_of_silence));
    }
    assert_eq!(r.correlation_rows.len(), 6 * 5);
}

#[test]
fn rule6_is_bit_deterministic() {
    let a = run_with_client(&small_cfg(DecisionMode::Rule6dim), None).unwrap();
    let b = run_with_client(&small_cfg(DecisionMode::Rule6dim), None).unwrap();
    assert_eq!(a.metrics_rows.len(), b.metrics_rows.len());
    for (ra, rb) in a.metrics_rows.iter().zip(b.metrics_rows.iter()) {
        assert_eq!(ra.t, rb.t);
        assert!((ra.silence_rate - rb.silence_rate).abs() < 1e-15);
        assert!((ra.motive_mix_ineffectual - rb.motive_mix_ineffectual).abs() < 1e-15);
        assert!((ra.motive_mix_defensive - rb.motive_mix_defensive).abs() < 1e-15);
        assert!((ra.motive_vec_mean_diffident - rb.motive_vec_mean_diffident).abs() < 1e-15);
        assert!((ra.kl_to_reference - rb.kl_to_reference).abs() < 1e-15);
    }
    // Agent end-state must be byte-identical too.
    assert_eq!(a.agent_rows.len(), b.agent_rows.len());
    for (ra, rb) in a.agent_rows.iter().zip(b.agent_rows.iter()) {
        assert_eq!(ra.expression, rb.expression);
        assert_eq!(ra.primary_motive, rb.primary_motive);
        assert!((ra.motive_defensive - rb.motive_defensive).abs() < 1e-15);
    }
}

#[test]
fn rule_4dim_collapses_diffident_disengaged_primaries() {
    // In 4-dim mode no silent agent should ever have diffident/disengaged as a
    // *primary* motive (their mass is folded into ineffectual/relational).
    let r = run_with_client(&small_cfg(DecisionMode::Rule4dim), None).unwrap();
    for row in &r.metrics_rows {
        assert!(row.motive_mix_diffident.abs() < 1e-12);
        assert!(row.motive_mix_disengaged.abs() < 1e-12);
    }
}

#[test]
fn rule_3dim_collapses_deviant_primary() {
    // In 3-dim (Van Dyne) mode deviant is folded into defensive; diffident /
    // disengaged into the acquiescent (ineffectual) carrier.
    let r = run_with_client(&small_cfg(DecisionMode::Rule3dim), None).unwrap();
    for row in &r.metrics_rows {
        assert!(row.motive_mix_deviant.abs() < 1e-12);
        assert!(row.motive_mix_diffident.abs() < 1e-12);
        assert!(row.motive_mix_disengaged.abs() < 1e-12);
    }
}

// --------------------------------------------------------------------------- //
// LLM-mode end-to-end (mock; no live LLM)
// --------------------------------------------------------------------------- //

#[test]
fn llm_mode_smoke_run_with_scripted_client() {
    let cfg = small_cfg(DecisionMode::Llm);
    let r = run_with_client(&cfg, Some(scripted_client())).unwrap();
    assert!(!r.metrics_rows.is_empty());
    assert!(r.metadata.total() > 0, "LLM mode must call the LLM");
    for row in &r.metrics_rows {
        let s = row.motive_mix_ineffectual
            + row.motive_mix_relational
            + row.motive_mix_defensive
            + row.motive_mix_diffident
            + row.motive_mix_disengaged
            + row.motive_mix_deviant;
        assert!(s.abs() < 1e-9 || (s - 1.0).abs() < 1e-9);
        // motive_vec_mean is always a normalised simplex (silent → soft vecs).
        let m = row.motive_vec_mean_ineffectual
            + row.motive_vec_mean_relational
            + row.motive_vec_mean_defensive
            + row.motive_vec_mean_diffident
            + row.motive_vec_mean_disengaged
            + row.motive_vec_mean_deviant;
        assert!((m - 1.0).abs() < 1e-9, "motive_vec_mean sum = {m}");
    }
}
