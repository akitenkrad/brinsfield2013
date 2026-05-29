//! Initialisation + run driver for the Brinsfield (2013) simulation.
//!
//! Two-layer determinism (design §4.3 RNG streams):
//! - `derive_seed(root, &[0])` — world init (employees + Watts–Strogatz network).
//! - `derive_seed(root, &[1])` — engine (scheduler / Bernoulli draws in rule mode).
//! - `derive_seed(root, &[2])` — retaliation stream (folded into the engine RNG).
//! - `derive_seed(root, &[3, agent_id, t])` — LLM `(agent, t)` seed.
//!
//! The rule modes are bit-reproducible; the LLM mode is pseudo-determinised by
//! `temperature=0` + the `(agent, t)` seed + the prompt→response cache.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use rand::Rng;
use serde::Serialize;
use socsim_core::{derive_seed, AgentId, SimClock, SimRng};
use socsim_engine::{RandomActivationScheduler, SimulationBuilder};
use socsim_llm::{LlmClient, MetadataCollector};
use socsim_net::SocialNetwork;

use crate::config::{Config, DecisionMode, NetworkKind};
use crate::llm::{build_live_client, SilenceClient};
use crate::mechanisms::{
    ClimateSilence, FearAppraisal, IssueSalience, MotiveDynamics, MotiveMetrics, OrgPerformance,
    PrefalseCascade, PsafetyUpdate, RetaliationEvent, SharedClient, SharedMetadata, SilenceSpiral,
    VoiceDecisionLlm, VoiceDecisionRule,
};
use crate::metrics::{
    climate_of_silence, defensive_share, kl_to_reference, motive_mix, motive_vec_mean,
    n_distinct_motives_active, pearson, silence_rate,
};
use crate::motives::MotiveLabel;
use crate::world::{Employee, Expression, SilenceWorld, Team};

/// RNG stream label: world init.
pub const RNG_WORLD_INIT: u64 = 0;
/// RNG stream label: socsim engine.
pub const RNG_ENGINE: u64 = 1;
/// RNG stream label: LLM `(agent, t)` seed root.
pub const RNG_PROMPT_ROOT: u64 = 3;

// --------------------------------------------------------------------------- //
// Result containers + rows
// --------------------------------------------------------------------------- //

/// Per-step metrics row written to `metrics.csv`.
#[derive(Debug, Clone, Serialize)]
pub struct MetricsRow {
    pub t: u64,
    pub silence_rate: f64,
    pub motive_mix_ineffectual: f64,
    pub motive_mix_relational: f64,
    pub motive_mix_defensive: f64,
    pub motive_mix_diffident: f64,
    pub motive_mix_disengaged: f64,
    pub motive_mix_deviant: f64,
    pub motive_vec_mean_ineffectual: f64,
    pub motive_vec_mean_relational: f64,
    pub motive_vec_mean_defensive: f64,
    pub motive_vec_mean_diffident: f64,
    pub motive_vec_mean_disengaged: f64,
    pub motive_vec_mean_deviant: f64,
    pub n_distinct_motives_active: usize,
    pub climate_of_silence: f64,
    pub org_performance: f64,
    pub issue_salience: f64,
    pub kl_to_reference: f64,
}

/// Per-(t, ineffectual..deviant) motive_mix row written to `motive_mix.csv`.
#[derive(Debug, Clone, Serialize)]
pub struct MotiveMixRow {
    pub t: u64,
    pub ineffectual: f64,
    pub relational: f64,
    pub defensive: f64,
    pub diffident: f64,
    pub disengaged: f64,
    pub deviant: f64,
}

/// Per-agent end-of-run state row written to `agents.csv`.
#[derive(Debug, Clone, Serialize)]
pub struct AgentRow {
    pub t: u64,
    pub agent_id: u64,
    pub team: usize,
    pub level: u8,
    pub tenure: u32,
    pub expression: String,
    pub primary_motive: String,
    pub motive_ineffectual: f64,
    pub motive_relational: f64,
    pub motive_defensive: f64,
    pub motive_diffident: f64,
    pub motive_disengaged: f64,
    pub motive_deviant: f64,
    pub fear: f64,
    pub psafety: f64,
    pub ivt: f64,
    pub neuroticism: f64,
    pub extraversion: f64,
    pub private_concern: f64,
}

/// Per-(motive, correlate) correlation row written to `correlations.csv`.
#[derive(Debug, Clone, Serialize)]
pub struct CorrelationRow {
    pub motive: String,
    pub correlate: String,
    pub pearson_r: f64,
}

/// Result of a single run.
pub struct SimulationResult {
    pub final_round: u64,
    pub world: SilenceWorld,
    pub metrics_rows: Vec<MetricsRow>,
    pub motive_mix_rows: Vec<MotiveMixRow>,
    pub agent_rows: Vec<AgentRow>,
    pub correlation_rows: Vec<CorrelationRow>,
    pub metadata: MetadataCollector,
    pub llm_model: String,
    pub llm_endpoint: String,
}

// --------------------------------------------------------------------------- //
// World initialisation
// --------------------------------------------------------------------------- //

/// Initialise a [`SilenceWorld`] with per-employee attributes from `rng`.
pub fn init_world(cfg: &Config, rng: &mut SimRng) -> SilenceWorld {
    let n = cfg.n_employees();
    let init_mv = cfg.motive_init.to_motive_vec();
    let mut employees: BTreeMap<AgentId, Employee> = BTreeMap::new();
    for i in 0..n {
        let team = i / cfg.team_size;
        let level = (i % cfg.n_levels.max(1) as usize) as u8;
        let tenure: u32 = rng.gen_range(1..120);
        let mut e = Employee::neutral(team, level, tenure);
        e.fear = rng.gen::<f64>().clamp(0.0, 1.0) * 0.6;
        e.psych_safety = (0.3 + 0.5 * rng.gen::<f64>()).clamp(0.0, 1.0);
        e.ivt_strength = rng.gen::<f64>().clamp(0.0, 1.0) * 0.6;
        e.neuroticism = rng.gen::<f64>().clamp(0.0, 1.0);
        e.extraversion = rng.gen::<f64>().clamp(0.0, 1.0);
        e.private_concern = rng.gen_range(-1.0..1.0);
        e.voice_threshold = (0.4 + 0.3 * rng.gen::<f64>()).clamp(0.0, 1.0);
        e.motive_vec = init_mv;
        employees.insert(AgentId(i as u64), e);
    }

    // Teams: supervisor openness interpolated toward a common value by η_sup.
    let common = rng.gen_range(-0.5..0.7);
    let mut teams = Vec::with_capacity(cfg.n_teams);
    for _ in 0..cfg.n_teams {
        let indiv = rng.gen_range(-0.5..0.7);
        let h = cfg.supervisor_homogeneity.clamp(0.0, 1.0);
        teams.push(Team {
            supervisor_openness: (1.0 - h) * indiv + h * common,
            ..Team::default()
        });
    }

    let ids: Vec<AgentId> = (0..n).map(|i| AgentId(i as u64)).collect();
    let network = match cfg.network_kind {
        NetworkKind::WattsStrogatz => {
            SocialNetwork::watts_strogatz(&ids, cfg.network_k.max(2), cfg.network_beta, rng)
        }
        NetworkKind::ErdosRenyi => SocialNetwork::erdos_renyi(&ids, cfg.network_beta, rng),
        NetworkKind::BarabasiAlbert => {
            SocialNetwork::barabasi_albert(&ids, cfg.network_k.max(1), rng)
        }
    };

    SilenceWorld::new(SimClock::new(cfg.t_max), employees, teams, network)
}

// --------------------------------------------------------------------------- //
// Run driver
// --------------------------------------------------------------------------- //

/// Build mechanisms + run one configuration (production LLM client built from env).
pub fn run(cfg: &Config) -> std::result::Result<SimulationResult, String> {
    if cfg.decision_mode.is_llm() {
        let client =
            build_live_client(&cfg.llm).map_err(|e| format!("LLM client build failed: {e}"))?;
        run_with_client(cfg, Some(client))
    } else {
        run_with_client(cfg, None)
    }
}

/// Run with an optional pre-built client (production via [`build_live_client`],
/// tests via [`crate::llm::wrap_client`] over a `ScriptedClient`).
pub fn run_with_client(
    cfg: &Config,
    client: Option<SilenceClient>,
) -> std::result::Result<SimulationResult, String> {
    let root = cfg.seed;

    let mut init_rng = SimRng::from_seed(derive_seed(root, &[RNG_WORLD_INIT]));
    let world = init_world(cfg, &mut init_rng);

    let shared_meta: SharedMetadata = Rc::new(RefCell::new(MetadataCollector::new()));
    let (llm_model, llm_endpoint, shared_client): (String, String, Option<SharedClient>) =
        match client {
            Some(c) => {
                let model = c.inner().model().to_string();
                let endpoint = c.inner().endpoint().to_string();
                (model, endpoint, Some(Rc::new(RefCell::new(c))))
            }
            None => ("none".to_string(), "none".to_string(), None),
        };

    let mut builder = SimulationBuilder::new(world)
        .scheduler(Box::new(RandomActivationScheduler))
        .seed(derive_seed(root, &[RNG_ENGINE]));

    // Environment
    builder = builder.add_mechanism(Box::new(IssueSalience::new(
        cfg.shock_t,
        cfg.shock_magnitude,
    )));
    builder = builder.add_mechanism(Box::new(RetaliationEvent::new(cfg.p_retaliate)));

    // Decision
    builder = builder.add_mechanism(Box::new(FearAppraisal::new()));
    match (cfg.decision_mode, &shared_client) {
        (DecisionMode::Llm, Some(sc)) => {
            builder = builder.add_mechanism(Box::new(VoiceDecisionLlm::new(
                Rc::clone(sc),
                Rc::clone(&shared_meta),
                cfg.llm.clone(),
                cfg.prompt_version,
                derive_seed(root, &[RNG_PROMPT_ROOT]),
            )));
        }
        (DecisionMode::Llm, None) => {
            return Err("LLM decision mode selected but no client supplied".to_string());
        }
        (mode, _) => {
            builder =
                builder.add_mechanism(Box::new(VoiceDecisionRule::new(cfg.beta, mode.n_dims())));
        }
    }

    // Interaction
    builder = builder.add_mechanism(Box::new(SilenceSpiral));
    builder = builder.add_mechanism(Box::new(PrefalseCascade));

    // Reward
    builder = builder.add_mechanism(Box::new(OrgPerformance::new()));

    // PostStep
    builder = builder.add_mechanism(Box::new(MotiveDynamics::new(cfg.motive_learn_rate)));
    builder = builder.add_mechanism(Box::new(PsafetyUpdate::new(cfg.psafety_learn)));
    builder = builder.add_mechanism(Box::new(ClimateSilence));
    builder = builder.add_mechanism(Box::new(MotiveMetrics));

    let mut sim = builder.build();

    let mut metrics_rows: Vec<MetricsRow> = Vec::new();
    let mut motive_mix_rows: Vec<MotiveMixRow> = Vec::new();
    let mut final_round = 0u64;

    sim.run_observed(|report| {
        let t = report.t;
        let world = report.world;
        let mm = motive_mix(world);
        let mv = motive_vec_mean(world);
        metrics_rows.push(MetricsRow {
            t,
            silence_rate: silence_rate(world),
            motive_mix_ineffectual: mm[0],
            motive_mix_relational: mm[1],
            motive_mix_defensive: mm[2],
            motive_mix_diffident: mm[3],
            motive_mix_disengaged: mm[4],
            motive_mix_deviant: mm[5],
            motive_vec_mean_ineffectual: mv[0],
            motive_vec_mean_relational: mv[1],
            motive_vec_mean_defensive: mv[2],
            motive_vec_mean_diffident: mv[3],
            motive_vec_mean_disengaged: mv[4],
            motive_vec_mean_deviant: mv[5],
            n_distinct_motives_active: n_distinct_motives_active(world, 0.05),
            climate_of_silence: climate_of_silence(world),
            org_performance: world.org_performance,
            issue_salience: world.issue_salience,
            kl_to_reference: kl_to_reference(mm),
        });
        motive_mix_rows.push(MotiveMixRow {
            t,
            ineffectual: mm[0],
            relational: mm[1],
            defensive: mm[2],
            diffident: mm[3],
            disengaged: mm[4],
            deviant: mm[5],
        });
        final_round = t;
    })
    .map_err(|e| format!("simulation run failed: {e}"))?;

    if let Some(sc) = &shared_client {
        if cfg.llm.cache_path.is_some() {
            sc.borrow()
                .cache()
                .save()
                .map_err(|e| format!("cache save failed: {e}"))?;
        }
    }

    let final_world = sim.world().clone();

    let mut agent_rows: Vec<AgentRow> = Vec::with_capacity(final_world.n_employees());
    for (&id, emp) in &final_world.employees {
        let a = emp.motive_vec.to_array();
        let pm = if emp.expression == Expression::Silence {
            emp.motive_vec.primary().label().to_string()
        } else {
            "-".to_string()
        };
        agent_rows.push(AgentRow {
            t: final_round,
            agent_id: id.0,
            team: emp.team,
            level: emp.level,
            tenure: emp.tenure,
            expression: emp.expression.label().to_string(),
            primary_motive: pm,
            motive_ineffectual: a[0],
            motive_relational: a[1],
            motive_defensive: a[2],
            motive_diffident: a[3],
            motive_disengaged: a[4],
            motive_deviant: a[5],
            fear: emp.fear,
            psafety: emp.psych_safety,
            ivt: emp.ivt_strength,
            neuroticism: emp.neuroticism,
            extraversion: emp.extraversion,
            private_concern: emp.private_concern,
        });
    }

    let correlation_rows = build_correlation_rows(&final_world);

    let metadata = shared_meta.borrow().clone();
    let _ = defensive_share; // available for downstream callers / tests
    Ok(SimulationResult {
        final_round,
        world: final_world,
        metrics_rows,
        motive_mix_rows,
        agent_rows,
        correlation_rows,
        metadata,
        llm_model,
        llm_endpoint,
    })
}

// --------------------------------------------------------------------------- //
// Correlations (agent-level Pearson r over the final step)
// --------------------------------------------------------------------------- //

fn build_correlation_rows(world: &SilenceWorld) -> Vec<CorrelationRow> {
    let mut psafety_vec: Vec<f64> = Vec::new();
    let mut fear_vec: Vec<f64> = Vec::new();
    let mut neuro_vec: Vec<f64> = Vec::new();
    let mut extra_vec: Vec<f64> = Vec::new();
    let mut climate_vec: Vec<f64> = Vec::new();
    // Per-motive soft-share columns (over the full population; VOICE = uniform).
    let mut motive_cols: [Vec<f64>; 6] = Default::default();
    let climates = crate::metrics::team_climates(world);

    for emp in world.employees.values() {
        psafety_vec.push(emp.psych_safety);
        fear_vec.push(emp.fear);
        neuro_vec.push(emp.neuroticism);
        extra_vec.push(emp.extraversion);
        climate_vec.push(climates[emp.team]);
        let a = emp.motive_vec.to_array();
        for i in 0..6 {
            // Only silent agents carry a meaningful motive; voicing agents
            // contribute their uniform vector (neutral on correlation).
            motive_cols[i].push(if emp.expression == Expression::Silence {
                a[i]
            } else {
                0.0
            });
        }
    }

    let correlates: [(&str, &[f64]); 5] = [
        ("psafety", &psafety_vec),
        ("fear", &fear_vec),
        ("neuroticism", &neuro_vec),
        ("extraversion", &extra_vec),
        ("climate_of_silence", &climate_vec),
    ];

    let mut rows = Vec::with_capacity(6 * correlates.len());
    for m in MotiveLabel::ALL {
        let col = &motive_cols[m.index()];
        for (name, corr) in &correlates {
            rows.push(CorrelationRow {
                motive: m.label().to_string(),
                correlate: (*name).to_string(),
                pearson_r: pearson(col, corr),
            });
        }
    }
    rows
}

// --------------------------------------------------------------------------- //
// Output writers
// --------------------------------------------------------------------------- //

pub fn ensure_output_dir(output_dir: &str) {
    socsim_results::ensure_dir(output_dir).expect("failed to create output directory");
}

pub fn save_metrics(result: &SimulationResult, output_dir: &str) {
    let path = format!("{output_dir}/metrics.csv");
    socsim_results::write_csv(&result.metrics_rows, &path).expect("failed to write metrics.csv");
}

pub fn save_motive_mix(result: &SimulationResult, output_dir: &str) {
    let path = format!("{output_dir}/motive_mix.csv");
    socsim_results::write_csv(&result.motive_mix_rows, &path)
        .expect("failed to write motive_mix.csv");
}

pub fn save_agents(result: &SimulationResult, output_dir: &str) {
    let path = format!("{output_dir}/agents.csv");
    socsim_results::write_csv(&result.agent_rows, &path).expect("failed to write agents.csv");
}

pub fn save_correlations(result: &SimulationResult, output_dir: &str) {
    let path = format!("{output_dir}/correlations.csv");
    socsim_results::write_csv(&result.correlation_rows, &path)
        .expect("failed to write correlations.csv");
}

/// `llm_meta.json` (model / endpoint / temperature / seed / cache stats).
#[derive(Serialize)]
pub struct LlmMetaJson {
    pub decision_mode: String,
    pub llm_model: String,
    pub llm_endpoint: String,
    pub llm_temperature: f32,
    pub llm_seed: u64,
    pub prompt_version: u8,
    pub total_calls: usize,
    pub cache_hits: usize,
    pub cache_hit_rate: f64,
    pub final_round: u64,
    pub determinism_note: &'static str,
}

pub fn save_llm_meta(result: &SimulationResult, cfg: &Config, output_dir: &str) {
    let meta = LlmMetaJson {
        decision_mode: cfg.decision_mode.label().to_string(),
        llm_model: result.llm_model.clone(),
        llm_endpoint: result.llm_endpoint.clone(),
        llm_temperature: cfg.llm.temperature,
        llm_seed: cfg.llm.seed,
        prompt_version: cfg.prompt_version,
        total_calls: result.metadata.total(),
        cache_hits: result.metadata.cache_hits(),
        cache_hit_rate: result.metadata.cache_hit_rate(),
        final_round: result.final_round,
        determinism_note: "LLM output is outside socsim bit-reproducibility; the prompt->response \
                           cache (temperature=0 + (agent_id, t)-derived seed) is the reproducibility \
                           mechanism. The socsim core (employee init, network, scheduling, the \
                           non-LLM mechanisms) is deterministic given the seed. The rule_* decision \
                           modes make zero LLM calls and are bit-reproducible.",
    };
    let path = format!("{output_dir}/llm_meta.json");
    socsim_results::write_json(&meta, &path).expect("failed to write llm_meta.json");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DecisionMode;

    fn small_cfg(mode: DecisionMode) -> Config {
        Config {
            n_teams: 2,
            team_size: 4,
            n_levels: 2,
            network_k: 2,
            t_max: 6,
            runs: 1,
            seed: 42,
            decision_mode: mode,
            ..Config::default()
        }
    }

    #[test]
    fn rule_run_is_deterministic() {
        let a = run_with_client(&small_cfg(DecisionMode::Rule6dim), None).unwrap();
        let b = run_with_client(&small_cfg(DecisionMode::Rule6dim), None).unwrap();
        assert_eq!(a.metrics_rows.len(), b.metrics_rows.len());
        for (ra, rb) in a.metrics_rows.iter().zip(b.metrics_rows.iter()) {
            assert!((ra.silence_rate - rb.silence_rate).abs() < 1e-15);
            assert!((ra.motive_mix_defensive - rb.motive_mix_defensive).abs() < 1e-15);
        }
        assert_eq!(a.metadata.total(), 0, "rule mode makes 0 LLM calls");
    }

    #[test]
    fn motive_mix_sums_to_one_or_zero() {
        for mode in [
            DecisionMode::Rule6dim,
            DecisionMode::Rule4dim,
            DecisionMode::Rule3dim,
        ] {
            let r = run_with_client(&small_cfg(mode), None).unwrap();
            for row in &r.metrics_rows {
                let s = row.motive_mix_ineffectual
                    + row.motive_mix_relational
                    + row.motive_mix_defensive
                    + row.motive_mix_diffident
                    + row.motive_mix_disengaged
                    + row.motive_mix_deviant;
                assert!(
                    s.abs() < 1e-9 || (s - 1.0).abs() < 1e-9,
                    "mode={:?} sum={s}",
                    mode
                );
            }
        }
    }

    #[test]
    fn correlation_matrix_shape() {
        let r = run_with_client(&small_cfg(DecisionMode::Rule6dim), None).unwrap();
        assert_eq!(r.correlation_rows.len(), 6 * 5);
    }
}
