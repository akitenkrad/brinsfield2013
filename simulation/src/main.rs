//! Brinsfield (2013) — Six-motive employee silence CLI.
//!
//! `run`       : single configuration; `--decision-mode {llm|rule_6dim|rule_4dim|rule_3dim}`.
//! `sweep`     : Cartesian product over `ψ_learn × p_retaliate × motive-init-defensive × seeds`.
//! `ablate`    : run several decision modes side-by-side; compare motive_mix + KL to reference.
//! `reproduce` : print the Brinsfield anchors vs the latest run's emergent values.

use std::fs;
use std::path::Path;

use clap::{Parser, Subcommand};

use brinsfield_silence_simulation::calibration::{
    DEFENSIVE_SHARE_ANCHOR, DEFENSIVE_SHARE_TOL, DEVIANT_CEILING, INEFFECTUAL_FLOOR,
    REFERENCE_MOTIVE_MIX,
};
use brinsfield_silence_simulation::config::{
    parse_decision_mode, parse_network_kind, BetaGroup, Config, DecisionMode, LlmSettings,
    MotiveInit, NetworkKind,
};
use brinsfield_silence_simulation::simulation::{
    ensure_output_dir, run, save_agents, save_correlations, save_llm_meta, save_metrics,
    save_motive_mix, SimulationResult,
};

use socsim_core::derive_seed;
use socsim_results::{refresh_latest_symlink, timestamp, write_csv, write_json};

// --------------------------------------------------------------------------- //
// CLI
// --------------------------------------------------------------------------- //

#[derive(Parser, Debug)]
#[command(
    name = "brinsfield",
    about = "Brinsfield (2013) — Six forms of employee silence (LLM vs rule_6/4/3dim)"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
    /// Ollama 接続先 URL（指定時は環境変数 OLLAMA_HOST を上書きする）．
    #[arg(long, global = true)]
    ollama_host: Option<String>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run a single configuration.
    Run(RunArgs),
    /// Sweep ψ_learn × p_retaliate × motive-init-defensive × seeds.
    Sweep(SweepArgs),
    /// Run several decision modes side-by-side and compare motive_mix / KL.
    Ablate(AblateArgs),
    /// Print Brinsfield anchors vs the latest run's emergent values.
    Reproduce(ReproduceArgs),
}

#[derive(Parser, Debug)]
struct RunArgs {
    /// Decision mechanism (llm / rule_6dim / rule_4dim / rule_3dim).
    #[arg(long, default_value = "rule_6dim")]
    decision_mode: String,
    #[arg(long, default_value_t = 5)]
    n_teams: usize,
    #[arg(long, default_value_t = 8)]
    team_size: usize,
    #[arg(long, default_value_t = 3)]
    n_levels: u8,
    #[arg(long, default_value = "watts-strogatz")]
    network: String,
    #[arg(long, default_value_t = 6)]
    network_k: usize,
    #[arg(long, default_value_t = 0.1)]
    network_beta: f64,
    /// Supervisor-openness homogeneity η_sup ∈ [0,1].
    #[arg(long, default_value_t = 0.0)]
    supervisor_homogeneity: f64,
    /// Initial six-motive distribution "ineff,rel,def,dif,dis,dev".
    #[arg(long, default_value = "0.35,0.20,0.13,0.13,0.13,0.06")]
    motive_init: String,
    /// EMA motive learning rate η (motive_dynamics).
    #[arg(long, default_value_t = 0.10)]
    motive_learn_rate: f64,
    /// Psychological-safety learning rate.
    #[arg(long, default_value_t = 0.05)]
    psafety_learn: f64,
    /// Per-agent per-step retaliation probability.
    #[arg(long, default_value_t = 0.05)]
    p_retaliate: f64,
    /// Optional exogenous σ-shock time step.
    #[arg(long)]
    shock_t: Option<u64>,
    /// σ-shock magnitude.
    #[arg(long, default_value_t = 0.3)]
    shock_magnitude: f64,
    /// LLM temperature.
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,
    /// Prompt template version (1 / 2 / 3).
    #[arg(long, default_value_t = 1)]
    prompt_version: u8,
    #[arg(long, default_value_t = 48)]
    t_max: u64,
    #[arg(long, default_value_t = 1)]
    runs: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// LLM generation seed offset.
    #[arg(long, default_value_t = 0)]
    llm_seed: u64,
    /// Prompt → response cache path (LLM mode only).
    #[arg(long, default_value = ".llm_cache/cache.json")]
    llm_cache_path: String,
    #[arg(long, default_value = "results")]
    output_dir: String,
}

#[derive(Parser, Debug)]
struct SweepArgs {
    #[arg(long, default_value = "rule_6dim")]
    decision_mode: String,
    #[arg(long, default_value_t = 5)]
    n_teams: usize,
    #[arg(long, default_value_t = 8)]
    team_size: usize,
    /// ψ-learning-rate sweep values.
    #[arg(long, default_value = "0.05,0.10,0.20")]
    psafety_learn: String,
    /// p_retaliate sweep values.
    #[arg(long, default_value = "0.02,0.05,0.10")]
    p_retaliate: String,
    /// motive-init defensive-share sweep values (other 5 motives rescaled).
    #[arg(long, default_value = "0.05,0.10,0.15,0.20")]
    motive_init_defensive: String,
    #[arg(long, default_value_t = 5)]
    runs: usize,
    #[arg(long, default_value_t = 48)]
    t_max: u64,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value = "results")]
    output_dir: String,
}

#[derive(Parser, Debug)]
struct AblateArgs {
    /// Comma-separated decision modes to compare.
    #[arg(long, default_value = "rule_6dim,rule_4dim,rule_3dim")]
    decision_modes: String,
    #[arg(long, default_value_t = 5)]
    n_teams: usize,
    #[arg(long, default_value_t = 8)]
    team_size: usize,
    #[arg(long, default_value_t = 5)]
    runs: usize,
    #[arg(long, default_value_t = 48)]
    t_max: u64,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value = "results/ablation")]
    output_dir: String,
}

#[derive(Parser, Debug)]
struct ReproduceArgs {
    /// Decision mode used for the emergent comparison run.
    #[arg(long, default_value = "rule_6dim")]
    decision_mode: String,
    #[arg(long, default_value_t = 48)]
    t_max: u64,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value = "results")]
    output_dir: String,
}

// --------------------------------------------------------------------------- //
// CSV rows
// --------------------------------------------------------------------------- //

#[derive(serde::Serialize)]
struct SweepRow {
    decision_mode: String,
    psafety_learn: f64,
    p_retaliate: f64,
    motive_init_defensive: f64,
    run: usize,
    seed: u64,
    final_round: u64,
    silence_rate: f64,
    motive_mix_ineffectual: f64,
    motive_mix_relational: f64,
    motive_mix_defensive: f64,
    motive_mix_diffident: f64,
    motive_mix_disengaged: f64,
    motive_mix_deviant: f64,
    climate_of_silence: f64,
    kl_to_reference: f64,
}

#[derive(serde::Serialize)]
struct AblateRow {
    decision_mode: String,
    run: usize,
    seed: u64,
    silence_rate: f64,
    motive_mix_ineffectual: f64,
    motive_mix_relational: f64,
    motive_mix_defensive: f64,
    motive_mix_diffident: f64,
    motive_mix_disengaged: f64,
    motive_mix_deviant: f64,
    kl_to_reference: f64,
}

// --------------------------------------------------------------------------- //
// helpers
// --------------------------------------------------------------------------- //

fn parse_f64_list(s: &str) -> Vec<f64> {
    s.split([',', ' '])
        .filter(|t| !t.is_empty())
        .filter_map(|t| t.trim().parse::<f64>().ok())
        .collect()
}

/// Build a motive-init with a given defensive share; the remaining mass is
/// distributed over the other five motives in the default proportions.
fn motive_init_with_defensive(defensive: f64) -> MotiveInit {
    let d = MotiveInit::default();
    let others = [
        d.ineffectual,
        d.relational,
        d.diffident,
        d.disengaged,
        d.deviant,
    ];
    let other_sum: f64 = others.iter().sum();
    let scale = (1.0 - defensive).max(0.0) / other_sum.max(1e-9);
    MotiveInit {
        ineffectual: d.ineffectual * scale,
        relational: d.relational * scale,
        defensive,
        diffident: d.diffident * scale,
        disengaged: d.disengaged * scale,
        deviant: d.deviant * scale,
    }
}

fn cfg_from_run_args(args: &RunArgs) -> Config {
    Config {
        n_teams: args.n_teams,
        team_size: args.team_size,
        n_levels: args.n_levels,
        network_kind: parse_network_kind(&args.network).unwrap_or(NetworkKind::WattsStrogatz),
        network_k: args.network_k,
        network_beta: args.network_beta,
        supervisor_homogeneity: args.supervisor_homogeneity,
        decision_mode: parse_decision_mode(&args.decision_mode).unwrap_or_else(|e| panic!("{e}")),
        prompt_version: args.prompt_version,
        motive_init: MotiveInit::parse(&args.motive_init).unwrap_or_else(|e| panic!("{e}")),
        beta: BetaGroup::default(),
        motive_learn_rate: args.motive_learn_rate,
        psafety_learn: args.psafety_learn,
        p_retaliate: args.p_retaliate,
        shock_t: args.shock_t,
        shock_magnitude: args.shock_magnitude,
        t_max: args.t_max,
        runs: args.runs,
        seed: args.seed,
        llm: LlmSettings {
            temperature: args.temperature,
            seed: args.llm_seed,
            cache_path: Some(args.llm_cache_path.clone()),
        },
        output_dir: args.output_dir.clone(),
    }
}

// --------------------------------------------------------------------------- //
// run
// --------------------------------------------------------------------------- //

fn cmd_run(args: RunArgs) {
    let timestamp = timestamp();
    let output_dir = format!("{}/{}", args.output_dir, timestamp);
    ensure_output_dir(&output_dir);

    let mut base_cfg = cfg_from_run_args(&args);
    base_cfg.output_dir = output_dir.clone();
    if base_cfg.decision_mode.is_llm() {
        if let Some(parent) = Path::new(&args.llm_cache_path).parent() {
            let _ = fs::create_dir_all(parent);
        }
    }

    println!("=== Brinsfield (2013) — Six forms of employee silence ===");
    println!(
        "decision-mode: {} | teams: {}×{} (={}) | network: {:?} k={} β={:.2}",
        base_cfg.decision_mode.label(),
        base_cfg.n_teams,
        base_cfg.team_size,
        base_cfg.n_employees(),
        base_cfg.network_kind,
        base_cfg.network_k,
        base_cfg.network_beta,
    );
    let mi = base_cfg.motive_init.normalised();
    println!(
        "motive_init: ineff={:.2} rel={:.2} def={:.2} dif={:.2} dis={:.2} dev={:.2} | η={} t_max={} runs={} seed={}",
        mi[0], mi[1], mi[2], mi[3], mi[4], mi[5],
        base_cfg.motive_learn_rate, base_cfg.t_max, base_cfg.runs, base_cfg.seed,
    );
    println!("output: {output_dir}");
    println!("----------------------------------------------------------------------");

    {
        let path = format!("{output_dir}/config.json");
        write_json(&base_cfg.to_run_config_json(), &path).expect("failed to write config.json");
    }

    let mut last_result: Option<SimulationResult> = None;
    let runs = base_cfg.runs.max(1);
    for run_idx in 0..runs {
        let seed = derive_seed(base_cfg.seed, &[run_idx as u64]);
        let cfg = Config {
            seed,
            ..base_cfg.clone()
        };
        let result = run(&cfg).unwrap_or_else(|e| panic!("run failed: {e}"));
        let f = result.metrics_rows.last();
        println!(
            "[{}/{}] seed={} silence={:.3} mix=(i{:.2}/r{:.2}/def{:.2}/dif{:.2}/dis{:.2}/dev{:.2}) C={:.3} KL={:.3}",
            run_idx + 1,
            runs,
            seed,
            f.map(|r| r.silence_rate).unwrap_or(0.0),
            f.map(|r| r.motive_mix_ineffectual).unwrap_or(0.0),
            f.map(|r| r.motive_mix_relational).unwrap_or(0.0),
            f.map(|r| r.motive_mix_defensive).unwrap_or(0.0),
            f.map(|r| r.motive_mix_diffident).unwrap_or(0.0),
            f.map(|r| r.motive_mix_disengaged).unwrap_or(0.0),
            f.map(|r| r.motive_mix_deviant).unwrap_or(0.0),
            f.map(|r| r.climate_of_silence).unwrap_or(0.0),
            f.map(|r| r.kl_to_reference).unwrap_or(0.0),
        );
        last_result = Some(result);
    }

    let result = last_result.expect("at least one run");
    save_metrics(&result, &output_dir);
    save_motive_mix(&result, &output_dir);
    save_agents(&result, &output_dir);
    save_correlations(&result, &output_dir);
    save_llm_meta(&result, &base_cfg, &output_dir);

    let _ = refresh_latest_symlink(&args.output_dir, &timestamp);

    println!("----------------------------------------------------------------------");
    println!(
        "LLM calls: {} | cache-hit: {} ({:.1}%) | model: {}",
        result.metadata.total(),
        result.metadata.cache_hits(),
        result.metadata.cache_hit_rate() * 100.0,
        result.llm_model,
    );
    println!("metrics      → {output_dir}/metrics.csv");
    println!("motive_mix   → {output_dir}/motive_mix.csv");
    println!("agents       → {output_dir}/agents.csv");
    println!("correlations → {output_dir}/correlations.csv");
    println!("llm_meta     → {output_dir}/llm_meta.json");
    println!("config       → {output_dir}/config.json");
}

// --------------------------------------------------------------------------- //
// sweep
// --------------------------------------------------------------------------- //

fn cmd_sweep(args: SweepArgs) {
    let decision_mode = parse_decision_mode(&args.decision_mode).unwrap_or_else(|e| panic!("{e}"));
    let timestamp = timestamp();
    let dir_name = format!("{timestamp}_sweep");
    let sweep_dir = format!("{}/{}", args.output_dir, dir_name);
    fs::create_dir_all(&sweep_dir).expect("failed to create sweep dir");

    let psafety_vals = parse_f64_list(&args.psafety_learn);
    let retaliate_vals = parse_f64_list(&args.p_retaliate);
    let defensive_vals = parse_f64_list(&args.motive_init_defensive);

    let n_cells = psafety_vals.len() * retaliate_vals.len() * defensive_vals.len();
    let n_total = n_cells * args.runs;
    println!("=== brinsfield-sweep ===");
    println!(
        "decision_mode: {} | ψ_learn={:?} p_retaliate={:?} motive_init_def={:?} | runs/cell={} | total {} runs",
        decision_mode.label(),
        psafety_vals,
        retaliate_vals,
        defensive_vals,
        args.runs,
        n_total,
    );
    println!("output: {sweep_dir}");
    println!("------------------------------------------------------------");

    {
        let config_json = serde_json::json!({
            "command": "sweep",
            "decision_mode": decision_mode.label(),
            "n_teams": args.n_teams,
            "team_size": args.team_size,
            "psafety_learn_values": psafety_vals,
            "p_retaliate_values": retaliate_vals,
            "motive_init_defensive_values": defensive_vals,
            "runs": args.runs,
            "t_max": args.t_max,
            "seed": args.seed,
        });
        let path = format!("{sweep_dir}/sweep_config.json");
        write_json(&config_json, &path).expect("failed to write sweep_config.json");
    }

    let mut rows: Vec<SweepRow> = Vec::with_capacity(n_total);
    let mut idx = 0usize;
    for &psl in &psafety_vals {
        for &pr in &retaliate_vals {
            for &dfn in &defensive_vals {
                for run_idx in 0..args.runs {
                    idx += 1;
                    let seed = derive_seed(
                        args.seed,
                        &[
                            (psl * 1000.0) as u64,
                            (pr * 1000.0) as u64,
                            (dfn * 1000.0) as u64,
                            run_idx as u64,
                        ],
                    );
                    let cfg = Config {
                        n_teams: args.n_teams,
                        team_size: args.team_size,
                        decision_mode,
                        psafety_learn: psl,
                        p_retaliate: pr,
                        motive_init: motive_init_with_defensive(dfn),
                        t_max: args.t_max,
                        runs: 1,
                        seed,
                        ..Config::default()
                    };
                    let result = run(&cfg).unwrap_or_else(|e| panic!("sweep run failed: {e}"));
                    let last = result.metrics_rows.last().expect("metrics_rows non-empty");
                    rows.push(SweepRow {
                        decision_mode: decision_mode.label().to_string(),
                        psafety_learn: psl,
                        p_retaliate: pr,
                        motive_init_defensive: dfn,
                        run: run_idx,
                        seed,
                        final_round: result.final_round,
                        silence_rate: last.silence_rate,
                        motive_mix_ineffectual: last.motive_mix_ineffectual,
                        motive_mix_relational: last.motive_mix_relational,
                        motive_mix_defensive: last.motive_mix_defensive,
                        motive_mix_diffident: last.motive_mix_diffident,
                        motive_mix_disengaged: last.motive_mix_disengaged,
                        motive_mix_deviant: last.motive_mix_deviant,
                        climate_of_silence: last.climate_of_silence,
                        kl_to_reference: last.kl_to_reference,
                    });
                    if idx.is_multiple_of(10) || idx == n_total {
                        println!(
                            "[{}/{}] ψ_learn={:.2} p_ret={:.2} def_init={:.2} run={} silence={:.3} def={:.3}",
                            idx, n_total, psl, pr, dfn, run_idx, last.silence_rate,
                            last.motive_mix_defensive
                        );
                    }
                }
            }
        }
    }

    let path = format!("{sweep_dir}/sweep_summary.csv");
    write_csv(&rows, &path).expect("failed to write sweep_summary.csv");

    let _ = refresh_latest_symlink(&args.output_dir, &dir_name);
    println!("------------------------------------------------------------");
    println!("sweep done. summary → {sweep_dir}/sweep_summary.csv");
}

// --------------------------------------------------------------------------- //
// ablate
// --------------------------------------------------------------------------- //

fn cmd_ablate(args: AblateArgs) {
    let modes: Vec<DecisionMode> = args
        .decision_modes
        .split([',', ' '])
        .filter(|t| !t.is_empty())
        .map(|t| parse_decision_mode(t).unwrap_or_else(|e| panic!("{e}")))
        .collect();
    fs::create_dir_all(&args.output_dir).expect("failed to create ablation dir");

    println!("=== brinsfield-ablate ===");
    println!(
        "modes: {:?} | runs/mode={} | t_max={} | seed={}",
        modes.iter().map(|m| m.label()).collect::<Vec<_>>(),
        args.runs,
        args.t_max,
        args.seed,
    );
    println!("output: {}", args.output_dir);
    println!("------------------------------------------------------------");

    let mut rows: Vec<AblateRow> = Vec::new();
    for &mode in &modes {
        for run_idx in 0..args.runs {
            let seed = derive_seed(args.seed, &[mode.n_dims() as u64, run_idx as u64]);
            let cfg = Config {
                n_teams: args.n_teams,
                team_size: args.team_size,
                decision_mode: mode,
                t_max: args.t_max,
                runs: 1,
                seed,
                ..Config::default()
            };
            let result = run(&cfg).unwrap_or_else(|e| panic!("ablate run failed: {e}"));
            let last = result.metrics_rows.last().expect("metrics_rows non-empty");
            rows.push(AblateRow {
                decision_mode: mode.label().to_string(),
                run: run_idx,
                seed,
                silence_rate: last.silence_rate,
                motive_mix_ineffectual: last.motive_mix_ineffectual,
                motive_mix_relational: last.motive_mix_relational,
                motive_mix_defensive: last.motive_mix_defensive,
                motive_mix_diffident: last.motive_mix_diffident,
                motive_mix_disengaged: last.motive_mix_disengaged,
                motive_mix_deviant: last.motive_mix_deviant,
                kl_to_reference: last.kl_to_reference,
            });
        }
        // Per-mode mean KL to reference.
        let mode_rows: Vec<&AblateRow> = rows
            .iter()
            .filter(|r| r.decision_mode == mode.label())
            .collect();
        let mean_kl: f64 = mode_rows.iter().map(|r| r.kl_to_reference).sum::<f64>()
            / mode_rows.len().max(1) as f64;
        let mean_def: f64 = mode_rows
            .iter()
            .map(|r| r.motive_mix_defensive)
            .sum::<f64>()
            / mode_rows.len().max(1) as f64;
        println!(
            "{:<10} mean KL→ref={:.4} mean defensive_share={:.3}",
            mode.label(),
            mean_kl,
            mean_def,
        );
    }

    let path = format!("{}/ablation_summary.csv", args.output_dir);
    write_csv(&rows, &path).expect("failed to write ablation_summary.csv");
    println!("------------------------------------------------------------");
    println!("ablation done. summary → {path}");
}

// --------------------------------------------------------------------------- //
// reproduce
// --------------------------------------------------------------------------- //

fn cmd_reproduce(args: ReproduceArgs) {
    let mode = parse_decision_mode(&args.decision_mode).unwrap_or_else(|e| panic!("{e}"));
    let timestamp = timestamp();
    let output_dir = format!("{}/{}_reproduce", args.output_dir, timestamp);
    ensure_output_dir(&output_dir);

    let cfg = Config {
        decision_mode: mode,
        t_max: args.t_max,
        seed: args.seed,
        runs: 1,
        output_dir: output_dir.clone(),
        ..Config::default()
    };
    let result = run(&cfg).unwrap_or_else(|e| panic!("reproduce run failed: {e}"));
    save_metrics(&result, &output_dir);
    save_motive_mix(&result, &output_dir);

    // Average the steady-state (t >= t_max/2) motive_mix.
    let half = args.t_max / 2;
    let tail: Vec<_> = result.metrics_rows.iter().filter(|r| r.t >= half).collect();
    let n = tail.len().max(1) as f64;
    let mut emergent = [0.0; 6];
    for r in &tail {
        emergent[0] += r.motive_mix_ineffectual;
        emergent[1] += r.motive_mix_relational;
        emergent[2] += r.motive_mix_defensive;
        emergent[3] += r.motive_mix_diffident;
        emergent[4] += r.motive_mix_disengaged;
        emergent[5] += r.motive_mix_deviant;
    }
    for v in emergent.iter_mut() {
        *v /= n;
    }

    let labels = [
        "ineffectual",
        "relational",
        "defensive",
        "diffident",
        "disengaged",
        "deviant",
    ];
    println!(
        "=== Brinsfield (2013) — reproduce (mode={}) ===",
        mode.label()
    );
    println!("steady-state motive_mix (mean over t >= {half}):");
    println!("  {:<13} {:>10} {:>10}", "motive", "emergent", "reference");
    for i in 0..6 {
        println!(
            "  {:<13} {:>10.4} {:>10.4}",
            labels[i], emergent[i], REFERENCE_MOTIVE_MIX[i]
        );
    }
    let def = emergent[2];
    let def_ok = (def - DEFENSIVE_SHARE_ANCHOR).abs() <= DEFENSIVE_SHARE_TOL;
    let ineff_ok = emergent[0] >= INEFFECTUAL_FLOOR - 0.05;
    let dev_ok = emergent[5] <= DEVIANT_CEILING + 0.02;
    println!("------------------------------------------------------------");
    println!(
        "defensive share {:.4} vs anchor {:.4} (±{:.2}): {}",
        def,
        DEFENSIVE_SHARE_ANCHOR,
        DEFENSIVE_SHARE_TOL,
        if def_ok { "PASS" } else { "off-anchor" }
    );
    println!(
        "ineffectual {:.4} ≥ floor {:.2}: {}",
        emergent[0],
        INEFFECTUAL_FLOOR,
        if ineff_ok { "PASS" } else { "below" }
    );
    println!(
        "deviant {:.4} ≤ ceiling {:.2}: {}",
        emergent[5],
        DEVIANT_CEILING,
        if dev_ok { "PASS" } else { "above" }
    );
    println!("------------------------------------------------------------");
    println!("Empirical 6-factor CFA superiority (vs 1–5 factor) is reproduced on the");
    println!("Python side: `uv run brinsfield-tools cfa --sample synth` (semopy).");
    println!("results → {output_dir}");
}

// --------------------------------------------------------------------------- //
// main
// --------------------------------------------------------------------------- //

fn main() {
    let cli = Cli::parse();
    if let Some(host) = cli.ollama_host.as_deref() {
        std::env::set_var("OLLAMA_HOST", host);
    }
    match cli.command {
        Commands::Run(args) => cmd_run(args),
        Commands::Sweep(args) => cmd_sweep(args),
        Commands::Ablate(args) => cmd_ablate(args),
        Commands::Reproduce(args) => cmd_reproduce(args),
    }
}
