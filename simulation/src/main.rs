//! Brinsfield (2013) — Six-motive employee silence CLI.
//!
//! `run`       : single configuration; `--decision-mode {llm|rule_6dim|rule_4dim|rule_3dim}`.
//! `sweep`     : Cartesian product over `ψ_learn × p_retaliate × motive-init-defensive × seeds`.
//! `ablate`    : run several decision modes side-by-side; compare motive_mix + KL to reference.
//! `reproduce` : print the Brinsfield anchors vs the emergent steady-state values.
//!
//! 出力の置き場と同一性は runvault が持つ．タイムスタンプ付きディレクトリも `latest`
//! シンボリックリンクもこちらでは作らず，`Run::start` が決めた run ディレクトリへ書く．
//!
//! `run` と `reproduce` は 1 本の run になる (`--runs N` は掃引ではなく，最後の反復
//! だけを残す従来どおりの動きなので子には割らない)．`sweep` と `ablate` はセル 1 つが
//! 子 run で，そのセルの試行は `events.jsonl` の `terminal` 行になる．割り当ての理由は
//! `brinsfield_silence_simulation::record` の冒頭を参照．

use std::fs;
use std::path::Path;

use clap::{Parser, Subcommand};
use runvault::{Lineage, Progress, Run, RunOptions, Stage};
use serde::Serialize;

use brinsfield_silence_simulation::calibration::{
    DEFENSIVE_SHARE_ANCHOR, DEFENSIVE_SHARE_TOL, DEVIANT_CEILING, INEFFECTUAL_FLOOR,
    REFERENCE_MOTIVE_MIX,
};
use brinsfield_silence_simulation::config::{
    parse_decision_mode, parse_network_kind, BetaGroup, Config, DecisionMode, LlmSettings,
    MotiveInit, NetworkKind,
};
use brinsfield_silence_simulation::llm::{build_live_client, SilenceClient};
use brinsfield_silence_simulation::record::{
    self, ConditionParameters, ReplicateGroupParameters, SingleRunParameters, DOMAIN, EXPERIMENT,
    GROUP_SEED_POINTERS, HASH_EXCLUDE, REPO_ID, SINGLE_SEED_POINTERS,
};
use brinsfield_silence_simulation::simulation::{run_with_client_observed, SimulationResult};

use socsim_llm::LlmClient;

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
    /// Print Brinsfield anchors vs the emergent steady-state values.
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
    /// runvault の results root（run ディレクトリの親）．
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
    /// runvault の results root（run ディレクトリの親）．
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
    /// runvault の results root（run ディレクトリの親）．
    #[arg(long, default_value = "results")]
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
    /// runvault の results root（run ディレクトリの親）．
    #[arg(long, default_value = "results")]
    output_dir: String,
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
    }
}

/// LLM モードのときだけキャッシュの置き場を掘る．
fn ensure_cache_dir(cfg: &Config) {
    if !cfg.decision_mode.is_llm() {
        return;
    }
    if let Some(path) = cfg.llm.cache_path.as_deref() {
        if let Some(parent) = Path::new(path).parent() {
            let _ = fs::create_dir_all(parent);
        }
    }
}

/// 本番の LLM クライアント（rule モードでは `None`）．
///
/// `Run::start` の前に組む — モデル名と endpoint を知っているのはクライアントを
/// 組んだ側だけで，`llm` ブロックはそこからしか埋められない．
fn build_client(cfg: &Config) -> Option<SilenceClient> {
    if !cfg.decision_mode.is_llm() {
        return None;
    }
    Some(build_live_client(&cfg.llm).unwrap_or_else(|e| panic!("LLM client build failed: {e}")))
}

/// 1 本のシミュレーションを回して run へ記録する（`run` / `reproduce` 共通）．
///
/// 進捗の 1 単位は 1 ステップ．費用がそこにあるからで，1 ステップは全従業員に
/// ついて決定を出し，`--decision-mode llm` ではその 1 つ 1 つがモデル呼び出しに
/// なる．反復 1 本を 1 単位にすると，ライブの 1 本は 0/1 と出したきり終わりまで
/// 黙る．
fn simulate_into(
    run: &mut Run,
    cfg: &Config,
    client: Option<SilenceClient>,
    stage: &mut Stage,
) -> SimulationResult {
    let result = run_with_client_observed(cfg, client, |_| stage.tick())
        .unwrap_or_else(|e| panic!("run failed: {e}"));
    record::log_simulation(run, &result);
    if cfg.decision_mode.is_llm() {
        record::log_llm_usage(run, &result);
    }
    record::log_paper_reference(run);
    result
}

fn print_result_line(index: usize, total: usize, seed: u64, result: &SimulationResult) {
    let f = result.metrics_rows.last();
    println!(
        "[{}/{}] seed={} silence={:.3} mix=(i{:.2}/r{:.2}/def{:.2}/dif{:.2}/dis{:.2}/dev{:.2}) C={:.3} KL={:.3}",
        index,
        total,
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
}

// --------------------------------------------------------------------------- //
// run
// --------------------------------------------------------------------------- //

fn cmd_run(args: RunArgs) {
    let base_cfg = cfg_from_run_args(&args);
    ensure_cache_dir(&base_cfg);
    let runs = base_cfg.runs.max(1);

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
    println!("----------------------------------------------------------------------");

    // 記録に残るのは最後の反復だけだが，途中の反復も移行前と同じ順序・同じシードで
    // 回す（LLM キャッシュの温まり方まで含めて振る舞いを変えないため）．run を起こす
    // 前に回しておくのは，status.json の duration_sec に «捨てる反復» の時間を
    // 混ぜないためである．
    //
    // 捨てる反復はまだ run が無いので，記録先の無い stderr だけの stage で数える．
    // 数えないという選択肢は無い — `--runs 20 --decision-mode llm` なら，ここが
    // このコマンドの時間のほとんどを占める．
    let mut discarded = (runs > 1)
        .then(|| Progress::to_stderr().stage("discarded", (runs - 1) * base_cfg.t_max as usize));
    for run_idx in 0..runs - 1 {
        let seed = record::replicate_seed(base_cfg.seed, run_idx);
        let cfg = Config {
            seed,
            ..base_cfg.clone()
        };
        let result = run_with_client_observed(&cfg, build_client(&cfg), |_| {
            if let Some(stage) = discarded.as_mut() {
                stage.tick();
            }
        })
        .unwrap_or_else(|e| panic!("run failed: {e}"));
        print_result_line(run_idx + 1, runs, seed, &result);
    }
    if let Some(stage) = discarded {
        stage.close();
    }

    // `--runs N` は掃引ではない．同じ条件を N 本回して最後の結果だけを残すのが
    // 移行前からの動きなので，master_seed には実際に世界を支配した
    // derive_seed(seed, [N-1]) を書き，replicate_index を N-1 にする．根のシードは
    // /base_seed にあり，seed_pointers 経由で execution_hash に残る．
    let last_seed = record::replicate_seed(base_cfg.seed, runs - 1);
    let last_cfg = Config {
        seed: last_seed,
        ..base_cfg.clone()
    };

    let client = build_client(&last_cfg);
    let llm = client.as_ref().map(|c| {
        record::llm_block(
            c.inner().model(),
            c.inner().endpoint(),
            last_cfg.llm.temperature,
        )
    });

    let mut options = RunOptions::new(EXPERIMENT, "run")
        .repo_id(REPO_ID)
        .domain(DOMAIN)
        .results_root(&args.output_dir)
        .parameters(&ReplicateGroupParameters {
            condition: ConditionParameters::from_config(&base_cfg),
            runs,
            base_seed: base_cfg.seed,
        })
        .expect("runvault: parameters の組み立てに失敗")
        .hash_exclude(HASH_EXCLUDE)
        .seed_pointers(GROUP_SEED_POINTERS)
        .master_seed(last_seed)
        .replicate_index((runs - 1) as u64)
        .replication(record::replication());
    if let Some(llm) = llm {
        options = options.llm(llm);
    }
    let mut rv = Run::start(options).expect("runvault: run の開始に失敗");

    let mut stage = rv.stage("steps", last_cfg.t_max as usize);
    let result = simulate_into(&mut rv, &last_cfg, client, &mut stage);
    // manifest.csv は finish() で封をされる．その後に 1 行足せば，manifest が
    // 食い違うダイジェストを持つことになる．
    stage.close();
    print_result_line(runs, runs, last_seed, &result);

    let calls = result.metadata.total();
    let cache_hits = result.metadata.cache_hits();
    let hit_rate = result.metadata.cache_hit_rate();
    let model = result.llm_model.clone();
    let dir = rv.finish().expect("runvault: run の完了に失敗");

    println!("----------------------------------------------------------------------");
    if last_cfg.decision_mode.is_llm() {
        println!(
            "LLM calls: {} | cache-hit: {} ({:.1}%) | model: {}",
            calls,
            cache_hits,
            hit_rate * 100.0,
            model,
        );
    }
    println!("run → {}", dir.display());
    println!("metrics.csv がステップごとの時系列と run スコープの指標，events.jsonl が従業員ごとの最終状態．");
}

// --------------------------------------------------------------------------- //
// sweep
// --------------------------------------------------------------------------- //

/// スイープ親 run の実験条件（グリッド定義そのもの）．
#[derive(Serialize)]
struct SweepParameters {
    decision_mode: &'static str,
    n_teams: usize,
    team_size: usize,
    psafety_learn_values: Vec<f64>,
    p_retaliate_values: Vec<f64>,
    motive_init_defensive_values: Vec<f64>,
    runs: usize,
    t_max: u64,
    base_seed: u64,
}

fn cmd_sweep(args: SweepArgs) {
    let decision_mode = parse_decision_mode(&args.decision_mode).unwrap_or_else(|e| panic!("{e}"));

    let psafety_vals = parse_f64_list(&args.psafety_learn);
    let retaliate_vals = parse_f64_list(&args.p_retaliate);
    let defensive_vals = parse_f64_list(&args.motive_init_defensive);

    let n_cells = psafety_vals.len() * retaliate_vals.len() * defensive_vals.len();
    let n_total = n_cells * args.runs;

    // 親 run: グリッド定義そのものを parameters に持つ．個別セルの指標は書かない．
    // 親は 1 本のシミュレーションではないので master_seed を名乗らない．base seed は
    // /base_seed と seed_pointers 経由で execution_hash に残る．
    let parent = Run::start(
        RunOptions::new(EXPERIMENT, "sweep")
            .repo_id(REPO_ID)
            .domain(DOMAIN)
            .results_root(&args.output_dir)
            .parameters(&SweepParameters {
                decision_mode: decision_mode.label(),
                n_teams: args.n_teams,
                team_size: args.team_size,
                psafety_learn_values: psafety_vals.clone(),
                p_retaliate_values: retaliate_vals.clone(),
                motive_init_defensive_values: defensive_vals.clone(),
                runs: args.runs,
                t_max: args.t_max,
                base_seed: args.seed,
            })
            .expect("runvault: sweep の parameters の組み立てに失敗")
            .seed_pointers(GROUP_SEED_POINTERS)
            .sweep_parent()
            .replication(record::replication()),
    )
    .expect("runvault: sweep 親 run の開始に失敗");

    let lineage = Lineage {
        sweep_id: parent.sweep_id().map(str::to_string),
        parent_run_uid: Some(parent.run_uid().to_string()),
        ..Default::default()
    };

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
    println!("base seed: {}", args.seed);
    println!("output: {}", parent.dir().display());
    println!("------------------------------------------------------------");

    // グリッド全体で stage を 1 つ．セルごとに開け直すと小さな 100% が並ぶだけで，
    // スイープ全体のどこにいるかは分からない．セルは決定モードを共有していて
    // （掃引しているのは ψ_learn・p_retaliate・防衛的動機の初期値で，どれも
    // 仕事の量を変えない）1 ステップの費用は全セルで同じなので，重みではなく数える．
    let mut stage = parent.stage("steps", n_total * args.t_max as usize);

    let mut idx = 0usize;
    for &psl in &psafety_vals {
        for &pr in &retaliate_vals {
            for &dfn in &defensive_vals {
                let cell_cfg = Config {
                    n_teams: args.n_teams,
                    team_size: args.team_size,
                    decision_mode,
                    psafety_learn: psl,
                    p_retaliate: pr,
                    motive_init: motive_init_with_defensive(dfn),
                    t_max: args.t_max,
                    runs: args.runs,
                    seed: args.seed,
                    ..Config::default()
                };
                ensure_cache_dir(&cell_cfg);

                // 子は «そのセルの試行群» そのもの．base seed とセル座標からすべての
                // 試行シードが決まるので master_seed は base seed であり，同一セルの
                // 繰り返しは無いので replicate_index は 0．
                let mut child = Run::start(
                    RunOptions::new(EXPERIMENT, "sweep-point")
                        .repo_id(REPO_ID)
                        .domain(DOMAIN)
                        .results_root(&args.output_dir)
                        .parameters(&ReplicateGroupParameters {
                            condition: ConditionParameters::from_config(&cell_cfg),
                            runs: args.runs,
                            base_seed: args.seed,
                        })
                        .expect("runvault: 子 run の parameters の組み立てに失敗")
                        .hash_exclude(HASH_EXCLUDE)
                        .seed_pointers(GROUP_SEED_POINTERS)
                        .master_seed(args.seed)
                        .replicate_index(0)
                        .lineage(lineage.clone())
                        .replication(record::replication()),
                )
                .expect("runvault: sweep 子 run の開始に失敗");

                let mut trials: Vec<record::TrialOutcome> = Vec::with_capacity(args.runs);
                for run_idx in 0..args.runs {
                    idx += 1;
                    let seed = record::sweep_trial_seed(args.seed, psl, pr, dfn, run_idx);
                    let cfg = Config {
                        seed,
                        runs: 1,
                        ..cell_cfg.clone()
                    };
                    let result =
                        run_with_client_observed(&cfg, build_client(&cfg), |_| stage.tick())
                            .unwrap_or_else(|e| panic!("sweep run failed: {e}"));
                    let outcome = record::TrialOutcome::from_result(&result);
                    record::log_trial(&mut child, run_idx, seed, args.t_max, &outcome);
                    if idx.is_multiple_of(10) || idx == n_total {
                        println!(
                            "[{}/{}] ψ_learn={:.2} p_ret={:.2} def_init={:.2} run={} silence={:.3} def={:.3}",
                            idx, n_total, psl, pr, dfn, run_idx, outcome.silence_rate,
                            outcome.motive_mix[2]
                        );
                    }
                    trials.push(outcome);
                }
                record::log_cell_summary(&mut child, &trials);
                child.finish().expect("runvault: sweep 子 run の完了に失敗");
            }
        }
    }

    stage.close();

    let dir = parent
        .finish()
        .expect("runvault: sweep 親 run の完了に失敗");
    println!("------------------------------------------------------------");
    println!("sweep done.");
    println!("親 run     → {}", dir.display());
    println!("セル {n_cells} 個 → 子 run (subcommand=sweep-point)．試行 1 本が events.jsonl の terminal 行 1 本．");
}

// --------------------------------------------------------------------------- //
// ablate
// --------------------------------------------------------------------------- //

/// アブレーション親 run の実験条件（比較する決定モードの一覧）．
///
/// モードごとに条件が違うので，どれか 1 つを親の条件として名乗ることはできない．
#[derive(Serialize)]
struct AblateParameters {
    decision_modes: Vec<&'static str>,
    n_teams: usize,
    team_size: usize,
    runs: usize,
    t_max: u64,
    base_seed: u64,
}

fn cmd_ablate(args: AblateArgs) {
    let modes: Vec<DecisionMode> = args
        .decision_modes
        .split([',', ' '])
        .filter(|t| !t.is_empty())
        .map(|t| parse_decision_mode(t).unwrap_or_else(|e| panic!("{e}")))
        .collect();

    // (モード × 試行) はそれぞれ模型の別々の実行なので，掃引と同じ «親 + セル子» に
    // する．セルは決定モード 1 つで，そのセルの試行が terminal 行になる．
    let parent = Run::start(
        RunOptions::new(EXPERIMENT, "ablate")
            .repo_id(REPO_ID)
            .domain(DOMAIN)
            .results_root(&args.output_dir)
            .parameters(&AblateParameters {
                decision_modes: modes.iter().map(|m| m.label()).collect(),
                n_teams: args.n_teams,
                team_size: args.team_size,
                runs: args.runs,
                t_max: args.t_max,
                base_seed: args.seed,
            })
            .expect("runvault: ablate の parameters の組み立てに失敗")
            .seed_pointers(GROUP_SEED_POINTERS)
            .sweep_parent()
            .replication(record::replication()),
    )
    .expect("runvault: ablate 親 run の開始に失敗");

    let lineage = Lineage {
        sweep_id: parent.sweep_id().map(str::to_string),
        parent_run_uid: Some(parent.run_uid().to_string()),
        ..Default::default()
    };

    println!("=== brinsfield-ablate ===");
    println!(
        "modes: {:?} | runs/mode={} | t_max={} | seed={}",
        modes.iter().map(|m| m.label()).collect::<Vec<_>>(),
        args.runs,
        args.t_max,
        args.seed,
    );
    println!("output: {}", parent.dir().display());
    println!("------------------------------------------------------------");

    for &mode in &modes {
        // モードごとに別の stage にする．llm と rule_* では 1 ステップの費用が
        // 桁で違う（前者は従業員 1 人ごとにモデル呼び出し，後者は算術だけ）ので，
        // 1 つの stage にまとめると数からの外挿が «自信をもって外れた見積もり» に
        // なる．比を重みで書くことはできない — その比はエンドポイント・モデル・
        // キャッシュの当たり方で決まり，走らせる前には誰も知らないからである．
        // 分ければ，stage の中の 1 ステップはすべて同じ費用になる．
        let mut stage = parent.stage(mode.label(), args.runs * args.t_max as usize);

        let cell_cfg = Config {
            n_teams: args.n_teams,
            team_size: args.team_size,
            decision_mode: mode,
            t_max: args.t_max,
            runs: args.runs,
            seed: args.seed,
            ..Config::default()
        };
        ensure_cache_dir(&cell_cfg);

        let mut child = Run::start(
            RunOptions::new(EXPERIMENT, "ablate-point")
                .repo_id(REPO_ID)
                .domain(DOMAIN)
                .results_root(&args.output_dir)
                .parameters(&ReplicateGroupParameters {
                    condition: ConditionParameters::from_config(&cell_cfg),
                    runs: args.runs,
                    base_seed: args.seed,
                })
                .expect("runvault: 子 run の parameters の組み立てに失敗")
                .hash_exclude(HASH_EXCLUDE)
                .seed_pointers(GROUP_SEED_POINTERS)
                .master_seed(args.seed)
                .replicate_index(0)
                .lineage(lineage.clone())
                .replication(record::replication()),
        )
        .expect("runvault: ablate 子 run の開始に失敗");

        let mut trials: Vec<record::TrialOutcome> = Vec::with_capacity(args.runs);
        for run_idx in 0..args.runs {
            let seed = record::ablate_trial_seed(args.seed, mode, run_idx);
            let cfg = Config {
                seed,
                runs: 1,
                ..cell_cfg.clone()
            };
            let result = run_with_client_observed(&cfg, build_client(&cfg), |_| stage.tick())
                .unwrap_or_else(|e| panic!("ablate run failed: {e}"));
            let outcome = record::TrialOutcome::from_result(&result);
            record::log_trial(&mut child, run_idx, seed, args.t_max, &outcome);
            trials.push(outcome);
        }
        record::log_cell_summary(&mut child, &trials);

        let n = trials.len().max(1) as f64;
        let mean_kl: f64 = trials.iter().map(|t| t.kl_to_reference).sum::<f64>() / n;
        let mean_def: f64 = trials.iter().map(|t| t.motive_mix[2]).sum::<f64>() / n;
        println!(
            "{:<10} mean KL→ref={:.4} mean defensive_share={:.3}",
            mode.label(),
            mean_kl,
            mean_def,
        );

        stage.close();
        child
            .finish()
            .expect("runvault: ablate 子 run の完了に失敗");
    }

    let dir = parent
        .finish()
        .expect("runvault: ablate 親 run の完了に失敗");
    println!("------------------------------------------------------------");
    println!("ablation done.");
    println!("親 run       → {}", dir.display());
    println!("モード {} 個 → 子 run (subcommand=ablate-point)．試行 1 本が events.jsonl の terminal 行 1 本．", modes.len());
}

// --------------------------------------------------------------------------- //
// reproduce
// --------------------------------------------------------------------------- //

fn cmd_reproduce(args: ReproduceArgs) {
    let mode = parse_decision_mode(&args.decision_mode).unwrap_or_else(|e| panic!("{e}"));

    let cfg = Config {
        decision_mode: mode,
        t_max: args.t_max,
        seed: args.seed,
        runs: 1,
        ..Config::default()
    };
    ensure_cache_dir(&cfg);

    let client = build_client(&cfg);
    let llm = client
        .as_ref()
        .map(|c| record::llm_block(c.inner().model(), c.inner().endpoint(), cfg.llm.temperature));

    let mut options = RunOptions::new(EXPERIMENT, "reproduce")
        .repo_id(REPO_ID)
        .domain(DOMAIN)
        .results_root(&args.output_dir)
        .parameters(&SingleRunParameters {
            condition: ConditionParameters::from_config(&cfg),
            seed: cfg.seed,
        })
        .expect("runvault: parameters の組み立てに失敗")
        .hash_exclude(HASH_EXCLUDE)
        .seed_pointers(SINGLE_SEED_POINTERS)
        .master_seed(cfg.seed)
        .replication(record::replication());
    if let Some(llm) = llm {
        options = options.llm(llm);
    }
    let mut rv = Run::start(options).expect("runvault: run の開始に失敗");

    let mut stage = rv.stage("steps", cfg.t_max as usize);
    let result = simulate_into(&mut rv, &cfg, client, &mut stage);
    stage.close();

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
    // 定常平均は «run 全体を 1 つの値で表す量» なので step を持たない run スコープの
    // 指標にする．ステップごとの `motive_mix_*` とは別の数なので二重記録ではない．
    let steady_names: Vec<String> = labels
        .iter()
        .map(|l| format!("steady_state_motive_mix_{l}"))
        .collect();
    let steady_values: Vec<(&str, f64)> = steady_names
        .iter()
        .zip(emergent.iter())
        .map(|(name, &v)| (name.as_str(), v))
        .collect();
    rv.log_metrics("run", &steady_values)
        .expect("定常動機分布の記録に失敗");

    println!(
        "=== Brinsfield (2013) — reproduce (mode={}) ===",
        mode.label()
    );
    println!("run: {}", rv.dir().display());
    println!("steady-state motive_mix (mean over t >= {half}):");
    println!("  {:<13} {:>10} {:>10}", "motive", "emergent", "reference");
    for i in 0..6 {
        println!(
            "  {:<13} {:>10.4} {:>10.4}",
            labels[i], emergent[i], REFERENCE_MOTIVE_MIX[i]
        );
    }
    // 許容幅と PASS/FAIL は論文の報告値でも観測値でもない（こちらが決めた判定基準
    // である）ので，記録には残さずコンソールにだけ出す．
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

    let dir = rv.finish().expect("runvault: run の完了に失敗");
    println!("run → {}", dir.display());
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
