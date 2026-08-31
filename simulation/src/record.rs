//! runvault への記録の共通部分．
//!
//! 論文メタデータ (research) は `run` / `sweep` / `ablate` / `reproduce` のどれでも
//! 同一なので，ここ 1 箇所で組み立てる．実験条件 (`parameters`)，ステップごとの指標，
//! run 全体を 1 つの値で表す指標，動機 × correlate の相関，従業員 1 人ごとの最終状態，
//! そして掃引 1 セルぶんの試行の書き方もここに集める．
//!
//! # どの形をどこへ置いたか
//!
//! - **ステップごとの集団指標** (`silence_rate` / `motive_mix_*` / …) は
//!   `metrics.csv` の `scope=run` のステップ指標．旧 `metrics.csv` の 1 行が
//!   1 ステップぶんの指標群になる．
//! - **run 全体を 1 つの値で表す量** (`final_round` / `n_units` / 動機 × correlate の
//!   相関 30 本) は `step` を持たない `scope=run` の指標．旧 `correlations.csv` の
//!   30 行がこれにあたる — `motive` と `correlate` は «どの数か» を指す名前であって
//!   値ではないので，`motive_mix_*` と同じく名前に畳む．
//! - **従業員 1 人ごとの最終状態** (旧 `agents.csv`) は `x.brinsfield2013.agent`
//!   イベント．`metrics.csv` の主キーは (`name`, `step`, `step_unit`, `scope`) なので，
//!   従業員ごとの `fear` を並べると全行が同じキーを名乗って衝突する．
//! - **掃引・アブレーションの試行** は `events.jsonl` の `terminal` 行．旧
//!   `sweep_summary.csv` / `ablation_summary.csv` の 1 行がこの 1 行になる．
//!
//! 旧 `motive_mix.csv` は書かない．6 列とも `metrics.csv` の `motive_mix_*` と
//! 同じ数で，同じ数を 2 箇所に置かないためである．
//!
//! # 反復 (`--runs N`) を子 run に割らない理由
//!
//! `run` の `--runs N` は掃引ではない．同じ条件を N 本回して**最後の結果だけ**を
//! 残すのが移行前からの動きで，途中の N-1 本はディスクに何も残さない．そこで
//! `master_seed` には実際に世界を支配した `derive_seed(seed, [N-1])` を書き，
//! `replicate_index` を N-1 にした (knoll2013 と同じ)．根のシードは
//! `/parameters.base_seed` にある．
//!
//! 一方 `sweep` / `ablate` の (セル × 試行) は模型の別々の実行である．こちらは
//! «セル 1 つ = 子 run 1 本，試行 = `terminal` 行 1 本» にした — 掃引が見るのは
//! 各試行の**最終ステップだけ**で，試行は自分の時系列を持たないからである
//! (fujimura2019 / hegselmann2002 と同じ形)．

use runvault::{Llm, Replication, Run, Target, Work};
use serde::Serialize;

use crate::config::{Config, DecisionMode};
use crate::simulation::{AgentRow, CorrelationRow, MetricsRow, SimulationResult};

/// runvault 上の実験名．`runvault path --experiment` に渡す値でもある．
pub const EXPERIMENT: &str = "brinsfield-silence";

/// リポジトリの安定 id．git remote の名前とは独立に固定する．
pub const REPO_ID: &str = "brinsfield2013";

/// 分野．
///
/// 従業員属性・Watts–Strogatz 網・スケジューラ・報復抽選がいずれも乱数駆動なので
/// `simulation` (= `master_seed` が必須)．`--decision-mode llm` では LLM が意思決定を
/// 担うが，測っているのはモデルの安全性ではなく組織に創発する沈黙動機の分布なので
/// `llm-safety` ではない．LLM 側の同一性は `run.json` の `llm` ブロックが持つ．
pub const DOMAIN: &str = "simulation";

/// 時間軸の単位．モデルの刻みは socsim エンジンの離散ステップなので語彙の `step`．
const T_UNIT: &str = "step";

/// 指標の粒度．集団指標も相関も従業員全体の集約なので `run`．
const SCOPE: &str = "run";

/// 従業員 1 人の最終状態を表す実験固有のイベント種別．コア語彙に無いので
/// `x.<repo_id>.<name>` を使う．
pub const AGENT_EVENT: &str = "x.brinsfield2013.agent";

/// 試行の終端．生存時間解析と同じ形なのでコア語彙の `terminal` をそのまま使う．
const TERMINAL: &str = "terminal";

/// 試行を «いつ観測したか» を残す行．`verify --deep` は `terminal` の `unit_id` が
/// `observation` にも現れることを要求する．
const OBSERVATION: &str = "observation";

// --------------------------------------------------------------------------- //
// 論文メタデータ
// --------------------------------------------------------------------------- //

/// この再現実験が対象としている論文．
///
/// target は 2 つ持つ．`run` / `reproduce` が照合するのは «defensive 動機は全沈黙の
/// 12.65% にすぎない» という Study 1 の主張で，`ablate` が照合するのは «沈黙動機は
/// 6 次元であって 1〜5 次元ではない» という Study 3 の主張である．
pub fn replication() -> Replication {
    let mut work = Work::doi("10.1002/job.1829")
        .title(
            "Employee Silence Motives: Investigation of Dimensionality and Development of Measures",
        )
        .year(2013)
        .source_version("published");
    // vault 側の同定にも使えるよう paper-id も残す (work_id は DOI 側)．
    work.paper_id = Some("P00001814".to_string());
    Replication::new(work)
        .target(Target::claim(
            "defensive-share-minority",
            "Fear-based (defensive) motives account for only 12.65% of employee silence",
        ))
        .target(Target::claim(
            "six-motive-dimensionality",
            "Employee silence motives form six distinct dimensions rather than one to five",
        ))
        .obsidian_note("研究/98_論文レポート/80-再現実験/実装完了/brinsfield2013/設計書.md")
}

/// 原著が報告した値のうち，この実装が run 1 本の中で直接測るもの．
///
/// 書けるのは defensive 動機の比率 12.65% (Study 1，288/2277) の 1 行だけである．
/// これは原著が «全沈黙のうち恐怖ベースの動機は 12.65%» と印字している数で，
/// ステップ指標 `motive_mix_defensive` と同じ量を指す．
///
/// 次のものは書かない．
///
/// - `calibration::REFERENCE_MOTIVE_MIX` の 6 値 — `kl_to_reference` の基準分布だが，
///   原著が印字した分布ではない．12.65% と «ineffectual が最頻出» «deviant は稀» と
///   いう Study 1 の記述から**本設計書が組み立てた**分布である．
/// - `DEFENSIVE_SHARE_TOL` (±3%) / `INEFFECTUAL_FLOOR` (30%) / `DEVIANT_CEILING` (8%)
///   — こちらが決めた許容幅と目標であって，原著の報告値ではない．
/// - Study 2/3/4 の EFA 分散説明率・CFA 適合度・Cronbach α・ΔR² — 原著が印字した数
///   だが，Rust 側の run はこれらを測らない (Python の Track A が測る)．run スコープに
///   対応する観測値が無い欄に論文値だけを置かない．
pub fn log_paper_reference(run: &mut Run) {
    run.log_reference("motive_mix_defensive", 0.1265)
        .target("defensive-share-minority")
        .source("Brinsfield (2013) Study 1: defensive (fear-based) motives = 288/2277 = 12.65% of all silence incidents")
        .send()
        .expect("原著の報告値の記録に失敗");
}

// --------------------------------------------------------------------------- //
// LLM ブロック
// --------------------------------------------------------------------------- //

/// 実際に応答したバックエンドを `llm` ブロックに落とす．
///
/// `model` / `endpoint` はクライアントが名乗った値をそのまま使う．`provider` は
/// runvault の語彙ではなく自由記述なので，endpoint から «どのゲートウェイが答えたか»
/// を決める (`mock://…` はオフラインの scripted クライアント)．
///
/// rule モードではこれを呼ばない．LLM を 1 度も叩かない run に `llm` ブロックを
/// 付けると，存在しないモデル (旧 `llm_meta.json` の `"model": "none"`) を名乗る
/// ことになる．
pub fn llm_block(model: &str, endpoint: &str, temperature: f32) -> Llm {
    let provider = if endpoint.starts_with("mock://") {
        "mock"
    } else if endpoint.contains("openai") {
        "openai"
    } else {
        "ollama"
    };
    Llm {
        provider: provider.to_string(),
        model_snapshot: model.to_string(),
        temperature: Some(temperature as f64),
        // プロンプトは従業員ごとに組み立てられ，固定の system prompt を持たない．
        // 無いものを hash しない．
        system_prompt_hash: None,
    }
}

// --------------------------------------------------------------------------- //
// 実験条件 (parameters)
// --------------------------------------------------------------------------- //

/// 条件そのもの．シードも反復数も含まない．
///
/// 旧 `sweep_config.json` / `ablate` の `config.json` は掃引軸と `n_teams` /
/// `team_size` / `t_max` しか持たず，`n_levels`・網の形・`motive_init` の土台・
/// `beta` 群・`motive_learn_rate`・衝撃・`prompt_version` はどこにも書かれていなかった
/// (すべて `Config::default()` から来る)．どれも結果を決める量なので，`config_hash` が
/// 条件の同一性を判定できるよう全部入れる．
#[derive(Serialize)]
pub struct ConditionParameters {
    pub n_teams: usize,
    pub team_size: usize,
    pub n_levels: u8,
    pub n_employees: usize,
    pub network_kind: crate::config::NetworkKind,
    pub network_k: usize,
    pub network_beta: f64,
    pub supervisor_homogeneity: f64,
    /// `DecisionMode::label()` の文字列．serde の派生形 (`rule6dim`) ではなく，
    /// CLI と旧 CSV が使ってきた綴り (`rule_6dim`) に揃える — 掃引親の
    /// `decision_modes` と同じ語彙でないと，条件を機械で突き合わせられない．
    pub decision_mode: &'static str,
    pub prompt_version: u8,
    pub motive_init: crate::config::MotiveInit,
    pub beta: crate::config::BetaGroup,
    pub motive_learn_rate: f64,
    pub psafety_learn: f64,
    pub p_retaliate: f64,
    pub shock_t: Option<u64>,
    pub shock_magnitude: f64,
    pub t_max: u64,
    pub llm_temperature: f32,
    pub llm_seed: u64,
    /// プロンプト → 応答キャッシュの置き場．条件ではなく置き場なので
    /// `hash_exclude` で `config_hash` から外す ([`HASH_EXCLUDE`])．
    pub llm_cache_path: Option<String>,
}

/// `config_hash` から外すポインタ．
///
/// キャッシュのパスは «どこに置いたか» であって条件ではない．同じ条件の run を
/// 別のキャッシュファイルで回しても同じ条件である．
pub const HASH_EXCLUDE: [&str; 1] = ["/llm_cache_path"];

impl ConditionParameters {
    /// [`Config`] から条件だけを取り出す．
    pub fn from_config(cfg: &Config) -> Self {
        ConditionParameters {
            n_teams: cfg.n_teams,
            team_size: cfg.team_size,
            n_levels: cfg.n_levels,
            n_employees: cfg.n_employees(),
            network_kind: cfg.network_kind,
            network_k: cfg.network_k,
            network_beta: cfg.network_beta,
            supervisor_homogeneity: cfg.supervisor_homogeneity,
            decision_mode: cfg.decision_mode.label(),
            prompt_version: cfg.prompt_version,
            motive_init: cfg.motive_init,
            beta: cfg.beta,
            motive_learn_rate: cfg.motive_learn_rate,
            psafety_learn: cfg.psafety_learn,
            p_retaliate: cfg.p_retaliate,
            shock_t: cfg.shock_t,
            shock_magnitude: cfg.shock_magnitude,
            t_max: cfg.t_max,
            llm_temperature: cfg.llm.temperature,
            llm_seed: cfg.llm.seed,
            llm_cache_path: cfg.llm.cache_path.clone(),
        }
    }
}

/// 「条件 1 つを 1 本回した」run の実験条件 (`reproduce` / `mock-smoke`)．
///
/// `seed` はその run が実際に使ったシードで `master_seed` と同じ値である．
/// `seed_pointers` で seed として宣言するので `config_hash` からは外れる．
#[derive(Serialize)]
pub struct SingleRunParameters {
    #[serde(flatten)]
    pub condition: ConditionParameters,
    pub seed: u64,
}

/// 「1 条件 + その反復群」の実験条件 (`run` の親，`sweep` / `ablate` のセル子)．
///
/// `base_seed` は反復ごと・試行ごとのシードを派生させる元であって，それ自体で
/// 世界を回す値ではない．`terminal` 行のシード欄を `seed` と別名にしてあるのは，
/// `runvault.read.sweep_events_table` が parameters の列を event の同名列に上書き
/// するためである — 同じ名前にすると試行ごとのシードが base seed で黙って潰れる．
#[derive(Serialize)]
pub struct ReplicateGroupParameters {
    #[serde(flatten)]
    pub condition: ConditionParameters,
    pub runs: usize,
    pub base_seed: u64,
}

/// `reproduce` / `mock-smoke` の seed ポインタ．
pub const SINGLE_SEED_POINTERS: [&str; 1] = ["/seed"];
/// `run` 親 / `sweep`・`ablate` のセル子の seed ポインタ．
pub const GROUP_SEED_POINTERS: [&str; 1] = ["/base_seed"];

// --------------------------------------------------------------------------- //
// シミュレーション 1 本ぶんの記録
// --------------------------------------------------------------------------- //

/// シミュレーション 1 本ぶんを run へ書く．
pub fn log_simulation(run: &mut Run, result: &SimulationResult) {
    for m in &result.metrics_rows {
        log_step(run, m);
    }
    log_run_scope(run, result);
    log_correlations(run, &result.correlation_rows);
    log_agents(run, result.final_round, &result.agent_rows);
}

/// [`MetricsRow`] の 18 の数値フィールドを 1 ステップぶんまとめて書く．
///
/// `t` は時間軸そのものなので値としては書かない．
///
/// `motive_mix_*` と `motive_vec_mean_*` の 6 本ずつは «動機というカテゴリに番号を
/// 振ったもの» ではない．各動機が沈黙者に占める割合／従業員の動機ベクトルの平均と
/// いう «ステップごとの数» が 6 つずつあるだけである (mix の 6 本の和は 1，沈黙者が
/// 1 人もいなければ 0)．逆に `agents.csv` の `primary_motive` はラベルなので指標に
/// しない．
///
/// `n_distinct_motives_active` は «活性な動機の本数» という数え上げで，カテゴリの
/// 番号ではない．
fn log_step(run: &mut Run, m: &MetricsRow) {
    run.log_metrics_at(
        m.t,
        T_UNIT,
        SCOPE,
        &[
            ("silence_rate", m.silence_rate),
            ("motive_mix_ineffectual", m.motive_mix_ineffectual),
            ("motive_mix_relational", m.motive_mix_relational),
            ("motive_mix_defensive", m.motive_mix_defensive),
            ("motive_mix_diffident", m.motive_mix_diffident),
            ("motive_mix_disengaged", m.motive_mix_disengaged),
            ("motive_mix_deviant", m.motive_mix_deviant),
            ("motive_vec_mean_ineffectual", m.motive_vec_mean_ineffectual),
            ("motive_vec_mean_relational", m.motive_vec_mean_relational),
            ("motive_vec_mean_defensive", m.motive_vec_mean_defensive),
            ("motive_vec_mean_diffident", m.motive_vec_mean_diffident),
            ("motive_vec_mean_disengaged", m.motive_vec_mean_disengaged),
            ("motive_vec_mean_deviant", m.motive_vec_mean_deviant),
            (
                "n_distinct_motives_active",
                m.n_distinct_motives_active as f64,
            ),
            ("climate_of_silence", m.climate_of_silence),
            ("org_performance", m.org_performance),
            ("issue_salience", m.issue_salience),
            ("kl_to_reference", m.kl_to_reference),
        ],
    )
    .unwrap_or_else(|e| panic!("step {} の指標の記録に失敗: {e}", m.t));
}

/// run 全体を 1 つの値で表す指標．
///
/// `n_units` は予約指標名で «観測主体の数» — このモデルでは従業員の数である．
/// 実行時間は `status.json` の `duration_sec` が正本なので指標にしない．
fn log_run_scope(run: &mut Run, result: &SimulationResult) {
    run.log_metrics(
        SCOPE,
        &[
            ("n_units", result.agent_rows.len() as f64),
            ("final_round", result.final_round as f64),
        ],
    )
    .expect("run スコープの指標の記録に失敗");
}

/// 動機 × correlate の Pearson $r$ を run スコープの指標として書く (旧
/// `correlations.csv` の 30 行)．
///
/// 動機 (6) と correlate (ψ / fear / 神経症 / 外向性 / 沈黙風土の 5) は，値そのもの
/// ではなく «どの数か» を指す名前なので，`motive_mix_*` と同じく名前に畳む．30 セルは
/// それぞれ別の名前を名乗るので主キー (name, step, step_unit, scope) は衝突しない．
/// 最終ステップの状態から求める集約であってステップごとの系列ではないので `step` は
/// 持たない．
///
/// `metrics::pearson` は分散 0 などの退化した入力に 0 を返す契約なので，その動機を
/// 1 人も持たなかったセルも 0 の行になる．欠測を 0 で埋めているのではなく，その関数が
/// 返した値をそのまま書いている．
fn log_correlations(run: &mut Run, rows: &[CorrelationRow]) {
    let names: Vec<String> = rows
        .iter()
        .map(|r| format!("corr_{}_{}", r.motive, r.correlate))
        .collect();
    let values: Vec<(&str, f64)> = names
        .iter()
        .zip(rows)
        .map(|(name, row)| (name.as_str(), row.pearson_r))
        .collect();
    run.log_metrics(SCOPE, &values)
        .expect("動機 × correlate 相関の記録に失敗");
}

/// LLM 呼び出しの内訳を run スコープの指標として書く (旧 `llm_meta.json`)．
///
/// rule モードでは呼ばない．0 回という数を書くこと自体は嘘ではないが，LLM を配線して
/// いない run に LLM の指標が並ぶと `llm` ブロックの有無と食い違って見える．
///
/// cache-hit 率は呼び出しが 1 本も無いとき «0» ではなく «定義できない» ので，
/// そのときは行そのものを書かない (欠測を 0 で埋めない)．
///
/// `tokens_in` / `tokens_out` / `cost_usd` は書かない — socsim-llm の
/// `MetadataCollector` はトークン数も費用も持たないので，予約名に入れる値が無い．
/// 旧 `llm_meta.json` の `determinism_note` は数でも条件でもないのでコードと
/// ドキュメントに残す．
pub fn log_llm_usage(run: &mut Run, result: &SimulationResult) {
    let calls = result.metadata.total();
    let mut values: Vec<(&str, f64)> = vec![
        ("llm_calls", calls as f64),
        ("llm_cache_hits", result.metadata.cache_hits() as f64),
    ];
    if calls > 0 {
        values.push(("llm_cache_hit_rate", result.metadata.cache_hit_rate()));
    }
    run.log_metrics(SCOPE, &values)
        .expect("LLM 呼び出しの内訳の記録に失敗");
}

// --------------------------------------------------------------------------- //
// 従業員 1 人ごとの最終状態
// --------------------------------------------------------------------------- //

/// `events.jsonl` に書く従業員 1 人の最終状態 (旧 `agents.csv` の 1 行)．
///
/// この行は `metrics.csv` には置けない．主キーが (name, step, step_unit, scope) なので，
/// 従業員ごとの `fear` を並べると全行が同じキーを名乗って衝突する (`scope=agent` に
/// しても行を分ける列が無い)．かといって従業員 1 人を子 run に割るのも実態と違う —
/// 1 回の実行は 1 つの組織を丸ごと回すのであって，起きていない N 本の実行を主張する
/// ことになる．従業員は «1 回の実行の中で観測された対象» なので，予約語 `unit_id`
/// (観測の主体) を持つイベントとして書く．
///
/// `expression` と `primary_motive` はラベルであって数ではないので，そもそも指標に
/// できない．沈黙していない従業員に沈黙動機は無いので欄そのものを落とす (旧 CSV は
/// `-` という番兵を書いていた．欠測は «無い» と書く方が後から見分けられる)．
///
/// コア語彙の `observation` ではなく実験固有の種別にしたのは，`observation` が到達
/// 時間の観測 1 点という意味を持つ行だからである．この run は `terminal` を書かない
/// ので observation の要求も生じない．
#[derive(Serialize)]
struct AgentEvent<'a> {
    unit_id: String,
    t: u64,
    t_unit: &'static str,
    agent_id: u64,
    team: usize,
    level: u8,
    tenure: u32,
    expression: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    primary_motive: Option<&'a str>,
    motive_ineffectual: f64,
    motive_relational: f64,
    motive_defensive: f64,
    motive_diffident: f64,
    motive_disengaged: f64,
    motive_deviant: f64,
    fear: f64,
    psafety: f64,
    ivt: f64,
    neuroticism: f64,
    extraversion: f64,
    private_concern: f64,
}

/// 従業員 1 人につき 1 行書く．
fn log_agents(run: &mut Run, final_round: u64, rows: &[AgentRow]) {
    for row in rows {
        let event = AgentEvent {
            unit_id: format!("agent-{}", row.agent_id),
            t: final_round,
            t_unit: T_UNIT,
            agent_id: row.agent_id,
            team: row.team,
            level: row.level,
            tenure: row.tenure,
            expression: &row.expression,
            primary_motive: if row.primary_motive == "-" {
                None
            } else {
                Some(&row.primary_motive)
            },
            motive_ineffectual: row.motive_ineffectual,
            motive_relational: row.motive_relational,
            motive_defensive: row.motive_defensive,
            motive_diffident: row.motive_diffident,
            motive_disengaged: row.motive_disengaged,
            motive_deviant: row.motive_deviant,
            fear: row.fear,
            psafety: row.psafety,
            ivt: row.ivt,
            neuroticism: row.neuroticism,
            extraversion: row.extraversion,
            private_concern: row.private_concern,
        };
        run.log_event(AGENT_EVENT, &event)
            .unwrap_or_else(|e| panic!("agent {} の記録に失敗: {e}", row.agent_id));
    }
}

// --------------------------------------------------------------------------- //
// 掃引 / アブレーションの試行 (events.jsonl)
// --------------------------------------------------------------------------- //

/// `events.jsonl` に書く観測行．
///
/// 数はここには書かない — 試行の最終値は [`TrialTerminal`] が正本なので，同じ数を
/// 2 箇所に置かない．この行が持つのは «その試行をいつ見たか» だけである．
#[derive(Serialize)]
struct TrialObservation<'a> {
    unit_id: &'a str,
    t: u64,
    t_unit: &'static str,
}

/// `events.jsonl` に書く試行の終端行．
///
/// 先頭 6 フィールドは runvault の予約語 (`terminal` はこれを全部要求する)．残りは
/// 自由欄で，旧 `sweep_summary.csv` / `ablation_summary.csv` の 1 行がこの 1 行に
/// 対応する．掃引軸 (`psafety_learn` / `p_retaliate` / `motive_init_defensive` /
/// `decision_mode`) はセル子 run の `parameters` にあるので，ここには書かない．
#[derive(Serialize)]
struct TrialTerminal<'a> {
    unit_id: &'a str,
    t: u64,
    t_unit: &'static str,
    outcome: &'static str,
    censored: bool,
    budget: u64,
    seed: u64,
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

/// 試行 1 本の最終値．セル 1 つぶんの集約の材料でもある．
pub struct TrialOutcome {
    pub final_round: u64,
    pub silence_rate: f64,
    pub motive_mix: [f64; 6],
    pub climate_of_silence: f64,
    pub kl_to_reference: f64,
}

impl TrialOutcome {
    /// [`SimulationResult`] の最終ステップから取り出す．
    pub fn from_result(result: &SimulationResult) -> Self {
        let last = result
            .metrics_rows
            .last()
            .expect("metrics_rows は空にならない");
        TrialOutcome {
            final_round: result.final_round,
            silence_rate: last.silence_rate,
            motive_mix: [
                last.motive_mix_ineffectual,
                last.motive_mix_relational,
                last.motive_mix_defensive,
                last.motive_mix_diffident,
                last.motive_mix_disengaged,
                last.motive_mix_deviant,
            ],
            climate_of_silence: last.climate_of_silence,
            kl_to_reference: last.kl_to_reference,
        }
    }
}

/// 試行 1 本を `observation` + `terminal` の 2 行として書く．
///
/// モデルは収束判定を持たず必ず `t_max` まで回るので，`outcome` は常に `horizon`，
/// `censored` は常に真である (打ち切りの行は `t == budget` でなければならず，
/// `final_round == t_max` なのでこれを満たす)．
pub fn log_trial(run: &mut Run, index: usize, seed: u64, t_max: u64, outcome: &TrialOutcome) {
    let unit_id = format!("trial-{index}");
    // 掃引が見るのは各試行の最終ステップだけなので，観測時刻もそこ 1 点．
    run.log_event(
        OBSERVATION,
        &TrialObservation {
            unit_id: &unit_id,
            t: outcome.final_round,
            t_unit: T_UNIT,
        },
    )
    .unwrap_or_else(|e| panic!("{unit_id} の observation の記録に失敗: {e}"));

    run.log_event(
        TERMINAL,
        &TrialTerminal {
            unit_id: &unit_id,
            t: outcome.final_round,
            t_unit: T_UNIT,
            outcome: "horizon",
            censored: true,
            budget: t_max,
            seed,
            silence_rate: outcome.silence_rate,
            motive_mix_ineffectual: outcome.motive_mix[0],
            motive_mix_relational: outcome.motive_mix[1],
            motive_mix_defensive: outcome.motive_mix[2],
            motive_mix_diffident: outcome.motive_mix[3],
            motive_mix_disengaged: outcome.motive_mix[4],
            motive_mix_deviant: outcome.motive_mix[5],
            climate_of_silence: outcome.climate_of_silence,
            kl_to_reference: outcome.kl_to_reference,
        },
    )
    .unwrap_or_else(|e| panic!("{unit_id} の terminal イベントの記録に失敗: {e}"));
}

/// 1 セルを 1 つの値で表す指標．
///
/// 試行ごとの値は `events.jsonl` の担当なので，ここには集約しか書かない．試行ごとの
/// `silence_rate` を指標にすると (`run_uid`, `step`, `scope`, `name`) が重複する．
/// 散らばりが要る図は `events.jsonl` から組み直す．
///
/// `mean_kl_to_reference` と `mean_motive_mix_defensive` は，旧 `ablate` が 1 モードに
/// つき 1 行コンソールへ印字していた 2 つの数そのものである．
pub fn log_cell_summary(run: &mut Run, trials: &[TrialOutcome]) {
    let n = trials.len();
    assert!(n > 0, "試行が 1 本もありません");
    let n_f = n as f64;
    let mean = |f: &dyn Fn(&TrialOutcome) -> f64| trials.iter().map(f).sum::<f64>() / n_f;

    run.log_metrics(
        SCOPE,
        &[
            ("n_units", n_f),
            ("mean_silence_rate", mean(&|t| t.silence_rate)),
            ("mean_motive_mix_defensive", mean(&|t| t.motive_mix[2])),
            ("mean_climate_of_silence", mean(&|t| t.climate_of_silence)),
            ("mean_kl_to_reference", mean(&|t| t.kl_to_reference)),
        ],
    )
    .expect("セル集約の記録に失敗");
}

// --------------------------------------------------------------------------- //
// シードの派生
// --------------------------------------------------------------------------- //

/// `run` の反復 1 本のシード．移行前の `cmd_run` と同じ導出である．
pub fn replicate_seed(base: u64, index: usize) -> u64 {
    socsim_core::derive_seed(base, &[index as u64])
}

/// `sweep` の試行 1 本のシード．移行前の `cmd_sweep` と同じ導出である．
pub fn sweep_trial_seed(
    base: u64,
    psafety_learn: f64,
    p_retaliate: f64,
    motive_init_defensive: f64,
    index: usize,
) -> u64 {
    socsim_core::derive_seed(
        base,
        &[
            (psafety_learn * 1000.0) as u64,
            (p_retaliate * 1000.0) as u64,
            (motive_init_defensive * 1000.0) as u64,
            index as u64,
        ],
    )
}

/// `ablate` の試行 1 本のシード．移行前の `cmd_ablate` と同じ導出である．
pub fn ablate_trial_seed(base: u64, mode: DecisionMode, index: usize) -> u64 {
    socsim_core::derive_seed(base, &[mode.n_dims() as u64, index as u64])
}

#[cfg(test)]
mod tests {
    use super::*;
    use runvault::meta::TargetKind;

    #[test]
    fn the_work_id_agrees_with_the_doi() {
        let research: runvault::meta::Research = replication().into();
        let work = research.work.expect("再現実験なので work がある");
        assert_eq!(work.work_id, "doi:10.1002/job.1829");
        assert_eq!(work.doi.as_deref(), Some("10.1002/job.1829"));
        assert_eq!(work.paper_id.as_deref(), Some("P00001814"));
    }

    #[test]
    fn the_reference_target_is_declared() {
        let research: runvault::meta::Research = replication().into();
        assert_eq!(research.targets.len(), 2);
        assert!(research
            .targets
            .iter()
            .any(|t| t.target_id == "defensive-share-minority"
                && matches!(t.kind, TargetKind::Claim)));
    }

    #[test]
    fn same_inputs_give_the_same_seed() {
        assert_eq!(replicate_seed(42, 3), replicate_seed(42, 3));
        assert_eq!(
            sweep_trial_seed(42, 0.10, 0.05, 0.15, 2),
            sweep_trial_seed(42, 0.10, 0.05, 0.15, 2)
        );
        assert_eq!(
            ablate_trial_seed(42, DecisionMode::Rule4dim, 1),
            ablate_trial_seed(42, DecisionMode::Rule4dim, 1)
        );
    }

    #[test]
    fn each_coordinate_changes_the_sweep_seed() {
        let base = sweep_trial_seed(42, 0.10, 0.05, 0.15, 0);
        assert_ne!(base, sweep_trial_seed(43, 0.10, 0.05, 0.15, 0), "base");
        assert_ne!(base, sweep_trial_seed(42, 0.20, 0.05, 0.15, 0), "ψ_learn");
        assert_ne!(
            base,
            sweep_trial_seed(42, 0.10, 0.10, 0.15, 0),
            "p_retaliate"
        );
        assert_ne!(base, sweep_trial_seed(42, 0.10, 0.05, 0.20, 0), "defensive");
        assert_ne!(base, sweep_trial_seed(42, 0.10, 0.05, 0.15, 1), "index");
    }

    /// 具体値を固定する．
    ///
    /// ここが変わるのは socsim の `derive_seed` が変わったときで，そのときは過去の
    /// run と結果を比較できなくなっている．Cargo.lock が socsim の commit を固定して
    /// いるので，この値は依存を上げたときにだけ動く．
    #[test]
    fn golden_values_are_pinned() {
        assert_eq!(replicate_seed(42, 0), 12_668_979_023_698_771_201);
        assert_eq!(
            sweep_trial_seed(42, 0.05, 0.02, 0.05, 0),
            10_684_939_167_478_375_031
        );
        assert_eq!(
            ablate_trial_seed(42, DecisionMode::Rule6dim, 0),
            3_554_960_822_400_118_533
        );
    }
}
