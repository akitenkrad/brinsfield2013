//! Offline (no live LLM) smoke: a scripted mock drives the LLM pipeline
//! end-to-end and records the same run shape as the production `run`．
//!
//! LLM 経路の «移行の前後で数が変わっていないか» はこれで確かめる．外部呼び出しが
//! 0 なので同じ入力に必ず同じ応答が返り，実 LLM を 2 回叩いて並べるのと違って
//! 比較が成立する．
//!
//! 反復は 1 本なので親 run は作らない．本番の `run` とは subcommand 名で区別する —
//! 中身は «scripted client で駆動した 1 本» であって，本番の LLM 経路とは別物である．
//!
//! Usage:
//!     cargo run --example mock_smoke -- results

use brinsfield_silence_simulation::config::{Config, DecisionMode, LlmSettings, MotiveInit};
use brinsfield_silence_simulation::llm::wrap_client;
use brinsfield_silence_simulation::record::{
    self, ConditionParameters, SingleRunParameters, DOMAIN, EXPERIMENT, HASH_EXCLUDE, REPO_ID,
    SINGLE_SEED_POINTERS,
};
use brinsfield_silence_simulation::simulation::run_with_client;

use runvault::{Run, RunOptions};
use socsim_llm::mock::ScriptedClient;
use socsim_llm::{LlmClient, PromptCache};

fn main() {
    let results_root = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "results".to_string());

    let cfg = Config {
        n_teams: 3,
        team_size: 6,
        n_levels: 2,
        network_k: 4,
        t_max: 10,
        runs: 1,
        seed: 1234,
        decision_mode: DecisionMode::Llm,
        motive_init: MotiveInit::default(),
        // cache_path = None → in-memory cache; run_with_client does not save().
        llm: LlmSettings::default(),
        ..Config::default()
    };

    // 三択の scripted backend．プロンプトの長さだけで応答を決めるので，プロンプトの
    // 文面が変わらないかぎり同じ列を返す．
    let backend = ScriptedClient::new("mock-brinsfield", |prompt: &str| match prompt.len() % 3 {
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
    let client = wrap_client(backend, PromptCache::in_memory());
    let llm = record::llm_block(
        client.inner().model(),
        client.inner().endpoint(),
        cfg.llm.temperature,
    );

    let mut rv = Run::start(
        RunOptions::new(EXPERIMENT, "mock-smoke")
            .repo_id(REPO_ID)
            .domain(DOMAIN)
            .results_root(&results_root)
            .parameters(&SingleRunParameters {
                condition: ConditionParameters::from_config(&cfg),
                seed: cfg.seed,
            })
            .expect("runvault: parameters の組み立てに失敗")
            .hash_exclude(HASH_EXCLUDE)
            .seed_pointers(SINGLE_SEED_POINTERS)
            .master_seed(cfg.seed)
            .llm(llm)
            .replication(record::replication()),
    )
    .expect("runvault: run の開始に失敗");

    let result = run_with_client(&cfg, Some(client)).expect("mock run failed");

    record::log_simulation(&mut rv, &result);
    record::log_llm_usage(&mut rv, &result);
    record::log_paper_reference(&mut rv);

    let last = result.metrics_rows.last().expect("metrics_rows は空でない");
    let (silence, defensive, kl) = (
        last.silence_rate,
        last.motive_mix_defensive,
        last.kl_to_reference,
    );
    let calls = result.metadata.total();
    let hit_rate = result.metadata.cache_hit_rate();

    let dir = rv.finish().expect("runvault: run の完了に失敗");

    println!("mock smoke wrote: {}", dir.display());
    println!("LLM calls: {} (cache-hit {:.1}%)", calls, hit_rate * 100.0);
    println!("final silence={silence:.3} defensive={defensive:.3} KL={kl:.3}");
}
