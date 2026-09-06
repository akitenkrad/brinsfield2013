# CLI リファレンス

[English](cli.md) | [日本語](cli.ja.md)

`brinsfield` バイナリは 4 つのサブコマンドを持つ．

出力の置き場と同一性は [runvault](https://github.com/akitenkrad/rs-runvault) が持つ．
どのサブコマンドも `<--output-dir>/brinsfield-silence/` の下に run ディレクトリを作る．
自前のタイムスタンプ付きディレクトリも `results/latest` シンボリックリンクも作らない．
run の解決は `runvault path --experiment brinsfield-silence --latest --subcommand <名前>`．

## `run`

単一設定の実行．run 1 本に `config.json`（条件），`metrics.csv`（ステップごとの指標と
run スコープの指標），`events.jsonl`（従業員 1 人 1 行の `x.brinsfield2013.agent`），
`reference.csv`（原著の defensive 12.65%），`manifest.csv` が入る．

`--runs N` は掃引ではない．同じ条件を N 本回して最後の結果だけを残す従来どおりの動き
なので，run には `master_seed = derive_seed(seed, [N-1])` と `replicate_index = N-1` を
記録する．根のシードは `parameters.base_seed` にある．

| フラグ | 既定 | 意味 |
|------|---------|------|
| `--decision-mode` | `rule_6dim` | `llm` / `rule_6dim` / `rule_4dim` / `rule_3dim` |
| `--n-teams` / `--team-size` | 5 / 8 | 組織形状 |
| `--n-levels` | 3 | 階層数（記述用） |
| `--network` | `watts-strogatz` | `watts-strogatz` / `erdos-renyi` / `barabasi-albert` |
| `--network-k` / `--network-beta` | 6 / 0.1 | WS 次数 `k` / rewire β（または ER `p`） |
| `--supervisor-homogeneity` | 0.0 | η_sup ∈ [0,1] 共通 supervisor openness への牽引 |
| `--motive-init` | `0.35,0.20,0.13,0.13,0.13,0.06` | 初期 6 動機分布 |
| `--motive-learn-rate` | 0.10 | `motive_dynamics` の EMA η |
| `--psafety-learn` | 0.05 | ψ 学習率 |
| `--p-retaliate` | 0.05 | エージェント別 step 別報復確率 |
| `--shock-t` / `--shock-magnitude` | なし / 0.3 | 外生 σ ショック |
| `--temperature` / `--prompt-version` | 0.0 / 1 | LLM 温度 / プロンプト template v1–v3 |
| `--t-max` / `--runs` / `--seed` | 48 / 1 / 42 | horizon / 反復 / コア seed |
| `--llm-seed` / `--llm-cache-path` | 0 / `.llm_cache/cache.json` | LLM seed offset / cache パス |
| `--output-dir` | `results` | runvault の results root |

## `sweep`

`--psafety-learn × --p-retaliate × --motive-init-defensive`（カンマ列）× `--runs` seed
の直積．親 run 1 本（`--subcommand sweep`）がグリッドを宣言し，セル 1 つにつき子 run
1 本（`--subcommand sweep-point`）がそのセルの試行を `events.jsonl` の `observation` +
`terminal` 行として持つ．セルの平均は子の run スコープ指標にある．掃引が見るのは各試行の
最終ステップだけなので，試行は自分の時系列を持たず `terminal` 行 1 本で言い尽くせる．

旧 `sweep_summary.csv` と同じ列の表は `brinsfield_tools.runs.sweep_summary_table` が
子 run から組み直す．

## `ablate`

`--decision-modes rule_6dim,rule_4dim,rule_3dim`（カンマ列）を各 `--runs` 回並走させ，
モード別の平均 `KL(mix‖reference)` と defensive 比率を表示する．記録は `sweep` と同じ
形で，親 run（`--subcommand ablate`）と決定モードごとの子 run（`--subcommand
ablate-point`）に分かれ，試行は `terminal` 行になる．旧 `ablation_summary.csv` と同じ
列の表は `brinsfield_tools.runs.ablation_summary_table` が組み直す．

## `reproduce`

1 設定を実行し，定常 motive_mix（`t ≥ t_max/2` の平均）を Brinsfield reference と対比
表示する．defensive 12.65% アンカー・ineffectual 下限・deviant 上限について
PASS/off-anchor を判定する．定常平均は run スコープ指標
`steady_state_motive_mix_<動機>` として記録する．許容幅と判定は記録しない — 原著の
報告値でも観測値でもなく，こちらが決めた判定基準だからである．

## `mock_smoke`（example）

`cargo run --example mock_smoke -- <results-root>` は scripted client で LLM 経路を
外部呼び出し 0 のまま端から端まで走らせ，`--subcommand mock-smoke` の run を記録する．
実 LLM 無しで LLM 経路を確かめるための入口である．
