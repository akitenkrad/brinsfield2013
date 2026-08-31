"""runs.py — どの run を読むか，そして旧 CSV と同じ形の表をどう組み直すか．

移行前は `results/<timestamp>/` に `metrics.csv` / `motive_mix.csv` / `agents.csv` /
`correlations.csv` / `sweep_summary.csv` が並んでいた．いまは出力の置き場と同一性を
runvault が持ち，同じ数が次の場所にある．

| 旧ファイル | いまの置き場 |
|---|---|
| `metrics.csv` (wide) | `metrics.csv` (long)．`t` が `step` |
| `motive_mix.csv` | 書かない．`metrics.csv` の `motive_mix_*` と同じ数 |
| `correlations.csv` | run スコープ指標 `corr_<motive>_<correlate>` |
| `agents.csv` | `events.jsonl` の `x.brinsfield2013.agent` |
| `sweep_summary.csv` | 子 run の `events.jsonl` の `terminal` 行 + 子の `parameters` |
| `ablation_summary.csv` | 同上 (`ablate-point` の子) |

run ディレクトリの解決は `runvault path` に任せる — `results/` を走査して新しそうな
ディレクトリを当てにいかない．run_slug には条件と環境のハッシュが入るので，こちらで
名前を組み立てることはできないし，できたとしてもすべきでない．

legacy な `results/<timestamp>/` は `--results-dir` に直接渡せば従来どおり読める．
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from runvault.read import (
    config_parameters,
    figures_dir,
    load_run_meta,
    run_scope_metrics,
    sweep_children,
)
from runvault.read import runvault_path as _runvault_path

#: Rust 側 `record::EXPERIMENT` と揃える．
EXPERIMENT = "brinsfield-silence"

#: 6 動機の正準順序．
MOTIVES = ["ineffectual", "relational", "defensive", "diffident", "disengaged", "deviant"]

#: `x.brinsfield2013.agent` — Rust 側 `record::AGENT_EVENT`．
AGENT_EVENT = "x.brinsfield2013.agent"


# --------------------------------------------------------------------------- #
# run の解決
# --------------------------------------------------------------------------- #

def resolve_run_dir(
    results_dir: str | os.PathLike | None,
    *,
    subcommand: str = "run",
    results_root: str = "results",
) -> Path:
    """読む run ディレクトリを決める．

    `results_dir` が与えられていればそれを使う (legacy な `results/<timestamp>/` も
    そのまま渡せる)．与えられていなければ `runvault path --latest` に聞く．
    """
    if results_dir is not None:
        path = Path(results_dir)
        # legacy の `results/latest` シンボリックリンクは実体へ解決する．
        if path.is_symlink():
            return Path(os.path.realpath(path))
        return path
    return Path(_runvault_path(EXPERIMENT, results_root, subcommand=subcommand))


def is_runvault_run(run_dir: str | os.PathLike) -> bool:
    """runvault の run ディレクトリか (`run.json` があるか)．"""
    return load_run_meta(run_dir, required=False) is not None


def analysis_output_dir(run_dir: str | os.PathLike, override: str | None) -> Path:
    """図の置き場．

    `manifest.csv` は `finish()` が確定させるので，run が終わったあとに作ったものを
    run ディレクトリの中に置くとハッシュを持たない (＝記録の一部でない) ファイルが
    混ざる．runvault の run には `runvault.read.figures_dir` が示す run の外を使う．
    """
    if override is not None:
        out = Path(override)
    elif is_runvault_run(run_dir):
        out = Path(figures_dir(run_dir))
    else:
        out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


# --------------------------------------------------------------------------- #
# 1 本の run を読む
# --------------------------------------------------------------------------- #

def _read_csv(path: str | os.PathLike) -> pd.DataFrame:
    """`float_precision="round_trip"` で読む．

    pandas の既定の C パーサは f64 を 1 ULP 落とすことがある
    (例: `0.26666666666666666` → `0.2666666666666666`)．ファイルの中身は正しいので，
    記録された数をそのまま読むためにこの引数を渡す．
    """
    return pd.read_csv(path, float_precision="round_trip")


def step_metrics(run_dir: str | os.PathLike) -> pd.DataFrame:
    """ステップごとの指標．旧 `metrics.csv` と同じ «1 ステップ 1 行» を返す．

    列名は旧 CSV と同じで，時間軸の列名も `t` に戻す．legacy な wide の
    `metrics.csv` はそのまま返す．
    """
    run_dir = Path(run_dir)
    df = _read_csv(run_dir / "metrics.csv")
    if "t" in df.columns:  # legacy wide
        return df
    stepped = df[df["step"].notna()]
    return (
        stepped.pivot_table(index="step", columns="name", values="value", aggfunc="last")
        .reset_index()
        .rename_axis(None, axis=1)
        .rename(columns={"step": "t"})
        .astype({"t": int})
        .sort_values("t")
        .reset_index(drop=True)
    )


def scope_metrics(run_dir: str | os.PathLike) -> dict[str, float]:
    """run 全体を 1 つの値で表す指標 (`final_round` / `n_units` / `corr_*` …)．"""
    if not is_runvault_run(run_dir):
        return {}
    return run_scope_metrics(run_dir)


def correlations(run_dir: str | os.PathLike) -> pd.DataFrame:
    """動機 × correlate の Pearson r．旧 `correlations.csv` と同じ 3 列を返す．

    移行後は run スコープ指標 `corr_<motive>_<correlate>` なので，名前を分解して
    表に戻す．動機名にはアンダースコアが無いので，先頭の 1 語が動機である．
    """
    run_dir = Path(run_dir)
    legacy = run_dir / "correlations.csv"
    if legacy.exists():
        return _read_csv(legacy)
    rows = []
    for name, value in scope_metrics(run_dir).items():
        if not name.startswith("corr_"):
            continue
        motive, correlate = name[len("corr_"):].split("_", 1)
        rows.append({"motive": motive, "correlate": correlate, "pearson_r": value})
    return pd.DataFrame(rows, columns=["motive", "correlate", "pearson_r"])


def agent_table(run_dir: str | os.PathLike) -> pd.DataFrame:
    """従業員ごとの最終状態．旧 `agents.csv` と同じ列を返す．

    移行後は `events.jsonl` の `x.brinsfield2013.agent` 行である．沈黙していない
    従業員には `primary_motive` の欄そのものが無いので (旧 CSV は `-` という番兵を
    書いていた)，読み側では欠測のままにする．
    """
    run_dir = Path(run_dir)
    legacy = run_dir / "agents.csv"
    if legacy.exists():
        return _read_csv(legacy)
    rows = []
    with (run_dir / "events.jsonl").open(encoding="utf-8") as f:
        for line in f:
            event = json.loads(line)
            if event.get("schema") == AGENT_EVENT:
                rows.append({k: v for k, v in event.items()
                             if k not in ("run_uid", "schema", "ts", "t_unit", "unit_id")})
    return pd.DataFrame(rows).sort_values("agent_id").reset_index(drop=True)


def parameters(run_dir: str | os.PathLike) -> dict:
    """その run の実験条件．legacy な flat な `config.json` もそのまま読む．"""
    run_dir = Path(run_dir)
    if is_runvault_run(run_dir):
        return config_parameters(run_dir, required=False) or {}
    for name in ("config.json", "sweep_config.json"):
        path = run_dir / name
        if path.exists():
            with path.open(encoding="utf-8") as f:
                return json.load(f)
    return {}


# --------------------------------------------------------------------------- #
# 掃引 / アブレーション: 試行の表を組み直す
# --------------------------------------------------------------------------- #

def _terminal_rows(child: Path) -> list[dict]:
    rows = []
    with (child / "events.jsonl").open(encoding="utf-8") as f:
        for line in f:
            event = json.loads(line)
            if event.get("schema") == "terminal":
                rows.append(event)
    return rows


def _children_or_stop(parent_dir: str | os.PathLike) -> list[Path]:
    children = [Path(c) for c in sweep_children(parent_dir)]
    if not children:
        raise SystemExit(
            f"エラー: この親 run に紐づく子 run が見つかりません: {parent_dir}\n"
            "  子 run は lineage.parent_run_uid で親を指します．"
            "親と子が同じ results ルートにあるか確認してください．"
        )
    return children


#: 試行の `terminal` 行から取る列 (旧 sweep_summary.csv と同じ名前)．
_TRIAL_COLUMNS = (
    ["silence_rate"]
    + [f"motive_mix_{m}" for m in MOTIVES]
    + ["climate_of_silence", "kl_to_reference"]
)


def sweep_summary_table(sweep_dir: str | os.PathLike) -> pd.DataFrame:
    """1 行 1 試行の掃引サマリ．旧 `sweep_summary.csv` と同じ列を返す．

    runvault はこの表をディスクに持たない．掃引親の子 run (`lineage.parent_run_uid`
    が親の `run_uid`) を集め，子の `parameters` と `events.jsonl` の `terminal` 行から
    組み直す．
    """
    sweep_dir = Path(sweep_dir)
    legacy = sweep_dir / "sweep_summary.csv"
    if legacy.exists():
        return _read_csv(legacy)

    rows: list[dict] = []
    for child in _children_or_stop(sweep_dir):
        params = parameters(child)
        for event in _terminal_rows(child):
            row = {
                "decision_mode": params.get("decision_mode"),
                "psafety_learn": params.get("psafety_learn"),
                "p_retaliate": params.get("p_retaliate"),
                # 掃引軸は «初期分布の defensive 成分» なので，条件の中では
                # motive_init の中にある．
                "motive_init_defensive": (params.get("motive_init") or {}).get("defensive"),
                "run": int(event["unit_id"].split("-")[-1]),
                "seed": event["seed"],
                "final_round": event["t"],
            }
            row.update({column: event[column] for column in _TRIAL_COLUMNS})
            row["run_dir"] = str(child)
            rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values(["psafety_learn", "p_retaliate", "motive_init_defensive", "run"])
        .reset_index(drop=True)
    )


def ablation_summary_table(ablate_dir: str | os.PathLike) -> pd.DataFrame:
    """1 行 1 試行のアブレーションサマリ．旧 `ablation_summary.csv` と同じ列を返す．"""
    ablate_dir = Path(ablate_dir)
    legacy = ablate_dir / "ablation_summary.csv"
    if legacy.exists():
        return _read_csv(legacy)

    rows: list[dict] = []
    for child in _children_or_stop(ablate_dir):
        params = parameters(child)
        for event in _terminal_rows(child):
            row = {
                "decision_mode": params.get("decision_mode"),
                "run": int(event["unit_id"].split("-")[-1]),
                "seed": event["seed"],
            }
            row.update({column: event[column] for column in _TRIAL_COLUMNS})
            row["run_dir"] = str(child)
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["decision_mode", "run"]).reset_index(drop=True)
