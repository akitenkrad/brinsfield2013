#!/usr/bin/env python3
"""show_experiment_settings.py — print a run directory's settings.

runvault の run では，条件は `config.json` の `parameters`，どの LLM が答えたかは
`run.json` の `llm` ブロック，呼び出し数と cache-hit は `metrics.csv` の run スコープ
指標 `llm_calls` / `llm_cache_hits` / `llm_cache_hit_rate` にある (移行前は
`config.json` / `sweep_config.json` / `llm_meta.json` の 3 ファイルだった)．

legacy な `results/<timestamp>/` を `--results-dir` に渡した場合は，従来どおり
`config.json` (または `sweep_config.json`) と `llm_meta.json` を読む．

run ディレクトリのパスは次でも取れる:
    runvault path --experiment brinsfield-silence --latest --subcommand run
    runvault path --experiment brinsfield-silence --latest --subcommand sweep

Usage:
    uv run brinsfield-tools show-experiment-settings
    uv run brinsfield-tools show-experiment-settings --subcommand sweep
    uv run brinsfield-tools show-experiment-settings --results-dir results/20260529_143329
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from socsim_tools.settings import render_run_metadata

from brinsfield_tools import runs

#: Config field → display label (left-padded so colons align)．
#: 掃引のセル子 run は `base_seed`，単発の run は `seed` を持つので，両方を並べて
#: «その run が実際に持っているキーだけ» を出す．
FIELD_LABELS = {
    "decision_mode": "decision_mode         ",
    "n_teams": "n_teams               ",
    "team_size": "team_size             ",
    "n_levels": "n_levels              ",
    "n_employees": "n_employees           ",
    "network_kind": "network_kind          ",
    "network_k": "network_k             ",
    "network_beta": "network_beta          ",
    "supervisor_homogeneity": "supervisor_homogeneity",
    "prompt_version": "prompt_version        ",
    "motive_learn_rate": "motive_learn_rate (η) ",
    "psafety_learn": "psafety_learn         ",
    "p_retaliate": "p_retaliate           ",
    "shock_t": "shock_t               ",
    "shock_magnitude": "shock_magnitude       ",
    "t_max": "t_max                 ",
    "runs": "runs                  ",
    "seed": "seed                  ",
    "base_seed": "base_seed             ",
    "llm_temperature": "LLM temperature       ",
    "llm_seed": "LLM seed              ",
    "llm_cache_path": "LLM cache_path        ",
}

#: 掃引・アブレーションの親 run だけが持つキー．
PARENT_KEYS = (
    "psafety_learn_values",
    "p_retaliate_values",
    "motive_init_defensive_values",
    "decision_modes",
)


def render_run_config(cfg: dict, source: Path, kind: str) -> str:
    """条件の表．その run が持っているキーだけを並べる．"""
    lines = ["=" * 70, f"experiment settings ({kind})", "=" * 70]
    lines.append(f"settings file: {source}")
    lines.append("-" * 70)
    for field, label in FIELD_LABELS.items():
        if field in cfg:
            lines.append(f"{label}: {cfg[field]}")
    init = cfg.get("motive_init") or {}
    if init:
        lines.append(
            "motive_init           : "
            + " ".join(f"{m[:4]}={init[m]}" for m in runs.MOTIVES if m in init)
        )
    for name, value in (cfg.get("beta") or {}).items():
        lines.append(f"{name:<22}: {value}")
    lines.append("=" * 70)
    return "\n".join(lines)


def render_parent_config(cfg: dict, source: Path, kind: str) -> str:
    """掃引・アブレーション親のグリッド宣言．"""
    lines = ["=" * 70, f"experiment settings ({kind} parent)", "=" * 70]
    lines.append(f"settings file: {source}")
    lines.append("-" * 70)
    for key, label in (
        ("decision_mode", "decision_mode          "),
        ("decision_modes", "decision_modes         "),
        ("n_teams", "n_teams                "),
        ("team_size", "team_size              "),
        ("psafety_learn_values", "ψ_learn values         "),
        ("p_retaliate_values", "p_retaliate values     "),
        ("motive_init_defensive_values", "motive_init_def values "),
        ("runs", "runs/cell              "),
        ("t_max", "t_max                  "),
    ):
        if key in cfg:
            lines.append(f"{label}: {cfg[key]}")
    lines.append(f"{'seed (base)':<23}: {cfg.get('base_seed', cfg.get('seed', '-'))}")
    lines.append("=" * 70)
    return "\n".join(lines)


def render_llm(meta: dict, scoped: dict[str, float]) -> str | None:
    """LLM 由来情報．rule モードは LLM 層に触れないので何も出さない．

    移行前は `llm_meta.json` が持っていた．モデル・provider・温度は `run.json` の
    `llm` ブロック，呼び出し数と cache-hit 率は run スコープの指標が正本になった．
    `"model": "none"` を名乗る行はもう出ない — LLM を 1 度も叩かない run には
    `llm` ブロックそのものが無い．
    """
    llm = meta.get("llm")
    if llm is None:
        return None
    lines = ["LLM provenance", "-" * 70]
    lines.append(f"provider         : {llm.get('provider', '-')}")
    lines.append(f"model            : {llm.get('model_snapshot', '-')}")
    lines.append(f"temperature      : {llm.get('temperature', '-')}")
    calls = scoped.get("llm_calls")
    if calls is not None:
        hits = scoped.get("llm_cache_hits", 0.0)
        rate = scoped.get("llm_cache_hit_rate")
        rate_text = "-" if rate is None else f"{rate * 100:.1f}%"
        lines.append(f"calls / cache-hit: {int(calls)} / {int(hits)} ({rate_text})")
    lines.append("=" * 70)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="brinsfield-tools show-experiment-settings",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        "--results_dir",
        default=None,
        help="run ディレクトリ (省略時は runvault path が返す直近の run)",
    )
    parser.add_argument("--results-root", "--results_root", default="results")
    parser.add_argument(
        "--subcommand",
        default="run",
        help="省略時に探すサブコマンド (run / sweep / ablate / reproduce / sweep-point …)",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table.")
    args = parser.parse_args(argv)

    run_dir = runs.resolve_run_dir(
        args.results_dir, subcommand=args.subcommand, results_root=args.results_root
    )
    if not run_dir.exists():
        print(f"error: directory does not exist: {run_dir}", file=sys.stderr)
        return 1

    cfg = runs.parameters(run_dir)
    if not cfg:
        print(
            f"error: no settings in: {run_dir}\n"
            "  expected: runvault の config.json，または legacy な config.json /"
            " sweep_config.json",
            file=sys.stderr,
        )
        return 1

    if runs.is_runvault_run(run_dir):
        from runvault.read import load_run_meta

        meta = load_run_meta(run_dir) or {}
        kind = str(meta.get("subcommand", "run"))
        scoped = runs.scope_metrics(run_dir)
        source = run_dir / "config.json"
        legacy_meta = None
    else:
        meta, scoped, legacy_meta = {}, {}, None
        kind = "sweep" if (run_dir / "sweep_config.json").exists() else "run"
        source = run_dir / ("sweep_config.json" if kind == "sweep" else "config.json")
        path = run_dir / "llm_meta.json"
        if path.exists():
            with path.open(encoding="utf-8") as f:
                legacy_meta = json.load(f)

    is_parent = any(key in cfg for key in PARENT_KEYS)

    if args.json:
        payload = {
            "source": str(source),
            "kind": kind,
            "config": cfg,
            "llm": meta.get("llm") or legacy_meta,
            "run_scope_metrics": scoped,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    if is_parent:
        print(render_parent_config(cfg, source, kind))
    else:
        print(render_run_config(cfg, source, kind))
    if legacy_meta is not None:
        print(render_run_metadata(legacy_meta))
    else:
        block = render_llm(meta, scoped)
        if block is not None:
            print(block)
    return 0
