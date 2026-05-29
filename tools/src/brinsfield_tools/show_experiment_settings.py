#!/usr/bin/env python3
"""show_experiment_settings.py — print a results directory's settings.

Reads `config.json` (run) or `sweep_config.json` (sweep) plus `llm_meta.json`
and renders them as a readable table, or as JSON with `--json`. Shared helpers
from `socsim_tools.settings` handle the run-config + metadata rendering; the
sweep-config table is repo-specific.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from socsim_tools.io import resolve_results_dir
from socsim_tools.settings import render_run_config, render_run_metadata

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
    "seed": "seed (core)           ",
    "llm_temperature": "LLM temperature       ",
    "llm_seed": "LLM seed              ",
    "llm_cache_path": "LLM cache_path        ",
    "output_dir": "output_dir            ",
}


def _find_config_file(results_dir: Path) -> tuple[Path, str]:
    run_cfg = results_dir / "config.json"
    sweep_cfg = results_dir / "sweep_config.json"
    if run_cfg.exists():
        return run_cfg, "run"
    if sweep_cfg.exists():
        return sweep_cfg, "sweep"
    raise FileNotFoundError(
        f"no settings file in: {results_dir}\n"
        f"  expected: config.json (run) or sweep_config.json (sweep)"
    )


def _load_llm_meta(results_dir: Path) -> dict | None:
    path = results_dir / "llm_meta.json"
    if path.exists():
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    return None


def render_sweep_config(cfg: dict, source: Path) -> str:
    lines = ["=" * 70, "experiment settings (sweep)", "=" * 70]
    lines.append(f"settings file: {source}")
    lines.append("-" * 70)
    lines.append(f"decision_mode             : {cfg.get('decision_mode', '-')}")
    lines.append(f"n_teams                   : {cfg.get('n_teams', '-')}")
    lines.append(f"team_size                 : {cfg.get('team_size', '-')}")
    lines.append(f"ψ_learn values            : {cfg.get('psafety_learn_values', '-')}")
    lines.append(f"p_retaliate values        : {cfg.get('p_retaliate_values', '-')}")
    lines.append(f"motive_init_def values    : {cfg.get('motive_init_defensive_values', '-')}")
    lines.append(f"runs/cell                 : {cfg.get('runs', '-')}")
    lines.append(f"t_max                     : {cfg.get('t_max', '-')}")
    lines.append(f"seed (base)               : {cfg.get('seed', '-')}")
    lines.append("=" * 70)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="brinsfield-tools show-experiment-settings",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results-dir", "--results_dir", default="results/latest")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table.")
    args = parser.parse_args(argv)

    results_dir = resolve_results_dir(args.results_dir)
    if not results_dir.exists():
        print(f"error: directory does not exist: {results_dir}", file=sys.stderr)
        return 1

    try:
        cfg_path, kind = _find_config_file(results_dir)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    with cfg_path.open(encoding="utf-8") as f:
        cfg = json.load(f)
    meta = _load_llm_meta(results_dir)

    if args.json:
        payload = {"source": str(cfg_path), "kind": kind, "config": cfg, "llm_meta": meta}
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        if kind == "run":
            print(render_run_config(cfg, cfg_path, FIELD_LABELS))
        else:
            print(render_sweep_config(cfg, cfg_path))
        if meta is not None:
            print(render_run_metadata(meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
