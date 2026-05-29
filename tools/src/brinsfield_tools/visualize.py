#!/usr/bin/env python3
"""visualize.py — single-run visualization for the Brinsfield 2013 silence model.

Reads `results/latest` (or `--results-dir`) and produces:
  - motive_mix_stack.png       : 6-region stacked motive-mix time series
  - silence_kl_timeseries.png  : silence rate + KL(mix || reference) per step
  - motive_correlate_bar.png   : final-step Pearson r (motive × ψ / neuroticism)

Usage:
    uv run brinsfield-tools visualize
    uv run brinsfield-tools visualize --results-dir results/latest --output-dir out
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLOR_BG = "#FAFAF8"
# Six categorical motive colours (repo palette + two extra hues).
MOTIVE_COLORS = {
    "ineffectual": "#534AB7",  # purple
    "relational": "#4C97C9",   # blue
    "defensive": "#0F6E56",    # teal
    "diffident": "#F4A259",    # amber
    "disengaged": "#B5546A",   # rose
    "deviant": "#6E8B3D",      # olive
}
MOTIVES = ["ineffectual", "relational", "defensive", "diffident", "disengaged", "deviant"]


def load_config(results_dir: str) -> dict | None:
    path = os.path.join(results_dir, "config.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def plot_motive_stack(results_dir: str, output_dir: str, cfg: dict | None) -> None:
    path = os.path.join(results_dir, "motive_mix.csv")
    if not os.path.exists(path):
        path = os.path.join(results_dir, "metrics.csv")
        if not os.path.exists(path):
            print(f"[visualize] no motive_mix/metrics at {results_dir}; skipping stack")
            return
        df = pd.read_csv(path)
        cols = [f"motive_mix_{m}" for m in MOTIVES]
        data = {m: df[c] for m, c in zip(MOTIVES, cols)}
    else:
        df = pd.read_csv(path)
        data = {m: df[m] for m in MOTIVES}

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(COLOR_BG)
    ax.stackplot(
        df["t"],
        *[data[m] for m in MOTIVES],
        labels=MOTIVES,
        colors=[MOTIVE_COLORS[m] for m in MOTIVES],
        alpha=0.9,
    )
    ax.set_xlabel("step t")
    ax.set_ylabel("primary-motive share within silent")
    ax.set_ylim(0, 1)
    ax.set_facecolor(COLOR_BG)
    title = "Six-motive mix over time"
    if cfg:
        title += f"  (decision_mode={cfg.get('decision_mode')})"
    ax.set_title(title)
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    fig.tight_layout()
    out = os.path.join(output_dir, "motive_mix_stack.png")
    fig.savefig(out, dpi=150, facecolor=COLOR_BG)
    plt.close(fig)
    print(f"[visualize] wrote {out}")


def plot_silence_kl(results_dir: str, output_dir: str) -> None:
    path = os.path.join(results_dir, "metrics.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor(COLOR_BG)
    ax1.plot(df["t"], df["silence_rate"], color="#444444", lw=2, label="silence rate")
    ax1.set_xlabel("step t")
    ax1.set_ylabel("silence rate")
    ax1.set_title("Silence rate over time")
    ax1.set_facecolor(COLOR_BG)
    ax1.legend()
    if "kl_to_reference" in df.columns:
        ax2.plot(df["t"], df["kl_to_reference"], color="#7f7f7f", lw=2)
        ax2.axhline(0.0, color="gray", ls=":", lw=0.8)
        ax2.set_xlabel("step t")
        ax2.set_ylabel("KL(mix || Brinsfield reference)")
        ax2.set_title("KL divergence to Brinsfield reference mix")
        ax2.set_facecolor(COLOR_BG)
    fig.tight_layout()
    out = os.path.join(output_dir, "silence_kl_timeseries.png")
    fig.savefig(out, dpi=150, facecolor=COLOR_BG)
    plt.close(fig)
    print(f"[visualize] wrote {out}")


def plot_motive_correlate_bar(results_dir: str, output_dir: str) -> None:
    path = os.path.join(results_dir, "correlations.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor(COLOR_BG)
    x = np.arange(len(MOTIVES))
    width = 0.4
    for offset, corr, color in [(-width / 2, "psafety", "#4C97C9"), (width / 2, "neuroticism", "#B5546A")]:
        rs = []
        for m in MOTIVES:
            sub = df[(df["motive"] == m) & (df["correlate"] == corr)]
            rs.append(float(sub["pearson_r"].iloc[0]) if not sub.empty else 0.0)
        ax.bar(x + offset, rs, width, label=f"r(motive, {corr})", color=color, alpha=0.85)
    ax.axhline(0.0, color="gray", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(MOTIVES, rotation=20, ha="right")
    ax.set_ylabel("Pearson r")
    ax.set_title("Motive × correlate (final step): ψ-negative for def/dif/rel; n-positive for dev/dif")
    ax.set_facecolor(COLOR_BG)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(output_dir, "motive_correlate_bar.png")
    fig.savefig(out, dpi=150, facecolor=COLOR_BG)
    plt.close(fig)
    print(f"[visualize] wrote {out}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="brinsfield-tools visualize")
    parser.add_argument("--results-dir", default="results/latest")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    results_dir = args.results_dir
    output_dir = args.output_dir or results_dir
    os.makedirs(output_dir, exist_ok=True)
    cfg = load_config(results_dir)
    plot_motive_stack(results_dir, output_dir, cfg)
    plot_silence_kl(results_dir, output_dir)
    plot_motive_correlate_bar(results_dir, output_dir)


if __name__ == "__main__":
    main()
