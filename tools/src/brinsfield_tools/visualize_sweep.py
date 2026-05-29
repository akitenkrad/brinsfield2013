#!/usr/bin/env python3
"""visualize_sweep.py — sweep visualization for the Brinsfield 2013 silence model.

Reads `results/<timestamp>_sweep/sweep_summary.csv` and produces:
  - sweep_defensive_heatmap.png : ψ_learn × p_retaliate heatmap of motive_mix_defensive
  - sweep_kl_contour.png        : motive_init_defensive × p_retaliate KL-to-reference contour
  - sweep_motive_response.png   : motive shares vs motive_init_defensive

Usage:
    uv run brinsfield-tools visualize-sweep
    uv run brinsfield-tools visualize-sweep --results-dir results/<ts>_sweep
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLOR_BG = "#FAFAF8"
MOTIVES = ["ineffectual", "relational", "defensive", "diffident", "disengaged", "deviant"]
MOTIVE_COLORS = ["#534AB7", "#4C97C9", "#0F6E56", "#F4A259", "#B5546A", "#6E8B3D"]


def plot_defensive_heatmap(df: pd.DataFrame, output_dir: str) -> None:
    pivot = df.pivot_table(
        index="p_retaliate", columns="psafety_learn", values="motive_mix_defensive", aggfunc="mean"
    )
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(COLOR_BG)
    im = ax.imshow(pivot.values, cmap="magma", origin="lower", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.2f}" for v in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.2f}" for v in pivot.index])
    ax.set_xlabel("ψ learning rate")
    ax.set_ylabel("p_retaliate")
    ax.set_title("Mean defensive motive share across ψ_learn × p_retaliate")
    fig.colorbar(im, ax=ax, label="motive_mix_defensive")
    fig.tight_layout()
    out = os.path.join(output_dir, "sweep_defensive_heatmap.png")
    fig.savefig(out, dpi=150, facecolor=COLOR_BG)
    plt.close(fig)
    print(f"[visualize-sweep] wrote {out}")


def plot_kl_contour(df: pd.DataFrame, output_dir: str) -> None:
    if "kl_to_reference" not in df.columns:
        return
    pivot = df.pivot_table(
        index="p_retaliate",
        columns="motive_init_defensive",
        values="kl_to_reference",
        aggfunc="mean",
    )
    if pivot.shape[0] < 2 or pivot.shape[1] < 2:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(COLOR_BG)
    xs = pivot.columns.to_numpy(dtype=float)
    ys = pivot.index.to_numpy(dtype=float)
    cs = ax.contourf(xs, ys, pivot.values, levels=12, cmap="viridis")
    ax.set_xlabel("motive_init defensive share")
    ax.set_ylabel("p_retaliate")
    ax.set_title("KL(mix || Brinsfield reference) contour")
    fig.colorbar(cs, ax=ax, label="KL to reference")
    fig.tight_layout()
    out = os.path.join(output_dir, "sweep_kl_contour.png")
    fig.savefig(out, dpi=150, facecolor=COLOR_BG)
    plt.close(fig)
    print(f"[visualize-sweep] wrote {out}")


def plot_motive_response(df: pd.DataFrame, output_dir: str) -> None:
    grouped = df.groupby("motive_init_defensive", as_index=True)[
        [f"motive_mix_{m}" for m in MOTIVES]
    ].mean()
    if grouped.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(COLOR_BG)
    for m, color in zip(MOTIVES, MOTIVE_COLORS):
        ax.plot(grouped.index, grouped[f"motive_mix_{m}"], marker="o", color=color, label=m)
    ax.axhline(0.1265, color="gray", ls="--", lw=1, label="defensive anchor .1265")
    ax.set_xlabel("initial defensive share")
    ax.set_ylabel("steady-state motive share")
    ax.set_title("Motive-mix response to initial defensive share")
    ax.set_facecolor(COLOR_BG)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    out = os.path.join(output_dir, "sweep_motive_response.png")
    fig.savefig(out, dpi=150, facecolor=COLOR_BG)
    plt.close(fig)
    print(f"[visualize-sweep] wrote {out}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="brinsfield-tools visualize-sweep")
    parser.add_argument("--results-dir", default="results/latest")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    results_dir = args.results_dir
    output_dir = args.output_dir or results_dir
    os.makedirs(output_dir, exist_ok=True)
    sweep_path = os.path.join(results_dir, "sweep_summary.csv")
    if not os.path.exists(sweep_path):
        print(f"[visualize-sweep] no sweep summary at {sweep_path}; nothing to plot")
        return
    df = pd.read_csv(sweep_path)
    _ = np  # numpy reserved for future contour smoothing
    plot_defensive_heatmap(df, output_dir)
    plot_kl_contour(df, output_dir)
    plot_motive_response(df, output_dir)


if __name__ == "__main__":
    main()
