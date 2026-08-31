#!/usr/bin/env python3
"""visualize.py — single-run visualization for the Brinsfield 2013 silence model.

runvault の run ディレクトリを読んで次の 3 枚を出す:
  - motive_mix_stack.png       : 6-region stacked motive-mix time series
  - silence_kl_timeseries.png  : silence rate + KL(mix || reference) per step
  - motive_correlate_bar.png   : final-step Pearson r (motive × ψ / neuroticism)

`--results-dir` を省略すると `runvault path --experiment brinsfield-silence --latest
--subcommand run` が返す run ディレクトリを対象にする (`runvault` が PATH にあるか，
環境変数 `RUNVAULT` が実行ファイルを指している必要がある)．legacy な
`results/<timestamp>/` を直接渡すこともできる．

図は run の外 (`<results-root>/brinsfield-silence/figures/<run_slug>/`) に出す．
run が終わった後に作るものは `manifest.csv` に載らないので，run ディレクトリの中には
置かない．

Usage:
    uv run brinsfield-tools visualize
    uv run brinsfield-tools visualize --results-dir results/20260529_143329   # legacy も可
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from brinsfield_tools import runs  # noqa: E402

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
MOTIVES = runs.MOTIVES


def plot_motive_stack(df: pd.DataFrame, output_dir: str, cfg: dict | None) -> None:
    data = {m: df[f"motive_mix_{m}"] for m in MOTIVES}
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


def plot_silence_kl(df: pd.DataFrame, output_dir: str) -> None:
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


def plot_motive_correlate_bar(corr: pd.DataFrame, output_dir: str) -> None:
    """動機 × correlate の Pearson r．

    移行前は `correlations.csv` の 1 行だった値が，run スコープの指標
    `corr_<motive>_<correlate>` になっている (run 全体で 1 つしか無い値なので `step`
    を持たない)．`runs.correlations` が旧 CSV と同じ 3 列に戻す．
    """
    if corr.empty:
        print("[visualize] no motive × correlate correlations; skipping")
        return
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor(COLOR_BG)
    x = np.arange(len(MOTIVES))
    width = 0.4
    for offset, name, color in [(-width / 2, "psafety", "#4C97C9"), (width / 2, "neuroticism", "#B5546A")]:
        rs = []
        for m in MOTIVES:
            sub = corr[(corr["motive"] == m) & (corr["correlate"] == name)]
            rs.append(float(sub["pearson_r"].iloc[0]) if not sub.empty else 0.0)
        ax.bar(x + offset, rs, width, label=f"r(motive, {name})", color=color, alpha=0.85)
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
    parser.add_argument(
        "--results-dir",
        "--results_dir",
        default=None,
        help="run ディレクトリ (省略時は runvault path が返す直近の run)",
    )
    parser.add_argument("--results-root", "--results_root", default="results")
    parser.add_argument("--output-dir", "--output_dir", default=None)
    args = parser.parse_args(argv)

    run_dir = runs.resolve_run_dir(
        args.results_dir, subcommand="run", results_root=args.results_root
    )
    output_dir = str(runs.analysis_output_dir(run_dir, args.output_dir))
    print(f"[visualize] run: {run_dir}")

    df = runs.step_metrics(run_dir)
    cfg = runs.parameters(run_dir)
    plot_motive_stack(df, output_dir, cfg)
    plot_silence_kl(df, output_dir)
    plot_motive_correlate_bar(runs.correlations(run_dir), output_dir)


if __name__ == "__main__":
    main()
