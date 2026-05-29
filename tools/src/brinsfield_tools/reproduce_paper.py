#!/usr/bin/env python3
"""reproduce_paper.py — one-command Brinsfield (2013) reproduction.

End-to-end on the synthetic Track A path (no real data required):

  1. synthesise a calibrated N-row sample (if not already present),
  2. fit the competing CFA family and confirm 6-factor superiority,
  3. report the motive distribution + defensive 12.65% anchor (from a Track B
     ABM `metrics.csv` if `--track-b-dir` is given, else from the synthetic
     factor-score prevalences),
  4. reconcile the Study 4 incremental-validity ΔR² for VOICE per motive.

Writes `three_way_comparison.csv` to the sample directory.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

from brinsfield_tools._track_a_common import (
    ITEMS_PER_SUBSCALE,
    SUBSCALES,
    load_sample,
)

# Brinsfield anchors (design §5).
DEFENSIVE_ANCHOR = 0.1265
DELTA_R2_TARGET = {
    "relational": 0.05,
    "defensive": 0.05,
    "ineffectual": 0.04,
    "disengaged": 0.03,
}


def _subscale_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Mean item score per subscale (a factor-score proxy)."""
    out = {}
    for sub in SUBSCALES:
        cols = [f"{sub}{i + 1}" for i in range(ITEMS_PER_SUBSCALE)]
        cols = [c for c in cols if c in df.columns]
        if cols:
            out[sub] = df[cols].mean(axis=1)
    return pd.DataFrame(out)


def _motive_prevalence(scores: pd.DataFrame) -> dict[str, float]:
    """Primary-motive prevalence: each respondent's argmax subscale (z-scored)."""
    z = (scores - scores.mean()) / scores.std(ddof=0).replace(0, 1)
    primary = z.idxmax(axis=1)
    counts = primary.value_counts(normalize=True)
    return {sub: float(counts.get(sub, 0.0)) for sub in SUBSCALES}


def _delta_r2(df: pd.DataFrame, scores: pd.DataFrame) -> dict[str, float]:
    """Incremental R² of each motive subscale for VOICE over the other 5.

    Closed-form via the squared semi-partial correlation from a least-squares
    fit (no statsmodels dependency).
    """
    if "voice" not in df.columns:
        return {}
    y = df["voice"].to_numpy(dtype=float)
    out = {}
    for sub in SUBSCALES:
        others = [s for s in SUBSCALES if s != sub and s in scores.columns]
        x_full = np.column_stack([np.ones(len(df))] + [scores[s].to_numpy() for s in others + [sub]])
        x_red = np.column_stack([np.ones(len(df))] + [scores[s].to_numpy() for s in others])
        r2_full = _ols_r2(x_full, y)
        r2_red = _ols_r2(x_red, y)
        out[sub] = max(r2_full - r2_red, 0.0)
    return out


def _ols_r2(x: np.ndarray, y: np.ndarray) -> float:
    coef, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _run_cfa(sample: str, output_base: str) -> pd.DataFrame | None:
    from brinsfield_tools.cfa import build_models, fit_one

    df = load_sample(sample, output_base)
    item_cols = [c for c in df.columns if any(c.startswith(s) for s in SUBSCALES)]
    data = df[item_cols].apply(pd.to_numeric, errors="coerce").dropna()
    models = build_models()
    rows = []
    for name in ["M1", "M2", "M3", "M4", "M5", "M6"]:
        try:
            stats = fit_one(models[name], data)
        except Exception as exc:  # noqa: BLE001
            print(f"warning: CFA {name} failed ({exc})", file=sys.stderr)
            stats = {"cfi": float("nan"), "rmsea": float("nan"), "aic": float("nan"), "bic": float("nan")}
        rows.append({"model": name, **{k: stats.get(k, float("nan")) for k in ("cfi", "rmsea", "aic", "bic")}})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="brinsfield-tools reproduce")
    parser.add_argument("--sample", default="synth")
    parser.add_argument("--synthesize-n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-base", default="results/track_a")
    parser.add_argument(
        "--track-b-dir",
        default=None,
        help="optional ABM results dir with metrics.csv (for the emergent motive mix)",
    )
    args = parser.parse_args(argv)

    sample_path = os.path.join(args.output_base, args.sample, "loaded.csv")
    if not os.path.exists(sample_path):
        from brinsfield_tools.survey_loader import synthesize

        os.makedirs(os.path.dirname(sample_path), exist_ok=True)
        synthesize(args.synthesize_n, args.seed).to_csv(sample_path, index=False)
        print(f"[reproduce] synthesised {args.synthesize_n}-row sample → {sample_path}")

    df = load_sample(args.sample, args.output_base)
    scores = _subscale_scores(df)

    print("=" * 64)
    print("Brinsfield (2013) — one-command reproduction (synthetic Track A)")
    print("=" * 64)

    # 1. CFA 6-factor superiority.
    cfa = _run_cfa(args.sample, args.output_base)
    print("\n[1] Competing CFA fit (6-factor M6 vs 1–5-factor):")
    print(cfa.to_string(index=False))
    verdict_cfa = "n/a"
    if cfa is not None and "M6" in cfa["model"].values:
        m6_cfi = cfa.loc[cfa["model"] == "M6", "cfi"].iloc[0]
        best_other = cfa.loc[cfa["model"].isin(["M1", "M2", "M3", "M4", "M5"]), "cfi"].max()
        if pd.notna(m6_cfi):
            verdict_cfa = "PASS" if m6_cfi >= best_other else "review"
            print(f"  → M6 CFI={m6_cfi:.3f} vs best 1–5-factor CFI={best_other:.3f}: {verdict_cfa}")

    # 2. Motive distribution (ABM emergent if provided, else synthetic prevalence).
    if args.track_b_dir and os.path.exists(os.path.join(args.track_b_dir, "metrics.csv")):
        m = pd.read_csv(os.path.join(args.track_b_dir, "metrics.csv"))
        tail = m[m["t"] >= m["t"].max() // 2]
        prevalence = {sub: float(tail[f"motive_mix_{sub}"].mean()) for sub in SUBSCALES}
        src = f"ABM {args.track_b_dir}"
    else:
        prevalence = _motive_prevalence(scores)
        src = "synthetic factor-score prevalence"
    print(f"\n[2] Motive distribution ({src}):")
    for sub in SUBSCALES:
        print(f"  {sub:<13} {prevalence[sub]:.4f}")
    def_share = prevalence["defensive"]
    def_ok = abs(def_share - DEFENSIVE_ANCHOR) <= 0.06
    print(f"  → defensive {def_share:.4f} vs anchor {DEFENSIVE_ANCHOR:.4f}: "
          f"{'PASS' if def_ok else 'off-anchor'}")

    # 3. Study 4 incremental ΔR².
    dr2 = _delta_r2(df, scores)
    print("\n[3] Study 4 incremental validity ΔR² for VOICE:")
    print(f"  {'motive':<13} {'ΔR²':>8} {'target':>8} {'verdict':>8}")
    rows = []
    for sub, target in DELTA_R2_TARGET.items():
        val = dr2.get(sub, float("nan"))
        ok = pd.notna(val) and val >= target - 0.02
        print(f"  {sub:<13} {val:>8.4f} {target:>8.3f} {'PASS' if ok else 'low':>8}")
        rows.append({"indicator": f"delta_r2_{sub}", "target": target, "value": val})

    # Write comparison CSV.
    rows.insert(0, {"indicator": "defensive_share", "target": DEFENSIVE_ANCHOR, "value": def_share})
    rows.insert(1, {"indicator": "cfa_6factor_superiority", "target": 1.0,
                    "value": 1.0 if verdict_cfa == "PASS" else 0.0})
    out_path = os.path.join(args.output_base, args.sample, "three_way_comparison.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print("=" * 64)
    print(f"[reproduce] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
