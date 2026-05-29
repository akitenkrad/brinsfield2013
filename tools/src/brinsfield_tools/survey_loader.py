#!/usr/bin/env python3
"""survey_loader.py — Track A survey loader.

Loads a real survey CSV (`--csv`) or synthesises a plausible 30-item (6-factor)
+ 4-correlate dataset (`--synthesize-n N`) calibrated to Brinsfield Study 2/3
anchors. Output is written to `results/track_a/<sample>/loaded.csv` and consumed
by `cfa` / `reproduce`.

The synthetic path is **explicitly synthetic** (a `synthetic=True` column) so it
is never mistaken for real data. It draws a clean 6-correlated-factor mixture so
that a 6-factor CFA fits markedly better than 1–5-factor competitors — the
structure Brinsfield's Study 3 establishes — while keeping the correlate signs
(ψ → defensive/diffident/relational negative; neuroticism → deviant/diffident
positive) consistent with Study 4.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

from brinsfield_tools._track_a_common import (
    CORRELATE_NAMES,
    ITEM_NAMES,
    ITEMS_PER_SUBSCALE,
    SUBSCALES,
)


def synthesize(n: int, seed: int = 42) -> pd.DataFrame:
    """Synthesise a 6-factor 30-item + 4-correlate dataset.

    Items load 0.72 on their own subscale factor; the six latent factors carry
    modest positive inter-correlations (≈ .25–.35). Correlate scales reproduce
    the Study 4 sign pattern.
    """
    rng = np.random.default_rng(seed)
    n_factors = len(SUBSCALES)
    n_items = len(ITEM_NAMES)

    # Loadings (rows = items, cols = factors).
    loadings = np.zeros((n_items, n_factors))
    for i in range(n_items):
        loadings[i, i // ITEMS_PER_SUBSCALE] = 0.72

    # Six correlated factors; deviant slightly more isolated.
    base = 0.30
    factor_corr = np.full((n_factors, n_factors), base)
    np.fill_diagonal(factor_corr, 1.0)
    dev = SUBSCALES.index("deviant")
    factor_corr[dev, :] = 0.15
    factor_corr[:, dev] = 0.15
    factor_corr[dev, dev] = 1.0
    # Defensive & diffident correlate a little more (both fear-of-self adjacent).
    di, df = SUBSCALES.index("diffident"), SUBSCALES.index("defensive")
    factor_corr[di, df] = factor_corr[df, di] = 0.45

    chol = np.linalg.cholesky(factor_corr)
    factors = rng.standard_normal((n, n_factors)) @ chol.T

    raw = factors @ loadings.T + 0.40 * rng.standard_normal((n, n_items))
    items = np.clip(np.round(raw * 1.1 + 4.0), 1, 7)
    df_out = pd.DataFrame(items.astype(int), columns=ITEM_NAMES)

    f = {sub: factors[:, i] for i, sub in enumerate(SUBSCALES)}
    # Study 4 correlates (signs per design §5 #11–17).
    voice = (
        -0.30 * f["ineffectual"]
        - 0.28 * f["relational"]
        - 0.30 * f["defensive"]
        - 0.22 * f["disengaged"]
        - 0.18 * f["diffident"]
        + 0.50 * rng.standard_normal(n)
    )
    psych_safety = (
        -0.40 * f["defensive"]
        - 0.32 * f["diffident"]
        - 0.30 * f["relational"]
        + 0.55 * rng.standard_normal(n)
    )
    neuroticism = (
        0.42 * f["deviant"] + 0.38 * f["diffident"] + 0.55 * rng.standard_normal(n)
    )
    extraversion = (
        -0.25 * f["disengaged"] + 0.10 * f["relational"] + 0.6 * rng.standard_normal(n)
    )

    df_out["voice"] = voice
    df_out["psych_safety"] = psych_safety
    df_out["neuroticism"] = neuroticism
    df_out["extraversion"] = extraversion
    df_out["synthetic"] = True
    return df_out


def load_real_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in ITEM_NAMES if c not in df.columns]
    if missing:
        print(f"[survey-loader] warning: missing item columns: {missing}", file=sys.stderr)
    df["synthetic"] = False
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="brinsfield-tools survey-loader")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="path to a real survey CSV (Phase 3 real-data)")
    src.add_argument(
        "--synthesize-n",
        type=int,
        default=None,
        help="synthesise an N-row dataset calibrated to Brinsfield Study 2/3",
    )
    parser.add_argument("--sample", required=True, help="sample identifier")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-base", default="results/track_a")
    args = parser.parse_args(argv)

    if args.csv:
        df = load_real_csv(args.csv)
        src_msg = f"real CSV {args.csv}"
    else:
        df = synthesize(args.synthesize_n, args.seed)
        src_msg = f"synthesised n={args.synthesize_n} seed={args.seed}"

    out_dir = os.path.join(args.output_base, args.sample)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "loaded.csv")
    df.to_csv(out_path, index=False)
    print(f"[survey-loader] wrote {out_path} ({len(df)} rows, {df.shape[1]} cols)")
    print(f"  correlates: {CORRELATE_NAMES}")
    print(f"  source: {src_msg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
