#!/usr/bin/env python3
"""cfa.py — Track A: competing CFA models on the 30 Brinsfield items (semopy).

Fits the family Brinsfield Study 3 contrasts, plus a bifactor extension:

  M1 : 1-factor (single global silence)
  M2 : 2-factor (fear-based {defensive,diffident} vs the rest)
  M3 : 3-factor (Van Dyne: acquiescent {ineff+diff+diseng} / defensive+deviant / prosocial=relational)
  M4 : 4-factor (Knoll: ineffectual+diffident / relational+disengaged / defensive / deviant)
  M5 : 5-factor (merge diffident into ineffectual)
  M6 : 6-factor correlated (Brinsfield Study 3 winner)
  Mbi: bifactor (general silence + 6 specific factors)

Loads `results/track_a/<sample>/loaded.csv`; writes `cfa_summary.csv`. The
synthetic loader produces a clean 6-correlated-factor structure, so M6 should
beat M1–M5 on CFI / RMSEA / AIC / BIC (the Study 3 superiority finding).
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

from brinsfield_tools._track_a_common import ITEMS_PER_SUBSCALE, SUBSCALES, load_sample


def _items(sub: str) -> list[str]:
    return [f"{sub}{i + 1}" for i in range(ITEMS_PER_SUBSCALE)]


def _factor(name: str, subs: list[str]) -> str:
    items = [it for sub in subs for it in _items(sub)]
    return f"{name} =~ " + " + ".join(items)


def _correlated(factor_names: list[str]) -> str:
    lines = []
    for i in range(len(factor_names)):
        for j in range(i + 1, len(factor_names)):
            lines.append(f"{factor_names[i]} ~~ {factor_names[j]}")
    return "\n".join(lines)


def build_models() -> dict[str, str]:
    six = [_factor(sub.capitalize(), [sub]) for sub in SUBSCALES]
    six_names = [sub.capitalize() for sub in SUBSCALES]

    models: dict[str, str] = {}
    # M1: single factor.
    models["M1"] = _factor("Silence", SUBSCALES)
    # M2: fear-based vs rest.
    models["M2"] = (
        _factor("Fearbased", ["defensive", "diffident"])
        + "\n"
        + _factor("Other", ["ineffectual", "relational", "disengaged", "deviant"])
        + "\nFearbased ~~ Other"
    )
    # M3: Van Dyne 3-factor.
    models["M3"] = (
        _factor("Acquiescent", ["ineffectual", "diffident", "disengaged"])
        + "\n"
        + _factor("Defensive", ["defensive", "deviant"])
        + "\n"
        + _factor("Prosocial", ["relational"])
        + "\n"
        + _correlated(["Acquiescent", "Defensive", "Prosocial"])
    )
    # M4: Knoll 4-factor.
    models["M4"] = (
        _factor("Ineffectual", ["ineffectual", "diffident"])
        + "\n"
        + _factor("Relational", ["relational", "disengaged"])
        + "\n"
        + _factor("Defensive", ["defensive"])
        + "\n"
        + _factor("Deviant", ["deviant"])
        + "\n"
        + _correlated(["Ineffectual", "Relational", "Defensive", "Deviant"])
    )
    # M5: 5-factor (merge diffident into ineffectual).
    m5_names = ["Ineffectual", "Relational", "Defensive", "Disengaged", "Deviant"]
    models["M5"] = (
        _factor("Ineffectual", ["ineffectual", "diffident"])
        + "\n"
        + _factor("Relational", ["relational"])
        + "\n"
        + _factor("Defensive", ["defensive"])
        + "\n"
        + _factor("Disengaged", ["disengaged"])
        + "\n"
        + _factor("Deviant", ["deviant"])
        + "\n"
        + _correlated(m5_names)
    )
    # M6: 6-factor correlated (Brinsfield winner).
    models["M6"] = "\n".join(six) + "\n" + _correlated(six_names)
    # Mbi: bifactor (general + 6 specifics; orthogonal specifics).
    all_items = [it for sub in SUBSCALES for it in _items(sub)]
    general = "Silence =~ " + " + ".join(all_items)
    models["Mbi"] = general + "\n" + "\n".join(six)
    return models


def fit_one(model_spec: str, data: pd.DataFrame) -> dict[str, float]:
    from semopy import Model, calc_stats

    m = Model(model_spec)
    m.fit(data)
    stats = calc_stats(m)
    if isinstance(stats, pd.DataFrame) and not stats.empty:
        row = stats.iloc[0].to_dict()
    elif isinstance(stats, dict):
        row = stats
    else:
        row = {}

    def _get(k: str) -> float:
        for key in (k, k.lower(), k.upper()):
            if key in row:
                try:
                    return float(row[key])
                except (TypeError, ValueError):
                    pass
        return float("nan")

    return {
        "chi2": _get("chi2"),
        "df": _get("DoF"),
        "cfi": _get("CFI"),
        "tli": _get("TLI"),
        "rmsea": _get("RMSEA"),
        "aic": _get("AIC"),
        "bic": _get("BIC"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="brinsfield-tools cfa")
    parser.add_argument("--sample", default="synth")
    parser.add_argument("--input", default=None, help="(unused; kept for CLI symmetry)")
    parser.add_argument(
        "--models",
        default="M1,M2,M3,M4,M5,M6,Mbi",
        help="comma-separated subset of {M1..M6, Mbi}",
    )
    parser.add_argument("--output-base", default="results/track_a")
    args = parser.parse_args(argv)

    all_models = build_models()
    item_cols = [c for c in load_sample(args.sample, args.output_base).columns
                 if any(c.startswith(s) for s in SUBSCALES)]
    df = load_sample(args.sample, args.output_base)
    data = df[item_cols].apply(pd.to_numeric, errors="coerce").dropna()

    chosen = [m.strip() for m in args.models.split(",") if m.strip()]
    rows = []
    for name in chosen:
        if name not in all_models:
            print(f"warning: unknown model {name}; skipping", file=sys.stderr)
            continue
        try:
            stats = fit_one(all_models[name], data)
        except ImportError as exc:
            print(f"error: semopy not installed ({exc}); run `uv sync` first", file=sys.stderr)
            return 1
        except Exception as exc:  # noqa: BLE001 — synthetic / tiny data may fail
            print(f"warning: {name} failed to fit ({exc}); NaN row", file=sys.stderr)
            stats = {k: float("nan") for k in ("chi2", "df", "cfi", "tli", "rmsea", "aic", "bic")}
        rows.append({"model": name, **stats})

    out = pd.DataFrame(rows)
    out_path = os.path.join(args.output_base, args.sample, "cfa_summary.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print()

    # Report the 6-factor superiority verdict.
    if "M6" in out["model"].values:
        m6 = out[out["model"] == "M6"].iloc[0]
        others = out[out["model"].isin(["M1", "M2", "M3", "M4", "M5"])]
        if not others.empty and pd.notna(m6["cfi"]):
            best_other_cfi = others["cfi"].max()
            verdict = "PASS" if m6["cfi"] >= best_other_cfi else "review"
            print(
                f"[cfa] M6 (6-factor) CFI={m6['cfi']:.3f} vs best 1–5-factor CFI="
                f"{best_other_cfi:.3f}: 6-factor superiority {verdict}"
            )
    print(f"[cfa] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
