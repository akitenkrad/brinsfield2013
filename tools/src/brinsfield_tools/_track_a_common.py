"""Shared helpers for Track A modules — six-motive item layout + sample loading."""

from __future__ import annotations

import os
import sys

import pandas as pd

# Six Brinsfield subscales in canonical order; 5 items each (59 items in the
# paper across 6 factors; we use a balanced 30-item synthetic instrument).
SUBSCALES = ["ineffectual", "relational", "defensive", "diffident", "disengaged", "deviant"]
ITEMS_PER_SUBSCALE = 5

ITEM_NAMES: list[str] = [
    f"{sub}{i + 1}" for sub in SUBSCALES for i in range(ITEMS_PER_SUBSCALE)
]
SUBSCALE_OF: dict[str, str] = {
    f"{sub}{i + 1}": sub for sub in SUBSCALES for i in range(ITEMS_PER_SUBSCALE)
}

# 4 Study-4 correlate scales (supervisor-rated voice, psych safety, neuroticism,
# extraversion) — the nomological network targets.
CORRELATE_NAMES = ["voice", "psych_safety", "neuroticism", "extraversion"]


def sample_dir(output_base: str, sample: str) -> str:
    """Resolve `<output_base>/<sample>`. Created lazily by the loader."""
    return os.path.join(output_base, sample)


def load_sample(sample: str, output_base: str = "results/track_a") -> pd.DataFrame:
    """Load the `loaded.csv` produced by `survey-loader` for a named sample.

    Raises `SystemExit` with a clear hint if the sample is missing.
    """
    path = os.path.join(output_base, sample, "loaded.csv")
    if not os.path.exists(path):
        print(
            f"error: no loaded sample at {path}\n"
            f"  hint: run `brinsfield-tools survey-loader --synthesize-n 300 --sample {sample}` first",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return pd.read_csv(path)
