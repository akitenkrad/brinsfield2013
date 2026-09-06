# Reproduction

[English](reproduction.md) | [日本語](reproduction.ja.md)

How Brinsfield (2013) Study 1–4 anchors map to the simulation outputs (design §5). The
anchors live in `simulation/src/calibration.rs`.

Of these, only the defensive 12.65% share is written into a run's `reference.csv`: it is
a number the paper printed *and* the same quantity the run measures as
`motive_mix_defensive`. The reference distribution behind `kl_to_reference` is not — it
was assembled by the design document from Study 1's prose, not printed by the paper. Nor
are the ±3% band, the ineffectual floor and the deviant ceiling: those are acceptance
criteria we chose, so they stay in `calibration.rs` and on the console.

| Anchor | Paper value | Where reproduced | Output |
|--------|-------------|------------------|--------|
| Study 1 — defensive share of all silence | 12.65% (288/2277) | ABM steady-state | `metrics.csv` `motive_mix_defensive` / `steady_state_motive_mix_defensive`; `reference.csv`; `reproduce` defensive verdict |
| Study 1 — ineffectual is most frequent | item 48 reported 230× | ABM steady-state | `reproduce` ineffectual floor (≥ .30) |
| Study 1 — deviant is rare | 0.48% | ABM steady-state | `reproduce` deviant ceiling (≤ .08) |
| Study 2 — EFA total variance | 58.27% | Track A (CFA proxy) | `cfa` fit indices |
| Study 3 — 6-factor CFI / NNFI | .96 / .96 | Track A (synthetic CFA) | `cfa_summary.csv` M6 CFI/TLI |
| Study 3 — 6-factor RMSEA | .087 | Track A | `cfa_summary.csv` M6 RMSEA |
| Study 3 — 6-factor superiority over 1–5-factor | M6 best AIC/BIC | Track A | `cfa` superiority verdict |
| Study 4 — ΔR² for VOICE (rel/def/ineff/diseng) | .05/.05/.04/.03 | Track A (synthetic) | `reproduce` ΔR² table |
| Study 4 — ψ → defensive/diffident/relational | negative | ABM correlations | `metrics.csv` run-scope `corr_<motive>_psafety` |
| Study 4 — neuroticism → deviant/diffident | positive | ABM correlations | `metrics.csv` run-scope `corr_<motive>_neuroticism` |
| ABM — competing-model KL | 6-dim < 4-dim < 3-dim | `ablate` | `mean_kl_to_reference` on each `ablate-point` child |

## One-command reproduction

```bash
# Track B (ABM) anchors:
cargo run --release -- reproduce --decision-mode rule_6dim --t-max 48 --seed 42

# Track A (psychometric) 6-factor superiority + ΔR² (synthetic data):
uv run brinsfield-tools reproduce --sample synth --synthesize-n 300
```

`reproduce` (Rust) prints the steady-state motive mix against the reference and PASS/off-anchor verdicts. `reproduce` (Python) fits the competing CFA family, confirms 6-factor superiority, reports the motive distribution and the Study 4 ΔR², and writes `three_way_comparison.csv`.
