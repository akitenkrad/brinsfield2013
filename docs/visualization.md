# Visualization

[English](visualization.md) | [日本語](visualization.ja.md)

The `brinsfield-tools` Python package renders the ABM outputs.

Which run to read is resolved by `runvault path --latest`, so `--results-dir` is
optional; pass it to pick a specific run, including a legacy `results/<timestamp>/`
directory from before the runvault migration. Figures are written outside the run
(`<results-root>/brinsfield-silence/figures/<run_slug>/`): `manifest.csv` is settled by
`finish()`, so anything made after a run ended must not land inside it.

## `visualize` (single run)

```bash
uv run brinsfield-tools visualize
uv run brinsfield-tools visualize --results-dir results/20260529_143329   # legacy
```

- `motive_mix_stack.png` — six-region stacked area chart of the primary-motive shares over time.
- `silence_kl_timeseries.png` — silence rate and `KL(mix‖reference)` per step.
- `motive_correlate_bar.png` — final-step Pearson r of each motive against ψ (negative for defensive/diffident/relational) and neuroticism (positive for deviant/diffident).

## `visualize-sweep`

```bash
uv run brinsfield-tools visualize-sweep
```

The `sweep_summary.csv` table is no longer on disk: it is rebuilt from the sweep
parent's children (`brinsfield_tools.runs.sweep_summary_table`).

- `sweep_defensive_heatmap.png` — defensive share across ψ_learn × p_retaliate.
- `sweep_kl_contour.png` — `KL(mix‖reference)` contour over motive-init-defensive × p_retaliate.
- `sweep_motive_response.png` — steady-state motive shares vs initial defensive share (with the 12.65% anchor line).

## `show-experiment-settings`

```bash
uv run brinsfield-tools show-experiment-settings
uv run brinsfield-tools show-experiment-settings --subcommand sweep
```

Prints the conditions from the run's `config.json` plus, for LLM runs only, the LLM
provenance: provider / model / temperature from the `run.json` `llm` block and
calls / cache-hit from the run-scope `llm_*` metrics. A rule-mode run has no `llm`
block, so it no longer claims a model called `none`. Add `--json` for machine-readable
output.
