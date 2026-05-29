"""brinsfield-tools — Track A (psychometric) + Track B (ABM) utilities for the
Brinsfield (2013) six-motive employee-silence replication.

Track B (ABM visualization): `visualize`, `visualize_sweep`,
`show_experiment_settings`.

Track A (psychometric replication): `survey_loader` (with a calibrated
`--synthesize-n` path), `cfa` (semopy 6-factor vs 1–5-factor + bifactor),
`reproduce_paper` (one-command 6-factor superiority + motive distribution +
Study 4 ΔR² reconcile).

All subcommands dispatch through `brinsfield_tools.cli:main` —
see `brinsfield-tools --help`.
"""

__version__ = "0.1.0"
