"""brinsfield-tools — unified CLI dispatcher.

Track B (ABM):
    brinsfield-tools visualize                 # 6-region stacked motive time-series
    brinsfield-tools visualize-sweep           # motive × ψ heatmap + KL contour
    brinsfield-tools show-experiment-settings  # print config / sweep_config / llm_meta

Track A (psychometrics):
    brinsfield-tools survey-loader   # real --csv or calibrated --synthesize-n
    brinsfield-tools cfa             # 6-factor vs 1–5-factor + bifactor/ESEM (semopy)
    brinsfield-tools reproduce       # one-command 6-factor superiority + motive dist

Arguments after the subcommand are passed verbatim to that subcommand's argparse.
Add `--help` after a subcommand for its own help.

The dispatcher assembly is delegated to the shared helper
`socsim_tools.cli.build_dispatcher`.
"""

from __future__ import annotations

from socsim_tools.cli import build_dispatcher

main = build_dispatcher(
    prog="brinsfield-tools",
    description="Brinsfield (2013) six-motive employee silence — Track A + Track B utilities",
    subcommands={
        # ── Track B (ABM visualization) ─────────────────────────────────────
        "visualize": (
            "single-run visualization (6-region stacked motive time-series + climate)",
            "brinsfield_tools.visualize:main",
        ),
        "visualize-sweep": (
            "sweep visualization (motive × ψ heatmap + KL-to-reference contour)",
            "brinsfield_tools.visualize_sweep:main",
        ),
        "show-experiment-settings": (
            "print a results directory's settings (config / sweep_config / llm_meta)",
            "brinsfield_tools.show_experiment_settings:main",
        ),
        # ── Track A (psychometric replication) ─────────────────────────────
        "survey-loader": (
            "Track A: load survey CSV (real --csv or synthesised --synthesize-n)",
            "brinsfield_tools.survey_loader:main",
        ),
        "cfa": (
            "Track A: competing CFA (6-factor vs 1–5-factor + bifactor/ESEM, semopy)",
            "brinsfield_tools.cfa:main",
        ),
        "reproduce": (
            "Track A+B: one-command 6-factor superiority + motive distribution reconcile",
            "brinsfield_tools.reproduce_paper:main",
        ),
    },
)


if __name__ == "__main__":
    main()
