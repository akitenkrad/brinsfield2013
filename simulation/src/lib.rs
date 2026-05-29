//! Brinsfield (2013) — Six-motive employee silence simulation (Track B).
//!
//! A socsim-based ABM that operationalises the Brinsfield six-motive silence
//! taxonomy (`ineffectual / relational / defensive / diffident / disengaged /
//! deviant`) as a per-agent probability simplex ([`motives::MotiveVec6`]) on a
//! Watts–Strogatz organisational network.
//!
//! Four **mutually exclusive** decision modes are wired by `config.decision_mode`:
//!
//! - `--decision-mode llm` — [`mechanisms::VoiceDecisionLlm`]: an LLM (Ollama-first,
//!   OpenAI fallback via `socsim-llm`) decides VOICE/SILENCE and emits a six-motive
//!   distribution given the Brinsfield operational definitions.
//! - `--decision-mode rule_6dim` — [`mechanisms::VoiceDecisionRule`] with a six-motive
//!   sign-constrained softmax (the primary rule ablation).
//! - `--decision-mode rule_4dim` / `rule_3dim` — the same softmax collapsed to
//!   Knoll's 4 forms / Van Dyne's 3 forms (competing-model KL ablation).
//!
//! Ten further mechanisms run each step across socsim's 6-phase loop, including
//! the EMA [`mechanisms::MotiveDynamics`] that lets the cross-sectional Brinsfield
//! distribution emerge as a learning-dynamic steady state. See `main.rs` for the
//! `run` / `sweep` / `ablate` / `reproduce` CLI.

pub mod calibration;
pub mod config;
pub mod llm;
pub mod mechanisms;
pub mod metrics;
pub mod motives;
pub mod prompts;
pub mod simulation;
pub mod world;
