//! Brinsfield (2013) Study 1–4 numeric anchors (design doc §5).
//!
//! These constants pin the ABM (and the synthetic-data Track A path) to the
//! paper's empirical targets so that `metrics`/`reproduce` can report distance-
//! to-anchor. They are the single source of truth for the "did we hit the
//! number" checks; nothing else hard-codes these values.

use crate::motives::MotiveLabel;

/// Study 1: defensive (fear-based) motive share of all silence (288 / 2277).
pub const DEFENSIVE_SHARE_ANCHOR: f64 = 0.1265;
/// Acceptance band around the defensive anchor (±3%, design §5 #2).
pub const DEFENSIVE_SHARE_TOL: f64 = 0.03;

/// Study 1 fact 3: ineffectual is the most frequent motive (target ≥ 30%).
pub const INEFFECTUAL_FLOOR: f64 = 0.30;
/// Deviant is rare (target ≤ 8%; Study 1 0.48% + social-desirability correction).
pub const DEVIANT_CEILING: f64 = 0.08;

/// Study 1 Fleiss κ for incident / target / reason coding (design §5 #1).
pub const FLEISS_KAPPA_INCIDENT: f64 = 0.63;
pub const FLEISS_KAPPA_TARGET: f64 = 0.83;
pub const FLEISS_KAPPA_REASON: f64 = 0.73;

/// Study 2 EFA total variance explained (58.27%) and the largest eigenvalue.
pub const EFA_VARIANCE_EXPLAINED: f64 = 0.5827;
pub const EFA_LARGEST_EIGENVALUE: f64 = 20.25;

/// Study 3 six-factor CFA fit indices.
pub const CFA_CFI: f64 = 0.96;
pub const CFA_NNFI: f64 = 0.96;
pub const CFA_RMSEA: f64 = 0.087;
pub const CFA_CHI2_DF: f64 = 4.36;

/// Study 3 Cronbach α lower-bound targets per subscale (design §5 #9–10).
/// Order = canonical [`MotiveLabel`] order.
pub const ALPHA_TARGET: [f64; 6] = [
    0.90, // ineffectual
    0.92, // relational
    0.92, // defensive
    0.89, // diffident
    0.83, // disengaged
    0.95, // deviant
];

/// Study 4 incremental validity `ΔR²` for VOICE per motive (design §5 #12–15).
/// Order = canonical [`MotiveLabel`] order; ineffectual/deviant not in Study 4
/// hierarchical regression are set to the .03 floor / 0.0 respectively.
pub const DELTA_R2_TARGET: [f64; 6] = [
    0.04, // ineffectual
    0.05, // relational
    0.05, // defensive
    0.03, // diffident (proxy)
    0.03, // disengaged
    0.0,  // deviant (not modelled in Study 4 hierarchical regression)
];

/// Sign of the expected ψ → motive correlation (Study 4 #16): defensive,
/// diffident and relational are negatively related to psychological safety.
pub fn psafety_sign(m: MotiveLabel) -> f64 {
    match m {
        MotiveLabel::Defensive | MotiveLabel::Diffident | MotiveLabel::Relational => -1.0,
        _ => 0.0,
    }
}

/// Sign of the expected neuroticism → motive correlation (Study 4 #17):
/// deviant and diffident are positively related to neuroticism.
pub fn neuroticism_sign(m: MotiveLabel) -> f64 {
    match m {
        MotiveLabel::Deviant | MotiveLabel::Diffident => 1.0,
        _ => 0.0,
    }
}

/// Reference six-motive distribution (canonical order) used as the KL target
/// and the "paper distribution" in `reproduce`. Built from the Study 1 facts:
/// ineffectual dominant, defensive ≈ 12.65%, deviant rare.
pub const REFERENCE_MOTIVE_MIX: [f64; 6] = [
    0.37,   // ineffectual (dominant)
    0.20,   // relational
    0.1265, // defensive (anchor)
    0.15,   // diffident
    0.12,   // disengaged
    0.0335, // deviant (rare)
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_mix_is_normalised() {
        let s: f64 = REFERENCE_MOTIVE_MIX.iter().sum();
        assert!((s - 1.0).abs() < 1e-9, "reference mix sum = {s}");
    }

    #[test]
    fn reference_mix_respects_anchors() {
        // Bind to locals so clippy does not flag these as constant assertions.
        let mix = REFERENCE_MOTIVE_MIX;
        let (anchor, floor, ceil) = (DEFENSIVE_SHARE_ANCHOR, INEFFECTUAL_FLOOR, DEVIANT_CEILING);
        assert!((mix[2] - anchor).abs() < 1e-9);
        assert!(mix[0] >= floor);
        assert!(mix[5] <= ceil);
    }
}
