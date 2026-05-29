//! `MotiveVec6` — the Brinsfield (2013) six-motive probability simplex.
//!
//! Replaces knoll2013's 4-value `Motive` enum with a 6-dimensional soft
//! assignment over the Brinsfield silence-motive taxonomy:
//!
//! - `ineffectual` — "speaking up makes no difference" (most frequent; Study 1
//!   item 48 reported 230×).
//! - `relational`  — preserving relationships / sparing others.
//! - `defensive`   — fear-based self-protection. The paper's **12.65%** anchor.
//! - `diffident`   — lack of confidence / self-efficacy (links to the spiral of
//!   silence, Noelle-Neumann 1974).
//! - `disengaged`  — withdrawal / detachment (lowest reliability, α = .83).
//! - `deviant`     — deviant withholding (rare: 0.48%; social-desirability bias).
//!
//! The vector is a categorical distribution (entries ≥ 0, Σ = 1). [`primary`]
//! projects back to a single [`MotiveLabel`] (the paper's categorical view);
//! [`entropy`], [`kl_to`] and [`normalize`] support the motive-dynamics EMA and
//! the LLM-vs-rule KL comparison.

use serde::{Deserialize, Serialize};

/// The six Brinsfield motive labels in canonical order.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MotiveLabel {
    Ineffectual,
    Relational,
    Defensive,
    Diffident,
    Disengaged,
    Deviant,
}

impl MotiveLabel {
    /// All six labels in canonical order.
    pub const ALL: [MotiveLabel; 6] = [
        MotiveLabel::Ineffectual,
        MotiveLabel::Relational,
        MotiveLabel::Defensive,
        MotiveLabel::Diffident,
        MotiveLabel::Disengaged,
        MotiveLabel::Deviant,
    ];

    /// Stable lowercase label (CSV / JSON friendly).
    pub fn label(&self) -> &'static str {
        match self {
            MotiveLabel::Ineffectual => "ineffectual",
            MotiveLabel::Relational => "relational",
            MotiveLabel::Defensive => "defensive",
            MotiveLabel::Diffident => "diffident",
            MotiveLabel::Disengaged => "disengaged",
            MotiveLabel::Deviant => "deviant",
        }
    }

    /// Index into 0..6 in canonical order.
    pub fn index(&self) -> usize {
        match self {
            MotiveLabel::Ineffectual => 0,
            MotiveLabel::Relational => 1,
            MotiveLabel::Defensive => 2,
            MotiveLabel::Diffident => 3,
            MotiveLabel::Disengaged => 4,
            MotiveLabel::Deviant => 5,
        }
    }

    /// Map a 0..6 index back to a label (canonical order).
    pub fn from_index(i: usize) -> MotiveLabel {
        MotiveLabel::ALL[i.min(5)]
    }

    /// Parse a lowercase label (lenient: accepts 3-letter prefixes).
    pub fn parse(s: &str) -> Option<MotiveLabel> {
        match s.trim().to_ascii_lowercase().as_str() {
            "ineffectual" | "ineff" => Some(MotiveLabel::Ineffectual),
            "relational" | "rel" => Some(MotiveLabel::Relational),
            "defensive" | "def" => Some(MotiveLabel::Defensive),
            "diffident" | "dif" => Some(MotiveLabel::Diffident),
            "disengaged" | "dis" => Some(MotiveLabel::Disengaged),
            "deviant" | "dev" => Some(MotiveLabel::Deviant),
            _ => None,
        }
    }
}

/// Brinsfield six-motive probability simplex.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MotiveVec6 {
    pub ineffectual: f64,
    pub relational: f64,
    pub defensive: f64,
    pub diffident: f64,
    pub disengaged: f64,
    pub deviant: f64,
}

impl MotiveVec6 {
    /// Build from a 6-array in canonical order.
    pub fn from_array(a: [f64; 6]) -> Self {
        MotiveVec6 {
            ineffectual: a[0],
            relational: a[1],
            defensive: a[2],
            diffident: a[3],
            disengaged: a[4],
            deviant: a[5],
        }
    }

    /// Canonical-order 6-array view.
    pub fn to_array(&self) -> [f64; 6] {
        [
            self.ineffectual,
            self.relational,
            self.defensive,
            self.diffident,
            self.disengaged,
            self.deviant,
        ]
    }

    /// A uniform `1/6` simplex (used when VOICE: no motive is assigned, so the
    /// "decision motive" for the EMA is the maximum-entropy point).
    pub fn uniform() -> Self {
        MotiveVec6::from_array([1.0 / 6.0; 6])
    }

    /// argmax label — the paper's categorical (primary-motive) projection.
    /// Ties resolve to the earliest canonical index (deterministic).
    pub fn primary(&self) -> MotiveLabel {
        let a = self.to_array();
        let mut best = 0usize;
        for i in 1..6 {
            if a[i] > a[best] {
                best = i;
            }
        }
        MotiveLabel::from_index(best)
    }

    /// Shannon entropy in nats (motive uncertainty). 0 for a one-hot vector,
    /// `ln 6` for the uniform simplex.
    pub fn entropy(&self) -> f64 {
        let mut h = 0.0;
        for p in self.to_array() {
            if p > 0.0 {
                h -= p * p.ln();
            }
        }
        h
    }

    /// KL divergence `D_KL(self ‖ other)` (nats). A tiny floor on `other`
    /// avoids `log(0)`; `self_i = 0` rows contribute 0 (standard convention).
    pub fn kl_to(&self, other: &Self) -> f64 {
        let p = self.to_array();
        let q = other.to_array();
        let eps = 1e-12;
        let mut acc = 0.0;
        for i in 0..6 {
            if p[i] <= 0.0 {
                continue;
            }
            let qi = q[i].max(eps);
            acc += p[i] * (p[i] / qi).ln();
        }
        acc.max(0.0)
    }

    /// L1-normalise to the simplex in place. Negative entries are clamped to 0;
    /// an all-zero vector falls back to uniform.
    pub fn normalize(&mut self) {
        let mut a = self.to_array();
        for v in a.iter_mut() {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
        let s: f64 = a.iter().sum();
        if s <= 0.0 {
            *self = MotiveVec6::uniform();
            return;
        }
        for v in a.iter_mut() {
            *v /= s;
        }
        *self = MotiveVec6::from_array(a);
    }

    /// EMA blend toward `decision` with learning rate `eta`:
    /// `self ← (1-eta)·self + eta·decision`, re-normalised.
    pub fn ema_update(&mut self, decision: &Self, eta: f64) {
        let cur = self.to_array();
        let dec = decision.to_array();
        let mut next = [0.0; 6];
        for i in 0..6 {
            next[i] = (1.0 - eta) * cur[i] + eta * dec[i];
        }
        let mut v = MotiveVec6::from_array(next);
        v.normalize();
        *self = v;
    }
}

impl Default for MotiveVec6 {
    fn default() -> Self {
        MotiveVec6::uniform()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_sums_to_one() {
        let mut v = MotiveVec6::from_array([2.0, 1.0, 1.0, 1.0, 1.0, 0.0]);
        v.normalize();
        let s: f64 = v.to_array().iter().sum();
        assert!((s - 1.0).abs() < 1e-12, "sum = {s}");
    }

    #[test]
    fn normalize_clamps_negatives_and_handles_zero() {
        let mut v = MotiveVec6::from_array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        v.normalize();
        // All-nonpositive → falls back to uniform.
        for p in v.to_array() {
            assert!((p - 1.0 / 6.0).abs() < 1e-12);
        }
    }

    #[test]
    fn primary_is_argmax() {
        let v = MotiveVec6::from_array([0.05, 0.1, 0.5, 0.1, 0.1, 0.15]);
        assert_eq!(v.primary(), MotiveLabel::Defensive);
    }

    #[test]
    fn kl_non_negative_and_zero_at_identity() {
        let v = MotiveVec6::from_array([0.3, 0.2, 0.13, 0.13, 0.13, 0.11]);
        assert!(v.kl_to(&v).abs() < 1e-12);
        let w = MotiveVec6::uniform();
        assert!(v.kl_to(&w) >= 0.0);
        assert!(w.kl_to(&v) >= 0.0);
    }

    #[test]
    fn entropy_bounds() {
        let onehot = MotiveVec6::from_array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert!(onehot.entropy().abs() < 1e-12);
        let u = MotiveVec6::uniform();
        assert!((u.entropy() - 6f64.ln()).abs() < 1e-9);
    }

    #[test]
    fn ema_moves_toward_decision_and_stays_simplex() {
        let mut v = MotiveVec6::uniform();
        let dec = MotiveVec6::from_array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        v.ema_update(&dec, 0.1);
        assert!(v.ineffectual > 1.0 / 6.0);
        let s: f64 = v.to_array().iter().sum();
        assert!((s - 1.0).abs() < 1e-12);
    }

    #[test]
    fn label_round_trips() {
        for m in MotiveLabel::ALL {
            assert_eq!(MotiveLabel::from_index(m.index()), m);
            assert_eq!(MotiveLabel::parse(m.label()), Some(m));
        }
    }
}
