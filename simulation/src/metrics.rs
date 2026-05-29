//! Aggregate metrics for the Brinsfield (2013) six-motive silence model.
//!
//! - **silence_rate** — fraction of employees with `expression == Silence`.
//! - **motive_mix** — primary-motive shares within the silent population
//!   (Σ = 1 when any silence exists; all-zero otherwise).
//! - **motive_vec_mean** — population mean of the soft six-motive vectors over
//!   silent employees.
//! - **defensive_share** — `motive_mix[defensive]` (the 12.65% anchor).
//! - **n_distinct_motives_active** — count of motives whose mean soft share
//!   exceeds a floor (face-validity of the 6-factor structure).
//! - **climate_of_silence** — `C(t) = (1/N) Σ 1[Silence ∧ private_concern<0]`.
//! - **pearson** — paired Pearson correlation (motive ↔ correlate).
//! - **kl_to_reference / kl_between** — KL to the Brinsfield reference mix, and
//!   between two motive vectors (LLM vs rule).
//! - **cronbach_alpha** — internal-consistency estimate over k repeated items.

use crate::calibration::REFERENCE_MOTIVE_MIX;
use crate::motives::{MotiveLabel, MotiveVec6};
use crate::world::{Expression, SilenceWorld};

/// Fraction of employees in `Silence` expression.
pub fn silence_rate(world: &SilenceWorld) -> f64 {
    let n = world.n_employees();
    if n == 0 {
        return 0.0;
    }
    let silent = world
        .employees
        .values()
        .filter(|e| e.expression == Expression::Silence)
        .count();
    silent as f64 / n as f64
}

/// Six-vector of primary-motive shares within the silent population
/// (Σ = 1 when any silence exists; all-zero otherwise). Canonical order.
pub fn motive_mix(world: &SilenceWorld) -> [f64; 6] {
    let mut counts = [0u64; 6];
    let mut total = 0u64;
    for e in world.employees.values() {
        if e.expression == Expression::Silence {
            counts[e.motive_vec.primary().index()] += 1;
            total += 1;
        }
    }
    if total == 0 {
        return [0.0; 6];
    }
    let mut out = [0.0; 6];
    for i in 0..6 {
        out[i] = counts[i] as f64 / total as f64;
    }
    out
}

/// Population mean of the soft six-motive vectors over silent employees
/// (re-normalised; uniform when no silence). Canonical order.
pub fn motive_vec_mean(world: &SilenceWorld) -> [f64; 6] {
    let mut acc = [0.0; 6];
    let mut n = 0u64;
    for e in world.employees.values() {
        if e.expression == Expression::Silence {
            let a = e.motive_vec.to_array();
            for i in 0..6 {
                acc[i] += a[i];
            }
            n += 1;
        }
    }
    if n == 0 {
        return [1.0 / 6.0; 6];
    }
    for v in acc.iter_mut() {
        *v /= n as f64;
    }
    acc
}

/// Defensive primary-motive share (the Brinsfield 12.65% anchor).
pub fn defensive_share(world: &SilenceWorld) -> f64 {
    motive_mix(world)[MotiveLabel::Defensive.index()]
}

/// Number of motives whose mean soft share exceeds `floor` (default 0.05).
pub fn n_distinct_motives_active(world: &SilenceWorld, floor: f64) -> usize {
    motive_vec_mean(world)
        .iter()
        .filter(|&&p| p > floor)
        .count()
}

/// Climate of silence `C(t) = (1/N) Σ 1[Silence ∧ private_concern < 0]`.
pub fn climate_of_silence(world: &SilenceWorld) -> f64 {
    let n = world.n_employees();
    if n == 0 {
        return 0.0;
    }
    let cnt = world
        .employees
        .values()
        .filter(|e| e.expression == Expression::Silence && e.private_concern < 0.0)
        .count();
    cnt as f64 / n as f64
}

/// Per-team climate-of-silence values, one entry per team.
pub fn team_climates(world: &SilenceWorld) -> Vec<f64> {
    let n_teams = world.teams.len();
    let mut counts = vec![0u64; n_teams];
    let mut sizes = vec![0u64; n_teams];
    for e in world.employees.values() {
        sizes[e.team] += 1;
        if e.expression == Expression::Silence && e.private_concern < 0.0 {
            counts[e.team] += 1;
        }
    }
    let mut out = vec![0.0; n_teams];
    for k in 0..n_teams {
        out[k] = if sizes[k] == 0 {
            0.0
        } else {
            counts[k] as f64 / sizes[k] as f64
        };
    }
    out
}

/// Pearson correlation between paired `x` and `y`. Returns 0 on degenerate
/// inputs (length mismatch / < 2 points / zero variance).
pub fn pearson(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }
    let n = x.len() as f64;
    let mean_x: f64 = x.iter().sum::<f64>() / n;
    let mean_y: f64 = y.iter().sum::<f64>() / n;
    let (mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0);
    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    let denom = (sxx * syy).sqrt();
    if denom <= 0.0 {
        0.0
    } else {
        sxy / denom
    }
}

/// KL divergence `D_KL(motive_mix ‖ reference)` to the Brinsfield reference.
pub fn kl_to_reference(mix: [f64; 6]) -> f64 {
    let p = mix;
    let q = REFERENCE_MOTIVE_MIX;
    let eps = 1e-9;
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

/// Symmetric-free KL between two motive vectors `D_KL(a ‖ b)` (LLM vs rule).
pub fn kl_between(a: &MotiveVec6, b: &MotiveVec6) -> f64 {
    a.kl_to(b)
}

/// Cronbach's α over a matrix of `n` respondents × `k` items.
/// `items[i]` is the i-th respondent's k-vector. Returns 0 for k < 2.
pub fn cronbach_alpha(items: &[Vec<f64>]) -> f64 {
    let n = items.len();
    if n < 2 {
        return 0.0;
    }
    let k = items[0].len();
    if k < 2 {
        return 0.0;
    }
    // Per-item variances.
    let mut item_var_sum = 0.0;
    for j in 0..k {
        let col: Vec<f64> = items.iter().map(|row| row[j]).collect();
        item_var_sum += variance(&col);
    }
    // Total-score variance.
    let totals: Vec<f64> = items.iter().map(|row| row.iter().sum()).collect();
    let total_var = variance(&totals);
    if total_var <= 0.0 {
        return 0.0;
    }
    (k as f64 / (k as f64 - 1.0)) * (1.0 - item_var_sum / total_var)
}

fn variance(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 {
        return 0.0;
    }
    let mean = x.iter().sum::<f64>() / n as f64;
    x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::{Employee, SilenceWorld, Team};
    use socsim_core::{AgentId, SimClock, SimRng};
    use socsim_net::SocialNetwork;
    use std::collections::BTreeMap;

    fn mini_world(specs: &[(Expression, MotiveVec6, f64)]) -> SilenceWorld {
        let mut rng = SimRng::from_seed(0);
        let ids: Vec<AgentId> = (0..specs.len()).map(|i| AgentId(i as u64)).collect();
        let net = SocialNetwork::erdos_renyi(&ids, 0.5, &mut rng);
        let mut emps: BTreeMap<AgentId, Employee> = BTreeMap::new();
        for (i, &id) in ids.iter().enumerate() {
            let mut e = Employee::neutral(0, 0, 0);
            e.expression = specs[i].0;
            e.motive_vec = specs[i].1;
            e.private_concern = specs[i].2;
            emps.insert(id, e);
        }
        SilenceWorld::new(SimClock::new(1), emps, vec![Team::default()], net)
    }

    fn one_hot(label: MotiveLabel) -> MotiveVec6 {
        let mut a = [0.0; 6];
        a[label.index()] = 1.0;
        MotiveVec6::from_array(a)
    }

    #[test]
    fn motive_mix_sums_to_one_with_silence() {
        let w = mini_world(&[
            (Expression::Silence, one_hot(MotiveLabel::Ineffectual), -0.5),
            (Expression::Silence, one_hot(MotiveLabel::Defensive), -0.5),
            (Expression::Voice, MotiveVec6::uniform(), 0.3),
        ]);
        let mix = motive_mix(&w);
        assert!((mix.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!((mix[MotiveLabel::Defensive.index()] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn motive_mix_zero_without_silence() {
        let w = mini_world(&[(Expression::Voice, MotiveVec6::uniform(), 0.1)]);
        assert_eq!(motive_mix(&w), [0.0; 6]);
    }

    #[test]
    fn defensive_share_matches_mix() {
        let w = mini_world(&[
            (Expression::Silence, one_hot(MotiveLabel::Defensive), -0.5),
            (Expression::Silence, one_hot(MotiveLabel::Defensive), -0.5),
            (Expression::Silence, one_hot(MotiveLabel::Relational), -0.5),
        ]);
        assert!((defensive_share(&w) - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn climate_counts_disagreeing_silent() {
        let w = mini_world(&[
            (Expression::Silence, MotiveVec6::uniform(), -0.5),
            (Expression::Silence, MotiveVec6::uniform(), -0.5),
        ]);
        assert!((climate_of_silence(&w) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn pearson_perfect() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        assert!((pearson(&x, &y) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn kl_to_reference_zero_at_reference() {
        let kl = kl_to_reference(REFERENCE_MOTIVE_MIX);
        assert!(kl.abs() < 1e-9, "kl = {kl}");
    }

    #[test]
    fn cronbach_alpha_high_for_consistent_items() {
        // 3 respondents, 4 items; items strongly correlated → high α.
        let items = vec![
            vec![1.0, 1.1, 0.9, 1.0],
            vec![5.0, 5.1, 4.9, 5.0],
            vec![3.0, 3.1, 2.9, 3.0],
        ];
        let a = cronbach_alpha(&items);
        assert!(a > 0.9, "alpha = {a}");
    }
}
