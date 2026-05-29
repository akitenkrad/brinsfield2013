//! World state for the Brinsfield (2013) six-motive silence model.
//!
//! Implements socsim's [`WorldState`] over employees living on a
//! [`SocialNetwork`] (Watts–Strogatz by default). Each employee carries a
//! context vector plus a six-motive probability simplex ([`MotiveVec6`]) that
//! replaces knoll2013's 4-value `Motive` enum.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use socsim_core::{AgentId, SimClock, WorldState};
use socsim_net::SocialNetwork;

use crate::motives::MotiveVec6;

// --------------------------------------------------------------------------- //
// Expression
// --------------------------------------------------------------------------- //

/// Public expression at step `t`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Expression {
    Voice,
    Silence,
    Neutral,
}

impl Expression {
    pub fn label(&self) -> &'static str {
        match self {
            Expression::Voice => "voice",
            Expression::Silence => "silence",
            Expression::Neutral => "neutral",
        }
    }
}

// --------------------------------------------------------------------------- //
// Employee / Team
// --------------------------------------------------------------------------- //

/// Per-employee state (design §4.3).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Employee {
    /// Hierarchical level (`0` = lowest).
    pub level: u8,
    /// Tenure in months.
    pub tenure: u32,
    /// Team membership index.
    pub team: usize,
    /// Private concern intensity `b_i ∈ [-1, 1]`.
    pub private_concern: f64,
    /// Current public expression `b̂_i`.
    pub expression: Expression,
    /// Fear `f_i ∈ [0, 1]` (Kish-Gephart 2009 — defensive driver).
    pub fear: f64,
    /// Psychological safety `ψ_i ∈ [0, 1]` (Edmondson 1999).
    pub psych_safety: f64,
    /// Implicit-voice-theory strength `ι_i ∈ [0, 1]` (Detert 2011).
    pub ivt_strength: f64,
    /// Neuroticism `n_i ∈ [0, 1]` (Study 4 — deviant / diffident driver).
    pub neuroticism: f64,
    /// Extraversion `e_i ∈ [0, 1]` (Study 4).
    pub extraversion: f64,
    /// Perceived neighbour-silence ratio `ρ_i ∈ [0, 1]` (set by `silence_spiral`).
    pub perceived_silence: f64,
    /// ★ Brinsfield six-motive vector (probability simplex, Σ = 1).
    pub motive_vec: MotiveVec6,
    /// VOICE threshold `θ_i ∈ [0, 1]` (Kuran 1995).
    pub voice_threshold: f64,
}

impl Employee {
    /// Initialise a "neutral" employee with mid-range context and a uniform
    /// motive vector. Context fields are typically overwritten at world init.
    pub fn neutral(team: usize, level: u8, tenure: u32) -> Self {
        Employee {
            level,
            tenure,
            team,
            private_concern: 0.0,
            expression: Expression::Neutral,
            fear: 0.3,
            psych_safety: 0.5,
            ivt_strength: 0.3,
            neuroticism: 0.4,
            extraversion: 0.4,
            perceived_silence: 0.5,
            motive_vec: MotiveVec6::uniform(),
            voice_threshold: 0.5,
        }
    }
}

/// Per-team state.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Team {
    /// Supervisor openness `u_k ∈ [-1, 1]`.
    pub supervisor_openness: f64,
    /// Cumulative team knowledge stock `K_k(t)`.
    pub knowledge_stock: f64,
    /// Team-level climate-of-silence proxy `C_k(t)`.
    pub climate: f64,
}

impl Default for Team {
    fn default() -> Self {
        Team {
            supervisor_openness: 0.0,
            knowledge_stock: 0.0,
            climate: 0.0,
        }
    }
}

// --------------------------------------------------------------------------- //
// SilenceWorld
// --------------------------------------------------------------------------- //

/// World state for the six-motive silence model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SilenceWorld {
    pub clock: SimClock,
    /// Employees keyed by sorted [`AgentId`] (sorted order = determinism).
    pub employees: BTreeMap<AgentId, Employee>,
    pub teams: Vec<Team>,
    /// Inter-employee social network (Watts–Strogatz by default).
    pub network: SocialNetwork,
    /// Issue salience `σ(t) ∈ [0, 1]`.
    pub issue_salience: f64,
    /// Whole-organisation climate of silence `C(t)`.
    pub climate_of_silence: f64,
    /// Organisation performance / knowledge proxy `Π(t)`.
    pub org_performance: f64,
    /// Agents touched by retaliation in the current step
    /// (cleared at the start of each step by `retaliation_event`).
    pub retaliation_this_step: Vec<AgentId>,
}

impl SilenceWorld {
    /// Build a world from employees + teams + an inter-employee network.
    pub fn new(
        clock: SimClock,
        employees: BTreeMap<AgentId, Employee>,
        teams: Vec<Team>,
        network: SocialNetwork,
    ) -> Self {
        SilenceWorld {
            clock,
            employees,
            teams,
            network,
            issue_salience: 0.5,
            climate_of_silence: 0.0,
            org_performance: 1.0,
            retaliation_this_step: Vec::new(),
        }
    }

    /// Total number of employees.
    pub fn n_employees(&self) -> usize {
        self.employees.len()
    }

    /// Perceived-silence ratio `ρ_i` over network neighbours. Isolated → 0.
    pub fn neighbour_silence_ratio(&self, id: AgentId) -> f64 {
        let neighbours = self.network.neighbors(id);
        if neighbours.is_empty() {
            return 0.0;
        }
        let mut silent = 0usize;
        for nb in &neighbours {
            if let Some(e) = self.employees.get(nb) {
                if e.expression == Expression::Silence {
                    silent += 1;
                }
            }
        }
        silent as f64 / neighbours.len() as f64
    }

    /// VOICE ratio over network neighbours (used by `prefalse_cascade`).
    pub fn neighbour_voice_ratio(&self, id: AgentId) -> f64 {
        let neighbours = self.network.neighbors(id);
        if neighbours.is_empty() {
            return 0.0;
        }
        let mut voice = 0usize;
        for nb in &neighbours {
            if let Some(e) = self.employees.get(nb) {
                if e.expression == Expression::Voice {
                    voice += 1;
                }
            }
        }
        voice as f64 / neighbours.len() as f64
    }

    /// Mean neighbour motive vector (used by `motive_dynamics` social term).
    pub fn neighbour_motive_mean(&self, id: AgentId) -> MotiveVec6 {
        let neighbours = self.network.neighbors(id);
        let mut acc = [0.0; 6];
        let mut n = 0usize;
        for nb in &neighbours {
            if let Some(e) = self.employees.get(nb) {
                if e.expression == Expression::Silence {
                    let a = e.motive_vec.to_array();
                    for i in 0..6 {
                        acc[i] += a[i];
                    }
                    n += 1;
                }
            }
        }
        if n == 0 {
            return MotiveVec6::uniform();
        }
        for v in acc.iter_mut() {
            *v /= n as f64;
        }
        let mut v = MotiveVec6::from_array(acc);
        v.normalize();
        v
    }
}

impl WorldState for SilenceWorld {
    fn agent_ids(&self) -> Vec<AgentId> {
        self.employees.keys().copied().collect()
    }

    fn clock(&self) -> &SimClock {
        &self.clock
    }

    fn clock_mut(&mut self) -> &mut SimClock {
        &mut self.clock
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use socsim_core::SimRng;

    #[test]
    fn neighbour_ratios_isolated_is_zero() {
        let mut rng = SimRng::from_seed(7);
        let ids: Vec<AgentId> = (0..4).map(|i| AgentId(i as u64)).collect();
        let net = SocialNetwork::erdos_renyi(&ids, 0.0, &mut rng);
        let mut emps: BTreeMap<AgentId, Employee> = BTreeMap::new();
        for &id in &ids {
            emps.insert(id, Employee::neutral(0, 0, 0));
        }
        let world = SilenceWorld::new(SimClock::new(1), emps, vec![Team::default()], net);
        assert_eq!(world.neighbour_silence_ratio(AgentId(0)), 0.0);
        assert_eq!(world.neighbour_voice_ratio(AgentId(0)), 0.0);
    }

    #[test]
    fn neighbour_motive_mean_uniform_when_no_silent_neighbours() {
        let mut rng = SimRng::from_seed(7);
        let ids: Vec<AgentId> = (0..4).map(|i| AgentId(i as u64)).collect();
        let net = SocialNetwork::watts_strogatz(&ids, 2, 0.1, &mut rng);
        let mut emps: BTreeMap<AgentId, Employee> = BTreeMap::new();
        for &id in &ids {
            emps.insert(id, Employee::neutral(0, 0, 0)); // all neutral
        }
        let world = SilenceWorld::new(SimClock::new(1), emps, vec![Team::default()], net);
        let m = world.neighbour_motive_mean(AgentId(0));
        for p in m.to_array() {
            assert!((p - 1.0 / 6.0).abs() < 1e-12);
        }
    }
}
