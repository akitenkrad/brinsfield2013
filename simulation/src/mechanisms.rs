//! 11 mechanisms across the socsim 6-phase loop (design §4.3 table).
//!
//! | # | Mechanism                | Phase        | Role |
//! |---|--------------------------|--------------|------|
//! | 1 | `IssueSalience`          | Environment  | Update σ(t); optional shock |
//! | 2 | `RetaliationEvent`       | Environment  | Mark agents touched by retaliation |
//! | 3 | `FearAppraisal`          | Decision     | Threat appraisal of fear f_i |
//! | 4 | `VoiceDecisionRule{6,4,3}dim` / `VoiceDecisionLlm` | Decision | **★ mutually exclusive** |
//! | 5 | `SilenceSpiral`          | Interaction  | ρ_i ← neighbour silence ratio |
//! | 6 | `PrefalseCascade`        | Interaction  | Flip to VOICE if neighbour-VOICE > θ_i |
//! | 7 | `OrgPerformance`         | Reward       | Knowledge-stock update (ineffectual/disengaged drain) |
//! | 8 | `MotiveDynamics`         | PostStep     | **★ EMA update of the six-motive vector** |
//! | 9 | `PsafetyUpdate`          | PostStep     | ψ_i ← ψ_i + η·voiced − ν·retaliated |
//! |10 | `ClimateSilence`         | PostStep     | Aggregate C(t) and per-team climate |
//! |11 | `MotiveMetrics`          | PostStep     | Snapshot org_performance proxy for metrics |
//!
//! The decision mechanisms **snapshot all employees at step start** and write
//! the new expressions/motives from that snapshot (synchronous update).

use std::cell::RefCell;
use std::rc::Rc;

use rand::Rng;
use socsim_core::{
    derive_seed, AgentId, Mechanism, Phase, Result, SocsimError, StepContext, WorldState,
};
use socsim_llm::MetadataCollector;

use crate::config::{BetaGroup, LlmSettings};
use crate::llm::{llm_config, SilenceClient};
use crate::motives::{MotiveLabel, MotiveVec6};
use crate::prompts::{build_silence_prompt, parse_voice_decision};
use crate::world::{Expression, SilenceWorld};

// --------------------------------------------------------------------------- //
// Shared LLM client / metadata wrappers (mirrors knoll2013)
// --------------------------------------------------------------------------- //

pub type SharedClient = Rc<RefCell<SilenceClient>>;
pub type SharedMetadata = Rc<RefCell<MetadataCollector>>;

// --------------------------------------------------------------------------- //
// 1. IssueSalience  (Environment)
// --------------------------------------------------------------------------- //

/// Mean-reverting σ(t) with an optional step-`shock_t` exogenous bump.
pub struct IssueSalience {
    decay: f64,
    target: f64,
    shock_t: Option<u64>,
    shock_magnitude: f64,
}

impl IssueSalience {
    pub fn new(shock_t: Option<u64>, shock_magnitude: f64) -> Self {
        IssueSalience {
            decay: 0.10,
            target: 0.5,
            shock_t,
            shock_magnitude,
        }
    }
}

impl Mechanism<SilenceWorld> for IssueSalience {
    fn name(&self) -> &str {
        "issue_salience"
    }
    fn phases(&self) -> &'static [Phase] {
        &[Phase::Environment]
    }
    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, SilenceWorld>) -> Result<()> {
        let sigma = ctx.world.issue_salience;
        let mut new_sigma = sigma + self.decay * (self.target - sigma);
        if let Some(t_shock) = self.shock_t {
            if ctx.clock.t() == t_shock {
                new_sigma = (new_sigma + self.shock_magnitude).clamp(0.0, 1.0);
            }
        }
        ctx.world.issue_salience = new_sigma.clamp(0.0, 1.0);
        Ok(())
    }
}

// --------------------------------------------------------------------------- //
// 2. RetaliationEvent  (Environment)
// --------------------------------------------------------------------------- //

/// With probability `p_retaliate` per agent, mark them retaliated this step.
pub struct RetaliationEvent {
    p_retaliate: f64,
}

impl RetaliationEvent {
    pub fn new(p_retaliate: f64) -> Self {
        RetaliationEvent {
            p_retaliate: p_retaliate.clamp(0.0, 1.0),
        }
    }
}

impl Mechanism<SilenceWorld> for RetaliationEvent {
    fn name(&self) -> &str {
        "retaliation_event"
    }
    fn phases(&self) -> &'static [Phase] {
        &[Phase::Environment]
    }
    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, SilenceWorld>) -> Result<()> {
        ctx.world.retaliation_this_step.clear();
        if self.p_retaliate <= 0.0 {
            return Ok(());
        }
        let ids: Vec<AgentId> = ctx.world.agent_ids();
        for id in ids {
            if ctx.rng.gen::<f64>() < self.p_retaliate {
                ctx.world.retaliation_this_step.push(id);
            }
        }
        Ok(())
    }
}

// --------------------------------------------------------------------------- //
// 3. FearAppraisal  (Decision)
// --------------------------------------------------------------------------- //

/// `f_i ← clamp(f_i + α·retaliated − γ·max(u_team,0), 0, 1)`.
pub struct FearAppraisal {
    alpha: f64,
    gamma: f64,
}

impl FearAppraisal {
    pub fn new() -> Self {
        FearAppraisal {
            alpha: 0.30,
            gamma: 0.10,
        }
    }
}

impl Default for FearAppraisal {
    fn default() -> Self {
        Self::new()
    }
}

impl Mechanism<SilenceWorld> for FearAppraisal {
    fn name(&self) -> &str {
        "fear_appraisal"
    }
    fn phases(&self) -> &'static [Phase] {
        &[Phase::Decision]
    }
    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, SilenceWorld>) -> Result<()> {
        let retaliated: std::collections::HashSet<AgentId> =
            ctx.world.retaliation_this_step.iter().copied().collect();
        let ids: Vec<AgentId> = ctx.world.agent_ids();
        for id in ids {
            let team_idx = ctx.world.employees[&id].team;
            let u = ctx.world.teams[team_idx].supervisor_openness;
            let emp = ctx.world.employees.get_mut(&id).expect("agent missing");
            let r = if retaliated.contains(&id) { 1.0 } else { 0.0 };
            emp.fear = (emp.fear + self.alpha * r - self.gamma * u.max(0.0)).clamp(0.0, 1.0);
        }
        Ok(())
    }
}

// --------------------------------------------------------------------------- //
// 4a. VoiceDecisionRule — six-motive softmax (with 4/3-dim collapse)
// --------------------------------------------------------------------------- //

#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Context vector `(ψ, f, ι, ρ, σ, n, e, u)` for an agent.
fn context_vec(world: &SilenceWorld, id: AgentId) -> [f64; 8] {
    let emp = &world.employees[&id];
    let team = &world.teams[emp.team];
    [
        emp.psych_safety,
        emp.fear,
        emp.ivt_strength,
        emp.perceived_silence,
        world.issue_salience,
        emp.neuroticism,
        emp.extraversion,
        team.supervisor_openness,
    ]
}

/// Sign-constrained motive-row logit (design §4.3).
/// `x = (ψ, f, ι, ρ, σ, n, e, u)`.
fn motive_row_logit(motive: MotiveLabel, x: &[f64; 8], gain: f64) -> f64 {
    let (psi, fear, ivt, rho, _sigma, neuro, extra, _u) =
        (x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]);
    let raw = match motive {
        // ineffectual: driven by ι (futility) and ρ (everyone silent → futile)
        MotiveLabel::Ineffectual => 0.6 + 1.0 * ivt + 0.5 * rho - 0.3 * psi,
        // relational: negatively related to ψ; mildly raised by extraversion
        MotiveLabel::Relational => 0.2 - 0.6 * psi + 0.4 * extra,
        // defensive: fear-driven; strongly negative on ψ (Study 4 #16)
        MotiveLabel::Defensive => -0.4 + 1.4 * fear - 0.9 * psi,
        // diffident: neuroticism + spiral of silence ρ; negative on ψ
        MotiveLabel::Diffident => -0.2 + 0.8 * neuro + 0.6 * rho - 0.5 * psi,
        // disengaged: detachment; low extraversion, low salience involvement
        MotiveLabel::Disengaged => 0.1 - 0.5 * extra + 0.3 * ivt,
        // deviant: rare; neuroticism-positive (Study 4 #17), low ψ
        MotiveLabel::Deviant => -1.6 + 0.7 * neuro - 0.3 * psi,
    };
    gain * raw
}

/// Six-motive softmax over the canonical motive order.
fn motive_softmax6(x: &[f64; 8], gain: f64) -> [f64; 6] {
    let logits: [f64; 6] = [
        motive_row_logit(MotiveLabel::Ineffectual, x, gain),
        motive_row_logit(MotiveLabel::Relational, x, gain),
        motive_row_logit(MotiveLabel::Defensive, x, gain),
        motive_row_logit(MotiveLabel::Diffident, x, gain),
        motive_row_logit(MotiveLabel::Disengaged, x, gain),
        motive_row_logit(MotiveLabel::Deviant, x, gain),
    ];
    softmax6(&logits)
}

fn softmax6(logits: &[f64; 6]) -> [f64; 6] {
    let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut exps = [0.0; 6];
    let mut s = 0.0;
    for i in 0..6 {
        exps[i] = (logits[i] - m).exp();
        s += exps[i];
    }
    if s <= 0.0 {
        return [1.0 / 6.0; 6];
    }
    for v in exps.iter_mut() {
        *v /= s;
    }
    exps
}

/// Collapse a six-motive vector into a coarser dimensionality, then re-expand
/// to the canonical 6-array (mass folded into representative motives).
///
/// - 4-dim (Knoll): merge diffident→ineffectual and disengaged→relational,
///   keeping {ineffectual, relational, defensive, deviant} as the 4 carriers.
/// - 3-dim (Van Dyne): collapse to {acquiescent≈ineffectual+diffident+disengaged,
///   defensive, prosocial≈relational}; deviant folded into defensive.
fn collapse_to_dims(p: [f64; 6], n_dims: usize) -> [f64; 6] {
    match n_dims {
        4 => {
            let mut out = [0.0; 6];
            out[MotiveLabel::Ineffectual.index()] =
                p[MotiveLabel::Ineffectual.index()] + p[MotiveLabel::Diffident.index()];
            out[MotiveLabel::Relational.index()] =
                p[MotiveLabel::Relational.index()] + p[MotiveLabel::Disengaged.index()];
            out[MotiveLabel::Defensive.index()] = p[MotiveLabel::Defensive.index()];
            out[MotiveLabel::Deviant.index()] = p[MotiveLabel::Deviant.index()];
            out
        }
        3 => {
            let mut out = [0.0; 6];
            // acquiescent carrier = ineffectual + diffident + disengaged
            out[MotiveLabel::Ineffectual.index()] = p[MotiveLabel::Ineffectual.index()]
                + p[MotiveLabel::Diffident.index()]
                + p[MotiveLabel::Disengaged.index()];
            // prosocial carrier = relational
            out[MotiveLabel::Relational.index()] = p[MotiveLabel::Relational.index()];
            // defensive carrier = defensive + deviant
            out[MotiveLabel::Defensive.index()] =
                p[MotiveLabel::Defensive.index()] + p[MotiveLabel::Deviant.index()];
            out
        }
        _ => p,
    }
}

/// Sample a 0..6 index from a categorical distribution.
fn sample_categorical6(probs: &[f64; 6], u: f64) -> usize {
    let mut acc = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if u < acc {
            return i;
        }
    }
    5
}

/// Rule-mode voice/motive decision at a chosen dimensionality.
pub struct VoiceDecisionRule {
    beta: BetaGroup,
    n_dims: usize,
}

impl VoiceDecisionRule {
    pub fn new(beta: BetaGroup, n_dims: usize) -> Self {
        VoiceDecisionRule { beta, n_dims }
    }

    /// Name reflecting the dimensionality (`voice_decision_rule_{6,4,3}dim`).
    fn dim_name(&self) -> &'static str {
        match self.n_dims {
            4 => "voice_decision_rule_4dim",
            3 => "voice_decision_rule_3dim",
            _ => "voice_decision_rule_6dim",
        }
    }
}

impl Mechanism<SilenceWorld> for VoiceDecisionRule {
    fn name(&self) -> &str {
        self.dim_name()
    }
    fn phases(&self) -> &'static [Phase] {
        &[Phase::Decision]
    }
    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, SilenceWorld>) -> Result<()> {
        let ids: Vec<AgentId> = ctx.world.agent_ids();
        let snapshot: Vec<([f64; 8], AgentId)> = ids
            .iter()
            .map(|id| (context_vec(ctx.world, *id), *id))
            .collect();

        let mut updates: Vec<(AgentId, Expression, MotiveVec6)> = Vec::with_capacity(ids.len());
        for (x, id) in snapshot {
            // VOICE Bernoulli (ψ, u, σ raise voice; f, ι, ρ lower it).
            let voice_logit = self.beta.voice_intercept
                + self.beta.beta_psafety * x[0]
                + self.beta.beta_supervisor * x[7]
                + self.beta.beta_salience * x[4]
                - self.beta.beta_fear * x[1]
                - self.beta.beta_ivt * x[2]
                - self.beta.beta_rho * x[3];
            let p_voice = sigmoid(voice_logit);
            let u_voice: f64 = ctx.rng.gen();
            if u_voice < p_voice {
                updates.push((id, Expression::Voice, MotiveVec6::uniform()));
            } else {
                let mut probs = motive_softmax6(&x, self.beta.motive_gain);
                if self.n_dims != 6 {
                    probs = collapse_to_dims(probs, self.n_dims);
                    let s: f64 = probs.iter().sum();
                    if s > 0.0 {
                        for v in probs.iter_mut() {
                            *v /= s;
                        }
                    }
                }
                // The realised primary motive is sampled; the stored motive_vec
                // is the soft distribution (a one-hot of the sample blended with
                // the soft probs keeps the EMA meaningful).
                let u_motive: f64 = ctx.rng.gen();
                let idx = sample_categorical6(&probs, u_motive);
                let mut mv = MotiveVec6::from_array(probs);
                // Nudge mass toward the sampled motive so primary() is stable.
                let mut a = mv.to_array();
                a[idx] += 0.5;
                mv = MotiveVec6::from_array(a);
                mv.normalize();
                updates.push((id, Expression::Silence, mv));
            }
        }
        for (id, expr, mv) in updates {
            let emp = ctx.world.employees.get_mut(&id).expect("agent missing");
            emp.expression = expr;
            // Decision motive is recorded into a transient slot via motive_vec;
            // MotiveDynamics blends it as the EMA target on PostStep. For the
            // rule path we set the motive_vec directly so primary() reads true.
            if expr == Expression::Silence {
                emp.motive_vec = mv;
            }
        }
        Ok(())
    }
}

// --------------------------------------------------------------------------- //
// 4b. VoiceDecisionLlm  (Decision)
// --------------------------------------------------------------------------- //

/// LLM-driven voice decision. Per-agent prompt built from the local context +
/// Brinsfield operational definitions; `temperature=0` + `(agent_id, t)` seed +
/// prompt→response cache pseudo-determinise generation.
pub struct VoiceDecisionLlm {
    client: SharedClient,
    metadata: SharedMetadata,
    settings: LlmSettings,
    prompt_version: u8,
    llm_seed_root: u64,
}

impl VoiceDecisionLlm {
    pub fn new(
        client: SharedClient,
        metadata: SharedMetadata,
        settings: LlmSettings,
        prompt_version: u8,
        llm_seed_root: u64,
    ) -> Self {
        VoiceDecisionLlm {
            client,
            metadata,
            settings,
            prompt_version,
            llm_seed_root,
        }
    }
}

impl Mechanism<SilenceWorld> for VoiceDecisionLlm {
    fn name(&self) -> &str {
        "voice_decision"
    }
    fn phases(&self) -> &'static [Phase] {
        &[Phase::Decision]
    }
    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, SilenceWorld>) -> Result<()> {
        let ids: Vec<AgentId> = ctx.world.agent_ids();
        let t = ctx.clock.t();
        let mut prompts: Vec<(AgentId, String, u64)> = Vec::with_capacity(ids.len());
        for id in ids {
            let prompt = build_silence_prompt(ctx.world, id, self.prompt_version);
            let llm_seed = derive_seed(self.llm_seed_root, &[3, id.0, t]);
            prompts.push((id, prompt, llm_seed));
        }

        let mut updates: Vec<(AgentId, Expression, MotiveVec6)> = Vec::with_capacity(prompts.len());
        for (id, prompt, llm_seed) in prompts {
            let mut cfg = llm_config(&self.settings);
            cfg.seed = llm_seed;
            let text = {
                let mut client = self.client.borrow_mut();
                let resp = client.complete(&prompt, &cfg).map_err(|e| {
                    SocsimError::Mechanism(format!("voice_decision LLM call failed: {e}"))
                })?;
                self.metadata.borrow_mut().record(resp.metadata.clone());
                resp.text
            };
            let verdict = parse_voice_decision(&text);
            updates.push((id, verdict.expression, verdict.motive_vec));
        }
        for (id, expr, mv) in updates {
            let emp = ctx.world.employees.get_mut(&id).expect("agent missing");
            emp.expression = expr;
            if expr == Expression::Silence {
                emp.motive_vec = mv;
            }
        }
        Ok(())
    }
}

// --------------------------------------------------------------------------- //
// 5. SilenceSpiral  (Interaction)
// --------------------------------------------------------------------------- //

/// `ρ_i ← neighbour silence ratio` (Noelle-Neumann 1974).
pub struct SilenceSpiral;

impl Mechanism<SilenceWorld> for SilenceSpiral {
    fn name(&self) -> &str {
        "silence_spiral"
    }
    fn phases(&self) -> &'static [Phase] {
        &[Phase::Interaction]
    }
    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, SilenceWorld>) -> Result<()> {
        let ids: Vec<AgentId> = ctx.world.agent_ids();
        let new_rho: Vec<(AgentId, f64)> = ids
            .iter()
            .map(|id| (*id, ctx.world.neighbour_silence_ratio(*id)))
            .collect();
        for (id, r) in new_rho {
            if let Some(e) = ctx.world.employees.get_mut(&id) {
                e.perceived_silence = r;
            }
        }
        Ok(())
    }
}

// --------------------------------------------------------------------------- //
// 6. PrefalseCascade  (Interaction)
// --------------------------------------------------------------------------- //

/// Threshold cascade (Kuran 1995): a silent agent flips to VOICE if its
/// neighbour-VOICE ratio exceeds `θ_i`.
pub struct PrefalseCascade;

impl Mechanism<SilenceWorld> for PrefalseCascade {
    fn name(&self) -> &str {
        "prefalse_cascade"
    }
    fn phases(&self) -> &'static [Phase] {
        &[Phase::Interaction]
    }
    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, SilenceWorld>) -> Result<()> {
        let ids: Vec<AgentId> = ctx.world.agent_ids();
        let mut flips: Vec<AgentId> = Vec::new();
        for id in ids {
            let emp = &ctx.world.employees[&id];
            if emp.expression == Expression::Silence {
                let rv = ctx.world.neighbour_voice_ratio(id);
                if rv > emp.voice_threshold {
                    flips.push(id);
                }
            }
        }
        for id in flips {
            if let Some(e) = ctx.world.employees.get_mut(&id) {
                e.expression = Expression::Voice;
            }
        }
        Ok(())
    }
}

// --------------------------------------------------------------------------- //
// 7. OrgPerformance  (Reward)
// --------------------------------------------------------------------------- //

/// Per-team knowledge-stock update. VOICE contributes; silence whose primary
/// motive is ineffectual/disengaged drains the stock most.
pub struct OrgPerformance {
    delta: f64,
}

impl OrgPerformance {
    pub fn new() -> Self {
        OrgPerformance { delta: 0.10 }
    }
}

impl Default for OrgPerformance {
    fn default() -> Self {
        Self::new()
    }
}

impl Mechanism<SilenceWorld> for OrgPerformance {
    fn name(&self) -> &str {
        "org_performance"
    }
    fn phases(&self) -> &'static [Phase] {
        &[Phase::Reward]
    }
    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, SilenceWorld>) -> Result<()> {
        let n_teams = ctx.world.teams.len();
        let mut sizes = vec![0u64; n_teams];
        let mut voice_cnt = vec![0u64; n_teams];
        let mut drain_cnt = vec![0u64; n_teams];
        for e in ctx.world.employees.values() {
            sizes[e.team] += 1;
            match e.expression {
                Expression::Voice => voice_cnt[e.team] += 1,
                Expression::Silence => {
                    let p = e.motive_vec.primary();
                    if matches!(p, MotiveLabel::Ineffectual | MotiveLabel::Disengaged) {
                        drain_cnt[e.team] += 1;
                    }
                }
                Expression::Neutral => {}
            }
        }
        for k in 0..n_teams {
            let n = sizes[k].max(1) as f64;
            let v = voice_cnt[k] as f64 / n;
            let d = drain_cnt[k] as f64 / n;
            let team = &mut ctx.world.teams[k];
            team.knowledge_stock = ((1.0 - self.delta) * team.knowledge_stock + v - d).max(0.0);
        }
        Ok(())
    }
}

// --------------------------------------------------------------------------- //
// 8. MotiveDynamics  (PostStep) — ★ EMA update of the six-motive vector
// --------------------------------------------------------------------------- //

/// `motive_vec_i ← (1-η)·motive_vec_i + η·decision_motive_i` plus a small social
/// pull toward the mean *silent* neighbour motive. For silent agents the
/// decision motive is the freshly-set `motive_vec`; for voicing agents the
/// decision motive is uniform (max entropy), gently relaxing the vector.
pub struct MotiveDynamics {
    eta: f64,
    social_weight: f64,
}

impl MotiveDynamics {
    pub fn new(eta: f64) -> Self {
        MotiveDynamics {
            eta: eta.clamp(0.0, 1.0),
            social_weight: 0.25,
        }
    }
}

impl Mechanism<SilenceWorld> for MotiveDynamics {
    fn name(&self) -> &str {
        "motive_dynamics"
    }
    fn phases(&self) -> &'static [Phase] {
        &[Phase::PostStep]
    }
    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, SilenceWorld>) -> Result<()> {
        let ids: Vec<AgentId> = ctx.world.agent_ids();
        // Snapshot decision targets (current motive_vec / expression) + social
        // means before mutating (synchronous update).
        let mut targets: Vec<(AgentId, MotiveVec6)> = Vec::with_capacity(ids.len());
        for &id in &ids {
            let emp = &ctx.world.employees[&id];
            let own = emp.motive_vec;
            let social = ctx.world.neighbour_motive_mean(id);
            // Decision target: silent → own decision; voicing → uniform.
            let decision = if emp.expression == Expression::Silence {
                own
            } else {
                MotiveVec6::uniform()
            };
            // Blend in the social mean.
            let d = decision.to_array();
            let s = social.to_array();
            let mut blended = [0.0; 6];
            for i in 0..6 {
                blended[i] = (1.0 - self.social_weight) * d[i] + self.social_weight * s[i];
            }
            let mut tgt = MotiveVec6::from_array(blended);
            tgt.normalize();
            targets.push((id, tgt));
        }
        for (id, tgt) in targets {
            if let Some(e) = ctx.world.employees.get_mut(&id) {
                let mut mv = e.motive_vec;
                mv.ema_update(&tgt, self.eta);
                e.motive_vec = mv;
            }
        }
        Ok(())
    }
}

// --------------------------------------------------------------------------- //
// 9. PsafetyUpdate  (PostStep)
// --------------------------------------------------------------------------- //

/// `ψ_i ← clamp(ψ_i + η·voiced − ν·retaliated, 0, 1)`.
pub struct PsafetyUpdate {
    eta: f64,
    nu: f64,
}

impl PsafetyUpdate {
    pub fn new(eta: f64) -> Self {
        PsafetyUpdate { eta, nu: 0.15 }
    }
}

impl Mechanism<SilenceWorld> for PsafetyUpdate {
    fn name(&self) -> &str {
        "psafety_update"
    }
    fn phases(&self) -> &'static [Phase] {
        &[Phase::PostStep]
    }
    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, SilenceWorld>) -> Result<()> {
        let retaliated: std::collections::HashSet<AgentId> =
            ctx.world.retaliation_this_step.iter().copied().collect();
        let ids: Vec<AgentId> = ctx.world.agent_ids();
        for id in ids {
            let emp = ctx.world.employees.get_mut(&id).expect("agent missing");
            let voiced = matches!(emp.expression, Expression::Voice);
            let was_retaliated = retaliated.contains(&id);
            let delta =
                self.eta * (voiced as i32 as f64) - self.nu * (was_retaliated as i32 as f64);
            emp.psych_safety = (emp.psych_safety + delta).clamp(0.0, 1.0);
        }
        Ok(())
    }
}

// --------------------------------------------------------------------------- //
// 10. ClimateSilence  (PostStep)
// --------------------------------------------------------------------------- //

/// Updates `world.climate_of_silence` and per-team `team.climate`.
pub struct ClimateSilence;

impl Mechanism<SilenceWorld> for ClimateSilence {
    fn name(&self) -> &str {
        "climate_silence"
    }
    fn phases(&self) -> &'static [Phase] {
        &[Phase::PostStep]
    }
    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, SilenceWorld>) -> Result<()> {
        ctx.world.climate_of_silence = crate::metrics::climate_of_silence(ctx.world);
        let per_team = crate::metrics::team_climates(ctx.world);
        for (k, c) in per_team.into_iter().enumerate() {
            if let Some(team) = ctx.world.teams.get_mut(k) {
                team.climate = c;
            }
        }
        Ok(())
    }
}

// --------------------------------------------------------------------------- //
// 11. MotiveMetrics  (PostStep)
// --------------------------------------------------------------------------- //

/// Snapshots the organisation-level knowledge proxy into `org_performance`
/// (per-step metric monitoring of the Brinsfield anchors lives in
/// `simulation::run_with_client`, which reads the world after this).
pub struct MotiveMetrics;

impl Mechanism<SilenceWorld> for MotiveMetrics {
    fn name(&self) -> &str {
        "motive_metrics"
    }
    fn phases(&self) -> &'static [Phase] {
        &[Phase::PostStep]
    }
    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, SilenceWorld>) -> Result<()> {
        let n_teams = ctx.world.teams.len().max(1) as f64;
        let total: f64 = ctx.world.teams.iter().map(|t| t.knowledge_stock).sum();
        ctx.world.org_performance = total / n_teams;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_at_zero_is_half() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn motive_softmax6_sums_to_one() {
        let x = [0.5, 0.3, 0.3, 0.5, 0.5, 0.4, 0.4, 0.0];
        let p = motive_softmax6(&x, 1.0);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn high_fear_raises_defensive_share() {
        // Holding the other dims fixed, raising fear must monotonically raise
        // the defensive share (defensive is the only fear-driven row).
        let lo = [0.3, 0.1, 0.1, 0.1, 0.5, 0.3, 0.3, 0.0];
        let hi = [0.3, 0.95, 0.1, 0.1, 0.5, 0.3, 0.3, 0.0];
        let p_lo = motive_softmax6(&lo, 1.0);
        let p_hi = motive_softmax6(&hi, 1.0);
        let d = MotiveLabel::Defensive.index();
        assert!(
            p_hi[d] > p_lo[d],
            "defensive share should rise with fear: {} -> {}",
            p_lo[d],
            p_hi[d]
        );
    }

    #[test]
    fn collapse_preserves_mass() {
        let p = [0.2, 0.2, 0.2, 0.15, 0.15, 0.1];
        for dims in [3usize, 4] {
            let c = collapse_to_dims(p, dims);
            assert!((c.iter().sum::<f64>() - 1.0).abs() < 1e-9, "dims={dims}");
        }
    }

    #[test]
    fn collapse_4dim_zeroes_diffident_disengaged() {
        let p = [0.2, 0.2, 0.2, 0.15, 0.15, 0.1];
        let c = collapse_to_dims(p, 4);
        assert_eq!(c[MotiveLabel::Diffident.index()], 0.0);
        assert_eq!(c[MotiveLabel::Disengaged.index()], 0.0);
        // mass folded in
        assert!((c[MotiveLabel::Ineffectual.index()] - 0.35).abs() < 1e-9);
    }

    #[test]
    fn sample_categorical6_edges() {
        let p = [0.5, 0.0, 0.0, 0.0, 0.0, 0.5];
        assert_eq!(sample_categorical6(&p, 0.1), 0);
        assert_eq!(sample_categorical6(&p, 0.7), 5);
    }
}
