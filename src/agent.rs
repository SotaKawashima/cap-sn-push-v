use approx::UlpsEq;
use num_traits::{Float, NumAssign};
use rand::Rng;
use rand_distr::{uniform::SampleUniform, Distribution, Exp1, Open01, Standard, StandardNormal};
use serde_with::{serde_as, TryFromInto};
use std::{
    array,
    collections::{BTreeMap, BTreeSet},
    fmt,
    iter::Sum,
};
use tracing::{debug, info};

use crate::{
    decision::{CptParams, LossParams, Prospect, CPT},
    dist::{IValue, IValueParam},
    info::{Info, TrustParams},
    opinion::{
        ConditionalOpinions, GlobalBaseRates, InitialConditions, InitialOpinions, MyFloat,
        Opinions, TempOpinions,
    },
    value::{EValue, EValueParam},
};

#[serde_as]
#[derive(Debug, serde::Deserialize)]
pub struct AgentParams<V>
where
    V: Float + NumAssign + UlpsEq,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
{
    pub initial_opinions: InitialOpinions<V>,
    pub initial_conditions: InitialConditions<V>,
    pub base_rates: GlobalBaseRates<V>,
    #[serde_as(as = "TryFromInto<EValueParam<V>>")]
    pub access_prob: EValue<V>,
    #[serde_as(as = "TryFromInto<EValueParam<V>>")]
    pub friend_access_prob: EValue<V>,
    #[serde_as(as = "TryFromInto<EValueParam<V>>")]
    pub social_access_prob: EValue<V>,
    #[serde_as(as = "TryFromInto<EValueParam<V>>")]
    pub friend_arrival_prob: EValue<V>,
    #[serde_as(as = "TryFromInto<IValueParam<V>>")]
    pub delay_selfish: IValue<V>,
    pub trust_params: TrustParams<V>,
    pub loss_params: LossParams<V>,
    pub cpt_params: CptParams<V>,
}

#[derive(Debug)]
pub struct BehaviorByInfo {
    pub sharing: bool,
    pub first_access: bool,
}

#[derive(Default)]
pub struct Agent<V: Float> {
    cpt: CPT<V>,
    prospect: Prospect<V>,
    ops: Opinions<V>,
    conds: ConditionalOpinions<V>,
    access_prob: V,
    friend_access_prob: V,
    social_access_prob: V,
    friend_arrival_rate: V,
    info_trust_map: BTreeMap<usize, V>,
    infos_accessed: BTreeSet<usize>,
    selfish_status: DelayActionStatus,
    sharing_statuses: BTreeMap<usize, ActionStatus>,
    delay_selfish: u32,
}

#[derive(Default, Debug)]
enum ActionStatus {
    #[default]
    NotYet,
    Done,
}

impl ActionStatus {
    fn is_done(&self) -> bool {
        matches!(self, Self::Done)
    }

    /// Index 0 of `values` indicates 'do not perform this action', and index 1 indicates 'perform this action.'
    fn decide<V: Float>(&mut self, values: [V; 2]) {
        if values[0] < values[1] {
            *self = Self::Done;
        }
    }
}

#[derive(Default, Debug)]
enum DelayActionStatus {
    #[default]
    NotYet,
    Willing(u32),
    Done,
}

impl DelayActionStatus {
    fn reset(&mut self) {
        *self = Self::NotYet;
    }

    #[inline]
    fn is_willing(&self) -> bool {
        matches!(self, Self::Willing(_))
    }

    #[inline]
    fn is_done(&self) -> bool {
        matches!(self, Self::Done)
    }

    /// Index 0 of `values` indicates 'do not perform this action', and index 1 indicates 'perform this action.'
    fn decide<V: Float>(&mut self, values: [V; 2], delay: u32) {
        let perform = values[0] < values[1];
        match self {
            Self::NotYet if perform => {
                *self = Self::Willing(delay);
            }
            Self::Willing(_) if !perform => {
                *self = Self::NotYet;
            }
            _ => {}
        }
    }

    fn progress(&mut self) -> bool {
        match self {
            Self::Willing(0) => {
                *self = Self::Done;
                true
            }
            Self::Willing(r) => {
                *r -= 1;
                false
            }
            _ => false,
        }
    }
}

impl<V> Agent<V>
where
    V: Float + UlpsEq + SampleUniform,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
{
    pub fn reset_with<R>(&mut self, agent_params: &AgentParams<V>, rng: &mut R)
    where
        V: MyFloat,
        R: Rng,
        StandardNormal: Distribution<V>,
        Exp1: Distribution<V>,
        Open01: Distribution<V>,
    {
        self.prospect.reset_with(&agent_params.loss_params, rng);
        self.cpt.reset_with(&agent_params.cpt_params, rng);

        // if agent_params.pi_prob > rng.gen::<V>() {
        //     agent_params.pi_rate.sample(rng)
        // } else {
        //     V::zero()
        // },

        self.delay_selfish = agent_params.delay_selfish.sample(rng);
        self.access_prob = agent_params.access_prob.sample(rng);
        self.friend_access_prob = agent_params.friend_access_prob.sample(rng);
        self.social_access_prob = agent_params.social_access_prob.sample(rng);
        self.friend_arrival_rate = agent_params.friend_arrival_prob.sample(rng);

        let sample = agent_params.initial_conditions.sample(rng);
        self.ops = Opinions::new(
            agent_params.initial_opinions.clone(),
            &agent_params.base_rates,
            &sample,
        );
        self.conds = ConditionalOpinions::from_sample(sample, &agent_params.base_rates);

        self.info_trust_map.clear();
        self.selfish_status.reset();
        self.sharing_statuses.clear();
        self.infos_accessed.clear();
    }

    pub fn is_willing_selfish(&self) -> bool {
        self.selfish_status.is_willing()
    }

    pub fn progress_selfish_status(&mut self) -> bool {
        let p = self.selfish_status.progress();
        debug!(target: "selfsh", status = ?self.selfish_status);
        p
    }

    #[inline]
    pub fn access_prob(&self) -> V {
        self.access_prob
    }

    pub fn read_info(
        &mut self,
        info: &Info<V>,
        receipt_prob: V,
        agent_params: &AgentParams<V>,
        rng: &mut impl Rng,
    ) -> BehaviorByInfo
    where
        V: Sum + Default + fmt::Debug + NumAssign,
        StandardNormal: Distribution<V>,
        Exp1: Distribution<V>,
        Open01: Distribution<V>,
    {
        let first_access = self.infos_accessed.insert(info.idx);

        let trust = *self
            .info_trust_map
            .entry(info.idx)
            .or_insert_with(|| agent_params.trust_params.gen_map(rng)(info));
        let friend_trust = receipt_prob * self.friend_access_prob * trust;
        let social_trust = receipt_prob * self.social_access_prob * trust;

        let mut new_ops = self
            .ops
            .update(info.content, trust, friend_trust, social_trust);
        let temp_ops = new_ops.compute(
            info.content,
            social_trust,
            &self.conds,
            &agent_params.base_rates,
        );

        // compute values of prospects
        let sharing_status = self.sharing_statuses.entry(info.idx).or_default();
        let sharing = 'a: {
            if sharing_status.is_done() {
                break 'a false;
            }
            let (pred_new_fop, ps) = new_ops.predict(
                &temp_ops,
                info.content,
                friend_trust,
                self.friend_arrival_rate * self.friend_access_prob * trust,
                &self.conds,
                &agent_params.base_rates,
            );
            let values: [V; 2] =
                array::from_fn(|i| self.cpt.valuate(&self.prospect.sharing[i], &ps[i]));
            info!(target: "     Y", V = ?values);
            sharing_status.decide(values);
            if sharing_status.is_done() {
                new_ops.replace_pred_fop(pred_new_fop);
                true
            } else {
                false
            }
        };
        self.ops = new_ops;
        self.decide_selfish(temp_ops);

        BehaviorByInfo {
            sharing,
            first_access,
        }
    }

    pub fn set_info_opinions(&mut self, info: &Info<V>, base_rates: &GlobalBaseRates<V>)
    where
        V: Sum + Default + fmt::Debug + NumAssign,
    {
        let trust = V::one();
        let friend_trust = V::zero(); // * self.friend_access_prob * trust;
        let social_trust = V::zero(); // * self.social_access_prob * trust;

        let mut new_ops = self
            .ops
            .update(info.content, trust, friend_trust, social_trust);
        let temp = new_ops.compute(info.content, social_trust, &self.conds, base_rates);

        // posting info is equivalent to sharing it to friends with max trust.
        let (pred_fop, _) = new_ops.predict(
            &temp,
            info.content,
            friend_trust,
            self.friend_arrival_rate * self.friend_access_prob, // * trust,
            &self.conds,
            base_rates,
        );
        new_ops.replace_pred_fop(pred_fop);
        self.ops = new_ops;
        self.decide_selfish(temp);
    }

    fn decide_selfish(&mut self, temp_ops: TempOpinions<V>)
    where
        V: NumAssign + Sum + fmt::Debug + Default,
    {
        if self.selfish_status.is_done() {
            return;
        }
        let p = temp_ops.get_theta_projection();
        debug!(target: "    TH", P = ?p);
        let values: [V; 2] = array::from_fn(|i| self.cpt.valuate(&self.prospect.selfish[i], &p));
        info!(target: "     X", V = ?values);
        self.selfish_status.decide(values, self.delay_selfish);
        info!(target: "selfsh", status = ?self.selfish_status);
    }
}

#[cfg(test)]
mod tests {
    use crate::agent::ActionStatus;

    use super::DelayActionStatus;

    #[test]
    fn test_action_status() {
        let mut s = ActionStatus::default();
        assert!(matches!(s, ActionStatus::NotYet));
        s.decide([1.0, 0.0]);
        assert!(matches!(s, ActionStatus::NotYet));
        s.decide([0.0, 1.0]);
        assert!(s.is_done());

        let mut s = DelayActionStatus::default();
        assert!(matches!(s, DelayActionStatus::NotYet));
        s.decide([0.0, 1.0], 1);
        assert!(matches!(s, DelayActionStatus::Willing(1)));
        s.decide([0.0, 1.0], 1);
        assert!(matches!(s, DelayActionStatus::Willing(1)));
        s.progress();
        assert!(matches!(s, DelayActionStatus::Willing(0)));
        s.progress();
        assert!(s.is_done());

        s.reset();
        assert!(matches!(s, DelayActionStatus::NotYet));
        s.decide([0.0, 1.0], 2);
        s.progress();
        assert!(matches!(s, DelayActionStatus::Willing(1)));
        s.progress();
        assert!(matches!(s, DelayActionStatus::Willing(0)));
        s.decide([1.0, 0.0], 2);
        s.progress();
        assert!(matches!(s, DelayActionStatus::NotYet));
        s.reset();
        s.decide([1.0, 0.0], 1);
        s.progress();
        assert!(matches!(s, DelayActionStatus::NotYet));
        s.decide([0.0, 1.0], 1);
        s.progress();
        assert!(matches!(s, DelayActionStatus::Willing(0)));
        s.decide([0.0, 1.0], 1);
        assert!(matches!(s, DelayActionStatus::Willing(0)));
        s.progress();
        assert!(s.is_done());
    }
}
