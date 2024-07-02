use num_traits::Float;
use rand::Rng;
use rand_distr::{Distribution, Exp1, Open01, Standard, StandardNormal};
use serde_with::{serde_as, TryFromInto};
use std::{
    array,
    collections::{BTreeMap, BTreeSet},
};
use tracing::*;

use crate::{
    decision::{CptParams, LossParams, Prospect, CPT},
    dist::{IValue, IValueParam},
    info::{Info, InfoLabel, InfoTrustParams},
    opinion::{
        gen2::{self, DeducedOpinions, InitialOpinions, MyOpinionsUpd, Trusts},
        MyFloat,
    },
    value::{EValue, EValueParam},
};

#[serde_as]
#[derive(Debug, serde::Deserialize)]
pub struct AgentParams<V>
where
    V: MyFloat,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
{
    pub initial_opinions: InitialOpinions<V>,
    #[serde_as(as = "TryFromInto<IValueParam<V>>")]
    pub delay_selfish: IValue<V>,
    pub loss_params: LossParams<V>,
    pub cpt_params: CptParams<V>,
    pub trust_params: TrustParams<V>,
    #[serde_as(as = "TryFromInto<EValueParam<V>>")]
    pub access_prob: EValue<V>,
}

#[serde_as]
#[derive(Debug, serde::Deserialize)]
pub struct TrustParams<V>
where
    V: Float,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
{
    #[serde_as(as = "TryFromInto<EValueParam<V>>")]
    friend_access_prob: EValue<V>,
    #[serde_as(as = "TryFromInto<EValueParam<V>>")]
    social_access_prob: EValue<V>,
    #[serde_as(as = "TryFromInto<EValueParam<V>>")]
    friend_arrival_prob: EValue<V>,
    info_trust_params: InfoTrustParams<V>,
    #[serde_as(as = "TryFromInto<EValueParam<V>>")]
    friend_misinfo_trust: EValue<V>,
    #[serde_as(as = "TryFromInto<EValueParam<V>>")]
    social_misinfo_trust: EValue<V>,
}

#[derive(Debug)]
pub struct BehaviorByInfo {
    pub sharing: bool,
    pub first_access: bool,
}

#[derive(Default)]
pub struct Agent<V: MyFloat> {
    ops_gen2: gen2::MyOpinions<V>,
    infos_accessed: BTreeSet<usize>,
    decision: Decision<V>,
    trust: Trust<V>,
    access_prob: V,
}

#[derive(Default)]
struct Trust<V: Float> {
    friend_access_prob: V,
    social_access_prob: V,
    friend_arrival_rate: V,
    info_trust_map: BTreeMap<usize, V>,
    corr_misinfo_trust_map: BTreeMap<usize, V>,
    misinfo_friend: V,
    misinfo_social: V,
}

impl<V: Float> Trust<V> {
    fn reset<R: Rng>(&mut self, trust_params: &TrustParams<V>, rng: &mut R)
    where
        V: MyFloat,
        Open01: Distribution<V>,
        Standard: Distribution<V>,
        StandardNormal: Distribution<V>,
        Exp1: Distribution<V>,
    {
        self.friend_access_prob = trust_params.friend_access_prob.sample(rng);
        self.social_access_prob = trust_params.social_access_prob.sample(rng);
        self.friend_arrival_rate = trust_params.friend_arrival_prob.sample(rng);
        self.misinfo_friend = trust_params.friend_misinfo_trust.sample(rng);
        self.misinfo_social = trust_params.social_misinfo_trust.sample(rng);
        self.info_trust_map.clear();
        self.corr_misinfo_trust_map.clear();
    }

    fn to_inform(&self) -> Trusts<V>
    where
        V: MyFloat,
    {
        Trusts {
            info: V::one(),
            corr_misinfo: V::zero(),
            friend: V::zero(),
            social: V::zero(),
            pi_friend: V::one(),
            pi_social: V::one(),
            pred_friend: self.friend_arrival_rate * self.friend_access_prob,
        }
    }

    fn to_sharer<R: Rng>(
        &mut self,
        info: &Info<V>,
        receipt_prob: V,
        params: &TrustParams<V>,
        rng: &mut R,
    ) -> Trusts<V>
    where
        V: MyFloat,
        Open01: Distribution<V>,
        Standard: Distribution<V>,
        StandardNormal: Distribution<V>,
        Exp1: Distribution<V>,
    {
        let trust_sampler = params.info_trust_params.get_sampler(info.label());
        let info_trust = *self
            .info_trust_map
            .entry(info.idx)
            .or_insert_with(|| trust_sampler.sample(rng));
        let corr_misinfo_trust =
            *self
                .corr_misinfo_trust_map
                .entry(info.idx)
                .or_insert_with(|| {
                    params
                        .info_trust_params
                        .get_sampler(&InfoLabel::Misinfo)
                        .sample(rng)
                });

        Trusts {
            info: info_trust,
            corr_misinfo: corr_misinfo_trust,
            friend: self.friend_access_prob * receipt_prob,
            social: self.social_access_prob * receipt_prob,
            pi_friend: self.misinfo_friend,
            pi_social: self.misinfo_social,
            pred_friend: self.friend_arrival_rate * self.friend_access_prob,
        }
    }
}

#[derive(Default)]
struct Decision<V: Float> {
    cpt: CPT<V>,
    prospect: Prospect<V>,
    selfish_status: DelayActionStatus,
    sharing_statuses: BTreeMap<usize, ActionStatus>,
    delay_selfish: u32,
}

impl<V: MyFloat> Decision<V> {
    fn values_selfish(&self, ded: &DeducedOpinions<V>) -> [V; 2] {
        let p_theta = ded.p_theta();
        info!(target: "    TH", P = ?p_theta);
        let values: [V; 2] =
            array::from_fn(|i| self.cpt.valuate(&self.prospect.selfish[i], &p_theta));
        info!(target: "     X", V = ?values);
        values
    }

    fn try_decide_selfish(&mut self, upd: &MyOpinionsUpd<V>) {
        if !self.selfish_status.is_done() {
            upd.decide1(|ded| {
                self.selfish_status
                    .decide(self.values_selfish(ded), self.delay_selfish);
                info!(target: "selfsh", status = ?self.selfish_status);
            });
        }
    }

    fn predict(&self, upd: &mut MyOpinionsUpd<V>) {
        upd.decide2(|_, _| true);
    }

    fn try_decide_sharing(&mut self, upd: &mut MyOpinionsUpd<V>, info_idx: usize) -> bool {
        let sharing_status = self.sharing_statuses.entry(info_idx).or_default();
        if sharing_status.is_done() {
            false
        } else {
            upd.decide2(|ded, pred_ded| {
                let p_a_thetad = ded.p_a_thetad();
                let pred_p_a_thetad = pred_ded.p_a_thetad();
                info!(target: "   THd", P = ?p_a_thetad);
                info!(target: "  ~THd", P = ?pred_p_a_thetad);

                let values = [
                    self.cpt.valuate(&self.prospect.sharing[0], &p_a_thetad),
                    self.cpt
                        .valuate(&self.prospect.sharing[1], &pred_p_a_thetad),
                ];
                info!(target: "     Y", V = ?values);
                sharing_status.decide(values);
                info!(target: "sharng", status = ?sharing_status);
                sharing_status.is_done()
            })
        }
    }

    fn reset<R: Rng>(&mut self, agent_params: &AgentParams<V>, rng: &mut R)
    where
        Open01: Distribution<V>,
        Standard: Distribution<V>,
        StandardNormal: Distribution<V>,
        Exp1: Distribution<V>,
    {
        self.prospect.reset_with(&agent_params.loss_params, rng);
        self.cpt.reset_with(&agent_params.cpt_params, rng);
        self.selfish_status.reset();
        self.sharing_statuses.clear();
        self.delay_selfish = agent_params.delay_selfish.sample(rng);
    }
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

impl<V: MyFloat> Agent<V>
where
    Open01: Distribution<V>,
    Standard: Distribution<V>,
{
    pub fn reset<R>(&mut self, agent_params: &AgentParams<V>, rng: &mut R)
    where
        V: MyFloat,
        R: Rng,
        StandardNormal: Distribution<V>,
        Exp1: Distribution<V>,
        Open01: Distribution<V>,
    {
        self.decision.reset(agent_params, rng);
        self.trust.reset(&agent_params.trust_params, rng);

        self.infos_accessed.clear();
        self.access_prob = agent_params.access_prob.sample(rng);

        self.ops_gen2.reset(&agent_params.initial_opinions, rng);
    }

    pub fn is_willing_selfish(&self) -> bool {
        self.decision.selfish_status.is_willing()
    }

    pub fn progress_selfish_status(&mut self) -> bool {
        let p = self.decision.selfish_status.progress();
        info!(target: "selfsh", status = ?self.decision.selfish_status);
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
        trust_params: &TrustParams<V>,
        rng: &mut impl Rng,
    ) -> BehaviorByInfo
    where
        StandardNormal: Distribution<V>,
        Exp1: Distribution<V>,
        Open01: Distribution<V>,
    {
        let first_access = self.infos_accessed.insert(info.idx);
        let trusts = self.trust.to_sharer(info, receipt_prob, trust_params, rng);

        // compute values of prospects
        let mut upd = self.ops_gen2.receive(info.p, trusts);
        self.decision.try_decide_selfish(&upd);
        let sharing = self.decision.try_decide_sharing(&mut upd, info.idx);

        BehaviorByInfo {
            sharing,
            first_access,
        }
    }

    pub fn set_info_opinions(&mut self, info: &Info<V>) {
        let trusts = self.trust.to_inform();

        let mut upd = self.ops_gen2.receive(info.p, trusts);
        self.decision.try_decide_selfish(&upd);
        self.decision.predict(&mut upd);
    }
}

#[cfg(test)]
mod tests {
    use serde::Deserialize;
    use std::fs::read_to_string;

    use super::{ActionStatus, AgentParams, DelayActionStatus};

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

    #[test]
    fn test_toml() -> anyhow::Result<()> {
        let initial_opinions = toml::from_str::<toml::Value>(&read_to_string(
            "./test/config/test_initial_opinions.toml",
        )?)?;
        let mut agent_params = toml::from_str::<toml::Value>(&read_to_string(
            "./test/config/test_agent_params.toml",
        )?)?;
        agent_params
            .as_table_mut()
            .unwrap()
            .insert("initial_opinions".to_string(), initial_opinions);

        AgentParams::<f32>::deserialize(agent_params)?;
        Ok(())
    }
}
