use approx::UlpsEq;
use either::Either;
use log::debug;
use num_traits::{Float, NumAssign};
use rand::Rng;
use rand_distr::{uniform::SampleUniform, Distribution, Open01, Standard};
use serde_with::{serde_as, TryFromInto};
use std::{array, collections::BTreeMap, iter::Sum};

use crate::{
    cpt::{CptParams, Prospect, CPT},
    dist::{sample, Dist, DistParam},
    info::{Info, TrustDists},
    opinion::{
        compute_opinions, reset_opinions, FriendOpinions, GlobalBaseRates, InitialOpinions,
        Opinions,
    },
};

#[serde_as]
#[derive(serde::Deserialize)]
pub struct AgentParams<V>
where
    V: Float + NumAssign + UlpsEq + SampleUniform,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
{
    pub initial_opinions: InitialOpinions<V>,
    pub base_rates: GlobalBaseRates<V>,
    #[serde_as(as = "TryFromInto<DistParam<V>>")]
    pub read_dist: Either<V, Dist<V>>,
    #[serde_as(as = "TryFromInto<DistParam<V>>")]
    pub farrival_dist: Either<V, Dist<V>>,
    #[serde_as(as = "TryFromInto<DistParam<V>>")]
    pub fread_dist: Either<V, Dist<V>>,
    #[serde_as(as = "TryFromInto<DistParam<V>>")]
    pub pi_dist: Either<V, Dist<V>>,
    /// probability whether people has plural ignorance
    pub pi_prob: V,
    pub trust_dists: TrustDists<V>,
    pub cpt_params: CptParams<V>,
}

#[derive(Debug)]
pub struct Behavior {
    pub selfish: bool,
    pub sharing: bool,
}

pub struct Agent<V: Float> {
    pub cpt: CPT<V>,
    pub prospect: Prospect<V>,
    op: Opinions<V>,
    fop: FriendOpinions<V>,
    read_prob: V,
    friend_arrival_prob: V,
    friend_read_prob: V,
    done_selfish: bool,
    info_trust_map: BTreeMap<usize, V>,
    info_shared: BTreeMap<usize, bool>,
}

impl<V> Agent<V>
where
    V: Float + UlpsEq + NumAssign + Sum + Default + std::fmt::Debug + SampleUniform,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
{
    pub fn new() -> Self {
        Self {
            cpt: Default::default(),
            prospect: Default::default(),
            op: Default::default(),
            fop: Default::default(),
            read_prob: V::zero(),
            friend_arrival_prob: V::zero(),
            friend_read_prob: V::zero(),
            done_selfish: Default::default(),
            info_trust_map: Default::default(),
            info_shared: Default::default(),
        }
    }

    pub fn reset(
        &mut self,
        read_prob: V,
        friend_arrival_prob: V,
        friend_read_prob: V,
        pi_rate: V,
        initial_opinions: InitialOpinions<V>,
        base_rates: &GlobalBaseRates<V>,
    ) {
        self.read_prob = read_prob;
        self.friend_arrival_prob = friend_arrival_prob;
        self.friend_read_prob = friend_read_prob;

        reset_opinions(
            &mut self.op,
            &mut self.fop,
            pi_rate,
            initial_opinions,
            base_rates,
        );

        self.info_trust_map.clear();
        self.info_shared.clear();
        self.done_selfish = false;
    }

    pub fn reset_with<R: Rng>(&mut self, agent_params: &AgentParams<V>, rng: &mut R) {
        self.cpt.reset_with(&agent_params.cpt_params, rng);
        self.prospect.reset_with(&agent_params.cpt_params, rng);

        let r = rng.gen::<V>();
        let s = sample(&agent_params.pi_dist, rng);
        self.reset(
            sample(&agent_params.read_dist, rng),
            sample(&agent_params.farrival_dist, rng),
            sample(&agent_params.fread_dist, rng),
            if agent_params.pi_prob > r {
                s
            } else {
                V::zero()
            },
            agent_params.initial_opinions.clone(),
            &agent_params.base_rates,
        );
    }

    pub fn read_info_trustfully(
        &mut self,
        info: &Info<V>,
        receipt_prob: V,
        agent_params: &AgentParams<V>,
    ) -> Behavior {
        self.read_info_with_trust(info, receipt_prob, &agent_params.base_rates, V::one())
    }

    pub fn read_info(
        &mut self,
        info: &Info<V>,
        receipt_prob: V,
        agent_params: &AgentParams<V>,
        rng: &mut impl Rng,
    ) -> Option<Behavior>
    where
        V: SampleUniform,
        Open01: Distribution<V>,
        Standard: Distribution<V>,
    {
        let trust = *self
            .info_trust_map
            .entry(info.idx)
            .or_insert_with(|| agent_params.trust_dists.gen_map(rng)(info));
        if rng.gen::<V>() <= self.read_prob {
            Some(self.read_info_with_trust(info, receipt_prob, &agent_params.base_rates, trust))
        } else {
            None
        }
    }

    fn read_info_with_trust(
        &mut self,
        info: &Info<V>,
        receipt_prob: V,
        base_rates: &GlobalBaseRates<V>,
        trust: V,
    ) -> Behavior {
        let mut temp = compute_opinions(
            &mut self.op,
            &mut self.fop,
            info,
            self.friend_read_prob,
            receipt_prob,
            trust,
            base_rates,
        );
        // compute values of prospects
        let shared = self.info_shared.entry(info.idx).or_insert(false);
        let sharing = if *shared {
            false
        } else {
            let (prob, pred_prob) = temp.predicate_friend_opinions(
                &self.op,
                &self.fop,
                info,
                self.friend_arrival_prob * self.friend_read_prob * trust,
                base_rates,
            );
            let value_sharing: [V; 2] = [
                self.cpt.valuate(&self.prospect.sharing[0], &prob),
                self.cpt.valuate(&self.prospect.sharing[1], &pred_prob),
            ];
            debug!("V_Y:{:?}", value_sharing);
            *shared = value_sharing[0] < value_sharing[1];
            *shared
        };

        let theta_prob = temp.update_for_sharing(&mut self.op, &mut self.fop);
        let selfish = if self.done_selfish {
            false
        } else {
            let value_selfish: [V; 2] =
                array::from_fn(|i| self.cpt.valuate(&self.prospect.selfish[i], &theta_prob));
            debug!("V_X:{:?}", value_selfish);
            self.done_selfish = value_selfish[0] < value_selfish[1];
            self.done_selfish
        };

        Behavior { selfish, sharing }
    }
}
