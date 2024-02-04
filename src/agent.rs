use approx::UlpsEq;
use either::Either;
use log::debug;
use num_traits::{Float, NumAssign};
use rand::Rng;
use rand_distr::{uniform::SampleUniform, Distribution, Open01, Standard};
use std::{array, iter::Sum};

use crate::{
    cpt::{CptParams, Prospect, CPT},
    dist::{sample, Dist},
    info::{Info, TrustDists},
    opinion::{
        compute_opinions, FriendOpinions, FriendStaticOpinions, GlobalBaseRates, Opinions,
        PluralIgnorance, StaticOpinions,
    },
};

pub struct Constants<V>
where
    V: Float + SampleUniform,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
{
    pub base_rates: GlobalBaseRates<V>,
    pub read_dist: Either<V, Dist<V>>,
    pub farrival_dist: Either<V, Dist<V>>,
    pub fread_dist: Either<V, Dist<V>>,
    pub pi_dist: Either<V, Dist<V>>,
    pub pi_prob: V,
    pub trust_dists: TrustDists<V>,
    pub cpt_params: CptParams<V>,
}

#[derive(Clone, Default)]
pub struct ParamsForInfo<V: Float> {
    trust: V,
    shared: bool,
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
    so: StaticOpinions<V>,
    fop: FriendOpinions<V>,
    fso: FriendStaticOpinions<V>,
    read_prob: V,
    friend_arrival_prob: V,
    friend_read_prob: V,
    done_selfish: bool,
    params_for_info: Vec<ParamsForInfo<V>>,
}

impl<V> Agent<V>
where
    V: Float
        + UlpsEq
        + NumAssign
        + Sum
        + Default
        + PluralIgnorance
        + std::fmt::Debug
        + SampleUniform,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
{
    pub fn new(
        op: Opinions<V>,
        so: StaticOpinions<V>,
        fop: FriendOpinions<V>,
        fso: FriendStaticOpinions<V>,
        info_count: usize,
    ) -> Self {
        Self {
            cpt: Default::default(),
            prospect: Default::default(),
            op,
            so,
            fop,
            fso,
            read_prob: V::zero(),
            friend_arrival_prob: V::zero(),
            friend_read_prob: V::zero(),
            done_selfish: Default::default(),
            params_for_info: Vec::with_capacity(info_count),
        }
    }

    pub fn reset<F: FnMut(&Info<V>) -> V>(
        &mut self,
        read_prob: V,
        friend_arrival_prob: V,
        friend_read_prob: V,
        pi_rate: V,
        mut trust_map: F,
        base_rates: &GlobalBaseRates<V>,
        infos: &[Info<V>],
    ) {
        self.read_prob = read_prob;
        self.friend_arrival_prob = friend_arrival_prob;
        self.friend_read_prob = friend_read_prob;

        self.op.reset(base_rates);
        self.so.reset(pi_rate);
        self.fop.reset(base_rates);
        self.fso.reset(pi_rate);

        self.params_for_info.clear();
        for info in infos {
            self.params_for_info.push(ParamsForInfo {
                trust: trust_map(info),
                shared: false,
            });
        }
        self.done_selfish = false;
    }

    pub fn reset_with<R: Rng>(&mut self, constants: &Constants<V>, infos: &[Info<V>], rng: &mut R) {
        self.cpt.reset_with(&constants.cpt_params, rng);
        self.prospect.reset_with(&constants.cpt_params, rng);

        let r = rng.gen::<V>();
        let s = sample(&constants.pi_dist, rng);
        self.reset(
            sample(&constants.read_dist, rng),
            sample(&constants.farrival_dist, rng),
            sample(&constants.fread_dist, rng),
            if constants.pi_prob > r { s } else { V::zero() },
            constants.trust_dists.gen_map(rng),
            &constants.base_rates,
            infos,
        );
    }

    pub fn read_info_trustfully(
        &mut self,
        info: &Info<V>,
        receipt_prob: V,
        base_rates: &GlobalBaseRates<V>,
    ) -> Behavior {
        self.read_info_with_trust(info, receipt_prob, V::one(), base_rates)
    }

    pub fn read_info<R: Rng>(
        &mut self,
        rng: &mut R,
        info: &Info<V>,
        receipt_prob: V,
        base_rates: &GlobalBaseRates<V>,
    ) -> Option<Behavior> {
        if rng.gen::<V>() <= self.read_prob {
            Some(self.read_info_with_trust(
                info,
                receipt_prob,
                self.params_for_info[info.id].trust,
                base_rates,
            ))
        } else {
            None
        }
    }

    fn read_info_with_trust(
        &mut self,
        info: &Info<V>,
        receipt_prob: V,
        trust: V,
        base_rates: &GlobalBaseRates<V>,
    ) -> Behavior {
        let mut temp = compute_opinions(
            &mut self.op,
            &mut self.fop,
            &self.so,
            &self.fso,
            info,
            self.friend_read_prob,
            receipt_prob,
            trust,
            base_rates,
        );
        // compute values of prospects
        let shared = &mut self.params_for_info[info.id].shared;
        let sharing = if *shared {
            false
        } else {
            let (prob, pred_prob) = temp.predicate_friend_opinions(
                &self.op,
                &self.fop,
                &self.so,
                &self.fso,
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
