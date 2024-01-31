use approx::UlpsEq;
use log::debug;
use num_traits::{Float, NumAssign};
use rand::Rng;
use rand_distr::{Beta, Distribution, Open01, Standard};
use std::{array, iter::Sum};

use crate::{
    cpt::{LevelSet, CPT},
    info::{Info, TrustDists},
    opinion::{
        compute_opinions, FriendOpinions, FriendStaticOpinions, GlobalBaseRates, Opinions,
        PluralIgnorance, StaticOpinions,
    },
};

pub struct Constants<V: Float>
where
    Open01: Distribution<V>,
{
    pub base_rates: GlobalBaseRates<V>,
    pub read_dist: Beta<V>,
    pub fclose_dist: Beta<V>,
    pub fread_dist: Beta<V>,
    pub pi_prob: V,
    pub pi_dist: Beta<V>,
    pub trust_dists: TrustDists<V>,
}

#[derive(Debug)]
pub struct Behavior {
    pub selfish: bool,
    pub sharing: bool,
}

pub struct Prospect<V: Float> {
    selfish: [LevelSet<usize, V>; 2],
    sharing: [LevelSet<[usize; 2], V>; 2],
}

#[derive(Clone, Default)]
pub struct ParamsForInfo<V: Float> {
    trust: V,
    shared: bool,
}

pub struct Agent<V: Float> {
    cpt: CPT<V>,
    prospect: Prospect<V>,
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
    V: Float + UlpsEq + NumAssign + Sum + Default + PluralIgnorance + std::fmt::Debug,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
{
    pub fn new(
        op: Opinions<V>,
        so: StaticOpinions<V>,
        fop: FriendOpinions<V>,
        fso: FriendStaticOpinions<V>,
        cpt: CPT<V>,
        selfish: [LevelSet<usize, V>; 2],
        sharing: [LevelSet<[usize; 2], V>; 2],
        info_count: usize,
    ) -> Self {
        Self {
            cpt,
            prospect: Prospect { selfish, sharing },
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
        let r = rng.gen::<V>();
        let s = constants.pi_dist.sample(rng);

        self.reset(
            constants.read_dist.sample(rng),
            constants.fclose_dist.sample(rng),
            constants.fread_dist.sample(rng),
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
        // debug!("b_PA|PTH_1:{:?}", self.so.cond_pa[1].belief);

        // FuseOp::Wgh.fuse_assign(&mut self.op.s, &info.content.s.discount(trust));
        // FuseOp::Wgh.fuse_assign(&mut self.op.psi, &info.content.psi.discount(trust));
        // FuseOp::Wgh.fuse_assign(&mut self.op.phi, &info.content.phi.discount(trust));
        // for i in 0..info.content.cond_theta_phi.len() {
        //     FuseOp::Wgh.fuse_assign(
        //         &mut self.fop.cond_ftheta_fphi[i],
        //         &info.content.cond_theta_phi[i].discount(trust),
        //     )
        // }

        // let pa = {
        //     let mut pa = self
        //         .op
        //         .s
        //         .deduce(&self.so.cond_ppsi)
        //         .unwrap_or_else(|| Opinion::vacuous_with(base_rates.ppsi))
        //         .deduce(&self.so.cond_ptheta)
        //         .unwrap_or_else(|| Opinion::vacuous_with(base_rates.ptheta))
        //         .deduce(&self.so.cond_pa)
        //         .unwrap_or_else(|| Opinion::vacuous_with(base_rates.pa));

        //     // an opinion of PA is computed by using aleatory cummulative fusion.
        //     FuseOp::ACm.fuse_assign(&mut pa, &info.content.pa.discount(trust));
        //     pa
        // };

        // debug!(" P_PA  :{:?}", pa.projection());

        // // compute friends opinions
        // let fpsi_ded = self
        //     .op
        //     .s
        //     .deduce(&self.so.cond_fpsi)
        //     .unwrap_or_else(|| Opinion::vacuous_with(base_rates.fpsi));
        // let (fop, fa) = self.fop.compute_new_friend_op(
        //     &self.fso,
        //     info,
        //     &fpsi_ded,
        //     receipt_prob * self.friend_read_prob * trust,
        //     base_rates,
        // );
        // self.fop.update(fop);

        // let theta_ded_2 = self
        //     .op
        //     .phi
        //     .deduce(&self.op.cond_theta_phi)
        //     .unwrap_or_else(|| Opinion::vacuous_with(self.op.theta.base_rate));

        // let theta = {
        //     let mut theta_ded_1 = Opinion::product3(pa.as_ref(), self.op.psi.as_ref(), fa.as_ref())
        //         .deduce(&self.so.cond_theta)
        //         .unwrap_or_else(|| Opinion::vacuous_with(self.op.theta.base_rate));
        //     FuseOp::Wgh.fuse_assign(&mut theta_ded_1, &theta_ded_2);
        //     theta_ded_1
        // };

        // // predict new friends' opinions in case of sharing
        // let (pred_fop, pred_fa) = self.fop.compute_new_friend_op(
        //     &self.fso,
        //     info,
        //     &fpsi_ded,
        //     self.friend_arrival_prob * self.friend_read_prob * trust,
        //     base_rates,
        // );
        // let pred_theta = {
        //     let mut pred_theta_ded_1 =
        //         Opinion::product3(pa.as_ref(), self.op.psi.as_ref(), pred_fa.as_ref())
        //             .deduce(&self.so.cond_theta)
        //             .unwrap_or_else(|| Opinion::vacuous_with(self.op.theta.base_rate));
        //     FuseOp::Wgh.fuse_assign(&mut pred_theta_ded_1, &theta_ded_2);
        //     pred_theta_ded_1
        // };

        // debug!(" P_FA  :{:?}", fa.projection());
        // debug!("~P_FA  :{:?}", pred_fa.projection());
        // debug!(" P_TH  :{:?}", theta.projection());
        // debug!("~P_TH  :{:?}", pred_theta.projection());

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
