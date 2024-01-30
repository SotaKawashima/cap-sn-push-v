use approx::UlpsEq;
use log::debug;
use num_traits::{Float, NumAssign};
use rand::Rng;
use rand_distr::{Beta, Distribution, Open01, Standard};
use std::{array, iter::Sum, ops::AddAssign};
use subjective_logic::{
    harr2, harr3,
    mul::{
        op::{Deduction, Fuse, FuseAssign, FuseOp},
        prod::{HigherArr2, HigherArr3, Product2, Product3},
        Discount, Opinion, Opinion1d, Projection, Simplex,
    },
};

use crate::{
    cpt::{LevelSet, CPT},
    info::{Info, InfoType},
    opinion::{
        A, FP_A, FP_PSI, FP_THETA, F_A, F_PHI, F_PSI, F_S, F_THETA, PHI, PSI, P_A, P_PSI, P_THETA,
        S, THETA,
    },
};

fn reset_opinion<V: Float, const N: usize>(w: &mut Opinion1d<V, N>, base_rate: &[V; N]) {
    w.simplex.belief = [V::zero(); N];
    w.simplex.uncertainty = V::one();
    w.base_rate = *base_rate;
}

fn reset_simplex<V: Float, const N: usize, const M: usize>(cond: &mut [Simplex<V, N>; M]) {
    for i in 0..M {
        cond[i].belief = [V::zero(); N];
        cond[i].uncertainty = V::one();
    }
}

pub struct Constants<V: Float>
where
    Open01: Distribution<V>,
{
    pub br_psi: [V; PSI],
    pub br_ppsi: [V; PSI],
    pub br_s: [V; S],
    pub br_fs: [V; S],
    pub br_pa: [V; A],
    pub br_fa: [V; A],
    pub br_fpa: [V; A],
    pub br_phi: [V; PHI],
    pub br_fpsi: [V; PSI],
    pub br_fppsi: [V; PSI],
    pub br_fphi: [V; PHI],
    pub br_theta: [V; THETA],
    pub br_ptheta: [V; THETA],
    pub br_ftheta: [V; THETA],
    pub br_fptheta: [V; THETA],
    pub read_dist: Beta<V>,
    pub fclose_dist: Beta<V>,
    pub fread_dist: Beta<V>,
    pub pi_prob: V,
    pub pi_dist: Beta<V>,
    pub misinfo_trust_dist: Beta<V>,
    pub correction_trust_dist: Beta<V>,
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
    op: AgentOpinion<V>,
    so: AgentStaticOpinion<V>,
    fop: FriendOpinion<V>,
    fso: FriendStaticOpinion<V>,
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
        op: AgentOpinion<V>,
        so: AgentStaticOpinion<V>,
        fop: FriendOpinion<V>,
        fso: FriendStaticOpinion<V>,
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

    pub fn reset<F: FnMut(&InfoType) -> V>(
        &mut self,
        read_prob: V,
        friend_arrival_prob: V,
        friend_read_prob: V,
        pi_rate: V,
        mut trust_map: F,
        constants: &Constants<V>,
        info_types: &[InfoType],
    ) {
        self.read_prob = read_prob;
        self.friend_arrival_prob = friend_arrival_prob;
        self.friend_read_prob = friend_read_prob;

        self.op.reset(
            &constants.br_theta,
            &constants.br_psi,
            &constants.br_phi,
            &constants.br_s,
        );
        self.so.reset(pi_rate);
        self.fop.reset(&constants.br_fs, &constants.br_fphi);
        self.fso.reset(pi_rate);

        self.params_for_info.clear();
        for i in 0..self.params_for_info.capacity() {
            self.params_for_info.push(ParamsForInfo {
                trust: trust_map(&info_types[i]),
                shared: false,
            });
        }
        self.done_selfish = false;
    }

    pub fn reset_with<R: Rng>(
        &mut self,
        constants: &Constants<V>,
        info_types: &[InfoType],
        rng: &mut R,
    ) {
        let r = rng.gen::<V>();
        let s = constants.pi_dist.sample(rng);

        self.reset(
            constants.read_dist.sample(rng),
            constants.fclose_dist.sample(rng),
            constants.fread_dist.sample(rng),
            if constants.pi_prob > r { s } else { V::zero() },
            |_| constants.misinfo_trust_dist.sample(rng),
            constants,
            info_types,
        );
    }

    pub fn read_info_trustfully(
        &mut self,
        info: &Info<V>,
        receipt_prob: V,
        constants: &Constants<V>,
    ) -> Behavior {
        self.read_info_with_trust(info, receipt_prob, V::one(), constants)
    }

    pub fn read_info<R: Rng>(
        &mut self,
        rng: &mut R,
        info: &Info<V>,
        receipt_prob: V,
        constants: &Constants<V>,
    ) -> Option<Behavior> {
        if rng.gen::<V>() <= self.read_prob {
            Some(self.read_info_with_trust(
                info,
                receipt_prob,
                self.params_for_info[info.id].trust,
                constants,
            ))
        } else {
            None
        }
    }

    fn read_info_with_trust(
        &mut self,
        info: &Info<V>,
        receipt_prob: V,
        t: V,
        constants: &Constants<V>,
    ) -> Behavior {
        debug!("b_PA|PTH_1:{:?}", self.so.cond_pa[1].belief);

        FuseOp::Wgh.fuse_assign(&mut self.op.s, &info.content.s.discount(t));
        FuseOp::Wgh.fuse_assign(&mut self.op.psi, &info.content.psi.discount(t));
        FuseOp::Wgh.fuse_assign(&mut self.op.phi, &info.content.phi.discount(t));
        for i in 0..PHI {
            FuseOp::Wgh.fuse_assign(
                &mut self.fop.cond_ftheta_fphi[i],
                &info.content.cond_theta_phi[i].discount(t),
            )
        }

        let pa = {
            let mut pa = self
                .op
                .s
                .deduce(&self.so.cond_ppsi)
                .unwrap_or_else(|| Opinion::<_, V>::vacuous_with(constants.br_ppsi))
                .deduce(&self.so.cond_ptheta)
                .unwrap_or_else(|| Opinion::<_, V>::vacuous_with(constants.br_ptheta))
                .deduce(&self.so.cond_pa)
                .unwrap_or_else(|| Opinion::<_, V>::vacuous_with(constants.br_pa));

            // an opinion of PA is computed by using aleatory cummulative fusion.
            FuseOp::ACm.fuse_assign(&mut pa, &info.content.pa.discount(t));
            pa
        };

        debug!(" P_PA  :{:?}", pa.projection());

        // compute friends opinions
        let fpsi_ded = self
            .op
            .s
            .deduce(&self.so.cond_fpsi)
            .unwrap_or_else(|| Opinion::<_, V>::vacuous_with(constants.br_fpsi));
        let (fop, fa) = self.fop.compute_new_friend_op(
            &self.fso,
            info,
            &fpsi_ded,
            receipt_prob * self.friend_read_prob * t,
            constants,
        );
        self.fop.update(fop);

        let theta_ded_2 = self
            .op
            .phi
            .deduce(&self.op.cond_theta_phi)
            .unwrap_or_else(|| Opinion::<_, V>::vacuous_with(self.op.theta.base_rate));

        let theta = {
            let mut theta_ded_1 = Opinion::product3(pa.as_ref(), self.op.psi.as_ref(), fa.as_ref())
                .deduce(&self.so.cond_theta)
                .unwrap_or_else(|| Opinion::<_, V>::vacuous_with(self.op.theta.base_rate));
            FuseOp::Wgh.fuse_assign(&mut theta_ded_1, &theta_ded_2);
            theta_ded_1
        };

        // predict new friends' opinions in case of sharing
        let (pred_fop, pred_fa) = self.fop.compute_new_friend_op(
            &self.fso,
            info,
            &fpsi_ded,
            self.friend_arrival_prob * self.friend_read_prob * t,
            constants,
        );
        let pred_theta = {
            let mut pred_theta_ded_1 =
                Opinion::product3(pa.as_ref(), self.op.psi.as_ref(), pred_fa.as_ref())
                    .deduce(&self.so.cond_theta)
                    .unwrap_or_else(|| Opinion::<_, V>::vacuous_with(self.op.theta.base_rate));
            FuseOp::Wgh.fuse_assign(&mut pred_theta_ded_1, &theta_ded_2);
            pred_theta_ded_1
        };

        debug!(" P_FA  :{:?}", fa.projection());
        debug!("~P_FA  :{:?}", pred_fa.projection());
        debug!(" P_TH  :{:?}", theta.projection());
        debug!("~P_TH  :{:?}", pred_theta.projection());

        // compute values of prospects
        let fa_theta = Opinion::product2(fa.as_ref(), theta.as_ref());
        let pred_fa_theta = Opinion::product2(pred_fa.as_ref(), pred_theta.as_ref());

        let value_sharing: [V; 2] = [
            self.cpt
                .valuate(&self.prospect.sharing[0], &fa_theta.projection()),
            self.cpt
                .valuate(&self.prospect.sharing[1], &pred_fa_theta.projection()),
        ];

        let sharing = &mut self.params_for_info[info.id].shared;
        let prev_sharing = *sharing;
        let prev_done_selfish = self.done_selfish;
        if !prev_sharing && value_sharing[0] < value_sharing[1] {
            // sharing info
            self.op.theta = pred_theta;
            self.fop.update(pred_fop);
            *sharing = true;
        } else {
            self.op.theta = theta;
        }
        let value_selfish: [V; 2] = array::from_fn(|i| {
            self.cpt
                .valuate(&self.prospect.selfish[i], &self.op.theta.projection())
        });

        debug!("V_X,V_Y:{:?},{:?}", value_selfish, value_sharing);

        if !prev_done_selfish && value_selfish[0] < value_selfish[1] {
            // do selfish action
            self.done_selfish = true;
        }

        Behavior {
            selfish: !prev_done_selfish && self.done_selfish,
            sharing: !prev_sharing && *sharing,
        }
    }
}

#[derive(Debug)]
struct FriendOpinionUpd<V: Float> {
    fs: Opinion1d<V, F_S>,
    fphi: Opinion1d<V, F_PHI>,
    cond_ftheta_fphi: [Simplex<V, F_THETA>; F_PHI],
}

#[derive(Debug)]
pub struct FriendOpinion<V: Float> {
    fs: Opinion1d<V, F_S>,
    fphi: Opinion1d<V, F_PHI>,
    // F\Theta | F\Phi
    cond_ftheta_fphi: [Simplex<V, F_THETA>; F_PHI],
}

pub struct FriendStaticOpinion<V: Float> {
    // FP\Theta | FP\Psi
    cond_fptheta: [Simplex<V, FP_THETA>; FP_PSI],
    // FPA | FP\Theta
    cond_fpa: [Simplex<V, FP_A>; FP_THETA],
    // F\Theta | FPA. F\Psi
    cond_ftheta: HigherArr2<Simplex<V, F_THETA>, FP_A, F_PSI>,
    // FA | F\Theta
    cond_fa: [Simplex<V, F_A>; F_THETA],
    // FP\Psi | FS
    cond_fppsi: [Simplex<V, FP_PSI>; F_S],
}

pub trait PluralIgnorance
where
    Self: Float + AddAssign + UlpsEq,
{
    const U: Self;
    fn plural_ignorance(self) -> Simplex<Self, A> {
        let b1 = self * (Self::one() - Self::U);
        Simplex::new([Self::one() - Self::U - b1, b1], Self::U)
    }
}

macro_rules! impl_plural_ignorance {
    ($ft: ty) => {
        impl PluralIgnorance for $ft {
            const U: $ft = 0.01;
        }
    };
}

impl_plural_ignorance!(f32);
impl_plural_ignorance!(f64);

impl<V> FriendOpinion<V>
where
    Open01: Distribution<V>,
    V: Float + UlpsEq + NumAssign + Sum + Default + std::fmt::Debug,
{
    pub fn reset(&mut self, br_fs: &[V; F_S], br_fphi: &[V; F_PHI]) {
        reset_opinion(&mut self.fs, br_fs);
        reset_opinion(&mut self.fphi, br_fphi);
        reset_simplex(&mut self.cond_ftheta_fphi);
    }

    pub fn new() -> Self {
        Self {
            fs: Opinion::default(),
            fphi: Opinion::default(),
            cond_ftheta_fphi: [Simplex::default(), Simplex::default()],
        }
    }

    fn compute_new_friend_op(
        &self,
        fso: &FriendStaticOpinion<V>,
        info: &Info<V>,
        fpsi_ded: &Opinion1d<V, PSI>,
        ft: V,
        constants: &Constants<V>,
    ) -> (FriendOpinionUpd<V>, Opinion1d<V, F_A>) {
        let fs = FuseOp::Wgh.fuse(&self.fs, &info.content.s.discount(ft));
        let fpsi = FuseOp::Wgh.fuse(fpsi_ded, &info.content.psi.discount(ft));
        let fphi = FuseOp::Wgh.fuse(&self.fphi, &info.content.phi.discount(ft));
        let cond_ftheta_fphi = array::from_fn(|i| {
            FuseOp::Wgh.fuse(
                &self.cond_ftheta_fphi[i],
                &info.content.cond_theta_phi[i].discount(ft),
            )
        });
        let fop = FriendOpinionUpd {
            fs,
            fphi,
            cond_ftheta_fphi,
        };

        debug!(" P_FS  :{:?}", fop.fs.projection());

        let fpa = {
            let mut fpa = fop
                .fs
                .deduce(&fso.cond_fppsi)
                .unwrap_or_else(|| Opinion::<_, V>::vacuous_with(constants.br_fppsi))
                .deduce(&fso.cond_fptheta)
                .unwrap_or_else(|| Opinion::<_, V>::vacuous_with(constants.br_fptheta))
                .deduce(&fso.cond_fpa)
                .unwrap_or_else(|| Opinion::<_, V>::vacuous_with(constants.br_fpa));

            // compute a FPA opinion by using aleatory cummulative fusion.
            FuseOp::ACm.fuse_assign(&mut fpa, &info.content.pa.discount(ft));
            fpa
        };

        let ftheta = {
            let mut ftheta_ded_1 = Opinion::product2(&fpa, &fpsi)
                .deduce(&fso.cond_ftheta)
                .unwrap_or_else(|| Opinion::<_, V>::vacuous_with(constants.br_ftheta));
            let ftheta_ded_2 = fop
                .fphi
                .deduce(&fop.cond_ftheta_fphi)
                .unwrap_or_else(|| Opinion::<_, V>::vacuous_with(constants.br_ftheta));
            FuseOp::Wgh.fuse_assign(&mut ftheta_ded_1, &ftheta_ded_2);
            ftheta_ded_1
        };
        debug!(" P_FPA :{:?}", fpa.projection());
        debug!(" P_FTH :{:?}", ftheta.projection());
        (
            fop,
            ftheta
                .deduce(&fso.cond_fa)
                .unwrap_or_else(|| Opinion::<_, V>::vacuous_with(constants.br_fa)),
        )
    }

    fn update(&mut self, fop: FriendOpinionUpd<V>) {
        self.fs = fop.fs;
        self.fphi = fop.fphi;
        self.cond_ftheta_fphi = fop.cond_ftheta_fphi;
    }
}

impl<V: PluralIgnorance> FriendStaticOpinion<V> {
    fn reset(&mut self, pi_rate: V) {
        self.cond_fpa[1] = pi_rate.plural_ignorance();
    }
}

macro_rules! impl_friend_opinion {
    ($ft: ty) => {
        #[allow(dead_code)]
        impl FriendStaticOpinion<$ft> {
            pub fn new() -> Self {
                Self {
                    cond_fa: [
                        Simplex::<$ft, F_A>::new([0.95, 0.00], 0.05),
                        Simplex::<$ft, F_A>::new([0.00, 0.95], 0.05),
                        Simplex::<$ft, F_A>::new([0.95, 0.00], 0.05),
                    ],
                    cond_fpa: [
                        Simplex::<$ft, F_A>::new([0.90, 0.00], 0.10),
                        Simplex::default(),
                        Simplex::<$ft, F_A>::new([0.90, 0.00], 0.10),
                    ],
                    cond_ftheta: harr2![
                        [
                            Simplex::<$ft, F_THETA>::new([0.95, 0.00, 0.00], 0.05),
                            Simplex::<$ft, F_THETA>::new([0.00, 0.45, 0.45], 0.10),
                        ],
                        [
                            Simplex::<$ft, F_THETA>::new([0.00, 0.475, 0.475], 0.05),
                            Simplex::<$ft, F_THETA>::new([0.00, 0.495, 0.495], 0.01),
                        ]
                    ],
                    cond_fptheta: [
                        Simplex::<$ft, FP_THETA>::new([0.99, 0.000, 0.000], 0.01),
                        Simplex::<$ft, FP_THETA>::new([0.00, 0.495, 0.495], 0.01),
                    ],
                    cond_fppsi: [
                        Simplex::<$ft, FP_PSI>::new([0.99, 0.00], 0.01),
                        Simplex::<$ft, FP_PSI>::new([0.25, 0.65], 0.10),
                    ],
                }
            }
        }
    };
}

impl_friend_opinion!(f32);
impl_friend_opinion!(f64);

#[derive(Debug)]
pub struct AgentOpinion<V> {
    theta: Opinion1d<V, THETA>,
    psi: Opinion1d<V, PSI>,
    phi: Opinion1d<V, PHI>,
    s: Opinion1d<V, S>,

    // \Theta | \Phi
    cond_theta_phi: [Simplex<V, THETA>; PHI],
}

pub struct AgentStaticOpinion<V> {
    // plural ignorance conditions
    // P\Psi | S
    cond_ppsi: [Simplex<V, P_PSI>; S],
    // P\Theta | P\Psi
    cond_ptheta: [Simplex<V, P_THETA>; P_PSI],
    // PA | P\Theta
    cond_pa: [Simplex<V, P_A>; P_THETA],
    // F\Psi | S
    cond_fpsi: [Simplex<V, F_PSI>; S],
    // \Theta | PA.\Psi,FA
    cond_theta: HigherArr3<Simplex<V, THETA>, P_A, PSI, F_A>,
}

impl<V> AgentOpinion<V>
where
    V: Float + Default,
    Open01: Distribution<V>,
{
    pub fn reset(
        &mut self,
        br_theta: &[V; THETA],
        br_psi: &[V; PSI],
        br_phi: &[V; PSI],
        br_s: &[V; S],
    ) {
        reset_opinion(&mut self.theta, br_theta);
        reset_opinion(&mut self.psi, br_psi);
        reset_opinion(&mut self.phi, br_phi);
        reset_opinion(&mut self.s, br_s);
        reset_simplex(&mut self.cond_theta_phi);
    }

    pub fn new() -> Self {
        Self {
            theta: Opinion::default(),
            psi: Opinion::default(),
            phi: Opinion::default(),
            s: Opinion::default(),
            cond_theta_phi: [Simplex::default(), Simplex::default()],
        }
    }
}

impl<V: PluralIgnorance> AgentStaticOpinion<V> {
    fn reset(&mut self, pi_rate: V) {
        self.cond_pa[1] = pi_rate.plural_ignorance();
    }
}

macro_rules! impl_agent_static_opinion {
    ($ft: ty) => {
        #[allow(dead_code)]
        impl AgentStaticOpinion<$ft> {
            pub fn new() -> Self {
                Self {
                    cond_pa: [
                        Simplex::<$ft, P_A>::new([0.90, 0.00], 0.10),
                        Simplex::default(),
                        Simplex::<$ft, P_A>::new([0.90, 0.00], 0.10),
                    ],
                    cond_theta: harr3![
                        [
                            [
                                Simplex::<$ft, THETA>::new([0.95, 0.00, 0.00], 0.05),
                                Simplex::<$ft, THETA>::new([0.95, 0.00, 0.00], 0.05),
                            ],
                            [
                                Simplex::<$ft, THETA>::new([0.00, 0.45, 0.45], 0.10),
                                Simplex::<$ft, THETA>::new([0.00, 0.45, 0.45], 0.10),
                            ],
                        ],
                        [
                            [
                                Simplex::<$ft, THETA>::new([0.00, 0.475, 0.475], 0.05),
                                Simplex::<$ft, THETA>::new([0.00, 0.475, 0.475], 0.05),
                            ],
                            [
                                Simplex::<$ft, THETA>::new([0.00, 0.495, 0.495], 0.01),
                                Simplex::<$ft, THETA>::new([0.00, 0.495, 0.495], 0.01),
                            ],
                        ]
                    ],
                    cond_ptheta: [
                        Simplex::<$ft, P_THETA>::new([0.99, 0.00, 0.00], 0.01),
                        Simplex::<$ft, P_THETA>::new([0.00, 0.495, 0.495], 0.01),
                    ],
                    cond_ppsi: [
                        Simplex::<$ft, P_PSI>::new([0.99, 0.00], 0.01),
                        Simplex::<$ft, P_PSI>::new([0.25, 0.65], 0.10),
                    ],
                    cond_fpsi: [
                        Simplex::<$ft, F_PSI>::new([0.99, 0.00], 0.01),
                        Simplex::<$ft, F_PSI>::new([0.70, 0.20], 0.10),
                    ],
                }
            }
        }
    };
}

impl_agent_static_opinion!(f32);
impl_agent_static_opinion!(f64);
