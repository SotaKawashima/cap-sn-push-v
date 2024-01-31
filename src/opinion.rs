use crate::info::Info;
use approx::UlpsEq;
use log::debug;
use num_traits::{Float, NumAssign};
use rand_distr::{Distribution, Open01};
use std::{array, iter::Sum, ops::AddAssign};
use subjective_logic::{
    harr2, harr3,
    mul::{
        op::{Deduction, Fuse, FuseAssign, FuseOp},
        prod::{HigherArr2, HigherArr3, Product2, Product3},
        Discount, Opinion, Opinion1d, Projection, Simplex,
    },
};

pub const THETA: usize = 3;
pub const PSI: usize = 2;
pub const PHI: usize = 2;
pub const A: usize = 2;
pub const S: usize = 2;
pub const P_THETA: usize = THETA;
pub const P_PSI: usize = PSI;
pub const P_A: usize = A;
pub const F_THETA: usize = THETA;
pub const F_PSI: usize = PSI;
pub const F_PHI: usize = PHI;
pub const F_A: usize = A;
pub const F_S: usize = S;
pub const FP_THETA: usize = P_THETA;
pub const FP_PSI: usize = P_PSI;
pub const FP_A: usize = P_A;

pub struct GlobalBaseRates<V> {
    pub psi: [V; PSI],
    pub ppsi: [V; P_PSI],
    pub s: [V; S],
    pub fs: [V; F_S],
    pub pa: [V; P_A],
    pub fa: [V; F_A],
    pub fpa: [V; FP_A],
    pub phi: [V; PHI],
    pub fpsi: [V; F_PSI],
    pub fppsi: [V; FP_PSI],
    pub fphi: [V; F_PHI],
    pub theta: [V; THETA],
    pub ptheta: [V; P_THETA],
    pub ftheta: [V; F_THETA],
    pub fptheta: [V; FP_THETA],
}

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

#[derive(Debug)]
pub struct FriendOpinionsUpd<V: Float> {
    pub fs: Opinion1d<V, F_S>,
    pub fphi: Opinion1d<V, F_PHI>,
    pub cond_ftheta_fphi: [Simplex<V, F_THETA>; F_PHI],
}

#[derive(Debug)]
pub struct FriendOpinions<V: Float> {
    pub fs: Opinion1d<V, F_S>,
    pub fphi: Opinion1d<V, F_PHI>,
    // F\Theta | F\Phi
    pub cond_ftheta_fphi: [Simplex<V, F_THETA>; F_PHI],
}

pub struct FriendStaticOpinions<V: Float> {
    // FP\Theta | FP\Psi
    pub cond_fptheta: [Simplex<V, FP_THETA>; FP_PSI],
    // FPA | FP\Theta
    pub cond_fpa: [Simplex<V, FP_A>; FP_THETA],
    // F\Theta | FPA. F\Psi
    pub cond_ftheta: HigherArr2<Simplex<V, F_THETA>, FP_A, F_PSI>,
    // FA | F\Theta
    pub cond_fa: [Simplex<V, F_A>; F_THETA],
    // FP\Psi | FS
    pub cond_fppsi: [Simplex<V, FP_PSI>; F_S],
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

impl<V> FriendOpinions<V>
where
    Open01: Distribution<V>,
    V: Float + UlpsEq + NumAssign + Sum + Default + std::fmt::Debug,
{
    pub fn reset(&mut self, base_rates: &GlobalBaseRates<V>) {
        reset_opinion(&mut self.fs, &base_rates.fs);
        reset_opinion(&mut self.fphi, &base_rates.fphi);
        reset_simplex(&mut self.cond_ftheta_fphi);
    }

    pub fn new() -> Self {
        Self {
            fs: Opinion::default(),
            fphi: Opinion::default(),
            cond_ftheta_fphi: [Simplex::default(), Simplex::default()],
        }
    }

    pub fn compute_new_friend_op(
        &self,
        fso: &FriendStaticOpinions<V>,
        info: &Info<V>,
        fpsi_ded: &Opinion1d<V, F_PSI>,
        ft: V,
        base_rates: &GlobalBaseRates<V>,
    ) -> (FriendOpinionsUpd<V>, Opinion1d<V, F_A>) {
        let fs = FuseOp::Wgh.fuse(&self.fs, &info.content.s.discount(ft));
        let fpsi = FuseOp::Wgh.fuse(fpsi_ded, &info.content.psi.discount(ft));
        let fphi = FuseOp::Wgh.fuse(&self.fphi, &info.content.phi.discount(ft));
        let cond_ftheta_fphi = array::from_fn(|i| {
            FuseOp::Wgh.fuse(
                &self.cond_ftheta_fphi[i],
                &info.content.cond_theta_phi[i].discount(ft),
            )
        });
        let fop = FriendOpinionsUpd {
            fs,
            fphi,
            cond_ftheta_fphi,
        };

        debug!(" P_FS  :{:?}", fop.fs.projection());

        let fpa = {
            let mut fpa = fop
                .fs
                .deduce(&fso.cond_fppsi)
                .unwrap_or_else(|| Opinion::vacuous_with(base_rates.fppsi))
                .deduce(&fso.cond_fptheta)
                .unwrap_or_else(|| Opinion::vacuous_with(base_rates.fptheta))
                .deduce(&fso.cond_fpa)
                .unwrap_or_else(|| Opinion::vacuous_with(base_rates.fpa));

            // compute a FPA opinion by using aleatory cummulative fusion.
            FuseOp::ACm.fuse_assign(&mut fpa, &info.content.pa.discount(ft));
            fpa
        };

        let ftheta = {
            let mut ftheta_ded_1 = Opinion::product2(&fpa, &fpsi)
                .deduce(&fso.cond_ftheta)
                .unwrap_or_else(|| Opinion::<_, V>::vacuous_with(base_rates.ftheta));
            let ftheta_ded_2 = fop
                .fphi
                .deduce(&fop.cond_ftheta_fphi)
                .unwrap_or_else(|| Opinion::<_, V>::vacuous_with(base_rates.ftheta));
            FuseOp::Wgh.fuse_assign(&mut ftheta_ded_1, &ftheta_ded_2);
            ftheta_ded_1
        };
        debug!(" P_FPA :{:?}", fpa.projection());
        debug!(" P_FTH :{:?}", ftheta.projection());
        (
            fop,
            ftheta
                .deduce(&fso.cond_fa)
                .unwrap_or_else(|| Opinion::vacuous_with(base_rates.fa)),
        )
    }

    pub fn update(&mut self, fop: FriendOpinionsUpd<V>) {
        self.fs = fop.fs;
        self.fphi = fop.fphi;
        self.cond_ftheta_fphi = fop.cond_ftheta_fphi;
    }
}

impl<V: PluralIgnorance> FriendStaticOpinions<V> {
    pub fn reset(&mut self, pi_rate: V) {
        self.cond_fpa[1] = pi_rate.plural_ignorance();
    }
}

macro_rules! impl_friend_opinion {
    ($ft: ty) => {
        #[allow(dead_code)]
        impl FriendStaticOpinions<$ft> {
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
pub struct Opinions<V> {
    pub theta: Opinion1d<V, THETA>,
    pub psi: Opinion1d<V, PSI>,
    pub phi: Opinion1d<V, PHI>,
    pub s: Opinion1d<V, S>,

    // \Theta | \Phi
    pub cond_theta_phi: [Simplex<V, THETA>; PHI],
}

pub struct StaticOpinions<V> {
    // plural ignorance conditions
    // P\Psi | S
    pub cond_ppsi: [Simplex<V, P_PSI>; S],
    // P\Theta | P\Psi
    pub cond_ptheta: [Simplex<V, P_THETA>; P_PSI],
    // PA | P\Theta
    pub cond_pa: [Simplex<V, P_A>; P_THETA],
    // F\Psi | S
    pub cond_fpsi: [Simplex<V, F_PSI>; S],
    // \Theta | PA.\Psi,FA
    pub cond_theta: HigherArr3<Simplex<V, THETA>, P_A, PSI, F_A>,
}

impl<V> Opinions<V>
where
    V: Float + Default,
    Open01: Distribution<V>,
{
    pub fn reset(&mut self, base_rates: &GlobalBaseRates<V>) {
        reset_opinion(&mut self.theta, &base_rates.theta);
        reset_opinion(&mut self.psi, &base_rates.psi);
        reset_opinion(&mut self.phi, &base_rates.phi);
        reset_opinion(&mut self.s, &base_rates.s);
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

impl<V: PluralIgnorance> StaticOpinions<V> {
    pub fn reset(&mut self, pi_rate: V) {
        self.cond_pa[1] = pi_rate.plural_ignorance();
    }
}

macro_rules! impl_agent_static_opinion {
    ($ft: ty) => {
        #[allow(dead_code)]
        impl StaticOpinions<$ft> {
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

pub struct Temporal<V: Float> {
    fa: Opinion1d<V, F_A>,
    theta: Opinion1d<V, THETA>,
    fpsi_ded: Opinion1d<V, F_PSI>,
    theta_ded_2: Opinion1d<V, THETA>,
    pa: Opinion1d<V, P_A>,
    pred: Option<PredTemporal<V>>,
}

pub fn compute_opinions<V>(
    op: &mut Opinions<V>,
    fop: &mut FriendOpinions<V>,
    so: &StaticOpinions<V>,
    fso: &FriendStaticOpinions<V>,
    info: &Info<V>,
    friend_read_prob: V,
    receipt_prob: V,
    trust: V,
    base_rates: &GlobalBaseRates<V>,
) -> Temporal<V>
where
    V: Float + UlpsEq + NumAssign + Sum + Default + PluralIgnorance + std::fmt::Debug,
    Open01: Distribution<V>,
{
    debug!("b_PA|PTH_1:{:?}", so.cond_pa[1].belief);

    FuseOp::Wgh.fuse_assign(&mut op.s, &info.content.s.discount(trust));
    FuseOp::Wgh.fuse_assign(&mut op.psi, &info.content.psi.discount(trust));
    FuseOp::Wgh.fuse_assign(&mut op.phi, &info.content.phi.discount(trust));
    for i in 0..info.content.cond_theta_phi.len() {
        FuseOp::Wgh.fuse_assign(
            &mut fop.cond_ftheta_fphi[i],
            &info.content.cond_theta_phi[i].discount(trust),
        )
    }

    let pa = {
        let mut pa =
            op.s.deduce(&so.cond_ppsi)
                .unwrap_or_else(|| Opinion::vacuous_with(base_rates.ppsi))
                .deduce(&so.cond_ptheta)
                .unwrap_or_else(|| Opinion::vacuous_with(base_rates.ptheta))
                .deduce(&so.cond_pa)
                .unwrap_or_else(|| Opinion::vacuous_with(base_rates.pa));

        // an opinion of PA is computed by using aleatory cummulative fusion.
        FuseOp::ACm.fuse_assign(&mut pa, &info.content.pa.discount(trust));
        pa
    };

    debug!(" P_PA  :{:?}", pa.projection());

    // compute friends opinions
    let fpsi_ded =
        op.s.deduce(&so.cond_fpsi)
            .unwrap_or_else(|| Opinion::vacuous_with(base_rates.fpsi));
    let (fop_upd, fa) = fop.compute_new_friend_op(
        &fso,
        info,
        &fpsi_ded,
        receipt_prob * friend_read_prob * trust,
        base_rates,
    );
    fop.update(fop_upd);

    let theta_ded_2 = op
        .phi
        .deduce(&op.cond_theta_phi)
        .unwrap_or_else(|| Opinion::vacuous_with(op.theta.base_rate));

    let theta = {
        let mut theta_ded_1 = Opinion::product3(pa.as_ref(), op.psi.as_ref(), fa.as_ref())
            .deduce(&so.cond_theta)
            .unwrap_or_else(|| Opinion::vacuous_with(op.theta.base_rate));
        FuseOp::Wgh.fuse_assign(&mut theta_ded_1, &theta_ded_2);
        theta_ded_1
    };

    Temporal {
        fa,
        theta,
        fpsi_ded,
        theta_ded_2,
        pa,
        pred: None,
    }
}

struct PredTemporal<V: Float> {
    pred_theta: Opinion1d<V, THETA>,
    pred_fop: FriendOpinionsUpd<V>,
}

impl<V> Temporal<V>
where
    V: Float + UlpsEq + NumAssign + Sum + Default + PluralIgnorance + std::fmt::Debug,
    Open01: Distribution<V>,
{
    pub fn predicate_friend_opinions(
        &mut self,
        op: &Opinions<V>,
        fop: &FriendOpinions<V>,
        so: &StaticOpinions<V>,
        fso: &FriendStaticOpinions<V>,
        info: &Info<V>,
        ft: V,
        base_rates: &GlobalBaseRates<V>,
    ) -> (HigherArr2<V, F_A, THETA>, HigherArr2<V, F_A, THETA>)
    where
        V: Float + UlpsEq + NumAssign + Sum + Default + PluralIgnorance + std::fmt::Debug,
        Open01: Distribution<V>,
    {
        // predict new friends' opinions in case of sharing
        let (pred_fop, pred_fa) =
            fop.compute_new_friend_op(&fso, info, &self.fpsi_ded, ft, base_rates);
        let pred_theta = {
            let mut pred_theta_ded_1 =
                Opinion::product3(self.pa.as_ref(), op.psi.as_ref(), pred_fa.as_ref())
                    .deduce(&so.cond_theta)
                    .unwrap_or_else(|| Opinion::vacuous_with(op.theta.base_rate));
            FuseOp::Wgh.fuse_assign(&mut pred_theta_ded_1, &self.theta_ded_2);
            pred_theta_ded_1
        };

        let fa_theta = Opinion::product2(self.fa.as_ref(), self.theta.as_ref());
        let pred_fa_theta = Opinion::product2(pred_fa.as_ref(), pred_theta.as_ref());
        self.pred = Some(PredTemporal {
            pred_theta,
            pred_fop,
        });
        (fa_theta.projection(), pred_fa_theta.projection())
    }

    pub fn update_for_sharing(
        self,
        op: &mut Opinions<V>,
        fop: &mut FriendOpinions<V>,
    ) -> [V; THETA] {
        match self.pred {
            // sharing info
            Some(PredTemporal {
                pred_theta,
                pred_fop,
            }) => {
                op.theta = pred_theta;
                fop.update(pred_fop);
            }
            None => {
                op.theta = self.theta;
            }
        }
        op.theta.projection()
    }
}
