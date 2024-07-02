use std::{fmt::Debug, mem};

use rand::Rng;
use rand_distr::{Distribution, Exp1, Open01, Standard, StandardNormal};
use serde_with::{serde_as, TryFromInto};
use subjective_logic::{
    domain::{Domain, DomainConv, Keys},
    iter::FromFn,
    mul::{
        labeled::{OpinionD1, OpinionD2, OpinionD3, OpinionRefD1, SimplexD1},
        InverseCondition, MergeJointConditions2,
    },
    multi_array::labeled::{MArrD1, MArrD2, MArrD3},
    ops::{Deduction, Discount, Fuse, FuseAssign, FuseOp, Product2, Product3, Projection},
};
use tracing::debug;

use crate::info::gen2::InfoContent;
use crate::value::{EValue, EValueParam};

use super::{
    paramter::{ConditionParams, DependentParam, SimplexDist, SimplexParam},
    FPhi, FPsi, KPhi, KPsi, MyFloat, Phi, Psi, Theta, Thetad, A, B, FH, FO, H, KH, KO, O,
};

#[serde_as]
#[derive(Debug, serde::Deserialize, Clone)]
#[serde(bound(deserialize = "V: serde::Deserialize<'de>"))]
struct InitialBaseRates<V> {
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    a: MArrD1<A, V>,
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    b: MArrD1<B, V>,
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    h: MArrD1<H, V>,
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    fh: MArrD1<FH, V>,
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    kh: MArrD1<KH, V>,
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    theta: MArrD1<Theta, V>,
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    thetad: MArrD1<Thetad, V>,
}

#[serde_as]
#[derive(Debug, serde::Deserialize, Clone)]
#[serde(bound(deserialize = "V: serde::Deserialize<'de>"))]
struct InitialState<V: MyFloat> {
    #[serde_as(as = "TryFromInto<(Vec<V>, V, Vec<V>)>")]
    psi: OpinionD1<Psi, V>,
    #[serde_as(as = "TryFromInto<(Vec<V>, V, Vec<V>)>")]
    phi: OpinionD1<Phi, V>,
    #[serde_as(as = "TryFromInto<(Vec<V>, V, Vec<V>)>")]
    o: OpinionD1<O, V>,
    #[serde_as(as = "TryFromInto<(Vec<V>, V, Vec<V>)>")]
    fo: OpinionD1<FO, V>,
    #[serde_as(as = "TryFromInto<(Vec<V>, V, Vec<V>)>")]
    ko: OpinionD1<KO, V>,
    #[serde_as(as = "TryFromInto<(Vec<V>, V, Vec<V>)>")]
    fpsi: OpinionD1<FPsi, V>,
    #[serde_as(as = "TryFromInto<(Vec<V>, V, Vec<V>)>")]
    fphi: OpinionD1<FPhi, V>,
    #[serde_as(as = "TryFromInto<(Vec<V>, V, Vec<V>)>")]
    kpsi: OpinionD1<KPsi, V>,
    #[serde_as(as = "TryFromInto<(Vec<V>, V, Vec<V>)>")]
    kphi: OpinionD1<KPhi, V>,
    #[serde_as(as = "TryFromInto<Vec<(Vec<V>, V)>>")]
    h_psi_if_phi1: MArrD1<Psi, SimplexD1<H, V>>,
    #[serde_as(as = "TryFromInto<Vec<(Vec<V>, V)>>")]
    h_b_if_phi1: MArrD1<B, SimplexD1<H, V>>,
    #[serde_as(as = "TryFromInto<Vec<(Vec<V>, V)>>")]
    fh_fpsi_if_fphi1: MArrD1<FPsi, SimplexD1<FH, V>>,
    #[serde_as(as = "TryFromInto<Vec<(Vec<V>, V)>>")]
    kh_kpsi_if_kphi1: MArrD1<KPsi, SimplexD1<KH, V>>,
}

#[serde_as]
#[derive(Debug, serde::Deserialize)]
#[serde(bound(deserialize = "V: serde::Deserialize<'de>"))]
struct InitialFixed<V>
where
    V: MyFloat,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    Open01: Distribution<V>,
{
    /// $H^F \implies A$
    #[serde_as(as = "TryFromInto<Vec<SimplexParam<V>>>")]
    a_fh: MArrD1<FH, SimplexDist<A, V>>,
    /// $H^K \implies B$
    #[serde_as(as = "TryFromInto<Vec<SimplexParam<V>>>")]
    b_kh: MArrD1<KH, SimplexDist<B, V>>,
    /// $B \implies O$
    #[serde_as(as = "TryFromInto<Vec<SimplexParam<V>>>")]
    o_b: MArrD1<B, SimplexDist<O, V>>,
    /// $H \implies \Theta$
    #[serde_as(as = "TryFromInto<Vec<SimplexParam<V>>>")]
    theta_h: MArrD1<H, SimplexDist<Theta, V>>,
    /// $H \implies \Theta'$
    #[serde_as(as = "TryFromInto<Vec<SimplexParam<V>>>")]
    thetad_h: MArrD1<H, SimplexDist<Thetad, V>>,
    /// parameters of conditional opinions $\Psi \implies H$ and $B \implies H$ when $\phi_0$ is true
    h_psi_b_if_phi0: ConditionParams<Psi, B, H, V>,
    /// parameters of conditional opinions $\Psi^F \implies H^F$ when $\phi^F_0$ is true
    #[serde_as(as = "TryFromInto<DependentParam<FH, V, Vec<SimplexParam<V>>>>")]
    fh_fpsi_if_fphi0: DependentParam<FH, V, Vec<SimplexDist<FH, V>>>,
    /// parameters of conditional opinions $\Psi^K \implies H^K$ when $\phi^K_0$ is true
    #[serde_as(as = "TryFromInto<DependentParam<KH, V, Vec<SimplexParam<V>>>>")]
    kh_kpsi_if_kphi0: DependentParam<KH, V, Vec<SimplexDist<KH, V>>>,

    #[serde_as(as = "TryFromInto<Vec<EValueParam<V>>>")]
    uncertainty_fh_fo_if_fphi0: MArrD1<FO, EValue<V>>,
    #[serde_as(as = "TryFromInto<Vec<EValueParam<V>>>")]
    uncertainty_fh_fo_if_fphi1: MArrD1<FO, EValue<V>>,
}

#[serde_as]
#[derive(Debug, serde::Deserialize)]
#[serde(bound(deserialize = "V: serde::Deserialize<'de>"))]
pub struct InitialOpinions<V>
where
    V: MyFloat,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    Open01: Distribution<V>,
{
    state: InitialState<V>,
    fixed: InitialFixed<V>,
    deduction_base_rates: InitialBaseRates<V>,
}

#[derive(Default, Debug)]
struct StateOpinions<V: MyFloat> {
    psi: OpinionD1<Psi, V>,
    phi: OpinionD1<Phi, V>,
    o: OpinionD1<O, V>,
    fo: OpinionD1<FO, V>,
    ko: OpinionD1<KO, V>,
    h_psi_if_phi1: MArrD1<Psi, SimplexD1<H, V>>,
    h_b_if_phi1: MArrD1<B, SimplexD1<H, V>>,
    fpsi: OpinionD1<FPsi, V>,
    fphi: OpinionD1<FPhi, V>,
    fh_fpsi_if_fphi1: MArrD1<FPsi, SimplexD1<FH, V>>,
    kpsi: OpinionD1<KPsi, V>,
    kphi: OpinionD1<KPhi, V>,
    kh_kpsi_if_kphi1: MArrD1<KPsi, SimplexD1<KH, V>>,
}

#[derive(Default, Debug)]
struct FixedOpinions<V: MyFloat> {
    o_b: MArrD1<B, SimplexD1<O, V>>,
    b_kh: MArrD1<KH, SimplexD1<B, V>>,
    a_fh: MArrD1<FH, SimplexD1<A, V>>,
    theta_h: MArrD1<H, SimplexD1<Theta, V>>,
    thetad_h: MArrD1<H, SimplexD1<Thetad, V>>,
    h_psi_if_phi0: MArrD1<Psi, SimplexD1<H, V>>,
    h_b_if_phi0: MArrD1<B, SimplexD1<H, V>>,
    fh_fpsi_if_fphi0: MArrD1<FPsi, SimplexD1<FH, V>>,
    kh_kpsi_if_kphi0: MArrD1<KPsi, SimplexD1<KH, V>>,
    uncertainty_fh_fo_if_fphi0: MArrD1<FO, V>,
    uncertainty_fh_fo_if_fphi1: MArrD1<FO, V>,
}

#[derive(Debug)]
enum DiffOpinions<V: MyFloat> {
    Causal {
        psi: OpinionD1<Psi, V>,
        fpsi: OpinionD1<FPsi, V>,
        kpsi: OpinionD1<KPsi, V>,
    },
    Observed {
        o: OpinionD1<O, V>,
        fo: OpinionD1<FO, V>,
        ko: OpinionD1<KO, V>,
    },
    Inhibition {
        phi: OpinionD1<Phi, V>,
        fphi: OpinionD1<FPhi, V>,
        kphi: OpinionD1<KPhi, V>,
        h_psi_if_phi1: MArrD1<Psi, SimplexD1<H, V>>,
        h_b_if_phi1: MArrD1<B, SimplexD1<H, V>>,
        fh_fpsi_if_fphi1: MArrD1<FPsi, SimplexD1<FH, V>>,
        kh_kpsi_if_kphi1: MArrD1<KPsi, SimplexD1<KH, V>>,
    },
}

#[derive(Debug)]
enum PredDiffOpinions<V: MyFloat> {
    Causal {
        fpsi: OpinionD1<FPsi, V>,
    },
    Observed {
        fo: OpinionD1<FO, V>,
    },
    Inhibition {
        fphi: OpinionD1<FPhi, V>,
        fh_fpsi_if_fphi1: MArrD1<FPsi, SimplexD1<FH, V>>,
    },
}

#[derive(Debug, Default)]
pub struct DeducedOpinions<V: MyFloat> {
    h: OpinionD1<H, V>,
    fh: OpinionD1<FH, V>,
    kh: OpinionD1<KH, V>,
    a: OpinionD1<A, V>,
    b: OpinionD1<B, V>,
    theta: OpinionD1<Theta, V>,
    thetad: OpinionD1<Thetad, V>,
}

pub struct Trusts<V: MyFloat> {
    pub info: V,
    pub corr_misinfo: V,
    pub friend: V,
    pub social: V,
    pub pi_friend: V,
    pub pi_social: V,
    pub pred_friend: V,
}

fn avg<D: Domain<Idx: Copy> + Keys<D::Idx>, V: MyFloat>(
    p: &MArrD1<D, V>,
    a: &MArrD1<D, V>,
    r: V,
) -> MArrD1<D, V> {
    MArrD1::from_fn(|i| r * p[i] + (V::one() - r) * a[i])
}

fn transform<
    D1: Domain<Idx: Copy> + Keys<D1::Idx>,
    D2: Domain<Idx: Debug> + From<D1> + Keys<D2::Idx>,
    V: MyFloat,
>(
    w: OpinionRefD1<D1, V>,
    e: V,
) -> OpinionD1<D2, V> {
    let p: MArrD1<D2, _> = w.projection().conv();
    OpinionD1::new(
        MArrD1::from_fn(|i| p[i] * e),
        V::one() - e,
        w.base_rate.clone().conv(),
    )
}

fn transform_simplex<
    D1: Domain<Idx: Copy> + Keys<D1::Idx>,
    D2: Domain<Idx: Debug> + From<D1> + Keys<D2::Idx>,
    V: MyFloat,
>(
    p: MArrD1<D1, V>,
    e: V,
) -> SimplexD1<D2, V> {
    let q: MArrD1<D2, _> = p.conv();
    SimplexD1::new_unchecked(MArrD1::from_fn(|i| q[i] * e), V::one() - e)
}

impl<V: MyFloat> StateOpinions<V> {
    fn receive(
        &self,
        p: &InfoContent<V>,
        trusts: &Trusts<V>,
        ded: &DeducedOpinions<V>,
    ) -> DiffOpinions<V> {
        match p {
            InfoContent::Misinfo { op } => {
                let op = op.discount(trusts.info);
                let psi = FuseOp::Wgh.fuse(&self.psi, &op);
                let fpsi = FuseOp::Wgh.fuse(&self.fpsi, &transform(op.as_ref(), trusts.friend));
                let kpsi = FuseOp::Wgh.fuse(&self.kpsi, &transform(op.as_ref(), trusts.social));
                DiffOpinions::Causal { psi, fpsi, kpsi }
            }
            InfoContent::Correction { op, misinfo } => {
                let op = op.discount(trusts.info);
                let m = misinfo.discount(trusts.corr_misinfo);
                let psi = FuseOp::Wgh.fuse(&self.psi, &op);
                let mut fpsi = FuseOp::Wgh.fuse(&self.fpsi, &transform(op.as_ref(), trusts.friend));
                FuseOp::Wgh.fuse_assign(&mut fpsi, &transform(m.as_ref(), trusts.pi_friend));
                let mut kpsi = FuseOp::Wgh.fuse(&self.kpsi, &transform(op.as_ref(), trusts.social));
                FuseOp::Wgh.fuse_assign(&mut kpsi, &transform(m.as_ref(), trusts.pi_social));
                DiffOpinions::Causal { psi, fpsi, kpsi }
            }
            InfoContent::Observation { op } => {
                let op = op.discount(trusts.info);
                let o = FuseOp::Wgh.fuse(&self.o, &op);
                let fo = FuseOp::Wgh.fuse(&self.fo, &transform(op.as_ref(), trusts.friend));
                let ko = FuseOp::Wgh.fuse(&self.ko, &transform(op.as_ref(), trusts.social));
                DiffOpinions::Observed { o, fo, ko }
            }
            InfoContent::Inhibition { op1, op2, op3 } => {
                let op1_dsc = op1.discount(trusts.info);
                let phi = FuseOp::Wgh.fuse(&self.phi, &op1_dsc);
                let fphi =
                    FuseOp::Wgh.fuse(&self.fphi, &transform(op1_dsc.as_ref(), trusts.friend));
                let kphi =
                    FuseOp::Wgh.fuse(&self.kphi, &transform(op1_dsc.as_ref(), trusts.social));
                let op2_dsc = op2
                    .iter()
                    .map(|o| o.discount(trusts.info))
                    .collect::<Vec<_>>();
                let op3_dsc = op3
                    .iter()
                    .map(|o| o.discount(trusts.info))
                    .collect::<Vec<_>>();

                let h_psi_if_phi1 = MArrD1::from_iter(
                    self.h_psi_if_phi1
                        .iter()
                        .zip(&op2_dsc)
                        .map(|(c, o)| FuseOp::Wgh.fuse(c, o)),
                );
                let h_b_if_phi1 = MArrD1::from_iter(
                    self.h_b_if_phi1
                        .iter()
                        .zip(&op3_dsc)
                        .map(|(c, o)| FuseOp::Wgh.fuse(c, o)),
                );
                let ah = &ded.h.base_rate;
                let fh_fpsi_if_fphi1 =
                    MArrD1::from_iter(self.fh_fpsi_if_fphi1.iter().zip(&op2_dsc).map(|(c, o)| {
                        FuseOp::Wgh.fuse(c, &transform_simplex(o.projection(ah), trusts.friend))
                    }));
                let kh_kpsi_if_kphi1 =
                    MArrD1::from_iter(self.kh_kpsi_if_kphi1.iter().zip(&op2_dsc).map(|(c, o)| {
                        FuseOp::Wgh.fuse(c, &transform_simplex(o.projection(ah), trusts.social))
                    }));
                DiffOpinions::Inhibition {
                    phi,
                    fphi,
                    kphi,
                    h_psi_if_phi1,
                    h_b_if_phi1,
                    fh_fpsi_if_fphi1,
                    kh_kpsi_if_kphi1,
                }
            }
        }
    }

    fn predict(
        &self,
        p: &InfoContent<V>,
        trusts: &Trusts<V>,
        ded: &DeducedOpinions<V>,
    ) -> PredDiffOpinions<V> {
        match p {
            InfoContent::Misinfo { op } => {
                let fpsi = FuseOp::Wgh.fuse(
                    &self.fpsi,
                    &transform(op.discount(trusts.info).as_ref(), trusts.pred_friend),
                );
                PredDiffOpinions::Causal { fpsi }
            }
            InfoContent::Observation { op } => {
                let fo = FuseOp::Wgh.fuse(
                    &self.fo,
                    &transform(op.discount(trusts.info).as_ref(), trusts.pred_friend),
                );
                PredDiffOpinions::Observed { fo }
            }
            InfoContent::Correction { op, .. } => {
                let fpsi = FuseOp::Wgh.fuse(
                    &self.fpsi,
                    &transform(op.discount(trusts.info).as_ref(), trusts.pred_friend),
                );
                PredDiffOpinions::Causal { fpsi }
            }
            InfoContent::Inhibition { op1, op2, .. } => {
                let fphi = FuseOp::Wgh.fuse(
                    &self.fphi,
                    &transform(op1.discount(trusts.info).as_ref(), trusts.pred_friend),
                );
                let ah = &ded.h.base_rate;
                let fh_fpsi_if_fphi1 = MArrD1::from_iter(
                    self.fh_fpsi_if_fphi1.iter().zip(op2.iter()).map(|(c, o)| {
                        FuseOp::Wgh.fuse(
                            c,
                            &transform_simplex(
                                o.discount(trusts.info).projection(ah),
                                trusts.friend,
                            ),
                        )
                    }),
                );
                PredDiffOpinions::Inhibition {
                    fphi,
                    fh_fpsi_if_fphi1,
                }
            }
        }
    }

    fn h_psi_b_if_phi1(
        &self,
        b: &MArrD1<B, V>,
        h: &MArrD1<H, V>,
    ) -> MArrD2<Psi, B, SimplexD1<H, V>> {
        let h_psi_b_if_phi1 = MArrD1::<H, _>::merge_cond2(
            &self.h_psi_if_phi1,
            &self.h_b_if_phi1,
            &self.psi.base_rate,
            b,
            h,
        );
        h_psi_b_if_phi1
    }

    fn reset(&mut self, state: &InitialState<V>)
    where
        Standard: Distribution<V>,
        StandardNormal: Distribution<V>,
        Exp1: Distribution<V>,
        Open01: Distribution<V>,
    {
        let InitialState {
            psi,
            phi,
            o,
            fo,
            ko,
            fpsi,
            fphi,
            kpsi,
            kphi,
            h_psi_if_phi1,
            h_b_if_phi1,
            fh_fpsi_if_fphi1,
            kh_kpsi_if_kphi1,
        } = state.clone();

        *self = Self {
            psi,
            phi,
            o,
            fo,
            ko,
            h_psi_if_phi1,
            h_b_if_phi1,
            fpsi,
            fphi,
            fh_fpsi_if_fphi1,
            kpsi,
            kphi,
            kh_kpsi_if_kphi1,
        };
    }
}

impl<V: MyFloat> FixedOpinions<V> {
    fn h_psi_b_if_phi0(
        &self,
        psi: &MArrD1<Psi, V>,
        b: &MArrD1<B, V>,
        h: &MArrD1<H, V>,
    ) -> MArrD2<Psi, B, SimplexD1<H, V>> {
        let h_psi_b_if_phi0 =
            MArrD1::<H, _>::merge_cond2(&self.h_psi_if_phi0, &self.h_b_if_phi0, psi, b, h);
        h_psi_b_if_phi0
    }

    fn b_kh_o(
        &self,
        b: &MArrD1<B, V>,
        kh: &MArrD1<KH, V>,
        o: &MArrD1<O, V>,
    ) -> MArrD2<KH, O, SimplexD1<B, V>> {
        let b_o = self.o_b.inverse(b, o);
        let b_kh_o = MArrD1::<B, _>::merge_cond2(&self.b_kh, &b_o, kh, o, b);
        debug!("{:?}", b_kh_o);
        b_kh_o
    }

    fn new<R: Rng>(fixed: &InitialFixed<V>, rng: &mut R) -> Self
    where
        Standard: Distribution<V>,
        StandardNormal: Distribution<V>,
        Exp1: Distribution<V>,
        Open01: Distribution<V>,
    {
        let (h_b_if_phi0, h_psi_if_phi0) = {
            let mut x = fixed.h_psi_b_if_phi0.sample(rng);
            (
                MArrD1::<B, _>::new(x.pop().unwrap()),
                MArrD1::<Psi, _>::new(x.pop().unwrap()),
            )
        };
        let fh_fpsi_if_fphi0 = {
            let base =
                MArrD1::<FPhi, SimplexD1<FH, _>>::from_fn(|i| h_psi_if_phi0[i].clone().conv());
            let x = fixed.fh_fpsi_if_fphi0.samples1(rng, base.iter());
            MArrD1::<FPsi, _>::new(x)
        };
        let kh_kpsi_if_kphi0 = {
            let base =
                MArrD1::<KPhi, SimplexD1<KH, _>>::from_fn(|i| h_psi_if_phi0[i].clone().conv());
            let x = fixed.kh_kpsi_if_kphi0.samples1(rng, base.iter());
            MArrD1::<KPsi, _>::new(x)
        };

        Self {
            o_b: MArrD1::from_fn(|i| fixed.o_b[i].sample(rng)),
            b_kh: MArrD1::from_fn(|i| fixed.b_kh[i].sample(rng)),
            a_fh: MArrD1::from_fn(|i| fixed.a_fh[i].sample(rng)),
            theta_h: MArrD1::from_fn(|i| fixed.theta_h[i].sample(rng)),
            thetad_h: MArrD1::from_fn(|i| fixed.thetad_h[i].sample(rng)),
            h_b_if_phi0,
            h_psi_if_phi0,
            fh_fpsi_if_fphi0,
            kh_kpsi_if_kphi0,
            uncertainty_fh_fo_if_fphi0: MArrD1::from_fn(|i| {
                fixed.uncertainty_fh_fo_if_fphi0[i].sample(rng)
            }),
            uncertainty_fh_fo_if_fphi1: MArrD1::from_fn(|i| {
                fixed.uncertainty_fh_fo_if_fphi1[i].sample(rng)
            }),
        }
    }
}

fn deduce_fh<V: MyFloat>(
    state: &StateOpinions<V>,
    fixed: &FixedOpinions<V>,
    b_o: &MArrD1<O, SimplexD1<B, V>>,
    b: &MArrD1<B, V>,
    h: &MArrD1<H, V>,
    fh: &MArrD1<FH, V>,
) -> OpinionD1<FH, V> {
    let fh_fo_if_fphi0 = MArrD1::<FO, _>::from_fn(|fo| {
        transform_simplex::<_, FH, _>(
            OpinionRefD1::from((&b_o[fo], b))
                .deduce_with(&fixed.h_b_if_phi0, || h.clone())
                .projection(),
            fixed.uncertainty_fh_fo_if_fphi0[fo],
        )
    });
    let fh_fo_if_fphi1 = MArrD1::<FO, _>::from_fn(|fo| {
        transform_simplex::<_, FH, _>(
            OpinionRefD1::from((&b_o[fo], b))
                .deduce_with(&state.h_b_if_phi1, || h.clone())
                .projection(),
            fixed.uncertainty_fh_fo_if_fphi1[fo],
        )
    });
    let fh_fpsi_fo_if_fphi0 = MArrD1::<FH, _>::merge_cond2(
        &fixed.fh_fpsi_if_fphi0,
        &fh_fo_if_fphi0,
        &state.fpsi.base_rate,
        &state.fo.base_rate,
        fh,
    );
    let fh_fpsi_fo_if_fphi1 = MArrD1::<FH, _>::merge_cond2(
        &state.fh_fpsi_if_fphi1,
        &fh_fo_if_fphi1,
        &state.fpsi.base_rate,
        &state.fo.base_rate,
        fh,
    );
    let fh_fphi_fpsi_fo =
        MArrD3::<FPhi, FPsi, FO, _>::new(vec![fh_fpsi_fo_if_fphi0, fh_fpsi_fo_if_fphi1]);

    debug!("{:?}", fh_fphi_fpsi_fo);

    let fh = OpinionD3::product3(&state.fphi, &state.fpsi, &state.fo)
        .deduce_with(&fh_fphi_fpsi_fo, || fh.clone());
    fh
}

fn deduce_kh<V: MyFloat>(
    state: &StateOpinions<V>,
    fixed: &FixedOpinions<V>,
    b_o: &MArrD1<O, SimplexD1<B, V>>,
    b: &MArrD1<B, V>,
    h: &MArrD1<H, V>,
    kh: &MArrD1<KH, V>,
) -> OpinionD1<KH, V> {
    let kh_ko_if_kphi0 = MArrD1::<KO, _>::from_fn(|ko| {
        transform_simplex::<_, KH, _>(
            OpinionRefD1::from((&b_o[ko], b))
                .deduce_with(&fixed.h_b_if_phi0, || h.clone())
                .projection(),
            V::zero(),
        )
    });
    let kh_ko_if_kphi1 = MArrD1::<KO, _>::from_fn(|ko| {
        transform_simplex::<_, KH, _>(
            OpinionRefD1::from((&b_o[ko], b))
                .deduce_with(&state.h_b_if_phi1, || h.clone())
                .projection(),
            V::zero(),
        )
    });
    let kh_kpsi_ko_if_kphi0 = MArrD1::<KH, _>::merge_cond2(
        &fixed.kh_kpsi_if_kphi0,
        &kh_ko_if_kphi0,
        &state.kpsi.base_rate,
        &state.ko.base_rate,
        kh,
    );
    let kh_kpsi_ko_if_kphi1 = MArrD1::<KH, _>::merge_cond2(
        &state.kh_kpsi_if_kphi1,
        &kh_ko_if_kphi1,
        &state.kpsi.base_rate,
        &state.ko.base_rate,
        kh,
    );
    let kh_kphi_kpsi_ko =
        MArrD3::<KPhi, KPsi, KO, _>::new(vec![kh_kpsi_ko_if_kphi0, kh_kpsi_ko_if_kphi1]);

    debug!("{:?}", kh_kphi_kpsi_ko);

    let kh = OpinionD3::product3(&state.kphi, &state.kpsi, &state.ko)
        .deduce_with(&kh_kphi_kpsi_ko, || kh.clone());
    kh
}

fn deduce_b<V: MyFloat>(
    state: &StateOpinions<V>,
    fixed: &FixedOpinions<V>,
    b_o: &MArrD1<O, SimplexD1<B, V>>,
    kh: &OpinionD1<KH, V>,
    b: &MArrD1<B, V>,
) -> OpinionD1<B, V> {
    let b_kh_o =
        MArrD1::<B, _>::merge_cond2(&fixed.b_kh, b_o, &kh.base_rate, &state.o.base_rate, b);
    debug!("{:?}", b_kh_o);
    let b = OpinionD2::product2(kh, &state.o).deduce_with(&b_kh_o, || b.clone());
    b
}

fn deduce_h<V: MyFloat>(
    state: &StateOpinions<V>,
    fixed: &FixedOpinions<V>,
    b: &OpinionD1<B, V>,
    h: &MArrD1<H, V>,
) -> OpinionD1<H, V> {
    let h_psi_b_if_phi0 = fixed.h_psi_b_if_phi0(&state.psi.base_rate, &b.base_rate, h);
    let h_psi_b_if_phi1 = state.h_psi_b_if_phi1(&b.base_rate, h);
    let h_phi_psi_b = MArrD3::<Phi, _, _, _>::new(vec![h_psi_b_if_phi0, h_psi_b_if_phi1]);
    debug!("{:?}", h_phi_psi_b);

    let h = OpinionD3::product3(&state.phi, &state.psi, &b).deduce_with(&h_phi_psi_b, || h.clone());
    h
}

impl<V: MyFloat> DeducedOpinions<V> {
    fn deduce(&self, state: &StateOpinions<V>, fixed: &FixedOpinions<V>) -> Self {
        let ab = &self.b.base_rate;
        let b_o = fixed.o_b.inverse(ab, &state.o.base_rate);
        let fh = deduce_fh(
            state,
            fixed,
            &b_o,
            &self.b.base_rate,
            &self.h.base_rate,
            &self.fh.base_rate,
        );
        let kh = deduce_kh(
            state,
            fixed,
            &b_o,
            &self.b.base_rate,
            &self.h.base_rate,
            &self.kh.base_rate,
        );
        let b = deduce_b(state, fixed, &b_o, &kh, &self.b.base_rate);
        let h = deduce_h(state, fixed, &b, &self.h.base_rate);
        let a = fh.deduce_with(&fixed.a_fh, || self.a.base_rate.clone());
        let theta = h.deduce_with(&fixed.theta_h, || self.theta.base_rate.clone());
        let thetad = h.deduce_with(&fixed.thetad_h, || self.thetad.base_rate.clone());

        Self {
            h,
            fh,
            kh,
            a,
            b,
            theta,
            thetad,
        }
    }

    pub fn p_theta(&self) -> MArrD1<Theta, V> {
        self.theta.projection()
    }

    pub fn p_a_thetad(&self) -> MArrD2<A, Thetad, V> {
        OpinionD2::product2(&self.a, &self.thetad).projection()
    }

    fn reset(&mut self, base_rates: &InitialBaseRates<V>) {
        let InitialBaseRates {
            a,
            b,
            h,
            fh,
            kh,
            theta,
            thetad,
        } = base_rates.clone();
        *self = Self {
            a: OpinionD1::vacuous_with(a),
            b: OpinionD1::vacuous_with(b),
            h: OpinionD1::vacuous_with(h),
            fh: OpinionD1::vacuous_with(fh),
            kh: OpinionD1::vacuous_with(kh),
            theta: OpinionD1::vacuous_with(theta),
            thetad: OpinionD1::vacuous_with(thetad),
        }
    }
}

impl<V: MyFloat> DiffOpinions<V> {
    fn swap(&mut self, state: &mut StateOpinions<V>) {
        match self {
            DiffOpinions::Causal { psi, fpsi, kpsi } => {
                mem::swap(&mut state.psi, psi);
                mem::swap(&mut state.fpsi, fpsi);
                mem::swap(&mut state.kpsi, kpsi);
            }
            DiffOpinions::Observed { o, fo, ko } => {
                mem::swap(&mut state.o, o);
                mem::swap(&mut state.fo, fo);
                mem::swap(&mut state.ko, ko);
            }
            DiffOpinions::Inhibition {
                phi,
                fphi,
                kphi,
                h_psi_if_phi1,
                h_b_if_phi1,
                fh_fpsi_if_fphi1,
                kh_kpsi_if_kphi1,
            } => {
                mem::swap(&mut state.phi, phi);
                mem::swap(&mut state.fphi, fphi);
                mem::swap(&mut state.kphi, kphi);
                mem::swap(&mut state.h_psi_if_phi1, h_psi_if_phi1);
                mem::swap(&mut state.h_b_if_phi1, h_b_if_phi1);
                mem::swap(&mut state.fh_fpsi_if_fphi1, fh_fpsi_if_fphi1);
                mem::swap(&mut state.kh_kpsi_if_kphi1, kh_kpsi_if_kphi1);
            }
        }
    }
}

impl<V: MyFloat> PredDiffOpinions<V> {
    fn swap(&mut self, state: &mut StateOpinions<V>) {
        match self {
            PredDiffOpinions::Causal { fpsi } => {
                mem::swap(fpsi, &mut state.fpsi);
            }
            PredDiffOpinions::Observed { fo } => {
                mem::swap(fo, &mut state.fo);
            }
            PredDiffOpinions::Inhibition {
                fphi,
                fh_fpsi_if_fphi1,
            } => {
                mem::swap(fphi, &mut state.fphi);
                mem::swap(fh_fpsi_if_fphi1, &mut state.fh_fpsi_if_fphi1);
            }
        }
    }
}

#[derive(Default)]
pub struct MyOpinions<V: MyFloat> {
    state: StateOpinions<V>,
    ded: DeducedOpinions<V>,
    fixed: FixedOpinions<V>,
}

impl<V: MyFloat> MyOpinions<V> {
    pub fn reset<R: Rng>(&mut self, opinions: &InitialOpinions<V>, rng: &mut R)
    where
        Standard: Distribution<V>,
        StandardNormal: Distribution<V>,
        Exp1: Distribution<V>,
        Open01: Distribution<V>,
    {
        self.state.reset(&opinions.state);
        self.ded.reset(&opinions.deduction_base_rates);
        self.fixed = FixedOpinions::new(&opinions.fixed, rng);
        debug!("{:?}", self.fixed);
    }

    pub fn receive<'a>(
        &'a mut self,
        p: &'a InfoContent<V>,
        trusts: Trusts<V>,
    ) -> MyOpinionsUpd<'a, V> {
        let mut diff = self.state.receive(p, &trusts, &self.ded);
        debug!("{:?}", &diff);

        diff.swap(&mut self.state);
        debug!("{:?}", &self.state);

        self.ded = self.ded.deduce(&self.state, &self.fixed);
        debug!("{:?}", &self.ded);

        MyOpinionsUpd {
            inner: self,
            p,
            trusts,
        }
    }

    fn predict(
        &mut self,
        p: &InfoContent<V>,
        trusts: &Trusts<V>,
    ) -> (PredDiffOpinions<V>, DeducedOpinions<V>) {
        let mut pred_diff = self.state.predict(p, trusts, &self.ded);
        debug!("{:?}", pred_diff);

        pred_diff.swap(&mut self.state);
        let pred_ded = self.ded.deduce(&self.state, &self.fixed);
        (pred_diff, pred_ded)
    }
}

pub struct MyOpinionsUpd<'a, V: MyFloat> {
    inner: &'a mut MyOpinions<V>,
    p: &'a InfoContent<V>,
    trusts: Trusts<V>,
}

impl<'a, V: MyFloat> MyOpinionsUpd<'a, V> {
    pub fn decide1<F>(&self, mut f: F)
    where
        F: FnMut(&DeducedOpinions<V>),
    {
        f(&self.inner.ded);
    }

    pub fn decide2<F>(&mut self, mut f: F) -> bool
    where
        F: FnMut(&DeducedOpinions<V>, &DeducedOpinions<V>) -> bool,
    {
        let (mut pred_diff, pred_ded) = self.inner.predict(self.p, &self.trusts);
        if f(&self.inner.ded, &pred_ded) {
            self.inner.ded = pred_ded;
            true
        } else {
            pred_diff.swap(&mut self.inner.state);
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::UlpsEq;
    use num_traits::{Float, NumAssign};
    use std::fmt::Debug;
    use std::fs::read_to_string;
    use std::iter::Sum;
    use subjective_logic::domain::DomainConv;
    use subjective_logic::iter::FromFn;
    use subjective_logic::mul::labeled::OpinionD1;
    use subjective_logic::mul::{mbr, MergeJointConditions2};
    use subjective_logic::multi_array::labeled::MArrD1;
    use subjective_logic::ops::{Deduction, Discount, FuseAssign, FuseOp, Projection};
    use subjective_logic::{marr_d1, mul::labeled::SimplexD1};

    use crate::opinion::gen2::transform;
    use crate::opinion::{FPsi, MyFloat, Psi, B, FH, H};

    use super::super::paramter::SimplexDist;
    use super::InitialOpinions;

    #[test]
    fn test_toml() -> anyhow::Result<()> {
        let initial_opinions = toml::from_str::<InitialOpinions<f32>>(&read_to_string(
            "./test/config/test_initial_opinions.toml",
        )?)?;
        assert!(matches!(
            &initial_opinions.fixed.theta_h[0],
            SimplexDist::Fixed(s) if s.b() == &marr_d1![1.0, 0.0] && s.u() == &0.0,
        ));
        assert!(matches!(
            &initial_opinions.fixed.theta_h[1],
            SimplexDist::Fixed(s) if s.b() == &marr_d1![0.0, 0.7] && s.u() == &0.3,
        ));
        assert!(initial_opinions.deduction_base_rates.a == marr_d1![0.99, 0.01]);
        Ok(())
    }

    fn deduce<V: Float + Sum + NumAssign + UlpsEq + Debug>(
        psi: &OpinionD1<Psi, V>,
        fpsi: &OpinionD1<FPsi, V>,
        c: &MArrD1<Psi, SimplexD1<H, V>>,
        cf: &MArrD1<FPsi, SimplexD1<FH, V>>,
        ah: &MArrD1<H, V>,
    ) {
        let h = psi.deduce_with(c, || ah.clone());
        let fh = fpsi.deduce(cf);
        println!("h {:?}", h);
        println!("Ph {:?}", h.projection());
        println!("fh {:?}", fh);
        println!("Pfh {:?}", fh.unwrap().projection());
    }

    enum Info<V> {
        Mis(OpinionD1<Psi, V>),
        Cor(OpinionD1<Psi, V>, OpinionD1<Psi, V>),
    }

    struct Trust<V> {
        mis: V,
        cor: V,
    }

    impl<V> Trust<V> {
        fn new(mis: V, cor: V) -> Self {
            Self { mis, cor }
        }
    }

    #[derive(Default)]
    struct Body<V> {
        e: V,
        psi: OpinionD1<Psi, V>,
        fpsi: OpinionD1<FPsi, V>,
        c: MArrD1<Psi, SimplexD1<H, V>>,
        cf: MArrD1<FPsi, SimplexD1<FH, V>>,
        ah: MArrD1<H, V>,
        pi: V,
    }

    impl Body<f32> {
        fn reset(&mut self, pi: f32) {
            *self = Body {
                e: 0.2,
                psi: OpinionD1::vacuous_with(marr_d1![0.99, 0.01]),
                fpsi: OpinionD1::vacuous_with(marr_d1![0.99, 0.01]),
                c: marr_d1![
                    SimplexD1::new(marr_d1![0.9, 0.0], 0.1),
                    SimplexD1::new(marr_d1![0.0, 0.8], 0.2),
                ],
                cf: marr_d1![
                    SimplexD1::new(marr_d1![0.7, 0.1], 0.2),
                    SimplexD1::new(marr_d1![0.1, 0.7], 0.2),
                ],
                ah: marr_d1![0.99, 0.01],
                pi,
            };
        }
    }

    impl<V: MyFloat> Body<V> {
        fn update(&mut self, p: &Info<V>, t: &Trust<V>) {
            println!("[update]");
            match p {
                Info::Mis(op) => {
                    let rcv = op.discount(t.mis);
                    let est = transform(rcv.as_ref(), self.e);
                    println!("-");
                    println!("rcv {:?}", rcv);
                    println!("est {:?}", est);
                    FuseOp::Wgh.fuse_assign(&mut self.psi, &rcv);
                    FuseOp::Wgh.fuse_assign(&mut self.fpsi, &est);
                }
                Info::Cor(op, m) => {
                    let rcv = op.discount(t.cor);
                    let est_op = transform(rcv.as_ref(), self.e);
                    let est_m = transform(m.discount(t.mis).as_ref(), self.pi);
                    println!("-");
                    println!("rcv {:?}", rcv);
                    println!("est {:?}", est_op);
                    println!("estm {:?}", est_m);
                    FuseOp::Wgh.fuse_assign(&mut self.psi, &rcv);
                    FuseOp::Wgh.fuse_assign(&mut self.fpsi, &est_op);
                    FuseOp::Wgh.fuse_assign(&mut self.fpsi, &est_m);
                }
            }
            let h = self.psi.deduce_with(&self.c, || self.ah.clone());
            let fh = self.fpsi.deduce(&self.cf).unwrap();
            println!("-");
            println!("psi {:?}", self.psi);
            println!("h {:?}", h);
            println!("Ph {:?}", h.projection());
            println!("-");
            println!("fpsi {:?}", self.fpsi);
            println!("fh {:?}", fh);
            println!("Pfh {:?}", fh.projection());
        }

        fn gen_cf(
            apsi: &MArrD1<Psi, V>,
            c: &MArrD1<Psi, SimplexD1<H, V>>,
            ah: &MArrD1<H, V>,
        ) -> MArrD1<FPsi, SimplexD1<FH, V>> {
            let mah = mbr(apsi, c).unwrap_or_else(|| ah.clone());
            let cf = MArrD1::<FPsi, _>::from_fn(|i| {
                SimplexD1::<FH, _>::new(
                    MArrD1::from_iter(
                        c[i].projection(&mah)
                            .iter()
                            .map(|p| *p * (V::one() - *c[i].u())),
                    ),
                    *c[i].u(),
                )
            });
            cf
        }

        fn gen_cf2(
            apsi: &MArrD1<Psi, V>,
            c: &MArrD1<Psi, SimplexD1<H, V>>,
            ah: &MArrD1<H, V>,
        ) -> MArrD1<FPsi, SimplexD1<FH, V>>
        where
            V: Float + Debug + Sum + UlpsEq + NumAssign,
        {
            let cf0 = OpinionD1::new(marr_d1![V::one(), V::zero()], V::zero(), apsi.clone())
                .deduce_with(c, || ah.clone());
            let cf1 = OpinionD1::new(marr_d1![V::zero(), V::one()], V::zero(), apsi.clone())
                .deduce_with(c, || ah.clone());
            let cf = MArrD1::new(vec![cf0.simplex.conv(), cf1.simplex.conv()]);
            cf
        }
    }

    #[test]
    fn test_misinfo() {
        let mis = [
            Info::Mis(OpinionD1::new(
                marr_d1![0.05, 0.9],
                0.05,
                marr_d1![0.99, 0.01],
            )),
            Info::Mis(OpinionD1::new(
                marr_d1![0.05, 0.9],
                0.05,
                marr_d1![0.80, 0.20],
            )),
        ];
        let cis = [
            Info::Cor(
                OpinionD1::new(marr_d1![0.9, 0.05], 0.05, marr_d1![0.99, 0.01]),
                OpinionD1::new(marr_d1![0.05, 0.9], 0.05, marr_d1![0.50, 0.50]),
            ),
            Info::Cor(
                OpinionD1::new(marr_d1![0.9, 0.05], 0.05, marr_d1![0.90, 0.10]),
                OpinionD1::new(marr_d1![0.05, 0.9], 0.05, marr_d1![0.80, 0.20]),
            ),
        ];

        let mut body = Body::default();
        let ts = [Trust::new(0.5, 0.9), Trust::new(0.5, 0.5)];
        let pis = [0.0, 0.25, 0.5];

        for (h, t) in ts.iter().enumerate() {
            for (i, ci) in cis.iter().enumerate() {
                for (k, pi) in pis.iter().enumerate() {
                    println!("\n-- case x.{h}.{i}.{k} --");
                    body.reset(*pi);
                    body.update(ci, &t);
                }
            }
        }

        for (h, t) in ts.iter().enumerate() {
            for (i, mi) in mis.iter().enumerate() {
                for (j, ci) in cis.iter().enumerate() {
                    for (k, pi) in pis.iter().enumerate() {
                        println!("\n-- Case {h}.{i}.{j}.{k} --");
                        body.reset(*pi);
                        body.update(mi, t);
                        body.update(ci, t);
                    }
                }
            }
        }
    }

    #[test]
    fn merge() {
        let h_psi_if_phi0 = marr_d1!(Psi; [
            SimplexD1::new(marr_d1!(H; [0.95f32, 0.01]), 0.04),
            SimplexD1::new(marr_d1!(H; [0.01, 0.95]), 0.04),
        ]);

        let h_b_if_phi0 = marr_d1!(B; [
            SimplexD1::new(marr_d1!(H; [0.95, 0.01]), 0.04),
            SimplexD1::new(marr_d1!(H; [1.8101491e-5, 0.9998891]),9.2769165e-5),
        ]);
        // let h_psi_if_phi1 = marr_d1!(Psi; [SimplexD1::vacuous(), SimplexD1::vacuous()]);
        // let h_b_if_phi1 = marr_d1!(Psi; [SimplexD1::vacuous(), SimplexD1::vacuous()]);

        let psi = marr_d1!(Psi; [0.1, 0.9]);
        let b = marr_d1!(B; [0.5, 0.5]);
        let h = marr_d1!(H; [0.99, 0.01]);
        let c = MArrD1::<H, _>::merge_cond2(&h_psi_if_phi0, &h_b_if_phi0, &psi, &b, &h);

        println!("{:?}", c);
    }
}
