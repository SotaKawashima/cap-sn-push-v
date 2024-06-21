use std::mem;

use rand::Rng;
use rand_distr::{Distribution, Exp1, Open01, Standard, StandardNormal};
use serde_with::{serde_as, TryFromInto};
use subjective_logic::{
    domain::DomainConv,
    iter::FromFn,
    mul::{
        labeled::{OpinionD1, OpinionD2, OpinionD3, SimplexD1},
        InverseCondition, MergeJointConditions2, OpinionRef,
    },
    multi_array::labeled::{MArrD1, MArrD2, MArrD3},
    ops::{Deduction, Discount, Fuse, FuseAssign, FuseOp, Product2, Product3, Projection},
};
use tracing::debug;

use crate::info::gen2::InfoContent;

use super::{
    paramter::{ConditionParams, DependentParam, SimplexDist, SimplexParam},
    FPhi, FPsi, KPhi, KPsi, MyFloat, Phi, Psi, Theta, Thetad, A, B, FH, H, KH, O,
};

#[serde_as]
#[derive(Debug, serde::Deserialize)]
#[serde(bound(deserialize = "V: serde::Deserialize<'de>"))]
struct InitialBaseRates<V> {
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    psi: MArrD1<Psi, V>,
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    phi: MArrD1<Phi, V>,
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    o: MArrD1<O, V>,
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    fpsi: MArrD1<FPsi, V>,
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    fphi: MArrD1<FPhi, V>,
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    kpsi: MArrD1<KPsi, V>,
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    kphi: MArrD1<KPhi, V>,
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
#[derive(Debug, serde::Deserialize)]
#[serde(bound(deserialize = "V: serde::Deserialize<'de>"))]
struct InitialSimplexes<V: MyFloat> {
    #[serde_as(as = "TryFromInto<(Vec<V>, V)>")]
    psi: SimplexD1<Psi, V>,
    #[serde_as(as = "TryFromInto<(Vec<V>, V)>")]
    phi: SimplexD1<Phi, V>,
    #[serde_as(as = "TryFromInto<(Vec<V>, V)>")]
    o: SimplexD1<O, V>,
    #[serde_as(as = "TryFromInto<(Vec<V>, V)>")]
    fpsi: SimplexD1<FPsi, V>,
    #[serde_as(as = "TryFromInto<(Vec<V>, V)>")]
    fphi: SimplexD1<FPhi, V>,
    #[serde_as(as = "TryFromInto<(Vec<V>, V)>")]
    kpsi: SimplexD1<KPsi, V>,
    #[serde_as(as = "TryFromInto<(Vec<V>, V)>")]
    kphi: SimplexD1<KPhi, V>,
}

#[serde_as]
#[derive(Debug, serde::Deserialize)]
#[serde(bound(deserialize = "V: serde::Deserialize<'de>"))]
struct InitialConditions<V>
where
    V: MyFloat,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    Open01: Distribution<V>,
{
    /// parameters of conditional opinions $\Psi \implies H$ and $B \implies H$ when $\phi_0$ is true
    params_h_psi_b_if_phi0: ConditionParams<Psi, B, H, V>,
    #[serde_as(as = "TryFromInto<Vec<(Vec<V>, V)>>")]
    h_psi_if_phi1: MArrD1<Psi, SimplexD1<H, V>>,
    #[serde_as(as = "TryFromInto<Vec<(Vec<V>, V)>>")]
    h_b_if_phi1: MArrD1<B, SimplexD1<H, V>>,

    /// parameters of conditional opinions $\Psi^F \implies H^F$ when $\phi^F_0$ is true
    #[serde_as(as = "TryFromInto<DependentParam<FH, V, Vec<SimplexParam<V>>>>")]
    params_fh_fpsi_if_fphi0: DependentParam<FH, V, Vec<SimplexDist<FH, V>>>,
    #[serde_as(as = "TryFromInto<Vec<(Vec<V>, V)>>")]
    fh_fpsi_if_fphi1: MArrD1<FPsi, SimplexD1<FH, V>>,

    /// parameters of conditional opinions $\Psi^K \implies H^K$ when $\phi^K_0$ is true
    #[serde_as(as = "TryFromInto<DependentParam<KH, V, Vec<SimplexParam<V>>>>")]
    params_kh_kpsi_if_kphi0: DependentParam<KH, V, Vec<SimplexDist<KH, V>>>,
    #[serde_as(as = "TryFromInto<Vec<(Vec<V>, V)>>")]
    kh_kpsi_if_kphi1: MArrD1<KPsi, SimplexD1<KH, V>>,

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
    base_rates: InitialBaseRates<V>,
    simplexes: InitialSimplexes<V>,
    conditions: InitialConditions<V>,
}

#[derive(Default)]
struct StateOpinions<V: MyFloat> {
    psi: OpinionD1<Psi, V>,
    phi: OpinionD1<Phi, V>,
    o: OpinionD1<O, V>,
    h_psi_if_phi0: MArrD1<Psi, SimplexD1<H, V>>,
    h_b_if_phi0: MArrD1<B, SimplexD1<H, V>>,
    h_psi_if_phi1: MArrD1<Psi, SimplexD1<H, V>>,
    h_b_if_phi1: MArrD1<B, SimplexD1<H, V>>,
    fpsi: OpinionD1<FPsi, V>,
    fphi: OpinionD1<FPhi, V>,
    fh_fphi_fpsi: MArrD2<FPhi, FPsi, SimplexD1<FH, V>>,
    kpsi: OpinionD1<KPsi, V>,
    kphi: OpinionD1<KPhi, V>,
    kh_kphi_kpsi: MArrD2<KPhi, KPsi, SimplexD1<KH, V>>,
    o_b: MArrD1<B, SimplexD1<O, V>>,
    b_kh: MArrD1<KH, SimplexD1<B, V>>,
    a_fh: MArrD1<FH, SimplexD1<A, V>>,
    theta_h: MArrD1<H, SimplexD1<Theta, V>>,
    thetad_h: MArrD1<H, SimplexD1<Thetad, V>>,
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
    Inhibition {
        fphi: OpinionD1<FPhi, V>,
        fh_fpsi_if_fphi1: MArrD1<FPsi, SimplexD1<FH, V>>,
    },
    None,
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
    pub friend: V,
    pub social: V,
    pub misinfo_friend: V,
    pub misinfo_social: V,
    pub pred_friend: V,
}

impl<V: MyFloat> StateOpinions<V> {
    fn receive(&self, p: &InfoContent<V>, trusts: &Trusts<V>) -> DiffOpinions<V> {
        match p {
            InfoContent::Misinfo { op } => {
                let op = op.discount(trusts.info);
                let psi = FuseOp::Wgh.fuse(&self.psi, &op);
                let fpsi = FuseOp::Wgh.fuse(&self.fpsi, &op.discount(trusts.friend).conv());
                let kpsi = FuseOp::Wgh.fuse(&self.kpsi, &op.discount(trusts.social).conv());
                DiffOpinions::Causal { psi, fpsi, kpsi }
            }
            InfoContent::Correction {
                misinfo,
                trust_misinfo,
                op,
            } => {
                let op = op.discount(trusts.info);
                let mut psi = FuseOp::Wgh.fuse(&self.psi, &op);
                FuseOp::Wgh.fuse_assign(&mut psi, &misinfo.discount(*trust_misinfo));
                let mut fpsi = FuseOp::Wgh.fuse(&self.fpsi, &op.discount(trusts.friend).conv());
                FuseOp::Wgh.fuse_assign(&mut fpsi, &misinfo.discount(trusts.misinfo_friend).conv());
                let mut kpsi = FuseOp::Wgh.fuse(&self.kpsi, &op.discount(trusts.social).conv());
                FuseOp::Wgh.fuse_assign(&mut kpsi, &misinfo.discount(trusts.misinfo_social).conv());
                DiffOpinions::Causal { psi, fpsi, kpsi }
            }
            InfoContent::Observation { op } => {
                let op = op.discount(trusts.info);
                let o = FuseOp::Wgh.fuse(&self.o, &op);
                DiffOpinions::Observed { o }
            }
            InfoContent::Inhibition { op1, op2, op3 } => {
                let op1 = op1.discount(trusts.info);
                let phi = FuseOp::Wgh.fuse(&self.phi, &op1);
                let fphi = FuseOp::Wgh.fuse(&self.fphi, &op1.discount(trusts.friend).conv());
                let kphi = FuseOp::Wgh.fuse(&self.kphi, &op1.discount(trusts.social).conv());
                let op2_iter = op2.iter().map(|o| o.discount(trusts.info));
                let op3_iter = op3.iter().map(|o| o.discount(trusts.info));

                let h_psi_if_phi1 = MArrD1::from_iter(
                    self.h_psi_if_phi1
                        .iter()
                        .zip(op2_iter.clone())
                        .map(|(c, o)| FuseOp::Wgh.fuse(c, &o)),
                );
                let h_b_if_phi1 = MArrD1::from_iter(
                    self.h_b_if_phi1
                        .iter()
                        .zip(op3_iter.clone())
                        .map(|(c, o)| FuseOp::Wgh.fuse(c, &o)),
                );
                let fh_fpsi_if_fphi1 = MArrD1::from_iter(
                    self.fh_fphi_fpsi
                        .down(1)
                        .iter()
                        .zip(op2_iter.clone())
                        .map(|(c, o)| FuseOp::Wgh.fuse(c, &o.discount(trusts.friend).conv())),
                );
                let kh_kpsi_if_kphi1 = MArrD1::from_iter(
                    self.kh_kphi_kpsi
                        .down(1)
                        .iter()
                        .zip(op2_iter)
                        .map(|(c, o)| FuseOp::Wgh.fuse(c, &o.discount(trusts.social).conv())),
                );
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

    fn predict(&self, p: &InfoContent<V>, trusts: &Trusts<V>) -> PredDiffOpinions<V> {
        match p {
            InfoContent::Misinfo { op } => {
                let fpsi = FuseOp::Wgh.fuse(
                    &self.fpsi,
                    &op.discount(trusts.info * trusts.pred_friend).conv(),
                );
                PredDiffOpinions::Causal { fpsi }
            }
            InfoContent::Correction { op, misinfo, .. } => {
                let mut fpsi = FuseOp::Wgh.fuse(
                    &self.fpsi,
                    &op.discount(trusts.info * trusts.pred_friend).conv(),
                );
                FuseOp::Wgh.fuse_assign(&mut fpsi, &misinfo.discount(trusts.misinfo_friend).conv());
                PredDiffOpinions::Causal { fpsi }
            }
            InfoContent::Inhibition { op1, op2, .. } => {
                let fphi = FuseOp::Wgh.fuse(
                    &self.fphi,
                    &op1.discount(trusts.info * trusts.pred_friend).conv(),
                );
                let fh_fpsi_if_fphi1 =
                    MArrD1::from_iter(self.fh_fphi_fpsi.down(1).iter().zip(op2.iter()).map(
                        |(c, o)| {
                            FuseOp::Wgh.fuse(c, &o.discount(trusts.info * trusts.friend).conv())
                        },
                    ));
                PredDiffOpinions::Inhibition {
                    fphi,
                    fh_fpsi_if_fphi1,
                }
            }
            _ => PredDiffOpinions::None,
        }
    }

    fn h_phi_psi_b(
        &self,
        b: &MArrD1<B, V>,
        h: &MArrD1<H, V>,
    ) -> MArrD3<Phi, Psi, B, SimplexD1<H, V>> {
        let h_psi_b_if_phi0 = MArrD1::<H, _>::merge_cond2(
            &self.h_psi_if_phi0,
            &self.h_b_if_phi0,
            &self.psi.base_rate,
            b,
            h,
        )
        .expect(&format!(
            "failed to merge {:?} & {:?} & {:?} & {:?} & {:?}",
            &self.h_psi_if_phi0, &self.h_b_if_phi0, &self.psi.base_rate, &b, &h
        ));
        let h_psi_b_if_phi1 = MArrD1::<H, _>::merge_cond2(
            &self.h_psi_if_phi1,
            &self.h_b_if_phi1,
            &self.psi.base_rate,
            b,
            h,
        )
        .expect(&format!(
            "failed to merge {:?} & {:?} & {:?} & {:?} & {:?}",
            &self.h_psi_if_phi1, &self.h_b_if_phi1, &self.psi.base_rate, &b, &h
        ));
        MArrD3::new(vec![h_psi_b_if_phi0, h_psi_b_if_phi1])
    }

    fn b_kh_o(&self, b: &MArrD1<B, V>, kh: &MArrD1<KH, V>) -> MArrD2<KH, O, SimplexD1<B, V>> {
        let b_o = self
            .o_b
            .inverse(b, &self.o.base_rate)
            .expect("failed to invert B=>O");
        MArrD1::<B, _>::merge_cond2(&self.b_kh, &b_o, kh, &self.o.base_rate, b).expect(&format!(
            "failed to merge {:?} & {:?} & {:?} & {:?} & {:?}",
            &self.b_kh, &b_o, &kh, &self.o.base_rate, &b
        ))
    }

    fn reset<R: Rng>(
        &mut self,
        simplexes: &InitialSimplexes<V>,
        conditions: &InitialConditions<V>,
        base_rates: &InitialBaseRates<V>,
        rng: &mut R,
    ) where
        Standard: Distribution<V>,
        StandardNormal: Distribution<V>,
        Exp1: Distribution<V>,
        Open01: Distribution<V>,
    {
        let (h_b_if_phi0, h_psi_if_phi0) = {
            let mut x = conditions.params_h_psi_b_if_phi0.sample(rng);
            (
                MArrD1::<B, _>::new(x.pop().unwrap()),
                MArrD1::<Psi, _>::new(x.pop().unwrap()),
            )
        };

        let fh_fpsi_if_fphi0 = {
            let base =
                MArrD1::<FPhi, SimplexD1<FH, _>>::from_fn(|i| h_psi_if_phi0[i].clone().conv());
            let x = conditions
                .params_fh_fpsi_if_fphi0
                .samples1(rng, base.iter());
            MArrD1::<FPsi, _>::new(x)
        };
        let fh_fphi_fpsi = MArrD2::new(vec![fh_fpsi_if_fphi0, conditions.fh_fpsi_if_fphi1.clone()]);

        let kh_kpsi_if_kphi0 = {
            let base =
                MArrD1::<KPhi, SimplexD1<KH, _>>::from_fn(|i| h_psi_if_phi0[i].clone().conv());
            let x = conditions
                .params_kh_kpsi_if_kphi0
                .samples1(rng, base.iter());
            MArrD1::<KPsi, _>::new(x)
        };
        let kh_kphi_kpsi = MArrD2::new(vec![kh_kpsi_if_kphi0, conditions.kh_kpsi_if_kphi1.clone()]);

        *self = Self {
            psi: OpinionRef::from((&simplexes.psi, &base_rates.psi)).cloned(),
            phi: OpinionRef::from((&simplexes.phi, &base_rates.phi)).cloned(),
            o: OpinionRef::from((&simplexes.o, &base_rates.o)).cloned(),
            fpsi: OpinionRef::from((&simplexes.fpsi, &base_rates.fpsi)).cloned(),
            fphi: OpinionRef::from((&simplexes.fphi, &base_rates.fphi)).cloned(),
            kpsi: OpinionRef::from((&simplexes.kpsi, &base_rates.kpsi)).cloned(),
            kphi: OpinionRef::from((&simplexes.kphi, &base_rates.kphi)).cloned(),
            h_psi_if_phi0,
            h_b_if_phi0,
            h_psi_if_phi1: conditions.h_psi_if_phi1.clone(),
            h_b_if_phi1: conditions.h_b_if_phi1.clone(),
            fh_fphi_fpsi,
            kh_kphi_kpsi,
            o_b: MArrD1::from_fn(|i| conditions.o_b[i].sample(rng)),
            b_kh: MArrD1::from_fn(|i| conditions.b_kh[i].sample(rng)),
            a_fh: MArrD1::from_fn(|i| conditions.a_fh[i].sample(rng)),
            theta_h: MArrD1::from_fn(|i| conditions.theta_h[i].sample(rng)),
            thetad_h: MArrD1::from_fn(|i| conditions.thetad_h[i].sample(rng)),
        }
    }
}

impl<V: MyFloat> DeducedOpinions<V> {
    fn deduce(&self, state: &StateOpinions<V>) -> Self {
        let kh = OpinionD2::product2(&state.kphi, &state.kpsi)
            .deduce_with(&state.kh_kphi_kpsi, self.kh.base_rate.clone());
        let b = OpinionD2::product2(&kh, &state.o).deduce_with(
            &state.b_kh_o(&self.b.base_rate, &kh.base_rate),
            self.b.base_rate.clone(),
        );
        let h = OpinionD3::product3(&state.phi, &state.psi, &b).deduce_with(
            &state.h_phi_psi_b(&b.base_rate, &self.h.base_rate),
            self.h.base_rate.clone(),
        );
        let fh = OpinionD2::product2(&state.fphi, &state.fpsi)
            .deduce_with(&state.fh_fphi_fpsi, self.fh.base_rate.clone());
        let a = fh.deduce_with(&state.a_fh, self.a.base_rate.clone());
        let theta = h.deduce_with(&state.theta_h, self.theta.base_rate.clone());
        let thetad = h.deduce_with(&state.thetad_h, self.thetad.base_rate.clone());

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
        self.a = OpinionD1::vacuous_with(base_rates.a.clone());
        self.b = OpinionD1::vacuous_with(base_rates.b.clone());
        self.h = OpinionD1::vacuous_with(base_rates.h.clone());
        self.fh = OpinionD1::vacuous_with(base_rates.fh.clone());
        self.kh = OpinionD1::vacuous_with(base_rates.kh.clone());
        self.theta = OpinionD1::vacuous_with(base_rates.theta.clone());
        self.thetad = OpinionD1::vacuous_with(base_rates.thetad.clone());
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
            DiffOpinions::Observed { o } => {
                mem::swap(&mut state.o, o);
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
                mem::swap(state.fh_fphi_fpsi.down_mut(1), fh_fpsi_if_fphi1);
                mem::swap(state.kh_kphi_kpsi.down_mut(1), kh_kpsi_if_kphi1);
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
            PredDiffOpinions::Inhibition {
                fphi,
                fh_fpsi_if_fphi1,
            } => {
                mem::swap(fphi, &mut state.fphi);
                mem::swap(fh_fpsi_if_fphi1, state.fh_fphi_fpsi.down_mut(1));
            }
            PredDiffOpinions::None => {}
        }
    }
}

#[derive(Default)]
pub struct MyOpinions<V: MyFloat> {
    state: StateOpinions<V>,
    ded: DeducedOpinions<V>,
}

impl<V: MyFloat> MyOpinions<V> {
    pub fn reset<R: Rng>(&mut self, opinions: &InitialOpinions<V>, rng: &mut R)
    where
        Standard: Distribution<V>,
        StandardNormal: Distribution<V>,
        Exp1: Distribution<V>,
        Open01: Distribution<V>,
    {
        self.state.reset(
            &opinions.simplexes,
            &opinions.conditions,
            &opinions.base_rates,
            rng,
        );
        self.ded.reset(&opinions.base_rates);
    }

    pub fn receive<'a>(
        &'a mut self,
        p: &'a InfoContent<V>,
        trusts: Trusts<V>,
    ) -> MyOpinionsUpd<'a, V> {
        let mut diff = self.state.receive(p, &trusts);
        debug!("{:?}", diff);

        diff.swap(&mut self.state);
        self.ded = self.ded.deduce(&self.state);
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
        let mut pred_diff = self.state.predict(p, trusts);
        debug!("{:?}", pred_diff);

        pred_diff.swap(&mut self.state);
        let pred_ded = self.ded.deduce(&self.state);
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
    use std::fs::read_to_string;
    use subjective_logic::{marr_d1, mul::labeled::SimplexD1};

    use super::super::paramter::SimplexDist;
    use super::InitialOpinions;

    #[test]
    fn test_toml() -> anyhow::Result<()> {
        let initial_opinions = toml::from_str::<InitialOpinions<f32>>(&read_to_string(
            "./test/config/test_initial_opinions.toml",
        )?)?;
        assert_eq!(initial_opinions.simplexes.fphi, SimplexD1::vacuous());
        assert!(matches!(
            &initial_opinions.conditions.theta_h[0],
            SimplexDist::Fixed(s) if s.b() == &marr_d1![1.0, 0.0] && s.u() == &0.0,
        ));
        assert!(matches!(
            &initial_opinions.conditions.theta_h[1],
            SimplexDist::Fixed(s) if s.b() == &marr_d1![0.0, 0.7] && s.u() == &0.3,
        ));
        assert!(initial_opinions.base_rates.a == marr_d1![0.999, 0.001]);
        Ok(())
    }
}
