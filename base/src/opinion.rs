use std::{fmt::Debug, iter::Sum, mem};

use approx::UlpsEq;
use num_traits::{Float, FromPrimitive, NumAssign, ToPrimitive};
use rand_distr::{uniform::SampleUniform, Distribution, Exp1, Open01, Standard, StandardNormal};
use subjective_logic::{
    domain::{Domain, DomainConv, Keys},
    iter::FromFn,
    mul::{
        labeled::{OpinionD1, OpinionD2, OpinionD3, OpinionRefD1, SimplexD1},
        InverseCondition, MergeJointConditions2,
    },
    multi_array::labeled::{MArrD1, MArrD2, MArrD3},
    new_type_domain,
    ops::{Deduction, Discount, Fuse, FuseAssign, FuseOp, Product2, Product3, Projection},
};
use tracing::debug;

use crate::info::InfoContent;

pub trait MyFloat
where
    Self: Float
        + NumAssign
        + UlpsEq
        + Sum
        + FromPrimitive
        + ToPrimitive
        + SampleUniform
        + Default
        + Send
        + Sync
        + Debug,
{
}

impl MyFloat for f32 {}
impl MyFloat for f64 {}

new_type_domain!(pub Psi = 2);
new_type_domain!(pub FPsi from Psi);
new_type_domain!(pub KPsi from Psi);
new_type_domain!(pub Phi = 2);
new_type_domain!(pub FPhi from Phi);
new_type_domain!(pub KPhi from Phi);
new_type_domain!(pub O = 2);
new_type_domain!(pub FO from O);
new_type_domain!(pub KO from O);
new_type_domain!(pub A = 2);
new_type_domain!(pub B = 2);
new_type_domain!(pub H = 2);
new_type_domain!(pub FH from H);
new_type_domain!(pub KH from H);
new_type_domain!(pub Theta from H);
new_type_domain!(pub Thetad from H);

#[derive(Default, Debug)]
pub struct StateOpinions<V> {
    pub psi: OpinionD1<Psi, V>,
    pub phi: OpinionD1<Phi, V>,
    pub o: OpinionD1<O, V>,
    pub fo: OpinionD1<FO, V>,
    pub ko: OpinionD1<KO, V>,
    pub h_psi_if_phi1: MArrD1<Psi, SimplexD1<H, V>>,
    pub h_b_if_phi1: MArrD1<B, SimplexD1<H, V>>,
    pub fpsi: OpinionD1<FPsi, V>,
    pub fphi: OpinionD1<FPhi, V>,
    pub fh_fpsi_if_fphi1: MArrD1<FPsi, SimplexD1<FH, V>>,
    pub kpsi: OpinionD1<KPsi, V>,
    pub kphi: OpinionD1<KPhi, V>,
    pub kh_kpsi_if_kphi1: MArrD1<KPsi, SimplexD1<KH, V>>,
}

#[derive(Default, Debug)]
pub struct FixedOpinions<V> {
    pub o_b: MArrD1<B, SimplexD1<O, V>>,
    pub b_kh: MArrD1<KH, SimplexD1<B, V>>,
    pub a_fh: MArrD1<FH, SimplexD1<A, V>>,
    pub theta_h: MArrD1<H, SimplexD1<Theta, V>>,
    pub thetad_h: MArrD1<H, SimplexD1<Thetad, V>>,
    pub h_psi_if_phi0: MArrD1<Psi, SimplexD1<H, V>>,
    pub h_b_if_phi0: MArrD1<B, SimplexD1<H, V>>,
    pub uncertainty_fh_fpsi_if_fphi0: MArrD1<FPsi, V>,
    pub uncertainty_kh_kpsi_if_kphi0: MArrD1<KPsi, V>,
    pub uncertainty_fh_fphi_fo: MArrD2<FPhi, FO, V>,
    pub uncertainty_kh_kphi_ko: MArrD2<KPhi, KO, V>,
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
pub struct DeducedOpinions<V> {
    pub h: OpinionD1<H, V>,
    pub fh: OpinionD1<FH, V>,
    pub kh: OpinionD1<KH, V>,
    pub a: OpinionD1<A, V>,
    pub b: OpinionD1<B, V>,
    pub theta: OpinionD1<Theta, V>,
    pub thetad: OpinionD1<Thetad, V>,
}

#[derive(Debug)]
pub struct Trusts<V> {
    // pub m: V,
    pub my_trust: V,
    pub friend_trusts: OtherTrusts<V>,
    pub social_trusts: OtherTrusts<V>,
    pub friend_misinfo_trusts: OtherTrusts<V>,
    pub social_misinfo_trusts: OtherTrusts<V>,
    pub pred_friend_trusts: OtherTrusts<V>,
}

fn transform<
    D1: Domain<Idx: Copy> + Keys<D1::Idx>,
    D2: Domain<Idx: Debug> + From<D1> + Keys<D2::Idx>,
    V: MyFloat,
>(
    w: OpinionRefD1<D1, V>,
    certainty: V,
) -> OpinionD1<D2, V> {
    let p: MArrD1<D2, _> = w.projection().conv();
    OpinionD1::new(
        MArrD1::from_fn(|i| p[i] * certainty),
        V::one() - certainty,
        w.base_rate.clone().conv(),
    )
}

fn transform_simplex<
    D1: Domain<Idx: Copy> + Keys<D1::Idx>,
    D2: Domain<Idx: Debug> + From<D1> + Keys<D2::Idx>,
    V: MyFloat,
>(
    p: MArrD1<D1, V>,
    c: V,
) -> SimplexD1<D2, V> {
    let q: MArrD1<D2, _> = p.conv();
    SimplexD1::new_unchecked(MArrD1::from_fn(|i| q[i] * c), V::one() - c)
}

#[derive(Debug)]
pub struct OtherTrusts<V> {
    pub trust: V,
    pub certainty: V,
}

impl<V> OtherTrusts<V> {
    fn approximate<D1, D2, O>(&self, w: O) -> OpinionD1<D2, V>
    where
        V: MyFloat,
        D1: Domain<Idx: Copy> + Keys<D1::Idx>,
        O: AsRef<OpinionD1<D1, V>>,
        D2: Domain<Idx: Debug> + From<D1> + Keys<D2::Idx>,
    {
        transform(w.as_ref().discount(self.trust).as_ref(), self.certainty)
    }
    fn approximate_simplex<D1, D2>(
        &self,
        s: &SimplexD1<D1, V>,
        base_rate: &MArrD1<D1, V>,
    ) -> SimplexD1<D2, V>
    where
        V: MyFloat,
        D1: Domain<Idx: Copy> + Keys<D1::Idx>,
        D2: Domain<Idx: Debug> + From<D1> + Keys<D2::Idx>,
    {
        transform_simplex(s.discount(self.trust).projection(base_rate), self.certainty)
    }
}

impl<V: MyFloat> StateOpinions<V> {
    pub fn reset(
        &mut self,
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
    ) where
        Standard: Distribution<V>,
        StandardNormal: Distribution<V>,
        Exp1: Distribution<V>,
        Open01: Distribution<V>,
    {
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

    fn receive(
        &self,
        p: &InfoContent<V>,
        trusts: &Trusts<V>,
        // ap: &AccessProb<V>,
        ded: &DeducedOpinions<V>,
    ) -> DiffOpinions<V> {
        match p {
            InfoContent::Misinfo { op } => {
                let psi = FuseOp::Wgh.fuse(&self.psi, &op.discount(trusts.my_trust));
                let fpsi = FuseOp::Wgh.fuse(&self.fpsi, &trusts.friend_trusts.approximate(op));
                let kpsi = FuseOp::Wgh.fuse(&self.kpsi, &trusts.social_trusts.approximate(op));
                DiffOpinions::Causal { psi, fpsi, kpsi }
            }
            InfoContent::Correction { op, misinfo } => {
                let psi = FuseOp::Wgh.fuse(&self.psi, &op.discount(trusts.my_trust));
                let mut fpsi = FuseOp::Wgh.fuse(&self.fpsi, &trusts.friend_trusts.approximate(op));
                FuseOp::Wgh.fuse_assign(
                    &mut fpsi,
                    &trusts.friend_misinfo_trusts.approximate(misinfo),
                );
                let mut kpsi = FuseOp::Wgh.fuse(&self.kpsi, &trusts.social_trusts.approximate(op));
                FuseOp::Wgh.fuse_assign(
                    &mut kpsi,
                    &trusts.social_misinfo_trusts.approximate(misinfo),
                );
                DiffOpinions::Causal { psi, fpsi, kpsi }
            }
            InfoContent::Observation { op } => {
                let o = FuseOp::ACm.fuse(&self.o, &op.discount(trusts.my_trust));
                let fo = FuseOp::Wgh.fuse(&self.fo, &trusts.friend_trusts.approximate(op));
                let ko = FuseOp::Wgh.fuse(&self.ko, &trusts.social_trusts.approximate(op));
                DiffOpinions::Observed { o, fo, ko }
            }
            InfoContent::Inhibition { op1, op2, op3 } => {
                let phi = FuseOp::Wgh.fuse(&self.phi, &op1.discount(trusts.my_trust));
                let fphi = FuseOp::Wgh.fuse(&self.fphi, &trusts.friend_trusts.approximate(op1));
                let kphi = FuseOp::Wgh.fuse(&self.kphi, &trusts.social_trusts.approximate(op1));

                let h_psi_if_phi1 = MArrD1::from_iter(
                    self.h_psi_if_phi1
                        .iter()
                        .zip(op2.iter().map(|o| o.discount(trusts.my_trust)))
                        .map(|(c, o)| FuseOp::Wgh.fuse(c, &o)),
                );
                let h_b_if_phi1 = MArrD1::from_iter(
                    self.h_b_if_phi1
                        .iter()
                        .zip(op3.iter().map(|o| o.discount(trusts.my_trust)))
                        .map(|(c, o)| FuseOp::Wgh.fuse(c, &o)),
                );
                let ah = &ded.h.base_rate;
                let fh_fpsi_if_fphi1 = MArrD1::from_iter(
                    self.fh_fpsi_if_fphi1
                        .iter()
                        .zip(op2.iter())
                        // .zip(op2.iter().map(|o| o.discount(trusts.fp)))
                        .map(|(c, s)| {
                            FuseOp::Wgh.fuse(c, &trusts.friend_trusts.approximate_simplex(s, ah))
                        }),
                );
                let kh_kpsi_if_kphi1 = MArrD1::from_iter(
                    self.kh_kpsi_if_kphi1.iter().zip(op2.iter()).map(|(c, s)| {
                        FuseOp::Wgh.fuse(c, &trusts.social_trusts.approximate_simplex(s, ah))
                    }),
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

    fn predict(
        &self,
        p: &InfoContent<V>,
        trusts: &Trusts<V>,
        ded: &DeducedOpinions<V>,
    ) -> PredDiffOpinions<V> {
        match p {
            InfoContent::Misinfo { op } => {
                let fpsi = FuseOp::Wgh.fuse(&self.fpsi, &trusts.pred_friend_trusts.approximate(op));
                PredDiffOpinions::Causal { fpsi }
            }
            InfoContent::Observation { op } => {
                let fo = FuseOp::Wgh.fuse(&self.fo, &trusts.pred_friend_trusts.approximate(op));
                PredDiffOpinions::Observed { fo }
            }
            InfoContent::Correction { op, .. } => {
                let fpsi = FuseOp::Wgh.fuse(&self.fpsi, &trusts.pred_friend_trusts.approximate(op));
                PredDiffOpinions::Causal { fpsi }
            }
            InfoContent::Inhibition { op1, op2, .. } => {
                let fphi =
                    FuseOp::Wgh.fuse(&self.fphi, &trusts.pred_friend_trusts.approximate(op1));
                let ah = &ded.h.base_rate;
                let fh_fpsi_if_fphi1 = MArrD1::from_iter(
                    self.fh_fpsi_if_fphi1.iter().zip(op2.iter()).map(|(c, s)| {
                        FuseOp::Wgh.fuse(c, &trusts.pred_friend_trusts.approximate_simplex(s, ah))
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

    pub fn reset(
        &mut self,
        o_b: MArrD1<B, SimplexD1<O, V>>,
        b_kh: MArrD1<KH, SimplexD1<B, V>>,
        a_fh: MArrD1<FH, SimplexD1<A, V>>,
        theta_h: MArrD1<H, SimplexD1<Theta, V>>,
        thetad_h: MArrD1<H, SimplexD1<Thetad, V>>,
        h_psi_if_phi0: MArrD1<Psi, SimplexD1<H, V>>,
        h_b_if_phi0: MArrD1<B, SimplexD1<H, V>>,
        uncertainty_fh_fpsi_if_fphi0: MArrD1<FPsi, V>,
        uncertainty_kh_kpsi_if_kphi0: MArrD1<KPsi, V>,
        uncertainty_fh_fphi_fo: MArrD2<FPhi, FO, V>,
        uncertainty_kh_kphi_ko: MArrD2<KPhi, KO, V>,
    ) {
        *self = Self {
            o_b,
            b_kh,
            a_fh,
            theta_h,
            thetad_h,
            h_psi_if_phi0,
            h_b_if_phi0,
            uncertainty_fh_fpsi_if_fphi0,
            uncertainty_kh_kpsi_if_kphi0,
            uncertainty_fh_fphi_fo,
            uncertainty_kh_kphi_ko,
        }
    }
}

fn deduce_fh<V: MyFloat>(
    state: &StateOpinions<V>,
    fixed: &FixedOpinions<V>,
    b_o: &MArrD1<O, SimplexD1<B, V>>,
    base_rate_b: &MArrD1<B, V>,
    base_rate_h: &MArrD1<H, V>,
    base_rate_fh: &MArrD1<FH, V>,
) -> OpinionD1<FH, V> {
    let fh_fpsi_if_fphi0 = MArrD1::<FPsi, _>::from_fn(|fpsi| {
        transform_simplex::<_, FH, _>(
            fixed.h_psi_if_phi0[fpsi.into()].projection(base_rate_h),
            V::one() - fixed.uncertainty_fh_fpsi_if_fphi0[fpsi.into()],
        )
    });
    let h_o_if_phi0 = MArrD1::<O, _>::from_fn(|o| {
        OpinionRefD1::<_, V>::from((&b_o[o], base_rate_b))
            .deduce_with(&fixed.h_b_if_phi0, || base_rate_h.clone())
    });
    debug!(target: "H|O,phi0", Cond=?h_o_if_phi0);

    let fh_fo_if_fphi0 = MArrD1::<FO, _>::from_fn(|fo| {
        transform_simplex::<_, FH, _>(
            h_o_if_phi0[fo.into()].projection(),
            // OpinionRefD1::from((&b_o[fo.into()], base_rate_b))
            //     .deduce_with(&fixed.h_b_if_phi0, || base_rate_h.clone())
            //     .projection(),
            V::one() - fixed.uncertainty_fh_fphi_fo[(FPhi(0), fo)],
        )
    });

    let fh_fo_if_fphi1 = MArrD1::<FO, _>::from_fn(|fo| {
        transform_simplex::<_, FH, _>(
            OpinionRefD1::from((&b_o[fo.into()], base_rate_b))
                .deduce_with(&state.h_b_if_phi1, || base_rate_h.clone())
                .projection(),
            V::one() - fixed.uncertainty_fh_fphi_fo[(FPhi(1), fo)],
        )
    });
    let fh_fpsi_fo_if_fphi0 = MArrD1::<FH, _>::merge_cond2(
        &fh_fpsi_if_fphi0,
        &fh_fo_if_fphi0,
        &state.fpsi.base_rate,
        &state.fo.base_rate,
        base_rate_fh,
    );
    let fh_fpsi_fo_if_fphi1 = MArrD1::<FH, _>::merge_cond2(
        &state.fh_fpsi_if_fphi1,
        &fh_fo_if_fphi1,
        &state.fpsi.base_rate,
        &state.fo.base_rate,
        base_rate_fh,
    );

    debug!(target: "FPsi", a=?state.fpsi.base_rate);
    debug!(target: "  FO", a=?state.fo.base_rate);
    debug!(target: "  FH", a=?base_rate_fh);
    debug!(target: "FH|Fphi0,FPsi", Cond=?fh_fpsi_if_fphi0);
    debug!(target: "FH|Fphi0,  FO", Cond=?fh_fo_if_fphi0);
    debug!(target: "FH|Fphi0,FPsi,FO", Cond=?fh_fpsi_fo_if_fphi0);
    debug!(target: "FH|Fphi1,FPsi", Cond=?state.fh_fpsi_if_fphi1);
    debug!(target: "FH|Fphi1,  FO", Cond=?fh_fo_if_fphi1);
    debug!(target: "FH|Fphi1,FPsi,FO", Cond=?fh_fpsi_fo_if_fphi1);

    let fh_fphi_fpsi_fo =
        MArrD3::<FPhi, FPsi, FO, _>::new(vec![fh_fpsi_fo_if_fphi0, fh_fpsi_fo_if_fphi1]);
    let fh = OpinionD3::product3(&state.fphi, &state.fpsi, &state.fo)
        .deduce_with(&fh_fphi_fpsi_fo, || base_rate_fh.clone());

    debug!(target:"FH", w=?fh);
    fh
}

fn deduce_kh<V: MyFloat>(
    state: &StateOpinions<V>,
    fixed: &FixedOpinions<V>,
    b_o: &MArrD1<O, SimplexD1<B, V>>,
    base_rate_b: &MArrD1<B, V>,
    base_rate_h: &MArrD1<H, V>,
    base_rate_kh: &MArrD1<KH, V>,
) -> OpinionD1<KH, V> {
    let kh_kpsi_if_kphi0 = MArrD1::<KPsi, _>::from_fn(|kpsi| {
        transform_simplex::<_, KH, _>(
            fixed.h_psi_if_phi0[kpsi.into()].projection(base_rate_h),
            V::one() - fixed.uncertainty_kh_kpsi_if_kphi0[kpsi],
        )
    });
    let kh_ko_if_kphi0 = MArrD1::<KO, _>::from_fn(|ko| {
        transform_simplex::<_, KH, _>(
            OpinionRefD1::from((&b_o[ko.into()], base_rate_b))
                .deduce_with(&fixed.h_b_if_phi0, || base_rate_h.clone())
                .projection(),
            V::one() - fixed.uncertainty_kh_kphi_ko[(KPhi(0), ko)],
        )
    });
    let kh_ko_if_kphi1 = MArrD1::<KO, _>::from_fn(|ko| {
        transform_simplex::<_, KH, _>(
            OpinionRefD1::from((&b_o[ko.into()], base_rate_b))
                .deduce_with(&state.h_b_if_phi1, || base_rate_h.clone())
                .projection(),
            V::one() - fixed.uncertainty_kh_kphi_ko[(KPhi(1), ko)],
        )
    });
    debug!(target: "KH|Kphi0,KPsi", Cond=?kh_kpsi_if_kphi0);
    debug!(target: "KH|Kphi0,  KO", Cond=?kh_ko_if_kphi0);
    debug!(target: "KH|Kphi1,KPsi", Cond=?state.kh_kpsi_if_kphi1);
    debug!(target: "KH|Kphi1,  KO", Cond=?kh_ko_if_kphi1);
    let kh_kpsi_ko_if_kphi0 = MArrD1::<KH, _>::merge_cond2(
        &kh_kpsi_if_kphi0,
        &kh_ko_if_kphi0,
        &state.kpsi.base_rate,
        &state.ko.base_rate,
        base_rate_kh,
    );
    let kh_kpsi_ko_if_kphi1 = MArrD1::<KH, _>::merge_cond2(
        &state.kh_kpsi_if_kphi1,
        &kh_ko_if_kphi1,
        &state.kpsi.base_rate,
        &state.ko.base_rate,
        base_rate_kh,
    );

    let kh_kphi_kpsi_ko =
        MArrD3::<KPhi, KPsi, KO, _>::new(vec![kh_kpsi_ko_if_kphi0, kh_kpsi_ko_if_kphi1]);
    let kh = OpinionD3::product3(&state.kphi, &state.kpsi, &state.ko)
        .deduce_with(&kh_kphi_kpsi_ko, || base_rate_kh.clone());

    debug!(target:"KH", w=?kh);
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
    pub fn reset(
        &mut self,
        h: OpinionD1<H, V>,
        fh: OpinionD1<FH, V>,
        kh: OpinionD1<KH, V>,
        a: OpinionD1<A, V>,
        b: OpinionD1<B, V>,
        theta: OpinionD1<Theta, V>,
        thetad: OpinionD1<Thetad, V>,
    ) {
        *self = Self {
            a,
            b,
            fh,
            h,
            kh,
            theta,
            thetad,
        }
    }

    fn deduce(&self, state: &StateOpinions<V>, fixed: &FixedOpinions<V>) -> Self {
        let ab = &self.b.base_rate;
        let b_o = fixed.o_b.inverse(ab, &state.o.base_rate);
        debug!(target: "B|O", Cond=?b_o);
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

#[derive(Default, Debug)]
pub struct MyOpinions<V> {
    pub state: StateOpinions<V>,
    pub ded: DeducedOpinions<V>,
    pub fixed: FixedOpinions<V>,
}

impl<V> MyOpinions<V>
where
    V: MyFloat,
{
    pub fn receive<'a>(
        &'a mut self,
        p: &'a InfoContent<V>,
        trusts: Trusts<V>,
    ) -> MyOpinionsUpd<'a, V> {
        debug!("{:?}", &trusts);
        debug!("before: {:?}", &self.state);

        let mut diff = self.state.receive(p, &trusts, &self.ded);
        diff.swap(&mut self.state);
        self.ded = self.ded.deduce(&self.state, &self.fixed);

        debug!("after: {:?}", &self.state);
        debug!("after: {:?}", &self.ded);

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
    p: &'a InfoContent<'a, V>,
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
    use std::iter::Sum;
    use subjective_logic::domain::DomainConv;
    use subjective_logic::iter::FromFn;
    use subjective_logic::mul::labeled::OpinionD1;
    use subjective_logic::mul::{mbr, MergeJointConditions2};
    use subjective_logic::multi_array::labeled::MArrD1;
    use subjective_logic::ops::{Deduction, Discount, FuseAssign, FuseOp, Projection};
    use subjective_logic::{marr_d1, mul::labeled::SimplexD1};

    use super::{transform, FPsi, MyFloat, Psi, B, FH, H};

    #[allow(dead_code)]
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

    #[allow(dead_code)]
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
            let cf = MArrD1::<FPsi, _>::from_fn(|fpsi| {
                let psi = fpsi.into();
                SimplexD1::<FH, _>::new(
                    MArrD1::from_iter(
                        c[psi]
                            .projection(&mah)
                            .iter()
                            .map(|p| *p * (V::one() - *c[psi].u())),
                    ),
                    *c[psi].u(),
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
