use std::mem;

use subjective_logic::{
    domain::DomainConv,
    iter::FromFn,
    mul::labeled::{OpinionD1, OpinionD2, OpinionD3, SimplexD1},
    multi_array::labeled::{MArrD1, MArrD2, MArrD3},
    ops::{Deduction, Discount, Fuse, FuseAssign, FuseOp, Product2, Product3, Projection},
};

use super::{FPhi, FPsi, KPhi, KPsi, MyFloat, Phi, Psi, Theta, Thetad, A, B, FH, H, KH, O};

pub enum InfoContent<V: MyFloat> {
    Misinfo {
        op: SimplexD1<Psi, V>,
    },
    Correction {
        op: SimplexD1<Psi, V>,
        misinfo: SimplexD1<Psi, V>,
        trust_misinfo: V,
    },
    Observation {
        op: SimplexD1<O, V>,
    },
    Inhibition {
        op1: SimplexD1<Phi, V>,
        op2: MArrD3<Phi, Psi, B, SimplexD1<H, V>>,
    },
}

#[derive(Default)]
struct StateOpinions<V: MyFloat> {
    psi: OpinionD1<Psi, V>,
    phi: OpinionD1<Phi, V>,
    o: OpinionD1<O, V>,
    h_phi_psi_b: MArrD3<Phi, Psi, B, SimplexD1<H, V>>,
    fpsi: OpinionD1<FPsi, V>,
    fphi: OpinionD1<FPhi, V>,
    fh_fphi_fpsi: MArrD2<FPhi, FPsi, SimplexD1<FH, V>>,
    kpsi: OpinionD1<KPsi, V>,
    kphi: OpinionD1<KPhi, V>,
    kh_kphi_kpsi: MArrD2<KPhi, KPsi, SimplexD1<KH, V>>,
    b_kh_o: MArrD2<KH, O, SimplexD1<B, V>>,
    a_fh: MArrD1<FH, SimplexD1<A, V>>,
    theta_h: MArrD1<H, SimplexD1<Theta, V>>,
    thetad_h: MArrD1<H, SimplexD1<Thetad, V>>,
}

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
        h_phi_psi_b: MArrD3<Phi, Psi, B, SimplexD1<H, V>>,
        fh_fphi_fpsi: MArrD2<FPhi, FPsi, SimplexD1<FH, V>>,
        kh_kphi_kpsi: MArrD2<KPhi, KPsi, SimplexD1<KH, V>>,
    },
}

enum PredDiffOpinions<V: MyFloat> {
    Causal {
        fpsi: OpinionD1<FPsi, V>,
    },
    Inhibition {
        fphi: OpinionD1<FPhi, V>,
        fh_fphi_fpsi: MArrD2<FPhi, FPsi, SimplexD1<FH, V>>,
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
    pub mis_friend: V,
    pub mis_social: V,
    pub pred_friend: V,
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
                FuseOp::Wgh.fuse_assign(&mut fpsi, &misinfo.discount(trusts.mis_friend).conv());
                let mut kpsi = FuseOp::Wgh.fuse(&self.kpsi, &op.discount(trusts.social).conv());
                FuseOp::Wgh.fuse_assign(&mut kpsi, &misinfo.discount(trusts.mis_social).conv());
                DiffOpinions::Causal { psi, fpsi, kpsi }
            }
            InfoContent::Observation { op } => {
                let op = op.discount(trusts.info);
                let o = FuseOp::Wgh.fuse(&self.o, &op);
                DiffOpinions::Observed { o }
            }
            InfoContent::Inhibition { op1, op2 } => {
                let op1 = op1.discount(trusts.info);
                let phi = FuseOp::Wgh.fuse(&self.phi, &op1);
                let fphi = FuseOp::Wgh.fuse(&self.fphi, &op1.discount(trusts.friend).conv());
                let kphi = FuseOp::Wgh.fuse(&self.kphi, &op1.discount(trusts.social).conv());
                let op2_iter = op2.iter().map(|o| o.discount(trusts.info));

                let h_phi_psi_b = MArrD3::from_iter(
                    self.h_phi_psi_b
                        .iter()
                        .zip(op2_iter.clone())
                        .map(|(c, o)| FuseOp::Wgh.fuse(c, &o)),
                );
                let fh_fphi_fpsi = MArrD2::from_fn(|(i_phi, i_psi)| {
                    let c = MArrD1::<B, _>::from_fn(|i_b| {
                        op2[(i_phi, i_psi, i_b)]
                            .discount(trusts.info * trusts.friend)
                            .conv()
                    });
                    let w = OpinionD1::vacuous_with(ded.b.base_rate.clone())
                        .deduce_with(&c, ded.fh.base_rate.clone());
                    FuseOp::Wgh.fuse(&self.fh_fphi_fpsi[(i_phi, i_psi)], &w.simplex)
                });
                let kh_kphi_kpsi = MArrD2::from_fn(|(i_phi, i_psi)| {
                    let c = MArrD1::<B, _>::from_fn(|i_b| {
                        op2[(i_phi, i_psi, i_b)]
                            .discount(trusts.info * trusts.friend)
                            .conv()
                    });
                    let w = OpinionD1::vacuous_with(ded.b.base_rate.clone())
                        .deduce_with(&c, ded.kh.base_rate.clone());
                    FuseOp::Wgh.fuse(&self.kh_kphi_kpsi[(i_phi, i_psi)], &w.simplex)
                });
                DiffOpinions::Inhibition {
                    phi,
                    fphi,
                    kphi,
                    h_phi_psi_b,
                    fh_fphi_fpsi,
                    kh_kphi_kpsi,
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
                    &op.discount(trusts.info * trusts.pred_friend).conv(),
                );
                PredDiffOpinions::Causal { fpsi }
            }
            InfoContent::Correction { op, misinfo, .. } => {
                let mut fpsi = FuseOp::Wgh.fuse(
                    &self.fpsi,
                    &op.discount(trusts.info * trusts.pred_friend).conv(),
                );
                FuseOp::Wgh.fuse_assign(&mut fpsi, &misinfo.discount(trusts.mis_friend).conv());
                PredDiffOpinions::Causal { fpsi }
            }
            InfoContent::Inhibition { op1, op2 } => {
                let fphi = FuseOp::Wgh.fuse(
                    &self.fphi,
                    &op1.discount(trusts.info * trusts.pred_friend).conv(),
                );
                let fh_fphi_fpsi = MArrD2::from_fn(|(i_phi, i_psi)| {
                    let c = MArrD1::<B, _>::from_fn(|i_b| {
                        op2[(i_phi, i_psi, i_b)]
                            .discount(trusts.info * trusts.friend)
                            .conv()
                    });
                    let w = OpinionD1::vacuous_with(ded.b.base_rate.clone())
                        .deduce_with(&c, ded.fh.base_rate.clone());
                    FuseOp::Wgh.fuse(&self.fh_fphi_fpsi[(i_phi, i_psi)], &w.simplex)
                });
                PredDiffOpinions::Inhibition { fphi, fh_fphi_fpsi }
            }
            _ => PredDiffOpinions::None,
        }
    }
}

impl<V: MyFloat> DeducedOpinions<V> {
    fn deduce(&self, state: &StateOpinions<V>) -> Self {
        let kh = OpinionD2::product2(&state.kphi, &state.kpsi)
            .deduce_with(&state.kh_kphi_kpsi, self.kh.base_rate.clone());
        let b =
            OpinionD2::product2(&kh, &state.o).deduce_with(&state.b_kh_o, self.b.base_rate.clone());
        let h = OpinionD3::product3(&state.phi, &state.psi, &b)
            .deduce_with(&state.h_phi_psi_b, self.h.base_rate.clone());
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
                h_phi_psi_b,
                fh_fphi_fpsi,
                kh_kphi_kpsi,
            } => {
                mem::swap(&mut state.phi, phi);
                mem::swap(&mut state.fphi, fphi);
                mem::swap(&mut state.kphi, kphi);
                mem::swap(&mut state.h_phi_psi_b, h_phi_psi_b);
                mem::swap(&mut state.fh_fphi_fpsi, fh_fphi_fpsi);
                mem::swap(&mut state.kh_kphi_kpsi, kh_kphi_kpsi);
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
            PredDiffOpinions::Inhibition { fphi, fh_fphi_fpsi } => {
                mem::swap(fphi, &mut state.fphi);
                mem::swap(fh_fphi_fpsi, &mut state.fh_fphi_fpsi);
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
    pub fn receive<F: FnMut(&DeducedOpinions<V>, &DeducedOpinions<V>) -> bool>(
        &mut self,
        p: &InfoContent<V>,
        trusts: Trusts<V>,
        mut decide: F,
    ) {
        let mut diff = self.state.receive(p, &trusts, &self.ded);
        diff.swap(&mut self.state);
        self.ded = self.ded.deduce(&self.state);

        let mut pred_diff = self.state.predict(p, &trusts, &self.ded);
        pred_diff.swap(&mut self.state);
        let pred_ded = self.ded.deduce(&self.state);

        if decide(&self.ded, &pred_ded) {
            self.ded = pred_ded;
        } else {
            pred_diff.swap(&mut self.state);
        }
    }
}
