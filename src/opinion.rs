use crate::info::Info;
use approx::UlpsEq;
use core::fmt;
use num_traits::{Float, NumAssign};
use serde_with::{serde_as, TryFromInto};
use std::{iter::Sum, ops::AddAssign};
use subjective_logic::mul::{
    op::{Deduction, Fuse, FuseAssign, FuseOp},
    prod::{HigherArr2, HigherArr3, Product2, Product3},
    Discount, Opinion, Opinion1d, Projection, Simplex,
};

pub const THETA: usize = 2;
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

#[derive(Debug, serde::Deserialize)]
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

#[serde_as]
#[derive(Debug, serde::Deserialize, Clone)]
#[serde(bound(deserialize = "V: serde::Deserialize<'de>"))]
pub struct InitialOpinions<V: Float + AddAssign + UlpsEq> {
    #[serde_as(as = "TryFromInto<([V; PSI], V)>")]
    pub psi: Simplex<V, PSI>,
    #[serde_as(as = "TryFromInto<([V; PHI], V)>")]
    pub phi: Simplex<V, PHI>,
    #[serde_as(as = "TryFromInto<([V; S], V)>")]
    pub s: Simplex<V, S>,
    // #[serde_as(as = "TryFromInto<([V; THETA], V)>")]
    // pub theta: Simplex<V, THETA>,
    // #[serde_as(as = "TryFromInto<([V; THETA], V)>")]
    // pub thetad: Simplex<V, THETA>,
    #[serde_as(as = "TryFromInto<([V; F_S], V)>")]
    pub fs: Simplex<V, F_S>,
    #[serde_as(as = "TryFromInto<([V; F_PHI], V)>")]
    pub fphi: Simplex<V, F_PHI>,
}

#[serde_as]
#[derive(Debug, serde::Deserialize, Clone)]
#[serde(bound(deserialize = "V: serde::Deserialize<'de>"))]
pub struct InitialConditions<V: Float + AddAssign + UlpsEq> {
    #[serde_as(as = "[TryFromInto<([V; THETA], V)>; PHI]")]
    pub cond_theta_phi: [Simplex<V, THETA>; PHI],
    #[serde_as(as = "[TryFromInto<([V; THETA], V)>; PHI]")]
    pub cond_thetad_phi: [Simplex<V, THETA>; PHI],
    // F\Theta | F\Phi
    #[serde_as(as = "[TryFromInto<([V; F_THETA], V)>; F_PHI]")]
    pub cond_ftheta_fphi: [Simplex<V, F_THETA>; F_PHI],

    // P\Psi | S
    #[serde_as(as = "[TryFromInto<([V; P_PSI], V)>; S]")]
    pub cond_ppsi: [Simplex<V, P_PSI>; S],
    // P\Theta | P\Psi
    #[serde_as(as = "[TryFromInto<([V; P_THETA], V)>; P_PSI]")]
    pub cond_ptheta: [Simplex<V, P_THETA>; P_PSI],
    // PA | P\Theta
    #[serde_as(as = "[TryFromInto<([V; P_A], V)>; P_THETA]")]
    pub cond_pa: [Simplex<V, P_A>; P_THETA],
    // F\Psi | S
    #[serde_as(as = "[TryFromInto<([V; F_PSI], V)>; S]")]
    pub cond_fpsi: [Simplex<V, F_PSI>; S],
    /// $\Theta | PA,\Psi$
    #[serde_as(as = "TryFromInto<[[([V; THETA], V); PSI]; P_A]>")]
    pub cond_theta: HigherArr2<Simplex<V, THETA>, P_A, PSI>,
    /// $\Theta' | PA,\Psi,FA$
    #[serde_as(as = "TryFromInto<[[[([V; THETA], V); F_A]; PSI]; P_A]>")]
    pub cond_thetad: HigherArr3<Simplex<V, THETA>, P_A, PSI, F_A>,

    // FP\Theta | FP\Psi
    #[serde_as(as = "[TryFromInto<([V; FP_THETA], V)>; FP_PSI]")]
    pub cond_fptheta: [Simplex<V, FP_THETA>; FP_PSI],
    // FPA | FP\Theta
    #[serde_as(as = "[TryFromInto<([V; FP_A], V)>; FP_THETA]")]
    pub cond_fpa: [Simplex<V, FP_A>; FP_THETA],
    // F\Theta | FPA. F\Psi
    #[serde_as(as = "TryFromInto<[[([V; F_THETA], V); F_PSI]; FP_A]>")]
    pub cond_ftheta: HigherArr2<Simplex<V, F_THETA>, FP_A, F_PSI>,
    // FA | F\Theta
    #[serde_as(as = "[TryFromInto<([V; F_A], V)>; F_THETA]")]
    pub cond_fa: [Simplex<V, F_A>; F_THETA],
    // FP\Psi | FS
    #[serde_as(as = "[TryFromInto<([V; FP_PSI], V)>; F_S]")]
    pub cond_fppsi: [Simplex<V, FP_PSI>; F_S],
}

#[derive(Debug, Default)]
pub struct ConditionalOpinions<V: Float> {
    // \Theta | \Phi
    cond_theta_phi: [Simplex<V, THETA>; PHI],
    // \Theta' | \Phi
    cond_thetad_phi: [Simplex<V, THETA>; PHI],
    // P\Psi | S
    cond_ppsi: [Simplex<V, P_PSI>; S],
    // P\Theta | P\Psi
    cond_ptheta: [Simplex<V, P_THETA>; P_PSI],
    // PA | P\Theta
    cond_pa: [Simplex<V, P_A>; P_THETA],
    // F\Psi | S
    cond_fpsi: [Simplex<V, F_PSI>; S],
    /// $\Theta | PA,\Psi$
    cond_theta: HigherArr2<Simplex<V, THETA>, P_A, PSI>,
    /// $\Theta' | PA,\Psi,FA$
    cond_thetad: HigherArr3<Simplex<V, THETA>, P_A, PSI, F_A>,
    // F\Theta | F\Phi
    cond_ftheta_fphi: [Simplex<V, F_THETA>; F_PHI],
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

#[derive(Debug, Default)]
pub struct Opinions<V: Float> {
    op: BaseOpinions<V>,
    fop: FriendOpinions<V>,
}

#[derive(Debug, Default)]
struct BaseOpinions<V: Float> {
    psi: Opinion1d<V, PSI>,
    phi: Opinion1d<V, PHI>,
    s: Opinion1d<V, S>,
}

#[derive(Debug, Default)]
pub struct FriendOpinions<V: Float> {
    fs: Opinion1d<V, F_S>,
    fphi: Opinion1d<V, F_PHI>,
    // cond_ftheta_fphi: [Simplex<V, F_THETA>; F_PHI],
}

#[derive(Debug)]
pub struct TempOpinions<V: Float> {
    theta: Opinion1d<V, THETA>,
    pa: Opinion1d<V, P_A>,
    fpsi_ded: Opinion1d<V, F_PSI>,
    // fa: Opinion1d<V, F_A>,
    // thetad: Opinion1d<V, THETA>,
}

impl<V: Float> TempOpinions<V> {
    #[inline]
    pub fn get_theta_projection(&self) -> [V; THETA]
    where
        V: NumAssign,
    {
        self.theta.projection()
    }
}

impl<V: Float> BaseOpinions<V> {
    fn new(&self, info: &Info<V>, trust: V) -> Self
    where
        V: UlpsEq + NumAssign,
    {
        let psi = FuseOp::Wgh.fuse(&self.psi, &info.content.psi.discount(trust));
        let phi = FuseOp::Wgh.fuse(&self.phi, &info.content.phi.discount(trust));
        let s = FuseOp::Wgh.fuse(&self.s, &info.content.s.discount(trust));
        // for i in 0..info.content.cond_theta_phi.len() {
        //     FuseOp::Wgh.fuse_assign(
        //         &mut self.fop.cond_ftheta_fphi[i],
        //         &info.content.cond_theta_phi[i].discount(trust),
        //     )
        // }

        Self { psi, phi, s }
    }

    fn compute_pa(
        &self,
        info: &Info<V>,
        trust: V,
        conds: &ConditionalOpinions<V>,
        base_rates: &GlobalBaseRates<V>,
    ) -> Opinion1d<V, P_A>
    where
        V: Float + UlpsEq + NumAssign + Sum + Default,
    {
        let mut pa = self
            .s
            .deduce(&conds.cond_ppsi)
            .unwrap_or_else(|| Opinion::vacuous_with(base_rates.ppsi))
            .deduce(&conds.cond_ptheta)
            .unwrap_or_else(|| Opinion::vacuous_with(base_rates.ptheta))
            .deduce(&conds.cond_pa)
            .unwrap_or_else(|| Opinion::vacuous_with(base_rates.pa));
        // an opinion of PA is computed by using aleatory cummulative fusion.
        FuseOp::ACm.fuse_assign(&mut pa, &info.content.pa.discount(trust));

        pa
    }

    fn compute_theta(
        &self,
        pa: &Opinion1d<V, P_A>,
        conds: &ConditionalOpinions<V>,
        base_rates: &GlobalBaseRates<V>,
    ) -> Opinion1d<V, THETA>
    where
        V: Float + UlpsEq + NumAssign + Sum + Default,
    {
        let mut theta_ded = Opinion::product2(pa.as_ref(), self.psi.as_ref())
            .deduce(&conds.cond_theta)
            .unwrap_or_else(|| Opinion::vacuous_with(base_rates.theta));
        let theta_ded_2 = self
            .phi
            .deduce(&conds.cond_theta_phi)
            .unwrap_or_else(|| Opinion::vacuous_with(base_rates.theta));
        FuseOp::Wgh.fuse_assign(&mut theta_ded, &theta_ded_2);
        theta_ded
    }

    fn compute_thetad(
        &self,
        pa: &Opinion1d<V, P_A>,
        fa: &Opinion1d<V, F_A>,
        conds: &ConditionalOpinions<V>,
        base_rates: &GlobalBaseRates<V>,
    ) -> Opinion1d<V, THETA>
    where
        V: Float + UlpsEq + NumAssign + Sum + Default,
    {
        let thetad_ded_2 = self
            .phi
            .deduce(&conds.cond_thetad_phi)
            .unwrap_or_else(|| Opinion::vacuous_with(base_rates.theta));
        let mut thetad_ded_1 = Opinion::product3(pa.as_ref(), self.psi.as_ref(), fa.as_ref())
            .deduce(&conds.cond_thetad)
            .unwrap_or_else(|| Opinion::vacuous_with(base_rates.theta));
        FuseOp::Wgh.fuse_assign(&mut thetad_ded_1, &thetad_ded_2);
        thetad_ded_1
    }
}

impl<V: Float> FriendOpinions<V> {
    fn new(&self, info: &Info<V>, ft: V) -> Self
    where
        V: UlpsEq + NumAssign,
    {
        // compute friends opinions
        let fs = FuseOp::Wgh.fuse(&self.fs, &info.content.s.discount(ft));
        let fphi = FuseOp::Wgh.fuse(&self.fphi, &info.content.phi.discount(ft));
        // let cond_ftheta_fphi = array::from_fn(|i| {
        //     FuseOp::Wgh.fuse(
        //         &self.cond_ftheta_fphi[i],
        //         &info.content.cond_theta_phi[i].discount(ft),
        //     )
        // });

        Self {
            fs,
            fphi,
            // cond_ftheta_fphi,
        }
    }

    fn compute_fa(
        &self,
        info: &Info<V>,
        fpsi_ded: &Opinion1d<V, F_PSI>,
        ft: V,
        conds: &ConditionalOpinions<V>,
        base_rates: &GlobalBaseRates<V>,
    ) -> Opinion1d<V, F_A>
    where
        V: UlpsEq + NumAssign + Sum + fmt::Debug + Default,
    {
        let fpa = self.compute_fpa(info, ft, conds, base_rates);
        let fpsi = self.compute_fpsi(info, fpsi_ded, ft);
        log::debug!(" w_FPSI:{:?}", fpsi.simplex);
        log::debug!(" w_FPA :{:?}", fpa.simplex);

        let mut ftheta = Opinion::product2(&fpa, &fpsi)
            .deduce(&conds.cond_ftheta)
            .unwrap_or_else(|| Opinion::<_, V>::vacuous_with(base_rates.ftheta));
        let ftheta_ded_2 = self
            .fphi
            .deduce(&conds.cond_ftheta_fphi)
            .unwrap_or_else(|| Opinion::<_, V>::vacuous_with(base_rates.ftheta));
        FuseOp::Wgh.fuse_assign(&mut ftheta, &ftheta_ded_2);

        log::debug!(" w_FTH :{:?}", ftheta.simplex);
        ftheta
            .deduce(&conds.cond_fa)
            .unwrap_or_else(|| Opinion::vacuous_with(base_rates.fa))
    }

    fn compute_fpa(
        &self,
        info: &Info<V>,
        ft: V,
        conds: &ConditionalOpinions<V>,
        base_rates: &GlobalBaseRates<V>,
    ) -> Opinion1d<V, FP_A>
    where
        V: UlpsEq + NumAssign + Sum + fmt::Debug + Default,
    {
        let mut fpa = self
            .fs
            .deduce(&conds.cond_fppsi)
            .unwrap_or_else(|| Opinion::vacuous_with(base_rates.fppsi))
            .deduce(&conds.cond_fptheta)
            .unwrap_or_else(|| Opinion::vacuous_with(base_rates.fptheta))
            .deduce(&conds.cond_fpa)
            .unwrap_or_else(|| Opinion::vacuous_with(base_rates.fpa));

        // compute a FPA opinion by using aleatory cummulative fusion.
        FuseOp::ACm.fuse_assign(&mut fpa, &info.content.pa.discount(ft));

        fpa
    }

    fn compute_fpsi(
        &self,
        info: &Info<V>,
        fpsi_ded: &Opinion1d<V, F_PSI>,
        ft: V,
    ) -> Opinion1d<V, FP_A>
    where
        V: UlpsEq + NumAssign + Sum + fmt::Debug + Default,
    {
        FuseOp::Wgh.fuse(fpsi_ded, &info.content.psi.discount(ft))
    }
}

impl<V: Float> Opinions<V> {
    pub fn reset(&mut self, initial_opinions: InitialOpinions<V>, base_rates: &GlobalBaseRates<V>)
    where
        V: UlpsEq + NumAssign,
    {
        let InitialOpinions {
            psi,
            phi,
            s,
            fs,
            fphi,
        } = initial_opinions;

        // self.op.theta.simplex = theta;
        // self.op.thetad.simplex = thetad;
        // self.op.theta.base_rate = base_rates.theta;
        self.op.psi.simplex = psi;
        self.op.psi.base_rate = base_rates.psi;
        self.op.phi.simplex = phi;
        self.op.phi.base_rate = base_rates.phi;
        self.op.s.simplex = s;
        self.op.s.base_rate = base_rates.s;

        self.fop.fs.simplex = fs;
        self.fop.fs.base_rate = base_rates.fs;
        self.fop.fphi.simplex = fphi;
        self.fop.fphi.base_rate = base_rates.fphi;
    }

    pub fn new(&self, info: &Info<V>, trust: V, ft: V, conds: &ConditionalOpinions<V>) -> Self
    where
        V: UlpsEq + NumAssign + std::fmt::Debug,
    {
        log::debug!("b_PA|PTH_1:{:?}", conds.cond_pa[1].belief);
        // let ft = friend_receipt_prob * friend_read_prob * trust;

        let op = self.op.new(info, trust);
        let fop = self.fop.new(info, ft);

        Self { op, fop }
    }

    pub fn compute(
        &self,
        info: &Info<V>,
        trust: V,
        conds: &ConditionalOpinions<V>,
        base_rates: &GlobalBaseRates<V>,
    ) -> TempOpinions<V>
    where
        V: UlpsEq + NumAssign + Sum + Default + std::fmt::Debug,
    {
        let pa = self.op.compute_pa(info, trust, conds, base_rates);
        log::debug!(" w_PA  :{:?}", pa.simplex);
        let fpsi_ded = self
            .op
            .s
            .deduce(&conds.cond_fpsi)
            .unwrap_or_else(|| Opinion::vacuous_with(base_rates.fpsi));
        let theta = self.op.compute_theta(&pa, conds, base_rates);

        TempOpinions {
            theta,
            pa,
            fpsi_ded,
        }
    }

    pub fn predicate(
        &self,
        temp: &TempOpinions<V>,
        info: &Info<V>,
        ft: V,
        pred_ft: V,
        conds: &ConditionalOpinions<V>,
        base_rates: &GlobalBaseRates<V>,
    ) -> (FriendOpinions<V>, [HigherArr2<V, F_A, THETA>; 2])
    where
        V: UlpsEq + Sum + Default + fmt::Debug + NumAssign,
    {
        // predict new friends' opinions in case of sharing
        let pred_fop = self.fop.new(info, pred_ft);

        // current friends' opinions
        let fa = self
            .fop
            .compute_fa(info, &temp.fpsi_ded, ft, conds, base_rates);
        log::debug!(" w_FA  :{:?}", fa.simplex);
        let thetad = self.op.compute_thetad(&temp.pa, &fa, conds, base_rates);
        let fa_thetad = Opinion::product2(fa.as_ref(), thetad.as_ref());

        log::debug!(" w_TH  :{:?}", temp.theta.simplex);
        log::debug!(" w_TH' :{:?}", thetad.simplex);

        let pred_fa = pred_fop.compute_fa(info, &temp.fpsi_ded, pred_ft, conds, base_rates);
        log::debug!("~w_FA  :{:?}", pred_fa.simplex);
        let pred_thetad = self
            .op
            .compute_thetad(&temp.pa, &pred_fa, conds, base_rates);
        let pred_fa_thetad = Opinion::product2(pred_fa.as_ref(), pred_thetad.as_ref());

        let ps = [fa_thetad.projection(), pred_fa_thetad.projection()];
        log::debug!(" P_FA-TH:{:?}", ps[0]);
        log::debug!("~P_FA-TH:{:?}", ps[1]);

        (pred_fop, ps)
    }

    pub fn replace_pred_fop(&mut self, pred_fop: FriendOpinions<V>) {
        self.fop = pred_fop;
    }
}

impl<V: Float> ConditionalOpinions<V> {
    pub fn reset(&mut self, initial_conds: InitialConditions<V>, pi_rate: V)
    where
        V: Float + UlpsEq + NumAssign + Default,
    {
        let InitialConditions {
            cond_theta_phi,
            cond_thetad_phi,
            cond_ftheta_fphi,
            cond_ppsi,
            cond_ptheta,
            cond_pa,
            cond_fpsi,
            cond_theta,
            cond_thetad,
            cond_fptheta,
            cond_fpa,
            cond_ftheta,
            cond_fa,
            cond_fppsi,
        } = initial_conds;

        self.cond_theta_phi = cond_theta_phi;
        self.cond_thetad_phi = cond_thetad_phi;
        self.cond_ppsi = cond_ppsi;
        self.cond_ptheta = cond_ptheta;
        self.cond_fpsi = cond_fpsi;
        self.cond_theta = cond_theta;
        self.cond_thetad = cond_thetad;

        self.cond_pa = cond_pa;
        let u = self.cond_pa[1].uncertainty;
        let b1 = pi_rate * self.cond_pa[1].belief[1];
        self.cond_pa[1].belief[0] = V::one() - u - b1;
        self.cond_pa[1].belief[1] = b1;
        self.cond_pa[1].uncertainty = u;

        self.cond_ftheta_fphi = cond_ftheta_fphi;
        self.cond_fptheta = cond_fptheta;
        self.cond_ftheta = cond_ftheta;
        self.cond_fa = cond_fa;
        self.cond_fppsi = cond_fppsi;

        self.cond_fpa = cond_fpa;
        let u = self.cond_fpa[1].uncertainty;
        let b1 = pi_rate * self.cond_fpa[1].belief[1];
        self.cond_fpa[1].belief[0] = V::one() - u - b1;
        self.cond_fpa[1].belief[1] = b1;
        self.cond_fpa[1].uncertainty = u;
    }
}
