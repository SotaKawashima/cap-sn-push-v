use crate::info::Info;
use approx::UlpsEq;
use core::fmt;
use num_traits::{Float, NumAssign};
use serde_with::{serde_as, TryFromInto};
use std::{array, iter::Sum, ops::AddAssign};
use subjective_logic::mul::{
    op::{Deduction, Fuse, FuseAssign, FuseOp},
    prod::{HigherArr2, HigherArr3, Product2, Product3},
    Discount, Opinion, Opinion1d, Projection, Simplex,
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
    #[serde_as(as = "TryFromInto<([V; THETA], V)>")]
    pub theta: Simplex<V, THETA>,
    #[serde_as(as = "TryFromInto<([V; PSI], V)>")]
    pub psi: Simplex<V, PSI>,
    #[serde_as(as = "TryFromInto<([V; PHI], V)>")]
    pub phi: Simplex<V, PHI>,
    #[serde_as(as = "TryFromInto<([V; S], V)>")]
    pub s: Simplex<V, S>,
    #[serde_as(as = "[TryFromInto<([V; THETA], V)>; PHI]")]
    pub cond_theta_phi: [Simplex<V, THETA>; PHI],

    #[serde_as(as = "TryFromInto<([V; F_S], V)>")]
    pub fs: Simplex<V, F_S>,
    #[serde_as(as = "TryFromInto<([V; F_PHI], V)>")]
    pub fphi: Simplex<V, F_PHI>,
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
    /// $\Theta | PA.\Psi,FA$
    #[serde_as(as = "TryFromInto<[[[([V; THETA], V); F_A]; PSI]; P_A]>")]
    pub cond_theta: HigherArr3<Simplex<V, THETA>, P_A, PSI, F_A>,

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
pub struct Opinions<V: Float> {
    op: BaseOpinions<V>,
    fop: FriendOpinions<V>,
}

#[derive(Debug, Default)]
struct BaseOpinions<V: Float> {
    theta: Opinion1d<V, THETA>,
    psi: Opinion1d<V, PSI>,
    phi: Opinion1d<V, PHI>,
    s: Opinion1d<V, S>,
    // \Theta | \Phi
    cond_theta_phi: [Simplex<V, THETA>; PHI],
    // P\Psi | S
    cond_ppsi: [Simplex<V, P_PSI>; S],
    // P\Theta | P\Psi
    cond_ptheta: [Simplex<V, P_THETA>; P_PSI],
    // PA | P\Theta
    cond_pa: [Simplex<V, P_A>; P_THETA],
    // F\Psi | S
    cond_fpsi: [Simplex<V, F_PSI>; S],
    /// $\Theta | PA.\Psi,FA$
    cond_theta: HigherArr3<Simplex<V, THETA>, P_A, PSI, F_A>,
}

#[derive(Debug)]
struct FriendOpinionsUpd<V: Float> {
    fs: Opinion1d<V, F_S>,
    fphi: Opinion1d<V, F_PHI>,
    cond_ftheta_fphi: [Simplex<V, F_THETA>; F_PHI],
}

#[derive(Debug, Default)]
pub struct FriendOpinions<V: Float> {
    pub fs: Opinion1d<V, F_S>,
    pub fphi: Opinion1d<V, F_PHI>,
    // F\Theta | F\Phi
    pub cond_ftheta_fphi: [Simplex<V, F_THETA>; F_PHI],

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

impl<V: Float> FriendOpinions<V> {
    fn compute_new_friend_op(
        &self,
        info: &Info<V>,
        fpsi_ded: &Opinion1d<V, F_PSI>,
        ft: V,
        base_rates: &GlobalBaseRates<V>,
    ) -> (FriendOpinionsUpd<V>, Opinion1d<V, F_A>)
    where
        V: Float + UlpsEq + NumAssign + Sum + Default + std::fmt::Debug,
    {
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

        log::debug!(" w_FPSI:{:?}", fpsi.simplex);

        let fpa = {
            let mut fpa = fop
                .fs
                .deduce(&self.cond_fppsi)
                .unwrap_or_else(|| Opinion::vacuous_with(base_rates.fppsi))
                .deduce(&self.cond_fptheta)
                .unwrap_or_else(|| Opinion::vacuous_with(base_rates.fptheta))
                .deduce(&self.cond_fpa)
                .unwrap_or_else(|| Opinion::vacuous_with(base_rates.fpa));

            // compute a FPA opinion by using aleatory cummulative fusion.
            FuseOp::ACm.fuse_assign(&mut fpa, &info.content.pa.discount(ft));
            fpa
        };

        let ftheta = {
            let mut ftheta_ded_1 = Opinion::product2(&fpa, &fpsi)
                .deduce(&self.cond_ftheta)
                .unwrap_or_else(|| Opinion::<_, V>::vacuous_with(base_rates.ftheta));
            let ftheta_ded_2 = fop
                .fphi
                .deduce(&self.cond_ftheta_fphi)
                .unwrap_or_else(|| Opinion::<_, V>::vacuous_with(base_rates.ftheta));
            FuseOp::Wgh.fuse_assign(&mut ftheta_ded_1, &ftheta_ded_2);
            ftheta_ded_1
        };
        log::debug!(" w_FPA :{:?}", fpa.simplex);
        log::debug!(" w_FTH :{:?}", ftheta.simplex);
        (
            fop,
            ftheta
                .deduce(&self.cond_fa)
                .unwrap_or_else(|| Opinion::vacuous_with(base_rates.fa)),
        )
    }

    fn update(&mut self, fop: FriendOpinionsUpd<V>) {
        self.fs = fop.fs;
        self.fphi = fop.fphi;
        self.cond_ftheta_fphi = fop.cond_ftheta_fphi;
    }
}

pub struct Temporal<V: Float> {
    fa: Opinion1d<V, F_A>,
    theta: Opinion1d<V, THETA>,
    fpsi_ded: Opinion1d<V, F_PSI>,
    theta_ded_2: Opinion1d<V, THETA>,
    pa: Opinion1d<V, P_A>,
    pred: Option<PredTemporal<V>>,
}

struct PredTemporal<V: Float> {
    pred_theta: Opinion1d<V, THETA>,
    pred_fop: FriendOpinionsUpd<V>,
}

impl<V> Temporal<V> where V: Float + UlpsEq + NumAssign + Sum + Default + std::fmt::Debug {}

impl<V: Float> Opinions<V> {
    pub fn reset_opinions(
        &mut self,
        pi_rate: V,
        initial_opinions: InitialOpinions<V>,
        base_rates: &GlobalBaseRates<V>,
    ) where
        V: Float + UlpsEq + NumAssign + Default,
    {
        let InitialOpinions {
            theta,
            psi,
            phi,
            s,
            cond_theta_phi,
            fs,
            fphi,
            cond_ftheta_fphi,
            cond_ppsi,
            cond_ptheta,
            cond_pa,
            cond_fpsi,
            cond_theta,
            cond_fptheta,
            cond_fpa,
            cond_ftheta,
            cond_fa,
            cond_fppsi,
        } = initial_opinions;

        self.op.theta.simplex = theta;
        self.op.theta.base_rate = base_rates.theta;
        self.op.psi.simplex = psi;
        self.op.psi.base_rate = base_rates.psi;
        self.op.phi.simplex = phi;
        self.op.phi.base_rate = base_rates.phi;
        self.op.s.simplex = s;
        self.op.s.base_rate = base_rates.s;

        self.op.cond_theta_phi = cond_theta_phi;
        self.op.cond_ppsi = cond_ppsi;
        self.op.cond_ptheta = cond_ptheta;
        self.op.cond_fpsi = cond_fpsi;
        self.op.cond_theta = cond_theta;

        self.op.cond_pa = cond_pa;
        let u = self.op.cond_pa[1].uncertainty;
        let b1 = pi_rate * self.op.cond_pa[1].belief[1];
        self.op.cond_pa[1].belief[0] = V::one() - u - b1;
        self.op.cond_pa[1].belief[1] = b1;
        self.op.cond_pa[1].uncertainty = u;

        self.fop.fs.simplex = fs;
        self.fop.fs.base_rate = base_rates.fs;
        self.fop.fphi.simplex = fphi;
        self.fop.fphi.base_rate = base_rates.fphi;
        self.fop.cond_ftheta_fphi = cond_ftheta_fphi;

        self.fop.cond_fptheta = cond_fptheta;
        self.fop.cond_ftheta = cond_ftheta;
        self.fop.cond_fa = cond_fa;
        self.fop.cond_fppsi = cond_fppsi;

        self.fop.cond_fpa = cond_fpa;
        let u = self.fop.cond_fpa[1].uncertainty;
        let b1 = pi_rate * self.fop.cond_fpa[1].belief[1];
        self.fop.cond_fpa[1].belief[0] = V::one() - u - b1;
        self.fop.cond_fpa[1].belief[1] = b1;
        self.fop.cond_fpa[1].uncertainty = u;
    }

    pub fn compute_opinions(
        &mut self,
        info: &Info<V>,
        friend_read_prob: V,
        friend_receipt_prob: V,
        trust: V,
        base_rates: &GlobalBaseRates<V>,
    ) -> Temporal<V>
    where
        V: UlpsEq + NumAssign + Sum + Default + std::fmt::Debug,
    {
        log::debug!("b_PA|PTH_1:{:?}", self.op.cond_pa[1].belief);

        FuseOp::Wgh.fuse_assign(&mut self.op.s, &info.content.s.discount(trust));
        FuseOp::Wgh.fuse_assign(&mut self.op.psi, &info.content.psi.discount(trust));
        FuseOp::Wgh.fuse_assign(&mut self.op.phi, &info.content.phi.discount(trust));
        for i in 0..info.content.cond_theta_phi.len() {
            FuseOp::Wgh.fuse_assign(
                &mut self.fop.cond_ftheta_fphi[i],
                &info.content.cond_theta_phi[i].discount(trust),
            )
        }

        let pa = {
            let a = self
                .op
                .s
                .deduce(&self.op.cond_ppsi)
                .unwrap_or_else(|| Opinion::vacuous_with(base_rates.ppsi));
            let mut pa = a
                .deduce(&self.op.cond_ptheta)
                .unwrap_or_else(|| Opinion::vacuous_with(base_rates.ptheta))
                .deduce(&self.op.cond_pa)
                .unwrap_or_else(|| Opinion::vacuous_with(base_rates.pa));

            // an opinion of PA is computed by using aleatory cummulative fusion.
            FuseOp::ACm.fuse_assign(&mut pa, &info.content.pa.discount(trust));
            pa
        };

        log::debug!(" w_PA  :{:?}", pa.simplex);

        // compute friends opinions
        let fpsi_ded = self
            .op
            .s
            .deduce(&self.op.cond_fpsi)
            .unwrap_or_else(|| Opinion::vacuous_with(base_rates.fpsi));
        let (fop_upd, fa) = self.fop.compute_new_friend_op(
            info,
            &fpsi_ded,
            friend_receipt_prob * friend_read_prob * trust,
            base_rates,
        );

        log::debug!(" w_FA  :{:?}", fa.simplex);
        self.fop.update(fop_upd);

        let theta_ded_2 = self
            .op
            .phi
            .deduce(&self.op.cond_theta_phi)
            .unwrap_or_else(|| Opinion::vacuous_with(self.op.theta.base_rate));

        let theta = {
            let mut theta_ded_1 = Opinion::product3(pa.as_ref(), self.op.psi.as_ref(), fa.as_ref())
                .deduce(&self.op.cond_theta)
                .unwrap_or_else(|| Opinion::vacuous_with(self.op.theta.base_rate));
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

    pub fn predicate_friend_opinions(
        &self,
        temp: &mut Temporal<V>,
        info: &Info<V>,
        ft: V,
        base_rates: &GlobalBaseRates<V>,
    ) -> [HigherArr2<V, F_A, THETA>; 2]
    where
        V: UlpsEq + Sum + Default + fmt::Debug + NumAssign,
    {
        // predict new friends' opinions in case of sharing
        let (pred_fop, pred_fa) =
            self.fop
                .compute_new_friend_op(info, &temp.fpsi_ded, ft, base_rates);
        let pred_theta = {
            let mut pred_theta_ded_1 =
                Opinion::product3(temp.pa.as_ref(), self.op.psi.as_ref(), pred_fa.as_ref())
                    .deduce(&self.op.cond_theta)
                    .unwrap_or_else(|| Opinion::vacuous_with(self.op.theta.base_rate));
            FuseOp::Wgh.fuse_assign(&mut pred_theta_ded_1, &temp.theta_ded_2);
            pred_theta_ded_1
        };

        log::debug!("~w_FA  :{:?}", pred_fa.simplex);

        let fa_theta = Opinion::product2(temp.fa.as_ref(), temp.theta.as_ref());
        let pred_fa_theta = Opinion::product2(pred_fa.as_ref(), pred_theta.as_ref());
        temp.pred = Some(PredTemporal {
            pred_theta,
            pred_fop,
        });
        [fa_theta.projection(), pred_fa_theta.projection()]
    }

    pub fn update_for_sharing(&mut self, temp: Temporal<V>) {
        match temp.pred {
            // sharing info
            Some(PredTemporal {
                pred_theta,
                pred_fop,
            }) => {
                self.op.theta = pred_theta;
                self.fop.update(pred_fop);
            }
            None => {
                self.op.theta = temp.theta;
            }
        }
    }

    #[inline]
    pub fn get_theta_projection(&self) -> [V; THETA]
    where
        V: NumAssign,
    {
        self.op.theta.projection()
    }
}
