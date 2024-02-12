use crate::info::Info;
use approx::UlpsEq;
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
pub struct Opinions<V> {
    pub theta: Opinion1d<V, THETA>,
    pub psi: Opinion1d<V, PSI>,
    pub phi: Opinion1d<V, PHI>,
    pub s: Opinion1d<V, S>,
    // \Theta | \Phi
    pub cond_theta_phi: [Simplex<V, THETA>; PHI],
    // P\Psi | S
    pub cond_ppsi: [Simplex<V, P_PSI>; S],
    // P\Theta | P\Psi
    pub cond_ptheta: [Simplex<V, P_THETA>; P_PSI],
    // PA | P\Theta
    pub cond_pa: [Simplex<V, P_A>; P_THETA],
    // F\Psi | S
    pub cond_fpsi: [Simplex<V, F_PSI>; S],
    /// $\Theta | PA.\Psi,FA$
    pub cond_theta: HigherArr3<Simplex<V, THETA>, P_A, PSI, F_A>,
}

pub fn reset_opinions<V>(
    op: &mut Opinions<V>,
    fop: &mut FriendOpinions<V>,
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

    op.theta.simplex = theta;
    op.theta.base_rate = base_rates.theta;
    op.psi.simplex = psi;
    op.psi.base_rate = base_rates.psi;
    op.phi.simplex = phi;
    op.phi.base_rate = base_rates.phi;
    op.s.simplex = s;
    op.s.base_rate = base_rates.s;

    op.cond_theta_phi = cond_theta_phi;
    op.cond_ppsi = cond_ppsi;
    op.cond_ptheta = cond_ptheta;
    op.cond_fpsi = cond_fpsi;
    op.cond_theta = cond_theta;

    op.cond_pa = cond_pa;
    let u = op.cond_pa[1].uncertainty;
    let b1 = pi_rate * op.cond_pa[1].belief[1];
    op.cond_pa[1].belief[0] = V::one() - u - b1;
    op.cond_pa[1].belief[1] = b1;
    op.cond_pa[1].uncertainty = u;

    fop.fs.simplex = fs;
    fop.fs.base_rate = base_rates.fs;
    fop.fphi.simplex = fphi;
    fop.fphi.base_rate = base_rates.fphi;
    fop.cond_ftheta_fphi = cond_ftheta_fphi;

    fop.cond_fptheta = cond_fptheta;
    fop.cond_ftheta = cond_ftheta;
    fop.cond_fa = cond_fa;
    fop.cond_fppsi = cond_fppsi;

    fop.cond_fpa = cond_fpa;
    let u = fop.cond_fpa[1].uncertainty;
    let b1 = pi_rate * fop.cond_fpa[1].belief[1];
    fop.cond_fpa[1].belief[0] = V::one() - u - b1;
    fop.cond_fpa[1].belief[1] = b1;
    fop.cond_fpa[1].uncertainty = u;
}

#[derive(Debug)]
pub struct FriendOpinionsUpd<V: Float> {
    pub fs: Opinion1d<V, F_S>,
    pub fphi: Opinion1d<V, F_PHI>,
    pub cond_ftheta_fphi: [Simplex<V, F_THETA>; F_PHI],
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
    pub fn compute_new_friend_op(
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

        log::debug!(" P_FS  :{:?}", fop.fs.projection());

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
        log::debug!(" P_FPA :{:?}", fpa.projection());
        log::debug!(" P_FTH :{:?}", ftheta.projection());
        (
            fop,
            ftheta
                .deduce(&self.cond_fa)
                .unwrap_or_else(|| Opinion::vacuous_with(base_rates.fa)),
        )
    }

    pub fn update(&mut self, fop: FriendOpinionsUpd<V>) {
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

pub fn compute_opinions<V>(
    op: &mut Opinions<V>,
    fop: &mut FriendOpinions<V>,
    // so: &StaticOpinions<V>,
    // fso: &FriendStaticOpinions<V>,
    info: &Info<V>,
    friend_read_prob: V,
    receipt_prob: V,
    trust: V,
    base_rates: &GlobalBaseRates<V>,
) -> Temporal<V>
where
    V: Float + UlpsEq + NumAssign + Sum + Default + std::fmt::Debug,
{
    log::debug!("b_PA|PTH_1:{:?}", op.cond_pa[1].belief);

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
        let a =
            op.s.deduce(&op.cond_ppsi)
                .unwrap_or_else(|| Opinion::vacuous_with(base_rates.ppsi));
        log::debug!(" S  :{:?}", op.cond_ppsi);
        let mut pa = a
            .deduce(&op.cond_ptheta)
            .unwrap_or_else(|| Opinion::vacuous_with(base_rates.ptheta))
            .deduce(&op.cond_pa)
            .unwrap_or_else(|| Opinion::vacuous_with(base_rates.pa));

        // an opinion of PA is computed by using aleatory cummulative fusion.
        FuseOp::ACm.fuse_assign(&mut pa, &info.content.pa.discount(trust));
        pa
    };

    log::debug!(" P_PA  :{:?}", pa.projection());

    // compute friends opinions
    let fpsi_ded =
        op.s.deduce(&op.cond_fpsi)
            .unwrap_or_else(|| Opinion::vacuous_with(base_rates.fpsi));
    let (fop_upd, fa) = fop.compute_new_friend_op(
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
            .deduce(&op.cond_theta)
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
    V: Float + UlpsEq + NumAssign + Sum + Default + std::fmt::Debug,
{
    pub fn predicate_friend_opinions(
        &mut self,
        op: &Opinions<V>,
        fop: &FriendOpinions<V>,
        info: &Info<V>,
        ft: V,
        base_rates: &GlobalBaseRates<V>,
    ) -> (HigherArr2<V, F_A, THETA>, HigherArr2<V, F_A, THETA>) {
        // predict new friends' opinions in case of sharing
        let (pred_fop, pred_fa) = fop.compute_new_friend_op(info, &self.fpsi_ded, ft, base_rates);
        let pred_theta = {
            let mut pred_theta_ded_1 =
                Opinion::product3(self.pa.as_ref(), op.psi.as_ref(), pred_fa.as_ref())
                    .deduce(&op.cond_theta)
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
