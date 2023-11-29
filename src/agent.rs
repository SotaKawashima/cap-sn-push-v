use once_cell::sync::Lazy;
use std::{array, ops::Deref};

use crate::cpt::{LevelSet, CPT};
use subjective_logic::mul::{
    op::{Deduction, Fuse, FuseAssign, FuseOp},
    prod::{HigherArr2, HigherArr3, Product2, Product3},
    Discount, Opinion, Opinion1d, OpinionRef, Projection, Simplex,
};

const THETA: usize = 3;
const PSI: usize = 2;
const PHI: usize = 2;
const A: usize = 2;

const THETA_VACUOUS: Lazy<Simplex<f32, THETA>> = Lazy::new(|| Simplex::<f32, THETA>::vacuous());
const A_VACUOUS: Lazy<Simplex<f32, A>> = Lazy::new(|| Simplex::<f32, A>::vacuous());

pub struct InfoContent {
    psi: Opinion1d<f32, PSI>,
    ppsi: Opinion1d<f32, PSI>,
    pa: Opinion1d<f32, A>,
    phi: Opinion1d<f32, PHI>,
    cond_theta_phi: [Simplex<f32, THETA>; PHI],
}

pub struct Info {
    content: InfoContent,
    id: usize,
    num_shared: usize,
}

pub struct Agent<const NUM_INFOS: usize> {
    op: AgentOpinion,
    cpt: CPT,
    selfish: [LevelSet<usize, f32>; 2],
    sharing: [LevelSet<[usize; 2], f32>; 2],
    trusts: [f32; NUM_INFOS],
    rho: f32,
}

impl<const NUM_INFOS: usize> Agent<NUM_INFOS> {
    pub fn new(
        op: AgentOpinion,
        cpt: CPT,
        selfish: [LevelSet<usize, f32>; 2],
        sharing: [LevelSet<[usize; 2], f32>; 2],
        trusts: [f32; NUM_INFOS],
        rho: f32,
    ) -> Self {
        Self {
            op,
            cpt,
            selfish,
            sharing,
            trusts,
            rho,
        }
    }

    pub fn valuate(&self) -> [f32; 2] {
        let p = self.op.theta.projection();
        array::from_fn(|i| self.cpt.valuate(&self.selfish[i], &p))
    }

    fn update_friend(&mut self, info: &Info, ft: f32) -> Option<Opinion1d<f32, A>> {
        FuseOp::Wgh.fuse_assign(&mut self.op.fppsi, &info.content.ppsi.discount(ft));
        let fptheta = self.op.fppsi.deduce(&self.op.cond_fptheta);
        let fpa_fptheta = fptheta
            .as_ref()
            .map(|w| w.as_ref())
            .unwrap_or(OpinionRef::from((
                THETA_VACUOUS.deref(),
                &self.op.br_fptheta,
            )))
            .deduce(&self.op.cond_fpa);
        let fpa = FuseOp::ACm.fuse(
            fpa_fptheta
                .as_ref()
                .map(|w| w.as_ref())
                .unwrap_or(OpinionRef::from((A_VACUOUS.deref(), &self.op.br_fpa))),
            info.content.pa.discount(ft).as_ref(),
        );
        FuseOp::Wgh.fuse_assign(&mut self.op.fpsi, &info.content.psi.discount(ft));
        FuseOp::Wgh.fuse_assign(&mut self.op.fphi, &info.content.phi.discount(ft));
        for i in 0..PHI {
            FuseOp::Wgh.fuse_assign(
                &mut self.op.cond_ftheta_fphi[i],
                &info.content.cond_theta_phi[i].discount(ft),
            )
        }
        let ftheta_fpsi_fpa = Opinion::product2(&fpa, &self.op.fpsi).deduce(&self.op.cond_ftheta);
        let ftheta_fphi = self.op.fphi.deduce(&self.op.cond_ftheta_fphi);
        let ftheta = FuseOp::Wgh.fuse(
            ftheta_fpsi_fpa
                .as_ref()
                .map(|w| w.as_ref())
                .unwrap_or(OpinionRef::from((
                    THETA_VACUOUS.deref(),
                    &self.op.br_ftheta,
                ))),
            ftheta_fphi
                .as_ref()
                .map(|w| w.as_ref())
                .unwrap_or(OpinionRef::from((
                    THETA_VACUOUS.deref(),
                    &self.op.br_ftheta,
                ))),
        );
        return ftheta.deduce(&self.op.cond_fa);
    }

    pub fn update(&mut self, info: &Info, read_prob: f32, kappa: f32) {
        let t = self.trusts[info.id];
        FuseOp::Avg.fuse_assign(&mut self.op.ppsi, &info.content.ppsi.discount(t));
        let ptheta = self.op.ppsi.deduce(&self.op.cond_ptheta);
        let pa_ptheta = ptheta
            .as_ref()
            .map(|w| w.as_ref())
            .unwrap_or(OpinionRef::from((
                THETA_VACUOUS.deref(),
                &self.op.br_ptheta,
            )))
            .deduce(&self.op.cond_pa);
        let pa = FuseOp::ACm.fuse(
            pa_ptheta
                .as_ref()
                .map(|w| w.as_ref())
                .unwrap_or(OpinionRef::from((
                    &Simplex::<f32, A>::vacuous(),
                    &self.op.br_pa,
                ))),
            info.content.pa.discount(t).as_ref(),
        );
        FuseOp::Wgh.fuse_assign(&mut self.op.psi, &info.content.psi.discount(t));
        FuseOp::Wgh.fuse_assign(&mut self.op.phi, &info.content.phi.discount(t));
        for i in 0..PHI {
            FuseOp::Wgh.fuse_assign(
                &mut self.op.cond_ftheta_fphi[i],
                &info.content.cond_theta_phi[i].discount(t),
            )
        }
        let fa_ftheta = self.update_friend(info, read_prob * t);
        let temp = A_VACUOUS;
        let fa = fa_ftheta
            .as_ref()
            .map(|w| w.as_ref())
            .unwrap_or(OpinionRef::from((temp.deref(), &self.op.br_fa)));
        let theta_pa_psi_fa =
            Opinion::product3(pa.as_ref(), self.op.psi.as_ref(), fa).deduce(&self.op.cond_theta);
        let theta_phi = self.op.phi.deduce(&self.op.cond_theta_phi);
        let theta = FuseOp::Wgh.fuse(
            theta_pa_psi_fa
                .as_ref()
                .map(|w| w.as_ref())
                .unwrap_or(OpinionRef::from((THETA_VACUOUS.deref(), &self.op.br_theta))),
            theta_phi
                .as_ref()
                .map(|w| w.as_ref())
                .unwrap_or(OpinionRef::from((THETA_VACUOUS.deref(), &self.op.br_theta))),
        );

        let pred_fa_ftheta = self.update_friend(info, self.rho * kappa * t);
        let pred_fa = pred_fa_ftheta.unwrap_or(Opinion1d::from_simplex_unchecked(
            Simplex::<f32, A>::vacuous(),
            self.op.br_fa.clone(),
        ));
        let pred_theta_pa_psi_fa =
            Opinion::product3(pa.as_ref(), self.op.psi.as_ref(), pred_fa.as_ref())
                .deduce(&self.op.cond_theta);
        let pred_theta = FuseOp::Wgh.fuse(
            pred_theta_pa_psi_fa
                .as_ref()
                .map(|w| w.as_ref())
                .unwrap_or(OpinionRef::from((THETA_VACUOUS.deref(), &self.op.br_theta))),
            theta_phi
                .as_ref()
                .map(|w| w.as_ref())
                .unwrap_or(OpinionRef::from((THETA_VACUOUS.deref(), &self.op.br_theta))),
        );

        let pred_fa_theta = Opinion::product2(&pred_fa, &pred_theta);
        let value_fa: [f32; 2] = array::from_fn(|i| {
            self.cpt
                .valuate(&self.sharing[i], &pred_fa_theta.projection())
        });
        if value_fa[0] < value_fa[1] {
            // do sharing
            println!("do sharing");
            self.op.theta = pred_theta;
        } else {
            self.op.theta = theta;
        }
        let value_theta: [f32; 2] = array::from_fn(|i| {
            self.cpt
                .valuate(&self.selfish[i], &self.op.theta.projection())
        });
        if value_theta[0] < value_theta[1] {
            // do selfish action
            println!("do selfish action");
        }
    }
}

#[derive(Debug)]
pub struct AgentOpinion {
    theta: Opinion1d<f32, THETA>,
    psi: Opinion1d<f32, PSI>,
    phi: Opinion1d<f32, PHI>,
    ppsi: Opinion1d<f32, PSI>,
    fppsi: Opinion1d<f32, PSI>,
    fpsi: Opinion1d<f32, PSI>,
    fphi: Opinion1d<f32, PHI>,
    br_pa: [f32; A],
    br_ptheta: [f32; THETA],
    br_fpa: [f32; A],
    br_fptheta: [f32; THETA],
    br_fa: [f32; A],
    br_ftheta: [f32; THETA],
    br_theta: [f32; THETA],

    // P\Theta | P\Psi
    cond_ptheta: [Simplex<f32, THETA>; PSI],
    // PA | P\Theta
    cond_pa: [Simplex<f32, A>; THETA],
    // \Theta | PA.\Psi,FA
    cond_theta: HigherArr3<Simplex<f32, THETA>, A, PSI, A>,
    // \Theta | \Phi
    cond_theta_phi: [Simplex<f32, THETA>; PHI],

    // FP\Theta | FP\Psi
    cond_fptheta: [Simplex<f32, THETA>; PSI],
    // FPA | P\Theta
    cond_fpa: [Simplex<f32, A>; THETA],
    // F\Theta | PA.\Psi
    cond_ftheta: HigherArr2<Simplex<f32, THETA>, A, PSI>,
    // F\Theta | \Phi
    cond_ftheta_fphi: [Simplex<f32, THETA>; PHI],
    // FA | F\Theta
    cond_fa: [Simplex<f32, A>; THETA],
}

impl AgentOpinion {
    pub fn new(
        theta: Opinion1d<f32, THETA>,
        psi: Opinion1d<f32, PSI>,
        phi: Opinion1d<f32, PHI>,
        ppsi: Opinion1d<f32, PSI>,
        fppsi: Opinion1d<f32, PSI>,
        fpsi: Opinion1d<f32, PSI>,
        fphi: Opinion1d<f32, PHI>,
        br_pa: [f32; A],
        br_ptheta: [f32; THETA],
        br_fpa: [f32; A],
        br_fptheta: [f32; THETA],
        br_fa: [f32; A],
        br_ftheta: [f32; THETA],
        br_theta: [f32; THETA],
        cond_ptheta: [Simplex<f32, THETA>; PSI],
        cond_pa: [Simplex<f32, A>; THETA],
        cond_theta: HigherArr3<Simplex<f32, THETA>, A, PSI, A>,
        cond_theta_phi: [Simplex<f32, THETA>; PHI],
        cond_fptheta: [Simplex<f32, THETA>; PSI],
        cond_fpa: [Simplex<f32, A>; THETA],
        cond_ftheta: HigherArr2<Simplex<f32, THETA>, A, PSI>,
        cond_ftheta_fphi: [Simplex<f32, THETA>; PHI],
        cond_fa: [Simplex<f32, A>; THETA],
    ) -> Self {
        Self {
            theta,
            psi,
            phi,
            ppsi,
            fppsi,
            fpsi,
            fphi,
            br_pa,
            br_ptheta,
            br_fpa,
            br_fptheta,
            br_fa,
            br_ftheta,
            br_theta,
            cond_ptheta,
            cond_pa,
            cond_theta,
            cond_theta_phi,
            cond_fptheta,
            cond_fpa,
            cond_ftheta,
            cond_ftheta_fphi,
            cond_fa,
        }
    }
}
