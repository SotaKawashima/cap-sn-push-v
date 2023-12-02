use once_cell::sync::Lazy;
use std::{array, ops::Deref};

use crate::cpt::{LevelSet, CPT};
use subjective_logic::{
    harr2, harr3,
    mul::{
        op::{Deduction, Fuse, FuseAssign, FuseOp},
        prod::{HigherArr2, HigherArr3, Product2, Product3},
        Discount, Opinion, Opinion1d, OpinionRef, Projection, Simplex,
    },
};

pub const THETA: usize = 3;
pub const PSI: usize = 2;
pub const PHI: usize = 2;
pub const A: usize = 2;

const THETA_VACUOUS: Lazy<Simplex<f32, THETA>> = Lazy::new(|| Simplex::<f32, THETA>::vacuous());
const A_VACUOUS: Lazy<Simplex<f32, A>> = Lazy::new(|| Simplex::<f32, A>::vacuous());

pub struct InfoContent {
    psi: Opinion1d<f32, PSI>,
    ppsi: Opinion1d<f32, PSI>,
    pa: Opinion1d<f32, A>,
    phi: Opinion1d<f32, PHI>,
    cond_theta_phi: [Simplex<f32, THETA>; PHI],
}

impl InfoContent {
    pub fn new(
        psi: Opinion1d<f32, PSI>,
        ppsi: Opinion1d<f32, PSI>,
        pa: Opinion1d<f32, A>,
        phi: Opinion1d<f32, PHI>,
        cond_theta_phi: [Simplex<f32, THETA>; PHI],
    ) -> Self {
        Self {
            psi,
            ppsi,
            pa,
            phi,
            cond_theta_phi,
        }
    }
}

pub struct Info {
    id: usize,
    content: InfoContent,
    num_shared: usize,
}

impl Info {
    pub fn new(id: usize, content: InfoContent) -> Self {
        Self {
            id,
            content,
            num_shared: 0,
        }
    }

    pub fn shared(&mut self) {
        self.num_shared += 1;
    }

    #[inline]
    pub fn num_shared(&self) -> usize {
        self.num_shared
    }
}

#[derive(Debug)]
pub struct Behavior {
    pub selfish: bool,
    pub sharing: bool,
}

#[derive(Default)]
pub struct Prospect {
    selfish: [LevelSet<usize, f32>; 2],
    sharing: [LevelSet<[usize; 2], f32>; 2],
}

#[derive(Default)]
pub struct Agent {
    op: AgentOpinion,
    cpt: CPT,
    prospect: Prospect,
    trusts: Vec<f32>,
    arrival_prob: f32,
    read_prob: f32,
    // delay_selfish: usize,
    done_selfish: bool,
    shared: Vec<bool>,
}

// <const NUM_INFOS: usize>
impl Agent {
    pub fn reset(
        &mut self,
        num_info: usize,
        op: AgentOpinion,
        cpt: CPT,
        prospect: Prospect,
        arrival_prob: f32,
        read_prob: f32,
        trusts: Vec<f32>,
    ) {
        self.op = op;
        self.cpt = cpt;
        self.prospect = prospect;
        self.trusts = trusts;
        self.arrival_prob = arrival_prob;
        self.read_prob = read_prob;
        self.done_selfish = false;
        self.shared = vec![false; num_info];
    }

    pub fn valuate(&self) -> [f32; 2] {
        let p = self.op.theta.projection();
        array::from_fn(|i| self.cpt.valuate(&self.prospect.selfish[i], &p))
    }

    fn update_friend_op(&mut self, info: &Info, ft: f32) -> Option<Opinion1d<f32, A>> {
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

    pub fn receive_info(&mut self, info: &Info, receipt_prob: f32) -> Behavior {
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
        let fa_ftheta = self.update_friend_op(info, receipt_prob * self.read_prob * t);
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

        let pred_fa_ftheta = self.update_friend_op(info, self.arrival_prob * self.read_prob * t);
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
                .valuate(&self.prospect.sharing[i], &pred_fa_theta.projection())
        });
        let sharing = &mut self.shared[info.id];
        let prev_sharing = *sharing;
        let prev_done_selfish = self.done_selfish;
        if !prev_sharing && value_fa[0] < value_fa[1] {
            // do sharing
            println!("do sharing");
            self.op.theta = pred_theta;
            *sharing = true;
        } else {
            self.op.theta = theta;
        }
        let value_theta: [f32; 2] = array::from_fn(|i| {
            self.cpt
                .valuate(&self.prospect.selfish[i], &self.op.theta.projection())
        });
        if !prev_done_selfish && value_theta[0] < value_theta[1] {
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

impl Default for AgentOpinion {
    fn default() -> Self {
        Self {
            theta: Opinion::from_simplex_unchecked(
                Simplex::<f32, THETA>::vacuous(),
                [1.0, 0.0, 0.0],
            ),
            psi: Opinion::from_simplex_unchecked(Simplex::<f32, PSI>::vacuous(), [1.0, 0.0]),
            phi: Opinion::from_simplex_unchecked(Simplex::<f32, PHI>::vacuous(), [1.0, 0.0]),
            ppsi: Opinion::from_simplex_unchecked(Simplex::<f32, PSI>::vacuous(), [1.0, 0.0]),
            fppsi: Opinion::from_simplex_unchecked(Simplex::<f32, PSI>::vacuous(), [1.0, 0.0]),
            fpsi: Opinion::from_simplex_unchecked(Simplex::<f32, PSI>::vacuous(), [1.0, 0.0]),
            fphi: Opinion::from_simplex_unchecked(Simplex::<f32, PHI>::vacuous(), [1.0, 0.0]),
            br_pa: [1.0, 0.0],
            br_ptheta: [1.0, 0.0, 0.0],
            br_fpa: [1.0, 0.0],
            br_fptheta: [1.0, 0.0, 0.0],
            br_fa: [1.0, 0.0],
            br_ftheta: [1.0, 0.0, 0.0],
            br_theta: [1.0, 0.0, 0.0],
            cond_ptheta: [
                Simplex::<f32, THETA>::vacuous(),
                Simplex::<f32, THETA>::vacuous(),
            ],
            cond_pa: [
                Simplex::<f32, A>::vacuous(),
                Simplex::<f32, A>::vacuous(),
                Simplex::<f32, A>::vacuous(),
            ],
            cond_theta: harr3![
                [
                    [
                        Simplex::<f32, THETA>::vacuous(),
                        Simplex::<f32, THETA>::vacuous(),
                    ],
                    [
                        Simplex::<f32, THETA>::vacuous(),
                        Simplex::<f32, THETA>::vacuous(),
                    ],
                ],
                [
                    [
                        Simplex::<f32, THETA>::vacuous(),
                        Simplex::<f32, THETA>::vacuous(),
                    ],
                    [
                        Simplex::<f32, THETA>::vacuous(),
                        Simplex::<f32, THETA>::vacuous(),
                    ],
                ]
            ],
            cond_theta_phi: [
                Simplex::<f32, THETA>::vacuous(),
                Simplex::<f32, THETA>::vacuous(),
            ],
            cond_fptheta: [
                Simplex::<f32, THETA>::vacuous(),
                Simplex::<f32, THETA>::vacuous(),
            ],
            cond_fpa: [
                Simplex::<f32, A>::vacuous(),
                Simplex::<f32, A>::vacuous(),
                Simplex::<f32, A>::vacuous(),
            ],
            cond_ftheta: harr2![
                [
                    Simplex::<f32, THETA>::vacuous(),
                    Simplex::<f32, THETA>::vacuous(),
                ],
                [
                    Simplex::<f32, THETA>::vacuous(),
                    Simplex::<f32, THETA>::vacuous(),
                ]
            ],
            cond_ftheta_fphi: [
                Simplex::<f32, THETA>::vacuous(),
                Simplex::<f32, THETA>::vacuous(),
            ],
            cond_fa: [
                Simplex::<f32, A>::vacuous(),
                Simplex::<f32, A>::vacuous(),
                Simplex::<f32, A>::vacuous(),
            ],
        }
    }
}

impl AgentOpinion {
    pub fn reset(
        &mut self,
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
