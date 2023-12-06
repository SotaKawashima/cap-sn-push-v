use once_cell::sync::Lazy;
use rand::Rng;
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

pub struct Prospect {
    selfish: [LevelSet<usize, f32>; 2],
    sharing: [LevelSet<[usize; 2], f32>; 2],
}

pub struct Agent {
    cpt: CPT,
    prospect: Prospect,
    op: AgentOpinion,
    fop: FriendOpinion,
    trusts: Vec<f32>,
    read_prob: f32,
    friend_arrival_prob: f32,
    friend_read_prob: f32,
    done_selfish: bool,
    shared: Vec<bool>,
    br_pa: [f32; A],
    br_ptheta: [f32; THETA],
    // br_theta: [f32; THETA],
    br_fa: [f32; A],
}

impl Agent {
    pub fn new(
        num_info: usize,
        op: AgentOpinion,
        fop: FriendOpinion,
        cpt: CPT,
        selfish: [LevelSet<usize, f32>; 2],
        sharing: [LevelSet<[usize; 2], f32>; 2],
        read_prob: f32,
        friend_arrival_prob: f32,
        friend_read_prob: f32,
        trusts: Vec<f32>,
        br_pa: [f32; A],
        br_ptheta: [f32; THETA],
        br_fa: [f32; A],
    ) -> Self {
        Self {
            cpt,
            prospect: Prospect { selfish, sharing },
            op,
            fop,
            trusts,
            read_prob,
            friend_arrival_prob,
            friend_read_prob,
            done_selfish: false,
            shared: vec![false; num_info],
            br_pa,
            br_ptheta,
            br_fa,
            // br_theta,
        }
    }

    pub fn valuate(&self) -> [f32; 2] {
        let p = self.op.theta.projection();
        array::from_fn(|i| self.cpt.valuate(&self.prospect.selfish[i], &p))
    }

    pub fn try_read<R: Rng>(&self, rng: &mut R) -> bool {
        rng.gen::<f32>() <= self.read_prob
    }

    pub fn read_info(&mut self, info: &Info, receipt_prob: f32) -> Behavior {
        let t = self.trusts[info.id];
        FuseOp::Avg.fuse_assign(&mut self.op.ppsi, &info.content.ppsi.discount(t));
        FuseOp::Wgh.fuse_assign(&mut self.op.psi, &info.content.psi.discount(t));
        FuseOp::Wgh.fuse_assign(&mut self.op.phi, &info.content.phi.discount(t));
        for i in 0..PHI {
            FuseOp::Wgh.fuse_assign(
                &mut self.fop.cond_ftheta_fphi[i],
                &info.content.cond_theta_phi[i].discount(t),
            )
        }
        let a_vacuous = A_VACUOUS;

        let ptheta = self.op.ppsi.deduce(&self.op.cond_ptheta);
        let pa_ptheta = ptheta
            .as_ref()
            .map(|w| w.as_ref())
            .unwrap_or(OpinionRef::from((THETA_VACUOUS.deref(), &self.br_ptheta)))
            .deduce(&self.op.cond_pa);
        let pa_ptheta = pa_ptheta
            .as_ref()
            .map(|w| w.as_ref())
            .unwrap_or(OpinionRef::from((a_vacuous.deref(), &self.br_pa)));
        let pa = FuseOp::ACm.fuse(pa_ptheta, info.content.pa.discount(t).as_ref());

        let (fop, fa) = self
            .fop
            .compute_new_friend_op(info, receipt_prob * self.friend_read_prob * t);
        self.fop.update(fop);

        let fa_ref = fa
            .as_ref()
            .map(|w| w.as_ref())
            .unwrap_or(OpinionRef::from((a_vacuous.deref(), &self.br_fa)));
        let theta_pa_psi_fa = Opinion::product3(pa.as_ref(), self.op.psi.as_ref(), fa_ref.clone())
            .deduce(&self.op.cond_theta);
        let theta_phi = self.op.phi.deduce(&self.op.cond_theta_phi);
        let theta = FuseOp::Wgh.fuse(
            theta_pa_psi_fa
                .as_ref()
                .map(|w| w.as_ref())
                .unwrap_or(OpinionRef::from((
                    THETA_VACUOUS.deref(),
                    &self.op.theta.base_rate,
                ))),
            theta_phi
                .as_ref()
                .map(|w| w.as_ref())
                .unwrap_or(OpinionRef::from((
                    THETA_VACUOUS.deref(),
                    &self.op.theta.base_rate,
                ))),
        );

        let (pred_fop, pred_fa) = self
            .fop
            .compute_new_friend_op(info, self.friend_arrival_prob * self.friend_read_prob * t);

        let pred_fa_ref = pred_fa
            .as_ref()
            .map(|w| w.as_ref())
            .unwrap_or(OpinionRef::from((a_vacuous.deref(), &self.br_fa)));
        let pred_theta_pa_psi_fa =
            Opinion::product3(pa.as_ref(), self.op.psi.as_ref(), pred_fa_ref.clone())
                .deduce(&self.op.cond_theta);
        let pred_theta = FuseOp::Wgh.fuse(
            pred_theta_pa_psi_fa
                .as_ref()
                .map(|w| w.as_ref())
                .unwrap_or(OpinionRef::from((
                    THETA_VACUOUS.deref(),
                    &self.op.theta.base_rate,
                ))),
            theta_phi
                .as_ref()
                .map(|w| w.as_ref())
                .unwrap_or(OpinionRef::from((
                    THETA_VACUOUS.deref(),
                    &self.op.theta.base_rate,
                ))),
        );

        println!("~P_FA = {:?}", pred_fa_ref);
        println!("~P_TH = {:?}", pred_theta);

        let fa_theta = Opinion::product2(fa_ref, theta.as_ref());
        let pred_fa_theta = Opinion::product2(pred_fa_ref, pred_theta.as_ref());

        let value_sharing: [f32; 2] = [
            self.cpt
                .valuate(&self.prospect.sharing[0], &fa_theta.projection()),
            self.cpt
                .valuate(&self.prospect.sharing[1], &pred_fa_theta.projection()),
        ];

        println!(" P_FA,TH = {:?}", fa_theta.projection());
        println!("~P_FA,TH = {:?}", pred_fa_theta.projection());
        println!(" V_S = {:?}", value_sharing);

        let sharing = &mut self.shared[info.id];
        let prev_sharing = *sharing;
        let prev_done_selfish = self.done_selfish;
        if !prev_sharing && value_sharing[0] < value_sharing[1] {
            // sharing info
            self.op.theta = pred_theta;
            self.fop.update(pred_fop);
            *sharing = true;
        } else {
            self.op.theta = theta;
        }
        let value_selfish: [f32; 2] = array::from_fn(|i| {
            self.cpt
                .valuate(&self.prospect.selfish[i], &self.op.theta.projection())
        });
        if !prev_done_selfish && value_selfish[0] < value_selfish[1] {
            // do selfish action
            self.done_selfish = true;
        }

        println!("P_TH = {:?}", self.op.theta.projection());
        println!(" V_A = {:?}", value_selfish);

        Behavior {
            selfish: !prev_done_selfish && self.done_selfish,
            sharing: !prev_sharing && *sharing,
        }
    }
}

#[derive(Debug)]
struct FriendOpinionUpd {
    fppsi: Opinion1d<f32, PSI>,
    fpsi: Opinion1d<f32, PSI>,
    fphi: Opinion1d<f32, PHI>,
    cond_ftheta_fphi: [Simplex<f32, THETA>; PHI],
}

#[derive(Debug)]
pub struct FriendOpinion {
    fppsi: Opinion1d<f32, PSI>,
    fpsi: Opinion1d<f32, PSI>,
    fphi: Opinion1d<f32, PHI>,
    // F\Theta | \Phi
    cond_ftheta_fphi: [Simplex<f32, THETA>; PHI],
    // FP\Theta | FP\Psi
    cond_fptheta: [Simplex<f32, THETA>; PSI],
    // FPA | P\Theta
    cond_fpa: [Simplex<f32, A>; THETA],
    // F\Theta | PA.\Psi
    cond_ftheta: HigherArr2<Simplex<f32, THETA>, A, PSI>,
    // FA | F\Theta
    cond_fa: [Simplex<f32, A>; THETA],

    br_fpa: [f32; A],
    br_fptheta: [f32; THETA],
    br_ftheta: [f32; THETA],
}

impl FriendOpinion {
    pub fn new(
        br_fppsi: [f32; PSI],
        br_fpsi: [f32; PSI],
        br_fphi: [f32; PSI],
        br_fpa: [f32; A],
        br_fptheta: [f32; THETA],
        br_ftheta: [f32; THETA],
    ) -> Self {
        Self {
            fppsi: Opinion::from_simplex_unchecked(Simplex::<f32, PSI>::vacuous(), br_fppsi),
            fpsi: Opinion::from_simplex_unchecked(Simplex::<f32, PSI>::vacuous(), br_fpsi),
            fphi: Opinion::from_simplex_unchecked(Simplex::<f32, PHI>::vacuous(), br_fphi),
            cond_fa: [
                Simplex::<f32, A>::new([0.95, 0.00], 0.05),
                Simplex::<f32, A>::new([0.00, 0.95], 0.05),
                Simplex::<f32, A>::new([0.95, 0.00], 0.05),
            ],
            cond_fpa: [
                Simplex::<f32, A>::new([0.90, 0.00], 0.10),
                Simplex::<f32, A>::new([0.00, 0.90], 0.10),
                Simplex::<f32, A>::new([0.90, 0.00], 0.10),
            ],
            cond_ftheta: harr2![
                [
                    Simplex::<f32, THETA>::new([0.95, 0.00, 0.00], 0.05),
                    Simplex::<f32, THETA>::new([0.00, 0.43, 0.47], 0.10),
                ],
                [
                    Simplex::<f32, THETA>::new([0.00, 0.45, 0.50], 0.05),
                    Simplex::<f32, THETA>::new([0.00, 0.47, 0.52], 0.01),
                ]
            ],
            cond_ftheta_fphi: [
                Simplex::<f32, THETA>::vacuous(),
                Simplex::<f32, THETA>::vacuous(),
            ],
            cond_fptheta: [
                Simplex::<f32, THETA>::new([0.99, 0.00, 0.00], 0.01),
                Simplex::<f32, THETA>::new([0.00, 0.47, 0.52], 0.01),
            ],
            br_fpa,
            br_fptheta,
            br_ftheta,
        }
    }

    fn compute_new_friend_op(
        &self,
        info: &Info,
        ft: f32,
    ) -> (FriendOpinionUpd, Option<Opinion1d<f32, A>>) {
        let fppsi = FuseOp::Wgh.fuse(&self.fppsi, &info.content.ppsi.discount(ft));
        let fpsi = FuseOp::Wgh.fuse(&self.fpsi, &info.content.psi.discount(ft));
        let fphi = FuseOp::Wgh.fuse(&self.fphi, &info.content.phi.discount(ft));
        let cond_ftheta_fphi = array::from_fn(|i| {
            FuseOp::Wgh.fuse(
                &self.cond_ftheta_fphi[i],
                &info.content.cond_theta_phi[i].discount(ft),
            )
        });
        let fop = FriendOpinionUpd {
            fppsi,
            fpsi,
            fphi,
            cond_ftheta_fphi,
        };

        let fptheta = fop.fppsi.deduce(&self.cond_fptheta);
        let fpa_fptheta = fptheta
            .as_ref()
            .map(|w| w.as_ref())
            .unwrap_or(OpinionRef::from((THETA_VACUOUS.deref(), &self.br_fptheta)))
            .deduce(&self.cond_fpa);
        let fpa = FuseOp::ACm.fuse(
            fpa_fptheta
                .as_ref()
                .map(|w| w.as_ref())
                .unwrap_or(OpinionRef::from((A_VACUOUS.deref(), &self.br_fpa))),
            info.content.pa.discount(ft).as_ref(),
        );
        let ftheta_fpsi_fpa = Opinion::product2(&fpa, &fop.fpsi).deduce(&self.cond_ftheta);
        let ftheta_fphi = fop.fphi.deduce(&fop.cond_ftheta_fphi);
        let ftheta = FuseOp::Wgh.fuse(
            ftheta_fpsi_fpa
                .as_ref()
                .map(|w| w.as_ref())
                .unwrap_or(OpinionRef::from((THETA_VACUOUS.deref(), &self.br_ftheta))),
            ftheta_fphi
                .as_ref()
                .map(|w| w.as_ref())
                .unwrap_or(OpinionRef::from((THETA_VACUOUS.deref(), &self.br_ftheta))),
        );
        (fop, ftheta.deduce(&self.cond_fa))
    }

    fn update(&mut self, fop: FriendOpinionUpd) {
        self.fppsi = fop.fppsi;
        self.fpsi = fop.fpsi;
        self.fphi = fop.fphi;
        self.cond_ftheta_fphi = fop.cond_ftheta_fphi;
    }
}

#[derive(Debug)]
pub struct AgentOpinion {
    theta: Opinion1d<f32, THETA>,
    psi: Opinion1d<f32, PSI>,
    phi: Opinion1d<f32, PHI>,
    ppsi: Opinion1d<f32, PSI>,

    // P\Theta | P\Psi
    cond_ptheta: [Simplex<f32, THETA>; PSI],
    // PA | P\Theta
    cond_pa: [Simplex<f32, A>; THETA],
    // \Theta | PA.\Psi,FA
    cond_theta: HigherArr3<Simplex<f32, THETA>, A, PSI, A>,
    // \Theta | \Phi
    cond_theta_phi: [Simplex<f32, THETA>; PHI],
}

impl AgentOpinion {
    pub fn new(
        br_theta: [f32; THETA],
        br_psi: [f32; PSI],
        br_phi: [f32; PSI],
        br_ppsi: [f32; PSI],
    ) -> Self {
        Self {
            theta: Opinion::from_simplex_unchecked(
                Simplex::<f32, THETA>::vacuous(),
                br_theta.clone(),
            ),
            psi: Opinion::from_simplex_unchecked(Simplex::<f32, PSI>::vacuous(), br_psi),
            phi: Opinion::from_simplex_unchecked(Simplex::<f32, PHI>::vacuous(), br_phi),
            ppsi: Opinion::from_simplex_unchecked(Simplex::<f32, PSI>::vacuous(), br_ppsi),
            cond_pa: [
                Simplex::<f32, A>::new([0.90, 0.00], 0.10),
                Simplex::<f32, A>::new([0.00, 0.90], 0.10),
                Simplex::<f32, A>::new([0.90, 0.00], 0.10),
            ],
            cond_theta: harr3![
                [
                    [
                        Simplex::<f32, THETA>::new([0.95, 0.00, 0.00], 0.05),
                        Simplex::<f32, THETA>::new([0.95, 0.00, 0.00], 0.05),
                    ],
                    [
                        Simplex::<f32, THETA>::new([0.00, 0.43, 0.47], 0.10),
                        Simplex::<f32, THETA>::new([0.00, 0.43, 0.47], 0.10),
                    ],
                ],
                [
                    [
                        Simplex::<f32, THETA>::new([0.00, 0.45, 0.50], 0.05),
                        Simplex::<f32, THETA>::new([0.00, 0.45, 0.50], 0.05),
                    ],
                    [
                        Simplex::<f32, THETA>::new([0.00, 0.47, 0.52], 0.01),
                        Simplex::<f32, THETA>::new([0.00, 0.47, 0.52], 0.01),
                    ],
                ]
            ],
            cond_theta_phi: [
                Simplex::<f32, THETA>::vacuous(),
                Simplex::<f32, THETA>::vacuous(),
            ],
            cond_ptheta: [
                Simplex::<f32, THETA>::new([0.99, 0.00, 0.00], 0.01),
                Simplex::<f32, THETA>::new([0.00, 0.47, 0.52], 0.01),
            ],
        }
    }
}
