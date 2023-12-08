use rand::Rng;
use rand_distr::{Beta, Distribution};
use std::{array, ops::Deref};
use subjective_logic::{
    harr2, harr3,
    mul::{
        op::{Deduction, Fuse, FuseAssign, FuseOp},
        prod::{HigherArr2, HigherArr3, Product2, Product3},
        Discount, Opinion, Opinion1d, OpinionRef, Projection, Simplex,
    },
};

use crate::{
    cpt::{LevelSet, CPT},
    info::Info,
    opinion::{A, A_VACUOUS, PHI, PSI, THETA, THETA_VACUOUS},
};

fn reset_opinion<const N: usize>(w: &mut Opinion1d<f32, N>, base_rate: &[f32; N]) {
    w.simplex.belief = [0.0; N];
    w.simplex.uncertainty = 1.0;
    w.base_rate = *base_rate;
}

fn reset_simplex<const N: usize, const M: usize>(cond: &mut [Simplex<f32, N>; M]) {
    for i in 0..M {
        cond[i].belief = [0.0; N];
        cond[i].uncertainty = 1.0;
    }
}

pub struct Constants {
    pub br_psi: [f32; PSI],
    pub br_ppsi: [f32; PSI],
    pub br_pa: [f32; A],
    pub br_fa: [f32; A],
    pub br_fpa: [f32; A],
    pub br_phi: [f32; PHI],
    pub br_fpsi: [f32; PSI],
    pub br_fppsi: [f32; PSI],
    pub br_fphi: [f32; PHI],
    pub br_theta: [f32; THETA],
    pub br_ptheta: [f32; THETA],
    pub br_ftheta: [f32; THETA],
    pub br_fptheta: [f32; THETA],
    pub read_dist: Beta<f32>,
    pub fclose_dist: Beta<f32>,
    pub fread_dist: Beta<f32>,
    pub misinfo_trust_dist: Beta<f32>,
}

#[derive(Debug)]
pub struct Behavior {
    pub selfish: bool,
    pub sharing: bool,
}

#[derive(Clone)]
pub struct Prospect {
    selfish: [LevelSet<usize, f32>; 2],
    sharing: [LevelSet<[usize; 2], f32>; 2],
}

#[derive(Clone)]
pub struct Agent {
    cpt: CPT,
    prospect: Prospect,
    pub op: AgentOpinion,
    pub fop: FriendOpinion,
    trusts: Vec<f32>,
    read_prob: f32,
    friend_arrival_prob: f32,
    friend_read_prob: f32,
    done_selfish: bool,
    shared: Vec<bool>,
    br_pa: [f32; A],
    br_ptheta: [f32; THETA],
    br_fa: [f32; A],
}

impl Agent {
    pub fn new(
        op: AgentOpinion,
        fop: FriendOpinion,
        cpt: CPT,
        selfish: [LevelSet<usize, f32>; 2],
        sharing: [LevelSet<[usize; 2], f32>; 2],
        br_pa: [f32; A],
        br_ptheta: [f32; THETA],
        br_fa: [f32; A],
    ) -> Self {
        Self {
            cpt,
            prospect: Prospect { selfish, sharing },
            op,
            fop,
            trusts: Vec::new(),
            read_prob: Default::default(),
            friend_arrival_prob: Default::default(),
            friend_read_prob: Default::default(),
            done_selfish: Default::default(),
            shared: Vec::new(),
            br_pa,
            br_ptheta,
            br_fa,
        }
    }

    pub fn reset(
        &mut self,
        read_prob: f32,
        friend_arrival_prob: f32,
        friend_read_prob: f32,
        trusts: Vec<f32>,
    ) {
        self.read_prob = read_prob;
        self.friend_arrival_prob = friend_arrival_prob;
        self.friend_read_prob = friend_read_prob;
        self.trusts = trusts;
    }

    pub fn reset_with<R: Rng>(&mut self, constants: &Constants, rng: &mut R) {
        self.read_prob = constants.read_dist.sample(rng);
        self.friend_arrival_prob = constants.fclose_dist.sample(rng);
        self.friend_read_prob = constants.fread_dist.sample(rng);
        self.trusts = vec![constants.misinfo_trust_dist.sample(rng)];

        self.op.reset(
            &constants.br_theta,
            &constants.br_psi,
            &constants.br_phi,
            &constants.br_ppsi,
        );
        self.fop
            .reset(&constants.br_fppsi, &constants.br_fpsi, &constants.br_fphi);

        if self.shared.is_empty() {
            self.shared = vec![false; self.trusts.len()];
        } else {
            for i in 0..self.trusts.len() {
                self.shared[i] = false;
            }
        }
        self.done_selfish = false;
    }

    pub fn valuate(&self) -> [f32; 2] {
        let p = self.op.theta.projection();
        array::from_fn(|i| self.cpt.valuate(&self.prospect.selfish[i], &p))
    }

    pub fn read_info_trustfully(&mut self, info: &Info, receipt_prob: f32) -> Behavior {
        self.read_info_with_trust(info, receipt_prob, 1.0)
    }

    pub fn read_info<R: Rng>(
        &mut self,
        rng: &mut R,
        info: &Info,
        receipt_prob: f32,
    ) -> Option<Behavior> {
        if rng.gen::<f32>() <= self.read_prob {
            Some(self.read_info_with_trust(info, receipt_prob, self.trusts[info.id]))
        } else {
            None
        }
    }

    fn read_info_with_trust(&mut self, info: &Info, receipt_prob: f32, t: f32) -> Behavior {
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

        let fa_theta = Opinion::product2(fa_ref, theta.as_ref());
        let pred_fa_theta = Opinion::product2(pred_fa_ref, pred_theta.as_ref());

        let value_sharing: [f32; 2] = [
            self.cpt
                .valuate(&self.prospect.sharing[0], &fa_theta.projection()),
            self.cpt
                .valuate(&self.prospect.sharing[1], &pred_fa_theta.projection()),
        ];

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

#[derive(Clone, Debug)]
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
    pub fn reset(&mut self, br_fppsi: &[f32; PSI], br_fpsi: &[f32; PSI], br_fphi: &[f32; PSI]) {
        reset_opinion(&mut self.fppsi, br_fppsi);
        reset_opinion(&mut self.fpsi, br_fpsi);
        reset_opinion(&mut self.fphi, br_fphi);
        reset_simplex(&mut self.cond_ftheta_fphi);
    }

    pub fn new(br_fpa: [f32; A], br_fptheta: [f32; THETA], br_ftheta: [f32; THETA]) -> Self {
        Self {
            fppsi: Opinion::default(),
            fpsi: Opinion::default(),
            fphi: Opinion::default(),
            cond_ftheta_fphi: [Simplex::default(), Simplex::default()],
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

#[derive(Clone, Debug)]
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
    pub fn reset(
        &mut self,
        br_theta: &[f32; THETA],
        br_psi: &[f32; PSI],
        br_phi: &[f32; PSI],
        br_ppsi: &[f32; PSI],
    ) {
        reset_opinion(&mut self.theta, br_theta);
        reset_opinion(&mut self.psi, br_psi);
        reset_opinion(&mut self.phi, br_phi);
        reset_opinion(&mut self.ppsi, br_ppsi);
        reset_simplex(&mut self.cond_theta_phi);
    }

    pub fn new() -> Self {
        Self {
            theta: Opinion::default(),
            psi: Opinion::default(),
            phi: Opinion::default(),
            ppsi: Opinion::default(),
            cond_theta_phi: [Simplex::default(), Simplex::default()],
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
            cond_ptheta: [
                Simplex::<f32, THETA>::new([0.99, 0.00, 0.00], 0.01),
                Simplex::<f32, THETA>::new([0.00, 0.47, 0.52], 0.01),
            ],
        }
    }
}
