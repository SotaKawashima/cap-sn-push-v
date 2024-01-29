use log::debug;
use rand::Rng;
use rand_distr::{Beta, Distribution};
use std::array;
use subjective_logic::{
    harr2, harr3,
    mul::{
        op::{Deduction, Fuse, FuseAssign, FuseOp},
        prod::{HigherArr2, HigherArr3, Product2, Product3},
        Discount, Opinion, Opinion1d, Projection, Simplex,
    },
};

use crate::{
    cpt::{LevelSet, CPT},
    info::{Info, InfoType},
    opinion::{
        A, FP_A, FP_PSI, FP_THETA, F_A, F_PHI, F_PSI, F_S, F_THETA, PHI, PSI, P_A, P_PSI, P_THETA,
        S, THETA,
    },
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
    pub br_s: [f32; S],
    pub br_fs: [f32; S],
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
    pub pi_prob: f32,
    pub pi_dist: Beta<f32>,
    pub misinfo_trust_dist: Beta<f32>,
    pub correction_trust_dist: Beta<f32>,
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

#[derive(Clone, Default)]
pub struct ParamsForInfo {
    trust: f32,
    shared: bool,
}

pub struct Agent {
    cpt: CPT,
    prospect: Prospect,
    op: AgentOpinion,
    fop: FriendOpinion,
    read_prob: f32,
    friend_arrival_prob: f32,
    friend_read_prob: f32,
    done_selfish: bool,
    params_for_info: Vec<ParamsForInfo>,
}

impl Agent {
    pub fn new(
        op: AgentOpinion,
        fop: FriendOpinion,
        cpt: CPT,
        selfish: [LevelSet<usize, f32>; 2],
        sharing: [LevelSet<[usize; 2], f32>; 2],
        info_count: usize,
    ) -> Self {
        Self {
            cpt,
            prospect: Prospect { selfish, sharing },
            op,
            fop,
            read_prob: Default::default(),
            friend_arrival_prob: Default::default(),
            friend_read_prob: Default::default(),
            done_selfish: Default::default(),
            params_for_info: Vec::with_capacity(info_count),
        }
    }

    pub fn reset<F: FnMut(&InfoType) -> f32>(
        &mut self,
        read_prob: f32,
        friend_arrival_prob: f32,
        friend_read_prob: f32,
        pi_rate: f32,
        mut trust_map: F,
        constants: &Constants,
        info_types: &[InfoType],
    ) {
        self.read_prob = read_prob;
        self.friend_arrival_prob = friend_arrival_prob;
        self.friend_read_prob = friend_read_prob;

        self.op.reset(
            pi_rate,
            &constants.br_theta,
            &constants.br_psi,
            &constants.br_phi,
            &constants.br_s,
        );
        self.fop
            .reset(pi_rate, &constants.br_fs, &constants.br_fphi);

        self.params_for_info.clear();
        for i in 0..self.params_for_info.capacity() {
            self.params_for_info.push(ParamsForInfo {
                trust: trust_map(&info_types[i]),
                shared: false,
            });
        }
        self.done_selfish = false;
    }

    pub fn reset_with<R: Rng>(
        &mut self,
        constants: &Constants,
        info_types: &[InfoType],
        rng: &mut R,
    ) {
        let r = rng.gen::<f32>();
        let s = constants.pi_dist.sample(rng);

        self.reset(
            constants.read_dist.sample(rng),
            constants.fclose_dist.sample(rng),
            constants.fread_dist.sample(rng),
            if constants.pi_prob > r { s } else { 0.0 },
            |_| constants.misinfo_trust_dist.sample(rng),
            constants,
            info_types,
        );
    }

    pub fn valuate(&self) -> [f32; 2] {
        let p = self.op.theta.projection();
        array::from_fn(|i| self.cpt.valuate(&self.prospect.selfish[i], &p))
    }

    pub fn read_info_trustfully(
        &mut self,
        info: &Info,
        receipt_prob: f32,
        constants: &Constants,
    ) -> Behavior {
        self.read_info_with_trust(info, receipt_prob, 1.0, constants)
    }

    pub fn read_info<R: Rng>(
        &mut self,
        rng: &mut R,
        info: &Info,
        receipt_prob: f32,
        constants: &Constants,
    ) -> Option<Behavior> {
        if rng.gen::<f32>() <= self.read_prob {
            Some(self.read_info_with_trust(
                info,
                receipt_prob,
                self.params_for_info[info.id].trust,
                constants,
            ))
        } else {
            None
        }
    }

    fn read_info_with_trust(
        &mut self,
        info: &Info,
        receipt_prob: f32,
        t: f32,
        constants: &Constants,
    ) -> Behavior {
        debug!("b_PA|PTH_1:{:?}", self.op.cond_pa[1].belief);

        FuseOp::Wgh.fuse_assign(&mut self.op.s, &info.content.s.discount(t));
        FuseOp::Wgh.fuse_assign(&mut self.op.psi, &info.content.psi.discount(t));
        FuseOp::Wgh.fuse_assign(&mut self.op.phi, &info.content.phi.discount(t));
        for i in 0..PHI {
            FuseOp::Wgh.fuse_assign(
                &mut self.fop.cond_ftheta_fphi[i],
                &info.content.cond_theta_phi[i].discount(t),
            )
        }

        let pa = {
            let mut pa = self
                .op
                .s
                .deduce(&self.op.cond_ppsi)
                .unwrap_or_else(|| Opinion::<_, f32>::vacuous_with(constants.br_ppsi))
                .deduce(&self.op.cond_ptheta)
                .unwrap_or_else(|| Opinion::<_, f32>::vacuous_with(constants.br_ptheta))
                .deduce(&self.op.cond_pa)
                .unwrap_or_else(|| Opinion::<_, f32>::vacuous_with(constants.br_pa));

            // an opinion of PA is computed by using aleatory cummulative fusion.
            FuseOp::ACm.fuse_assign(&mut pa, &info.content.pa.discount(t));
            pa
        };

        debug!(" P_PA  :{:?}", pa.projection());

        // compute friends opinions
        let fpsi_ded = self
            .op
            .s
            .deduce(&self.op.cond_fpsi)
            .unwrap_or_else(|| Opinion::<_, f32>::vacuous_with(constants.br_fpsi));
        let (fop, fa) = self.fop.compute_new_friend_op(
            info,
            &fpsi_ded,
            receipt_prob * self.friend_read_prob * t,
            constants,
        );
        self.fop.update(fop);

        let theta_ded_2 = self
            .op
            .phi
            .deduce(&self.op.cond_theta_phi)
            .unwrap_or_else(|| Opinion::<_, f32>::vacuous_with(self.op.theta.base_rate));

        let theta = {
            let mut theta_ded_1 = Opinion::product3(pa.as_ref(), self.op.psi.as_ref(), fa.as_ref())
                .deduce(&self.op.cond_theta)
                .unwrap_or_else(|| Opinion::<_, f32>::vacuous_with(self.op.theta.base_rate));
            FuseOp::Wgh.fuse_assign(&mut theta_ded_1, &theta_ded_2);
            theta_ded_1
        };

        // predict new friends' opinions in case of sharing
        let (pred_fop, pred_fa) = self.fop.compute_new_friend_op(
            info,
            &fpsi_ded,
            self.friend_arrival_prob * self.friend_read_prob * t,
            constants,
        );
        let pred_theta = {
            let mut pred_theta_ded_1 =
                Opinion::product3(pa.as_ref(), self.op.psi.as_ref(), pred_fa.as_ref())
                    .deduce(&self.op.cond_theta)
                    .unwrap_or_else(|| Opinion::<_, f32>::vacuous_with(self.op.theta.base_rate));
            FuseOp::Wgh.fuse_assign(&mut pred_theta_ded_1, &theta_ded_2);
            pred_theta_ded_1
        };

        debug!(" P_FA  :{:?}", fa.projection());
        debug!("~P_FA  :{:?}", pred_fa.projection());
        debug!(" P_TH  :{:?}", theta.projection());
        debug!("~P_TH  :{:?}", pred_theta.projection());

        // compute values of prospects
        let fa_theta = Opinion::product2(fa.as_ref(), theta.as_ref());
        let pred_fa_theta = Opinion::product2(pred_fa.as_ref(), pred_theta.as_ref());

        let value_sharing: [f32; 2] = [
            self.cpt
                .valuate(&self.prospect.sharing[0], &fa_theta.projection()),
            self.cpt
                .valuate(&self.prospect.sharing[1], &pred_fa_theta.projection()),
        ];

        let sharing = &mut self.params_for_info[info.id].shared;
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

        debug!("V_X,V_Y:{:?},{:?}", value_selfish, value_sharing);

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
    fs: Opinion1d<f32, F_S>,
    fphi: Opinion1d<f32, F_PHI>,
    cond_ftheta_fphi: [Simplex<f32, F_THETA>; F_PHI],
}

#[derive(Debug)]
pub struct FriendOpinion {
    fs: Opinion1d<f32, F_S>,
    fphi: Opinion1d<f32, F_PHI>,

    // F\Theta | F\Phi
    cond_ftheta_fphi: [Simplex<f32, F_THETA>; F_PHI],
    // FP\Theta | FP\Psi
    cond_fptheta: [Simplex<f32, FP_THETA>; FP_PSI],
    // FPA | FP\Theta
    cond_fpa: [Simplex<f32, FP_A>; FP_THETA],
    // F\Theta | FPA. F\Psi
    cond_ftheta: HigherArr2<Simplex<f32, F_THETA>, FP_A, F_PSI>,
    // FA | F\Theta
    cond_fa: [Simplex<f32, F_A>; F_THETA],
    // FP\Psi | FS
    cond_fppsi: [Simplex<f32, FP_PSI>; F_S],
}

fn belief_plural_ignorance(pi_rate: f32) -> Simplex<f32, A> {
    let u = 0.01;
    let b1 = pi_rate * (1.0 - u);
    Simplex::<f32, A>::new([1.0 - u - b1, b1], u)
}

impl FriendOpinion {
    pub fn reset(&mut self, pi_rate: f32, br_fs: &[f32; F_S], br_fphi: &[f32; F_PHI]) {
        reset_opinion(&mut self.fs, br_fs);
        reset_opinion(&mut self.fphi, br_fphi);
        reset_simplex(&mut self.cond_ftheta_fphi);
        self.cond_fpa[1] = belief_plural_ignorance(pi_rate);
        // let b1 = pi_rate.powi(2);
        // self.cond_fpa[1] = Simplex::<f32, FP_A>::new([pi_rate - b1, b1], 1.0 - pi_rate);
    }

    pub fn new() -> Self {
        Self {
            fs: Opinion::default(),
            fphi: Opinion::default(),
            cond_ftheta_fphi: [Simplex::default(), Simplex::default()],
            cond_fa: [
                Simplex::<f32, F_A>::new([0.95, 0.00], 0.05),
                Simplex::<f32, F_A>::new([0.00, 0.95], 0.05),
                Simplex::<f32, F_A>::new([0.95, 0.00], 0.05),
            ],
            cond_fpa: [
                Simplex::<f32, F_A>::new([0.90, 0.00], 0.10),
                Simplex::<f32, F_A>::default(),
                Simplex::<f32, F_A>::new([0.90, 0.00], 0.10),
            ],
            cond_ftheta: harr2![
                [
                    Simplex::<f32, F_THETA>::new([0.95, 0.00, 0.00], 0.05),
                    Simplex::<f32, F_THETA>::new([0.00, 0.45, 0.45], 0.10),
                ],
                [
                    Simplex::<f32, F_THETA>::new([0.00, 0.475, 0.475], 0.05),
                    Simplex::<f32, F_THETA>::new([0.00, 0.495, 0.495], 0.01),
                ]
            ],
            cond_fptheta: [
                Simplex::<f32, FP_THETA>::new([0.99, 0.000, 0.000], 0.01),
                Simplex::<f32, FP_THETA>::new([0.00, 0.495, 0.495], 0.01),
            ],
            cond_fppsi: [
                Simplex::<f32, FP_PSI>::new([0.99, 0.00], 0.01),
                Simplex::<f32, FP_PSI>::new([0.25, 0.65], 0.10),
            ],
        }
    }

    fn compute_new_friend_op(
        &self,
        info: &Info,
        fpsi_ded: &Opinion1d<f32, PSI>,
        ft: f32,
        constants: &Constants,
    ) -> (FriendOpinionUpd, Opinion1d<f32, F_A>) {
        let fs = FuseOp::Wgh.fuse(&self.fs, &info.content.s.discount(ft));
        let fpsi = FuseOp::Wgh.fuse(fpsi_ded, &info.content.psi.discount(ft));
        let fphi = FuseOp::Wgh.fuse(&self.fphi, &info.content.phi.discount(ft));
        let cond_ftheta_fphi = array::from_fn(|i| {
            FuseOp::Wgh.fuse(
                &self.cond_ftheta_fphi[i],
                &info.content.cond_theta_phi[i].discount(ft),
            )
        });
        let fop = FriendOpinionUpd {
            fs,
            fphi,
            cond_ftheta_fphi,
        };

        debug!(" P_FS  :{:?}", fop.fs.projection());

        let fpa = {
            let mut fpa = fop
                .fs
                .deduce(&self.cond_fppsi)
                .unwrap_or_else(|| Opinion::<_, f32>::vacuous_with(constants.br_fppsi))
                .deduce(&self.cond_fptheta)
                .unwrap_or_else(|| Opinion::<_, f32>::vacuous_with(constants.br_fptheta))
                .deduce(&self.cond_fpa)
                .unwrap_or_else(|| Opinion::<_, f32>::vacuous_with(constants.br_fpa));

            // compute a FPA opinion by using aleatory cummulative fusion.
            FuseOp::ACm.fuse_assign(&mut fpa, &info.content.pa.discount(ft));
            fpa
        };

        let ftheta = {
            let mut ftheta_ded_1 = Opinion::product2(&fpa, &fpsi)
                .deduce(&self.cond_ftheta)
                .unwrap_or_else(|| Opinion::<_, f32>::vacuous_with(constants.br_ftheta));
            let ftheta_ded_2 = fop
                .fphi
                .deduce(&fop.cond_ftheta_fphi)
                .unwrap_or_else(|| Opinion::<_, f32>::vacuous_with(constants.br_ftheta));
            FuseOp::Wgh.fuse_assign(&mut ftheta_ded_1, &ftheta_ded_2);
            ftheta_ded_1
        };
        debug!(" P_FPA :{:?}", fpa.projection());
        debug!(" P_FTH :{:?}", ftheta.projection());
        (
            fop,
            ftheta
                .deduce(&self.cond_fa)
                .unwrap_or_else(|| Opinion::<_, f32>::vacuous_with(constants.br_fa)),
        )
    }

    fn update(&mut self, fop: FriendOpinionUpd) {
        self.fs = fop.fs;
        self.fphi = fop.fphi;
        self.cond_ftheta_fphi = fop.cond_ftheta_fphi;
    }
}

#[derive(Debug)]
pub struct AgentOpinion {
    theta: Opinion1d<f32, THETA>,
    psi: Opinion1d<f32, PSI>,
    phi: Opinion1d<f32, PHI>,
    s: Opinion1d<f32, S>,

    // F\Psi | S
    cond_fpsi: [Simplex<f32, F_PSI>; S],

    // plural ignorance conditions
    // P\Psi | S
    cond_ppsi: [Simplex<f32, P_PSI>; S],
    // P\Theta | P\Psi
    cond_ptheta: [Simplex<f32, P_THETA>; P_PSI],
    // PA | P\Theta
    cond_pa: [Simplex<f32, P_A>; P_THETA],

    // \Theta | PA.\Psi,FA
    cond_theta: HigherArr3<Simplex<f32, THETA>, P_A, PSI, F_A>,
    // \Theta | \Phi
    cond_theta_phi: [Simplex<f32, THETA>; PHI],
}

impl AgentOpinion {
    pub fn reset(
        &mut self,
        pi_rate: f32,
        br_theta: &[f32; THETA],
        br_psi: &[f32; PSI],
        br_phi: &[f32; PSI],
        br_s: &[f32; S],
    ) {
        reset_opinion(&mut self.theta, br_theta);
        reset_opinion(&mut self.psi, br_psi);
        reset_opinion(&mut self.phi, br_phi);
        reset_opinion(&mut self.s, br_s);
        reset_simplex(&mut self.cond_theta_phi);
        self.cond_pa[1] = belief_plural_ignorance(pi_rate);
        // let b1 = pi_rate.powi(2);
        // self.cond_pa[1] = Simplex::<f32, P_A>::new([pi_rate - b1, b1], 1.0 - pi_rate);
    }

    pub fn new() -> Self {
        Self {
            theta: Opinion::default(),
            psi: Opinion::default(),
            phi: Opinion::default(),
            s: Opinion::default(),
            cond_theta_phi: [Simplex::default(), Simplex::default()],
            cond_pa: [
                Simplex::<f32, P_A>::new([0.90, 0.00], 0.10),
                Simplex::default(),
                Simplex::<f32, P_A>::new([0.90, 0.00], 0.10),
            ],
            cond_theta: harr3![
                [
                    [
                        Simplex::<f32, THETA>::new([0.95, 0.00, 0.00], 0.05),
                        Simplex::<f32, THETA>::new([0.95, 0.00, 0.00], 0.05),
                    ],
                    [
                        Simplex::<f32, THETA>::new([0.00, 0.45, 0.45], 0.10),
                        Simplex::<f32, THETA>::new([0.00, 0.45, 0.45], 0.10),
                    ],
                ],
                [
                    [
                        Simplex::<f32, THETA>::new([0.00, 0.475, 0.475], 0.05),
                        Simplex::<f32, THETA>::new([0.00, 0.475, 0.475], 0.05),
                    ],
                    [
                        Simplex::<f32, THETA>::new([0.00, 0.495, 0.495], 0.01),
                        Simplex::<f32, THETA>::new([0.00, 0.495, 0.495], 0.01),
                    ],
                ]
            ],
            cond_ptheta: [
                Simplex::<f32, P_THETA>::new([0.99, 0.00, 0.00], 0.01),
                Simplex::<f32, P_THETA>::new([0.00, 0.495, 0.495], 0.01),
            ],
            cond_ppsi: [
                Simplex::<f32, P_PSI>::new([0.99, 0.00], 0.01),
                Simplex::<f32, P_PSI>::new([0.25, 0.65], 0.10),
            ],
            cond_fpsi: [
                Simplex::<f32, F_PSI>::new([0.99, 0.00], 0.01),
                Simplex::<f32, F_PSI>::new([0.70, 0.20], 0.10),
            ],
        }
    }
}
