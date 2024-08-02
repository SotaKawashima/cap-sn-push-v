use num_traits::Float;
use rand::Rng;
use rand_distr::{Distribution, Exp1, Open01, Standard, StandardNormal};
use std::{
    array,
    collections::{BTreeMap, BTreeSet},
};
use tracing::*;

use crate::{
    decision::{Prospect, CPT},
    info::Info,
    opinion::{AccessProb, DeducedOpinions, MyFloat, MyOpinions, MyOpinionsUpd, Trusts},
    util::Reset,
};

#[derive(Debug)]
pub struct BehaviorByInfo {
    pub sharing: bool,
    pub first_access: bool,
}

#[derive(Default)]
pub struct Agent<V> {
    pub ops: MyOpinions<V>,
    pub infos_accessed: BTreeSet<usize>,
    pub decision: Decision<V>,
}

#[derive(Default)]
pub struct Decision<V> {
    pub cpt: CPT<V>,
    pub prospect: Prospect<V>,
    selfish_status: DelayActionStatus,
    sharing_statuses: BTreeMap<usize, ActionStatus>,
    pub delay_selfish: u32,
}

impl<V: MyFloat> Decision<V> {
    fn values_selfish(&self, ded: &DeducedOpinions<V>) -> [V; 2] {
        let p_theta = ded.p_theta();
        info!(target: "    TH", P = ?p_theta);
        let values: [V; 2] =
            array::from_fn(|i| self.cpt.valuate(&self.prospect.selfish[i], &p_theta));
        info!(target: "     X", V = ?values);
        values
    }

    fn try_decide_selfish(&mut self, upd: &MyOpinionsUpd<V>) {
        if !self.selfish_status.is_done() {
            upd.decide1(|ded| {
                self.selfish_status
                    .decide(self.values_selfish(ded), self.delay_selfish);
                info!(target: "selfsh", status = ?self.selfish_status);
            });
        }
    }

    fn predict(&self, upd: &mut MyOpinionsUpd<V>) {
        upd.decide2(|_, _| true);
    }

    fn try_decide_sharing(&mut self, upd: &mut MyOpinionsUpd<V>, info_idx: usize) -> bool {
        let sharing_status = self.sharing_statuses.entry(info_idx).or_default();
        if sharing_status.is_done() {
            false
        } else {
            upd.decide2(|ded, pred_ded| {
                let p_a_thetad = ded.p_a_thetad();
                let pred_p_a_thetad = pred_ded.p_a_thetad();
                info!(target: "   THd", P = ?p_a_thetad);
                info!(target: "  ~THd", P = ?pred_p_a_thetad);

                let values = [
                    self.cpt.valuate(&self.prospect.sharing[0], &p_a_thetad),
                    self.cpt
                        .valuate(&self.prospect.sharing[1], &pred_p_a_thetad),
                ];
                info!(target: "     Y", V = ?values);
                sharing_status.decide(values);
                info!(target: "sharng", status = ?sharing_status);
                sharing_status.is_done()
            })
        }
    }

    pub fn reset<R: Rng, P: Reset<Self>>(&mut self, param: &P, rng: &mut R)
    where
        Open01: Distribution<V>,
        Standard: Distribution<V>,
        StandardNormal: Distribution<V>,
        Exp1: Distribution<V>,
    {
        self.selfish_status.reset();
        self.sharing_statuses.clear();
        param.reset(self, rng);
    }
}

#[derive(Default, Debug)]
enum ActionStatus {
    #[default]
    NotYet,
    Done,
}

impl ActionStatus {
    fn is_done(&self) -> bool {
        matches!(self, Self::Done)
    }

    /// Index 0 of `values` indicates 'do not perform this action', and index 1 indicates 'perform this action.'
    fn decide<V: Float>(&mut self, values: [V; 2]) {
        if values[0] < values[1] {
            *self = Self::Done;
        }
    }
}

#[derive(Default, Debug)]
enum DelayActionStatus {
    #[default]
    NotYet,
    Willing(u32),
    Done,
}

impl DelayActionStatus {
    fn reset(&mut self) {
        *self = Self::NotYet;
    }

    #[inline]
    fn is_willing(&self) -> bool {
        matches!(self, Self::Willing(_))
    }

    #[inline]
    fn is_done(&self) -> bool {
        matches!(self, Self::Done)
    }

    /// Index 0 of `values` indicates 'do not perform this action', and index 1 indicates 'perform this action.'
    fn decide<V: Float>(&mut self, values: [V; 2], delay: u32) {
        let perform = values[0] < values[1];
        match self {
            Self::NotYet if perform => {
                *self = Self::Willing(delay);
            }
            Self::Willing(_) if !perform => {
                *self = Self::NotYet;
            }
            _ => {}
        }
    }

    fn progress(&mut self) -> bool {
        match self {
            Self::Willing(0) => {
                *self = Self::Done;
                true
            }
            Self::Willing(r) => {
                *r -= 1;
                false
            }
            _ => false,
        }
    }
}

impl<V> Agent<V> {
    pub fn reset<P, R>(&mut self, param: &P, rng: &mut R)
    where
        R: Rng,
        P: Reset<Self>,
    {
        self.infos_accessed.clear();
        param.reset(self, rng);
    }

    pub fn is_willing_selfish(&self) -> bool {
        self.decision.selfish_status.is_willing()
    }

    pub fn progress_selfish_status(&mut self) -> bool {
        let p = self.decision.selfish_status.progress();
        info!(target: "selfsh", status = ?self.decision.selfish_status);
        p
    }

    pub fn read_info(
        &mut self,
        info: &Info<V>,
        trusts: Trusts<V>,
        ap: AccessProb<V>,
    ) -> BehaviorByInfo
    where
        V: MyFloat,
        StandardNormal: Distribution<V>,
        Exp1: Distribution<V>,
        Open01: Distribution<V>,
    {
        let first_access = self.infos_accessed.insert(info.idx);

        // compute values of prospects
        let mut upd = self.ops.receive(info.p, trusts, ap);
        self.decision.try_decide_selfish(&upd);
        let sharing = self.decision.try_decide_sharing(&mut upd, info.idx);

        BehaviorByInfo {
            sharing,
            first_access,
        }
    }

    pub fn set_info_opinions(&mut self, info: &Info<V>, trusts: Trusts<V>, ap: AccessProb<V>)
    where
        V: MyFloat,
        StandardNormal: Distribution<V>,
        Exp1: Distribution<V>,
        Open01: Distribution<V>,
    {
        let mut upd = self.ops.receive(info.p, trusts, ap);
        self.decision.try_decide_selfish(&upd);
        self.decision.predict(&mut upd);
    }
}

#[cfg(test)]
mod tests {
    use super::{ActionStatus, DelayActionStatus};

    #[test]
    fn test_action_status() {
        let mut s = ActionStatus::default();
        assert!(matches!(s, ActionStatus::NotYet));
        s.decide([1.0, 0.0]);
        assert!(matches!(s, ActionStatus::NotYet));
        s.decide([0.0, 1.0]);
        assert!(s.is_done());

        let mut s = DelayActionStatus::default();
        assert!(matches!(s, DelayActionStatus::NotYet));
        s.decide([0.0, 1.0], 1);
        assert!(matches!(s, DelayActionStatus::Willing(1)));
        s.decide([0.0, 1.0], 1);
        assert!(matches!(s, DelayActionStatus::Willing(1)));
        s.progress();
        assert!(matches!(s, DelayActionStatus::Willing(0)));
        s.progress();
        assert!(s.is_done());

        s.reset();
        assert!(matches!(s, DelayActionStatus::NotYet));
        s.decide([0.0, 1.0], 2);
        s.progress();
        assert!(matches!(s, DelayActionStatus::Willing(1)));
        s.progress();
        assert!(matches!(s, DelayActionStatus::Willing(0)));
        s.decide([1.0, 0.0], 2);
        s.progress();
        assert!(matches!(s, DelayActionStatus::NotYet));
        s.reset();
        s.decide([1.0, 0.0], 1);
        s.progress();
        assert!(matches!(s, DelayActionStatus::NotYet));
        s.decide([0.0, 1.0], 1);
        s.progress();
        assert!(matches!(s, DelayActionStatus::Willing(0)));
        s.decide([0.0, 1.0], 1);
        assert!(matches!(s, DelayActionStatus::Willing(0)));
        s.progress();
        assert!(s.is_done());
    }
}
