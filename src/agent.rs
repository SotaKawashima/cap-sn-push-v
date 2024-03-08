use approx::UlpsEq;
use num_traits::{Float, NumAssign};
use rand::Rng;
use rand_distr::{uniform::SampleUniform, Distribution, Exp1, Open01, Standard, StandardNormal};
use serde_with::{serde_as, TryFromInto};
use std::{
    array,
    collections::{BTreeMap, BTreeSet},
    fmt,
    iter::Sum,
};

use crate::{
    decision::{CptParams, LossParams, Prospect, CPT},
    dist::{IValue, IValueParam},
    info::{Info, TrustParams},
    opinion::{
        ConditionalOpinions, GlobalBaseRates, InitialConditions, InitialOpinions, Opinions,
        TempOpinions,
    },
    value::{EValue, EValueParam},
};

#[serde_as]
#[derive(Debug, serde::Deserialize)]
pub struct AgentParams<V>
where
    V: Float + NumAssign + UlpsEq,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
{
    pub initial_opinions: InitialOpinions<V>,
    pub initial_conditions: InitialConditions<V>,
    pub base_rates: GlobalBaseRates<V>,
    #[serde_as(as = "TryFromInto<EValueParam<V>>")]
    pub access_prob: EValue<V>,
    #[serde_as(as = "TryFromInto<EValueParam<V>>")]
    pub friend_access_prob: EValue<V>,
    #[serde_as(as = "TryFromInto<EValueParam<V>>")]
    pub social_access_prob: EValue<V>,
    #[serde_as(as = "TryFromInto<EValueParam<V>>")]
    pub friend_arrival_prob: EValue<V>,
    #[serde_as(as = "TryFromInto<IValueParam<V>>")]
    pub delay_selfish: IValue<V>,
    pub trust_params: TrustParams<V>,
    pub loss_params: LossParams<V>,
    pub cpt_params: CptParams<V>,
}

#[derive(Debug)]
pub struct BehaviorByInfo {
    pub sharing: bool,
    pub first_access: bool,
}

#[derive(Default)]
pub struct Agent<V: Float> {
    pub cpt: CPT<V>,
    pub prospect: Prospect<V>,
    ops: Opinions<V>,
    conds: ConditionalOpinions<V>,
    access_prob: V,
    friend_access_prob: V,
    social_access_prob: V,
    friend_arrival_rate: V,
    info_trust_map: BTreeMap<usize, V>,
    infos_accessed: BTreeSet<usize>,
    selfish_status: DelayActionStatus,
    sharing_statuses: BTreeMap<usize, ActionStatus>,
    delay_selfish: u32,
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
        log::debug!("{:?}", self);
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

impl<V> Agent<V>
where
    V: Float + UlpsEq + SampleUniform,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
{
    pub fn reset(
        &mut self,
        delay_selfish: u32,
        access_prob: V,
        friend_access_prob: V,
        social_access_prob: V,
        friend_arrival_prob: V,
        initial_opinions: InitialOpinions<V>,
        base_rates: &GlobalBaseRates<V>,
    ) where
        V: NumAssign + Default,
        StandardNormal: Distribution<V>,
        Exp1: Distribution<V>,
        Open01: Distribution<V>,
    {
        self.delay_selfish = delay_selfish;
        self.access_prob = access_prob;
        self.friend_access_prob = friend_access_prob;
        self.social_access_prob = social_access_prob;
        self.friend_arrival_rate = friend_arrival_prob;

        self.ops.reset(initial_opinions, base_rates);

        self.info_trust_map.clear();
        self.selfish_status.reset();
        self.sharing_statuses.clear();
        self.infos_accessed.clear();
    }

    pub fn reset_with<R>(&mut self, agent_params: &AgentParams<V>, rng: &mut R)
    where
        V: NumAssign + Sum + Default,
        R: Rng,
        StandardNormal: Distribution<V>,
        Exp1: Distribution<V>,
        Open01: Distribution<V>,
    {
        self.prospect.reset_with(&agent_params.loss_params, rng);
        self.cpt.reset_with(&agent_params.cpt_params, rng);

        self.conds = ConditionalOpinions::from_init(&agent_params.initial_conditions, rng);

        // if agent_params.pi_prob > rng.gen::<V>() {
        //     agent_params.pi_rate.sample(rng)
        // } else {
        //     V::zero()
        // },

        self.reset(
            agent_params.delay_selfish.sample(rng),
            agent_params.access_prob.sample(rng),
            agent_params.friend_access_prob.sample(rng),
            agent_params.social_access_prob.sample(rng),
            agent_params.friend_arrival_prob.sample(rng),
            agent_params.initial_opinions.clone(),
            &agent_params.base_rates,
        );
    }

    pub fn is_willing_selfish(&self) -> bool {
        self.selfish_status.is_willing()
    }

    pub fn progress_selfish_status(&mut self) -> bool {
        self.selfish_status.progress()
    }

    #[inline]
    pub fn access_prob(&self) -> V {
        self.access_prob
    }

    pub fn read_info(
        &mut self,
        info: &Info<V>,
        receipt_prob: V,
        agent_params: &AgentParams<V>,
        rng: &mut impl Rng,
    ) -> BehaviorByInfo
    where
        V: Sum + Default + fmt::Debug + NumAssign,
        StandardNormal: Distribution<V>,
        Exp1: Distribution<V>,
        Open01: Distribution<V>,
    {
        let first_access = self.infos_accessed.insert(info.idx);

        let trust = *self
            .info_trust_map
            .entry(info.idx)
            .or_insert_with(|| agent_params.trust_params.gen_map(rng)(info));
        let friend_trust = receipt_prob * self.friend_access_prob * trust;
        let social_trust = receipt_prob * self.social_access_prob * trust;

        let mut new_ops = self.ops.new(info, trust, friend_trust, social_trust);
        let temp_ops = new_ops.compute(info, social_trust, &self.conds, &agent_params.base_rates);

        // compute values of prospects
        let sharing_status = self.sharing_statuses.entry(info.idx).or_default();
        let sharing = 'a: {
            if sharing_status.is_done() {
                break 'a false;
            }
            let (pred_new_fop, ps) = new_ops.predict(
                &temp_ops,
                info,
                friend_trust,
                self.friend_arrival_rate * self.friend_access_prob * trust,
                &self.conds,
                &agent_params.base_rates,
            );
            let values: [V; 2] =
                array::from_fn(|i| self.cpt.valuate(&self.prospect.sharing[i], &ps[i]));
            log::info!("V_Y : {:?}", values);
            sharing_status.decide(values);
            if sharing_status.is_done() {
                new_ops.replace_pred_fop(pred_new_fop);
                true
            } else {
                false
            }
        };
        self.ops = new_ops;
        self.decide_selfish(temp_ops);

        BehaviorByInfo {
            sharing,
            first_access,
        }
    }

    pub fn set_info_opinions(&mut self, info: &Info<V>, base_rates: &GlobalBaseRates<V>)
    where
        V: Sum + Default + fmt::Debug + NumAssign,
    {
        let trust = V::one();
        let friend_trust = V::zero(); // * self.friend_access_prob * trust;
        let social_trust = V::zero(); // * self.social_access_prob * trust;

        let mut new_ops = self.ops.new(info, trust, friend_trust, social_trust);
        let temp = new_ops.compute(info, social_trust, &self.conds, base_rates);

        // posting info is equivalent to sharing it to friends with max trust.
        let (pred_fop, _) = new_ops.predict(
            &temp,
            info,
            friend_trust,
            self.friend_arrival_rate * self.friend_access_prob, // * trust,
            &self.conds,
            base_rates,
        );
        new_ops.replace_pred_fop(pred_fop);
        self.ops = new_ops;
        self.decide_selfish(temp);
    }

    fn decide_selfish(&mut self, temp_ops: TempOpinions<V>)
    where
        V: NumAssign + Sum + fmt::Debug + Default,
    {
        if self.selfish_status.is_done() {
            return;
        }
        let p = temp_ops.get_theta_projection();
        let values: [V; 2] = array::from_fn(|i| self.cpt.valuate(&self.prospect.selfish[i], &p));
        log::info!("V_X : {:?}", values);
        self.selfish_status.decide(values, self.delay_selfish);
        log::info!("selfish: {:?}", self.selfish_status);
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        agent::{ActionStatus, Agent, AgentParams},
        decision::{CptParams, LossParams},
        dist::IValue,
        info::{InfoBuilder, InfoObject, TrustParams},
        opinion::{
            BaseRates, CondFThetaDist, CondKThetaDist, CondThetaDist, CondThetadDist,
            FriendBaseRates, GlobalBaseRates, InitialBaseConditions, InitialBaseSimplexes,
            InitialConditions, InitialFriendConditions, InitialFriendSimplexes, InitialOpinions,
            InitialSocialConditions, InitialSocialSimplexes, RelativeParam, SimplexDist,
            SimplexParam, SocialBaseRates,
        },
        value::EValue,
    };

    use rand::thread_rng;
    use subjective_logic::{harr2, mul::Simplex};

    use super::DelayActionStatus;

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

    #[test]
    fn test_agent() {
        let agent_params = AgentParams {
            initial_opinions: InitialOpinions {
                base: InitialBaseSimplexes {
                    psi: Simplex::new([1.0, 0.0], 0.0),
                    phi: Simplex::new([0.9, 0.0], 0.1),
                    s: Simplex::new([0.8, 0.0], 0.2),
                    o: Simplex::new([0.7, 0.0], 0.3),
                },
                friend: InitialFriendSimplexes {
                    fphi: Simplex::new([0.6, 0.0], 0.4),
                    fs: Simplex::new([0.5, 0.0], 0.5),
                    fo: Simplex::new([0.4, 0.0], 0.6),
                },
                social: InitialSocialSimplexes {
                    kphi: Simplex::new([0.3, 0.0], 0.7),
                    ks: Simplex::new([0.2, 0.0], 0.8),
                    ko: Simplex::new([0.1, 0.0], 0.9),
                },
            },
            initial_conditions: InitialConditions {
                base: InitialBaseConditions {
                    // B => O
                    cond_o: [
                        SimplexParam::Fixed([1.0, 0.0], 0.0).try_into().unwrap(),
                        SimplexDist::Fixed(Simplex::new([0.0, 1.0], 0.0)),
                    ],
                    // K\Theta => B
                    cond_b: [
                        SimplexDist::Fixed(Simplex::new([0.9, 0.1], 0.0)),
                        SimplexDist::Fixed(Simplex::new([0.1, 0.9], 0.0)),
                    ],
                    // B,\Psi => \Theta
                    cond_theta: CondThetaDist {
                        b0psi0: SimplexDist::Fixed(Simplex::new([0.95, 0.00], 0.05)),
                        b1psi1: SimplexDist::Fixed(Simplex::new([0.50, 0.40], 0.10)),
                        b0psi1: RelativeParam {
                            belief: EValue::fixed(1.0),
                            uncertainty: EValue::fixed(1.0),
                        },
                        b1psi0: RelativeParam {
                            belief: EValue::fixed(1.0),
                            uncertainty: EValue::fixed(1.0),
                        },
                    },
                    // \Phi => \Theta
                    cond_theta_phi: [
                        SimplexDist::Fixed(Simplex::new([0.7, 0.3], 0.0)),
                        SimplexDist::Fixed(Simplex::new([0.3, 0.7], 0.0)),
                    ],
                    // F\Theta => A
                    cond_a: [
                        SimplexDist::Fixed(Simplex::new([0.6, 0.4], 0.0)),
                        SimplexDist::Fixed(Simplex::new([0.4, 0.6], 0.0)),
                    ],
                    // B,\Psi,A => \Theta'
                    cond_thetad: CondThetadDist {
                        a0b0psi0: SimplexDist::Fixed(Simplex::new([0.95, 0.00], 0.05)),
                        a0b1psi1: SimplexDist::Fixed(Simplex::new([0.50, 0.40], 0.10)),
                        a0b0psi1: RelativeParam {
                            belief: EValue::fixed(1.0),
                            uncertainty: EValue::fixed(1.0),
                        },
                        a0b1psi0: RelativeParam {
                            belief: EValue::fixed(1.0),
                            uncertainty: EValue::fixed(1.0),
                        },
                        a1: harr2![
                            [
                                RelativeParam {
                                    belief: EValue::fixed(1.0),
                                    uncertainty: EValue::fixed(1.0),
                                },
                                RelativeParam {
                                    belief: EValue::fixed(1.0),
                                    uncertainty: EValue::fixed(1.0),
                                },
                            ],
                            [
                                RelativeParam {
                                    belief: EValue::fixed(1.0),
                                    uncertainty: EValue::fixed(1.0),
                                },
                                RelativeParam {
                                    belief: EValue::fixed(1.0),
                                    uncertainty: EValue::fixed(1.0),
                                },
                            ]
                        ],
                    },
                    // \Phi => \Theta'
                    cond_thetad_phi: [
                        SimplexDist::Fixed(Simplex::new([0.4, 0.6], 0.0)),
                        SimplexDist::Fixed(Simplex::new([0.6, 0.4], 0.0)),
                    ],
                },
                friend: InitialFriendConditions {
                    // S -> F\Psi
                    cond_fpsi: [
                        SimplexDist::Fixed(Simplex::new([0.9, 0.0], 0.1)),
                        SimplexDist::Fixed(Simplex::new([0.0, 0.9], 0.1)),
                    ],
                    // FB => FO
                    cond_fo: [
                        SimplexDist::Fixed(Simplex::new([0.8, 0.1], 0.1)),
                        SimplexDist::Fixed(Simplex::new([0.1, 0.8], 0.1)),
                    ],
                    // FS => FB
                    cond_fb: [
                        SimplexDist::Fixed(Simplex::new([0.7, 0.2], 0.1)),
                        SimplexDist::Fixed(Simplex::new([0.2, 0.7], 0.1)),
                    ],
                    // FB,F\Psi => F\Theta
                    cond_ftheta: CondFThetaDist {
                        fb0fpsi0: SimplexDist::Fixed(Simplex::new([0.95, 0.00], 0.05)),
                        fb1fpsi1: SimplexDist::Fixed(Simplex::new([0.50, 0.40], 0.10)),
                        fb0fpsi1: RelativeParam {
                            belief: EValue::fixed(1.0),
                            uncertainty: EValue::fixed(1.0),
                        },
                        fb1fpsi0: RelativeParam {
                            belief: EValue::fixed(1.0),
                            uncertainty: EValue::fixed(1.0),
                        },
                    },
                    // F\Phi => F\Theta
                    cond_ftheta_fphi: [
                        SimplexDist::Fixed(Simplex::new([0.5, 0.4], 0.1)),
                        SimplexDist::Fixed(Simplex::new([0.4, 0.5], 0.1)),
                    ],
                },
                social: InitialSocialConditions {
                    // S -> K\Psi
                    cond_kpsi: [
                        SimplexDist::Fixed(Simplex::new([0.8, 0.0], 0.2)),
                        SimplexDist::Fixed(Simplex::new([0.0, 0.8], 0.2)),
                    ],
                    // KB => KO
                    cond_ko: [
                        SimplexDist::Fixed(Simplex::new([0.7, 0.1], 0.2)),
                        SimplexDist::Fixed(Simplex::new([0.1, 0.7], 0.2)),
                    ],
                    // KS => KB
                    cond_kb: [
                        SimplexDist::Fixed(Simplex::new([0.6, 0.2], 0.2)),
                        SimplexDist::Fixed(Simplex::new([0.2, 0.6], 0.2)),
                    ],
                    // KB,K\Psi => K\Theta
                    cond_ktheta: CondKThetaDist {
                        kb0kpsi0: SimplexDist::Fixed(Simplex::new([0.95, 0.00], 0.05)),
                        kb1kpsi1: SimplexDist::Fixed(Simplex::new([0.50, 0.40], 0.10)),
                        kb0kpsi1: RelativeParam {
                            belief: EValue::fixed(1.0),
                            uncertainty: EValue::fixed(1.0),
                        },
                        kb1kpsi0: RelativeParam {
                            belief: EValue::fixed(1.0),
                            uncertainty: EValue::fixed(1.0),
                        },
                    },
                    // K\Phi => K\Theta
                    cond_ktheta_kphi: [
                        SimplexDist::Fixed(Simplex::new([0.41, 0.39], 0.2)),
                        SimplexDist::Fixed(Simplex::new([0.39, 0.41], 0.2)),
                    ],
                },
            },
            base_rates: GlobalBaseRates {
                base: BaseRates {
                    psi: [0.99, 0.01],
                    phi: [0.98, 0.02],
                    s: [0.97, 0.03],
                    o: [0.96, 0.04],
                    b: [0.95, 0.05],
                    theta: [0.94, 0.06],
                    a: [0.93, 0.07],
                    thetad: [0.92, 0.08],
                },
                friend: FriendBaseRates {
                    fpsi: [0.90, 0.10],
                    fphi: [0.89, 0.11],
                    fs: [0.88, 0.12],
                    fo: [0.87, 0.13],
                    fb: [0.86, 0.14],
                    ftheta: [0.85, 0.15],
                },
                social: SocialBaseRates {
                    kpsi: [0.80, 0.20],
                    kphi: [0.79, 0.21],
                    ks: [0.78, 0.22],
                    ko: [0.77, 0.23],
                    kb: [0.76, 0.24],
                    ktheta: [0.75, 0.25],
                },
            },
            delay_selfish: IValue::Fixed(1),
            access_prob: EValue::fixed(0.0),
            friend_access_prob: EValue::fixed(0.01),
            social_access_prob: EValue::fixed(0.02),
            friend_arrival_prob: EValue::fixed(0.03),
            // pi_rate: DistValue::fixed(0.04),
            // pi_prob: 0.05,
            trust_params: TrustParams {
                misinfo: EValue::fixed(0.10),
                corrective: EValue::fixed(0.11),
                observed: EValue::fixed(0.12),
                inhibitive: EValue::fixed(0.13),
            },
            cpt_params: CptParams {
                alpha: EValue::fixed(1.3),
                beta: EValue::fixed(1.4),
                lambda: EValue::fixed(1.5),
                gamma: EValue::fixed(1.6),
                delta: EValue::fixed(1.7),
            },
            loss_params: LossParams {
                x0: EValue::fixed(-1.0),
                x1_of_x0: EValue::fixed(1.1),
                y_of_x0: EValue::fixed(1.2),
            },
        };

        let info_objects = [InfoObject::Misinfo {
            psi: Simplex::new([0.00, 0.99], 0.01),
        }];
        let info_builder = InfoBuilder::new();
        let info = info_builder.build(0, &info_objects[0]);

        let mut a = Agent::default();
        a.prospect.reset(-0.1, -2.0, -0.001);
        a.cpt.reset(0.88, 0.88, 2.25, 0.61, 0.69);
        a.reset_with(&agent_params, &mut thread_rng());

        println!("{:?}", a.set_info_opinions(&info, &agent_params.base_rates));
    }
}
