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
    cpt::{CptParams, Prospect, CPT},
    info::{Info, TrustParams},
    opinion::{
        ConditionalOpinions, GlobalBaseRates, InitialConditions, InitialOpinions, Opinions,
        TempOpinions,
    },
    value::{DistValue, ParamValue},
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
    #[serde_as(as = "TryFromInto<ParamValue<V>>")]
    pub access_prob: DistValue<V>,
    #[serde_as(as = "TryFromInto<ParamValue<V>>")]
    pub friend_access_prob: DistValue<V>,
    #[serde_as(as = "TryFromInto<ParamValue<V>>")]
    pub social_access_prob: DistValue<V>,
    #[serde_as(as = "TryFromInto<ParamValue<V>>")]
    pub friend_arrival_prob: DistValue<V>,
    // #[serde_as(as = "TryFromInto<ParamValue<V>>")]
    // pub pi_rate: DistValue<V>,
    // probability whether people has plural ignorance
    // pub pi_prob: V,
    pub trust_params: TrustParams<V>,
    pub cpt_params: CptParams<V>,
}

#[derive(Debug)]
pub struct Behavior {
    pub selfish: bool,
    pub sharing: bool,
    pub first_reading: bool,
}

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
    selfish: ActionStatus,
    sharing: BTreeMap<usize, ActionStatus>,
    reading: BTreeSet<usize>,
}

#[derive(Default)]
enum ActionStatus {
    #[default]
    NotYet,
    Done,
}

impl ActionStatus {
    fn reset(&mut self) {
        *self = ActionStatus::NotYet;
    }

    fn is_done(&self) -> bool {
        matches!(self, ActionStatus::Done)
    }

    /// retrun true if doing action is chosen, or false otherwise.
    fn decide<V: Float>(&mut self, values: [V; 2]) -> bool {
        if values[0] < values[1] {
            *self = ActionStatus::Done;
            true
        } else {
            false
        }
    }
}

impl<V> Agent<V>
where
    V: Float + UlpsEq + SampleUniform,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
{
    pub fn new() -> Self
    where
        V: Default,
    {
        Self {
            cpt: Default::default(),
            prospect: Default::default(),
            ops: Default::default(),
            conds: Default::default(),
            access_prob: V::zero(),
            friend_arrival_rate: V::zero(),
            friend_access_prob: V::zero(),
            social_access_prob: V::zero(),
            info_trust_map: Default::default(),
            selfish: Default::default(),
            sharing: Default::default(),
            reading: Default::default(),
        }
    }

    pub fn reset(
        &mut self,
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
        self.access_prob = access_prob;
        self.friend_access_prob = friend_access_prob;
        self.social_access_prob = social_access_prob;
        self.friend_arrival_rate = friend_arrival_prob;

        self.ops.reset(initial_opinions, base_rates);

        self.info_trust_map.clear();
        self.selfish.reset();
        self.sharing.clear();
        self.reading.clear();
    }

    pub fn reset_with<R>(&mut self, agent_params: &AgentParams<V>, rng: &mut R)
    where
        V: NumAssign + Sum + Default,
        R: Rng,
        StandardNormal: Distribution<V>,
        Exp1: Distribution<V>,
        Open01: Distribution<V>,
    {
        self.cpt.reset_with(&agent_params.cpt_params, rng);
        self.prospect.reset_with(&agent_params.cpt_params, rng);

        self.conds = ConditionalOpinions::from_init(&agent_params.initial_conditions, rng);

        // if agent_params.pi_prob > rng.gen::<V>() {
        //     agent_params.pi_rate.sample(rng)
        // } else {
        //     V::zero()
        // },
        // agent_params.initial_conditions.clone(),

        self.reset(
            agent_params.access_prob.sample(rng),
            agent_params.friend_access_prob.sample(rng),
            agent_params.social_access_prob.sample(rng),
            agent_params.friend_arrival_prob.sample(rng),
            agent_params.initial_opinions.clone(),
            &agent_params.base_rates,
        );
    }

    pub fn read_info(
        &mut self,
        info: &Info<V>,
        receipt_prob: V,
        agent_params: &AgentParams<V>,
        rng: &mut impl Rng,
    ) -> Option<Behavior>
    where
        V: Sum + Default + fmt::Debug + NumAssign,
        StandardNormal: Distribution<V>,
        Exp1: Distribution<V>,
        Open01: Distribution<V>,
    {
        if rng.gen::<V>() > self.access_prob {
            return None;
        }
        let first_reading = self.reading.insert(info.idx);

        let trust = *self
            .info_trust_map
            .entry(info.idx)
            .or_insert_with(|| agent_params.trust_params.gen_map(rng)(info));
        let friend_trust = receipt_prob * self.friend_access_prob * trust;
        let social_trust = receipt_prob * self.social_access_prob * trust;

        let mut new_ops = self.ops.new(info, trust, friend_trust, social_trust);
        let temp = new_ops.compute(info, social_trust, &self.conds, &agent_params.base_rates);

        // compute values of prospects
        let sharing_status = self.sharing.entry(info.idx).or_default();
        let mut sharing = false;
        if !sharing_status.is_done() {
            let (pred_new_fop, ps) = new_ops.predict(
                &temp,
                info,
                friend_trust,
                self.friend_arrival_rate * self.friend_access_prob * trust,
                &self.conds,
                &agent_params.base_rates,
            );
            let values: [V; 2] =
                array::from_fn(|i| self.cpt.valuate(&self.prospect.sharing[i], &ps[i]));
            log::info!("V_Y:{:?}", values);
            if sharing_status.decide(values) {
                new_ops.replace_pred_fop(pred_new_fop);
                sharing = true;
            }
        };
        self.ops = new_ops;
        let selfish = self.decide_selfish(temp);

        Some(Behavior {
            selfish,
            sharing,
            first_reading,
        })
    }

    pub fn set_info_opinions(&mut self, info: &Info<V>, base_rates: &GlobalBaseRates<V>) -> bool
    where
        V: Sum + Default + fmt::Debug + NumAssign,
    {
        let trust = V::one();
        let friend_trust = V::zero(); // * self.friend_access_prob * trust;
        let social_trust = V::zero(); // * self.social_access_prob * trust;

        let mut new_ops = self.ops.new(info, trust, friend_trust, social_trust);
        let temp = new_ops.compute(info, social_trust, &self.conds, base_rates);
        // log::debug!("{new_ops:?}");
        // log::debug!("{temp:?}");
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
        self.decide_selfish(temp)
    }

    fn decide_selfish(&mut self, temp: TempOpinions<V>) -> bool
    where
        V: NumAssign + Sum + fmt::Debug + Default,
    {
        if self.selfish.is_done() {
            return false;
        }
        let p = temp.get_theta_projection();
        log::debug!("P_TH: {:?}", p);
        let values: [V; 2] = array::from_fn(|i| self.cpt.valuate(&self.prospect.selfish[i], &p));
        log::info!("V_X  : {:?}", values);
        self.selfish.decide(values)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        agent::{Agent, AgentParams},
        cpt::CptParams,
        info::{Info, InfoContent, InfoObject, TrustParams},
        opinion::{
            BaseRates, FriendBaseRates, GlobalBaseRates, InitialBaseConditions,
            InitialBaseSimplexes, InitialConditions, InitialFriendConditions,
            InitialFriendSimplexes, InitialOpinions, InitialSocialConditions,
            InitialSocialSimplexes, SimplexDist, SimplexParam, SocialBaseRates,
        },
        value::DistValue,
    };

    use rand::thread_rng;
    use subjective_logic::{harr2, harr3, mul::Simplex};

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
                    cond_theta: harr2![
                        [
                            SimplexDist::Fixed(Simplex::new([0.8, 0.2], 0.0)),
                            SimplexDist::Fixed(Simplex::new([0.2, 0.8], 0.0)),
                        ],
                        [
                            SimplexDist::Fixed(Simplex::new([0.85, 0.15], 0.0)),
                            SimplexDist::Fixed(Simplex::new([0.15, 0.85], 0.0)),
                        ]
                    ],
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
                    cond_thetad: harr3![
                        [
                            [
                                SimplexDist::Fixed(Simplex::new([0.51, 0.49], 0.0)),
                                SimplexDist::Fixed(Simplex::new([0.49, 0.51], 0.0)),
                            ],
                            [
                                SimplexDist::Fixed(Simplex::new([0.52, 0.48], 0.0)),
                                SimplexDist::Fixed(Simplex::new([0.48, 0.52], 0.0)),
                            ],
                        ],
                        [
                            [
                                SimplexDist::Fixed(Simplex::new([0.53, 0.47], 0.0)),
                                SimplexDist::Fixed(Simplex::new([0.47, 0.53], 0.0)),
                            ],
                            [
                                SimplexDist::Fixed(Simplex::new([0.54, 0.46], 0.0)),
                                SimplexDist::Fixed(Simplex::new([0.46, 0.54], 0.0)),
                            ],
                        ]
                    ],
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
                    cond_ftheta: harr2![
                        [
                            SimplexDist::Fixed(Simplex::new([0.60, 0.30], 0.1)),
                            SimplexDist::Fixed(Simplex::new([0.30, 0.60], 0.1)),
                        ],
                        [
                            SimplexDist::Fixed(Simplex::new([0.61, 0.29], 0.1)),
                            SimplexDist::Fixed(Simplex::new([0.29, 0.61], 0.1)),
                        ]
                    ],
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
                    cond_ktheta: harr2![
                        [
                            SimplexDist::Fixed(Simplex::new([0.50, 0.30], 0.2)),
                            SimplexDist::Fixed(Simplex::new([0.30, 0.50], 0.2)),
                        ],
                        [
                            SimplexDist::Fixed(Simplex::new([0.51, 0.29], 0.2)),
                            SimplexDist::Fixed(Simplex::new([0.29, 0.51], 0.2)),
                        ]
                    ],
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
            access_prob: DistValue::fixed(0.0),
            friend_access_prob: DistValue::fixed(0.01),
            social_access_prob: DistValue::fixed(0.02),
            friend_arrival_prob: DistValue::fixed(0.03),
            // pi_rate: DistValue::fixed(0.04),
            // pi_prob: 0.05,
            trust_params: TrustParams {
                misinfo: DistValue::fixed(0.10),
                corrective: DistValue::fixed(0.11),
                observed: DistValue::fixed(0.12),
                inhibitive: DistValue::fixed(0.13),
            },
            cpt_params: CptParams {
                x0: DistValue::fixed(1.0),
                x1: DistValue::fixed(1.1),
                y: DistValue::fixed(1.2),
                alpha: DistValue::fixed(1.3),
                beta: DistValue::fixed(1.4),
                lambda: DistValue::fixed(1.5),
                gamma: DistValue::fixed(1.6),
                delta: DistValue::fixed(1.7),
            },
        };

        let info_contents = [InfoContent::<f32>::from(InfoObject::Misinfo {
            psi: Simplex::new([0.00, 0.99], 0.01),
        })];
        let info = Info::new(0, &info_contents[0]);

        let mut a = Agent::new();
        a.prospect.reset(-0.1, -2.0, -0.001);
        a.cpt.reset(0.88, 0.88, 2.25, 0.61, 0.69);
        a.reset_with(&agent_params, &mut thread_rng());

        println!("{:?}", a.set_info_opinions(&info, &agent_params.base_rates));
    }
}
