use approx::UlpsEq;
use num_traits::{Float, NumAssign};
use rand::Rng;
use rand_distr::{uniform::SampleUniform, Distribution, Open01, Standard};
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
{
    pub initial_opinions: InitialOpinions<V>,
    pub initial_conditions: InitialConditions<V>,
    pub base_rates: GlobalBaseRates<V>,
    #[serde_as(as = "TryFromInto<ParamValue<V>>")]
    pub read_prob: DistValue<V>,
    #[serde_as(as = "TryFromInto<ParamValue<V>>")]
    pub farrival_prob: DistValue<V>,
    #[serde_as(as = "TryFromInto<ParamValue<V>>")]
    pub fread_prob: DistValue<V>,
    #[serde_as(as = "TryFromInto<ParamValue<V>>")]
    pub pi_rate: DistValue<V>,
    /// probability whether people has plural ignorance
    pub pi_prob: V,
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
    read_prob: V,
    friend_arrival_prob: V,
    friend_read_prob: V,
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
            read_prob: V::zero(),
            friend_arrival_prob: V::zero(),
            friend_read_prob: V::zero(),
            info_trust_map: Default::default(),
            selfish: Default::default(),
            sharing: Default::default(),
            reading: Default::default(),
        }
    }

    pub fn reset(
        &mut self,
        read_prob: V,
        friend_arrival_prob: V,
        friend_read_prob: V,
        pi_rate: V,
        initial_opinions: InitialOpinions<V>,
        initial_conds: InitialConditions<V>,
        base_rates: &GlobalBaseRates<V>,
    ) where
        V: NumAssign + Default,
    {
        self.read_prob = read_prob;
        self.friend_arrival_prob = friend_arrival_prob;
        self.friend_read_prob = friend_read_prob;

        self.ops.reset(initial_opinions, base_rates);
        self.conds.reset(initial_conds, pi_rate);

        self.info_trust_map.clear();
        self.selfish.reset();
        self.sharing.clear();
        self.reading.clear();
    }

    pub fn reset_with<R>(&mut self, agent_params: &AgentParams<V>, rng: &mut R)
    where
        V: NumAssign + Sum + Default,
        R: Rng,
    {
        self.cpt.reset_with(&agent_params.cpt_params, rng);
        self.prospect.reset_with(&agent_params.cpt_params, rng);

        self.reset(
            agent_params.read_prob.sample(rng),
            agent_params.farrival_prob.sample(rng),
            agent_params.fread_prob.sample(rng),
            if agent_params.pi_prob > rng.gen::<V>() {
                agent_params.pi_rate.sample(rng)
            } else {
                V::zero()
            },
            agent_params.initial_opinions.clone(),
            agent_params.initial_conditions.clone(),
            &agent_params.base_rates,
        );
    }

    pub fn read_info(
        &mut self,
        info: &Info<V>,
        friend_receipt_prob: V,
        agent_params: &AgentParams<V>,
        rng: &mut impl Rng,
    ) -> Option<Behavior>
    where
        V: Sum + Default + fmt::Debug + NumAssign,
    {
        if rng.gen::<V>() > self.read_prob {
            return None;
        }
        let first_reading = self.reading.insert(info.idx);

        let trust = *self
            .info_trust_map
            .entry(info.idx)
            .or_insert_with(|| agent_params.trust_params.gen_map(rng)(info));
        let ft = friend_receipt_prob * self.friend_read_prob * trust;

        let mut new_ops = self.ops.new(info, trust, ft, &self.conds);
        let temp = new_ops.compute(info, trust, &self.conds, &agent_params.base_rates);

        // compute values of prospects
        let sharing_status = self.sharing.entry(info.idx).or_default();
        let mut sharing = false;
        if !sharing_status.is_done() {
            let (pred_new_fop, ps) = new_ops.predicate(
                &temp,
                info,
                ft,
                self.friend_arrival_prob * self.friend_read_prob * trust,
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
        let ft = V::zero();
        let mut new_ops = self.ops.new(info, trust, ft, &self.conds);
        let temp = new_ops.compute(info, trust, &self.conds, base_rates);
        // posting info is equivalent to sharing it to friends with max trust.
        let (pred_fop, _) = new_ops.predicate(
            &temp,
            info,
            ft,
            self.friend_arrival_prob * self.friend_read_prob,
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
        log::debug!("P_TH:{:?}", p);
        let values: [V; 2] = array::from_fn(|i| self.cpt.valuate(&self.prospect.selfish[i], &p));
        log::info!("V_X:{:?}", values);
        self.selfish.decide(values)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        agent::{Agent, AgentParams},
        cpt::CptParams,
        info::{Info, InfoContent, InfoObject, TrustParams},
        opinion::{GlobalBaseRates, InitialConditions, InitialOpinions},
        value::DistValue,
    };

    use subjective_logic::{harr2, harr3, mul::Simplex};

    #[test]
    fn test_agent() {
        let agent_params = AgentParams {
            initial_opinions: InitialOpinions {
                psi: Simplex::vacuous(),
                phi: Simplex::vacuous(),
                s: Simplex::vacuous(),
                fs: Simplex::vacuous(),
                fphi: Simplex::vacuous(),
            },
            initial_conditions: InitialConditions {
                cond_theta_phi: [Simplex::vacuous(), Simplex::vacuous()],
                cond_thetad_phi: [Simplex::vacuous(), Simplex::vacuous()],
                cond_ftheta_fphi: [Simplex::vacuous(), Simplex::vacuous()],
                cond_pa: [
                    Simplex::new([0.90, 0.00], 0.10),
                    Simplex::new([0.00, 0.99], 0.01),
                    Simplex::new([0.90, 0.00], 0.10),
                ],
                cond_theta: harr2![
                    [
                        Simplex::new([0.95, 0.00, 0.00], 0.05),
                        Simplex::new([0.00, 0.45, 0.45], 0.10),
                    ],
                    [
                        Simplex::new([0.00, 0.475, 0.475], 0.05),
                        Simplex::new([0.00, 0.495, 0.495], 0.01),
                    ]
                ],
                cond_thetad: harr3![
                    [
                        [
                            Simplex::new([0.95, 0.00, 0.00], 0.05),
                            Simplex::new([0.95, 0.00, 0.00], 0.05),
                        ],
                        [
                            Simplex::new([0.00, 0.45, 0.45], 0.10),
                            Simplex::new([0.00, 0.45, 0.45], 0.10),
                        ],
                    ],
                    [
                        [
                            Simplex::new([0.00, 0.475, 0.475], 0.05),
                            Simplex::new([0.00, 0.475, 0.475], 0.05),
                        ],
                        [
                            Simplex::new([0.00, 0.495, 0.495], 0.01),
                            Simplex::new([0.00, 0.495, 0.495], 0.01),
                        ],
                    ]
                ],
                cond_ptheta: [
                    Simplex::new([0.99, 0.00, 0.00], 0.01),
                    Simplex::new([0.00, 0.495, 0.495], 0.01),
                ],
                cond_ppsi: [
                    Simplex::new([0.99, 0.00], 0.01),
                    Simplex::new([0.25, 0.65], 0.10),
                ],
                cond_fpsi: [
                    Simplex::new([0.99, 0.00], 0.01),
                    Simplex::new([0.70, 0.20], 0.10),
                ],
                cond_fa: [
                    Simplex::new([0.95, 0.00], 0.05),
                    Simplex::new([0.00, 0.95], 0.05),
                    Simplex::new([0.95, 0.00], 0.05),
                ],
                cond_fpa: [
                    Simplex::new([0.90, 0.00], 0.10),
                    Simplex::new([0.00, 0.99], 0.01),
                    Simplex::new([0.90, 0.00], 0.10),
                ],
                cond_ftheta: harr2![
                    [
                        Simplex::new([0.95, 0.00, 0.00], 0.05),
                        Simplex::new([0.00, 0.45, 0.45], 0.10),
                    ],
                    [
                        Simplex::new([0.00, 0.475, 0.475], 0.05),
                        Simplex::new([0.00, 0.495, 0.495], 0.01),
                    ]
                ],
                cond_fptheta: [
                    Simplex::new([0.99, 0.000, 0.000], 0.01),
                    Simplex::new([0.00, 0.495, 0.495], 0.01),
                ],
                cond_fppsi: [
                    Simplex::new([0.99, 0.00], 0.01),
                    Simplex::new([0.25, 0.65], 0.10),
                ],
            },
            base_rates: GlobalBaseRates {
                s: [0.999, 0.001],
                fs: [0.999, 0.001],
                psi: [0.999, 0.001],
                ppsi: [0.999, 0.001],
                pa: [0.999, 0.001],
                fa: [0.999, 0.001],
                fpa: [0.999, 0.001],
                phi: [0.999, 0.001],
                fpsi: [0.999, 0.001],
                fppsi: [0.999, 0.001],
                fphi: [0.999, 0.001],
                theta: [0.999, 0.0005, 0.0005],
                ptheta: [0.999, 0.0005, 0.0005],
                ftheta: [0.999, 0.0005, 0.0005],
                fptheta: [0.999, 0.0005, 0.0005],
            },
            read_prob: DistValue::fixed(0.5),
            farrival_prob: DistValue::fixed(0.5),
            fread_prob: DistValue::fixed(0.5),
            pi_rate: DistValue::fixed(0.5),
            pi_prob: 0.5,
            trust_params: TrustParams {
                misinfo: DistValue::fixed(0.5),
                corrective: DistValue::fixed(0.5),
                observed: DistValue::fixed(0.5),
                inhibitive: DistValue::fixed(0.5),
            },
            cpt_params: CptParams {
                x0: DistValue::fixed(0.5),
                x1: DistValue::fixed(0.5),
                y: DistValue::fixed(0.5),
                alpha: DistValue::fixed(0.88),
                beta: DistValue::fixed(0.88),
                lambda: DistValue::fixed(2.25),
                gamma: DistValue::fixed(0.61),
                delta: DistValue::fixed(0.69),
            },
        };

        let info_contents = [InfoContent::<f32>::from(InfoObject::Misinfo {
            psi: Simplex::new([0.00, 0.99], 0.01),
        })];
        let info = Info::new(0, &info_contents[0]);

        let mut a = Agent::new();
        a.prospect.reset(-0.1, -2.0, -0.001);
        a.cpt.reset(0.88, 0.88, 2.25, 0.61, 0.69);
        a.reset(
            0.5,
            0.5,
            0.5,
            0.0,
            agent_params.initial_opinions.clone(),
            agent_params.initial_conditions.clone(),
            &agent_params.base_rates,
        );

        println!("{:?}", a.set_info_opinions(&info, &agent_params.base_rates));
    }
}
