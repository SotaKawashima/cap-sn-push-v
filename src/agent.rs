use approx::UlpsEq;
use either::Either;
use log::debug;
use num_traits::{Float, NumAssign};
use rand::Rng;
use rand_distr::{uniform::SampleUniform, Distribution, Open01, Standard};
use serde_with::{serde_as, TryFromInto};
use std::{array, collections::BTreeMap, fmt, iter::Sum};

use crate::{
    cpt::{CptParams, Prospect, CPT},
    dist::{sample, Dist, DistParam},
    info::{Info, TrustDists},
    opinion::{
        compute_opinions, reset_opinions, FriendOpinions, GlobalBaseRates, InitialOpinions,
        Opinions,
    },
};

#[serde_as]
#[derive(serde::Deserialize)]
pub struct AgentParams<V>
where
    V: Float + NumAssign + UlpsEq + SampleUniform,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
{
    pub initial_opinions: InitialOpinions<V>,
    pub base_rates: GlobalBaseRates<V>,
    #[serde_as(as = "TryFromInto<DistParam<V>>")]
    pub read_dist: Either<V, Dist<V>>,
    #[serde_as(as = "TryFromInto<DistParam<V>>")]
    pub farrival_dist: Either<V, Dist<V>>,
    #[serde_as(as = "TryFromInto<DistParam<V>>")]
    pub fread_dist: Either<V, Dist<V>>,
    #[serde_as(as = "TryFromInto<DistParam<V>>")]
    pub pi_dist: Either<V, Dist<V>>,
    /// probability whether people has plural ignorance
    pub pi_prob: V,
    pub trust_dists: TrustDists<V>,
    pub cpt_params: CptParams<V>,
}

#[derive(Debug)]
pub struct Behavior {
    pub selfish: bool,
    pub sharing: bool,
}

pub struct Agent<V: Float> {
    pub cpt: CPT<V>,
    pub prospect: Prospect<V>,
    op: Opinions<V>,
    fop: FriendOpinions<V>,
    read_prob: V,
    friend_arrival_prob: V,
    friend_read_prob: V,
    done_selfish: bool,
    info_trust_map: BTreeMap<usize, V>,
    info_shared: BTreeMap<usize, bool>,
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
            op: Default::default(),
            fop: Default::default(),
            read_prob: V::zero(),
            friend_arrival_prob: V::zero(),
            friend_read_prob: V::zero(),
            done_selfish: Default::default(),
            info_trust_map: Default::default(),
            info_shared: Default::default(),
        }
    }

    pub fn reset(
        &mut self,
        read_prob: V,
        friend_arrival_prob: V,
        friend_read_prob: V,
        pi_rate: V,
        initial_opinions: InitialOpinions<V>,
        base_rates: &GlobalBaseRates<V>,
    ) where
        V: NumAssign + Default,
        // + Sum + Default + std::fmt::Debug + SampleUniform,
    {
        self.read_prob = read_prob;
        self.friend_arrival_prob = friend_arrival_prob;
        self.friend_read_prob = friend_read_prob;

        reset_opinions(
            &mut self.op,
            &mut self.fop,
            pi_rate,
            initial_opinions,
            base_rates,
        );

        self.info_trust_map.clear();
        self.info_shared.clear();
        self.done_selfish = false;
    }

    pub fn reset_with<R>(&mut self, agent_params: &AgentParams<V>, rng: &mut R)
    where
        V: NumAssign + Sum + Default,
        R: Rng,
    {
        self.cpt.reset_with(&agent_params.cpt_params, rng);
        self.prospect.reset_with(&agent_params.cpt_params, rng);

        let r = rng.gen::<V>();
        let s = sample(&agent_params.pi_dist, rng);
        self.reset(
            sample(&agent_params.read_dist, rng),
            sample(&agent_params.farrival_dist, rng),
            sample(&agent_params.fread_dist, rng),
            if agent_params.pi_prob > r {
                s
            } else {
                V::zero()
            },
            agent_params.initial_opinions.clone(),
            &agent_params.base_rates,
        );
    }

    pub fn read_info_trustfully(
        &mut self,
        info: &Info<V>,
        receipt_prob: V,
        agent_params: &AgentParams<V>,
    ) -> Behavior
    where
        V: Sum + Default + fmt::Debug + NumAssign,
    {
        self.read_info_with_trust(info, receipt_prob, &agent_params.base_rates, V::one())
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
    {
        let trust = *self
            .info_trust_map
            .entry(info.idx)
            .or_insert_with(|| agent_params.trust_dists.gen_map(rng)(info));
        if rng.gen::<V>() <= self.read_prob {
            Some(self.read_info_with_trust(info, receipt_prob, &agent_params.base_rates, trust))
        } else {
            None
        }
    }

    fn read_info_with_trust(
        &mut self,
        info: &Info<V>,
        receipt_prob: V,
        base_rates: &GlobalBaseRates<V>,
        trust: V,
    ) -> Behavior
    where
        V: Sum + Default + fmt::Debug + NumAssign,
    {
        let mut temp = compute_opinions(
            &mut self.op,
            &mut self.fop,
            info,
            self.friend_read_prob,
            receipt_prob,
            trust,
            base_rates,
        );
        // compute values of prospects
        let shared = self.info_shared.entry(info.idx).or_insert(false);
        let sharing = if *shared {
            false
        } else {
            let (prob, pred_prob) = temp.predicate_friend_opinions(
                &self.op,
                &self.fop,
                info,
                self.friend_arrival_prob * self.friend_read_prob * trust,
                base_rates,
            );
            let value_sharing: [V; 2] = [
                self.cpt.valuate(&self.prospect.sharing[0], &prob),
                self.cpt.valuate(&self.prospect.sharing[1], &pred_prob),
            ];
            debug!("V_Y:{:?}", value_sharing);
            *shared = value_sharing[0] < value_sharing[1];
            *shared
        };

        let theta_prob = temp.update_for_sharing(&mut self.op, &mut self.fop);
        let selfish = if self.done_selfish {
            false
        } else {
            let value_selfish: [V; 2] =
                array::from_fn(|i| self.cpt.valuate(&self.prospect.selfish[i], &theta_prob));
            debug!("V_X:{:?}", value_selfish);
            self.done_selfish = value_selfish[0] < value_selfish[1];
            self.done_selfish
        };

        Behavior { selfish, sharing }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        agent::{Agent, AgentParams},
        cpt::CptParams,
        info::{Info, InfoContent, InfoObject, TrustDists},
        opinion::{GlobalBaseRates, InitialOpinions},
    };

    use either::Either::Left;
    use subjective_logic::{harr2, harr3, mul::Simplex};

    #[test]
    fn test_agent() {
        let agent_params = AgentParams {
            initial_opinions: InitialOpinions {
                theta: Simplex::vacuous(),
                psi: Simplex::vacuous(),
                phi: Simplex::vacuous(),
                s: Simplex::vacuous(),
                cond_theta_phi: [Simplex::vacuous(), Simplex::vacuous()],
                fs: Simplex::vacuous(),
                fphi: Simplex::vacuous(),
                cond_ftheta_fphi: [Simplex::vacuous(), Simplex::vacuous()],
                cond_pa: [
                    Simplex::new([0.90, 0.00], 0.10),
                    Simplex::new([0.00, 0.99], 0.01),
                    Simplex::new([0.90, 0.00], 0.10),
                ],
                cond_theta: harr3![
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
            read_dist: Left(0.5),
            farrival_dist: Left(0.5),
            fread_dist: Left(0.5),
            pi_dist: Left(0.5),
            pi_prob: 0.5,
            trust_dists: TrustDists {
                misinfo: Left(0.5),
                corrective: Left(0.5),
                observed: Left(0.5),
                inhibitive: Left(0.5),
            },
            cpt_params: CptParams {
                x0_dist: Left(0.5),
                x1_dist: Left(0.5),
                y_dist: Left(0.5),
                alpha: Left(0.88),
                beta: Left(0.88),
                lambda: Left(2.25),
                gamma: Left(0.61),
                delta: Left(0.69),
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
            &agent_params.base_rates,
        );

        let receipt_prob = 0.0;
        println!(
            "{:?}",
            a.read_info_trustfully(&info, receipt_prob, &agent_params)
        );
    }
}
