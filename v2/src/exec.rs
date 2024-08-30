use std::{borrow::Cow, collections::BTreeMap};

use base::{
    agent::Decision,
    executor::{AgentExtTrait, AgentIdx, Executor, InfoIdx, InstanceExt, InstanceWrapper},
    info::{InfoContent, InfoLabel},
    opinion::{MyFloat, MyOpinions, OtherTrusts, Trusts},
};
use graph_lib::prelude::{Graph, GraphB};
use rand::{seq::IteratorRandom, seq::SliceRandom, Rng};
use rand_distr::{Distribution, Exp1, Open01, Standard, StandardNormal};

use crate::config::*;

pub struct Exec<V: MyFloat> {
    pub enable_inhibition: bool,
    pub graph: GraphB,
    pub fnum_agents: V,
    pub mean_degree: V,
    pub sharer_trust: SharerTrustSamples<V>,
    pub opinion: OpinionSamples<V>,
    pub information: InformationSamples<V>,
    pub informing: InformingParams<V>,
    pub community_psi1: SupportLevels<V>,
    pub probabilies: ProbabilitySamples<V>,
    pub prospect: ProspectSamples<V>,
    pub cpt: CptSamples<V>,
}

impl<V> Executor<V, AgentExt<V>, Instance> for Exec<V>
where
    V: MyFloat,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
{
    fn num_agents(&self) -> usize {
        self.graph.node_count()
    }

    fn graph(&self) -> &GraphB {
        &self.graph
    }
}

#[derive(Default)]
pub struct AgentExt<V> {
    trusts: InfoMap<V>,
    arrival_prob: Option<V>,
    /// (friend, social)
    viewing_probs: Option<(V, V)>,
    /// (friend, social)
    plural_ignores: Option<(V, V)>,
    visit_prob: Option<V>,
    psi1_support_level: V,
}

impl<V> AgentExt<V> {
    fn get_trust<R: Rng>(&mut self, label: InfoLabel, exec: &Exec<V>, rng: &mut R) -> V
    where
        V: MyFloat,
    {
        *self.trusts.entry(label).or_insert_with(|| match label {
            InfoLabel::Misinfo => {
                let p = self.psi1_support_level;
                let d = p.min(V::one() - p);
                let u = *exec.sharer_trust.misinfo.choose(rng).unwrap(); // 0<u<1
                let x = d * (u * (V::one() + V::one()) - V::one()); // -d < x < d
                p + x // lv - d < lv + x < lv + d
            }
            InfoLabel::Corrective => {
                let p = V::one() - self.psi1_support_level; // opposite of support level
                let d = p.min(V::one() - p);
                let u = *exec.sharer_trust.correction.choose(rng).unwrap(); // 0<u<1
                let x = d * (u * (V::one() + V::one()) - V::one()); // -d < x < d
                p + x // lv - d < lv + x < lv + d
            }
            InfoLabel::Observed => *exec.sharer_trust.obserbation.choose(rng).unwrap(),
            InfoLabel::Inhibitive => *exec.sharer_trust.inhibition.choose(rng).unwrap(),
        })
    }

    fn get_plural_ignorances<'a, R>(&mut self, exec: &Exec<V>, rng: &mut R) -> (V, V)
    where
        V: MyFloat,
        R: Rng,
    {
        *self.plural_ignores.get_or_insert_with(|| {
            (
                *exec.probabilies.plural_ignore_friend.choose(rng).unwrap(),
                *exec.probabilies.plural_ignore_social.choose(rng).unwrap(),
            )
        })
    }

    fn viewing_probs<'a, R>(&mut self, exec: &Exec<V>, rng: &mut R) -> (V, V)
    where
        V: MyFloat,
        R: Rng,
    {
        *self.viewing_probs.get_or_insert_with(|| {
            (
                *exec.probabilies.viewing_friend.choose(rng).unwrap(),
                *exec.probabilies.viewing_social.choose(rng).unwrap(),
            )
        })
    }

    fn arrival_prob<'a, R>(&mut self, exec: &Exec<V>, rng: &mut R) -> V
    where
        V: MyFloat,
        R: Rng,
    {
        *self
            .arrival_prob
            .get_or_insert_with(|| *exec.probabilies.arrival_friend.choose(rng).unwrap())
    }
}

fn new_trusts<V: MyFloat>(
    my_trust: V,
    trust_mis: V,
    receipt_prob: V,
    friend_viewing_probs: V,
    social_viewing_probs: V,
    arrival_prob: V,
    friend_plural_ignorance: V,
    social_plural_ignorance: V,
) -> Trusts<V> {
    Trusts {
        my_trust,
        social_trusts: OtherTrusts {
            trust: my_trust,
            certainty: social_viewing_probs * receipt_prob,
        },
        friend_trusts: OtherTrusts {
            trust: my_trust,
            certainty: friend_viewing_probs * receipt_prob,
        },
        pred_friend_trusts: OtherTrusts {
            trust: my_trust,
            certainty: friend_viewing_probs * arrival_prob,
        },
        social_misinfo_trusts: OtherTrusts {
            trust: trust_mis,
            certainty: social_plural_ignorance,
        },
        friend_misinfo_trusts: OtherTrusts {
            trust: trust_mis,
            certainty: friend_plural_ignorance,
        },
    }
}

impl<V> AgentExtTrait<V> for AgentExt<V>
where
    V: MyFloat,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    Open01: Distribution<V>,
{
    type Exec = Exec<V>;
    type Ix = Instance;

    fn informer_trusts<'a, R: Rng>(
        &mut self,
        ins: &mut InstanceWrapper<'a, Self::Exec, V, R, Self::Ix>,
        _: InfoIdx,
    ) -> Trusts<V> {
        let my_trust = V::one();
        let trust_mis = self.get_trust(InfoLabel::Misinfo, ins.exec, &mut ins.rng);
        let (fpi, kpi) = self.get_plural_ignorances(ins.exec, &mut ins.rng);
        new_trusts(
            my_trust,
            trust_mis,
            V::zero(),
            V::zero(),
            V::zero(),
            V::zero(),
            fpi,
            kpi,
        )
    }

    fn sharer_trusts<'a, R: Rng>(
        &mut self,
        ins: &mut InstanceWrapper<'a, Self::Exec, V, R, Self::Ix>,
        info_idx: InfoIdx,
    ) -> Trusts<V> {
        let my_trust = self.get_trust(*ins.get_info_label(info_idx), ins.exec, &mut ins.rng);
        let trust_mis = self.get_trust(InfoLabel::Misinfo, ins.exec, &mut ins.rng);
        let r = Instance::receipt_prob(ins, info_idx);
        let (fq, kq) = self.viewing_probs(ins.exec, &mut ins.rng);
        let arr = self.arrival_prob(ins.exec, &mut ins.rng);
        let (fpi, kpi) = self.get_plural_ignorances(ins.exec, &mut ins.rng);
        new_trusts(my_trust, trust_mis, r, fq, kq, arr, fpi, kpi)
    }

    fn visit_prob<R: Rng>(&mut self, exec: &Self::Exec, rng: &mut R) -> V {
        *self
            .visit_prob
            .get_or_insert_with(|| *exec.probabilies.viewing.choose(rng).unwrap())
    }

    fn reset_core<R: Rng>(
        ops: &mut MyOpinions<V>,
        decision: &mut Decision<V>,
        exec: &Self::Exec,
        rng: &mut R,
    ) {
        decision.reset(0, |prospect, cpt| {
            exec.opinion.reset_to(ops, rng);
            exec.prospect.reset_to(prospect, rng);
            exec.cpt.reset_to(cpt, rng);
        });
    }

    fn reset<R: Rng>(&mut self, idx: usize, exec: &Exec<V>, _: &mut R) {
        self.trusts = InfoMap::new();
        self.arrival_prob = None;
        self.viewing_probs = None;
        self.plural_ignores = None;
        self.visit_prob = None;
        self.psi1_support_level = exec.community_psi1.level(idx);
    }
}

#[derive(Default)]
struct InfoMap<T> {
    misinfo: Option<T>,
    correction: Option<T>,
    observation: Option<T>,
    inhibition: Option<T>,
}

impl<T> InfoMap<T> {
    fn new() -> Self {
        Self {
            misinfo: None,
            correction: None,
            observation: None,
            inhibition: None,
        }
    }
    fn entry(&mut self, label: InfoLabel) -> MyEntry<'_, T> {
        let value = match label {
            InfoLabel::Misinfo => &mut self.misinfo,
            InfoLabel::Corrective => &mut self.correction,
            InfoLabel::Observed => &mut self.observation,
            InfoLabel::Inhibitive => &mut self.inhibition,
        };
        MyEntry { value }
    }
}

struct MyEntry<'a, T> {
    value: &'a mut Option<T>,
}

impl<'a, T> MyEntry<'a, T> {
    fn or_insert_with<F: FnOnce() -> T>(self, default: F) -> &'a mut T {
        self.value.get_or_insert_with(default)
    }
}

pub struct Instance {
    misinfo_informers: BTreeMap<u32, Vec<AgentIdx>>,
    corection_informers: BTreeMap<u32, Vec<AgentIdx>>,
    inhibition_informers: BTreeMap<u32, Vec<AgentIdx>>,
    observation: Vec<AgentIdx>,
    max_step_num_observation: usize,
}

impl Instance {
    fn receipt_prob<'a, V, R>(
        ins: &InstanceWrapper<'a, Exec<V>, V, R, Self>,
        info_idx: InfoIdx,
    ) -> V
    where
        V: MyFloat,
    {
        V::one()
            - (V::one() - V::from_usize(ins.num_shared(info_idx)).unwrap() / ins.exec.fnum_agents)
                .powf(ins.exec.mean_degree)
    }
}

impl<V, R> InstanceExt<V, R, Exec<V>> for Instance
where
    V: MyFloat,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    R: Rng,
{
    fn from_exec(exec: &Exec<V>, rng: &mut R) -> Self {
        let mut is_observation = vec![true; exec.graph.node_count()];

        fn make_informers<V: MyFloat, R: Rng>(
            exec: &Exec<V>,
            rng: &mut R,
            sampling: &Sampling<V>,
            informings: &[Informing<V>],
            is_observation: &mut [bool],
        ) -> BTreeMap<u32, Vec<AgentIdx>> {
            let mut informers = BTreeMap::new();
            let samples = match sampling {
                &Sampling::Random(p) => exec
                    .community_psi1
                    .random(V::to_usize(&(exec.fnum_agents * p)).unwrap(), rng),
                &Sampling::Top(p) => exec
                    .community_psi1
                    .top(V::to_usize(&(exec.fnum_agents * p)).unwrap(), rng),
                &Sampling::Middle(p) => exec
                    .community_psi1
                    .middle(V::to_usize(&(exec.fnum_agents * p)).unwrap(), rng),
                &Sampling::Bottom(p) => exec
                    .community_psi1
                    .bottom(V::to_usize(&(exec.fnum_agents * p)).unwrap(), rng),
            };
            let mut iter = samples.into_iter();
            for Informing { step, pop_agents } in informings {
                if iter.len() == 0 {
                    continue;
                }
                if informers.contains_key(step) {
                    continue;
                }
                let n = V::to_usize(&(*pop_agents * exec.fnum_agents)).unwrap();
                let agents = (0..n)
                    .flat_map(|_| {
                        iter.next().map(|i| {
                            is_observation[i] = false;
                            AgentIdx(i)
                        })
                    })
                    .collect::<Vec<_>>();
                informers.insert(*step, agents);
            }
            informers
        }

        let misinfo_informers = make_informers(
            exec,
            rng,
            &Sampling::Top(exec.informing.max_pop_misinfo),
            &exec.informing.misinfo,
            &mut is_observation,
        );
        let corection_informers = make_informers(
            exec,
            rng,
            &Sampling::Bottom(exec.informing.max_pop_correction),
            &exec.informing.correction,
            &mut is_observation,
        );
        let inhibition_informers = make_informers(
            exec,
            rng,
            &exec.informing.max_pop_inhibition,
            &exec.informing.inhibition,
            &mut is_observation,
        );
        let mut observation = is_observation
            .into_iter()
            .enumerate()
            .filter_map(|(i, b)| if b { Some(i.into()) } else { None })
            .take(V::to_usize(&(exec.informing.max_pop_observation * exec.fnum_agents)).unwrap())
            .collect::<Vec<_>>();
        observation.shuffle(rng);

        Self {
            misinfo_informers,
            corection_informers,
            inhibition_informers: if exec.enable_inhibition {
                inhibition_informers
            } else {
                BTreeMap::new()
            },
            observation,
            max_step_num_observation: V::to_usize(
                &(exec.informing.max_step_pop_observation * exec.fnum_agents),
            )
            .unwrap(),
        }
    }

    fn is_continued(&self, _: &Exec<V>) -> bool {
        !self.misinfo_informers.is_empty()
            || !self.corection_informers.is_empty()
            || !self.inhibition_informers.is_empty()
    }

    fn get_informers_with<'a>(
        ins: &mut InstanceWrapper<'a, Exec<V>, V, R, Self>,
        t: u32,
    ) -> Vec<(AgentIdx, InfoContent<'a, V>)> {
        let mut informers = Vec::new();
        if let Some(d) = ins.ext.misinfo_informers.remove(&t) {
            for agent_idx in d {
                informers.push((
                    agent_idx.into(),
                    InfoContent::Misinfo {
                        op: Cow::Borrowed(
                            ins.exec.information.misinfo.choose(&mut ins.rng).unwrap(),
                        ),
                    },
                ));
            }
        }
        if let Some(d) = ins.ext.corection_informers.remove(&t) {
            for agent_idx in d {
                informers.push((
                    agent_idx.into(),
                    InfoContent::Correction {
                        op: Cow::Borrowed(
                            ins.exec
                                .information
                                .correction
                                .choose(&mut ins.rng)
                                .unwrap(),
                        ),
                        misinfo: Cow::Borrowed(
                            ins.exec.information.misinfo.choose(&mut ins.rng).unwrap(),
                        ),
                    },
                ));
            }
        }
        if !ins.ext.observation.is_empty() {
            let p = V::from_usize(ins.prev_num_selfish()).unwrap() / ins.exec.fnum_agents;
            for _ in 0..ins.ext.max_step_num_observation {
                if ins.exec.informing.prob_post_observation * p <= ins.rng.gen() {
                    continue;
                }
                if let Some(agent_idx) = ins.ext.observation.pop() {
                    let op = ins
                        .exec
                        .information
                        .observation
                        .iter()
                        .choose(&mut ins.rng)
                        .unwrap();
                    informers.push((
                        agent_idx,
                        InfoContent::Observation {
                            op: Cow::Borrowed(op),
                        },
                    ));
                }
            }
        }
        if let Some(d) = ins.ext.inhibition_informers.remove(&t) {
            for agent_idx in d {
                let (op1, op2, op3) = ins
                    .exec
                    .information
                    .inhibition
                    .choose(&mut ins.rng)
                    .unwrap();
                informers.push((
                    agent_idx.into(),
                    InfoContent::Inhibition {
                        op1: Cow::Borrowed(op1),
                        op2: Cow::Borrowed(op2),
                        op3: Cow::Borrowed(op3),
                    },
                ));
            }
        }

        informers
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use base::{
        agent::Agent,
        executor::InstanceExt,
        info::{Info, InfoContent},
        opinion::{FPsi, FH, FO},
    };
    use rand::{rngs::SmallRng, SeedableRng};
    use subjective_logic::{
        marr_d1, marr_d2,
        mul::{
            labeled::{OpinionD1, SimplexD1},
            MergeJointConditions2, Simplex,
        },
        multi_array::labeled::{MArrD1, MArrD2},
    };

    use super::{new_trusts, Config, Exec, Instance};

    #[test]
    fn test_instance() -> anyhow::Result<()> {
        let config = Config::try_new(
            "./test/network_config.toml",
            "./test/agent_config.toml",
            "./test/strategy_config.toml",
        )?;
        let exec: Exec<f32> = config.into_exec(true)?;
        let mut rng = SmallRng::seed_from_u64(0);
        let ins = Instance::from_exec(&exec, &mut rng);
        for t in [0, 1, 2] {
            assert_eq!(ins.misinfo_informers.get(&t).unwrap().len(), 2);
        }
        for t in [2, 3, 4] {
            assert_eq!(ins.corection_informers.get(&t).unwrap().len(), 2);
        }
        for t in [4, 5, 6] {
            assert_eq!(ins.inhibition_informers.get(&t).unwrap().len(), 2);
        }
        assert_eq!(ins.observation.len(), 40);
        Ok(())
    }

    #[tracing_test::traced_test]
    #[test]
    fn test_agent() -> anyhow::Result<()> {
        let mut agent = Agent::<f32>::default();
        agent.reset(|ops, dec| {
            dec.reset(0, |prs, cpt| {
                prs.reset(-1.0, -4.00, -0.001);
                cpt.reset(0.88, 0.88, 2.25, 0.61, 0.69);
            });
            let o_b = marr_d1![
                Simplex::new(marr_d1![0.95, 0.0], 0.05),
                Simplex::new(marr_d1![0.0, 0.95], 0.05)
            ];
            let b_kh = marr_d1![
                Simplex::new(marr_d1![0.95, 0.0], 0.05),
                Simplex::new(marr_d1![0.0, 0.9], 0.1)
            ];
            let a_fh = marr_d1![
                Simplex::new(marr_d1![0.95, 0.0], 0.05),
                Simplex::new(marr_d1![0.0, 0.9], 0.1)
            ];
            let theta_h = marr_d1![
                Simplex::new(marr_d1![0.5, 0.0], 0.5),
                Simplex::new(marr_d1![0.0, 0.90], 0.1)
            ];
            let thetad_h = marr_d1![
                Simplex::new(marr_d1![0.8, 0.0], 0.2),
                Simplex::new(marr_d1![0.0, 0.8], 0.2)
            ];
            let h_psi_if_phi0 = marr_d1![
                Simplex::new(marr_d1![0.25, 0.25], 0.5),
                Simplex::new(marr_d1![0.0, 0.9], 0.1)
            ];
            let h_b_if_phi0 = marr_d1![
                Simplex::new(marr_d1![0.7, 0.1], 0.2),
                Simplex::new(marr_d1![0.0, 0.925], 0.075)
            ];
            let uncertainty_fh_fpsi_if_fphi0 = marr_d1![0.3, 0.3];
            let uncertainty_kh_kpsi_if_kphi0 = marr_d1![0.3, 0.3];
            let uncertainty_fh_fphi_fo = marr_d2![[0.3, 0.3], [0.3, 0.3]];
            let uncertainty_kh_kphi_ko = marr_d2![[0.3, 0.3], [0.3, 0.3]];
            ops.fixed.reset(
                o_b,
                b_kh,
                a_fh,
                theta_h,
                thetad_h,
                h_psi_if_phi0,
                h_b_if_phi0,
                uncertainty_fh_fpsi_if_fphi0,
                uncertainty_kh_kpsi_if_kphi0,
                uncertainty_fh_fphi_fo,
                uncertainty_kh_kphi_ko,
            );
            ops.state.reset(
                OpinionD1::vacuous_with(vec![0.95, 0.05].try_into().unwrap()),
                OpinionD1::vacuous_with(vec![0.95, 0.05].try_into().unwrap()),
                OpinionD1::vacuous_with(vec![0.95, 0.05].try_into().unwrap()),
                OpinionD1::vacuous_with(vec![0.95, 0.05].try_into().unwrap()),
                OpinionD1::vacuous_with(vec![0.95, 0.05].try_into().unwrap()),
                vec![SimplexD1::vacuous(), SimplexD1::vacuous()]
                    .try_into()
                    .unwrap(),
                vec![SimplexD1::vacuous(), SimplexD1::vacuous()]
                    .try_into()
                    .unwrap(),
                OpinionD1::vacuous_with(vec![0.95, 0.05].try_into().unwrap()),
                OpinionD1::vacuous_with(vec![0.95, 0.05].try_into().unwrap()),
                vec![SimplexD1::vacuous(), SimplexD1::vacuous()]
                    .try_into()
                    .unwrap(),
                OpinionD1::vacuous_with(vec![0.95, 0.05].try_into().unwrap()),
                OpinionD1::vacuous_with(vec![0.95, 0.05].try_into().unwrap()),
                vec![SimplexD1::vacuous(), SimplexD1::vacuous()]
                    .try_into()
                    .unwrap(),
            );
            ops.ded.reset(
                OpinionD1::vacuous_with(marr_d1![0.95, 0.05].try_into().unwrap()),
                OpinionD1::vacuous_with(marr_d1![0.80, 0.2].try_into().unwrap()),
                OpinionD1::vacuous_with(marr_d1![0.80, 0.2].try_into().unwrap()),
                OpinionD1::vacuous_with(marr_d1![0.5, 0.5].try_into().unwrap()),
                OpinionD1::vacuous_with(marr_d1![0.5, 0.5].try_into().unwrap()),
                OpinionD1::vacuous_with(marr_d1![0.90, 0.1].try_into().unwrap()),
                OpinionD1::vacuous_with(marr_d1![0.90, 0.1].try_into().unwrap()),
            );
        });

        let p = InfoContent::Correction {
            op: Cow::Owned(OpinionD1::new(
                marr_d1![0.95, 0.0],
                0.05,
                marr_d1![0.95, 0.05],
            )),
            misinfo: Cow::Owned(OpinionD1::new(
                marr_d1![0.0, 0.90],
                0.10,
                marr_d1![0.3, 0.7],
            )),
        };
        let info = Info::new(0, p);
        let t = new_trusts(1.0, 0.9, 0.5, 0.05, 0.05, 0.8, 0.9, 0.5);
        agent.read_info(&info, t);
        Ok(())
    }

    #[test]
    fn test_cond() {
        let c0 = marr_d1!(FPsi; [
            SimplexD1::new(marr_d1!(FH;[0.52, 0.18]), 0.3),
            SimplexD1::new(marr_d1![0.07, 0.63], 0.3),
            // SimplexD1::new(marr_d1![0.02, 0.93], 0.05),
        ]);
        let c1 = marr_d1!(FO; [
            SimplexD1::new(marr_d1!(FH;[0.523, 0.177]), 0.3),
            SimplexD1::new(marr_d1![0.023, 0.677], 0.3),
            // SimplexD1::new(marr_d1![0.02, 0.93], 0.05),
        ]);
        let c: MArrD2<FPsi, FO, SimplexD1<FH, f32>> = MArrD1::<FH, _>::merge_cond2(
            &c0,
            &c1,
            &marr_d1![0.95, 0.05],
            &marr_d1![0.95, 0.05],
            &marr_d1![0.80, 0.20],
        );
        println!("{:?}", c[(FPsi(0), FO(0))]);
        println!("{:?}", c[(FPsi(0), FO(1))]);
        println!("{:?}", c[(FPsi(1), FO(0))]);
        println!("{:?}", c[(FPsi(1), FO(1))]);
    }
}
