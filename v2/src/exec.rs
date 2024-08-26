use std::{borrow::Cow, vec::Drain};

use base::{
    executor::{
        AgentExtTrait, AgentIdx, AgentWrapper, Executor, InfoIdx, InstanceExt, InstanceWrapper,
    },
    info::{InfoContent, InfoLabel},
    opinion::{AccessProb, MyFloat, Trusts},
};
use graph_lib::prelude::{Graph, GraphB};
use input::format::DataFormat;
use rand::{seq::IteratorRandom, seq::SliceRandom, Rng};
use rand_distr::{Distribution, Exp1, Open01, Standard, StandardNormal};
use serde::Deserialize;

use crate::config::*;

pub struct Exec<V: MyFloat> {
    graph: GraphB,
    fnum_agents: V,
    mean_degree: V,
    sharer_trust: SharerTrustSamples<V>,
    condition: ConditionSamples<V>,
    uncertainty: UncertaintySamples<V>,
    initial_opinions: InitialOpinions<V>,
    initial_base_rates: InitialBaseRates<V>,
    information: InformationSamples<V>,
    // informer: InformerParams,
    informing: InformingParams,
    /// descending ordered by level
    community_psi1: SupportLevels<V>,
    prob_post_observation: V,
    probabilies: ProbabilityParams<V>,
    prospect: ProspectSamples<V>,
    cpt: CptSamples<V>,
}

impl<V> Exec<V>
where
    V: MyFloat + for<'a> Deserialize<'a>,
{
    pub fn try_new(config: Config<V>) -> anyhow::Result<Self> {
        let graph: GraphB = config.graph.try_into()?;
        let fnum_agents = V::from_usize(graph.node_count()).unwrap();
        let mean_degree = V::from_usize(graph.edge_count()).unwrap() / fnum_agents;
        Ok(Self {
            graph,
            fnum_agents,
            mean_degree,
            sharer_trust: DataFormat::read(&config.sharer_trust_path)?.parse()?,
            condition: config.condition.try_into()?,
            uncertainty: config.uncertainty.try_into()?,
            initial_opinions: config.initial_opinions,
            initial_base_rates: config.initial_base_rate,
            information: config.information.try_into()?,
            informing: DataFormat::read(&config.informing_path)?.parse()?,
            community_psi1: SupportLevels::conv_from(config.community_psi1_path)?,
            prob_post_observation: config.prob_post_observation,
            probabilies: DataFormat::read(&config.probabilities_path)?.parse()?,
            prospect: ProspectSamples::conv_from(config.prospect_path)?,
            cpt: CptSamples::conv_from(config.cpt_path)?,
        })
    }
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
}

impl<V> AgentExt<V> {
    fn clear(&mut self) {
        self.trusts = InfoMap::new();
        self.arrival_prob = None;
        self.viewing_probs = None;
        self.plural_ignores = None;
        self.visit_prob = None;
    }

    fn get_trust<R: Rng>(&mut self, label: InfoLabel, exec: &Exec<V>, rng: &mut R) -> V
    where
        V: MyFloat,
    {
        *self.trusts.entry(label).or_insert_with(|| {
            *match label {
                InfoLabel::Misinfo => &exec.sharer_trust.misinfo,
                InfoLabel::Corrective => &exec.sharer_trust.correction,
                InfoLabel::Observed => &exec.sharer_trust.obserbation,
                InfoLabel::Inhibitive => &exec.sharer_trust.inhibition,
            }
            .choose(rng)
            .unwrap()
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
        let trust_mis = self.get_trust(InfoLabel::Misinfo, ins.exec, &mut ins.rng);
        Trusts {
            p: V::one(),
            fp: V::one(),
            kp: V::one(),
            fm: trust_mis,
            km: trust_mis,
        }
    }

    fn informer_access_probs<'a, R: Rng>(
        &mut self,
        ins: &mut InstanceWrapper<'a, Self::Exec, V, R, Self::Ix>,
        _: InfoIdx,
    ) -> AccessProb<V> {
        let (fm, km) = self.get_plural_ignorances(ins.exec, &mut ins.rng);
        AccessProb {
            fp: V::zero(),
            kp: V::zero(),
            pred_fp: V::zero(),
            fm,
            km,
        }
    }

    fn sharer_trusts<'a, R: Rng>(
        &mut self,
        ins: &mut InstanceWrapper<'a, Self::Exec, V, R, Self::Ix>,
        info_idx: InfoIdx,
    ) -> Trusts<V> {
        let trust = self.get_trust(*ins.get_info_label(info_idx), ins.exec, &mut ins.rng);
        let trust_mis = self.get_trust(InfoLabel::Misinfo, ins.exec, &mut ins.rng);
        Trusts {
            p: trust,
            fp: trust,
            kp: trust,
            fm: trust_mis,
            km: trust_mis,
        }
    }

    fn sharer_access_probs<'a, R: Rng>(
        &mut self,
        ins: &mut InstanceWrapper<'a, Self::Exec, V, R, Self::Ix>,
        info_idx: InfoIdx,
    ) -> AccessProb<V> {
        let r = Instance::receipt_prob(ins, info_idx);
        let (fq, kq) = self.viewing_probs(ins.exec, &mut ins.rng);
        let arr = self.arrival_prob(ins.exec, &mut ins.rng);
        let (fm, km) = self.get_plural_ignorances(ins.exec, &mut ins.rng);
        AccessProb {
            fp: fq * r,
            kp: kq * r,
            pred_fp: fq * arr,
            fm,
            km,
        }
    }

    fn visit_prob<R: Rng>(&mut self, exec: &Self::Exec, rng: &mut R) -> V {
        *self
            .visit_prob
            .get_or_insert_with(|| *exec.probabilies.viewing.choose(rng).unwrap())
    }

    fn reset<R: Rng>(wrapper: &mut AgentWrapper<V, Self>, exec: &Exec<V>, rng: &mut R) {
        wrapper.ext.clear();
        wrapper.core.reset(|ops, decision| {
            reset_fixed(&exec.condition, &exec.uncertainty, &mut ops.fixed, rng);
            exec.initial_opinions.clone().reset_to(&mut ops.state);
            exec.initial_base_rates.clone().reset_to(&mut ops.ded);
            decision.reset(0, |prospect, cpt| {
                exec.prospect.reset_to(prospect, rng);
                exec.cpt.reset_to(cpt, rng);
            });
        });
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
    /// desc ordered by support levels
    misinfo: Informers,
    /// asc ordered by support levels
    corection: Informers,
    observation: Vec<usize>,
    /// order is determined by inhibition sampling parameter
    inhibition: Informers,
}

struct Informers {
    agents: Vec<usize>,
    next_index: usize,
}

impl Informers {
    fn new(agents: Vec<usize>) -> Self {
        Self {
            agents,
            next_index: 0,
        }
    }
}

impl Informers {
    fn pick(&mut self, t: u32, informings: &[Informing]) -> Option<Drain<'_, usize>> {
        match informings.get(self.next_index) {
            Some(&Informing { step, num_agents }) if step == t => {
                self.next_index += 1;
                Some(self.agents.drain(0..(num_agents.min(self.agents.len()))))
            }
            _ => None,
        }
    }
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
        Self {
            misinfo: Informers::new(exec.community_psi1.top(exec.informing.informer.num_misinfo)),
            corection: Informers::new(
                exec.community_psi1
                    .bottom(exec.informing.informer.num_correction),
            ),
            observation: (0..exec.graph.node_count()).collect(),
            inhibition: match exec.informing.informer.inhibition {
                Sampling::Random(n) => Informers::new(exec.community_psi1.random(n, rng)),
                Sampling::Top(n) => Informers::new(exec.community_psi1.top(n)),
                Sampling::Middle(n) => Informers::new(exec.community_psi1.middle(n)),
                Sampling::Bottom(n) => Informers::new(exec.community_psi1.bottom(n)),
            },
        }
    }

    fn is_continued(&self, exec: &Exec<V>) -> bool {
        (self.misinfo.next_index < exec.informing.misinfo.len() - 1)
            && (self.corection.next_index < exec.informing.correction.len() - 1)
            && (self.inhibition.next_index < exec.informing.inhibition.len() - 1)
    }

    fn get_informers_with<'a>(
        ins: &mut InstanceWrapper<'a, Exec<V>, V, R, Self>,
        t: u32,
    ) -> Vec<(AgentIdx, InfoContent<'a, V>)> {
        let mut informers = Vec::new();
        if let Some(d) = ins.ext.misinfo.pick(t, &ins.exec.informing.misinfo) {
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
        if let Some(d) = ins.ext.corection.pick(t, &ins.exec.informing.correction) {
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
        if ins.total_num_selfish() > ins.exec.informing.obs_threshold_selfish {
            ins.ext.observation.retain(|&agent_idx| {
                if ins.exec.prob_post_observation <= ins.rng.gen() {
                    return true;
                }
                let op = ins
                    .exec
                    .information
                    .observation
                    .iter()
                    .choose(&mut ins.rng)
                    .unwrap();
                informers.push((
                    agent_idx.into(),
                    InfoContent::Observation {
                        op: Cow::Borrowed(op),
                    },
                ));
                false
            });
        }
        if let Some(d) = ins.ext.inhibition.pick(t, &ins.exec.informing.inhibition) {
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
