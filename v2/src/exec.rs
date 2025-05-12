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

use crate::parameter::{
    CptSamples, InformationSamples, Informing, InformingParams, OpinionSamples, PopSampleType,
    ProbabilitySamples, ProspectSamples, SharerTrustSamples, SupportLevelTable,
};

pub struct Exec<V: MyFloat>
where
    Open01: Distribution<V>,
{
    pub enable_inhibition: bool,
    pub graph: GraphB,
    pub fnum_agents: V,
    pub mean_degree: V,
    pub sharer_trust: SharerTrustSamples<V>,
    pub opinion: OpinionSamples<V>,
    pub information: InformationSamples<V>,
    pub informing: InformingParams<V>,
    pub community_psi1: SupportLevelTable<V>,
    pub probabilities: ProbabilitySamples<V>,
    pub prospect: ProspectSamples<V>,
    pub cpt: CptSamples<V>,
    pub delay_selfish: u32,
}

impl<V> Executor<V, AgentExt<V>, Instance> for Exec<V>
where
    V: MyFloat,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    <V as rand_distr::uniform::SampleUniform>::Sampler: Sync + Send,
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

impl<V> AgentExt<V>
where
    V: MyFloat,
    Open01: Distribution<V>,
    <V as rand_distr::uniform::SampleUniform>::Sampler: Sync + Send,
{
    fn get_trust<R: Rng>(&mut self, label: InfoLabel, exec: &Exec<V>, rng: &mut R) -> V {
        *self.trusts.entry(label).or_insert_with(|| match label {
            InfoLabel::Misinfo => {
                // let p = self.psi1_support_level;
                // let d = p.min(V::one() - p);
                // let u = *exec.sharer_trust.misinfo.choose(rng).unwrap(); // 0<u<1
                // let x = d * (u * (V::one() + V::one()) - V::one()); // -d < x < d
                // p + x // lv - d < lv + x < lv + d
                exec.sharer_trust.misinfo.choose(rng)
            }
            InfoLabel::Corrective => {
                // let p = V::one() - self.psi1_support_level; // opposite of support level
                // let d = p.min(V::one() - p);
                // let u = *exec.sharer_trust.correction.choose(rng).unwrap(); // 0<u<1
                // let x = d * (u * (V::one() + V::one()) - V::one()); // -d < x < d
                // p + x // lv - d < lv + x < lv + d
                exec.sharer_trust.correction.choose(rng)
            }
            InfoLabel::Observed => exec.sharer_trust.obserbation.choose(rng),
            InfoLabel::Inhibitive => exec.sharer_trust.inhibition.choose(rng),
        })
    }

    fn get_plural_ignorances<'a, R: Rng>(&mut self, exec: &Exec<V>, rng: &mut R) -> (V, V) {
        *self.plural_ignores.get_or_insert_with(|| {
            (
                exec.probabilities.plural_ignore_friend.choose(rng),
                exec.probabilities.plural_ignore_social.choose(rng),
            )
        })
    }

    fn viewing_probs<'a, R: Rng>(&mut self, exec: &Exec<V>, rng: &mut R) -> (V, V) {
        *self.viewing_probs.get_or_insert_with(|| {
            (
                exec.probabilities.viewing_friend.choose(rng),
                exec.probabilities.viewing_social.choose(rng),
            )
        })
    }

    fn arrival_prob<'a, R: Rng>(&mut self, exec: &Exec<V>, rng: &mut R) -> V {
        *self
            .arrival_prob
            .get_or_insert_with(|| exec.probabilities.arrival_friend.choose(rng))
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
    <V as rand_distr::uniform::SampleUniform>::Sampler: Sync + Send,
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
            .get_or_insert_with(|| exec.probabilities.viewing.choose(rng))
    }

    fn reset_core<R: Rng>(
        ops: &mut MyOpinions<V>,
        decision: &mut Decision<V>,
        exec: &Self::Exec,
        rng: &mut R,
    ) {
        decision.reset(exec.delay_selfish, |prospect, cpt| {
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
        Open01: Distribution<V>,
        <V as rand_distr::uniform::SampleUniform>::Sampler: Sync + Send,
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
    <V as rand_distr::uniform::SampleUniform>::Sampler: Sync + Send,
{
    fn from_exec(exec: &Exec<V>, rng: &mut R) -> Self {
        let mut is_observation = vec![true; exec.graph.node_count()];

        fn make_informers<V: MyFloat, R: Rng>(
            exec: &Exec<V>,
            rng: &mut R,
            sampling: &PopSampleType<V>,
            informings: &[Informing<V>],
            is_observation: &mut [bool],
        ) -> BTreeMap<u32, Vec<AgentIdx>>
        where
            Open01: Distribution<V>,
            <V as rand_distr::uniform::SampleUniform>::Sampler: Sync + Send,
        {
            let mut informers = BTreeMap::new();
            let samples = match sampling {
                &PopSampleType::Random(p) => exec
                    .community_psi1
                    .random(V::to_usize(&(exec.fnum_agents * p)).unwrap(), rng),
                &PopSampleType::Top(p) => exec
                    .community_psi1
                    .top(V::to_usize(&(exec.fnum_agents * p)).unwrap(), rng),
                &PopSampleType::Middle(p) => exec
                    .community_psi1
                    .middle(V::to_usize(&(exec.fnum_agents * p)).unwrap(), rng),
                &PopSampleType::Bottom(p) => exec
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
            &PopSampleType::Top(exec.informing.max_pop_misinfo),
            &exec.informing.misinfo,
            &mut is_observation,
        );
        let corection_informers = make_informers(
            exec,
            rng,
            &PopSampleType::Bottom(exec.informing.max_pop_correction),
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
        agent::{Agent, Decision},
        decision::{LevelSet, CPT},
        executor::InstanceExt,
        info::{Info, InfoContent},
        opinion::{FPsi, FixedOpinions, MyOpinions, Psi, Thetad, A, FH, FO},
    };
    use rand::{rngs::SmallRng, SeedableRng};
    use subjective_logic::{
        iter::FromFn,
        marr_d1, marr_d2,
        mul::{
            labeled::{OpinionD1, OpinionD2, OpinionRefD1, SimplexD1},
            InverseCondition, MergeJointConditions2, Simplex,
        },
        multi_array::labeled::{MArrD1, MArrD2},
        ops::{Product2, Projection},
    };

    use super::{new_trusts, Exec, Instance};
    use crate::config::Config;

    #[test]
    fn test_instance() -> anyhow::Result<()> {
        let config = Config::try_new(
            "./test/network_config.toml",
            "./test/agent_config.toml",
            "./test/strategy_config.toml",
        )?;
        let exec: Exec<f32> = config.into_exec(true, 0)?;
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

    fn fix_reset(fixed: &mut FixedOpinions<f32>) {
        let o_b = marr_d1![
            Simplex::new(marr_d1![0.95, 0.0], 0.05),
            Simplex::new(marr_d1![0.40, 0.2], 0.40)
        ];
        let a_fh = marr_d1![
            Simplex::new(marr_d1![0.95, 0.0], 0.05),
            Simplex::new(marr_d1![0.2, 0.6], 0.2)
        ];
        let b_kh = marr_d1![
            Simplex::new(marr_d1![0.95, 0.0], 0.05),
            Simplex::new(marr_d1![0.1, 0.5], 0.4)
        ];
        let theta_h = marr_d1![
            Simplex::new(marr_d1![0.95, 0.0], 0.05),
            Simplex::new(marr_d1![0.1, 0.5], 0.4)
        ];
        let thetad_h = marr_d1![
            Simplex::new(marr_d1![0.95, 0.0], 0.05),
            Simplex::new(marr_d1![0.1, 0.5], 0.4)
        ];
        let h_psi_if_phi0 = marr_d1![
            // Simplex::new(marr_d1![0.25, 0.25], 0.5),
            // Simplex::new(marr_d1![0.0, 0.0], 1.0),
            Simplex::new(marr_d1![0.0, 0.0], 1.0),
            Simplex::new(marr_d1![0.5, 0.25], 0.25)
        ];
        let h_b_if_phi0 = marr_d1![
            Simplex::new(marr_d1![0.5, 0.0], 0.5),
            Simplex::new(marr_d1![0.1, 0.8], 0.1)
        ];
        let uncertainty_fh_fpsi_if_fphi0 = marr_d1![0.3, 0.3];
        let uncertainty_kh_kpsi_if_kphi0 = marr_d1![0.3, 0.3];
        let uncertainty_fh_fphi_fo = marr_d2![[0.3, 0.3], [0.3, 0.3]];
        let uncertainty_kh_kphi_ko = marr_d2![[0.3, 0.3], [0.3, 0.3]];
        fixed.reset(
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
    }

    fn reset_agent(agent: &mut Agent<f32>) {
        agent.reset(|ops: &mut MyOpinions<f32>, dec: &mut Decision<f32>| {
            dec.reset(0, |prs, cpt| {
                prs.reset(-1.0, -8.00, -0.001);
                cpt.reset(0.88, 0.88, 2.25, 0.61, 0.69);
            });
            fix_reset(&mut ops.fixed);
            let psi = OpinionD1::vacuous_with(vec![0.95, 0.05].try_into().unwrap());
            let phi = OpinionD1::vacuous_with(vec![0.95, 0.05].try_into().unwrap());
            let o = OpinionD1::vacuous_with(vec![0.95, 0.05].try_into().unwrap());
            let h_psi_if_phi1 = vec![SimplexD1::vacuous(), SimplexD1::vacuous()]
                .try_into()
                .unwrap();
            let h_b_if_phi1 = vec![SimplexD1::vacuous(), SimplexD1::vacuous()]
                .try_into()
                .unwrap();
            let fo = OpinionD1::vacuous_with(vec![0.5, 0.5].try_into().unwrap());
            let fpsi = OpinionD1::vacuous_with(vec![0.95, 0.05].try_into().unwrap());
            let fphi = OpinionD1::vacuous_with(vec![0.95, 0.05].try_into().unwrap());
            let fh_fpsi_if_fphi1 = vec![SimplexD1::vacuous(), SimplexD1::vacuous()]
                .try_into()
                .unwrap();
            let ko = OpinionD1::vacuous_with(vec![0.5, 0.5].try_into().unwrap());
            let kpsi = OpinionD1::vacuous_with(vec![0.95, 0.05].try_into().unwrap());
            let kphi = OpinionD1::vacuous_with(vec![0.95, 0.05].try_into().unwrap());
            let kh_kpsi_if_kphi1 = vec![SimplexD1::vacuous(), SimplexD1::vacuous()]
                .try_into()
                .unwrap();
            ops.state.reset(
                psi,
                phi,
                o,
                fo,
                ko,
                h_psi_if_phi1,
                h_b_if_phi1,
                fpsi,
                fphi,
                fh_fpsi_if_fphi1,
                kpsi,
                kphi,
                kh_kpsi_if_kphi1,
            );
            let h = OpinionD1::vacuous_with(marr_d1![0.95, 0.05].try_into().unwrap());
            let fh = OpinionD1::vacuous_with(marr_d1![0.95, 0.05].try_into().unwrap());
            let kh = OpinionD1::vacuous_with(marr_d1![0.95, 0.05].try_into().unwrap());
            let a = OpinionD1::vacuous_with(marr_d1![0.95, 0.05].try_into().unwrap());
            let b = OpinionD1::vacuous_with(marr_d1![0.95, 0.05].try_into().unwrap());
            let theta = OpinionD1::vacuous_with(marr_d1![0.95, 0.05].try_into().unwrap());
            let thetad = OpinionD1::vacuous_with(marr_d1![0.95, 0.05].try_into().unwrap());
            ops.ded.reset(h, fh, kh, a, b, theta, thetad);
        });
    }

    #[tracing_test::traced_test]
    #[test]
    fn test_agent() -> anyhow::Result<()> {
        let m_op = Cow::<OpinionD1<Psi, f32>>::Owned(OpinionD1::new(
            marr_d1![0.0, 0.95],
            0.05,
            marr_d1![0.1, 0.9],
        ));
        let c_op = Cow::Owned(OpinionD1::new(
            marr_d1![0.95, 0.0],
            0.05,
            marr_d1![0.5, 0.5],
        ));
        let m_info = Info::new(0, InfoContent::Misinfo { op: m_op.clone() });
        let c_info = Info::new(
            1,
            InfoContent::Correction {
                op: c_op,
                misinfo: m_op.clone(),
            },
        );

        let mut agent = Agent::<f32>::default();
        // let o = InfoContent::Observation {
        //     op: Cow::Owned(OpinionD1::new(marr_d1![0.9, 0.0], 0.1, marr_d1![0.9, 0.1])),
        // };
        // let o_info = Info::new(2, o);
        // agent.reset(reset);
        // agent.read_info(&m_info, new_trusts(0.5, 0.9, 0.5, 0.1, 0.1, 0.5, 0.9, 0.5));
        // agent.reset(reset);
        agent.read_info(&c_info, new_trusts(1.0, 0.9, 0.5, 0.1, 0.5, 0.5, 0.9, 0.95));
        // agent.reset(reset);
        // agent.read_info(&o_info, new_trusts(1.0, 0.9, 0.5, 0.1, 0.5, 0.5, 0.9, 0.95));
        reset_agent(&mut agent);
        agent.read_info(&m_info, new_trusts(1.0, 0.9, 0.5, 0.5, 0.5, 1.0, 0.9, 0.95));
        // agent.reset(reset);
        // agent.read_info(&m_info, new_trusts(0.6, 0.9, 0.5, 0.1, 0.5, 0.5, 0.9, 0.95));
        // agent.reset(reset);
        // agent.read_info(&m_info, new_trusts(0.5, 0.9, 0.5, 0.1, 0.5, 0.5, 0.9, 0.95));
        // agent.reset(reset);
        // agent.read_info(&c_info, new_trusts(1.0, 0.9, 0.5, 0.8, 0.5, 0.5, 0.9, 0.95));
        // agent.reset(reset);
        // agent.read_info(&m_info, new_trusts(0.8, 0.9, 0.5, 0.8, 0.5, 0.5, 0.9, 0.95));
        // agent.read_info(&c_info, new_trusts(1.0, 0.9, 0.5, 0.8, 0.5, 0.5, 0.9, 0.95));
        // agent.read_info(&c_info, new_trusts(1.0, 0.9, 0.5, 0.8, 0.5, 0.5, 0.9, 0.95));
        Ok(())
    }

    #[test]
    fn test_cond() {
        let c0 = marr_d1!(FPsi; [
            SimplexD1::new(marr_d1!(FH;[0.89, 0.10]), 0.01),
            SimplexD1::new(marr_d1![0.10, 0.89], 0.01),
        ]);
        let c1 = marr_d1!(FO; [
            SimplexD1::new(marr_d1!(FH;[0.94, 0.05]), 0.01),
            SimplexD1::new(marr_d1![0.11, 0.88], 0.01),
        ]);

        let a0 = marr_d1!(FPsi; [0.37, 0.63]);
        let a1 = marr_d1!(FO; [0.5, 0.5]);
        let ay = marr_d1!(FH; [0.85, 0.15]); // marr_d1!(FPsi; [0.95, 0.05]);
        let inv_c0 = c0.inverse(&a0, &ay); // mbr(&a0, &c0).as_ref().unwrap());
        let inv_c1 = c1.inverse(&a1, &ay); // mbr(&a1, &c1).as_ref().unwrap());
        let inv_c01 = MArrD1::<FH, _>::from_fn(|fh| {
            let r0 = OpinionRefD1::from((&inv_c0[fh], &a0));
            let r1 = OpinionRefD1::from((&inv_c1[fh], &a1));
            OpinionD2::product2(r0, r1).simplex
        });
        let c01 = inv_c01.inverse(&ay, &MArrD2::product2(&a0, &a1));
        println!("{:?}", inv_c0);
        println!("{:?}", inv_c1);
        println!("{:?}", c01);
        // let x1_y = y_x1.inverse(ax1, mbr(ax1, y_x1).as_ref().unwrap_or(ay));
        // let x2_y = y_x2.inverse(ax2, mbr(ax2, y_x2).as_ref().unwrap_or(ay));
        // let x12_y = CX1X2Y::from_fn(|y| {
        //     let x1yr = OpinionRef::from((x1_y[y].borrow(), ax1));
        //     let x2yr = OpinionRef::from((x2_y[y].borrow(), ax2));
        //     let OpinionBase { simplex, .. } = Product2::product2(x1yr, x2yr);
        //     simplex
        // });
        // let ax12 = mbr(ay, &x12_y).unwrap_or_else(|| Product2::product2(ax1, ax2));
        // x12_y.inverse(ay, &ax12)

        let c: MArrD2<FPsi, FO, SimplexD1<FH, f32>> =
            MArrD1::<FH, _>::merge_cond2(&c0, &c1, &a0, &a1, &ay);
        println!("{:?}", c[(FPsi(0), FO(0))]);
        println!("{:?}", c[(FPsi(0), FO(1))]);
        println!("{:?}", c[(FPsi(1), FO(0))]);
        println!("{:?}", c[(FPsi(1), FO(1))]);
    }

    #[test]
    fn test_sharing() {
        let mut cpt = CPT::default();
        cpt.reset(0.88, 0.88, 0.69, 2.25, 0.61);

        let x0 = -1.0;
        let x1 = -10.0;
        let y = -0.001;
        let d0 = LevelSet::<(A, Thetad), f32>::new(&marr_d2!(A, Thetad; [[0.0, x1], [x0, x0]]));
        let d1 = LevelSet::<(A, Thetad), f32>::new(
            &marr_d2!(A, Thetad; [[y, x1 + y], [x0 + y, x0 + y]]),
        );
        // b=A[0.1856482,  0.20050177], u=0.61385006, a=A[0.49628967, 0.50371027]
        // b=A[0.15187897, 0.17525421], u=0.6728668 , a=A[0.48121205, 0.5187879]
        // b=Thetad[0.45818824, 0.12272677], u=0.41908503, a=Thetad[0.79127306, 0.2087269]
        // b=Thetad[0.44344428, 0.14268395], u=0.41387182, a=Thetad[0.7584604, 0.24153961]
        let wa0 = OpinionD1::<A, f32>::new(marr_d1![0.19, 0.20], 0.61, marr_d1![0.50, 0.50]);
        let wa1 = OpinionD1::<A, f32>::new(marr_d1![0.10, 0.29], 0.61, marr_d1![0.48, 0.52]);
        let wthetad0 =
            OpinionD1::<Thetad, f32>::new(marr_d1![0.46, 0.12], 0.42, marr_d1![0.80, 0.20]);
        let wthetad1 =
            OpinionD1::<Thetad, f32>::new(marr_d1![0.50, 0.08], 0.42, marr_d1![0.76, 0.24]);
        let w0 = OpinionD2::product2(&wa0, &wthetad0);
        let w1 = OpinionD2::product2(&wa1, &wthetad1);
        let p0 = w0.projection();
        let p1 = w1.projection();
        println!("{w0:?}");
        println!("{w1:?}");
        println!("{p0:?}");
        println!("{p1:?}");
        let v0 = cpt.valuate(&d0, &p0);
        let v1 = cpt.valuate(&d1, &p1);
        println!("{v0}, {v1}");
        // 2024-09-02T11:36:03.828566Z  INFO test_agent:    THd: P=A[Thetad[0.38723496, 0.10306068], Thetad[0.40256393, 0.1071404]]
        // 2024-09-02T11:36:03.828596Z  INFO test_agent:   ~THd: P=A[Thetad[0.36024895, 0.115421645], Thetad[0.39710066, 0.12722872]]
    }
}
