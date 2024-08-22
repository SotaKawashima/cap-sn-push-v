use std::borrow::Cow;

use base::{
    executor::{
        AgentExtTrait, AgentIdx, AgentWrapper, Executor, InstanceExt, InstanceWrapper, Memory,
    },
    info::InfoContent,
    opinion::{AccessProb, MyFloat, Trusts},
};
use graph_lib::prelude::{Graph, GraphB};
use rand::{seq::index::sample, Rng};
use rand_distr::{Distribution, Exp1, Open01, Standard, StandardNormal};
use subjective_logic::{marr1, marr_d1, mul::labeled::OpinionD1};

pub async fn start() -> anyhow::Result<()> {
    Ok(())
}

struct Communities<V> {
    misinfo: Vec<(usize, V)>,
}

struct Exec<V> {
    graph: GraphB,
    communities: Communities<V>,
    produce_prob: V,
}

impl<V> Executor<V, AgentExt, Instance> for Exec<V>
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

    fn reset<R: Rng>(&self, memory: &mut Memory<V, AgentExt>, rng: &mut R) {}
}

struct AgentExt;

impl<V> AgentExtTrait<V> for AgentExt {
    fn visit_prob(wrapper: &AgentWrapper<V, Self>) -> V {
        todo!()
    }
}

struct Instance {
    // info_contents: Vec<InfoContent<V>>,
    misinfo_producers: Vec<usize>,
    observable: Vec<usize>,
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
    fn from_exec(exec: &Exec<V>) -> Self {
        todo!()
    }

    fn is_continued(&self) -> bool {
        todo!()
    }

    fn get_producers_with<'a>(
        ins: &mut InstanceWrapper<'a, Exec<V>, V, R, Self>,
        _: u32,
    ) -> Vec<(AgentIdx, Cow<'a, InfoContent<V>>)> {
        let mut producers = Vec::new();
        // if let Some(observer) = &ins.exec.scenario.observer {
        //     ins.ext.observable.retain(|&agent_idx| {
        //         if ins.total_num_selfish <= observer.threshold {
        //             return true;
        //         }
        //         if observer.po <= ins.rng.gen() {
        //             return true;
        //         }
        //         if observer.pp <= ins.rng.gen() {
        //             return true;
        //         }
        //         producers.push((
        //             agent_idx.into(),
        //             Cow::Borrowed(&ins.exec.scenario.info_contents[observer.observed_info_obj_idx]),
        //         ));
        //         false
        //     });
        //     if producers.len() > 1 {
        //         producers.shuffle(&mut ins.rng);
        //     }
        // }
        ins.ext.misinfo_producers.retain(|&agent_idx| {
            if ins.exec.produce_prob >= ins.rng.gen() {
                return true;
            }
            let c = InfoContent::Misinfo {
                op: OpinionD1::vacuous_with(marr_d1![V::one(), V::zero()]),
            };
            producers.push((agent_idx.into(), Cow::Owned(c)));
            false
        });
        producers
    }

    fn get_informer<'a>(
        ins: &mut InstanceWrapper<'a, Exec<V>, V, R, Self>,
    ) -> (Trusts<V>, AccessProb<V>) {
        todo!()
    }

    fn get_sharer<'a>(
        ins: &mut InstanceWrapper<'a, Exec<V>, V, R, Self>,
        info_idx: usize,
    ) -> (Trusts<V>, AccessProb<V>) {
        todo!()
    }
}
