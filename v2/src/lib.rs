use base::{
    executor::{
        AgentExtTrait, AgentIdx, AgentWrapper, Executor, InstanceExt, InstanceWrapper, Memory,
    },
    opinion::{AccessProb, MyFloat, Trusts},
};
use graph_lib::prelude::{Graph, GraphB};
use rand::Rng;
use rand_distr::{Distribution, Exp1, Open01, Standard, StandardNormal};

pub async fn start() -> anyhow::Result<()> {
    Ok(())
}

struct Exec {
    graph: GraphB,
}

impl<V> Executor<V, AgentExt, Instance> for Exec
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

    fn instance_ext(&self) -> Instance {
        todo!()
    }
    fn reset<R: Rng>(&self, memory: &mut Memory<V, AgentExt>, rng: &mut R) {}
}

struct AgentExt;

impl<V> AgentExtTrait<V> for AgentExt {
    fn visit_prob(wrapper: &AgentWrapper<V, Self>) -> V {
        todo!()
    }
}

struct Instance;

impl<V, R> InstanceExt<V, R, Exec> for Instance
where
    V: MyFloat,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    R: Rng,
{
    fn is_continued(&self) -> bool {
        todo!()
    }

    fn get_producers_with<'a>(
        ins: &mut InstanceWrapper<'a, Exec, V, R, Self>,
        t: u32,
    ) -> Vec<(AgentIdx, &'a base::info::InfoContent<V>)> {
        todo!()
    }

    fn get_informer<'a>(
        ins: &mut InstanceWrapper<'a, Exec, V, R, Self>,
    ) -> (Trusts<V>, AccessProb<V>) {
        todo!()
    }

    fn get_sharer<'a>(
        ins: &mut InstanceWrapper<'a, Exec, V, R, Self>,
        info_idx: usize,
    ) -> (Trusts<V>, AccessProb<V>) {
        todo!()
    }
}
