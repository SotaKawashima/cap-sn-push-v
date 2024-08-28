use std::{collections::BTreeMap, mem};

use graph_lib::prelude::{Graph, GraphB};
use itertools::Itertools;
use rand::seq::SliceRandom;
use rand::Rng;
use rand_distr::{Distribution, Exp1, Open01, Standard, StandardNormal};
use tracing::{debug, span, Level};

use crate::{
    agent::{Agent, Decision},
    info::{Info, InfoContent, InfoLabel},
    opinion::{AccessProb, MyFloat, MyOpinions, Trusts},
    stat::{AgentStat, InfoData, InfoStat, PopData, PopStat, Stat},
};

pub struct Memory<V, Ax> {
    pub agents: Vec<AgentWrapper<V, Ax>>,
    pub id: usize,
}

impl<V: MyFloat, Ax> Memory<V, Ax> {
    pub fn new<E, X>(e: &E, id: usize) -> Self
    where
        E: Executor<V, Ax, X>,
        Ax: Default,
    {
        let agents = (0..(e.num_agents()))
            .map(|_| AgentWrapper::default())
            .collect::<Vec<_>>();
        Self { agents, id }
    }

    fn get_agent_mut(&mut self, idx: usize) -> &mut AgentWrapper<V, Ax> {
        &mut self.agents[idx]
    }

    fn reset<E, R: Rng>(&mut self, exec: &E, rng: &mut R)
    where
        Ax: AgentExtTrait<V, Exec = E>,
    {
        for (idx, agent) in self.agents.iter_mut().enumerate() {
            let span = span!(Level::INFO, "init", "#" = idx);
            let _guard = span.enter();
            agent.idx = idx;
            agent.ext.reset(idx, exec, rng);
            agent.core.reset(|ops, decision| {
                Ax::reset_core(ops, decision, exec, rng);
            });
        }
    }
}

#[derive(Default)]
pub struct AgentWrapper<V, X> {
    pub idx: usize,
    pub core: Agent<V>,
    pub ext: X,
}

pub trait AgentExtTrait<V: Clone>: Sized {
    type Exec;
    type Ix;
    fn reset_core<R: Rng>(
        ops: &mut MyOpinions<V>,
        decision: &mut Decision<V>,
        exec: &Self::Exec,
        rng: &mut R,
    );
    fn reset<R: Rng>(&mut self, idx: usize, exec: &Self::Exec, rng: &mut R);
    fn visit_prob<R: Rng>(&mut self, exec: &Self::Exec, rng: &mut R) -> V;
    fn informer_trusts<'a, R: Rng>(
        &mut self,
        ins: &mut InstanceWrapper<'a, Self::Exec, V, R, Self::Ix>,
        info_idx: InfoIdx,
    ) -> Trusts<V>;
    fn informer_access_probs<'a, R: Rng>(
        &mut self,
        ins: &mut InstanceWrapper<'a, Self::Exec, V, R, Self::Ix>,
        info_idx: InfoIdx,
    ) -> AccessProb<V>;
    fn sharer_trusts<'a, R: Rng>(
        &mut self,
        ins: &mut InstanceWrapper<'a, Self::Exec, V, R, Self::Ix>,
        info_idx: InfoIdx,
    ) -> Trusts<V>;
    fn sharer_access_probs<'a, R: Rng>(
        &mut self,
        ins: &mut InstanceWrapper<'a, Self::Exec, V, R, Self::Ix>,
        info_idx: InfoIdx,
    ) -> AccessProb<V>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct AgentIdx(pub usize);

impl From<usize> for AgentIdx {
    fn from(value: usize) -> Self {
        Self(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct InfoIdx(pub usize);

impl From<usize> for InfoIdx {
    fn from(value: usize) -> Self {
        Self(value)
    }
}

pub trait Executor<V, Ax, Ix> {
    fn num_agents(&self) -> usize;
    fn graph(&self) -> &GraphB;
    fn execute<R>(&self, memory: &mut Memory<V, Ax>, num_iter: u32, mut rng: R) -> Vec<Stat>
    where
        V: MyFloat,
        Open01: Distribution<V>,
        Standard: Distribution<V>,
        StandardNormal: Distribution<V>,
        Exp1: Distribution<V>,
        R: Rng,
        Ix: InstanceExt<V, R, Self>,
        Ax: AgentExtTrait<V, Exec = Self, Ix = Ix>,
        Self: Sized,
    {
        let span = span!(Level::INFO, "iter", i = num_iter);
        let _guard = span.enter();
        memory.reset(self, &mut rng);

        let instance = InstanceWrapper::new(self, Ix::from_exec(self, &mut rng), rng);
        instance.my_loop(memory, num_iter)
    }
}

pub struct InstanceWrapper<'a, E, V: Clone, R, X> {
    pub exec: &'a E,
    infos: Vec<Info<'a, V>>,
    pub info_data_table: BTreeMap<InfoLabel, InfoData>,
    pub rng: R,
    selfishes: Vec<AgentIdx>,
    info_stat: InfoStat,
    agent_stat: AgentStat,
    pop_stat: PopStat,
    total_num_selfish: usize,
    prev_num_selfish: usize,
    rps: Vec<(AgentIdx, InfoIdx)>,
    pub ext: X,
}

impl<'a, E, V: MyFloat, R, Ix> InstanceWrapper<'a, E, V, R, Ix> {
    pub fn new(e: &'a E, ext: Ix, rng: R) -> Self {
        let infos = Vec::<Info<V>>::new();
        let info_data_table = BTreeMap::<InfoLabel, InfoData>::new();
        let selfishes = Vec::new();
        let info_stat = InfoStat::default();
        let agent_stat = AgentStat::default();
        let pop_stat = PopStat::default();
        let total_num_selfish = 0;
        let rps = Vec::new();
        Self {
            exec: e,
            infos,
            info_data_table,
            rng,
            selfishes,
            info_stat,
            agent_stat,
            pop_stat,
            total_num_selfish,
            prev_num_selfish: 0,
            rps,
            ext,
        }
    }

    fn push_info_stats(&mut self, num_iter: u32, t: u32) {
        for (info_label, d) in &mut self.info_data_table {
            self.info_stat.push(num_iter, t, d, info_label);
            *d = InfoData::default();
        }
    }

    fn received_info(&mut self, info_idx: InfoIdx) {
        let info = &self.infos[info_idx.0];
        let d = self.info_data_table.get_mut(info.label()).unwrap();
        d.received();
        debug!(target: "recv", l = ?info.label(), "#" = info.idx);
    }

    fn get_info_mut(&mut self, info_idx: InfoIdx) -> (&mut Info<'a, V>, &mut InfoData) {
        let info = &mut self.infos[info_idx.0];
        let d = self.info_data_table.get_mut(info.label()).unwrap();
        (info, d)
    }

    pub fn get_info(&self, info_idx: InfoIdx) -> &Info<'_, V> {
        &self.infos[info_idx.0]
    }

    pub fn get_info_label(&self, info_idx: InfoIdx) -> &InfoLabel {
        self.infos[info_idx.0].label()
    }

    pub fn num_shared(&self, info_idx: InfoIdx) -> usize {
        self.infos[info_idx.0].num_shared()
    }

    pub fn total_num_selfish(&self) -> usize {
        self.total_num_selfish
    }

    pub fn prev_num_selfish(&self) -> usize {
        self.prev_num_selfish
    }
    fn new_info(&mut self, obj: InfoContent<'a, V>) -> InfoIdx {
        let info_idx = self.infos.len();
        let info = Info::new(info_idx, obj);
        let d = self.info_data_table.entry(*info.label()).or_default();
        d.posted();
        self.infos.push(info);
        info_idx.into()
    }

    fn my_loop<Ax>(mut self, memory: &mut Memory<V, Ax>, num_iter: u32) -> Vec<Stat>
    where
        E: Executor<V, Ax, Ix>,
        V: MyFloat,
        Open01: Distribution<V>,
        Standard: Distribution<V>,
        StandardNormal: Distribution<V>,
        Exp1: Distribution<V>,
        R: Rng,
        Ix: InstanceExt<V, R, E>,
        Ax: AgentExtTrait<V, Exec = E, Ix = Ix>,
    {
        let mut t = 0;
        while !self.rps.is_empty() || !self.selfishes.is_empty() || self.ext.is_continued(self.exec)
        {
            let span = span!(Level::INFO, "t", t = t);
            let _guard = span.enter();
            self.step(memory, num_iter, t);
            t += 1;
        }
        vec![
            self.info_stat.into(),
            self.agent_stat.into(),
            self.pop_stat.into(),
        ]
    }

    fn step<Ax>(&mut self, memory: &mut Memory<V, Ax>, num_iter: u32, t: u32)
    where
        E: Executor<V, Ax, Ix>,
        V: MyFloat,
        Open01: Distribution<V>,
        Standard: Distribution<V>,
        StandardNormal: Distribution<V>,
        Exp1: Distribution<V>,
        R: Rng,
        Ix: InstanceExt<V, R, E>,
        Ax: AgentExtTrait<V, Exec = E, Ix = Ix>,
    {
        let ips = self.info_producers(t);
        for &(agent_idx, info_idx) in &ips {
            let span = span!(Level::INFO, "IA", "#" = agent_idx.0);
            let _guard = span.enter();
            let agent = memory.get_agent_mut(agent_idx.0);
            let trusts = agent.ext.informer_trusts(self, info_idx);
            let ap = agent.ext.informer_access_probs(self, info_idx);
            let info = &self.infos[info_idx.0];
            debug!(target: "recv", l = ?info.label(), "#" = info.idx);
            agent.core.set_info_opinions(&info, trusts, ap);
            if agent.core.is_willing_selfish() {
                self.selfishes.push(agent_idx);
            }
        }

        let mut s = Vec::new();
        for (agent_idx, info_idx) in mem::take(&mut self.rps) {
            let span = span!(Level::INFO, "SA", "#" = agent_idx.0);
            let _guard = span.enter();
            self.received_info(info_idx);

            let agent = memory.get_agent_mut(agent_idx.0);
            if self.rng.gen::<V>() >= agent.ext.visit_prob(self.exec, &mut self.rng) {
                continue;
            }

            let trusts = agent.ext.sharer_trusts(self, info_idx);
            let ap = agent.ext.sharer_access_probs(self, info_idx);
            let (info, d) = self.get_info_mut(info_idx);
            let b = agent.core.read_info(info, trusts, ap);
            info.viewed();
            d.viewed();
            if b.sharing {
                info.shared();
                d.shared();
                s.push((agent_idx, info_idx));
            }
            if b.first_access {
                // println!("i={num_iter} t={t} agent_idx={agent_idx} info_idx={info_idx}");
                d.first_viewed();
            }
            if agent.core.is_willing_selfish() {
                self.selfishes.push(agent_idx);
            }
        }
        let mut next_rps = ips
            .into_iter()
            .chain(s.into_iter())
            .flat_map(|(agent_idx, info_idx)| {
                self.exec
                    .graph()
                    .successors(agent_idx.0)
                    .map(move |&bid| (bid.into(), info_idx))
            })
            .collect_vec();
        next_rps.shuffle(&mut self.rng);
        self.rps = next_rps;

        let mut pop_data = PopData::default();
        let mut temp = Vec::new(); // mem::take(&mut self.selfishes);
        for agent_idx in &self.selfishes {
            let agent = memory.get_agent_mut(agent_idx.0);
            let span = span!(Level::INFO, "Ag", "#" = agent_idx.0);
            let _guard = span.enter();
            if agent.core.progress_selfish_status() {
                self.agent_stat.push_selfish(num_iter, t, agent_idx.0);
                pop_data.selfish();
            }
            if agent.core.is_willing_selfish() {
                temp.push(*agent_idx);
            }
        }
        self.selfishes = temp;
        self.total_num_selfish += pop_data.num_selfish as usize;
        self.prev_num_selfish = pop_data.num_selfish as usize;
        self.pop_stat.push(num_iter, t, pop_data);
        self.push_info_stats(num_iter, t);
    }

    fn info_producers<M>(&mut self, t: u32) -> Vec<(AgentIdx, InfoIdx)>
    where
        E: Executor<V, M, Ix>,
        V: MyFloat,
        Open01: Distribution<V>,
        Standard: Distribution<V>,
        StandardNormal: Distribution<V>,
        Exp1: Distribution<V>,
        R: Rng,
        Ix: InstanceExt<V, R, E>,
    {
        let mut ips = Vec::new();
        for (agent_idx, obj) in Ix::get_informers_with(self, t) {
            ips.push((agent_idx, self.new_info(obj)));
        }
        ips
    }
}

pub trait InstanceExt<V: Clone, R, E>: Sized {
    fn from_exec(exec: &E, rng: &mut R) -> Self;
    fn is_continued(&self, exec: &E) -> bool;
    fn get_informers_with<'a>(
        ins: &mut InstanceWrapper<'a, E, V, R, Self>,
        t: u32,
    ) -> Vec<(AgentIdx, InfoContent<'a, V>)>;
}
