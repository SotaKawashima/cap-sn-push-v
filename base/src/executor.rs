use std::{collections::BTreeMap, mem};

use graph_lib::prelude::{Graph, GraphB};
use itertools::Itertools;
use rand::seq::SliceRandom;
use rand::Rng;
use rand_distr::{Distribution, Exp1, Open01, Standard, StandardNormal};
use tracing::{debug, span, Level};

use crate::{
    agent::Agent,
    info::{Info, InfoContent, InfoLabel},
    opinion::{AccessProb, MyFloat, Trusts},
    stat::{AgentStat, InfoData, InfoStat, PopData, PopStat, Stat},
    util::Reset,
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
}

#[derive(Default)]
pub struct AgentWrapper<V, X> {
    pub core: Agent<V>,
    pub ext: X,
}

impl<V, X> AgentWrapper<V, X> {
    pub fn reset<P, R: Rng>(&mut self, param: &P, rng: &mut R)
    where
        P: Reset<Agent<V>> + Reset<X>,
    {
        self.core.reset(param, rng);
        param.reset(&mut self.ext, rng);
    }
}

pub trait AgentExtTrait<V>: Sized {
    fn visit_prob(wrapper: &AgentWrapper<V, Self>) -> V;
}

#[derive(Debug, Clone, Copy)]
pub struct AgentIdx(pub usize);

impl From<usize> for AgentIdx {
    fn from(value: usize) -> Self {
        Self(value)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct InfoIdx(pub usize);

impl From<usize> for InfoIdx {
    fn from(value: usize) -> Self {
        Self(value)
    }
}

pub trait Executor<V, M, X> {
    fn num_agents(&self) -> usize;
    fn graph(&self) -> &GraphB;
    fn reset<R: Rng>(&self, memory: &mut Memory<V, M>, rng: &mut R);
    fn execute<R>(&self, memory: &mut Memory<V, M>, num_iter: u32, mut rng: R) -> Vec<Stat>
    where
        V: MyFloat,
        Open01: Distribution<V>,
        Standard: Distribution<V>,
        StandardNormal: Distribution<V>,
        Exp1: Distribution<V>,
        R: Rng,
        X: InstanceExt<V, R, Self>,
        M: AgentExtTrait<V>,
        Self: Sized,
    {
        let span = span!(Level::INFO, "iter", i = num_iter);
        let _guard = span.enter();
        self.reset(memory, &mut rng);

        let instance = InstanceWrapper::new(self, rng, X::from_exec(self));
        instance.my_loop(memory, num_iter)
    }
}

pub struct InstanceWrapper<'a, E, V, R, X> {
    pub exec: &'a E,
    pub infos: Vec<Info<'a, V>>,
    pub info_data_table: BTreeMap<InfoLabel, InfoData>,
    pub rng: R,
    pub selfishes: Vec<AgentIdx>,
    pub info_stat: InfoStat,
    pub agent_stat: AgentStat,
    pub pop_stat: PopStat,
    pub total_num_selfish: usize,
    pub rps: Vec<(AgentIdx, InfoIdx)>,
    pub ext: X,
}

impl<'a, E, V: MyFloat, R, X> InstanceWrapper<'a, E, V, R, X> {
    pub fn new(e: &'a E, rng: R, ext: X) -> Self {
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

    fn received_info(&mut self, info_idx: usize) {
        let info = &self.infos[info_idx];
        let d = self.info_data_table.get_mut(info.label()).unwrap();
        d.received();
        debug!(target: "recv", l = ?info.label(), "#" = info.idx);
    }

    fn get_info_mut(&mut self, info_idx: usize) -> (&mut Info<'a, V>, &mut InfoData) {
        let info = &mut self.infos[info_idx];
        let d = self.info_data_table.get_mut(info.label()).unwrap();
        (info, d)
    }
    pub fn new_info(&mut self, obj: &'a InfoContent<V>) -> InfoIdx {
        let info_idx = self.infos.len();
        let info = Info::new(info_idx, obj);
        self.info_data_table.entry(*info.label()).or_default();
        self.infos.push(info);
        info_idx.into()
    }

    fn my_loop<Ax>(mut self, memory: &mut Memory<V, Ax>, num_iter: u32) -> Vec<Stat>
    where
        E: Executor<V, Ax, X>,
        V: MyFloat,
        Open01: Distribution<V>,
        Standard: Distribution<V>,
        StandardNormal: Distribution<V>,
        Exp1: Distribution<V>,
        R: Rng,
        X: InstanceExt<V, R, E>,
        Ax: AgentExtTrait<V>,
    {
        let mut t = 0;
        while !self.rps.is_empty() || !self.selfishes.is_empty() || self.ext.is_continued() {
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
        E: Executor<V, Ax, X>,
        V: MyFloat,
        Open01: Distribution<V>,
        Standard: Distribution<V>,
        StandardNormal: Distribution<V>,
        Exp1: Distribution<V>,
        R: Rng,
        X: InstanceExt<V, R, E>,
        Ax: AgentExtTrait<V>,
    {
        let ips = self.info_producers(t);
        for &(agent_idx, info_idx) in &ips {
            let span = span!(Level::INFO, "IA", "#" = agent_idx.0);
            let _guard = span.enter();
            let (trusts, ap) = X::get_informer(self);
            let info = &self.infos[info_idx.0];
            debug!(target: "recv", l = ?info.label(), "#" = info.idx);
            let agent = memory.get_agent_mut(agent_idx.0);
            agent.core.set_info_opinions(&info, trusts, ap);
            if agent.core.is_willing_selfish() {
                self.selfishes.push(agent_idx);
            }
        }

        let mut s = Vec::new();
        for (agent_idx, info_idx) in mem::take(&mut self.rps) {
            let span = span!(Level::INFO, "SA", "#" = agent_idx.0);
            let _guard = span.enter();

            self.received_info(info_idx.0);
            if self.rng.gen::<V>() >= Ax::visit_prob(memory.get_agent_mut(agent_idx.0)) {
                continue;
            }

            let (trusts, ap) = X::get_sharer(self, info_idx.0);
            let (info, d) = self.get_info_mut(info_idx.0);
            let agent = memory.get_agent_mut(agent_idx.0);
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
                self.total_num_selfish += 1;
            }
            if agent.core.is_willing_selfish() {
                temp.push(*agent_idx);
            }
        }
        self.selfishes = temp;
        self.pop_stat.push(num_iter, t, pop_data);
        self.push_info_stats(num_iter, t);
    }

    fn info_producers<M>(&mut self, t: u32) -> Vec<(AgentIdx, InfoIdx)>
    where
        E: Executor<V, M, X>,
        V: MyFloat,
        Open01: Distribution<V>,
        Standard: Distribution<V>,
        StandardNormal: Distribution<V>,
        Exp1: Distribution<V>,
        R: Rng,
        X: InstanceExt<V, R, E>,
    {
        let mut ips = Vec::new();
        for (agent_idx, obj) in X::get_producers_with(self, t) {
            ips.push((agent_idx, self.new_info(obj)));
        }
        ips
    }
}

pub trait InstanceExt<V, R, E>: Sized {
    fn from_exec(exec: &E) -> Self;
    fn is_continued(&self) -> bool;
    fn get_producers_with<'a>(
        ins: &mut InstanceWrapper<'a, E, V, R, Self>,
        t: u32,
    ) -> Vec<(AgentIdx, &'a InfoContent<V>)>;
    fn get_informer<'a>(ins: &mut InstanceWrapper<'a, E, V, R, Self>)
        -> (Trusts<V>, AccessProb<V>);
    fn get_sharer<'a>(
        ins: &mut InstanceWrapper<'a, E, V, R, Self>,
        info_idx: usize,
    ) -> (Trusts<V>, AccessProb<V>);
}
