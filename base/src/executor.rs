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
};

pub struct Memory<V> {
    pub agents: Vec<Agent<V>>,
}

impl<V: MyFloat> Memory<V> {
    pub fn new(n: usize) -> Self {
        let agents = (0..n).map(|_| Agent::default()).collect::<Vec<_>>();
        Self { agents }
    }

    fn agent(&mut self, idx: usize) -> &mut Agent<V> {
        &mut self.agents[idx]
    }
}

pub trait Executor<V, X>
where
    V: MyFloat,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
{
    fn graph(&self) -> &GraphB;
    fn reset<R: Rng>(&self, core: &mut Memory<V>, rng: &mut R);
    fn instance_ext(&self) -> X;
    fn execute<R>(&self, memory: &mut Memory<V>, num_iter: u32, mut rng: R) -> Vec<Stat>
    where
        R: Rng,
        X: InstanceExt<V, R, Self>,
        Self: Sized,
    {
        let span = span!(Level::INFO, "iter", i = num_iter);
        let _guard = span.enter();
        self.reset(memory, &mut rng);

        let instance = InstanceWrapper::new(self, rng, self.instance_ext());
        instance.my_loop(memory, num_iter)
    }
}

pub struct InstanceWrapper<'a, E, V: MyFloat, R, X> {
    pub e: &'a E,
    pub infos: Vec<Info<'a, V>>,
    pub info_data_table: BTreeMap<InfoLabel, InfoData>,
    pub rng: R,
    pub selfishes: Vec<usize>,
    pub info_stat: InfoStat,
    pub agent_stat: AgentStat,
    pub pop_stat: PopStat,
    pub total_num_selfish: usize,
    pub rps: Vec<(usize, usize)>,
    pub ext: X,
}

impl<'a, E, V: MyFloat, R, X> InstanceWrapper<'a, E, V, R, X> {
    pub fn new(e: &'a E, rng: R, ext: X) -> Self {
        let infos = Vec::<Info<V>>::new();
        let info_data_table = BTreeMap::<InfoLabel, InfoData>::new();
        let selfishes = Vec::<usize>::new();
        let info_stat = InfoStat::default();
        let agent_stat = AgentStat::default();
        let pop_stat = PopStat::default();
        let total_num_selfish = 0;
        let rps = Vec::new();
        Self {
            e,
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

    fn to_stats(self) -> Vec<Stat> {
        vec![
            self.info_stat.into(),
            self.agent_stat.into(),
            self.pop_stat.into(),
        ]
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

    fn info(&mut self, info_idx: usize) -> (&mut Info<'a, V>, &mut InfoData) {
        let info = &mut self.infos[info_idx];
        let d = self.info_data_table.get_mut(info.label()).unwrap();
        (info, d)
    }
    pub fn new_info(&mut self, obj: &'a InfoContent<V>) -> usize {
        let info_idx = self.infos.len();
        let info = Info::new(info_idx, obj);
        self.info_data_table.entry(*info.label()).or_default();
        self.infos.push(info);
        info_idx
    }

    fn my_loop(mut self, memory: &mut Memory<V>, num_iter: u32) -> Vec<Stat>
    where
        E: Executor<V, X>,
        V: MyFloat,
        Open01: Distribution<V>,
        Standard: Distribution<V>,
        StandardNormal: Distribution<V>,
        Exp1: Distribution<V>,
        R: Rng,
        X: InstanceExt<V, R, E>,
    {
        let mut t = 0;
        // let mut rps = Vec::new();
        while !self.rps.is_empty() || !self.selfishes.is_empty() || self.ext.is_continued() {
            self.step(memory, num_iter, t);
            t += 1;
        }
        self.to_stats()
    }

    fn step(&mut self, memory: &mut Memory<V>, num_iter: u32, t: u32)
    where
        E: Executor<V, X>,
        V: MyFloat,
        Open01: Distribution<V>,
        Standard: Distribution<V>,
        StandardNormal: Distribution<V>,
        Exp1: Distribution<V>,
        R: Rng,
        X: InstanceExt<V, R, E>,
    {
        let ips = self.info_producers(t);
        for &(agent_idx, info_idx) in &ips {
            let span = span!(Level::INFO, "IA", "#" = agent_idx);
            let _guard = span.enter();
            let (trusts, ap) = X::get_informer(self);
            let info = &self.infos[info_idx];
            debug!(target: "recv", l = ?info.label(), "#" = info.idx);
            let agent = memory.agent(agent_idx);
            agent.set_info_opinions(&info, trusts, ap);
            if agent.is_willing_selfish() {
                self.selfishes.push(agent_idx);
            }
        }

        let mut s = Vec::new();
        for (agent_idx, info_idx) in mem::take(&mut self.rps) {
            let span = span!(Level::INFO, "SA", "#" = agent_idx);
            let _guard = span.enter();

            self.received_info(info_idx);
            if self.rng.gen::<V>() >= self.ext.q(&memory.agent(agent_idx)) {
                continue;
            }

            let (trusts, ap) = X::get_sharer(self, info_idx);
            // , self, agent_idx, info_idx
            let (info, d) = self.info(info_idx);
            let agent = memory.agent(agent_idx);
            let b = agent.read_info(info, trusts, ap);
            info.viewed();
            d.viewed();
            if b.sharing {
                info.shared();
                d.shared();
                s.push((agent_idx, info_idx));
            }
            if b.first_access {
                d.first_viewed();
            }
            if agent.is_willing_selfish() {
                self.selfishes.push(agent_idx);
            }
        }
        let mut next_rps = ips
            .into_iter()
            .chain(s.into_iter())
            .flat_map(|(agent_idx, info_idx)| {
                self.e
                    .graph()
                    .successors(agent_idx)
                    .map(move |bid| (*bid, info_idx))
            })
            .collect_vec();
        next_rps.shuffle(&mut self.rng);
        self.rps = next_rps;

        let mut pop_data = PopData::default();
        let mut temp = Vec::new(); // mem::take(&mut self.selfishes);
        for agent_idx in &self.selfishes {
            let agent = memory.agent(*agent_idx);
            let span = span!(Level::INFO, "Ag", "#" = agent_idx);
            let _guard = span.enter();
            if agent.progress_selfish_status() {
                self.agent_stat.push_selfish(num_iter, t, *agent_idx);
                pop_data.selfish();
                self.total_num_selfish += 1;
            }
            if agent.is_willing_selfish() {
                temp.push(agent_idx);
            }
        }
        self.pop_stat.push(num_iter, t, pop_data);
        self.push_info_stats(num_iter, t);
    }

    fn info_producers(&mut self, t: u32) -> Vec<(usize, usize)>
    where
        E: Executor<V, X>,
        V: MyFloat,
        Open01: Distribution<V>,
        Standard: Distribution<V>,
        StandardNormal: Distribution<V>,
        Exp1: Distribution<V>,
        R: Rng,
        X: InstanceExt<V, R, E>,
    {
        let mut ips = Vec::new();
        for (agent_idx, obj) in X::info_contents(self, t) {
            ips.push((agent_idx, self.new_info(obj)));
        }
        ips
    }
}

pub trait InstanceExt<V, R, E>: Sized
where
    V: MyFloat,
    E: Executor<V, Self>,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    R: Rng,
{
    fn is_continued(&self) -> bool;
    fn info_contents<'a>(
        ins: &mut InstanceWrapper<'a, E, V, R, Self>,
        t: u32,
    ) -> Vec<(usize, &'a InfoContent<V>)>;
    fn q(&self, agent: &Agent<V>) -> V;
    fn get_informer<'a>(ins: &mut InstanceWrapper<'a, E, V, R, Self>)
        -> (Trusts<V>, AccessProb<V>);
    fn get_sharer<'a>(
        ins: &mut InstanceWrapper<'a, E, V, R, Self>,
        info_idx: usize,
    ) -> (Trusts<V>, AccessProb<V>);
}
