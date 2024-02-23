mod agent;
pub mod config;
mod cpt;
mod dist;
mod info;
mod opinion;
mod stat;
mod value;

use std::{collections::BTreeMap, iter::Sum, sync::mpsc, thread};

use approx::UlpsEq;
use config::{Config, ConfigFormat, EventTable};
use graph_lib::prelude::{Graph, GraphB};
use num_traits::{Float, FromPrimitive, NumAssign};
use rand::{rngs::SmallRng, seq::SliceRandom, Rng, SeedableRng};
use rand_distr::{uniform::SampleUniform, Distribution, Exp1, Open01, Standard, StandardNormal};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

use agent::{Agent, AgentParams};
use info::{Info, InfoContent, InfoLabel};
use stat::{FileWriters, InfoData, InfoStat, Stat};

use crate::stat::{AgentStat, PopData, PopStat};

pub struct Runner<V>
where
    V: Float + UlpsEq + NumAssign + SampleUniform,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
{
    config: Config<V>,
    identifier: String,
    overwriting: bool,
}

impl<V> Runner<V>
where
    V: Float + NumAssign + UlpsEq + Default + Sum + std::fmt::Debug + SampleUniform + Send + Sync,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
{
    pub fn try_new(
        config_data: ConfigFormat,
        identifier: String,
        overwriting: bool,
    ) -> anyhow::Result<Self>
    where
        for<'de> V: serde::Deserialize<'de>,
    {
        let config: Config<V> = config_data.try_into()?;
        Ok(Self {
            config,
            identifier,
            overwriting,
        })
    }

    pub fn run(&mut self) -> anyhow::Result<()>
    where
        V: FromPrimitive + Sync,
        <V as SampleUniform>::Sampler: Sync,
    {
        let (sender, receiver) = mpsc::channel::<Stat>();
        let mut writers = FileWriters::try_new(
            &self.config.output,
            &self.config.runtime,
            &self.identifier,
            self.overwriting,
        )?;

        let handle = thread::spawn(move || {
            while let Ok(stat) = receiver.recv() {
                writers.write(stat).unwrap();
            }
            writers.finish().unwrap();
            println!("finished writing.");
        });

        let num_nodes = V::from_usize(self.config.runtime.graph.node_count()).unwrap();
        let mean_degree =
            V::from_usize(self.config.runtime.graph.edge_count()).unwrap() / num_nodes;

        let mut rng = SmallRng::seed_from_u64(self.config.runtime.seed_state);
        let rngs = (0..(self.config.runtime.num_parallel))
            .map(|_| SmallRng::from_rng(&mut rng))
            .collect::<Result<Vec<_>, _>>()?;

        println!("started.");
        rngs.into_par_iter()
            .enumerate()
            .for_each(|(num_par, mut rng)| {
                let mut env = Environment::new(
                    &self.config,
                    &mut rng,
                    num_nodes,
                    mean_degree,
                    sender.clone(),
                );
                let num_par = num_par as u32;
                for num_iter in 0..(self.config.runtime.iteration_count) {
                    env.execute(num_par, num_iter);
                }
            });
        drop(sender);
        handle.join().unwrap();
        println!("done.");
        Ok(())
    }
}

#[derive(Debug)]
struct Receiver {
    agent_idx: usize,
    info_idx: usize,
}

struct Environment<'a, V, R>
where
    V: Float + UlpsEq + SampleUniform + NumAssign,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
{
    graph: &'a GraphB,
    info_contents: &'a [InfoContent<V>],
    agent_params: &'a AgentParams<V>,
    event_table: &'a EventTable,
    rng: &'a mut R,
    num_nodes: V,
    mean_degree: V,
    agents: Vec<Agent<V>>,
    sender: mpsc::Sender<Stat>,
}

impl<'a, V, R> Environment<'a, V, R>
where
    R: Rng,
    V: Float + UlpsEq + SampleUniform + NumAssign,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
{
    fn new(
        config: &'a Config<V>,
        rng: &'a mut R,
        num_nodes: V,
        mean_degree: V,
        sender: mpsc::Sender<Stat>,
    ) -> Self
    where
        V: Default + NumAssign,
    {
        let agents = (0..config.runtime.graph.node_count())
            .map(|_| Agent::default())
            .collect::<Vec<_>>();
        Self {
            graph: &config.runtime.graph,
            info_contents: &config.scenario.info_contents,
            agent_params: &config.scenario.agent_params,
            event_table: &config.scenario.event_table,
            rng,
            num_nodes,
            mean_degree,
            agents,
            sender,
        }
    }

    fn execute(&mut self, num_par: u32, num_iter: u32)
    where
        V: FromPrimitive + Sum + Default + std::fmt::Debug,
    {
        log::info!("#i: {num_iter}");

        for agent in &mut self.agents {
            agent.reset_with(self.agent_params, self.rng);
        }

        let mut infos = Vec::<Info<V>>::new();
        let mut receivers = Vec::new();
        let mut info_data_map = BTreeMap::<InfoLabel, InfoData>::new();
        let mut event_table = self.event_table.0.clone();
        let mut agents_willing_selfish = Vec::<usize>::new();
        let mut senders = Vec::new();

        let mut info_stat = InfoStat::default();
        let mut agent_stat = AgentStat::default();
        let mut pop_stat = PopStat::default();
        let mut t = 0;

        while !receivers.is_empty() || !event_table.is_empty() || !agents_willing_selfish.is_empty()
        {
            log::info!("t = {t}");
            let mut pop_data = PopData::default();

            if let Some(informms) = event_table.remove(&t) {
                for (agent_idx, info_content_idx) in informms {
                    let info_idx = infos.len();
                    let info = Info::new(info_idx, &self.info_contents[info_content_idx]);
                    info_data_map.entry(info.content.label).or_default();

                    log::info!("Agent {agent_idx} (informer) <- {}", info.content.label);

                    let agent = &mut self.agents[agent_idx];
                    agent.set_info_opinions(&info, &self.agent_params.base_rates);
                    infos.push(info);
                    if agent.is_willing_selfish() {
                        agents_willing_selfish.push(agent_idx);
                    }

                    senders.push(Receiver {
                        agent_idx,
                        info_idx,
                    });
                }
            }

            receivers.shuffle(self.rng);

            for r @ Receiver {
                agent_idx,
                info_idx,
            } in receivers.drain(..)
            {
                let agent = &mut self.agents[agent_idx];
                let info = &mut infos[info_idx];
                let info_label = info.content.label;
                let info_data = info_data_map.get_mut(&info_label).unwrap();
                info_data.received();

                log::info!("Agent {agent_idx} (sharer) <- {info_label}");

                let friend_receipt_prob = V::one()
                    - (V::one() - V::from_usize(info.num_shared()).unwrap() / self.num_nodes)
                        .powf(self.mean_degree);
                log::info!("r^i_m = {friend_receipt_prob:?}");

                if self.rng.gen::<V>() > agent.access_prob() {
                    continue;
                }

                let b = agent.read_info(info, friend_receipt_prob, self.agent_params, self.rng);
                info.viewed();
                info_data.viewed();
                if b.sharing {
                    info.shared();
                    info_data.shared();
                    senders.push(r);
                }
                if b.first_access {
                    info_data.first_viewed();
                }

                if agent.is_willing_selfish() {
                    agents_willing_selfish.push(agent_idx);
                }
            }

            for Receiver {
                agent_idx,
                info_idx,
            } in senders.drain(..)
            {
                for bid in self.graph.successors(agent_idx) {
                    receivers.push(Receiver {
                        agent_idx: *bid,
                        info_idx,
                    });
                }
            }

            for (info_label, d) in &mut info_data_map {
                info_stat.push(num_par, num_iter, t, d, info_label);
                *d = InfoData::default();
            }
            agents_willing_selfish.retain(|&agent_idx| {
                let agent = &mut self.agents[agent_idx];
                if agent.progress_selfish_status() {
                    log::info!("Agent {agent_idx} : done selfish");
                    agent_stat.push_selfish(num_par, num_iter, t, agent_idx);
                    pop_data.selfish();
                }
                agent.is_willing_selfish()
            });
            pop_stat.push(num_par, num_iter, t, pop_data);
            t += 1;
        }
        self.sender.send(info_stat.into()).unwrap();
        self.sender.send(agent_stat.into()).unwrap();
        self.sender.send(pop_stat.into()).unwrap();
    }
}
