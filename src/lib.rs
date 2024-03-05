mod agent;
pub mod config;
mod cpt;
mod dist;
mod info;
mod opinion;
mod scenario;
mod stat;
mod value;

use std::{collections::BTreeMap, iter::Sum, path::Path, sync::mpsc, thread};

use approx::UlpsEq;
use graph_lib::prelude::Graph;
use num_traits::{Float, FromPrimitive, NumAssign, ToPrimitive};
use rand::{rngs::SmallRng, seq::SliceRandom, Rng, SeedableRng};
use rand_distr::{uniform::SampleUniform, Distribution, Exp1, Open01, Standard, StandardNormal};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

use agent::{Agent, AgentParams};
use config::{ConfigData, General, Runtime};
use info::{Info, InfoBuilder, InfoLabel};
use scenario::{Scenario, ScenarioParam};
use stat::{AgentStat, FileWriters, InfoData, InfoStat, PopData, PopStat, Stat};

pub struct Runner<P, V>
where
    P: AsRef<Path>,
    V: Float + UlpsEq + NumAssign + SampleUniform,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
{
    general: ConfigData<P, General>,
    runtime: ConfigData<P, Runtime>,
    agent_params: ConfigData<P, AgentParams<V>>,
    scenario: ConfigData<P, Scenario<V>>,
    identifier: String,
    overwriting: bool,
}

impl<P, V> Runner<P, V>
where
    P: AsRef<Path>,
    V: Float
        + NumAssign
        + UlpsEq
        + FromPrimitive
        + ToPrimitive
        + Default
        + Sum
        + std::fmt::Debug
        + SampleUniform
        + Send
        + Sync,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
{
    pub fn try_new(
        general_path: P,
        runtime_path: P,
        agent_params_path: P,
        scenario_path: P,
        identifier: String,
        overwriting: bool,
    ) -> anyhow::Result<Self>
    where
        for<'de> V: serde::Deserialize<'de>,
    {
        Ok(Self {
            general: ConfigData::try_new::<General>(general_path)?,
            runtime: ConfigData::try_new::<Runtime>(runtime_path)?,
            agent_params: ConfigData::try_new::<AgentParams<V>>(agent_params_path)?,
            scenario: ConfigData::try_new::<ScenarioParam<V>>(scenario_path)?,
            identifier,
            overwriting,
        })
    }

    fn create_metadata(&self) -> BTreeMap<String, String> {
        BTreeMap::from_iter([
            ("version".to_string(), env!("CARGO_PKG_VERSION").to_string()),
            ("general".to_string(), self.general.get_path_string()),
            ("runtime".to_string(), self.runtime.get_path_string()),
            (
                "agent_params".to_string(),
                self.agent_params.get_path_string(),
            ),
            ("scenario".to_string(), self.scenario.get_path_string()),
            (
                "num_parallel".to_string(),
                self.runtime.data.num_parallel.to_string(),
            ),
            (
                "iteration_count".to_string(),
                self.runtime.data.iteration_count.to_string(),
            ),
        ])
    }

    pub fn run(&mut self) -> anyhow::Result<()>
    where
        V: FromPrimitive + ToPrimitive + Sync,
        <V as SampleUniform>::Sampler: Sync,
    {
        let (sender, receiver) = mpsc::channel::<Stat>();
        let mut writers = FileWriters::try_new(
            &self.general.data.output,
            &self.identifier,
            self.overwriting,
            self.create_metadata(),
        )?;

        let handle = thread::spawn(move || {
            while let Ok(stat) = receiver.recv() {
                writers.write(stat).unwrap();
            }
            writers.finish().unwrap();
            println!("finished writing.");
        });

        let mut rng = SmallRng::seed_from_u64(self.runtime.data.seed_state);
        let rngs = (0..(self.runtime.data.num_parallel))
            .map(|_| SmallRng::from_rng(&mut rng))
            .collect::<Result<Vec<_>, _>>()?;

        println!("started.");
        rngs.into_par_iter()
            .enumerate()
            .for_each(|(num_par, mut rng)| {
                let mut env = Environment::new(
                    &self.agent_params.data,
                    &self.scenario.data,
                    &mut rng,
                    sender.clone(),
                );
                let num_par = num_par as u32;
                for num_iter in 0..(self.runtime.data.iteration_count) {
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
    agent_params: &'a AgentParams<V>,
    scenario: &'a Scenario<V>,
    rng: &'a mut R,
    info_builder: InfoBuilder<V>,
    agents: Vec<Agent<V>>,
    sender: mpsc::Sender<Stat>,
}

impl<'a, V, R> Environment<'a, V, R>
where
    R: Rng,
    V: Float + UlpsEq + SampleUniform + NumAssign + FromPrimitive + ToPrimitive,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
{
    fn new(
        agent_params: &'a AgentParams<V>,
        scenario: &'a Scenario<V>,
        rng: &'a mut R,
        sender: mpsc::Sender<Stat>,
    ) -> Self
    where
        V: Default + NumAssign,
    {
        let agents = (0..scenario.num_nodes)
            .map(|_| Agent::default())
            .collect::<Vec<_>>();
        Self {
            agent_params,
            scenario,
            rng,
            info_builder: InfoBuilder::new(),
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
        let info_objects = &self.scenario.info_objects;
        let mut event_table = self.scenario.table.clone();
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
                for (agent_idx, info_object_idx) in informms {
                    let info_idx = infos.len();
                    let info = self
                        .info_builder
                        .build(info_idx, &info_objects[info_object_idx]);
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
                    - (V::one()
                        - V::from_usize(info.num_shared()).unwrap() / self.scenario.fnum_nodes)
                        .powf(self.scenario.mean_degree);
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
                for bid in self.scenario.graph.successors(agent_idx) {
                    receivers.push(Receiver {
                        agent_idx: *bid,
                        info_idx,
                    });
                }
            }

            // update selfish states of agents
            agents_willing_selfish.retain(|&agent_idx| {
                let agent = &mut self.agents[agent_idx];
                if agent.progress_selfish_status() {
                    log::info!("Agent {agent_idx} : done selfish");
                    agent_stat.push_selfish(num_par, num_iter, t, agent_idx);
                    pop_data.selfish();
                }
                agent.is_willing_selfish()
            });

            // register observer agents
            if let Some(observer) = &self.scenario.observer {
                if pop_data.num_selfish > 0 {
                    let observer_prob =
                        V::from_u32(pop_data.num_selfish).unwrap() / self.scenario.fnum_nodes;
                    log::info!("q = {observer_prob:?}");
                    // E[k] = observer.k
                    let k = observer.k.trunc().to_usize().unwrap()
                        + if observer.k.fract() > self.rng.gen() {
                            1
                        } else {
                            0
                        };
                    for agent_idx in rand::seq::index::sample(self.rng, self.scenario.num_nodes, k)
                    {
                        if observer_prob > self.rng.gen() {
                            let informs = event_table.entry(t + 1).or_default();
                            // senders of observed info have priority over existing senders.
                            informs.push_front((agent_idx, observer.observed_info_obj_idx));
                        }
                    }
                }
            }

            for (info_label, d) in &mut info_data_map {
                info_stat.push(num_par, num_iter, t, d, info_label);
                *d = InfoData::default();
            }
            pop_stat.push(num_par, num_iter, t, pop_data);
            t += 1;
        }
        self.sender.send(info_stat.into()).unwrap();
        self.sender.send(agent_stat.into()).unwrap();
        self.sender.send(pop_stat.into()).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use crate::Runner;
    use arrow2::io::ipc::read::read_file_metadata;
    use std::fs;

    #[test]
    fn test_exec() -> anyhow::Result<()> {
        let general_path = "./test/config/general.toml";
        let runtime_path = "./test/config/runtime.toml";
        let agent_params_path = "./test/config/agent_params.toml";
        let scenario_path = "./test/config/scenario.toml";
        let mut runner = Runner::<_, f32>::try_new(
            general_path,
            runtime_path,
            agent_params_path,
            scenario_path,
            "run_test".to_string(),
            true,
        )?;
        runner.run()?;

        let mut reader = fs::File::open("./test/run_test_info_out.arrow")?;
        let metadata = read_file_metadata(&mut reader)?;
        let metadata = metadata.schema.metadata;
        assert_eq!(metadata["general"], general_path);
        assert_eq!(metadata["runtime"], runtime_path);
        assert_eq!(metadata["agent_params"], agent_params_path);
        assert_eq!(metadata["scenario"], scenario_path);
        Ok(())
    }
}
