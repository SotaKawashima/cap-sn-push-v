mod agent;
pub mod config;
mod cpt;
mod info;
mod opinion;
mod value;

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Write;
use std::iter::Sum;
use std::mem;

use approx::UlpsEq;
use arrow2::array::{Array, PrimitiveArray, Utf8Array};
use arrow2::chunk::Chunk;
use arrow2::datatypes::{DataType, Field, Schema};
use arrow2::io::ipc::write::{Compression, FileWriter, WriteOptions};
use config::{Config, ConfigFormat, EventTable};
use num_traits::{Float, FromPrimitive, NumAssign};
use rand::Rng;
use rand::{rngs::SmallRng, seq::SliceRandom, SeedableRng};

use agent::{Agent, AgentParams, Behavior};
use info::{Info, InfoContent, InfoLabel};

use graph_lib::prelude::{Graph, GraphB};
use rand_distr::uniform::SampleUniform;
use rand_distr::{Distribution, Exp1, Open01, Standard, StandardNormal};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

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
    V: Float + NumAssign + UlpsEq + Default + Sum + std::fmt::Debug + SampleUniform,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
{
    pub fn try_new(
        config_data: ConfigFormat,
        identifier: String,
        overwriting: bool,
    ) -> Result<Self, Box<dyn std::error::Error>>
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

    pub fn run(&mut self) -> Result<(), Box<dyn std::error::Error>>
    where
        V: FromPrimitive + Sync,
        <V as SampleUniform>::Sampler: Sync,
    {
        let output_path = self.config.output.location.join(format!(
            "{}_{}.arrow",
            self.identifier, self.config.output.suffix
        ));
        if !self.overwriting && output_path.exists() {
            panic!(
                "{} already exists. If you want to overwrite it, run with the overwriting option.",
                output_path.display()
            );
        }
        let writer = File::create(output_path)?;

        let n = V::from_usize(self.config.runtime.graph.node_count()).unwrap();
        let d = V::from_usize(self.config.runtime.graph.edge_count()).unwrap() / n; // average outdegree

        let mut rng = SmallRng::seed_from_u64(self.config.runtime.seed_state);
        let rngs = (0..(self.config.runtime.num_parallel))
            .map(|_| SmallRng::from_rng(&mut rng))
            .collect::<Result<Vec<_>, _>>()?;

        println!("started.");
        let stats = rngs
            .into_par_iter()
            .enumerate()
            .map(|(num_par, mut rng)| {
                let num_par = num_par as u32;
                let mut env = Environment::new(
                    &self.config.runtime.graph,
                    &self.config.scenario.info_contents,
                    &self.config.scenario.agent_params,
                    &self.config.scenario.event_table,
                    &mut rng,
                    n,
                    d,
                );
                for num_iter in 0..(self.config.runtime.iteration_count) {
                    env.execute(num_par, num_iter);
                }
                env.stat
            })
            .collect::<Vec<_>>();
        println!("finished.");
        self.write(writer, &stats)?;
        println!("done.");
        Ok(())
    }

    fn write<W: Write>(&self, writer: W, stats: &[Stat]) -> arrow2::error::Result<()> {
        let metadata = BTreeMap::from_iter([
            ("version".to_string(), env!("CARGO_PKG_VERSION").to_string()),
            (
                "num_parallel".to_string(),
                self.config.runtime.num_parallel.to_string(),
            ),
            (
                "iteration_count".to_string(),
                self.config.runtime.iteration_count.to_string(),
            ),
        ]);
        let compression: Option<Compression> = if self.config.output.compress {
            Some(Compression::ZSTD)
        } else {
            None
        };

        let mut writer = FileWriter::try_new(
            writer,
            Schema {
                fields: Stat::get_fields(),
                metadata,
            },
            None,
            WriteOptions { compression },
        )?;
        for stat in stats {
            writer.write(&stat.try_into()?, None)?;
        }
        writer.finish()
    }
}

#[derive(Debug)]
enum SendRole {
    Inform,
    Share,
}

#[derive(Debug)]
struct Receiver {
    agent_idx: usize,
    info_idx: usize,
    role: SendRole,
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
    n: V,
    d: V,
    agents: Vec<Agent<V>>,
    stat: Stat,
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
        graph: &'a GraphB,
        info_contents: &'a [InfoContent<V>],
        agent_params: &'a AgentParams<V>,
        event_table: &'a EventTable,
        rng: &'a mut R,
        n: V,
        d: V,
    ) -> Self
    where
        V: Default + NumAssign,
    {
        let agents = (0..graph.node_count())
            .map(|_| Agent::new())
            .collect::<Vec<_>>();
        Self {
            graph,
            info_contents,
            agent_params,
            event_table,
            rng,
            n,
            d,
            agents,
            stat: Stat::default(),
        }
    }

    fn execute(&mut self, num_par: u32, num_iter: u32)
    where
        V: FromPrimitive + Sum + Default + std::fmt::Debug,
    {
        for a in &mut self.agents {
            a.reset_with(self.agent_params, self.rng);
        }

        let mut infos = Vec::<Info<V>>::new();
        let mut stack = Vec::new();
        let mut t = 0;
        let mut info_data_map = BTreeMap::<InfoLabel, InfoData>::new();
        let mut event_table = self.event_table.0.clone();

        log::info!("#i: {num_iter}");

        while !stack.is_empty() || !event_table.is_empty() {
            log::info!("t = {t}");
            if let Some(informms) = event_table.remove(&t) {
                for (agent_idx, info_content_idx) in informms {
                    let info_idx = infos.len();
                    infos.push(Info::new(info_idx, &self.info_contents[info_content_idx]));
                    stack.push(Receiver {
                        agent_idx,
                        info_idx,
                        role: SendRole::Inform,
                    });
                }
            }
            stack.shuffle(self.rng);

            let mut num_selfish = 0;
            for Receiver {
                agent_idx,
                info_idx,
                role,
            } in mem::take(&mut stack)
            {
                let agent = &mut self.agents[agent_idx];
                let info = &mut infos[info_idx];
                let info_label = info.content.label;
                let info_data = info_data_map.entry(info_label).or_default();
                info_data.num_received += 1;

                log::info!("{info_label} -> #{agent_idx}({role:?})");

                let send = match role {
                    SendRole::Inform => {
                        if agent.set_info_opinions(info, &self.agent_params.base_rates) {
                            num_selfish += 1;
                        }
                        true
                    }
                    SendRole::Share => {
                        let friend_receipt_prob = V::one()
                            - (V::one() - V::from_usize(info.num_shared()).unwrap() / self.n)
                                .powf(self.d);
                        log::info!("r^i_m = {friend_receipt_prob:?}");

                        if let Some(b) =
                            agent.read_info(info, friend_receipt_prob, self.agent_params, self.rng)
                        {
                            if b.selfish {
                                num_selfish += 1;
                            }
                            info.update_stat(&b);
                            info_data.update(&b);
                            b.sharing
                        } else {
                            false
                        }
                    }
                };

                if send {
                    for bid in self.graph.successors(agent_idx) {
                        stack.push(Receiver {
                            agent_idx: *bid,
                            info_idx,
                            role: SendRole::Share,
                        });
                    }
                }
            }
            for (info_label, d) in mem::take(&mut info_data_map) {
                d.push_to_stat(&mut self.stat, num_par, num_iter, t, info_label);
            }
            self.stat
                .push(num_par, num_iter, t, num_selfish, "num_selfish".to_string());
            t += 1;
        }
    }
}

#[derive(Default)]
struct InfoData {
    num_received: u32,
    num_shared: u32,
    num_viewed: u32,
    num_fst_read: u32,
}

impl InfoData {
    fn update(&mut self, b: &Behavior) {
        self.num_viewed += 1;
        if b.first_reading {
            self.num_fst_read += 1;
        }
        if b.sharing {
            self.num_shared += 1;
        }
    }

    fn push_to_stat(
        self,
        stat: &mut Stat,
        num_par: u32,
        num_iter: u32,
        t: u32,
        info_label: InfoLabel,
    ) {
        stat.push(
            num_par,
            num_iter,
            t,
            self.num_received,
            format!("num_received:{info_label}"),
        );
        stat.push(
            num_par,
            num_iter,
            t,
            self.num_shared,
            format!("num_shared:{info_label}"),
        );
        stat.push(
            num_par,
            num_iter,
            t,
            self.num_viewed,
            format!("num_viewed:{info_label}"),
        );
        stat.push(
            num_par,
            num_iter,
            t,
            self.num_fst_read,
            format!("num_fst_read:{info_label}"),
        );
    }
}

#[derive(Default)]
struct Stat {
    num_par: Vec<u32>,
    num_iter: Vec<u32>,
    t: Vec<u32>,
    num_people: Vec<u32>,
    kind: Vec<String>,
}

impl TryFrom<&Stat> for Chunk<Box<dyn Array>> {
    type Error = arrow2::error::Error;

    fn try_from(value: &Stat) -> Result<Self, Self::Error> {
        Chunk::try_new(vec![
            PrimitiveArray::from_slice(&value.num_par).boxed(),
            PrimitiveArray::from_slice(&value.num_iter).boxed(),
            PrimitiveArray::from_slice(&value.t).boxed(),
            PrimitiveArray::from_slice(&value.num_people).boxed(),
            Utf8Array::<i32>::from_slice(&value.kind).boxed(),
        ])
    }
}

impl Stat {
    fn push(&mut self, num_par: u32, num_iter: u32, t: u32, num_people: u32, kind: String) {
        self.num_par.push(num_par);
        self.num_iter.push(num_iter);
        self.t.push(t);
        self.num_people.push(num_people);
        self.kind.push(kind);
    }

    fn get_fields() -> Vec<Field> {
        vec![
            Field::new("num_par", DataType::UInt32, false),
            Field::new("num_iter", DataType::UInt32, false),
            Field::new("t", DataType::UInt32, false),
            Field::new("num_people", DataType::UInt32, false),
            Field::new("kind", DataType::Utf8, false),
        ]
    }
}
