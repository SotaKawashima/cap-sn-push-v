mod agent;
pub mod config;
mod cpt;
mod dist;
mod info;
mod opinion;

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Write;
use std::iter::Sum;

use approx::UlpsEq;
use arrow2::array::{Array, PrimitiveArray, Utf8Array};
use arrow2::chunk::Chunk;
use arrow2::datatypes::{DataType, Field, Schema};
use arrow2::io::ipc::write::{Compression, FileWriter, WriteOptions};
use config::{Config, ConfigFormat, EventTable};
use num_traits::{Float, FromPrimitive, NumAssign};
use rand::Rng;
use rand::{rngs::SmallRng, seq::SliceRandom, SeedableRng};

use agent::{Agent, AgentParams};
use info::{Info, InfoContent, InfoLabel};

use graph_lib::prelude::{Graph, GraphB};
use rand_distr::uniform::SampleUniform;
use rand_distr::{Distribution, Open01, Standard};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

#[derive(Debug)]
struct Receipt {
    agent_idx: usize,
    force: bool,
    info_idx: usize,
}

pub struct Runner<V>
where
    V: Float + UlpsEq + NumAssign + SampleUniform,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
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

struct Environment<'a, V, R>
where
    V: Float + UlpsEq + SampleUniform + NumAssign,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
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
        let mut received = Vec::new();
        let mut t = 0;
        let mut num_view_map = BTreeMap::<InfoLabel, u32>::new();
        let mut num_sharing_map = BTreeMap::<InfoLabel, u32>::new();
        let mut num_receipt_map = BTreeMap::<InfoLabel, u32>::new();
        let mut event_table = self.event_table.0.clone();

        while !received.is_empty() || !event_table.is_empty() {
            if let Some(informms) = event_table.remove(&t) {
                for (agent_idx, info_content_idx) in informms {
                    let info_idx = infos.len();
                    infos.push(Info::new(info_idx, &self.info_contents[info_content_idx]));
                    received.push(Receipt {
                        agent_idx,
                        force: true,
                        info_idx,
                    });
                }
            }
            received.shuffle(self.rng);

            let mut num_selfish = 0;
            num_receipt_map.clear();
            num_view_map.clear();
            num_sharing_map.clear();
            received = received
                .into_iter()
                .filter_map(|r| {
                    let agent = &mut self.agents[r.agent_idx];
                    let info = &mut infos[r.info_idx];
                    let receipt_prob = V::one()
                        - (V::one() - V::from_usize(info.num_shared()).unwrap() / self.n)
                            .powf(self.d);

                    let num_receipt = num_receipt_map.entry(info.content.label).or_insert(0);
                    *num_receipt += 1;

                    log::info!(
                        "{} -> #{}:{}",
                        info.content.label,
                        r.agent_idx,
                        if r.force { "informer" } else { "sharer" },
                    );

                    let b = if r.force {
                        agent.read_info_trustfully(info, receipt_prob, self.agent_params)
                    } else {
                        agent.read_info(info, receipt_prob, self.agent_params, self.rng)?
                    };

                    if !r.force {
                        let num_view = num_view_map.entry(info.content.label).or_insert(0);
                        *num_view += 1;
                    }
                    if !b.sharing {
                        return None;
                    }
                    if !r.force {
                        info.shared();
                        let num_sharing = num_sharing_map.entry(info.content.label).or_insert(0);
                        *num_sharing += 1;
                    }
                    if b.selfish {
                        num_selfish += 1;
                    }
                    Some(
                        self.graph
                            .successors(r.agent_idx)
                            .map(|bid| Receipt {
                                agent_idx: *bid,
                                force: false,
                                info_idx: r.info_idx,
                            })
                            .collect::<Vec<_>>(),
                    )
                })
                .flatten()
                .collect();
            for (k, n) in &num_receipt_map {
                self.stat
                    .push(num_par, num_iter, t, *n, format!("num_receipt:{}", k));
            }
            for (k, n) in &num_view_map {
                self.stat
                    .push(num_par, num_iter, t, *n, format!("num_view:{}", k));
            }
            for (k, n) in &num_sharing_map {
                self.stat
                    .push(num_par, num_iter, t, *n, format!("num_sharing:{}", k));
            }
            self.stat
                .push(num_par, num_iter, t, num_selfish, "num_selfish".to_string());
            t += 1;
        }
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
