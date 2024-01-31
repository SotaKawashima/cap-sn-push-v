mod agent;
mod cpt;
mod info;
mod opinion;

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::iter::Sum;
use std::ops::Deref;

use agent::{Agent, Constants};
use approx::UlpsEq;
use arrow2::array::{PrimitiveArray, Utf8Array};
use arrow2::chunk::Chunk;
use arrow2::datatypes::{DataType, Field, Schema};
use arrow2::io::ipc::write::{Compression, FileWriter, WriteOptions};
use cpt::{LevelSet, CPT};
use info::{Info, InfoType, ToInfoContent, TrustDists};
use opinion::{
    FriendOpinions, FriendStaticOpinions, GlobalBaseRates, Opinions, PluralIgnorance,
    StaticOpinions,
};

use graph_lib::io::{DataFormat, ParseBuilder};
use graph_lib::prelude::{DiGraphB, Graph};
use log::debug;
use num_traits::{Float, FromPrimitive, NumAssign};
use rand::Rng;
use rand::{rngs::SmallRng, seq::SliceRandom, SeedableRng};
use rand_distr::{Beta, Distribution, Open01, Standard};
use subjective_logic::harr2;

#[derive(serde::Deserialize, Debug)]
struct Strategy<V: Float> {
    /// probability whether people has plural ignorance
    pi_prob: V,
    scenario: Vec<Event>,
}

fn extract_scenario<V>(
    scenario: Vec<Event>,
) -> (Vec<Info<V>>, BTreeMap<u32, BTreeMap<usize, usize>>)
where
    V: Float + UlpsEq,
    for<'a> &'a InfoType: ToInfoContent<V>,
{
    let mut info_types = Vec::new();
    let scenario = scenario
        .into_iter()
        .map(|Event { time, informers }| {
            (
                time,
                informers
                    .into_iter()
                    .map(
                        |Inform {
                             agent_idx,
                             info_type,
                         }| {
                            let info_id = info_types.len();
                            info_types.push(info_type);
                            (agent_idx, info_id)
                        },
                    )
                    .collect(),
            )
        })
        .collect();

    let infos = info_types
        .iter()
        .enumerate()
        .map(|(id, t)| Info::new(id, *t, t.into()))
        .collect::<Vec<_>>();

    (infos, scenario)
}

#[derive(serde::Deserialize, Debug)]
struct Event {
    time: u32,
    informers: Vec<Inform>,
}

#[derive(serde::Deserialize, Debug)]
struct Inform {
    agent_idx: usize,
    info_type: InfoType,
}

#[derive(Debug)]
struct Receipt {
    agent_idx: usize,
    force: bool,
    info_id: usize,
}

#[derive(Default)]
struct Stat {
    num_iter: Vec<u32>,
    t: Vec<u32>,
    num_people: Vec<u32>,
    kind: Vec<String>,
}

impl Stat {
    fn push(&mut self, num_iter: u32, t: u32, num_people: u32, kind: String) {
        self.num_iter.push(num_iter);
        self.t.push(t);
        self.num_people.push(num_people);
        self.kind.push(kind);
    }

    fn write<W: Write>(
        &self,
        writer: W,
        compression: Option<Compression>,
    ) -> arrow2::error::Result<()> {
        let metadata = BTreeMap::from_iter([
            ("version".to_string(), env!("CARGO_PKG_VERSION").to_string()),
            ("graph".to_string(), "filmtrust".to_string()),
        ]);
        let schema = Schema {
            fields: vec![
                Field::new("num_iter", DataType::UInt32, false),
                Field::new("t", DataType::UInt32, false),
                Field::new("num_people", DataType::UInt32, false),
                Field::new("kind", DataType::Utf8, false),
            ],
            metadata,
        };
        let chunk = Chunk::try_new(vec![
            PrimitiveArray::from_slice(&self.num_iter).boxed(),
            PrimitiveArray::from_slice(&self.t).boxed(),
            PrimitiveArray::from_slice(&self.num_people).boxed(),
            Utf8Array::<i32>::from_slice(&self.kind).boxed(),
        ])?;
        let mut writer = FileWriter::try_new(writer, schema, None, WriteOptions { compression })?;
        writer.write(&chunk, None)?;
        writer.finish()
    }
}

pub struct Executor<V>
where
    V: Float,
    Open01: Distribution<V>,
{
    scenario: BTreeMap<u32, BTreeMap<usize, usize>>,
    stat: Stat,
    seed_state: u64,
    agents: Vec<Agent<V>>,
    infos: Vec<Info<V>>,
    graph: DiGraphB,
    constants: Constants<V>,
}

macro_rules! impl_executor {
    ($ft: ty) => {
        impl Executor<$ft> {
            pub fn try_new<R: Read>(
                reader: R,
                strategy: &str,
                seed_state: u64,
            ) -> Result<Self, Box<dyn std::error::Error>> {
                let Strategy { pi_prob, scenario } = serde_json::from_str(strategy)?;
                let (infos, scenario) = extract_scenario(scenario);

                let constants = Constants::<$ft> {
                    base_rates: GlobalBaseRates {
                        s: [0.999, 0.001],
                        fs: [0.99, 0.01],
                        psi: [0.999, 0.001],
                        ppsi: [0.999, 0.001],
                        pa: [0.999, 0.001],
                        fa: [0.999, 0.001],
                        fpa: [0.999, 0.001],
                        phi: [0.999, 0.001],
                        fpsi: [0.999, 0.001],
                        fppsi: [0.999, 0.001],
                        fphi: [0.999, 0.001],
                        theta: [0.999, 0.0005, 0.0005],
                        ptheta: [0.999, 0.0005, 0.0005],
                        ftheta: [0.999, 0.0005, 0.0005],
                        fptheta: [0.999, 0.0005, 0.0005],
                    },
                    read_dist: Beta::new(7.0, 3.0)?,
                    fclose_dist: Beta::new(9.0, 1.0)?,
                    fread_dist: Beta::new(5.0, 5.0)?,
                    pi_dist: Beta::new(19.0, 1.0)?,
                    pi_prob,
                    trust_dists: TrustDists {
                        misinfo: Beta::new(1.5, 4.5)?,
                        corrective: Beta::new(4.5, 1.5)?,
                        observed: Beta::new(4.5, 1.5)?,
                        inhivitive: Beta::new(4.5, 1.5)?,
                    },
                };

                let x0 = -2.0;
                let x1 = -50.0;
                let y = -0.001;
                let selfish_outcome_maps = [[0.0, x1, 0.0], [x0, x0, x0]];
                let sharing_outcome_maps = [
                    harr2![[0.0, x1, 0.0], [x0, x0, x0]],
                    harr2![[y, x1 + y, y], [x0 + y, x0 + y, x0 + y]],
                ];
                let selfish = [
                    LevelSet::<_, $ft>::new(&selfish_outcome_maps[0]),
                    LevelSet::<_, $ft>::new(&selfish_outcome_maps[1]),
                ];
                let sharing = [
                    LevelSet::<_, $ft>::new(sharing_outcome_maps[0].deref()),
                    LevelSet::<_, $ft>::new(sharing_outcome_maps[1].deref()),
                ];

                let graph: DiGraphB = ParseBuilder::new(reader, DataFormat::EdgeList)
                    .parse()
                    .unwrap();

                let mut agents = Vec::with_capacity(graph.node_count());
                for _ in 0..agents.capacity() {
                    agents.push(Agent::new(
                        Opinions::new(),
                        StaticOpinions::<$ft>::new(),
                        FriendOpinions::new(),
                        FriendStaticOpinions::<$ft>::new(),
                        CPT::new(0.88, 0.88, 2.25, 0.61, 0.69),
                        selfish.clone(),
                        sharing.clone(),
                        infos.len(),
                    ));
                }

                Ok(Self {
                    scenario,
                    stat: Stat::default(),
                    seed_state,
                    agents,
                    infos,
                    graph,
                    constants,
                })
            }
        }
    };
}

impl_executor!(f32);
impl_executor!(f64);

impl<V> Executor<V>
where
    V: Float
        + FromPrimitive
        + NumAssign
        + UlpsEq
        + Default
        + Sum
        + PluralIgnorance
        + std::fmt::Debug,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
{
    pub fn exec<W: Write>(
        &mut self,
        iteration_count: u32,
        writer: &mut W,
        out_compression: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut rng = SmallRng::seed_from_u64(self.seed_state);
        let n = V::from_usize(self.graph.node_count()).unwrap();
        let d = V::from_usize(self.graph.edge_count()).unwrap() / n; // average outdegree

        println!("started.");
        for num_iter in 0..iteration_count {
            for a in &mut self.agents {
                a.reset_with(&self.constants, &self.infos, &mut rng);
            }
            for info in &mut self.infos {
                info.reset();
            }
            self.run_loop(num_iter, n, d, &mut rng);
        }
        println!("finished.");
        self.stat.write(
            writer,
            if out_compression {
                Some(Compression::ZSTD)
            } else {
                None
            },
        )?;
        println!("done.");
        Ok(())
    }

    fn run_loop<R>(&mut self, num_iter: u32, n: V, d: V, rng: &mut R)
    where
        R: Rng,
    {
        let mut received = Vec::new();
        let mut scenario = self.scenario.clone();
        let mut t = 0;
        let mut num_view_map = BTreeMap::<InfoType, u32>::new();
        let mut num_sharing_map = BTreeMap::<InfoType, u32>::new();
        let mut num_receipt_map = BTreeMap::<InfoType, u32>::new();
        while !received.is_empty() || !scenario.is_empty() {
            if let Some(informers) = scenario.remove(&t) {
                for (agent_idx, info_id) in informers {
                    received.push(Receipt {
                        agent_idx,
                        force: true,
                        info_id,
                    });
                }
            }
            received.shuffle(rng);

            let mut num_selfish = 0;
            num_receipt_map.clear();
            num_view_map.clear();
            num_sharing_map.clear();
            received = received
                .into_iter()
                .filter_map(|r| {
                    let agent = &mut self.agents[r.agent_idx];
                    let info = &mut self.infos[r.info_id];
                    let receipt_prob = V::one()
                        - (V::one() - V::from_usize(info.num_shared()).unwrap() / n).powf(d);

                    let num_receipt = num_receipt_map.entry(info.info_type).or_insert(0);
                    *num_receipt += 1;

                    let b = if r.force {
                        agent.read_info_trustfully(info, receipt_prob, &self.constants.base_rates)
                    } else {
                        agent.read_info(rng, info, receipt_prob, &self.constants.base_rates)?
                    };

                    debug!(
                        "{}: {}",
                        if r.force { "informer" } else { "sharer" },
                        info.info_type
                    );

                    if !r.force {
                        let num_view = num_view_map.entry(info.info_type).or_insert(0);
                        *num_view += 1;
                    }
                    if !b.sharing {
                        return None;
                    }
                    if !r.force {
                        info.shared();
                        let num_sharing = num_sharing_map.entry(info.info_type).or_insert(0);
                        *num_sharing += 1;
                    }
                    if b.selfish {
                        num_selfish += 1;
                    }
                    Some(
                        self.graph
                            .successors(r.agent_idx)
                            .unwrap()
                            .iter()
                            .map(|bid| Receipt {
                                agent_idx: *bid,
                                force: false,
                                info_id: r.info_id,
                            })
                            .collect::<Vec<_>>(),
                    )
                })
                .flatten()
                .collect();
            for (k, n) in &num_receipt_map {
                self.stat
                    .push(num_iter, t, *n, format!("num_receipt:{}", k));
            }
            for (k, n) in &num_view_map {
                self.stat.push(num_iter, t, *n, format!("num_view:{}", k));
            }
            for (k, n) in &num_sharing_map {
                self.stat
                    .push(num_iter, t, *n, format!("num_sharing:{}", k));
            }
            self.stat
                .push(num_iter, t, num_selfish, "num_selfish".to_string());
            t += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        agent::{Agent, Constants},
        cpt::{LevelSet, CPT},
        extract_scenario,
        info::TrustDists,
        info::{Info, InfoType},
        opinion::{
            FriendOpinions, FriendStaticOpinions, GlobalBaseRates, Opinions, StaticOpinions,
        },
        Strategy,
    };

    use rand_distr::Beta;
    use serde_json::json;
    use std::ops::Deref;
    use subjective_logic::harr2;

    #[test]
    fn test_agent() {
        let constants = Constants {
            base_rates: GlobalBaseRates {
                s: [0.999, 0.001],
                fs: [0.999, 0.001],
                psi: [0.999, 0.001],
                ppsi: [0.999, 0.001],
                pa: [0.999, 0.001],
                fa: [0.999, 0.001],
                fpa: [0.999, 0.001],
                phi: [0.999, 0.001],
                fpsi: [0.999, 0.001],
                fppsi: [0.999, 0.001],
                fphi: [0.999, 0.001],
                theta: [0.999, 0.0005, 0.0005],
                ptheta: [0.999, 0.0005, 0.0005],
                ftheta: [0.999, 0.0005, 0.0005],
                fptheta: [0.999, 0.0005, 0.0005],
            },
            read_dist: Beta::new(7.0, 3.0).unwrap(),
            fclose_dist: Beta::new(9.0, 1.0).unwrap(),
            fread_dist: Beta::new(5.0, 5.0).unwrap(),
            pi_dist: Beta::new(0.5 * 10.0, (1.0 - 0.5) * 10.0).unwrap(),
            pi_prob: 0.5,
            trust_dists: TrustDists {
                misinfo: Beta::new(1.5, 4.5).unwrap(),
                corrective: Beta::new(4.5, 1.5).unwrap(),
                observed: Beta::new(4.5, 1.5).unwrap(),
                inhivitive: Beta::new(4.5, 1.5).unwrap(),
            },
        };

        let info_types = [InfoType::Misinfo];
        let infos = info_types.map(|t| Info::new(0, t, t.into()));

        let x0 = -0.1;
        let x1 = -2.0;
        let y = -0.001;
        let selfish_outcome_maps = [[0.0, x1, 0.0], [x0, x0, x0]];
        let sharing_outcome_maps = [
            harr2![[0.0, x1, 0.0], [x0, x0, x0]],
            harr2![[y, x1 + y, y], [x0 + y, x0 + y, x0 + y]],
        ];

        let mut a = Agent::new(
            Opinions::new(),
            StaticOpinions::<f32>::new(),
            FriendOpinions::new(),
            FriendStaticOpinions::<f32>::new(),
            CPT::new(0.88, 0.88, 2.25, 0.61, 0.69),
            [
                LevelSet::<_, f32>::new(&selfish_outcome_maps[0]),
                LevelSet::<_, f32>::new(&selfish_outcome_maps[1]),
            ],
            [
                LevelSet::<_, f32>::new(sharing_outcome_maps[0].deref()),
                LevelSet::<_, f32>::new(sharing_outcome_maps[1].deref()),
            ],
            1,
        );

        a.reset(0.5, 0.5, 0.5, 0.0, |_| 0.90, &constants.base_rates, &infos);

        let receipt_prob = 0.0;
        println!(
            "{:?}",
            a.read_info_trustfully(&infos[0], receipt_prob, &constants.base_rates)
        );
    }

    #[test]
    fn test_strategy() {
        let v = json!(
            {
                "pi_prob": 1.0,
                "scenario": [
                    {
                        "time": 0,
                        "informers": [
                            {"agent_idx": 0, "info_type": "Misinfo"},
                            {"agent_idx": 0, "info_type": "Misinfo"}
                        ],
                    },
                    {
                        "time": 1,
                        "informers": [{"agent_idx": 2, "info_type": "Misinfo"}]
                    },
                ]
            }
        );
        let s: Strategy<f32> = serde_json::from_value(v).unwrap();
        println!("{:?}", s);
        let (is, sc) = extract_scenario::<f32>(s.scenario);
        println!("{:?}", is);
        println!("{:?}", sc);
    }
}
