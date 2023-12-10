pub mod agent;
pub mod cpt;
pub mod info;
pub mod opinion;
pub mod snippet;

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::ops::Deref;

use agent::{Agent, AgentOpinion, Constants, FriendOpinion};
use arrow2::array::{PrimitiveArray, Utf8Array};
use arrow2::chunk::Chunk;
use arrow2::datatypes::{DataType, Field, Schema};
use arrow2::io::ipc::write::{Compression, FileWriter, WriteOptions};
use cpt::{LevelSet, CPT};
use info::{Info, InfoType};

use graph_lib::io::{DataFormat, ParseBuilder};
use graph_lib::prelude::{DiGraphB, Graph};
use rand::Rng;
use rand::{rngs::SmallRng, seq::SliceRandom, SeedableRng};
use rand_distr::Beta;
use subjective_logic::harr2;

#[derive(serde::Deserialize, Debug)]
struct Scenario(Vec<Event>);

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

impl Scenario {
    fn extract(self) -> (Vec<InfoType>, BTreeMap<u32, BTreeMap<usize, usize>>) {
        let mut info_types = Vec::new();
        let scenario = self
            .0
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
        (info_types, scenario)
    }
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
    // num_view: Vec<u32>,
    // num_sharing: Vec<u32>,
    // num_selfish: Vec<u32>,
}

impl Stat {
    fn push(
        &mut self,
        num_iter: u32,
        t: u32,
        num_people: u32,
        kind: String,
        // num_receipt: u32,
        // num_sharing: u32,
        // num_selfish: u32,
    ) {
        self.num_iter.push(num_iter);
        self.t.push(t);
        self.num_people.push(num_people);
        self.kind.push(kind);
        // self.num_view.push(num_view);
        // self.num_receipt.push(num_receipt);
        // self.num_sharing.push(num_sharing);
        // self.num_selfish.push(num_selfish);
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
            // PrimitiveArray::from_slice(&self.num_sharing).boxed(),
            // PrimitiveArray::from_slice(&self.num_selfish).boxed(),
        ])?;
        let mut writer = FileWriter::try_new(writer, schema, None, WriteOptions { compression })?;
        writer.write(&chunk, None)?;
        writer.finish()
    }
}

pub struct Executor {
    scenario: BTreeMap<u32, BTreeMap<usize, usize>>,
    stat: Stat,
    seed_state: u64,
    agents: Vec<Agent>,
    info_types: Vec<InfoType>,
    infos: Vec<Info>,
    graph: DiGraphB,
    constants: Constants,
}

impl Executor {
    pub fn try_new<R: Read>(
        reader: R,
        strategy: &str,
        seed_state: u64,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let scenario: Scenario = serde_json::from_str(strategy)?;
        let (info_types, scenario) = scenario.extract();

        let constants = Constants {
            br_psi: [0.999, 0.001],
            br_ppsi: [0.999, 0.001],
            br_pa: [0.999, 0.001],
            br_fa: [0.999, 0.001],
            br_fpa: [0.999, 0.001],
            br_phi: [0.999, 0.001],
            br_fpsi: [0.999, 0.001],
            br_fppsi: [0.999, 0.001],
            br_fphi: [0.999, 0.001],
            br_theta: [0.999, 0.0005, 0.0005],
            br_ptheta: [0.999, 0.0005, 0.0005],
            br_ftheta: [0.999, 0.0005, 0.0005],
            br_fptheta: [0.999, 0.0005, 0.0005],
            read_dist: Beta::new(7.0, 3.0)?,
            fclose_dist: Beta::new(9.0, 1.0)?,
            fread_dist: Beta::new(5.0, 5.0)?,
            misinfo_trust_dist: Beta::new(1.5, 4.5)?,
            correction_trust_dist: Beta::new(4.5, 1.5)?,
        };

        let infos = info_types
            .iter()
            .enumerate()
            .map(|(id, t)| Info::new(id, *t, t.into()))
            .collect::<Vec<_>>();

        let x0 = -0.01;
        let x1 = -2.0;
        let y = -0.0002;
        let selfish_outcome_maps = [[0.0, x1, 0.0], [x0, x0, x0]];
        let sharing_outcome_maps = [
            harr2![[0.0, x1, 0.0], [x0, x0, x0]],
            harr2![[y, x1 + y, y], [x0 + y, x0 + y, x0 + y]],
        ];
        let selfish = [
            LevelSet::<_, f32>::new(&selfish_outcome_maps[0]),
            LevelSet::<_, f32>::new(&selfish_outcome_maps[1]),
        ];
        let sharing = [
            LevelSet::<_, f32>::new(sharing_outcome_maps[0].deref()),
            LevelSet::<_, f32>::new(sharing_outcome_maps[1].deref()),
        ];

        let graph: DiGraphB = ParseBuilder::new(reader, DataFormat::EdgeList)
            .parse()
            .unwrap();

        let mut agents = Vec::with_capacity(graph.node_count());
        for _ in 0..agents.capacity() {
            agents.push(Agent::new(
                AgentOpinion::new(),
                FriendOpinion::new(constants.br_fpa, constants.br_fptheta, constants.br_ftheta),
                CPT::new(0.88, 0.88, 2.25, 0.61, 0.69),
                selfish.clone(),
                sharing.clone(),
                constants.br_pa,
                constants.br_ptheta,
                constants.br_fa,
                infos.len(),
            ));
        }

        Ok(Self {
            scenario,
            stat: Stat::default(),
            seed_state,
            agents,
            info_types,
            infos,
            graph,
            constants,
        })
    }

    pub fn exec<W: Write>(
        &mut self,
        iteration_count: u32,
        writer: &mut W,
        out_compression: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut rng = SmallRng::seed_from_u64(self.seed_state);
        let n = self.graph.node_count() as f32;
        let d = self.graph.edge_count() as f32 / n; // average outdegree

        println!("started.");
        for num_iter in 0..iteration_count {
            for a in &mut self.agents {
                a.reset_with(&self.constants, &self.info_types, &mut rng);
            }
            for info in &mut self.infos {
                info.reset();
            }
            run_loop(
                Vec::new(),
                self.scenario.clone(),
                num_iter,
                n,
                d,
                &mut rng,
                &self.constants,
                &self.graph,
                &mut self.agents,
                &mut self.infos,
                &mut self.stat,
            );
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
}

fn run_loop<R: Rng>(
    mut received: Vec<Receipt>,
    mut event: BTreeMap<u32, BTreeMap<usize, usize>>,
    num_iter: u32,
    n: f32,
    d: f32,
    rng: &mut R,
    constants: &Constants,
    graph: &DiGraphB,
    agents: &mut [Agent],
    infos: &mut [Info],
    stat: &mut Stat,
) {
    let mut t = 0;
    let mut num_view_map = BTreeMap::<InfoType, u32>::new();
    let mut num_sharing_map = BTreeMap::<InfoType, u32>::new();
    let mut num_receipt_map = BTreeMap::<InfoType, u32>::new();
    while !received.is_empty() || !event.is_empty() {
        if let Some(informers) = event.remove(&t) {
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
                let agent = &mut agents[r.agent_idx];
                let info = &mut infos[r.info_id];
                let receipt_prob = 1.0 - (1.0 - info.num_shared() as f32 / n).powf(d);

                let num_receipt = num_receipt_map.entry(info.info_type).or_insert(0);
                *num_receipt += 1;

                let b = if r.force {
                    agent.read_info_trustfully(info, receipt_prob, constants)
                } else {
                    agent.read_info(rng, info, receipt_prob, constants)?
                };
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
                    graph
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
            stat.push(num_iter, t, *n, format!("num_receipt:{}", k));
        }
        for (k, n) in &num_view_map {
            stat.push(num_iter, t, *n, format!("num_view:{}", k));
        }
        for (k, n) in &num_sharing_map {
            stat.push(num_iter, t, *n, format!("num_sharing:{}", k));
        }
        stat.push(num_iter, t, num_selfish, "num_selfish".to_string());
        t += 1;
    }
}

#[cfg(test)]
mod tests {
    use crate::Scenario;

    use super::{
        agent::{Agent, AgentOpinion, Constants, FriendOpinion},
        cpt::{LevelSet, CPT},
        info::{Info, InfoType},
    };
    use rand_distr::Beta;
    use serde_json::json;
    use std::ops::Deref;
    use subjective_logic::harr2;

    #[test]
    fn test_agent() {
        let constants = Constants {
            br_psi: [0.999, 0.001],
            br_ppsi: [0.999, 0.001],
            br_pa: [0.999, 0.001],
            br_fa: [0.999, 0.001],
            br_fpa: [0.999, 0.001],
            br_phi: [0.999, 0.001],
            br_fpsi: [0.999, 0.001],
            br_fppsi: [0.999, 0.001],
            br_fphi: [0.999, 0.001],
            br_theta: [0.999, 0.0005, 0.0005],
            br_ptheta: [0.999, 0.0005, 0.0005],
            br_ftheta: [0.999, 0.0005, 0.0005],
            br_fptheta: [0.999, 0.0005, 0.0005],
            read_dist: Beta::new(7.0, 3.0).unwrap(),
            fclose_dist: Beta::new(9.0, 1.0).unwrap(),
            fread_dist: Beta::new(5.0, 5.0).unwrap(),
            misinfo_trust_dist: Beta::new(1.5, 4.5).unwrap(),
            correction_trust_dist: Beta::new(4.5, 1.5).unwrap(),
        };

        let info_types = [InfoType::Misinfo];
        let infos = info_types.map(|t| Info::new(0, t, t.into()));

        let x0 = -0.1;
        let x1 = -2.0;
        let y = -0.01;
        let selfish_outcome_maps = [[0.0, x1, 0.0], [x0, x0, x0]];
        let sharing_outcome_maps = [
            harr2![[0.0, x1, 0.0], [x0, x0, x0]],
            harr2![[y, x1 + y, y], [x0 + y, x0 + y, x0 + y]],
        ];

        let mut a = Agent::new(
            AgentOpinion::new(),
            FriendOpinion::new(constants.br_fpa, constants.br_fptheta, constants.br_ftheta),
            CPT::new(0.88, 0.88, 2.25, 0.61, 0.69),
            [
                LevelSet::<_, f32>::new(&selfish_outcome_maps[0]),
                LevelSet::<_, f32>::new(&selfish_outcome_maps[1]),
            ],
            [
                LevelSet::<_, f32>::new(sharing_outcome_maps[0].deref()),
                LevelSet::<_, f32>::new(sharing_outcome_maps[1].deref()),
            ],
            constants.br_pa,
            constants.br_ptheta,
            constants.br_fa,
            1,
        );

        a.reset(0.5, 0.5, 0.5, |_| 0.90, &constants, &info_types);

        let receipt_prob = 0.0;
        println!(
            "{:?}",
            a.read_info_trustfully(&infos[0], receipt_prob, &constants)
        );
    }

    #[test]
    fn test_strategy() {
        let v = json!(
            [
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
        );
        let s: Scenario = serde_json::from_value(v).unwrap();
        println!("{:?}", s);
        let (is, sc) = s.extract();
        println!("{:?}", is);
        println!("{:?}", sc);
    }
}
