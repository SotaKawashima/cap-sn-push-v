pub mod agent;
pub mod cpt;
pub mod info;
pub mod opinion;
pub mod snippet;

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::ops::Deref;

use agent::{Agent, AgentOpinion, Constants, FriendOpinion};
use arrow2::array::PrimitiveArray;
use arrow2::chunk::Chunk;
use arrow2::datatypes::{DataType, Field, Schema};
use arrow2::io::ipc::write::{Compression, FileWriter, WriteOptions};
use cpt::{LevelSet, CPT};
use info::{Info, InfoContent};
use opinion::{A, PHI, PSI, THETA};

use graph_lib::io::{DataFormat, ParseBuilder};
use graph_lib::prelude::{DiGraphB, Graph};
use rand::Rng;
use rand::{rngs::SmallRng, seq::SliceRandom, SeedableRng};
use rand_distr::Beta;
use subjective_logic::{
    harr2,
    mul::{Opinion1d, Simplex},
};

#[derive(Clone, serde::Deserialize, Debug)]
struct Informer {
    agent_idx: usize,
    time: u32,
    info_idx: usize,
}

struct Receipt {
    agent_idx: usize,
    force: bool,
    info_idx: usize,
}

struct Stat {
    num_iter: Vec<u32>,
    t: Vec<u32>,
    num_sharing: Vec<u32>,
    num_receipt: Vec<u32>,
}

impl Stat {
    fn push(&mut self, num_iter: u32, t: u32, num_sharing: u32, num_receipt: u32) {
        self.num_iter.push(num_iter);
        self.t.push(t);
        self.num_sharing.push(num_sharing);
        self.num_receipt.push(num_receipt);
    }

    fn write<W: Write>(
        &self,
        writer: W,
        compression: Option<Compression>,
    ) -> arrow2::error::Result<()> {
        let metadata = BTreeMap::from_iter([
            ("version".to_string(), env!("CARGO_PKG_VERSION").to_string()),
            ("graph".to_string(), "filmtrust".to_string()),
            ("info".to_string(), "#0: misinfo".to_string()),
        ]);
        let schema = Schema {
            fields: vec![
                Field::new("num_iter", DataType::UInt32, false),
                Field::new("t", DataType::UInt32, false),
                Field::new("num_receipt", DataType::UInt32, false),
                Field::new("num_sharing", DataType::UInt32, false),
            ],
            metadata,
        };
        let chunk = Chunk::try_new(vec![
            PrimitiveArray::from_slice(&self.num_iter).boxed(),
            PrimitiveArray::from_slice(&self.t).boxed(),
            PrimitiveArray::from_slice(&self.num_receipt).boxed(),
            PrimitiveArray::from_slice(&self.num_sharing).boxed(),
        ])?;
        let mut writer = FileWriter::try_new(writer, schema, None, WriteOptions { compression })?;
        writer.write(&chunk, None)?;
        writer.finish()
    }
}

pub struct Executor {
    informers: Vec<Informer>,
    stat: Stat,
    seed_state: u64,
    agents: Vec<Agent>,
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
        let informers = serde_json::from_str(strategy)?;

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
        };

        let infos = vec![Info::new(
            0,
            InfoContent::new(
                Opinion1d::<f32, PSI>::new([0.01, 0.98], 0.01, constants.br_psi),
                Opinion1d::<f32, PSI>::new([0.0, 0.0], 1.0, constants.br_ppsi),
                Opinion1d::<f32, A>::new([0.0, 0.0], 1.0, constants.br_pa),
                Opinion1d::<f32, PHI>::new([0.0, 0.0], 1.0, constants.br_phi),
                [
                    Simplex::<f32, THETA>::vacuous(),
                    Simplex::<f32, THETA>::vacuous(),
                ],
            ),
        )];

        let x0 = -0.1;
        let x1 = -2.0;
        let y = -0.01;
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

        let agent = Agent::new(
            AgentOpinion::new(),
            FriendOpinion::new(constants.br_fpa, constants.br_fptheta, constants.br_ftheta),
            CPT::new(0.88, 0.88, 2.25, 0.61, 0.69),
            selfish.clone(),
            sharing.clone(),
            constants.br_pa,
            constants.br_ptheta,
            constants.br_fa,
        );

        let mut agents = Vec::with_capacity(graph.node_count());
        for _ in 1..agents.capacity() {
            agents.push(agent.clone());
        }
        agents.push(agent);

        Ok(Self {
            informers,
            stat: Stat {
                num_iter: Vec::new(),
                t: Vec::new(),
                num_sharing: Vec::new(),
                num_receipt: Vec::new(),
            },
            seed_state,
            agents,
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
                a.reset_with(&self.constants, &mut rng);
            }
            for info in &mut self.infos {
                info.reset();
            }
            run_loop(
                Vec::new(),
                self.informers.clone(),
                num_iter,
                n,
                d,
                &mut rng,
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
    mut informers: Vec<Informer>,
    num_iter: u32,
    n: f32,
    d: f32,
    rng: &mut R,
    graph: &DiGraphB,
    agents: &mut [Agent],
    infos: &mut [Info],
    stat: &mut Stat,
) {
    let mut t = 0;
    while !received.is_empty() || !informers.is_empty() {
        if informers.first().map(|i| i.time == t).unwrap_or(false) {
            let Informer {
                agent_idx,
                info_idx,
                ..
            } = informers.pop().unwrap();
            received.push(Receipt {
                agent_idx,
                force: true,
                info_idx,
            });
        }
        received.shuffle(rng);

        let mut num_sharing = 0;
        received = received
            .into_iter()
            .filter_map(|r| {
                let agent = &mut agents[r.agent_idx];
                let info = &mut infos[r.info_idx];
                let receipt_prob = 1.0 - (1.0 - info.num_shared() as f32 / n).powf(d);
                let b = if r.force {
                    agent.read_info_trustfully(info, receipt_prob)
                } else {
                    agent.read_info(rng, info, receipt_prob)?
                };
                if !b.sharing {
                    return None;
                }
                if !r.force {
                    info.shared();
                    num_sharing += 1;
                }
                Some(
                    graph
                        .successors(r.agent_idx)
                        .unwrap()
                        .iter()
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
        stat.push(num_iter, t, num_sharing, received.len() as u32);
        t += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::{
        agent::{Agent, AgentOpinion, FriendOpinion},
        cpt::{LevelSet, CPT},
        info::{Info, InfoContent},
        opinion::{A, PHI, PSI, THETA},
    };
    use std::ops::Deref;

    use subjective_logic::{
        harr2,
        mul::{Opinion1d, Simplex},
    };

    #[test]
    fn test_agent() {
        let br_psi = [0.999, 0.001];
        let br_ppsi = [0.999, 0.001];
        let br_pa = [0.999, 0.001];
        let br_fa = [0.999, 0.001];
        let br_fpa = [0.999, 0.001];
        let br_phi = [0.999, 0.001];
        let br_fpsi = [0.999, 0.001];
        let br_fppsi = [0.999, 0.001];
        let br_fphi = [0.999, 0.001];
        let br_theta = [0.999, 0.0005, 0.0005];
        let br_ptheta = [0.999, 0.0005, 0.0005];
        let br_ftheta = [0.999, 0.0005, 0.0005];
        let br_fptheta = [0.999, 0.0005, 0.0005];
        // let read_dist = Beta::new(7.0, 3.0)?;
        // let fclose_dist = Beta::new(9.0, 1.0)?;
        // let fread_dist = Beta::new(5.0, 5.0)?;
        // let misinfo_trust_dist = Beta::new(1.5, 4.5)?;

        let infos = [Info::new(
            0,
            InfoContent::new(
                Opinion1d::<f32, PSI>::new([0.01, 0.98], 0.01, br_psi),
                Opinion1d::<f32, PSI>::new([0.0, 0.0], 1.0, br_ppsi),
                Opinion1d::<f32, A>::new([0.0, 0.0], 1.0, br_pa),
                Opinion1d::<f32, PHI>::new([0.0, 0.0], 1.0, br_phi),
                [
                    Simplex::<f32, THETA>::vacuous(),
                    Simplex::<f32, THETA>::vacuous(),
                ],
            ),
        )];

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
            FriendOpinion::new(br_fpa, br_fptheta, br_ftheta),
            CPT::new(0.88, 0.88, 2.25, 0.61, 0.69),
            [
                LevelSet::<_, f32>::new(&selfish_outcome_maps[0]),
                LevelSet::<_, f32>::new(&selfish_outcome_maps[1]),
            ],
            [
                LevelSet::<_, f32>::new(sharing_outcome_maps[0].deref()),
                LevelSet::<_, f32>::new(sharing_outcome_maps[1].deref()),
            ],
            br_pa,
            br_ptheta,
            br_fa,
        );

        a.reset(0.5, 0.5, 0.5, vec![0.90]);
        a.op.reset(&br_theta, &br_psi, &br_phi, &br_ppsi);
        a.fop.reset(&br_fppsi, &br_fpsi, &br_fphi);

        let receipt_prob = 0.0;
        println!("{:?}", a.read_info_trustfully(&infos[0], receipt_prob));
    }
}
