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
use arrow2::array::{PrimitiveArray, Utf8Array};
use arrow2::chunk::Chunk;
use arrow2::datatypes::{DataType, Field, Schema};
use arrow2::io::ipc::write::{Compression, FileWriter, WriteOptions};
use config::{Config, ConfigFormat, EventTable};
use log::debug;
use num_traits::{Float, FromPrimitive, NumAssign};
use rand::Rng;
use rand::{rngs::SmallRng, seq::SliceRandom, SeedableRng};

use agent::{Agent, AgentParams};
use info::{Info, InfoContent, InfoLabel};

use graph_lib::prelude::{Graph, GraphB};
use rand_distr::uniform::SampleUniform;
use rand_distr::{Distribution, Open01, Standard};

#[derive(Debug)]
struct Receipt {
    agent_idx: usize,
    force: bool,
    info_idx: usize,
}

pub struct Executor<V>
where
    V: Float + UlpsEq + NumAssign + SampleUniform,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
{
    config: Config<V>,
    agents: Vec<Agent<V>>,
    stat: Stat,
}

impl<V> Executor<V>
where
    V: Float + NumAssign + UlpsEq + Default + Sum + std::fmt::Debug + SampleUniform,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
{
    pub fn try_new(config_data: ConfigFormat) -> Result<Self, Box<dyn std::error::Error>>
    where
        for<'de> V: serde::Deserialize<'de>,
    {
        let config: Config<V> = config_data.try_into()?;
        let mut agents = Vec::with_capacity(config.runtime.graph.node_count());
        for _ in 0..agents.capacity() {
            agents.push(Agent::new());
        }

        Ok(Self {
            config,
            stat: Stat::default(),
            agents,
        })
    }

    pub fn exec(&mut self) -> Result<(), Box<dyn std::error::Error>>
    where
        V: FromPrimitive,
    {
        let writer = File::create(self.config.output.location.join(format!(
            "{}_{}.arrow",
            self.config.name, self.config.output.suffix
        )))?;
        let mut rng = SmallRng::seed_from_u64(self.config.runtime.seed_state);
        let n = V::from_usize(self.config.runtime.graph.node_count()).unwrap();
        let d = V::from_usize(self.config.runtime.graph.edge_count()).unwrap() / n; // average outdegree

        println!("started.");
        for num_iter in 0..self.config.runtime.iteration_count {
            for a in &mut self.agents {
                a.reset_with(&self.config.scenario.agent_params, &mut rng);
            }
            Self::run_loop(
                self.config.scenario.event_table.clone(),
                &self.config.scenario.info_contents,
                &mut self.agents,
                &mut self.stat,
                &self.config.runtime.graph,
                &self.config.scenario.agent_params,
                num_iter,
                n,
                d,
                &mut rng,
            );
        }
        println!("finished.");
        self.stat.write(
            writer,
            if self.config.output.compress {
                Some(Compression::ZSTD)
            } else {
                None
            },
        )?;
        println!("done.");
        Ok(())
    }

    fn run_loop<R>(
        mut event_table: EventTable,
        info_contents: &[InfoContent<V>],
        agents: &mut Vec<Agent<V>>,
        stat: &mut Stat,
        graph: &GraphB,
        agent_params: &AgentParams<V>,
        num_iter: u32,
        n: V,
        d: V,
        rng: &mut R,
    ) where
        V: FromPrimitive,
        R: Rng,
    {
        let mut infos = Vec::<Info<V>>::new();
        let mut received = Vec::new();
        let mut t = 0;
        let mut num_view_map = BTreeMap::<InfoLabel, u32>::new();
        let mut num_sharing_map = BTreeMap::<InfoLabel, u32>::new();
        let mut num_receipt_map = BTreeMap::<InfoLabel, u32>::new();
        while !received.is_empty() || !event_table.0.is_empty() {
            if let Some(informms) = event_table.0.remove(&t) {
                for (agent_idx, info_content_idx) in informms {
                    let info_idx = infos.len();
                    infos.push(Info::new(info_idx, &info_contents[info_content_idx]));
                    received.push(Receipt {
                        agent_idx,
                        force: true,
                        info_idx,
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
                    let info = &mut infos[r.info_idx];
                    let receipt_prob = V::one()
                        - (V::one() - V::from_usize(info.num_shared()).unwrap() / n).powf(d);

                    let num_receipt = num_receipt_map.entry(info.content.label).or_insert(0);
                    *num_receipt += 1;

                    let b = if r.force {
                        agent.read_info_trustfully(info, receipt_prob, &agent_params)
                    } else {
                        agent.read_info(info, receipt_prob, &agent_params, rng)?
                    };

                    debug!(
                        "{}: {}",
                        if r.force { "informer" } else { "sharer" },
                        info.content.label
                    );

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
                        graph
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
        let metadata =
            BTreeMap::from_iter([("version".to_string(), env!("CARGO_PKG_VERSION").to_string())]);
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

#[cfg(test)]
mod tests {
    use crate::{
        agent::{Agent, AgentParams},
        cpt::CptParams,
        info::{Info, InfoContent, InfoObject, TrustDists},
        opinion::{GlobalBaseRates, InitialOpinions},
        Config,
    };

    use either::Either::Left;
    use serde_json::json;
    use subjective_logic::{harr2, harr3, mul::Simplex};

    #[test]
    fn test_agent() {
        let agent_params = AgentParams {
            initial_opinions: InitialOpinions {
                theta: Simplex::vacuous(),
                psi: Simplex::vacuous(),
                phi: Simplex::vacuous(),
                s: Simplex::vacuous(),
                cond_theta_phi: [Simplex::vacuous(), Simplex::vacuous()],
                fs: Simplex::vacuous(),
                fphi: Simplex::vacuous(),
                cond_ftheta_fphi: [Simplex::vacuous(), Simplex::vacuous()],
                cond_pa: [
                    Simplex::new([0.90, 0.00], 0.10),
                    Simplex::new([0.00, 0.99], 0.01),
                    Simplex::new([0.90, 0.00], 0.10),
                ],
                cond_theta: harr3![
                    [
                        [
                            Simplex::new([0.95, 0.00, 0.00], 0.05),
                            Simplex::new([0.95, 0.00, 0.00], 0.05),
                        ],
                        [
                            Simplex::new([0.00, 0.45, 0.45], 0.10),
                            Simplex::new([0.00, 0.45, 0.45], 0.10),
                        ],
                    ],
                    [
                        [
                            Simplex::new([0.00, 0.475, 0.475], 0.05),
                            Simplex::new([0.00, 0.475, 0.475], 0.05),
                        ],
                        [
                            Simplex::new([0.00, 0.495, 0.495], 0.01),
                            Simplex::new([0.00, 0.495, 0.495], 0.01),
                        ],
                    ]
                ],
                cond_ptheta: [
                    Simplex::new([0.99, 0.00, 0.00], 0.01),
                    Simplex::new([0.00, 0.495, 0.495], 0.01),
                ],
                cond_ppsi: [
                    Simplex::new([0.99, 0.00], 0.01),
                    Simplex::new([0.25, 0.65], 0.10),
                ],
                cond_fpsi: [
                    Simplex::new([0.99, 0.00], 0.01),
                    Simplex::new([0.70, 0.20], 0.10),
                ],
                cond_fa: [
                    Simplex::new([0.95, 0.00], 0.05),
                    Simplex::new([0.00, 0.95], 0.05),
                    Simplex::new([0.95, 0.00], 0.05),
                ],
                cond_fpa: [
                    Simplex::new([0.90, 0.00], 0.10),
                    Simplex::new([0.00, 0.99], 0.01),
                    Simplex::new([0.90, 0.00], 0.10),
                ],
                cond_ftheta: harr2![
                    [
                        Simplex::new([0.95, 0.00, 0.00], 0.05),
                        Simplex::new([0.00, 0.45, 0.45], 0.10),
                    ],
                    [
                        Simplex::new([0.00, 0.475, 0.475], 0.05),
                        Simplex::new([0.00, 0.495, 0.495], 0.01),
                    ]
                ],
                cond_fptheta: [
                    Simplex::new([0.99, 0.000, 0.000], 0.01),
                    Simplex::new([0.00, 0.495, 0.495], 0.01),
                ],
                cond_fppsi: [
                    Simplex::new([0.99, 0.00], 0.01),
                    Simplex::new([0.25, 0.65], 0.10),
                ],
            },
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
            read_dist: Left(0.5),
            farrival_dist: Left(0.5),
            fread_dist: Left(0.5),
            pi_dist: Left(0.5),
            pi_prob: 0.5,
            trust_dists: TrustDists {
                misinfo: Left(0.5),
                corrective: Left(0.5),
                observed: Left(0.5),
                inhibitive: Left(0.5),
            },
            cpt_params: CptParams {
                x0_dist: Left(0.5),
                x1_dist: Left(0.5),
                y_dist: Left(0.5),
                alpha: Left(0.88),
                beta: Left(0.88),
                lambda: Left(2.25),
                gamma: Left(0.61),
                delta: Left(0.69),
            },
        };

        let info_contents = [InfoContent::<f32>::from(InfoObject::Misinfo {
            psi: Simplex::new([0.00, 0.99], 0.01),
        })];
        let info = Info::new(0, &info_contents[0]);

        let mut a = Agent::new();
        a.prospect.reset(-0.1, -2.0, -0.001);
        a.cpt.reset(0.88, 0.88, 2.25, 0.61, 0.69);
        a.reset(
            0.5,
            0.5,
            0.5,
            0.0,
            agent_params.initial_opinions.clone(),
            &agent_params.base_rates,
        );

        let receipt_prob = 0.0;
        println!(
            "{:?}",
            a.read_info_trustfully(&info, receipt_prob, &agent_params)
        );
    }

    #[test]
    fn test_json_config() {
        let v = json!(
            {
                "name": "test-senario",
                "output": {
                    "location": "./test/",
                },
                "runtime": {
                    "graph": {
                        "directed": true,
                        "location": {
                            "LocalFile": "./test/graph.txt",
                        },
                    },
                    "seed_state": 0,
                    "iteration_count": 1,
                },
                "scenario": {
                    "agent_params": {
                        "initial_opinions": {
                            "theta": [[0.0, 0.0, 0.0], 1.0],
                            "psi":   [[0.0, 0.0], 1.0],
                            "phi":   [[0.0, 0.0], 1.0],
                            "s":     [[0.0, 0.0], 1.0],
                            "cond_theta_phi": [[[0.0, 0.0, 0.0], 1.0], [[0.0, 0.0, 0.0], 1.0]],
                            "fs": [[0.0, 0.0], 1.0],
                            "fphi": [[0.0, 0.0], 1.0],
                            "cond_ftheta_fphi": [[[0.0, 0.0, 0.0], 1.0], [[0.0, 0.0, 0.0], 1.0]],
                            "cond_pa": [
                                [[0.90, 0.00], 0.10],
                                [[0.00, 0.99], 0.01],
                                [[0.90, 0.00], 0.10],
                            ],
                            "cond_theta": [
                                [
                                    [
                                        [[0.95, 0.00, 0.00], 0.05],
                                        [[0.95, 0.00, 0.00], 0.05],
                                    ],
                                    [
                                        [[0.00, 0.45, 0.45], 0.10],
                                        [[0.00, 0.45, 0.45], 0.10],
                                    ],
                                ],
                                [
                                    [
                                        [[0.00, 0.475, 0.475], 0.05],
                                        [[0.00, 0.475, 0.475], 0.05],
                                    ],
                                    [
                                        [[0.00, 0.495, 0.495], 0.01],
                                        [[0.00, 0.495, 0.495], 0.01],
                                    ],
                                ]
                            ],
                            "cond_ptheta": [
                                [[0.99, 0.00, 0.00], 0.01],
                                [[0.00, 0.495, 0.495], 0.01],
                            ],
                            "cond_ppsi": [
                                [[0.99, 0.00], 0.01],
                                [[0.25, 0.65], 0.10],
                            ],
                            "cond_fpsi": [
                                [[0.99, 0.00], 0.01],
                                [[0.70, 0.20], 0.10],
                            ],
                            "cond_fa": [
                                [[0.95, 0.00], 0.05],
                                [[0.00, 0.95], 0.05],
                                [[0.95, 0.00], 0.05],
                            ],
                            "cond_fpa": [
                                [[0.90, 0.00], 0.10],
                                [[0.00, 0.99], 0.01],
                                [[0.90, 0.00], 0.10],
                            ],
                            "cond_ftheta": [
                                [
                                    [[0.95, 0.00, 0.00], 0.05],
                                    [[0.00, 0.45, 0.45], 0.10],
                                ],
                                [
                                    [[0.00, 0.475, 0.475], 0.05],
                                    [[0.00, 0.495, 0.495], 0.01],
                                ]
                            ],
                            "cond_fptheta": [
                                [[0.99, 0.000, 0.000], 0.01],
                                [[0.00, 0.495, 0.495], 0.01],
                            ],
                            "cond_fppsi": [
                                [[0.99, 0.00], 0.01],
                                [[0.25, 0.65], 0.10],
                            ],
                        },
                        "base_rates": {
                            "s": [0.999, 0.001],
                            "fs": [0.99, 0.01],
                            "psi": [0.999, 0.001],
                            "ppsi": [0.999, 0.001],
                            "pa": [0.999, 0.001],
                            "fa": [0.999, 0.001],
                            "fpa": [0.999, 0.001],
                            "phi": [0.999, 0.001],
                            "fpsi": [0.999, 0.001],
                            "fppsi": [0.999, 0.001],
                            "fphi": [0.999, 0.001],
                            "theta": [0.999, 0.0005, 0.0005],
                            "ptheta": [0.999, 0.0005, 0.0005],
                            "ftheta": [0.999, 0.0005, 0.0005],
                            "fptheta": [0.999, 0.0005, 0.0005],
                        },
                        "pi_prob": 1.0,
                        "read_dist": { "Beta": { "alpha": 3.0, "beta": 3.0 } },
                        "farrival_dist": { "Beta": {"alpha": 3.0, "beta": 3.0}},
                        "fread_dist": {"Beta": {"alpha": 3.0, "beta": 3.0}},
                        "pi_dist": {"Fixed": {"value": 0.5}},
                        "trust_dists": {
                            "misinfo": {"Beta": { "alpha": 3.0, "beta": 3.0 }},
                            "corrective": {"Beta":{ "alpha": 3.0, "beta": 3.0 }},
                            "observed": "Standard",
                            "inhibitive": {"Uniform": { "low": 0.5, "high": 1.0 }},
                        },
                        "cpt_params": {
                            "x0_dist": {"Fixed": {"value": -2.0}},
                            "x1_dist": {"Fixed": {"value": -50.0}},
                            "y_dist": {"Fixed": {"value": -0.001}},
                            "alpha":  { "Fixed": { "value": 0.88 }},
                            "beta":   { "Fixed": { "value": 0.88 }},
                            "lambda": { "Fixed": { "value": 2.25 }},
                            "gamma":  { "Fixed": { "value": 0.61 }},
                            "delta":  { "Fixed": { "value": 0.69 }},
                        }
                    },
                    "info_contents": [
                        { "Misinfo": { "psi": ([0.00, 0.99], 0.01) } },
                        { "Corrective": { "psi": ([0.99, 0.00], 0.01), "s": ([0.0, 1.0], 0.0) } }
                    ],
                    "event_table": [
                        {
                            "time": 0,
                            "informs": [
                                { "agent_idx": 0, "info_content_idx": 0, },
                                { "agent_idx": 1, "info_content_idx": 0, },
                            ],
                        },
                        {
                            "time": 1,
                            "informs": [
                                { "agent_idx": 2, "info_content_idx": 1, },
                            ]
                        },
                    ],
                }
            }
        );
        let r = serde_json::from_value::<Config<f32>>(v);
        assert!(r.is_ok());
        let c = r.unwrap();
        println!("{:?}", c.output);
        println!("{:?}", c.runtime.graph);
        println!("{:?}", c.scenario.event_table);
        println!("{:?}", c.scenario.info_contents);
    }

    #[test]
    fn test_toml_config() {
        let r = toml::from_str::<Config<f32>>(
            r#"
            name = "test-senario"

            [runtime]
            graph = { directed = false, location = { LocalFile = "./test/graph.txt" } }
            seed_state = 0
            iteration_count = 1

            [scenario.agent_params.initial_opinions]
            theta = [[0.0, 0.0, 0.0], 1.0]
            psi =   [[0.0, 0.0], 1.0]
            phi =   [[0.0, 0.0], 1.0]
            s =     [[0.0, 0.0], 1.0]
            cond_theta_phi = [[[0.0, 0.0, 0.0], 1.0], [[0.0, 0.0, 0.0], 1.0]]
            fs = [[0.0, 0.0], 1.0]
            fphi = [[0.0, 0.0], 1.0]
            cond_ftheta_fphi = [[[0.0, 0.0, 0.0], 1.0], [[0.0, 0.0, 0.0], 1.0]]
            cond_pa = [
                [[0.90, 0.00], 0.10],
                [[0.00, 0.99], 0.01],
                [[0.90, 0.00], 0.10],
            ]
            cond_theta = [
                [
                    [
                        [[0.95, 0.00, 0.00], 0.05],
                        [[0.95, 0.00, 0.00], 0.05],
                    ],
                    [
                        [[0.00, 0.45, 0.45], 0.10],
                        [[0.00, 0.45, 0.45], 0.10],
                    ],
                ],
                [
                    [
                        [[0.00, 0.475, 0.475], 0.05],
                        [[0.00, 0.475, 0.475], 0.05],
                    ],
                    [
                        [[0.00, 0.495, 0.495], 0.01],
                        [[0.00, 0.495, 0.495], 0.01],
                    ],
                ]
            ]
            cond_ptheta = [
                [[0.99, 0.00, 0.00], 0.01],
                [[0.00, 0.495, 0.495], 0.01],
            ]
            cond_ppsi = [
                [[0.99, 0.00], 0.01],
                [[0.25, 0.65], 0.10],
            ]
            cond_fpsi = [
                [[0.99, 0.00], 0.01],
                [[0.70, 0.20], 0.10],
            ]
            cond_fa = [
                [[0.95, 0.00], 0.05],
                [[0.00, 0.95], 0.05],
                [[0.95, 0.00], 0.05],
            ]
            cond_fpa = [
                [[0.90, 0.00], 0.10],
                [[0.00, 0.99], 0.01],
                [[0.90, 0.00], 0.10],
            ]
            cond_ftheta = [
                [
                    [[0.95, 0.00, 0.00], 0.05],
                    [[0.00, 0.45, 0.45], 0.10],
                ],
                [
                    [[0.00, 0.475, 0.475], 0.05],
                    [[0.00, 0.495, 0.495], 0.01],
                ]
            ]
            cond_fptheta = [
                [[0.99, 0.000, 0.000], 0.01],
                [[0.00, 0.495, 0.495], 0.01],
            ]
            cond_fppsi = [
                [[0.99, 0.00], 0.01],
                [[0.25, 0.65], 0.10],
            ]

            [scenario.agent_params.base_rates]
            s = [0.999, 0.001]
            fs = [0.99, 0.01]
            psi = [0.999, 0.001]
            ppsi = [0.999, 0.001]
            pa = [0.999, 0.001]
            fa = [0.999, 0.001]
            fpa = [0.999, 0.001]
            phi = [0.999, 0.001]
            fpsi = [0.999, 0.001]
            fppsi = [0.999, 0.001]
            fphi = [0.999, 0.001]
            theta = [0.999, 0.0005, 0.0005]
            ptheta = [0.999, 0.0005, 0.0005]
            ftheta = [0.999, 0.0005, 0.0005]
            fptheta = [0.999, 0.0005, 0.0005]

            [scenario.agent_params]
            pi_prob = 1.0
            read_dist = { Beta = { alpha = 3.0, beta = 3.0 } }
            farrival_dist = { Beta = { alpha = 3.0, beta = 3.0 } }
            fread_dist = { Beta = { alpha = 3.0, beta = 3.0 } }
            pi_dist = { Fixed = { value = 0.5 }}

            [scenario.agent_params.trust_dists]
            misinfo = { Beta = { alpha = 3.0, beta = 3.0 } }
            corrective = { Beta = { alpha = 3.0, beta = 3.0 } }
            observed =  "Standard"
            inhibitive = { Uniform = { low = 0.5, high = 1.0 } }

            [scenario.agent_params.cpt_params]
            x0_dist = { Fixed = { value = -2.0 } }
            x1_dist = { Fixed = { value = -50.0 } }
            y_dist = { Fixed = { value = -0.001 } }
            alpha =  { Fixed = { value = 0.88 } }
            beta =   { Fixed = { value = 0.88 } }
            lambda = { Fixed = { value = 2.25 } }
            gamma =  { Fixed = { value = 0.61 } }
            delta =  { Fixed = { value = 0.69 } }

            [scenario]
            info_contents = [
                { Misinfo = { psi = [[0.00, 0.99], 0.01] } },
                { Corrective = { psi = [[0.99, 0.00], 0.01], s = [[0.0, 1.0], 0.0] } }
            ]

            [[scenario.event_table]]
            time = 0
            informs = [
                { agent_idx = 0, info_content_idx = 0 },
                { agent_idx = 1, info_content_idx = 0 },
            ]

            [[scenario.event_table]]
            time = 1
            informs = [{ agent_idx = 2, info_content_idx = 1 }]
            "#,
        );

        assert!(r.is_ok());
        let c = r.unwrap();
        println!("{:?}", c.output);
        println!("{:?}", c.runtime.graph);
        println!("{:?}", c.scenario.agent_params.initial_opinions);
        println!(
            "{:?}",
            c.scenario
                .info_contents
                .into_iter()
                .map(|i| i.label)
                .collect::<Vec<_>>()
        );
        println!("{:?}", c.scenario.event_table);
    }
}
