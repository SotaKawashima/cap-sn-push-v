mod agent;
mod cpt;
mod dist;
mod info;
mod opinion;

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::iter::Sum;

use approx::UlpsEq;
use arrow2::array::{PrimitiveArray, Utf8Array};
use arrow2::chunk::Chunk;
use arrow2::datatypes::{DataType, Field, Schema};
use arrow2::io::ipc::write::{Compression, FileWriter, WriteOptions};
use dist::DistParam;
use log::debug;
use num_traits::{Float, FromPrimitive, NumAssign};
use rand::Rng;
use rand::{rngs::SmallRng, seq::SliceRandom, SeedableRng};
use rand_distr::uniform::SampleUniform;
use rand_distr::{Distribution, Open01, Standard};

use agent::{Agent, Constants};
use cpt::CptParams;
use info::{Info, InfoType, ToInfoContent, TrustDists};
use opinion::{
    FriendOpinions, FriendStaticOpinions, GlobalBaseRates, Opinions, PluralIgnorance,
    StaticOpinions,
};

use graph_lib::io::{DataFormat, ParseBuilder};
use graph_lib::prelude::{DiGraphB, Graph};

#[derive(serde::Deserialize, Debug)]
struct Scenario<V: Float> {
    events: Vec<Event>,
    base_rates: GlobalBaseRates<V>,
    /// probability whether people has plural ignorance
    pi_prob: V,
    read_dist: DistParam<V>,
    farrival_dist: DistParam<V>,
    fread_dist: DistParam<V>,
    pi_dist: DistParam<V>,
    misinfo_trust: DistParam<V>,
    corrective_trust: DistParam<V>,
    observed_trust: DistParam<V>,
    inhibitive_trust: DistParam<V>,
    x0_dist: DistParam<V>,
    x1_dist: DistParam<V>,
    y_dist: DistParam<V>,
    alpha: DistParam<V>,
    beta: DistParam<V>,
    lambda: DistParam<V>,
    gamma: DistParam<V>,
    delta: DistParam<V>,
}

/// events: time -> agent_id -> info_id
fn extract_events<V>(events: Vec<Event>) -> (Vec<Info<V>>, BTreeMap<u32, BTreeMap<usize, usize>>)
where
    V: Float + UlpsEq,
    for<'a> &'a InfoType: ToInfoContent<V>,
{
    let mut infos = Vec::new();
    let mut info_id = 0;
    let scenario = events
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
                            let info = Info::new(info_id, info_type, info_type.into());
                            info_id += 1;
                            infos.push(info);
                            (agent_idx, info_id)
                        },
                    )
                    .collect(),
            )
        })
        .collect();

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
    V: Float + SampleUniform,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
{
    events: BTreeMap<u32, BTreeMap<usize, usize>>,
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
                scenario: &str,
                seed_state: u64,
            ) -> Result<Self, Box<dyn std::error::Error>> {
                let scenario: Scenario<$ft> = serde_json::from_str(scenario)?;
                let (infos, events) = extract_events(scenario.events);

                let constants = Constants::<$ft> {
                    base_rates: scenario.base_rates,
                    read_dist: scenario.read_dist.try_into()?,
                    farrival_dist: scenario.farrival_dist.try_into()?,
                    fread_dist: scenario.fread_dist.try_into()?,
                    pi_dist: scenario.pi_dist.try_into()?,
                    pi_prob: scenario.pi_prob,
                    trust_dists: TrustDists {
                        misinfo: scenario.misinfo_trust.try_into()?,
                        corrective: scenario.corrective_trust.try_into()?,
                        observed: scenario.observed_trust.try_into()?,
                        inhibitive: scenario.inhibitive_trust.try_into()?,
                    },
                    cpt_params: CptParams {
                        x0_dist: scenario.x0_dist.try_into()?,
                        x1_dist: scenario.x1_dist.try_into()?,
                        y_dist: scenario.y_dist.try_into()?,
                        alpha: scenario.alpha.try_into()?,
                        beta: scenario.beta.try_into()?,
                        lambda: scenario.lambda.try_into()?,
                        gamma: scenario.gamma.try_into()?,
                        delta: scenario.delta.try_into()?,
                    },
                };

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
                        infos.len(),
                    ));
                }

                Ok(Self {
                    events,
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
        + SampleUniform
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
        let mut events = self.events.clone();
        let mut t = 0;
        let mut num_view_map = BTreeMap::<InfoType, u32>::new();
        let mut num_sharing_map = BTreeMap::<InfoType, u32>::new();
        let mut num_receipt_map = BTreeMap::<InfoType, u32>::new();
        while !received.is_empty() || !events.is_empty() {
            if let Some(informers) = events.remove(&t) {
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
        cpt::CptParams,
        extract_events,
        info::TrustDists,
        info::{Info, InfoType},
        opinion::{
            FriendOpinions, FriendStaticOpinions, GlobalBaseRates, Opinions, StaticOpinions,
        },
        Scenario,
    };

    use either::Either::Left;
    use serde_json::json;

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

        let info_types = [InfoType::Misinfo];
        let infos = info_types.map(|t| Info::new(0, t, t.into()));

        let mut a = Agent::new(
            Opinions::new(),
            StaticOpinions::<f32>::new(),
            FriendOpinions::new(),
            FriendStaticOpinions::<f32>::new(),
            1,
        );

        a.prospect.reset(-0.1, -2.0, -0.001);
        a.cpt.reset(0.88, 0.88, 2.25, 0.61, 0.69);
        a.reset(0.5, 0.5, 0.5, 0.0, |_| 0.90, &constants.base_rates, &infos);

        let receipt_prob = 0.0;
        println!(
            "{:?}",
            a.read_info_trustfully(&infos[0], receipt_prob, &constants.base_rates)
        );
    }

    #[test]
    fn test_scenario() {
        let v = json!(
            {
                "events": [
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
                ],
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
                "misinfo_trust": {"Beta": { "alpha": 3.0, "beta": 3.0 }},
                "corrective_trust": {"Beta":{ "alpha": 3.0, "beta": 3.0 }},
                "observed_trust": "Standard",
                "inhibitive_trust": {"Uniform": { "low": 0.5, "high": 1.0 }},
                "x0_dist": {"Fixed": {"value": -2.0}},
                "x1_dist": {"Fixed": {"value": -50.0}},
                "y_dist": {"Fixed": {"value": -0.001}},
                "alpha":  { "Fixed": { "value": 0.88 }},
                "beta":   { "Fixed": { "value": 0.88 }},
                "lambda": { "Fixed": { "value": 2.25 }},
                "gamma":  { "Fixed": { "value": 0.61 }},
                "delta":  { "Fixed": { "value": 0.69 }},
            }
        );
        let s: Scenario<f32> = serde_json::from_value(v).unwrap();
        println!("{:?}", s);
        let (is, sc) = extract_events::<f32>(s.events);
        println!("{:?}", is);
        println!("{:?}", sc);
    }
}
