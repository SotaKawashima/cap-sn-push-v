use approx::UlpsEq;
use graph_lib::io::{DataFormat, ParseBuilder};
use graph_lib::prelude::{DiGraphB, Graph, GraphB, UndiGraphB};
use num_traits::{Float, FromPrimitive, NumAssign};
use rand_distr::uniform::SampleUniform;
use rand_distr::{Distribution, Exp1, Open01, Standard, StandardNormal};
use serde::Deserialize;
use serde_with::{serde_as, TryFromInto};
use std::collections::{BTreeMap, VecDeque};
use std::{fs::File, io, ops::AddAssign};
use subjective_logic::mul::labeled::SimplexD1;

use crate::info::InfoObject;
use crate::opinion::O;

#[derive(Debug, serde::Deserialize)]
struct GraphInfo {
    directed: bool,
    location: DataLocation,
}

#[derive(Debug, serde::Deserialize)]
enum DataLocation {
    LocalFile(String),
}

impl TryFrom<GraphInfo> for GraphB {
    type Error = io::Error;

    fn try_from(value: GraphInfo) -> Result<Self, Self::Error> {
        match value.location {
            DataLocation::LocalFile(path) => {
                let builder = ParseBuilder::new(File::open(path)?, DataFormat::EdgeList);
                if value.directed {
                    Ok(GraphB::Di(builder.parse::<DiGraphB>()?))
                } else {
                    Ok(GraphB::Ud(builder.parse::<UndiGraphB>()?))
                }
            }
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct ScenarioParam<V>
where
    V: Float + UlpsEq + AddAssign,
{
    graph: GraphInfo,
    info_objects: Vec<InfoObject<V>>,
    events: Vec<Event>,
    observer: Option<ObserverParam<V>>,
}

#[derive(Debug)]
pub struct Scenario<V>
where
    V: Float + UlpsEq + NumAssign + SampleUniform,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
{
    pub graph: GraphB,
    pub info_objects: Vec<InfoObject<V>>,
    /// time -> (agent_idx, info_obj_idx)
    pub table: BTreeMap<u32, VecDeque<(usize, usize)>>,
    pub observer: Option<Observer<V>>,
    pub fnum_nodes: V,
    pub mean_degree: V,
    pub num_nodes: usize,
}

#[derive(Deserialize, Debug)]
struct Event {
    time: u32,
    informs: Vec<Inform>,
}

#[derive(Deserialize, Debug)]
pub struct Inform {
    agent_idx: usize,
    info_obj_idx: usize,
}

#[serde_as]
#[derive(Debug, serde::Deserialize)]
struct ObserverParam<V: Float + UlpsEq + AddAssign> {
    observer_pop_rate: V,
    #[serde_as(as = "TryFromInto<(Vec<V>, V)>")]
    observed_info: SimplexD1<O, V>,
}

#[derive(Debug)]
pub struct Observer<V> {
    // pub observer_pop_rate: V,
    pub observed_info_obj_idx: usize,
    /// number of times to try to send observed info in a step
    pub k: V,
}

impl<V> TryFrom<ScenarioParam<V>> for Scenario<V>
where
    V: Float + UlpsEq + NumAssign + SampleUniform + FromPrimitive,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
{
    type Error = anyhow::Error;

    fn try_from(value: ScenarioParam<V>) -> anyhow::Result<Scenario<V>> {
        let ScenarioParam {
            graph,
            mut info_objects,
            events,
            observer,
        } = value;

        let graph = GraphB::try_from(graph)?;
        let mut table = BTreeMap::default();
        for Event { time, informs } in events {
            table.insert(
                time,
                informs
                    .into_iter()
                    .map(
                        |Inform {
                             agent_idx,
                             info_obj_idx: info_object_idx,
                         }| {
                            // let idx = match info {
                            //     InfoParam::Idx(idx) => idx,
                            //     InfoParam::Obj(obj) => {
                            //         info_objects.push(obj);
                            //         info_objects.len() - 1
                            //     }
                            // };
                            (agent_idx, info_object_idx)
                        },
                    )
                    .collect(),
            );
        }

        let num_nodes = graph.node_count();
        let fnum_nodes = V::from_usize(num_nodes).unwrap();
        let mean_degree = V::from_usize(graph.edge_count()).unwrap() / fnum_nodes;

        let observer = observer.map(
            |ObserverParam {
                 observer_pop_rate,
                 observed_info,
             }| {
                let k = fnum_nodes * observer_pop_rate;
                info_objects.push(InfoObject::Observed { o: observed_info });
                Observer {
                    // observer_pop_rate,
                    observed_info_obj_idx: info_objects.len() - 1,
                    k,
                }
            },
        );
        Ok(Self {
            graph,
            info_objects,
            table,
            num_nodes,
            fnum_nodes,
            mean_degree,
            observer,
        })
    }
}
