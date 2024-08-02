use std::{
    collections::{btree_map, BTreeMap, VecDeque},
    fs::File,
    io,
};

use graph_lib::{
    io::{DataFormat, ParseBuilder},
    prelude::{DiGraphB, Graph, GraphB, UndiGraphB},
};
use itertools::Itertools;
use rand_distr::{Distribution, Exp1, Open01, Standard, StandardNormal};
use serde::Deserialize;
use serde_with::{serde_as, TryFromInto};
use subjective_logic::{
    mul::labeled::{OpinionD1, SimplexD1},
    multi_array::labeled::MArrD1,
};

use base::opinion::{MyFloat, Phi, B, H, O};
use base::{info::InfoContent, opinion::Psi};

#[derive(Debug, serde::Deserialize)]
struct GraphInfo {
    directed: bool,
    location: DataLocation,
    #[serde(default = "default_transposed")]
    transposed: bool,
}

#[derive(Debug, serde::Deserialize)]
enum DataLocation {
    LocalFile(String),
}

fn default_transposed() -> bool {
    false
}

impl TryFrom<GraphInfo> for GraphB {
    type Error = io::Error;

    fn try_from(value: GraphInfo) -> Result<Self, Self::Error> {
        match value.location {
            DataLocation::LocalFile(path) => {
                let mut builder = ParseBuilder::new(File::open(path)?, DataFormat::EdgeList);
                if value.directed {
                    if value.transposed {
                        builder = builder.transpose();
                    }
                    Ok(GraphB::Di(builder.parse::<DiGraphB>()?))
                } else {
                    Ok(GraphB::Ud(builder.parse::<UndiGraphB>()?))
                }
            }
        }
    }
}

#[serde_as]
#[derive(Debug, serde::Deserialize, Clone)]
#[serde(bound(deserialize = "V: Deserialize<'de>"))]
pub enum InfoObject<V>
where
    V: MyFloat,
{
    Misinfo {
        #[serde_as(as = "TryFromInto<(Vec<V>, V, Vec<V>)>")]
        op: OpinionD1<Psi, V>,
    },
    Correction {
        #[serde_as(as = "TryFromInto<(Vec<V>, V, Vec<V>)>")]
        op: OpinionD1<Psi, V>,
        #[serde_as(as = "TryFromInto<(Vec<V>, V, Vec<V>)>")]
        misinfo: OpinionD1<Psi, V>,
    },
    Observation {
        #[serde_as(as = "TryFromInto<(Vec<V>, V, Vec<V>)>")]
        op: OpinionD1<O, V>,
    },
    Inhibition {
        #[serde_as(as = "TryFromInto<(Vec<V>, V, Vec<V>)>")]
        op1: OpinionD1<Phi, V>,
        #[serde_as(as = "TryFromInto<Vec<(Vec<V>, V)>>")]
        op2: MArrD1<Psi, SimplexD1<H, V>>,
        #[serde_as(as = "TryFromInto<Vec<(Vec<V>, V)>>")]
        op3: MArrD1<B, SimplexD1<H, V>>,
    },
}

impl<V: MyFloat> From<InfoObject<V>> for InfoContent<V> {
    fn from(value: InfoObject<V>) -> Self {
        match value {
            InfoObject::Misinfo { op } => Self::Misinfo { op },
            InfoObject::Correction { op, misinfo } => Self::Correction { op, misinfo },
            InfoObject::Observation { op } => Self::Observation { op },
            InfoObject::Inhibition { op1, op2, op3 } => Self::Inhibition { op1, op2, op3 },
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct ScenarioParam<V>
where
    V: MyFloat,
{
    graph: GraphInfo,
    infos: Vec<InfoObject<V>>,
    events: Vec<Event>,
    observer: Option<ObserverParam<V>>,
}

#[derive(Debug)]
pub struct Scenario<V>
where
    V: MyFloat,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
{
    pub graph: GraphB,
    pub info_contents: Vec<InfoContent<V>>,
    /// time -> Inform
    pub table: BTreeMap<u32, VecDeque<Inform>>,
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

#[derive(Deserialize, Debug, Clone)]
pub struct Inform {
    pub agent_idx: usize,
    pub info_obj_idx: usize,
}

#[serde_as]
#[derive(Debug, serde::Deserialize)]
struct ObserverParam<V: MyFloat> {
    observe_prob: V,
    post_prob: V,
    threashold_rate: V,
    #[serde_as(as = "TryFromInto<(Vec<V>, V, Vec<V>)>")]
    observed_info: OpinionD1<O, V>,
}

#[derive(Debug)]
pub struct Observer<V> {
    pub observed_info_obj_idx: usize,
    /// probability to observe info in a step (no duplication)
    pub po: V,
    // probability to post observation info if obsereved events.
    pub pp: V,
    // max number of selfish actions not to happen obserevable events.
    pub threshold: usize,
}

impl<V> TryFrom<ScenarioParam<V>> for Scenario<V>
where
    V: MyFloat,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
{
    type Error = anyhow::Error;

    fn try_from(value: ScenarioParam<V>) -> anyhow::Result<Scenario<V>> {
        let ScenarioParam {
            graph,
            infos,
            events,
            observer,
        } = value;

        let graph = GraphB::try_from(graph)?;
        let mut table = BTreeMap::default();
        for Event { time, informs } in events {
            match table.entry(time) {
                btree_map::Entry::Vacant(e) => {
                    e.insert(VecDeque::from(informs));
                }
                btree_map::Entry::Occupied(mut e) => {
                    e.get_mut().extend(informs);
                }
            }
        }

        let num_nodes = graph.node_count();
        let fnum_nodes = V::from_usize(num_nodes).unwrap();
        let mean_degree = V::from_usize(graph.edge_count()).unwrap() / fnum_nodes;

        let mut info_contents = infos.into_iter().map(|obj| obj.into()).collect_vec();
        let observer = observer.map(
            |ObserverParam {
                 observe_prob,
                 observed_info,
                 post_prob,
                 threashold_rate,
             }| {
                // let k = (fnum_nodes * observer_pop_rate).round().to_usize().unwrap();
                let threshold = (fnum_nodes * threashold_rate).round().to_usize().unwrap();
                info_contents.push(InfoContent::Observation { op: observed_info });
                Observer {
                    observed_info_obj_idx: info_contents.len() - 1,
                    po: observe_prob,
                    pp: post_prob,
                    threshold,
                }
            },
        );
        Ok(Self {
            graph,
            info_contents,
            table,
            num_nodes,
            fnum_nodes,
            mean_degree,
            observer,
        })
    }
}

#[cfg(test)]
mod tests {
    use graph_lib::prelude::Graph;
    use std::fs::read_to_string;

    use super::{Inform, Scenario, ScenarioParam};
    use base::info::InfoContent;

    #[test]
    fn test_toml() -> anyhow::Result<()> {
        let scenario: Scenario<f32> = toml::from_str::<ScenarioParam<f32>>(&read_to_string(
            "./test/config/test_scenario.toml",
        )?)?
        .try_into()?;
        assert_eq!(scenario.graph.node_count(), 12);
        assert_eq!(scenario.graph.directed(), false);

        assert!(matches!(
            scenario.info_contents[0],
            InfoContent::Misinfo { .. }
        ));
        assert!(matches!(
            scenario.info_contents[1],
            InfoContent::Correction { .. }
        ));
        assert!(matches!(
            scenario.info_contents[2],
            InfoContent::Inhibition { .. }
        ));

        assert!(
            matches!(scenario.table[&0][0], Inform {agent_idx, info_obj_idx} if agent_idx == 0 && info_obj_idx == 0)
        );
        assert!(
            matches!(scenario.table[&0][1], Inform {agent_idx, info_obj_idx} if agent_idx == 1 && info_obj_idx == 0)
        );
        assert!(
            matches!(scenario.table[&1][0], Inform {agent_idx, info_obj_idx} if agent_idx == 2 && info_obj_idx == 1)
        );

        let observer = scenario.observer.unwrap();
        assert_eq!(observer.po, 0.1);
        assert_eq!(observer.observed_info_obj_idx, 3);
        assert!(matches!(
            &scenario.info_contents[observer.observed_info_obj_idx],
            InfoContent::Observation { op } if op.b()[1] == 1.0
        ));
        Ok(())
    }
}
