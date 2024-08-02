use graph_lib::io::{DataFormat, ParseBuilder};
use graph_lib::prelude::{DiGraphB, Graph, GraphB, UndiGraphB};
use rand_distr::{Distribution, Exp1, Open01, Standard, StandardNormal};
use serde::Deserialize;
use serde_with::{serde_as, TryFromInto};
use std::collections::{btree_map, BTreeMap, VecDeque};
use std::{fs::File, io};
use subjective_logic::mul::labeled::OpinionD1;

use crate::info::gen2::InfoContent;
use crate::opinion::{MyFloat, O};

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

#[derive(Debug, Deserialize)]
pub struct ScenarioParam<V>
where
    V: MyFloat,
{
    graph: GraphInfo,
    infos: Vec<InfoContent<V>>,
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
    pub info_objects: Vec<InfoContent<V>>,
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
    observer_pop_rate: V,
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
            infos: mut info_objects,
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

        let observer = observer.map(
            |ObserverParam {
                 observer_pop_rate,
                 observed_info,
                 post_prob,
                 threashold_rate,
             }| {
                // let k = (fnum_nodes * observer_pop_rate).round().to_usize().unwrap();
                let threshold = (fnum_nodes * threashold_rate).round().to_usize().unwrap();
                info_objects.push(InfoContent::Observation { op: observed_info });
                Observer {
                    // observer_pop_rate,
                    observed_info_obj_idx: info_objects.len() - 1,
                    po: observer_pop_rate,
                    pp: post_prob,
                    threshold,
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

#[cfg(test)]
mod tests {
    use graph_lib::prelude::Graph;
    use std::fs::read_to_string;

    use super::{Inform, Scenario, ScenarioParam};
    use crate::info::gen2::InfoContent;

    #[test]
    fn test_toml() -> anyhow::Result<()> {
        let scenario: Scenario<f32> = toml::from_str::<ScenarioParam<f32>>(&read_to_string(
            "./test/config/test_scenario.toml",
        )?)?
        .try_into()?;
        assert_eq!(scenario.graph.node_count(), 12);
        assert_eq!(scenario.graph.directed(), false);

        assert!(matches!(
            scenario.info_objects[0],
            InfoContent::Misinfo { .. }
        ));
        assert!(matches!(
            scenario.info_objects[1],
            InfoContent::Correction { .. }
        ));
        assert!(matches!(
            scenario.info_objects[2],
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
        assert_eq!(observer.po, 0.0);
        assert_eq!(observer.observed_info_obj_idx, 3);
        assert!(matches!(
            &scenario.info_objects[observer.observed_info_obj_idx],
            InfoContent::Observation { op } if op.b()[1] == 1.0
        ));
        Ok(())
    }
}
