use std::collections::BTreeMap;
use std::fs::File;
use std::io;
use std::path::PathBuf;

use approx::UlpsEq;
use graph_lib::io::{DataFormat, ParseBuilder};
use graph_lib::prelude::{DiGraphB, GraphB, UndiGraphB};
use num_traits::{Float, NumAssign};
use rand_distr::uniform::SampleUniform;
use rand_distr::{Distribution, Open01, Standard};
use serde::Deserialize;
use serde_with::{serde_as, FromInto, TryFromInto};

use crate::agent::AgentParams;
use crate::info::{InfoContent, InfoObject};

#[derive(serde::Deserialize)]
pub struct Config<V>
where
    V: Float + UlpsEq + NumAssign + SampleUniform,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
{
    pub name: String,
    #[serde(default)]
    pub output: Output,
    pub runtime: Runtime,
    pub scenario: Scenario<V>,
}

#[derive(serde::Deserialize, Debug)]
pub struct Output {
    #[serde(default = "default_output_location")]
    pub location: PathBuf,
    #[serde(default = "default_output_suffix")]
    pub suffix: String,
    /// Compress output file as zstd format
    #[serde(default = "default_output_compress")]
    pub compress: bool,
}

fn default_output_location() -> PathBuf {
    PathBuf::from("./")
}
fn default_output_suffix() -> String {
    "out".to_string()
}
fn default_output_compress() -> bool {
    true
}

impl Default for Output {
    fn default() -> Self {
        Self {
            location: default_output_location(),
            suffix: default_output_suffix(),
            compress: default_output_compress(),
        }
    }
}

#[serde_as]
#[derive(Debug, serde::Deserialize)]
pub struct Runtime {
    #[serde_as(as = "TryFromInto<GraphInfo>")]
    pub graph: GraphB,
    pub seed_state: u64,
    pub iteration_count: u32,
}

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

#[serde_as]
#[derive(serde::Deserialize)]
pub struct Scenario<V>
where
    V: Float + UlpsEq + NumAssign + SampleUniform,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
{
    pub agent_params: AgentParams<V>,
    #[serde_as(as = "Vec<FromInto<InfoObject<V>>>")]
    pub info_contents: Vec<InfoContent<V>>,
    #[serde_as(as = "FromInto<Vec<Event>>")]
    pub event_table: EventTable,
}

#[derive(serde::Deserialize, Debug)]
struct Event {
    time: u32,
    informs: Vec<Inform>,
}

#[derive(serde::Deserialize, Debug)]
struct Inform {
    agent_idx: usize,
    info_content_idx: usize,
}

/// time -> agent_idx -> info_content_idx
#[derive(Debug, Clone)]
pub struct EventTable(pub BTreeMap<u32, BTreeMap<usize, usize>>);

impl From<Vec<Event>> for EventTable {
    fn from(value: Vec<Event>) -> Self {
        Self(
            value
                .into_iter()
                .map(|Event { time, informs }| {
                    (
                        time,
                        informs
                            .into_iter()
                            .map(
                                |Inform {
                                     agent_idx,
                                     info_content_idx,
                                 }| (agent_idx, info_content_idx),
                            )
                            .collect(),
                    )
                })
                .collect(),
        )
    }
}

pub enum ConfigFormat {
    JSON(String),
    TOML(String),
}

#[derive(Debug, thiserror::Error)]
pub enum ParseConfigError {
    #[error("{0}")]
    JSONError(#[from] serde_json::Error),
    #[error("{0}")]
    TOMLError(#[from] toml::de::Error),
}

impl<V> TryFrom<ConfigFormat> for Config<V>
where
    V: Float + UlpsEq + NumAssign + SampleUniform,
    for<'de> V: Deserialize<'de>,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
{
    type Error = ParseConfigError;

    fn try_from(value: ConfigFormat) -> Result<Self, Self::Error> {
        match value {
            ConfigFormat::JSON(s) => Ok(serde_json::from_str(&s)?),
            ConfigFormat::TOML(s) => Ok(toml::from_str(&s)?),
        }
    }
}
