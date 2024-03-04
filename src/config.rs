use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};

use approx::UlpsEq;
use graph_lib::io::{DataFormat as GraphDataFormat, ParseBuilder};
use graph_lib::prelude::{DiGraphB, GraphB, UndiGraphB};
use num_traits::{Float, NumAssign};
use rand_distr::uniform::SampleUniform;
use rand_distr::{Distribution, Exp1, Open01, Standard, StandardNormal};
use serde::de::DeserializeOwned;
use serde::Deserialize;
use serde_with::{serde_as, FromInto, TryFromInto};

use crate::info::{InfoContent, InfoObject};

pub struct ConfigData<P: AsRef<Path>, T> {
    pub path: P,
    pub data: T,
}

#[derive(thiserror::Error, Debug)]
pub enum ConfigError {
    #[error("{0}")]
    IOError(#[from] FileReadError),
    #[error("{0}")]
    ParseError(#[from] ParseError),
}

impl<P: AsRef<Path>, T> ConfigData<P, T> {
    pub fn try_new(path: P) -> Result<Self, ConfigError>
    where
        T: DeserializeOwned,
    {
        let data = DataFormat::read(&path)?.parse()?;
        Ok(Self { path, data })
    }

    pub fn get_path_string(&self) -> String {
        self.path.as_ref().to_str().unwrap().to_string()
    }
}

#[derive(Debug, serde::Deserialize)]
pub struct General {
    #[serde(default)]
    pub output: Output,
}

#[serde_as]
#[derive(Debug, serde::Deserialize)]
pub struct Runtime {
    pub seed_state: u64,
    pub num_parallel: u32,
    pub iteration_count: u32,
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
                let builder = ParseBuilder::new(File::open(path)?, GraphDataFormat::EdgeList);
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
#[derive(Debug, serde::Deserialize)]
#[serde(bound(deserialize = "V: Deserialize<'de>"))]
pub struct Scenario<V>
where
    V: Float + UlpsEq + NumAssign + SampleUniform,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
{
    #[serde_as(as = "TryFromInto<GraphInfo>")]
    pub graph: GraphB,
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

pub enum DataFormat {
    JSON(String),
    TOML(String),
}

#[derive(thiserror::Error, Debug)]
pub enum FileReadError {
    #[error("extension is needed.")]
    NoExtension,
    #[error("extension {0} is invalid.")]
    InvalidExtension(String),
    #[error("{0}")]
    IntoStringError(#[from] std::str::Utf8Error),
    #[error("{0}")]
    IOError(#[from] io::Error),
}

impl DataFormat {
    pub fn read<P: AsRef<Path>>(path: P) -> Result<Self, FileReadError> {
        match path.as_ref().extension() {
            None => Err(FileReadError::NoExtension),
            Some(ext) => match <&str>::try_from(ext)? {
                "json" => Ok(DataFormat::JSON(fs::read_to_string(path)?)),
                "toml" => Ok(DataFormat::TOML(fs::read_to_string(path)?)),
                s => Err(FileReadError::InvalidExtension(s.to_string())),
            },
        }
    }

    pub fn parse<'a, T: Deserialize<'a>>(&'a self) -> Result<T, ParseError> {
        match self {
            DataFormat::JSON(s) => Ok(serde_json::from_str(s)?),
            DataFormat::TOML(s) => Ok(T::deserialize(toml::Deserializer::new(s))?),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("{0}")]
    JSONError(#[from] serde_json::Error),
    #[error("{0}")]
    TOMLError(#[from] toml::de::Error),
}

#[cfg(test)]
mod tests {
    use super::{General, Runtime, Scenario};
    use crate::agent::AgentParams;
    use serde_json::json;

    #[test]
    fn test_json_config() {
        let g = json!({
            "output": {
                "location": "./test/",
            },
        });
        let runtime = json!({
            "seed_state": 0,
            "num_parallel": 1,
            "iteration_count": 1,
        });
        let agent_params = json!({
            "initial_opinions": {
                "base": {
                    "psi"  : [[0.0, 0.0], 1.0],
                    "phi"  : [[0.0, 0.0], 1.0],
                    "s"    : [[0.0, 0.0], 1.0],
                    "o"    : [[0.0, 0.0], 1.0],
                },
                "friend": {
                    "fphi" : [[0.0, 0.0], 1.0],
                    "fs"   : [[0.0, 0.0], 1.0],
                    "fo"   : [[0.0, 0.0], 1.0],
                },
                "social": {
                    "kphi" : [[0.0, 0.0], 1.0],
                    "ks"   : [[0.0, 0.0], 1.0],
                    "ko"   : [[0.0, 0.0], 1.0],
                }
            },
            "base_rates": {
                "base": {
                    "psi"    : [0.999, 0.001],
                    "phi"    : [0.999, 0.001],
                    "s"      : [0.999, 0.001],
                    "o"      : [0.999, 0.001],
                    "b"      : [0.999, 0.001],
                    "theta"  : [0.999, 0.001],
                    "a"      : [0.999, 0.001],
                    "thetad" : [0.999, 0.001],
                },
                "friend": {
                    "fpsi"    : [0.999, 0.001],
                    "fphi"    : [0.999, 0.001],
                    "fs"      : [0.999, 0.001],
                    "fo"      : [0.999, 0.001],
                    "fb"      : [0.999, 0.001],
                    "ftheta"  : [0.999, 0.001],
                },
                "social": {
                    "kpsi"    : [0.999, 0.001],
                    "kphi"    : [0.999, 0.001],
                    "ks"      : [0.999, 0.001],
                    "ko"      : [0.999, 0.001],
                    "kb"      : [0.999, 0.001],
                    "ktheta"  : [0.999, 0.001],
                },
            },
            "initial_conditions": {
                "base": {
                    "cond_o" : [
                        { "Fixed" : [[1.0, 0.00], 0.00] },
                        { "Fixed" : [[0.0, 0.70], 0.30] },
                    ],
                    "cond_b" : [
                        { "Fixed" : [[0.90, 0.00], 0.10] },
                        { "Fixed" : [[0.00, 0.99], 0.01] },
                    ],
                    "cond_theta" : {
                        "none": { "Fixed" : [[0.95, 0.00], 0.05] },
                        "possible": { "Fixed" : [[0.45, 0.45], 0.10] },
                        "rates": [[1.0, 1.0], [1.0, 1.0]]
                    },
                    "cond_theta_phi" : [
                        { "Fixed" : [[0.00, 0.0], 1.00] },
                        { "Fixed" : [[0.99, 0.0], 0.01] }
                    ],
                    "cond_a" : [
                        { "Fixed" : [[0.95, 0.00], 0.05] },
                        { "Fixed" : [[0.00, 0.95], 0.05] },
                    ],
                    "cond_thetad" : {
                        "none": { "Fixed" : [[0.95, 0.00], 0.05] },
                        "possible": { "Fixed" : [[0.45, 0.45], 0.10] },
                        "rates": [[1.0, 1.0], [1.0, 1.0]],
                        "avoid_u_rates": [1.0, 1.0, 1.0]
                    },
                    "cond_thetad_phi" : [
                        { "Fixed" : [[0.00, 0.0], 1.00] },
                        { "Fixed" : [[0.99, 0.0], 0.01] },
                    ],
                },
                "friend": {
                    "cond_fpsi" : [
                        { "Fixed" : [[0.99, 0.00], 0.01] },
                        { "Fixed" : [[0.70, 0.20], 0.10] },
                    ],
                    "cond_fo" : [
                        { "Fixed" : [[1.0, 0.00], 0.00] },
                        { "Fixed" : [[0.0, 0.70], 0.30] },
                    ],
                    "cond_fb" : [
                        { "Fixed" : [[0.90, 0.00], 0.10] },
                        { "Fixed" : [[0.00, 0.99], 0.01] },
                    ],
                    "cond_ftheta" : {
                        "none": { "Fixed" : [[0.95, 0.00], 0.05] },
                        "possible": { "Fixed" : [[0.45, 0.45], 0.10] },
                        "rates": [[1.0, 1.0], [1.0, 1.0]]
                    },
                    "cond_ftheta_fphi" : [
                        { "Fixed" : [[0.00, 0.0], 1.00] },
                        { "Fixed" : [[0.90, 0.0], 0.10] }
                    ],
                },
                "social": {
                    "cond_kpsi" : [
                        { "Fixed" : [[0.99, 0.00], 0.01] },
                        { "Fixed" : [[0.25, 0.65], 0.10] },
                    ],
                    "cond_ko" : [
                        { "Fixed" : [[1.0, 0.00], 0.00] },
                        { "Fixed" : [[0.0, 0.70], 0.30] },
                    ],
                    "cond_kb" : [
                        { "Fixed" : [[1.00, 0.00], 0.00] },
                        { "Fixed" : [[0.25, 0.65], 0.10] },
                    ],
                    "cond_ktheta" : {
                        "none": { "Fixed" : [[0.95, 0.00], 0.05] },
                        "possible": { "Fixed" : [[0.45, 0.45], 0.10] },
                        "rates": [[1.0, 1.0], [1.0, 1.0]]
                    },
                    "cond_ktheta_kphi" : [
                        { "Fixed" : [[0.00, 0.0], 1.00] },
                        { "Fixed" : [[0.90, 0.0], 0.10] },
                    ],
                },
            },
            // "pi_prob": 1.0,
            // "pi_rate": { "base": 0.5 },
            "delay_selfish": {"Fixed": 0 },
            "access_prob": { "base": 0.0, "error": { "dist": { "Beta": { "alpha": 3.0, "beta": 3.0 } }, "low": 0.0, "high": 1.0 } },
            "friend_access_prob": { "base": 0.0, "error": { "dist": { "Beta": { "alpha": 3.0, "beta": 3.0 } }, "low": 0.0, "high": 1.0 } },
            "social_access_prob": { "base": 0.0, "error": { "dist": { "Beta": { "alpha": 3.0, "beta": 3.0 } }, "low": 0.0, "high": 1.0 } },
            "friend_arrival_prob": { "base": 0.0, "error": { "dist": { "Beta": { "alpha": 3.0, "beta": 3.0 } }, "low": 0.0, "high": 1.0 } },
            "trust_params": {
                "misinfo": { "base": 0.0, "error": { "dist": { "Beta": { "alpha": 3.0, "beta": 3.0 } }, "low": 0.0, "high": 1.0 } },
                "corrective": { "base": 0.0, "error": { "dist": { "Beta": { "alpha": 3.0, "beta": 3.0 } }, "low": 0.0, "high": 1.0 } },
                "observed": { "base": 0.0, "error": { "dist": "Standard", "low": 0.0, "high": 1.0 } },
                "inhibitive": { "base": 0.0, "error": { "dist": "Standard", "low": 0.5, "high": 1.0 } },
            },
            "cpt_params": {
                "x0": {"base": -2.0},
                "x1": {"base": -50.0},
                "y":  {"base": -0.001},
                "alpha":  { "base": 0.88 },
                "beta":   { "base": 0.88 },
                "lambda": { "base": 2.25 },
                "gamma":  { "base": 0.61 },
                "delta":  { "base": 0.69 },
            }
        });
        let scenario = json!(
            {
                "graph": {
                    "directed": true,
                    "location": {
                        "LocalFile": "./test/graph/graph.txt",
                    },
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
        );
        let general = serde_json::from_value::<General>(g).unwrap();
        let runtime = serde_json::from_value::<Runtime>(runtime).unwrap();
        let agent_params = serde_json::from_value::<AgentParams<f32>>(agent_params).unwrap();
        let scenario = serde_json::from_value::<Scenario<f32>>(scenario).unwrap();
        println!("{:?}", general);
        println!("{:?}", runtime);
        println!("{:?}", agent_params.initial_opinions);
        println!("{:?}", scenario.graph);
        println!("{:?}", scenario.event_table);
        println!("{:?}", scenario.info_contents);
    }

    #[test]
    fn test_toml_config() {
        let general = toml::from_str::<General>(
            r#"
        [output]
        "#,
        )
        .unwrap();
        let runtime = toml::from_str::<Runtime>(
            r#"
            seed_state = 0
            num_parallel = 1
            iteration_count = 1
        "#,
        )
        .unwrap();
        let agent_params = toml::from_str::<AgentParams<f32>>(r#"
            delay_selfish       = { Fixed = 0 }
            access_prob         = { base = 0.0, error = { dist = { Beta = { alpha = 3.0, beta = 3.0 } } } }
            friend_access_prob  = { base = 0.0, error = { dist = { Beta = { alpha = 3.0, beta = 3.0 } } } }
            social_access_prob  = { base = 0.0, error = { dist = { Beta = { alpha = 3.0, beta = 3.0 } } } }
            friend_arrival_prob = { base = 0.0, error = { dist = { Beta = { alpha = 3.0, beta = 3.0 } } } }

            [base_rates.base]
            psi    = [0.999, 0.001]
            phi    = [0.999, 0.001]
            s      = [0.999, 0.001]
            o      = [0.999, 0.001]
            b      = [0.999, 0.001]
            theta  = [0.999, 0.001]
            a      = [0.999, 0.001]
            thetad = [0.999, 0.001]
            [base_rates.friend]
            fpsi    = [0.999, 0.001]
            fphi    = [0.999, 0.001]
            fs      = [0.999, 0.001]
            fo      = [0.999, 0.001]
            fb      = [0.999, 0.001]
            ftheta  = [0.999, 0.001]
            [base_rates.social]
            kpsi    = [0.999, 0.001]
            kphi    = [0.999, 0.001]
            ks      = [0.999, 0.001]
            ko      = [0.999, 0.001]
            kb      = [0.999, 0.001]
            ktheta  = [0.999, 0.001]

            [initial_opinions.base]
            psi  = [[0.0, 0.0], 1.0]
            phi  = [[0.0, 0.0], 1.0]
            s    = [[0.0, 0.0], 1.0]
            o    = [[0.0, 0.0], 1.0]
            [initial_opinions.friend]
            fphi = [[0.0, 0.0], 1.0]
            fs   = [[0.0, 0.0], 1.0]
            fo   = [[0.0, 0.0], 1.0]
            [initial_opinions.social]
            kphi = [[0.0, 0.0], 1.0]
            ks   = [[0.0, 0.0], 1.0]
            ko   = [[0.0, 0.0], 1.0]

            [initial_conditions.base.cond_thetad]
            none = { Fixed = [[0.95, 0.00], 0.05] }
            possible = { Fixed = [[0.45, 0.45], 0.10] }
            rates = [[1.0, 1.0], [1.0, 1.0]]
            avoid_u_rates = [1.0, 1.0, 1.0]

            [initial_conditions.base.cond_theta]
            none = { Fixed = [[0.95, 0.00], 0.05] }
            possible = { Fixed = [[0.45, 0.45], 0.10] }
            rates = [[1.0, 1.0], [1.0, 1.0]]

            [initial_conditions.friend.cond_ftheta]
            none = { Fixed = [[0.95, 0.00], 0.05] }
            possible = { Fixed = [[0.45, 0.45], 0.10] }
            rates = [[1.0, 1.0], [1.0, 1.0]]

            [initial_conditions.social.cond_ktheta]
            none = { Fixed = [[0.95, 0.00], 0.05] }
            possible = { Fixed = [[0.45, 0.45], 0.10] }
            rates = [[1.0, 1.0], [1.0, 1.0]]

            [initial_conditions.base]
            cond_o = [
                { Fixed = [[1.0, 0.00], 0.00] },
                { Dirichlet = { alpha = [5.0, 5.0, 5.0]} },
            ]
            cond_b = [
                { Fixed = [[0.90, 0.00], 0.10] },
                { Fixed = [[0.00, 0.99], 0.01] },
            ]
            cond_theta_phi= [
                { Fixed = [[0.00, 0.0], 1.00] },
                { Fixed = [[0.99, 0.0], 0.01] }
            ]
            cond_a = [
                { Fixed = [[0.95, 0.00], 0.05] },
                { Fixed = [[0.00, 0.95], 0.05] },
            ]
            cond_thetad_phi = [
                { Fixed = [[0.00, 0.0], 1.00] },
                { Fixed = [[0.99, 0.0], 0.01] },
            ]
            [initial_conditions.friend]
            cond_fpsi = [
                { Fixed = [[0.99, 0.00], 0.01] },
                { Fixed = [[0.70, 0.20], 0.10] },
            ]
            cond_fo = [
                { Fixed = [[1.0, 0.00], 0.00] },
                { Fixed = [[0.0, 0.70], 0.30] },
            ]
            cond_fb = [
                { Fixed = [[0.90, 0.00], 0.10] },
                { Fixed = [[0.00, 0.99], 0.01] },
            ]
            cond_ftheta_fphi = [
                { Fixed = [[0.00, 0.0], 1.00] },
                { Fixed = [[0.90, 0.0], 0.10] }
            ]
            [initial_conditions.social]
            cond_kpsi = [
                { Fixed = [[0.99, 0.00], 0.01] },
                { Fixed = [[0.25, 0.65], 0.10] },
            ]
            cond_ko = [
                { Fixed = [[1.0, 0.00], 0.00] },
                { Fixed = [[0.0, 0.70], 0.30] },
            ]
            cond_kb = [
                { Fixed = [[1.00, 0.00], 0.00] },
                { Fixed = [[0.25, 0.65], 0.10] },
            ]
            cond_ktheta_kphi = [
                { Fixed = [[0.00, 0.0], 1.00] },
                { Fixed = [[0.90, 0.0], 0.10] },
            ]

            [trust_params]
            misinfo    = { base = 0.0, error = { dist = { Beta = { alpha = 3.0, beta = 3.0 } } } }
            corrective = { base = 0.0, error = { dist = { Beta = { alpha = 3.0, beta = 3.0 } } } }
            observed   = { base = 0.0, error = { dist = "Standard" } }
            inhibitive = { base = 0.0, error = { dist = "Standard", low = 0.5, high = 1.0 } }

            [cpt_params]
            x0 = { base = -2.0 }
            x1 = { base = -50.0 }
            y  = { base = -0.001 }
            alpha  = { base = 0.88 }
            beta   = { base = 0.88 }
            lambda = { base = 2.25 }
            gamma  = { base = 0.61 }
            delta  = { base = 0.69 }
            "#).unwrap();
        let scenario = toml::from_str::<Scenario<f32>>(
            r#"
            graph = { directed = false, location = { LocalFile = "./test/graph/graph.txt" } }
            info_contents = [
                { Misinfo = { psi = [[0.00, 0.99], 0.01] } },
                { Corrective = { psi = [[0.99, 0.00], 0.01], s = [[0.0, 1.0], 0.0] } }
            ]
            [[event_table]]
            time = 0
            informs = [
                { agent_idx = 0, info_content_idx = 0 },
                { agent_idx = 1, info_content_idx = 0 },
            ]
            [[event_table]]
            time = 1
            informs = [{ agent_idx = 2, info_content_idx = 1 }]
            "#,
        )
        .unwrap();

        println!("{:?}", general);
        println!("{:?}", runtime);
        println!("{:?}", scenario.graph);
        println!("{:?}", agent_params.initial_opinions);
        println!("{:?}", agent_params.initial_conditions.base.cond_theta);
        println!(
            "{:?}",
            agent_params.initial_conditions.social.cond_ktheta_kphi
        );
        println!(
            "{:?}",
            scenario
                .info_contents
                .into_iter()
                .map(|i| i.label)
                .collect::<Vec<_>>()
        );
        println!("{:?}", scenario.event_table);
    }
}
