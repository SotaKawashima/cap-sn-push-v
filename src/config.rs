use std::collections::BTreeMap;
use std::fs::File;
use std::io;
use std::path::PathBuf;

use approx::UlpsEq;
use graph_lib::io::{DataFormat, ParseBuilder};
use graph_lib::prelude::{DiGraphB, GraphB, UndiGraphB};
use num_traits::{Float, NumAssign};
use rand_distr::uniform::SampleUniform;
use rand_distr::{Distribution, Exp1, Open01, Standard, StandardNormal};
use serde::Deserialize;
use serde_with::{serde_as, FromInto, TryFromInto};

use crate::agent::AgentParams;
use crate::info::{InfoContent, InfoObject};

#[derive(Debug, serde::Deserialize)]
pub struct Config<V>
where
    V: Float + UlpsEq + NumAssign + SampleUniform,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
{
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
    pub num_parallel: u32,
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
#[derive(Debug, serde::Deserialize)]
pub struct Scenario<V>
where
    V: Float + UlpsEq + NumAssign + SampleUniform,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
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
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
{
    type Error = ParseConfigError;

    fn try_from(value: ConfigFormat) -> Result<Self, Self::Error> {
        match value {
            ConfigFormat::JSON(s) => Ok(serde_json::from_str(&s)?),
            ConfigFormat::TOML(s) => Ok(toml::from_str(&s)?),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use super::Config;
    use serde_json::json;

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
                    "num_parallel": 1,
                    "iteration_count": 1,
                },
                "scenario": {
                    "agent_params": {
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
                                "cond_theta" : [
                                    [
                                        { "Fixed" : [[0.95, 0.00], 0.05] },
                                        { "Fixed" : [[0.45, 0.45], 0.10] },
                                    ],
                                    [
                                        { "Fixed" : [[0.475, 0.475], 0.05] },
                                        { "Fixed" : [[0.495, 0.495], 0.01] },
                                    ]
                                ],
                                "cond_theta_phi" : [
                                    { "Fixed" : [[0.00, 0.0], 1.00] },
                                    { "Fixed" : [[0.99, 0.0], 0.01] }
                                ],
                                "cond_a" : [
                                    { "Fixed" : [[0.95, 0.00], 0.05] },
                                    { "Fixed" : [[0.00, 0.95], 0.05] },
                                ],
                                "cond_thetad" : [
                                    [
                                        [
                                            { "Fixed" : [[0.95, 0.00], 0.05] },
                                            { "Fixed" : [[0.95, 0.00], 0.05] },
                                        ],
                                        [
                                            { "Fixed" : [[0.45, 0.45], 0.10] },
                                            { "Fixed" : [[0.45, 0.45], 0.10] },
                                        ],
                                    ],
                                    [
                                        [
                                            { "Fixed" : [[0.475, 0.475], 0.05] },
                                            { "Fixed" : [[0.00 , 0.95 ], 0.05] },
                                        ],
                                        [
                                            { "Fixed" : [[0.495, 0.495], 0.01] },
                                            { "Fixed" : [[0.00 , 0.99 ], 0.01] },
                                        ],
                                    ]
                                ],
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
                                "cond_ftheta" : [
                                    [
                                        { "Fixed" : [[0.95, 0.00], 0.05] },
                                        { "Fixed" : [[0.45, 0.45], 0.10] },
                                    ],
                                    [
                                        { "Fixed" : [[0.475, 0.475], 0.05] },
                                        { "Fixed" : [[0.495, 0.495], 0.01] },
                                    ]
                                ],
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
                                "cond_ktheta" : [
                                    [
                                        { "Fixed" : [[0.95, 0.00], 0.05] },
                                        { "Fixed" : [[0.45, 0.45], 0.10] },
                                    ],
                                    [
                                        { "Fixed" : [[0.475, 0.475], 0.05] },
                                        { "Fixed" : [[0.495, 0.495], 0.01] },
                                    ]
                                ],
                                "cond_ktheta_kphi" : [
                                    { "Fixed" : [[0.00, 0.0], 1.00] },
                                    { "Fixed" : [[0.90, 0.0], 0.10] },
                                ],
                            },
                        },
                        // "pi_prob": 1.0,
                        // "pi_rate": { "base": 0.5 },
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
        println!("{:?}", r);
        if let Err(e) = &r {
            println!("{:?}", e.source());
        }
        let c = r.unwrap();
        println!("{:?}", c.output);
        println!("{:?}", c.runtime.graph);
        println!("{:?}", c.scenario.event_table);
        println!("{:?}", c.scenario.info_contents);
    }

    #[test]
    fn test_toml_config() {
        let text = r#"
            name = "test-senario"

            [runtime]
            graph = { directed = false, location = { LocalFile = "./test/graph.txt" } }
            seed_state = 0
            num_parallel = 1
            iteration_count = 1

            [scenario.agent_params.base_rates.base]
            psi    = [0.999, 0.001]
            phi    = [0.999, 0.001]
            s      = [0.999, 0.001]
            o      = [0.999, 0.001]
            b      = [0.999, 0.001]
            theta  = [0.999, 0.001]
            a      = [0.999, 0.001]
            thetad = [0.999, 0.001]
            [scenario.agent_params.base_rates.friend]
            fpsi    = [0.999, 0.001]
            fphi    = [0.999, 0.001]
            fs      = [0.999, 0.001]
            fo      = [0.999, 0.001]
            fb      = [0.999, 0.001]
            ftheta  = [0.999, 0.001]
            [scenario.agent_params.base_rates.social]
            kpsi    = [0.999, 0.001]
            kphi    = [0.999, 0.001]
            ks      = [0.999, 0.001]
            ko      = [0.999, 0.001]
            kb      = [0.999, 0.001]
            ktheta  = [0.999, 0.001]

            [scenario.agent_params.initial_opinions.base]
            psi  = [[0.0, 0.0], 1.0]
            phi  = [[0.0, 0.0], 1.0]
            s    = [[0.0, 0.0], 1.0]
            o    = [[0.0, 0.0], 1.0]
            [scenario.agent_params.initial_opinions.friend]
            fphi = [[0.0, 0.0], 1.0]
            fs   = [[0.0, 0.0], 1.0]
            fo   = [[0.0, 0.0], 1.0]
            [scenario.agent_params.initial_opinions.social]
            kphi = [[0.0, 0.0], 1.0]
            ks   = [[0.0, 0.0], 1.0]
            ko   = [[0.0, 0.0], 1.0]

            [scenario.agent_params.initial_conditions.base]
            cond_o = [
                { Fixed = [[1.0, 0.00], 0.00] },
                { Fixed = [[0.0, 0.70], 0.30] },
            ]
            cond_b = [
                { Fixed = [[0.90, 0.00], 0.10] },
                { Fixed = [[0.00, 0.99], 0.01] },
            ]
            cond_theta = [
                [
                    { Fixed = [[0.95, 0.00], 0.05] },
                    { Fixed = [[0.45, 0.45], 0.10] },
                ],
                [
                    { Fixed = [[0.475, 0.475], 0.05] },
                    { Fixed = [[0.495, 0.495], 0.01] },
                ]
            ]
            cond_theta_phi= [
                { Fixed = [[0.00, 0.0], 1.00] },
                { Fixed = [[0.99, 0.0], 0.01] }
            ]
            cond_a = [
                { Fixed = [[0.95, 0.00], 0.05] },
                { Fixed = [[0.00, 0.95], 0.05] },
            ]
            cond_thetad = [
                [
                    [
                        { Fixed = [[0.95, 0.00], 0.05] },
                        { Fixed = [[0.95, 0.00], 0.05] },
                    ],
                    [
                        { Fixed = [[0.45, 0.45], 0.10] },
                        { Fixed = [[0.45, 0.45], 0.10] },
                    ],
                ],
                [
                    [
                        { Fixed = [[0.475, 0.475], 0.05] },
                        { Fixed = [[0.00 , 0.95 ], 0.05] },
                    ],
                    [
                        { Fixed = [[0.495, 0.495], 0.01] },
                        { Fixed = [[0.00 , 0.99 ], 0.01] },
                    ],
                ]
            ]
            cond_thetad_phi = [
                { Fixed = [[0.00, 0.0], 1.00] },
                { Fixed = [[0.99, 0.0], 0.01] },
            ]
            [scenario.agent_params.initial_conditions.friend]
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
            cond_ftheta = [
                [
                    { Fixed = [[0.95, 0.00], 0.05] },
                    { Fixed = [[0.45, 0.45], 0.10] },
                ],
                [
                    { Fixed = [[0.475, 0.475], 0.05] },
                    { Fixed = [[0.495, 0.495], 0.01] },
                ]
            ]
            cond_ftheta_fphi = [
                { Fixed = [[0.00, 0.0], 1.00] },
                { Fixed = [[0.90, 0.0], 0.10] }
            ]
            [scenario.agent_params.initial_conditions.social]
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
            cond_ktheta = [
                [
                    { Fixed = [[0.95, 0.00], 0.05] },
                    { Fixed = [[0.45, 0.45], 0.10] },
                ],
                [
                    { Fixed = [[0.475, 0.475], 0.05] },
                    { Fixed = [[0.495, 0.495], 0.01] },
                ]
            ]
            cond_ktheta_kphi = [
                { Fixed = [[0.00, 0.0], 1.00] },
                { Fixed = [[0.90, 0.0], 0.10] },
            ]

            [scenario.agent_params]
            access_prob         = { base = 0.0, error = { dist = { Beta = { alpha = 3.0, beta = 3.0 } } } }
            friend_access_prob  = { base = 0.0, error = { dist = { Beta = { alpha = 3.0, beta = 3.0 } } } }
            social_access_prob  = { base = 0.0, error = { dist = { Beta = { alpha = 3.0, beta = 3.0 } } } }
            friend_arrival_prob = { base = 0.0, error = { dist = { Beta = { alpha = 3.0, beta = 3.0 } } } }
            # pi_prob       = 1.0
            # pi_rate       = { base = 0.5 }

            [scenario.agent_params.trust_params]
            misinfo    = { base = 0.0, error = { dist = { Beta = { alpha = 3.0, beta = 3.0 } } } }
            corrective = { base = 0.0, error = { dist = { Beta = { alpha = 3.0, beta = 3.0 } } } }
            observed   = { base = 0.0, error = { dist = "Standard" } }
            inhibitive = { base = 0.0, error = { dist = "Standard", low = 0.5, high = 1.0 } }

            [scenario.agent_params.cpt_params]
            x0 = { base = -2.0 }
            x1 = { base = -50.0 }
            y  = { base = -0.001 }
            alpha  = { base = 0.88 }
            beta   = { base = 0.88 }
            lambda = { base = 2.25 }
            gamma  = { base = 0.61 }
            delta  = { base = 0.69 }

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
            "#;
        let r = toml::from_str::<Config<f32>>(&text);

        if let Err(e) = &r {
            let r = e.span().unwrap();
            let s = r.start.checked_sub(10).unwrap_or_default();
            let t = (r.end + 10).min(text.len());
            println!("{:?}", text.get(s..t));
        }
        let c = r.unwrap();
        println!("{:?}", c.output);
        println!("{:?}", c.runtime.graph);
        println!("{:?}", c.scenario.agent_params.initial_opinions);
        println!(
            "{:?}",
            c.scenario.agent_params.initial_conditions.base.cond_theta
        );
        println!(
            "{:?}",
            c.scenario
                .agent_params
                .initial_conditions
                .social
                .cond_ktheta_kphi
        );
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
