use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::de::DeserializeOwned;
use serde::Deserialize;
use serde_with::serde_as;

pub struct ConfigData<T> {
    pub path: String,
    pub data: T,
}

impl<T> ConfigData<T> {
    pub fn try_new<S>(path: String) -> anyhow::Result<Self>
    where
        S: DeserializeOwned,
        S: TryInto<T>,
        anyhow::Error: From<S::Error>,
    {
        let data = DataFormat::read(&path)?.parse::<S>()?.try_into()?;
        Ok(Self { path, data })
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
    #[error("{err}: {path}")]
    IOError { path: String, err: io::Error },
}

impl DataFormat {
    pub fn read(path: &str) -> Result<Self, FileReadError> {
        match Path::new(path).extension() {
            None => Err(FileReadError::NoExtension),
            Some(ext) => match <&str>::try_from(ext)? {
                "json" => Ok(DataFormat::JSON(fs::read_to_string(path).map_err(
                    |err| FileReadError::IOError {
                        path: path.to_string(),
                        err,
                    },
                )?)),
                "toml" => Ok(DataFormat::TOML(fs::read_to_string(path).map_err(
                    |err| FileReadError::IOError {
                        path: path.to_string(),
                        err,
                    },
                )?)),
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
    use std::fs::read_to_string;

    use super::{General, Runtime};
    use crate::agent::AgentParams;
    use crate::info::InfoObject;
    use crate::opinion::SimplexDist;
    use crate::scenario::{Inform, Scenario, ScenarioParam};
    use graph_lib::prelude::Graph;
    use serde_json::json;
    use subjective_logic::marr_d1;
    use subjective_logic::mul::labeled::SimplexD1;

    #[test]
    fn test_json_config() -> anyhow::Result<()> {
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
                    "psi" : [[0.0, 0.0], 1.0],
                    "phi" : [[0.0, 0.0], 1.0],
                    "m"   : [[0.0, 0.0], 1.0],
                    "o"   : [[0.0, 0.0], 1.0],
                    "h_by_phi1_psi1" : [[0.0, 0.0], 1.0],
                    "h_by_phi1_b1"   : [[0.0, 0.0], 1.0],
                },
                "friend": {
                    "fphi" : [[0.0, 0.0], 1.0],
                    "fm"   : [[0.0, 0.0], 1.0],
                    "fo"   : [[0.0, 0.0], 1.0],
                    "fh_by_fphi1_fpsi1" : [[0.0, 0.0], 1.0],
                    "fh_by_fphi1_fb1"   : [[0.0, 0.0], 1.0],
                },
                "social": {
                    "kphi" : [[0.0, 0.0], 1.0],
                    "km"   : [[0.0, 0.0], 1.0],
                    "ko"   : [[0.0, 0.0], 1.0],
                    "kh_by_kphi1_kpsi1" : [[0.0, 0.0], 1.0],
                    "kh_by_kphi1_kb1"   : [[0.0, 0.0], 1.0],
                }
            },
            "base_rates": {
                "base": {
                    "psi"    : [0.999, 0.001],
                    "phi"    : [0.999, 0.001],
                    "m"      : [0.999, 0.001],
                    "o"      : [0.999, 0.001],
                    "a"      : [0.999, 0.001],
                    "b"      : [0.999, 0.001],
                    "h"      : [0.999, 0.001],
                    "theta"  : [0.999, 0.001],
                    "thetad" : [0.999, 0.001],
                },
                "friend": {
                    "fpsi" : [0.999, 0.001],
                    "fphi" : [0.999, 0.001],
                    "fm"   : [0.999, 0.001],
                    "fo"   : [0.999, 0.001],
                    "fb"   : [0.999, 0.001],
                    "fh"   : [0.999, 0.001],
                },
                "social": {
                    "kpsi" : [0.999, 0.001],
                    "kphi" : [0.999, 0.001],
                    "km"   : [0.999, 0.001],
                    "ko"   : [0.999, 0.001],
                    "kb"   : [0.999, 0.001],
                    "kh"   : [0.999, 0.001],
                },
            },
            "initial_conditions": {
                "base": {
                    "a_fh" : [
                        { "Fixed" : [[0.95, 0.00], 0.05] },
                        { "Fixed" : [[0.00, 0.95], 0.05] },
                    ],
                    "b_kh" : [
                        { "Fixed" : [[0.90, 0.00], 0.10] },
                        { "Fixed" : [[0.00, 0.99], 0.01] },
                    ],
                    "o_b" : [
                        { "Fixed" : [[1.0, 0.00], 0.00] },
                        { "Fixed" : [[0.0, 0.70], 0.30] },
                    ],
                    "theta_h" : [
                        { "Fixed" : [[1.0, 0.00], 0.00] },
                        { "Fixed" : [[0.0, 0.70], 0.30] },
                    ],
                    "thetad_h" : [
                        { "Fixed" : [[1.0, 0.00], 0.00] },
                        { "Fixed" : [[0.0, 0.70], 0.30] },
                    ],
                    "h_psi_by_phi0" : [
                        { "Fixed" : [[0.95, 0.00], 0.05] },
                        { "Fixed" : [[0.60, 0.30], 0.10] },
                    ],
                    "h_b_by_phi0" : [
                        { "Fixed" : [[0.95, 0.00], 0.05] },
                        { "Fixed" : [[0.45, 0.45], 0.10] },
                    ],
                },
                "friend": {
                    "fpsi_m" : [
                        { "Fixed" : [[0.99, 0.00], 0.01] },
                        { "Fixed" : [[0.70, 0.20], 0.10] },
                    ],
                    "fo_fb" : [
                        { "Fixed" : [[1.0, 0.00], 0.00] },
                        { "Fixed" : [[0.0, 0.70], 0.30] },
                    ],
                    "fb_fm" : [
                        { "Fixed" : [[0.90, 0.00], 0.10] },
                        { "Fixed" : [[0.00, 0.99], 0.01] },
                    ],
                    "fh_fpsi_by_fphi0" : [
                        { "Fixed" : [[0.95, 0.00], 0.05] },
                        { "Fixed" : [[0.60, 0.30], 0.10] },
                    ],
                    "fh_fb_by_fphi0"  : [
                        { "Fixed" : [[0.95, 0.00], 0.05] },
                        { "Fixed" : [[0.45, 0.45], 0.10] },
                    ],
                },
                "social": {
                    "kpsi_m" : [
                        { "Fixed" : [[0.99, 0.00], 0.01] },
                        { "Fixed" : [[0.25, 0.65], 0.10] },
                    ],
                    "ko_kb" : [
                        { "Fixed" : [[1.0, 0.00], 0.00] },
                        { "Fixed" : [[0.0, 0.70], 0.30] },
                    ],
                    "kb_km" : [
                        { "Fixed" : [[1.00, 0.00], 0.00] },
                        { "Fixed" : [[0.25, 0.65], 0.10] },
                    ],
                    "kh_kpsi_by_kphi0" : [
                        { "Fixed" : [[0.95, 0.00], 0.05] },
                        { "Fixed" : [[0.60, 0.30], 0.10] },
                    ],
                    "kh_kb_by_kphi0"  : [
                        { "Fixed" : [[0.95, 0.00], 0.05] },
                        { "Fixed" : [[0.45, 0.45], 0.10] },
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
            "loss_params": {
                "x0": {"base": -1.0},
                "x1_of_x0": {"base": -10.0},
                "y_of_x0": {"base": -0.001},
            },
            "cpt_params": {
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
                "info_objects": [
                    { "Misinfo": { "psi": ([0.00, 0.99], 0.01) } },
                    { "Corrective": { "psi": ([0.99, 0.00], 0.01), "m": ([0.0, 1.0], 0.0) } }
                ],
                "events": [
                    {
                        "time": 0,
                        "informs": [
                            { "agent_idx": 0, "info_obj_idx": 0, },
                            { "agent_idx": 1, "info_obj_idx": 0, },
                        ],
                    },
                    {
                        "time": 1,
                        "informs": [
                            { "agent_idx": 2, "info_obj_idx": 1, },
                        ]
                    },
                ],
                "observer": {
                    "observer_pop_rate": 0.0,
                    "observed_info": ([0.0, 1.0], 0.0)
                }
            }
        );
        let general = serde_json::from_value::<General>(g)?;
        let runtime = serde_json::from_value::<Runtime>(runtime)?;
        let agent_params = serde_json::from_value::<AgentParams<f32>>(agent_params)?;
        let scenario = Scenario::try_from(serde_json::from_value::<ScenarioParam<f32>>(scenario)?)?;
        println!("{:?}", general);
        println!("{:?}", runtime);
        println!("{:?}", agent_params.initial_opinions);
        println!("{:?}", scenario.graph);
        println!("{:?}", scenario.info_objects);
        println!("{:?}", scenario.table);
        Ok(())
    }

    #[test]
    fn test_toml_config() -> anyhow::Result<()> {
        let runtime =
            toml::from_str::<Runtime>(&read_to_string("./test/config/test_runtime.toml")?)?;
        let agent_params = toml::from_str::<AgentParams<f32>>(&read_to_string(
            "./test/config/test_agent_params.toml",
        )?)?;
        let scenario: Scenario<f32> = toml::from_str::<ScenarioParam<f32>>(&read_to_string(
            "./test/config/test_scenario.toml",
        )?)?
        .try_into()?;

        assert_eq!(runtime.seed_state, 0);
        assert_eq!(runtime.iteration_count, 1);
        assert_eq!(
            agent_params.initial_opinions.base.h_by_phi1_b1,
            SimplexD1::vacuous()
        );
        assert!(matches!(
            &agent_params.initial_conditions.base.theta_h[0],
            SimplexDist::Fixed(s) if s.b() == &marr_d1![1.0, 0.0] && s.u() == &0.0,
        ));
        assert!(matches!(
            &agent_params.initial_conditions.base.theta_h[1],
            SimplexDist::Fixed(s) if s.b() == &marr_d1![0.0, 0.7] && s.u() == &0.3,
        ));
        assert!(matches!(
            &agent_params.initial_conditions.social.kh_kpsi_by_kphi0[0],
            SimplexDist::Fixed(s) if s.b() == &marr_d1![0.95, 0.0] && s.u() == &0.05,
        ));
        assert!(matches!(
            &agent_params.initial_conditions.social.kh_kpsi_by_kphi0[1],
            SimplexDist::Fixed(s) if s.b() == &marr_d1![0.60, 0.30] && s.u() == &0.10,
        ));

        assert_eq!(scenario.graph.node_count(), 12);
        assert_eq!(scenario.graph.directed(), false);

        assert!(matches!(
            scenario.info_objects[0],
            InfoObject::Misinfo { .. }
        ));
        assert!(matches!(
            scenario.info_objects[1],
            InfoObject::Corrective { .. }
        ));
        assert!(matches!(
            scenario.info_objects[2],
            InfoObject::Inhibitive { .. }
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
        assert_eq!(observer.k, 0.01 * 12.0);
        assert_eq!(observer.observed_info_obj_idx, 3);
        assert!(matches!(
            &scenario.info_objects[observer.observed_info_obj_idx],
            InfoObject::Observed { o } if o.b()[1] == 1.0
        ));

        Ok(())
    }
}
