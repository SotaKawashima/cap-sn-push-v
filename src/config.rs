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
    use serde_json::json;

    #[test]
    fn test_json_config() -> anyhow::Result<()> {
        let g = json!({
            "output": {
                "location": "./test/",
            },
        });
        let general = serde_json::from_value::<General>(g)?;
        println!("{:?}", general);

        let runtime = json!({
            "seed_state": 0,
            "num_parallel": 1,
            "iteration_count": 1,
        });
        let runtime = serde_json::from_value::<Runtime>(runtime)?;
        println!("{:?}", runtime);
        Ok(())
    }

    #[test]
    fn test_toml_config() -> anyhow::Result<()> {
        let runtime =
            toml::from_str::<Runtime>(&read_to_string("./test/config/test_runtime.toml")?)?;
        assert_eq!(runtime.seed_state, 0);
        assert_eq!(runtime.iteration_count, 1);

        Ok(())
    }
}
