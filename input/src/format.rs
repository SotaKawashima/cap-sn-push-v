use std::{fs, io, path::Path};

use serde::Deserialize;

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
    pub fn read<P: AsRef<Path>>(path: P) -> Result<Self, FileReadError> {
        let path = path.as_ref();
        match path.extension() {
            None => Err(FileReadError::NoExtension),
            Some(ext) => match <&str>::try_from(ext)? {
                "json" => Ok(DataFormat::JSON(fs::read_to_string(path).map_err(
                    |err| FileReadError::IOError {
                        path: path.to_string_lossy().to_string(),
                        err,
                    },
                )?)),
                "toml" => Ok(DataFormat::TOML(fs::read_to_string(path).map_err(
                    |err| FileReadError::IOError {
                        path: path.to_string_lossy().to_string(),
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
