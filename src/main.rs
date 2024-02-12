use std::{fs, io, path::PathBuf};

use cap_sn::{config::ConfigFormat, Runner};
use clap::Parser;

#[derive(clap::Parser)]
struct Cli {
    #[command(flatten)]
    config: InputConfig,
    /// string to identify given configuration data
    identifier: String,
}

#[derive(clap::Args)]
#[group(required = true, multiple = false)]
struct InputConfig {
    #[arg(short, long)]
    file_path: Option<PathBuf>,
    #[arg(short, long)]
    json_data: Option<String>,
    #[arg(short, long)]
    toml_data: Option<String>,
}

#[derive(thiserror::Error, Debug)]
enum ConfigError {
    #[error("extension is needed.")]
    NoExtension,
    #[error("extension {0} is invalid.")]
    InvalidExtension(String),
    #[error("{0}")]
    IntoStringError(#[from] std::str::Utf8Error),
    #[error("{0}")]
    IOError(#[from] io::Error),
}

impl TryFrom<InputConfig> for ConfigFormat {
    type Error = ConfigError;

    fn try_from(value: InputConfig) -> Result<Self, Self::Error> {
        match (value.file_path, value.json_data, value.toml_data) {
            (Some(path), None, None) => match path.extension() {
                None => Err(ConfigError::NoExtension),
                Some(ext) => match <&str>::try_from(ext)? {
                    "json" => Ok(ConfigFormat::JSON(fs::read_to_string(path)?)),
                    "toml" => Ok(ConfigFormat::TOML(fs::read_to_string(path)?)),
                    s => Err(ConfigError::InvalidExtension(s.to_string())),
                },
            },
            (None, Some(data), None) => Ok(ConfigFormat::JSON(data)),
            (None, None, Some(data)) => Ok(ConfigFormat::TOML(data)),
            _ => unreachable!(),
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let args = Cli::parse();
    let mut executor = Runner::<f32>::try_new(args.config.try_into()?, args.identifier)?;
    executor.run()
}
