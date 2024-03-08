use std::path::PathBuf;

use cap_sn::{
    config::{DataFormat, FileReadError},
    Runner,
};
use clap::Parser;

#[derive(clap::Parser)]
struct Cli {
    /// string to identify given configuration data
    identifier: String,
    /// the path of a general config file
    #[arg(short, long)]
    general: PathBuf,
    /// the path of a runtime config file
    #[arg(short, long)]
    runtime: PathBuf,
    /// the path of a agent parameters config file
    #[arg(short, long)]
    agent_params: PathBuf,
    /// the path of a scenario config file
    #[arg(short, long)]
    scenario: PathBuf,
    /// Enable overwriting of a output file
    #[arg(short, long, default_value_t = false)]
    overwriting: bool,
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

impl TryFrom<InputConfig> for DataFormat {
    type Error = FileReadError;

    fn try_from(value: InputConfig) -> Result<Self, Self::Error> {
        match (value.file_path, value.json_data, value.toml_data) {
            (Some(path), None, None) => DataFormat::read(path),
            (None, Some(data), None) => Ok(DataFormat::JSON(data)),
            (None, None, Some(data)) => Ok(DataFormat::TOML(data)),
            _ => unreachable!(),
        }
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let args = Cli::parse();
    let executor = Runner::<PathBuf, f32>::try_new(
        args.general,
        args.runtime,
        args.agent_params,
        args.scenario,
        args.identifier,
        args.overwriting,
    )?;
    executor.run()
}
