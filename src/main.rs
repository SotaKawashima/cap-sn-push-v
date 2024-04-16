use std::{io, path::PathBuf};

use cap_sn::{
    config::{DataFormat, FileReadError},
    Runner,
};
use clap::Parser;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(clap::Parser)]
struct Cli {
    /// string to identify given configuration data
    identifier: String,
    /// the path of output files
    output_dir: PathBuf,
    /// the path of a runtime config file
    #[arg(short, long)]
    runtime: String,
    /// the path of a agent parameters config file
    #[arg(short, long)]
    agent_params: String,
    /// the path of a scenario config file
    #[arg(short, long)]
    scenario: String,
    /// Enable overwriting of a output file
    #[arg(short, long, default_value_t = false)]
    overwriting: bool,
    /// Compress a output file
    #[arg(short, long, default_value_t = true)]
    compressing: bool,
}

#[derive(clap::Args)]
#[group(required = true, multiple = false)]
struct InputConfig {
    #[arg(short, long)]
    file_path: Option<String>,
    #[arg(short, long)]
    json_data: Option<String>,
    #[arg(short, long)]
    toml_data: Option<String>,
}

impl TryFrom<InputConfig> for DataFormat {
    type Error = FileReadError;

    fn try_from(value: InputConfig) -> Result<Self, Self::Error> {
        match (value.file_path, value.json_data, value.toml_data) {
            (Some(path), None, None) => DataFormat::read(&path),
            (None, Some(data), None) => Ok(DataFormat::JSON(data)),
            (None, None, Some(data)) => Ok(DataFormat::TOML(data)),
            _ => unreachable!(),
        }
    }
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::Registry::default()
        .with(tracing_subscriber::fmt::Layer::default().with_writer(io::stderr))
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .try_init()?;

    let args = Cli::parse();
    let executor = Runner::<f32>::try_new(
        args.runtime,
        args.agent_params,
        args.scenario,
        args.identifier,
        args.output_dir,
        args.overwriting,
        args.compressing,
    )?;
    executor.run()
}
