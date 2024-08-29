mod config;
mod exec;

use base::{
    opinion::MyFloat,
    runner::{run, RuntimeParams},
    stat::FileWriters,
};
use config::Config;
use exec::{AgentExt, Instance};
use input::format::DataFormat;
use polars_arrow::datatypes::Metadata;
use rand_distr::{Distribution, Exp1, Open01, Standard, StandardNormal};
use std::path::PathBuf;

#[derive(clap::Parser)]
pub struct Cli {
    /// string to identify given configuration data
    identifier: String,
    /// the path of output files
    output_dir: PathBuf,
    /// the path of a runtime config file
    #[arg(long)]
    runtime: String,
    /// the path of a network config file
    #[arg(long)]
    network: PathBuf,
    /// the path of a agent config file
    #[arg(long)]
    agent: PathBuf,
    /// the path of a strategy config file
    #[arg(long)]
    strategy: PathBuf,
    /// Enable inhibition information
    #[arg(short, default_value_t = false)]
    enable_inhibition: bool,
    /// Enable overwriting of a output file
    #[arg(short, default_value_t = false)]
    overwriting: bool,
    /// Compress a output file
    #[arg(short, default_value_t = false)]
    compressing: bool,
}

pub async fn start<V>(args: Cli) -> anyhow::Result<()>
where
    V: MyFloat + 'static,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    V::Sampler: Sync,
    for<'de> V: serde::Deserialize<'de>,
{
    let Cli {
        identifier,
        output_dir,
        runtime: runtime_path,
        network: network_config,
        agent: agent_config,
        strategy: strategy_config,
        enable_inhibition,
        overwriting,
        compressing,
    } = args;
    let runtime = DataFormat::read(&runtime_path)?.parse::<RuntimeParams>()?;
    let config = Config::try_new(&network_config, &agent_config, &strategy_config)?;
    let metadata = Metadata::from_iter([
        ("app".to_string(), env!("CARGO_PKG_NAME").to_string()),
        ("version".to_string(), env!("CARGO_PKG_VERSION").to_string()),
        ("runtime".to_string(), runtime_path),
        (
            "agent_config".to_string(),
            agent_config.to_string_lossy().to_string(),
        ),
        (
            "network_config".to_string(),
            network_config.to_string_lossy().to_string(),
        ),
        (
            "strategy_config".to_string(),
            strategy_config.to_string_lossy().to_string(),
        ),
        (
            "enable_inhibition".to_string(),
            enable_inhibition.to_string(),
        ),
        (
            "iteration_count".to_string(),
            runtime.iteration_count.to_string(),
        ),
    ]);
    let writers =
        FileWriters::try_new(&identifier, &output_dir, overwriting, compressing, metadata)?;
    let exec = config.into_exec(enable_inhibition)?;
    run::<V, _, AgentExt<V>, Instance>(writers, &runtime, exec, None).await
}

#[cfg(test)]
mod tests {
    use crate::{start, Cli};

    #[test]
    fn test_start() -> anyhow::Result<()> {
        let args = Cli {
            identifier: "test-start".to_string(),
            output_dir: "./test/result".into(),
            runtime: "./test/runtime.toml".to_string(),
            agent: "./test/agent_config.toml".into(),
            network: "./test/network_config.toml".into(),
            strategy: "./test/strategy_config.toml".into(),
            enable_inhibition: true,
            overwriting: true,
            compressing: true,
        };
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async { start::<f32>(args).await })?;
        Ok(())
    }
}
