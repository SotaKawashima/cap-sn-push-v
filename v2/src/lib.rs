mod config;
mod exec;
mod io;

use base::{
    opinion::MyFloat,
    runner::{run, RuntimeParams},
    stat::FileWriters,
};
use config::Config;
use exec::{AgentExt, Instance};
use input::format::DataFormat;
use io::MyPath;
use polars_arrow::datatypes::Metadata;
use rand_distr::{uniform::SampleUniform, Distribution, Exp1, Open01, Standard, StandardNormal};
use std::path::PathBuf;

#[derive(clap::Parser)]
pub struct Cli {
    /// string to identify given configuration data
    identifier: String,
    /// the path of output files
    output_dir: PathBuf,
    /// the path of a runtime config file
    #[arg(long)]
    runtime: MyPath,
    /// the path of a network config file
    #[arg(long)]
    network: MyPath,
    /// the path of a agent config file
    #[arg(long)]
    agent: MyPath,
    /// the path of a strategy config file
    #[arg(long)]
    strategy: MyPath,
    /// Enable inhibition information
    #[arg(short, default_value_t = false)]
    enable_inhibition: bool,
    /// delayed steps of selfish from receiving information
    #[arg(short, default_value_t = 0)]
    delay_selfish: u32,
    /// Enable overwriting of a output file
    #[arg(short, default_value_t = false)]
    overwriting: bool,
    /// Compress a output file
    #[arg(short, default_value_t = false)]
    compressing: bool,
}

pub async fn start<V>(args: Cli) -> anyhow::Result<()>
where
    V: MyFloat + SampleUniform + 'static,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    V::Sampler: Sync + Send,
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
        delay_selfish,
        overwriting,
        compressing,
    } = args;
    let runtime = DataFormat::read(runtime_path.verify()?)?.parse::<RuntimeParams>()?;
    let config = Config::try_new(
        network_config.verify()?,
        agent_config.verify()?,
        strategy_config.verify()?,
    )?;
    let metadata = Metadata::from_iter([
        ("app".to_string(), env!("CARGO_PKG_NAME").to_string()),
        ("version".to_string(), env!("CARGO_PKG_VERSION").to_string()),
        (
            "runtime".to_string(),
            runtime_path.to_string_lossy().to_string(),
        ),
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
        ("delay_selfish".to_string(), delay_selfish.to_string()),
        (
            "iteration_count".to_string(),
            runtime.iteration_count.to_string(),
        ),
    ]);
    let writers =
        FileWriters::try_new(&identifier, &output_dir, overwriting, compressing, metadata)?;
    let exec = config.into_exec(enable_inhibition, delay_selfish)?;
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
            runtime: "./test/runtime.toml".into(),
            agent: "./test/agent_config.toml".into(),
            network: "./test/network_config.toml".into(),
            strategy: "./test/strategy_config.toml".into(),
            enable_inhibition: true,
            delay_selfish: 0,
            overwriting: true,
            compressing: true,
        };
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async { start::<f32>(args).await })?;
        Ok(())
    }
}
