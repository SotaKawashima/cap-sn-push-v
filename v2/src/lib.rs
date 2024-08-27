mod config;
mod exec;

use std::path::PathBuf;

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

#[derive(clap::Parser)]
pub struct Cli {
    /// string to identify given configuration data
    identifier: String,
    /// the path of output files
    output_dir: PathBuf,
    /// the path of a runtime config file
    #[arg(long)]
    runtime: String,
    /// the path of a agent parameters config file
    #[arg(long)]
    config: PathBuf,
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
        config: config_path,
        overwriting,
        compressing,
    } = args;
    let runtime = DataFormat::read(&runtime_path)?.parse::<RuntimeParams>()?;
    let config: Config = DataFormat::read(&config_path)?.parse()?;
    let metadata = Metadata::from_iter([
        ("app".to_string(), env!("CARGO_PKG_NAME").to_string()),
        ("version".to_string(), env!("CARGO_PKG_VERSION").to_string()),
        ("runtime".to_string(), runtime_path),
        (
            "config".to_string(),
            config_path.to_string_lossy().to_string(),
        ),
        (
            "iteration_count".to_string(),
            runtime.iteration_count.to_string(),
        ),
    ]);
    let writers =
        FileWriters::try_new(&identifier, &output_dir, overwriting, compressing, metadata)?;
    let exec = config.into_exec(&config_path)?;
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
            config: "./test/config.toml".into(),
            overwriting: true,
            compressing: true,
        };
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async { start::<f32>(args).await })?;
        Ok(())
    }
}
