use std::io;

use clap::Parser;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use v2::{start, Cli};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::Registry::default()
        .with(tracing_subscriber::fmt::Layer::default().with_writer(io::stderr))
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .try_init()?;

    let args = Cli::parse();
    start::<f32>(args).await
}
