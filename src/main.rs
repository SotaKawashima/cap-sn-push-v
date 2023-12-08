use std::fs::File;

use cap_sn::Executor;
use clap::Parser;

#[derive(Parser)]
struct Args {
    input: String,
    output: String,
    strategy: String,
    seed_state: u64,
    iteration_count: u32,
    /// Enable zstd compression
    #[arg(short, long, default_value_t = false)]
    out_compression: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let reader = File::open(args.input)?;
    let mut writer = File::create(args.output)?;
    let mut executor = Executor::try_new(reader, &args.strategy, args.seed_state)?;
    executor.exec(args.iteration_count, &mut writer, args.out_compression)
}
