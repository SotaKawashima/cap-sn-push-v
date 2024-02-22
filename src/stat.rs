use arrow2::array::{Array, BooleanArray, PrimitiveArray};
use arrow2::chunk::Chunk;
use arrow2::datatypes::Schema;
use arrow2::datatypes::{DataType, Field};
use arrow2::io::ipc::write::{Compression, FileWriter, WriteOptions};
use std::collections::BTreeMap;
use std::fs::File;
use std::path::PathBuf;

use crate::config::{Output, Runtime};
use crate::info::InfoLabel;

#[derive(Default)]
pub struct InfoData {
    num_received: u32,
    num_shared: u32,
    num_viewed: u32,
    num_fst_viewed: u32,
}

impl InfoData {
    pub fn received(&mut self) {
        self.num_received += 1;
    }

    pub fn shared(&mut self) {
        self.num_shared += 1;
    }

    pub fn first_viewed(&mut self) {
        self.num_fst_viewed += 1;
    }

    pub fn viewed(&mut self) {
        self.num_viewed += 1;
    }
}

pub enum Stat {
    Info(InfoStat),
    Agent(AgentStat),
}

#[derive(Default)]
pub struct InfoStat {
    num_par: Vec<u32>,
    num_iter: Vec<u32>,
    t: Vec<u32>,
    info_label: Vec<u8>,
    num_received: Vec<u32>,
    num_shared: Vec<u32>,
    num_viewed: Vec<u32>,
    num_fst_viewed: Vec<u32>,
}

impl TryFrom<&InfoStat> for Chunk<Box<dyn Array>> {
    type Error = arrow2::error::Error;

    fn try_from(value: &InfoStat) -> Result<Self, Self::Error> {
        Chunk::try_new(vec![
            PrimitiveArray::from_slice(&value.num_par).boxed(),
            PrimitiveArray::from_slice(&value.num_iter).boxed(),
            PrimitiveArray::from_slice(&value.t).boxed(),
            PrimitiveArray::from_slice(&value.info_label).boxed(),
            PrimitiveArray::from_slice(&value.num_received).boxed(),
            PrimitiveArray::from_slice(&value.num_shared).boxed(),
            PrimitiveArray::from_slice(&value.num_viewed).boxed(),
            PrimitiveArray::from_slice(&value.num_fst_viewed).boxed(),
        ])
    }
}

impl From<InfoStat> for Stat {
    fn from(value: InfoStat) -> Self {
        Self::Info(value)
    }
}

impl InfoStat {
    pub fn push(&mut self, num_par: u32, num_iter: u32, t: u32, d: &InfoData, label: &InfoLabel) {
        self.num_par.push(num_par);
        self.num_iter.push(num_iter);
        self.t.push(t);
        self.info_label.push(label.into());
        self.num_received.push(d.num_received);
        self.num_shared.push(d.num_shared);
        self.num_viewed.push(d.num_viewed);
        self.num_fst_viewed.push(d.num_fst_viewed);
    }

    fn get_fields() -> Vec<Field> {
        vec![
            Field::new("num_par", DataType::UInt32, false),
            Field::new("num_iter", DataType::UInt32, false),
            Field::new("t", DataType::UInt32, false),
            Field::new("info_label", DataType::UInt8, false),
            Field::new("num_received", DataType::UInt32, false),
            Field::new("num_shared", DataType::UInt32, false),
            Field::new("num_viewed", DataType::UInt32, false),
            Field::new("num_fst_viewed", DataType::UInt32, false),
        ]
    }

    fn output_path(output: &Output, identifier: &str) -> PathBuf {
        output.location.join(format!(
            "{}.arrow",
            [&identifier, "info", &output.suffix].join("_")
        ))
    }
}

#[derive(Default)]
pub struct AgentStat {
    num_par: Vec<u32>,
    num_iter: Vec<u32>,
    t: Vec<u32>,
    agent_idx: Vec<u32>,
    selfish: Vec<bool>,
}

impl AgentStat {
    pub fn push_selfish(&mut self, num_par: u32, num_iter: u32, t: u32, agent_idx: usize) {
        self.num_par.push(num_par);
        self.num_iter.push(num_iter);
        self.t.push(t);
        self.agent_idx.push(agent_idx as u32);
        self.selfish.push(true);
    }

    fn get_fields() -> Vec<Field> {
        vec![
            Field::new("num_par", DataType::UInt32, false),
            Field::new("num_iter", DataType::UInt32, false),
            Field::new("t", DataType::UInt32, false),
            Field::new("agent_idx", DataType::UInt8, false),
            Field::new("selfish", DataType::Boolean, false),
        ]
    }

    fn output_path(output: &Output, identifier: &str) -> PathBuf {
        output.location.join(format!(
            "{}.arrow",
            [&identifier, "agent", &output.suffix].join("_")
        ))
    }
}

impl TryFrom<&AgentStat> for Chunk<Box<dyn Array>> {
    type Error = arrow2::error::Error;

    fn try_from(value: &AgentStat) -> Result<Self, Self::Error> {
        Chunk::try_new(vec![
            PrimitiveArray::from_slice(&value.num_par).boxed(),
            PrimitiveArray::from_slice(&value.num_iter).boxed(),
            PrimitiveArray::from_slice(&value.t).boxed(),
            PrimitiveArray::from_slice(&value.agent_idx).boxed(),
            BooleanArray::from_slice(&value.selfish).boxed(),
        ])
    }
}

impl From<AgentStat> for Stat {
    fn from(value: AgentStat) -> Self {
        Self::Agent(value)
    }
}

pub struct FileWriters {
    info: FileWriter<File>,
    agent: FileWriter<File>,
}

impl FileWriters {
    pub fn try_new(
        output: &Output,
        runtime: &Runtime,
        identifier: &str,
        overwriting: bool,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            info: create_writer(
                InfoStat::output_path(output, identifier),
                overwriting,
                output.compress,
                InfoStat::get_fields(),
                create_metadata(runtime),
            )?,
            agent: create_writer(
                AgentStat::output_path(output, identifier),
                overwriting,
                output.compress,
                AgentStat::get_fields(),
                create_metadata(runtime),
            )?,
        })
    }

    pub fn write(&mut self, stat: Stat) -> arrow2::error::Result<()> {
        match stat {
            Stat::Info(ref stat) => self.info.write(&stat.try_into()?, None)?,
            Stat::Agent(ref stat) => self.agent.write(&stat.try_into()?, None)?,
        }
        Ok(())
    }

    pub fn finish(&mut self) -> arrow2::error::Result<()> {
        self.info.finish()?;
        self.agent.finish()?;
        Ok(())
    }
}

fn create_metadata(runtime: &Runtime) -> BTreeMap<String, String> {
    BTreeMap::from_iter([
        ("version".to_string(), env!("CARGO_PKG_VERSION").to_string()),
        ("num_parallel".to_string(), runtime.num_parallel.to_string()),
        (
            "iteration_count".to_string(),
            runtime.iteration_count.to_string(),
        ),
    ])
}

fn create_writer(
    output_path: PathBuf,
    overwriting: bool,
    compress: bool,
    fields: Vec<Field>,
    metadata: BTreeMap<String, String>,
) -> anyhow::Result<FileWriter<File>> {
    if !overwriting && output_path.exists() {
        panic!(
            "{} already exists. If you want to overwrite it, run with the overwriting option.",
            output_path.display()
        );
    }

    let writer = File::create(output_path)?;
    let compression: Option<Compression> = if compress {
        Some(Compression::ZSTD)
    } else {
        None
    };
    Ok(FileWriter::try_new(
        writer,
        Schema { fields, metadata },
        None,
        WriteOptions { compression },
    )?)
}
