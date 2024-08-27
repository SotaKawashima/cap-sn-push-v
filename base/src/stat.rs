use std::{fs::File, marker::PhantomData, path::PathBuf, sync::Arc};

use polars_arrow::{
    array::{ArrayRef, BooleanArray, PrimitiveArray},
    datatypes::{ArrowDataType, ArrowSchema, Field, Metadata},
    io::ipc::write::{stream_async::WriteOptions, FileWriter},
    legacy::error::PolarsResult,
    record_batch::RecordBatch,
};

use crate::info::InfoLabel;

#[derive(Default)]
pub struct InfoData {
    num_posted: u32,
    num_received: u32,
    num_shared: u32,
    num_viewed: u32,
    num_fst_viewed: u32,
}

impl InfoData {
    pub fn posted(&mut self) {
        self.num_posted += 1;
    }

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

#[derive(Debug)]
pub enum Stat {
    Info(InfoStat),
    Agent(AgentStat),
    Pop(PopStat),
}

#[derive(Default, Debug)]
pub struct InfoStat {
    num_iter: Vec<u32>,
    t: Vec<u32>,
    info_label: Vec<u8>,
    num_posted: Vec<u32>,
    num_received: Vec<u32>,
    num_shared: Vec<u32>,
    num_viewed: Vec<u32>,
    num_fst_viewed: Vec<u32>,
}

impl StatTrait for InfoStat {
    fn fields() -> Vec<Field> {
        vec![
            Field::new("num_iter", ArrowDataType::UInt32, false),
            Field::new("t", ArrowDataType::UInt32, false),
            Field::new("info_label", ArrowDataType::UInt8, false),
            Field::new("num_posted", ArrowDataType::UInt32, false),
            Field::new("num_received", ArrowDataType::UInt32, false),
            Field::new("num_shared", ArrowDataType::UInt32, false),
            Field::new("num_viewed", ArrowDataType::UInt32, false),
            Field::new("num_fst_viewed", ArrowDataType::UInt32, false),
        ]
    }

    fn to_columns(self) -> Vec<ArrayRef> {
        vec![
            Box::new(PrimitiveArray::from_vec(self.num_iter)),
            Box::new(PrimitiveArray::from_vec(self.t)),
            Box::new(PrimitiveArray::from_vec(self.info_label)),
            Box::new(PrimitiveArray::from_vec(self.num_posted)),
            Box::new(PrimitiveArray::from_vec(self.num_received)),
            Box::new(PrimitiveArray::from_vec(self.num_shared)),
            Box::new(PrimitiveArray::from_vec(self.num_viewed)),
            Box::new(PrimitiveArray::from_vec(self.num_fst_viewed)),
        ]
    }

    fn label() -> &'static str {
        "info"
    }
}

impl From<InfoStat> for Stat {
    fn from(value: InfoStat) -> Self {
        Self::Info(value)
    }
}

impl InfoStat {
    pub fn push(&mut self, num_iter: u32, t: u32, d: &InfoData, label: &InfoLabel) {
        self.num_iter.push(num_iter);
        self.t.push(t);
        self.info_label.push(label.into());
        self.num_posted.push(d.num_posted);
        self.num_received.push(d.num_received);
        self.num_shared.push(d.num_shared);
        self.num_viewed.push(d.num_viewed);
        self.num_fst_viewed.push(d.num_fst_viewed);
    }
}

#[derive(Default, Debug)]
pub struct AgentStat {
    num_iter: Vec<u32>,
    t: Vec<u32>,
    agent_idx: Vec<u32>,
    selfish: Vec<bool>,
}

impl StatTrait for AgentStat {
    fn fields() -> Vec<Field> {
        vec![
            Field::new("num_iter", ArrowDataType::UInt32, false),
            Field::new("t", ArrowDataType::UInt32, false),
            Field::new("agent_idx", ArrowDataType::UInt32, false),
            Field::new("selfish", ArrowDataType::Boolean, false),
        ]
    }

    fn to_columns(self) -> Vec<ArrayRef> {
        vec![
            Box::new(PrimitiveArray::from_vec(self.num_iter)),
            Box::new(PrimitiveArray::from_vec(self.t)),
            Box::new(PrimitiveArray::from_vec(self.agent_idx)),
            Box::new(BooleanArray::from_slice(&self.selfish)),
        ]
    }

    fn label() -> &'static str {
        "agent"
    }
}

impl AgentStat {
    pub fn push_selfish(&mut self, num_iter: u32, t: u32, agent_idx: usize) {
        self.num_iter.push(num_iter);
        self.t.push(t);
        self.agent_idx.push(agent_idx as u32);
        self.selfish.push(true);
    }
}

impl From<AgentStat> for Stat {
    fn from(value: AgentStat) -> Self {
        Self::Agent(value)
    }
}

#[derive(Default)]
pub struct PopData {
    pub num_selfish: u32,
}

impl PopData {
    pub fn selfish(&mut self) {
        self.num_selfish += 1;
    }
}

#[derive(Default, Debug)]
pub struct PopStat {
    num_iter: Vec<u32>,
    t: Vec<u32>,
    num_selfish: Vec<u32>,
}

impl StatTrait for PopStat {
    fn fields() -> Vec<Field> {
        vec![
            Field::new("num_iter", ArrowDataType::UInt32, false),
            Field::new("t", ArrowDataType::UInt32, false),
            Field::new("num_selfish", ArrowDataType::UInt32, false),
        ]
    }

    fn to_columns(self) -> Vec<ArrayRef> {
        vec![
            Box::new(PrimitiveArray::from_vec(self.num_iter)),
            Box::new(PrimitiveArray::from_vec(self.t)),
            Box::new(PrimitiveArray::from_vec(self.num_selfish)),
        ]
    }

    fn label() -> &'static str {
        "pop"
    }
}

impl PopStat {
    pub fn push(&mut self, num_iter: u32, t: u32, d: PopData) {
        self.num_iter.push(num_iter);
        self.t.push(t);
        self.num_selfish.push(d.num_selfish);
    }
}

impl From<PopStat> for Stat {
    fn from(value: PopStat) -> Self {
        Self::Pop(value)
    }
}

pub trait StatTrait {
    fn label() -> &'static str;
    fn fields() -> Vec<Field>;
    fn to_columns(self) -> Vec<ArrayRef>;
}

struct MyWriter<T> {
    writer: FileWriter<File>,
    _marker: PhantomData<T>,
}

impl<T: StatTrait> MyWriter<T> {
    fn try_new(
        output_dir: &PathBuf,
        identifier: &str,
        metadata: Metadata,
        overwriting: bool,
        compress: bool,
    ) -> anyhow::Result<Self> {
        let output_path = output_dir.join(format!("{identifier}_{}.arrow", T::label()));
        if !overwriting && output_path.exists() {
            panic!(
                "{} already exists. If you want to overwrite it, run with the overwriting option.",
                output_path.display()
            );
        }

        let schema = Arc::new(ArrowSchema::from(T::fields()).with_metadata(metadata));
        let writer = FileWriter::try_new(
            File::create(output_path)?,
            schema,
            None,
            WriteOptions {
                compression: if compress {
                    Some(polars_arrow::io::ipc::write::Compression::ZSTD)
                } else {
                    None
                },
            },
        )?;
        Ok(Self {
            writer,
            _marker: PhantomData,
        })
    }

    fn write(&mut self, data: T) -> PolarsResult<()> {
        let batch = RecordBatch::try_new(data.to_columns())?;
        self.writer.write(&batch, None)
    }

    fn finish(&mut self) -> PolarsResult<()> {
        self.writer.finish()
    }
}

pub struct FileWriters {
    info: MyWriter<InfoStat>,
    agent: MyWriter<AgentStat>,
    pop: MyWriter<PopStat>,
}

impl FileWriters {
    pub fn try_new(
        identifier: &str,
        output_dir: &PathBuf,
        overwriting: bool,
        compressing: bool,
        metadata: Metadata,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            info: MyWriter::try_new(
                output_dir,
                identifier,
                metadata.clone(),
                overwriting,
                compressing,
            )?,
            agent: MyWriter::try_new(
                output_dir,
                identifier,
                metadata.clone(),
                overwriting,
                compressing,
            )?,
            pop: MyWriter::try_new(output_dir, identifier, metadata, overwriting, compressing)?,
        })
    }

    pub fn write(&mut self, stat: Stat) -> PolarsResult<()> {
        match stat {
            Stat::Info(stat) => self.info.write(stat)?,
            Stat::Agent(stat) => self.agent.write(stat)?,
            Stat::Pop(stat) => self.pop.write(stat)?,
        }
        Ok(())
    }

    pub fn finish(&mut self) -> PolarsResult<()> {
        self.info.finish()?;
        self.agent.finish()?;
        self.pop.finish()?;
        Ok(())
    }
}
