use arrow::array::{ArrayRef, BooleanArray, PrimitiveArray, RecordBatch};
use arrow::datatypes::SchemaRef;
use arrow::{
    datatypes::{DataType, Field, Schema},
    ipc::writer::{FileWriter, IpcWriteOptions},
    ipc::CompressionType,
};
use std::collections::HashMap;
use std::fs::File;
use std::marker::PhantomData;
use std::sync::Arc;

use crate::config::Output;
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
    Pop(PopStat),
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

impl StatTrait for InfoStat {
    fn fields() -> Vec<Field> {
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

    fn to_columns(self) -> Vec<ArrayRef> {
        vec![
            Arc::new(PrimitiveArray::from(self.num_par)),
            Arc::new(PrimitiveArray::from(self.num_iter)),
            Arc::new(PrimitiveArray::from(self.t)),
            Arc::new(PrimitiveArray::from(self.info_label)),
            Arc::new(PrimitiveArray::from(self.num_received)),
            Arc::new(PrimitiveArray::from(self.num_shared)),
            Arc::new(PrimitiveArray::from(self.num_viewed)),
            Arc::new(PrimitiveArray::from(self.num_fst_viewed)),
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
}

#[derive(Default)]
pub struct AgentStat {
    num_par: Vec<u32>,
    num_iter: Vec<u32>,
    t: Vec<u32>,
    agent_idx: Vec<u32>,
    selfish: Vec<bool>,
}

impl StatTrait for AgentStat {
    fn fields() -> Vec<Field> {
        vec![
            Field::new("num_par", DataType::UInt32, false),
            Field::new("num_iter", DataType::UInt32, false),
            Field::new("t", DataType::UInt32, false),
            Field::new("agent_idx", DataType::UInt32, false),
            Field::new("selfish", DataType::Boolean, false),
        ]
    }

    fn to_columns(self) -> Vec<ArrayRef> {
        vec![
            Arc::new(PrimitiveArray::from(self.num_par)),
            Arc::new(PrimitiveArray::from(self.num_iter)),
            Arc::new(PrimitiveArray::from(self.t)),
            Arc::new(PrimitiveArray::from(self.agent_idx)),
            Arc::new(BooleanArray::from(self.selfish)),
        ]
    }

    fn label() -> &'static str {
        "agent"
    }
}

impl AgentStat {
    pub fn push_selfish(&mut self, num_par: u32, num_iter: u32, t: u32, agent_idx: usize) {
        self.num_par.push(num_par);
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

#[derive(Default)]
pub struct PopStat {
    num_par: Vec<u32>,
    num_iter: Vec<u32>,
    t: Vec<u32>,
    num_selfish: Vec<u32>,
}

impl StatTrait for PopStat {
    fn fields() -> Vec<Field> {
        vec![
            Field::new("num_par", DataType::UInt32, false),
            Field::new("num_iter", DataType::UInt32, false),
            Field::new("t", DataType::UInt32, false),
            Field::new("num_selfish", DataType::UInt32, false),
        ]
    }

    fn to_columns(self) -> Vec<ArrayRef> {
        vec![
            Arc::new(PrimitiveArray::from(self.num_par)),
            Arc::new(PrimitiveArray::from(self.num_iter)),
            Arc::new(PrimitiveArray::from(self.t)),
            Arc::new(PrimitiveArray::from(self.num_selfish)),
        ]
    }

    fn label() -> &'static str {
        "pop"
    }
}

impl PopStat {
    pub fn push(&mut self, num_par: u32, num_iter: u32, t: u32, d: PopData) {
        self.num_par.push(num_par);
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
    schema: SchemaRef,
    _marker: PhantomData<T>,
}

impl<T: StatTrait> MyWriter<T> {
    fn try_new(
        output: &Output,
        identifier: &str,
        metadata: HashMap<String, String>,
        overwriting: bool,
        compress: bool,
    ) -> anyhow::Result<Self> {
        let output_path = output.location.join(format!(
            "{}.arrow",
            [identifier, T::label(), &output.suffix].join("_")
        ));

        if !overwriting && output_path.exists() {
            panic!(
                "{} already exists. If you want to overwrite it, run with the overwriting option.",
                output_path.display()
            );
        }

        let schema = SchemaRef::new(Schema::new_with_metadata(T::fields(), metadata));
        let writer = FileWriter::try_new_with_options(
            File::create(output_path)?,
            &schema,
            IpcWriteOptions::default().try_with_compression(if compress {
                Some(CompressionType::ZSTD)
            } else {
                None
            })?,
        )?;
        Ok(Self {
            writer,
            schema,
            _marker: PhantomData,
        })
    }

    fn write(&mut self, data: T) -> arrow::error::Result<()> {
        let batch = RecordBatch::try_new(self.schema.clone(), data.to_columns())?;
        self.writer.write(&batch)
    }

    fn finish(&mut self) -> arrow::error::Result<()> {
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
        output: &Output,
        identifier: &str,
        overwriting: bool,
        metadata: HashMap<String, String>,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            info: MyWriter::try_new(
                output,
                identifier,
                metadata.clone(),
                overwriting,
                output.compress,
            )?,
            agent: MyWriter::try_new(
                output,
                identifier,
                metadata.clone(),
                overwriting,
                output.compress,
            )?,
            pop: MyWriter::try_new(output, identifier, metadata, overwriting, output.compress)?,
        })
    }

    pub fn write(&mut self, stat: Stat) -> arrow::error::Result<()> {
        match stat {
            Stat::Info(stat) => self.info.write(stat)?,
            Stat::Agent(stat) => self.agent.write(stat)?,
            Stat::Pop(stat) => self.pop.write(stat)?,
        }
        Ok(())
    }

    pub fn finish(&mut self) -> arrow::error::Result<()> {
        self.info.finish()?;
        self.agent.finish()?;
        self.pop.finish()?;
        Ok(())
    }
}
