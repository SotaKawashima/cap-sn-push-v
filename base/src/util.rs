use std::{fs::File, io};

use graph_lib::{
    io::{DataFormat, ParseBuilder},
    prelude::{DiGraphB, GraphB, UndiGraphB},
};

#[derive(Debug, serde::Deserialize)]
pub struct GraphInfo {
    directed: bool,
    location: DataLocation,
    #[serde(default = "default_transposed")]
    transposed: bool,
}

#[derive(Debug, serde::Deserialize)]
pub enum DataLocation {
    LocalFile(String),
}

pub fn default_transposed() -> bool {
    false
}

impl TryFrom<GraphInfo> for GraphB {
    type Error = io::Error;

    fn try_from(value: GraphInfo) -> Result<Self, Self::Error> {
        match value.location {
            DataLocation::LocalFile(path) => {
                let mut builder = ParseBuilder::new(File::open(path)?, DataFormat::EdgeList);
                if value.directed {
                    if value.transposed {
                        builder = builder.transpose();
                    }
                    Ok(GraphB::Di(builder.parse::<DiGraphB>()?))
                } else {
                    Ok(GraphB::Ud(builder.parse::<UndiGraphB>()?))
                }
            }
        }
    }
}
