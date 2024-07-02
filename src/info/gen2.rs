use serde::Deserialize;
use serde_with::{serde_as, TryFromInto};
use subjective_logic::{
    mul::labeled::{OpinionD1, SimplexD1},
    multi_array::labeled::MArrD1,
};

use crate::opinion::{MyFloat, Phi, Psi, B, H, O};

use super::InfoLabel;

#[serde_as]
#[derive(Debug, serde::Deserialize, Clone)]
#[serde(bound(deserialize = "V: Deserialize<'de>"))]
pub enum InfoContent<V: MyFloat> {
    Misinfo {
        #[serde_as(as = "TryFromInto<(Vec<V>, V, Vec<V>)>")]
        op: OpinionD1<Psi, V>,
    },
    Correction {
        #[serde_as(as = "TryFromInto<(Vec<V>, V, Vec<V>)>")]
        op: OpinionD1<Psi, V>,
        #[serde_as(as = "TryFromInto<(Vec<V>, V, Vec<V>)>")]
        misinfo: OpinionD1<Psi, V>,
    },
    Observation {
        #[serde_as(as = "TryFromInto<(Vec<V>, V, Vec<V>)>")]
        op: OpinionD1<O, V>,
    },
    Inhibition {
        #[serde_as(as = "TryFromInto<(Vec<V>, V, Vec<V>)>")]
        op1: OpinionD1<Phi, V>,
        #[serde_as(as = "TryFromInto<Vec<(Vec<V>, V)>>")]
        op2: MArrD1<Psi, SimplexD1<H, V>>,
        #[serde_as(as = "TryFromInto<Vec<(Vec<V>, V)>>")]
        op3: MArrD1<B, SimplexD1<H, V>>,
    },
}

impl<V: MyFloat> From<&InfoContent<V>> for InfoLabel {
    fn from(value: &InfoContent<V>) -> Self {
        match value {
            InfoContent::Misinfo { .. } => InfoLabel::Misinfo,
            InfoContent::Correction { .. } => InfoLabel::Corrective,
            InfoContent::Observation { .. } => InfoLabel::Observed,
            InfoContent::Inhibition { .. } => InfoLabel::Inhibitive,
        }
    }
}
