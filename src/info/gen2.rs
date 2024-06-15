use serde::Deserialize;
use serde_with::{serde_as, TryFromInto};
use subjective_logic::{mul::labeled::SimplexD1, multi_array::labeled::MArrD3};

use crate::opinion::{MyFloat, Phi, Psi, B, H, O};

#[serde_as]
#[derive(Debug, serde::Deserialize, Clone)]
#[serde(bound(deserialize = "V: Deserialize<'de>"))]
pub enum InfoContent<V: MyFloat> {
    Misinfo {
        #[serde_as(as = "TryFromInto<(Vec<V>, V)>")]
        op: SimplexD1<Psi, V>,
    },
    Correction {
        #[serde_as(as = "TryFromInto<(Vec<V>, V)>")]
        op: SimplexD1<Psi, V>,
        #[serde_as(as = "TryFromInto<(Vec<V>, V)>")]
        misinfo: SimplexD1<Psi, V>,
        trust_misinfo: V,
    },
    Observation {
        #[serde_as(as = "TryFromInto<(Vec<V>, V)>")]
        op: SimplexD1<O, V>,
    },
    Inhibition {
        #[serde_as(as = "TryFromInto<(Vec<V>, V)>")]
        op1: SimplexD1<Phi, V>,
        #[serde_as(as = "TryFromInto<Vec<Vec<Vec<(Vec<V>, V)>>>>")]
        op2: MArrD3<Phi, Psi, B, SimplexD1<H, V>>,
    },
}
