use std::{fmt::Display, ops::AddAssign};

use approx::UlpsEq;
use num_traits::Float;
use rand::Rng;
use rand_distr::{uniform::SampleUniform, Distribution, Open01, Standard};
use serde::Deserialize;
use serde_with::{serde_as, TryFromInto};
use subjective_logic::mul::Simplex;

use crate::{
    opinion::{O, PHI, PSI, S},
    value::{EValue, EValueParam},
};

#[derive(Debug, Clone)]
pub struct InfoContent<'a, V: Float> {
    pub label: InfoLabel,
    pub psi: &'a Simplex<V, PSI>,
    pub s: &'a Simplex<V, S>,
    pub o: &'a Simplex<V, O>,
    pub phi: &'a Simplex<V, PHI>,
}

impl<'a, V: Float> InfoContent<'a, V> {
    pub fn new(
        label: InfoLabel,
        psi: &'a Simplex<V, PSI>,
        s: &'a Simplex<V, S>,
        o: &'a Simplex<V, O>,
        phi: &'a Simplex<V, PHI>,
    ) -> Self {
        Self {
            label,
            psi,
            s,
            o,
            phi,
        }
    }
}

#[serde_as]
#[derive(serde::Deserialize, Debug)]
#[serde(bound(deserialize = "V: Deserialize<'de>"))]
pub enum InfoParam<V>
where
    V: Float + UlpsEq + AddAssign,
{
    Idx(usize),
    Obj(InfoObject<V>),
}

#[derive(Debug)]
pub struct Info<'a, V: Float> {
    pub idx: usize,
    num_shared: usize,
    num_viewed: usize,
    pub content: InfoContent<'a, V>,
}

impl<'a, V: Float> Info<'a, V> {
    fn new(idx: usize, content: InfoContent<'a, V>) -> Self {
        Self {
            idx,
            content,
            num_shared: Default::default(),
            num_viewed: Default::default(),
        }
    }

    #[inline]
    pub fn viewed(&mut self) {
        self.num_viewed += 1;
    }

    #[inline]
    pub fn shared(&mut self) {
        self.num_shared += 1;
    }

    #[inline]
    pub fn num_shared(&self) -> usize {
        self.num_shared
    }
}

#[serde_as]
#[derive(Debug, serde::Deserialize, Clone)]
#[serde(bound(deserialize = "V: Deserialize<'de>"))]
pub enum InfoObject<V>
where
    V: Float + UlpsEq + AddAssign,
{
    Misinfo {
        #[serde_as(as = "TryFromInto<([V; PSI], V)>")]
        psi: Simplex<V, PSI>,
    },
    Corrective {
        #[serde_as(as = "TryFromInto<([V; PSI], V)>")]
        psi: Simplex<V, PSI>,
        #[serde_as(as = "TryFromInto<([V; S], V)>")]
        s: Simplex<V, S>,
    },
    Observed {
        #[serde_as(as = "TryFromInto<([V; O], V)>")]
        o: Simplex<V, O>,
    },
    Inhibitive {
        #[serde_as(as = "TryFromInto<([V; PHI], V)>")]
        phi: Simplex<V, PSI>,
    },
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Copy)]
pub enum InfoLabel {
    Misinfo,
    Corrective,
    Observed,
    Inhibitive,
}

impl From<&InfoLabel> for u8 {
    fn from(value: &InfoLabel) -> Self {
        match value {
            InfoLabel::Misinfo => 0,
            InfoLabel::Corrective => 1,
            InfoLabel::Observed => 2,
            InfoLabel::Inhibitive => 3,
        }
    }
}

impl Display for InfoLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InfoLabel::Misinfo => write!(f, "misinfo"),
            InfoLabel::Corrective => write!(f, "corrective"),
            InfoLabel::Observed => write!(f, "observed"),
            InfoLabel::Inhibitive => write!(f, "inhivitive"),
        }
    }
}

pub struct InfoBuilder<V: Float> {
    vacuous_psi: Simplex<V, PSI>,
    vacuous_s: Simplex<V, S>,
    vacuous_o: Simplex<V, O>,
    vacuous_phi: Simplex<V, PHI>,
}

impl<V> InfoBuilder<V>
where
    V: Float + UlpsEq + AddAssign,
{
    pub fn new() -> Self {
        Self {
            vacuous_psi: Simplex::vacuous(),
            vacuous_s: Simplex::vacuous(),
            vacuous_o: Simplex::vacuous(),
            vacuous_phi: Simplex::vacuous(),
        }
    }

    pub fn build<'a>(&'a self, idx: usize, obj: &'a InfoObject<V>) -> Info<'a, V> {
        let content = match obj {
            InfoObject::Misinfo { psi } => InfoContent::new(
                InfoLabel::Misinfo,
                psi,
                &self.vacuous_s,
                &self.vacuous_o,
                &self.vacuous_phi,
            ),
            InfoObject::Corrective { psi, s } => InfoContent::new(
                InfoLabel::Corrective,
                psi,
                s,
                &self.vacuous_o,
                &self.vacuous_phi,
            ),
            InfoObject::Observed { o } => InfoContent::new(
                InfoLabel::Observed,
                &self.vacuous_psi,
                &self.vacuous_s,
                o,
                &self.vacuous_phi,
            ),
            InfoObject::Inhibitive {
                phi,
                // cond_theta_phi,
            } => InfoContent::new(
                InfoLabel::Inhibitive,
                &self.vacuous_psi,
                &self.vacuous_s,
                &self.vacuous_o,
                phi,
            ),
        };
        Info::new(idx, content)
    }

    // fn merge(&self, a)
}

#[cfg(test)]
mod tests {
    use super::InfoLabel;

    #[test]
    fn test_info_type() {
        println!("{}", InfoLabel::Misinfo);
    }
}

#[serde_as]
#[derive(Debug, serde::Deserialize)]
#[serde(bound(deserialize = "V: serde::Deserialize<'de>"))]
pub struct TrustParams<V>
where
    V: Float,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
{
    #[serde_as(as = "TryFromInto<EValueParam<V>>")]
    pub misinfo: EValue<V>,
    #[serde_as(as = "TryFromInto<EValueParam<V>>")]
    pub corrective: EValue<V>,
    #[serde_as(as = "TryFromInto<EValueParam<V>>")]
    pub observed: EValue<V>,
    #[serde_as(as = "TryFromInto<EValueParam<V>>")]
    pub inhibitive: EValue<V>,
}

impl<V> TrustParams<V>
where
    V: Float + SampleUniform,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
{
    pub fn gen_map<R: Rng>(&self, rng: &mut R) -> impl Fn(&Info<V>) -> V
    where
        V: SampleUniform,
        Open01: Distribution<V>,
        Standard: Distribution<V>,
    {
        let misinfo = self.misinfo.sample(rng);
        let corrective = self.corrective.sample(rng);
        let observed = self.observed.sample(rng);
        let inhibitive = self.inhibitive.sample(rng);

        move |info: &Info<V>| match &info.content.label {
            InfoLabel::Misinfo => misinfo,
            InfoLabel::Corrective => corrective,
            InfoLabel::Observed => observed,
            InfoLabel::Inhibitive => inhibitive,
        }
    }
}
