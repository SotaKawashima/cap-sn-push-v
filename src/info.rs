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

#[derive(Debug)]
pub struct InfoContent<V: Float> {
    pub label: InfoLabel,
    pub psi: Simplex<V, PSI>,
    pub s: Simplex<V, S>,
    pub o: Simplex<V, O>,
    pub phi: Simplex<V, PHI>,
}

impl<V: Float> InfoContent<V> {
    pub fn new(
        label: InfoLabel,
        psi: Simplex<V, PSI>,
        s: Simplex<V, S>,
        o: Simplex<V, O>,
        phi: Simplex<V, PHI>,
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

#[derive(Debug)]
pub struct Info<'a, V: Float> {
    pub idx: usize,
    num_shared: usize,
    num_viewed: usize,
    pub content: &'a InfoContent<V>,
}

impl<'a, V: Float> Info<'a, V> {
    pub fn new(idx: usize, content: &'a InfoContent<V>) -> Self {
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
#[derive(Debug, serde::Deserialize)]
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

impl<V> From<InfoObject<V>> for InfoContent<V>
where
    V: Float + UlpsEq + AddAssign,
{
    fn from(value: InfoObject<V>) -> Self {
        match value {
            InfoObject::Misinfo { psi } => InfoContent::new(
                InfoLabel::Misinfo,
                psi,
                Simplex::vacuous(),
                Simplex::vacuous(),
                Simplex::vacuous(),
            ),
            InfoObject::Corrective { psi, s } => InfoContent::new(
                InfoLabel::Corrective,
                psi,
                s,
                Simplex::vacuous(),
                Simplex::vacuous(),
            ),
            InfoObject::Observed { o } => InfoContent::new(
                InfoLabel::Observed,
                Simplex::vacuous(),
                Simplex::vacuous(),
                o,
                Simplex::vacuous(),
            ),
            InfoObject::Inhibitive {
                phi,
                // cond_theta_phi,
            } => InfoContent::new(
                InfoLabel::Inhibitive,
                Simplex::vacuous(),
                Simplex::vacuous(),
                Simplex::vacuous(),
                phi,
            ),
        }
    }
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

        move |info: &Info<V>| match info.content.label {
            InfoLabel::Misinfo => misinfo,
            InfoLabel::Corrective => corrective,
            InfoLabel::Observed => observed,
            InfoLabel::Inhibitive => inhibitive,
        }
    }
}
