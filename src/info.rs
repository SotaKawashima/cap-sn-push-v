use std::{fmt::Display, ops::AddAssign};

use approx::UlpsEq;
use num_traits::Float;
use rand::Rng;
use rand_distr::{uniform::SampleUniform, Distribution, Open01, Standard};
use serde::Deserialize;
use serde_with::{serde_as, TryFromInto};
use subjective_logic::mul::Simplex;

use crate::{
    opinion::{PHI, PSI, P_A, S, THETA},
    value::{DistValue, ParamValue},
};

#[derive(Debug)]
pub struct InfoContent<V: Float> {
    pub label: InfoLabel,
    pub psi: Simplex<V, PSI>,
    pub s: Simplex<V, S>,
    pub pa: Simplex<V, P_A>,
    pub phi: Simplex<V, PHI>,
    pub cond_theta_phi: [Simplex<V, THETA>; PHI],
}

impl<V: Float> InfoContent<V> {
    pub fn new(
        label: InfoLabel,
        psi: Simplex<V, PSI>,
        s: Simplex<V, S>,
        pa: Simplex<V, P_A>,
        phi: Simplex<V, PHI>,
        cond_theta_phi: [Simplex<V, THETA>; PHI],
    ) -> Self {
        Self {
            label,
            psi,
            s,
            pa,
            phi,
            cond_theta_phi,
        }
    }
}

#[derive(Debug)]
pub struct Info<'a, V: Float> {
    pub idx: usize,
    pub num_shared: usize,
    pub content: &'a InfoContent<V>,
}

impl<'a, V: Float> Info<'a, V> {
    pub fn reset(&mut self) {
        self.num_shared = 0;
    }

    pub fn new(idx: usize, content: &'a InfoContent<V>) -> Self {
        Self {
            idx,
            content,
            num_shared: Default::default(),
        }
    }

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
        #[serde_as(as = "TryFromInto<([V; P_A], V)>")]
        pa: Simplex<V, P_A>,
    },
    Inhibitive {
        #[serde_as(as = "TryFromInto<([V; PHI], V)>")]
        phi: Simplex<V, PSI>,
        #[serde_as(as = "[TryFromInto<([V; THETA], V)>; 2]")]
        cond_theta_phi: [Simplex<V, THETA>; PHI],
    },
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Copy)]
pub enum InfoLabel {
    Misinfo,
    Corrective,
    Observed,
    Inhibitive,
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
                [Simplex::vacuous(), Simplex::vacuous()],
            ),
            InfoObject::Corrective { psi, s } => InfoContent::new(
                InfoLabel::Corrective,
                psi,
                s,
                Simplex::vacuous(),
                Simplex::vacuous(),
                [Simplex::vacuous(), Simplex::vacuous()],
            ),
            InfoObject::Observed { pa } => InfoContent::new(
                InfoLabel::Observed,
                Simplex::vacuous(),
                Simplex::vacuous(),
                pa,
                Simplex::vacuous(),
                [Simplex::vacuous(), Simplex::vacuous()],
            ),
            InfoObject::Inhibitive {
                phi,
                cond_theta_phi,
            } => InfoContent::new(
                InfoLabel::Inhibitive,
                Simplex::vacuous(),
                Simplex::vacuous(),
                Simplex::vacuous(),
                phi,
                cond_theta_phi,
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
    #[serde_as(as = "TryFromInto<ParamValue<V>>")]
    pub misinfo: DistValue<V>,
    #[serde_as(as = "TryFromInto<ParamValue<V>>")]
    pub corrective: DistValue<V>,
    #[serde_as(as = "TryFromInto<ParamValue<V>>")]
    pub observed: DistValue<V>,
    #[serde_as(as = "TryFromInto<ParamValue<V>>")]
    pub inhibitive: DistValue<V>,
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
