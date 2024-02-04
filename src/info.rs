use std::fmt::Display;

use approx::UlpsEq;
use either::Either;
use num_traits::Float;
use rand::Rng;
use rand_distr::{uniform::SampleUniform, Distribution, Open01, Standard};
use subjective_logic::mul::Simplex;

use crate::{
    dist::{sample, Dist},
    opinion::{PHI, PSI, P_A, S, THETA},
};

#[derive(Debug)]
pub struct InfoContent<V: Float> {
    pub psi: Simplex<V, PSI>,
    pub s: Simplex<V, S>,
    pub pa: Simplex<V, P_A>,
    pub phi: Simplex<V, PHI>,
    pub cond_theta_phi: [Simplex<V, THETA>; PHI],
}

impl<V: Float> InfoContent<V> {
    pub fn new(
        psi: Simplex<V, PSI>,
        s: Simplex<V, S>,
        pa: Simplex<V, P_A>,
        phi: Simplex<V, PHI>,
        cond_theta_phi: [Simplex<V, THETA>; PHI],
    ) -> Self {
        Self {
            psi,
            s,
            pa,
            phi,
            cond_theta_phi,
        }
    }
}

#[derive(Debug)]
pub struct Info<V: Float> {
    pub id: usize,
    pub info_type: InfoType,
    pub content: InfoContent<V>,
    pub num_shared: usize,
}

impl<V: Float> Info<V> {
    pub fn reset(&mut self) {
        self.num_shared = 0;
    }

    pub fn new(id: usize, info_type: InfoType, content: InfoContent<V>) -> Self {
        Self {
            id,
            info_type,
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

#[derive(Debug, serde::Deserialize, Clone, PartialEq, PartialOrd, Eq, Ord, Copy)]
pub enum InfoType {
    Misinfo,
    Corrective,
    Observed,
    Inhibitive,
}

impl Display for InfoType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InfoType::Misinfo => write!(f, "misinfo"),
            InfoType::Corrective => write!(f, "correction"),
            InfoType::Observed => write!(f, "observed"),
            InfoType::Inhibitive => write!(f, "inhivitive"),
        }
    }
}

pub trait ToInfoContent<V: Float> {
    fn psi(self) -> Simplex<V, PSI>;
    fn s(self) -> Simplex<V, S>;
    fn pa(self) -> Simplex<V, P_A>;
    fn phi(self) -> Simplex<V, PHI>;
    fn cond_theta_phi(self) -> [Simplex<V, THETA>; PHI];
}

macro_rules! impl_info_content {
    ($ft: ty) => {
        impl ToInfoContent<$ft> for &InfoType {
            fn psi(self) -> Simplex<$ft, PSI> {
                match self {
                    InfoType::Misinfo => Simplex::new([0.0, 0.99], 0.01),
                    InfoType::Corrective => Simplex::new([0.99, 0.0], 0.01),
                    _ => Simplex::vacuous(),
                }
            }
            fn s(self) -> Simplex<$ft, S> {
                match self {
                    InfoType::Corrective => Simplex::new([0.0, 1.0], 0.0),
                    _ => Simplex::vacuous(),
                }
            }
            fn pa(self) -> Simplex<$ft, P_A> {
                Simplex::vacuous()
            }
            fn phi(self) -> Simplex<$ft, PHI> {
                Simplex::vacuous()
            }
            fn cond_theta_phi(self) -> [Simplex<$ft, THETA>; PHI] {
                [Simplex::vacuous(), Simplex::vacuous()]
            }
        }
    };
}

impl<V> From<&InfoType> for InfoContent<V>
where
    for<'a> &'a InfoType: ToInfoContent<V>,
    V: Float + UlpsEq,
{
    fn from(value: &InfoType) -> Self {
        InfoContent::new(
            value.psi(),
            value.s(),
            value.pa(),
            value.phi(),
            value.cond_theta_phi(),
        )
    }
}

impl<V> From<InfoType> for InfoContent<V>
where
    for<'a> &'a InfoType: ToInfoContent<V>,
    V: Float + UlpsEq,
{
    fn from(value: InfoType) -> Self {
        (&value).into()
    }
}
impl_info_content!(f32);
impl_info_content!(f64);

#[cfg(test)]
mod tests {
    use super::InfoType;

    #[test]
    fn test_info_type() {
        println!("{}", InfoType::Misinfo);
    }
}

pub struct TrustDists<V>
where
    V: Float + SampleUniform,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
{
    pub misinfo: Either<V, Dist<V>>,
    pub corrective: Either<V, Dist<V>>,
    pub observed: Either<V, Dist<V>>,
    pub inhibitive: Either<V, Dist<V>>,
}

impl<V> TrustDists<V>
where
    V: Float + SampleUniform,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
{
    pub fn gen_map<R: Rng>(&self, rng: &mut R) -> impl Fn(&Info<V>) -> V {
        let misinfo = sample(&self.misinfo, rng);
        let corrective = sample(&self.corrective, rng);
        let observed = sample(&self.observed, rng);
        let inhibitive = sample(&self.inhibitive, rng);

        move |info: &Info<V>| match info.info_type {
            InfoType::Misinfo => misinfo,
            InfoType::Corrective => corrective,
            InfoType::Observed => observed,
            InfoType::Inhibitive => inhibitive,
        }
    }
}
