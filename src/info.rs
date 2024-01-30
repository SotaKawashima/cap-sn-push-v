use std::fmt::Display;

use num_traits::Float;
use subjective_logic::mul::Simplex;

use crate::opinion::{A, PHI, PSI, S, THETA};

pub struct InfoContent<V: Float> {
    pub psi: Simplex<V, PSI>,
    pub s: Simplex<V, S>,
    pub pa: Simplex<V, A>,
    pub phi: Simplex<V, PHI>,
    pub cond_theta_phi: [Simplex<V, THETA>; PHI],
}

impl<V: Float> InfoContent<V> {
    pub fn new(
        psi: Simplex<V, PSI>,
        s: Simplex<V, S>,
        pa: Simplex<V, A>,
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
    Correction,
}

impl Display for InfoType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InfoType::Misinfo => write!(f, "misinfo"),
            InfoType::Correction => write!(f, "correction"),
        }
    }
}

macro_rules! impl_info_content {
    ($ft: ty) => {
        impl From<&InfoType> for InfoContent<$ft> {
            fn from(value: &InfoType) -> Self {
                match value {
                    InfoType::Misinfo => InfoContent::new(
                        Simplex::<$ft, PSI>::new([0.0, 0.99], 0.01),
                        Simplex::<$ft, PSI>::new([0.0, 0.0], 1.0),
                        Simplex::<$ft, A>::new([0.0, 0.0], 1.0),
                        Simplex::<$ft, PHI>::new([0.0, 0.0], 1.0),
                        [
                            Simplex::<$ft, THETA>::vacuous(),
                            Simplex::<$ft, THETA>::vacuous(),
                        ],
                    ),
                    InfoType::Correction => InfoContent::new(
                        Simplex::<$ft, PSI>::new([0.99, 0.0], 0.01),
                        Simplex::<$ft, PSI>::new([0.0, 1.0], 0.0),
                        Simplex::<$ft, A>::new([0.0, 0.0], 1.0),
                        Simplex::<$ft, PHI>::new([0.0, 0.0], 1.0),
                        [
                            Simplex::<$ft, THETA>::vacuous(),
                            Simplex::<$ft, THETA>::vacuous(),
                        ],
                    ),
                }
            }
        }

        impl From<InfoType> for InfoContent<$ft> {
            fn from(value: InfoType) -> Self {
                (&value).into()
            }
        }
    };
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
