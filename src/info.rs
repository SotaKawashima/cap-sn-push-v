use std::fmt::Display;

use subjective_logic::mul::Simplex;

use crate::opinion::{A, PHI, PSI, THETA};

pub struct InfoContent {
    pub psi: Simplex<f32, PSI>,
    pub ppsi: Simplex<f32, PSI>,
    pub pa: Simplex<f32, A>,
    pub phi: Simplex<f32, PHI>,
    pub cond_theta_phi: [Simplex<f32, THETA>; PHI],
}

impl InfoContent {
    pub fn new(
        psi: Simplex<f32, PSI>,
        ppsi: Simplex<f32, PSI>,
        pa: Simplex<f32, A>,
        phi: Simplex<f32, PHI>,
        cond_theta_phi: [Simplex<f32, THETA>; PHI],
    ) -> Self {
        Self {
            psi,
            ppsi,
            pa,
            phi,
            cond_theta_phi,
        }
    }
}

pub struct Info {
    pub id: usize,
    pub info_type: InfoType,
    pub content: InfoContent,
    pub num_shared: usize,
}

impl Info {
    pub fn reset(&mut self) {
        self.num_shared = 0;
    }

    pub fn new(id: usize, info_type: InfoType, content: InfoContent) -> Self {
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

impl From<&InfoType> for InfoContent {
    fn from(value: &InfoType) -> Self {
        match value {
            InfoType::Misinfo => InfoContent::new(
                Simplex::<f32, PSI>::new([0.0, 0.5], 0.5),
                Simplex::<f32, PSI>::new([0.0, 0.0], 1.0),
                Simplex::<f32, A>::new([0.0, 0.0], 1.0),
                Simplex::<f32, PHI>::new([0.0, 0.0], 1.0),
                [
                    Simplex::<f32, THETA>::vacuous(),
                    Simplex::<f32, THETA>::vacuous(),
                ],
            ),
            InfoType::Correction => InfoContent::new(
                Simplex::<f32, PSI>::new([1.0, 0.0], 0.0),
                Simplex::<f32, PSI>::new([0.0, 0.0], 1.0),
                Simplex::<f32, A>::new([0.0, 0.0], 1.0),
                Simplex::<f32, PHI>::new([0.0, 0.0], 1.0),
                [
                    Simplex::<f32, THETA>::vacuous(),
                    Simplex::<f32, THETA>::vacuous(),
                ],
            ),
        }
    }
}

impl From<InfoType> for InfoContent {
    fn from(value: InfoType) -> Self {
        (&value).into()
    }
}

#[cfg(test)]
mod tests {
    use super::InfoType;

    #[test]
    fn test_info_type() {
        println!("{}", InfoType::Misinfo);
    }
}
