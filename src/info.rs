use subjective_logic::mul::{Opinion1d, Simplex};

use crate::opinion::{A, PHI, PSI, THETA};

pub struct InfoContent {
    pub psi: Opinion1d<f32, PSI>,
    pub ppsi: Opinion1d<f32, PSI>,
    pub pa: Opinion1d<f32, A>,
    pub phi: Opinion1d<f32, PHI>,
    pub cond_theta_phi: [Simplex<f32, THETA>; PHI],
}

impl InfoContent {
    pub fn new(
        psi: Opinion1d<f32, PSI>,
        ppsi: Opinion1d<f32, PSI>,
        pa: Opinion1d<f32, A>,
        phi: Opinion1d<f32, PHI>,
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
    pub content: InfoContent,
    pub num_shared: usize,
}

impl Info {
    pub fn reset(&mut self) {
        self.num_shared = 0;
    }

    pub fn new(id: usize, content: InfoContent) -> Self {
        Self {
            id,
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
