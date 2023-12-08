use once_cell::sync::Lazy;
use subjective_logic::mul::Simplex;

pub const THETA: usize = 3;
pub const PSI: usize = 2;
pub const PHI: usize = 2;
pub const A: usize = 2;

pub const THETA_VACUOUS: Lazy<Simplex<f32, THETA>> = Lazy::new(|| Simplex::<f32, THETA>::vacuous());
pub const A_VACUOUS: Lazy<Simplex<f32, A>> = Lazy::new(|| Simplex::<f32, A>::vacuous());
