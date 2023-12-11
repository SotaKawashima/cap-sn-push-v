use once_cell::sync::Lazy;
use subjective_logic::mul::Simplex;

pub const THETA: usize = 3;
pub const PSI: usize = 2;
pub const PHI: usize = 2;
pub const A: usize = 2;
pub const S: usize = 2;
pub const P_THETA: usize = THETA;
pub const P_PSI: usize = PSI;
pub const P_A: usize = A;
pub const F_THETA: usize = THETA;
pub const F_PSI: usize = PSI;
pub const F_PHI: usize = PHI;
pub const F_A: usize = A;
pub const F_S: usize = S;
pub const FP_THETA: usize = P_THETA;
pub const FP_PSI: usize = P_PSI;
pub const FP_A: usize = P_A;

pub const PSI_VACUOUS: Lazy<Simplex<f32, PSI>> = Lazy::new(|| Simplex::<f32, PSI>::vacuous());
pub const THETA_VACUOUS: Lazy<Simplex<f32, THETA>> = Lazy::new(|| Simplex::<f32, THETA>::vacuous());
pub const A_VACUOUS: Lazy<Simplex<f32, A>> = Lazy::new(|| Simplex::<f32, A>::vacuous());
