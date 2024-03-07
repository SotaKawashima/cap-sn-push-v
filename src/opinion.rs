use crate::{
    info::Info,
    value::{EValue, EValueParam},
};
use approx::UlpsEq;
use core::fmt;
use num_traits::{Float, NumAssign};
use rand::Rng;
use rand_distr::{Dirichlet, Distribution, Exp1, Open01, Standard, StandardNormal};
use serde_with::{serde_as, TryFromInto};
use std::{array, iter::Sum, ops::AddAssign};
use subjective_logic::{
    harr2, harr3,
    mul::{
        op::{Abduction, Deduction, Fuse, FuseAssign, FuseOp},
        prod::{HigherArr2, HigherArr3, Product2, Product3},
        Discount, Opinion, Opinion1d, Projection, Simplex,
    },
};

pub const PSI: usize = 2;
pub const PHI: usize = 2;
pub const S: usize = 2;
pub const O: usize = 2;
pub const A: usize = 2;
pub const B: usize = 2;
pub const THETA: usize = 2;
pub const THETAD: usize = THETA;

pub const F_O: usize = O;
pub const F_S: usize = S;
pub const F_PHI: usize = PHI;
pub const F_PSI: usize = PSI;
pub const F_B: usize = B;
pub const F_THETA: usize = THETA;

pub const K_PSI: usize = PSI;
pub const K_PHI: usize = PHI;
pub const K_S: usize = S;
pub const K_O: usize = O;
pub const K_THETA: usize = THETA;
pub const K_B: usize = B;

#[derive(Debug, serde::Deserialize)]
pub struct GlobalBaseRates<V> {
    pub base: BaseRates<V>,
    pub friend: FriendBaseRates<V>,
    pub social: SocialBaseRates<V>,
}

#[derive(Debug, serde::Deserialize)]
pub struct BaseRates<V> {
    pub psi: [V; PSI],
    pub phi: [V; PHI],
    pub s: [V; S],
    pub o: [V; O],
    pub b: [V; B],
    pub theta: [V; THETA],
    pub a: [V; A],
    pub thetad: [V; THETAD],
}

#[derive(Debug, serde::Deserialize)]
pub struct FriendBaseRates<V> {
    pub fpsi: [V; F_PSI],
    pub fphi: [V; F_PHI],
    pub fs: [V; F_S],
    pub fo: [V; F_S],
    pub fb: [V; F_B],
    pub ftheta: [V; F_THETA],
}

#[derive(Debug, serde::Deserialize)]
pub struct SocialBaseRates<V> {
    pub kpsi: [V; K_PSI],
    pub kphi: [V; K_PHI],
    pub ks: [V; K_S],
    pub ko: [V; K_S],
    pub kb: [V; K_B],
    pub ktheta: [V; K_THETA],
}

#[serde_as]
#[derive(Debug, serde::Deserialize, Clone)]
pub struct InitialOpinions<V: Float + AddAssign + UlpsEq> {
    pub base: InitialBaseSimplexes<V>,
    pub friend: InitialFriendSimplexes<V>,
    pub social: InitialSocialSimplexes<V>,
}

#[serde_as]
#[derive(Debug, serde::Deserialize, Clone)]
#[serde(bound(deserialize = "V: serde::Deserialize<'de>"))]
pub struct InitialBaseSimplexes<V: Float + AddAssign + UlpsEq> {
    #[serde_as(as = "TryFromInto<([V; PSI], V)>")]
    pub psi: Simplex<V, PSI>,
    #[serde_as(as = "TryFromInto<([V; PHI], V)>")]
    pub phi: Simplex<V, PHI>,
    #[serde_as(as = "TryFromInto<([V; S], V)>")]
    pub s: Simplex<V, S>,
    #[serde_as(as = "TryFromInto<([V; O], V)>")]
    pub o: Simplex<V, O>,
}

#[serde_as]
#[derive(Debug, serde::Deserialize, Clone)]
#[serde(bound(deserialize = "V: serde::Deserialize<'de>"))]
pub struct InitialFriendSimplexes<V: Float + AddAssign + UlpsEq> {
    #[serde_as(as = "TryFromInto<([V; F_PHI], V)>")]
    pub fphi: Simplex<V, F_PHI>,
    #[serde_as(as = "TryFromInto<([V; F_S], V)>")]
    pub fs: Simplex<V, F_S>,
    #[serde_as(as = "TryFromInto<([V; F_O], V)>")]
    pub fo: Simplex<V, F_O>,
}

#[serde_as]
#[derive(Debug, serde::Deserialize, Clone)]
#[serde(bound(deserialize = "V: serde::Deserialize<'de>"))]
pub struct InitialSocialSimplexes<V: Float + AddAssign + UlpsEq> {
    #[serde_as(as = "TryFromInto<([V; K_PHI], V)>")]
    pub kphi: Simplex<V, K_PHI>,
    #[serde_as(as = "TryFromInto<([V; K_S], V)>")]
    pub ks: Simplex<V, K_S>,
    #[serde_as(as = "TryFromInto<([V; K_O], V)>")]
    pub ko: Simplex<V, K_O>,
}

#[derive(Debug, serde::Deserialize, Clone)]
pub enum SimplexParam<T, V> {
    /// belief: T, uncertainty: V
    Fixed(T, V),
    /// alpha: Vec<V>
    Dirichlet {
        alpha: Vec<V>,
        zeros: Option<Vec<usize>>,
    },
}

#[derive(Debug)]
pub enum SimplexDist<V, const N: usize>
where
    V: Float,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    Open01: Distribution<V>,
{
    Fixed(Simplex<V, N>),
    Dirichlet {
        dist: Dirichlet<V>,
        b_zeros: [bool; N],
        u_zero: bool,
    },
}

#[derive(thiserror::Error, Debug)]
pub enum SimplexDistError {
    #[error("{0}")]
    SimplexError(#[from] subjective_logic::errors::InvalidValueError),
    #[error("alpha.len() + zeros.len() must be {0}.")]
    LengthExceed(usize),
    #[error("Index {0} exceeds the size of the domain + 1.")]
    ZeroIndexExceed(usize),
    #[error("Index {0} is duplicated.")]
    ZeroIndexDuplicated(usize),
    #[error("{0}")]
    DirichletError(#[from] rand_distr::DirichletError),
}

impl<V, const N: usize> TryFrom<SimplexParam<[V; N], V>> for SimplexDist<V, N>
where
    V: Float + UlpsEq + AddAssign,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    Open01: Distribution<V>,
{
    type Error = SimplexDistError;
    fn try_from(value: SimplexParam<[V; N], V>) -> Result<Self, Self::Error> {
        match value {
            SimplexParam::Fixed(b, u) => Ok(Self::Fixed(Simplex::try_new(b, u)?)),
            SimplexParam::Dirichlet { alpha, zeros } => {
                let l = alpha.len();
                let zero_idxs = zeros.unwrap_or(Vec::new());
                if l + zero_idxs.len() != N + 1 {
                    return Err(SimplexDistError::LengthExceed(N + 1));
                }
                let mut b_zeros = [false; N];
                let mut u_zero = false;
                for idx in zero_idxs {
                    if idx >= N + 1 {
                        return Err(SimplexDistError::ZeroIndexExceed(idx));
                    }
                    if idx == N {
                        if u_zero {
                            return Err(SimplexDistError::ZeroIndexDuplicated(idx));
                        }
                        u_zero = true;
                    } else {
                        if b_zeros[idx] {
                            return Err(SimplexDistError::ZeroIndexDuplicated(idx));
                        }
                        b_zeros[idx] = true;
                    }
                }
                Ok(Self::Dirichlet {
                    dist: Dirichlet::new(&alpha)?,
                    b_zeros,
                    u_zero,
                })
            }
        }
    }
}

impl<V, const N: usize> Distribution<Simplex<V, N>> for SimplexDist<V, N>
where
    V: Float + UlpsEq + AddAssign,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    Open01: Distribution<V>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Simplex<V, N> {
        match self {
            SimplexDist::Fixed(s) => s.clone(),
            SimplexDist::Dirichlet {
                dist,
                b_zeros,
                u_zero,
            } => {
                let mut s = dist.sample(rng);
                let u = if *u_zero { V::zero() } else { s.pop().unwrap() };
                let mut b = [V::zero(); N];

                for i in (0..N).rev() {
                    if !b_zeros[i] {
                        b[i] = s.pop().unwrap();
                    }
                }
                Simplex::new(b, u)
            }
        }
    }
}

#[serde_as]
#[derive(Debug, serde::Deserialize)]
pub struct InitialConditions<V>
where
    V: Float + UlpsEq + AddAssign,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    Open01: Distribution<V>,
{
    pub base: InitialBaseConditions<V>,
    pub friend: InitialFriendConditions<V>,
    pub social: InitialSocialConditions<V>,
}

#[serde_as]
#[derive(Debug, serde::Deserialize)]
#[serde(bound(deserialize = "V: serde::Deserialize<'de>"))]
pub struct InitialBaseConditions<V>
where
    V: Float + UlpsEq + AddAssign,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    Open01: Distribution<V>,
{
    /// $B \Rightarrow O$
    #[serde_as(as = "[TryFromInto<SimplexParam<[V; O], V>>; B]")]
    pub cond_o: [SimplexDist<V, O>; B],
    /// $K\Theta \Rightarrow B$
    #[serde_as(as = "[TryFromInto<SimplexParam<[V; B], V>>; K_THETA]")]
    pub cond_b: [SimplexDist<V, B>; K_THETA],
    /// $B,\Psi \Rightarrow \Theta$
    // #[serde_as(as = "TryFromInto<[[SimplexParam<[V; THETA], V>; PSI]; B]>")]
    // pub cond_theta: HigherArr2<SimplexDist<V, THETA>, B, PSI>,
    // #[serde_as(as = "TryFromInto<CondThetaParam<[V; THETA], V>>")]
    pub cond_theta: CondThetaDist<V>,
    /// $\Phi \Rightarrow \Theta$
    #[serde_as(as = "[TryFromInto<SimplexParam<[V; THETA], V>>; PHI]")]
    pub cond_theta_phi: [SimplexDist<V, THETA>; PHI],
    /// $K\Theta \Rightarrow A$
    #[serde_as(as = "[TryFromInto<SimplexParam<[V; A], V>>; K_THETA]")]
    pub cond_a: [SimplexDist<V, A>; K_THETA],
    /// $B,\Psi,A \Rightarrow \Theta'$
    // #[serde_as(as = "TryFromInto<[[[SimplexParam<[V; THETAD], V>; A]; PSI]; B]>")]
    // pub cond_thetad: HigherArr3<SimplexDist<V, THETAD>, B, PSI, A>,
    // #[serde_as(as = "TryFromInto<CondThetadParam<[V; THETAD], V>>")]
    pub cond_thetad: CondThetadDist<V>,
    /// $\Phi \Rightarrow \Theta'$
    #[serde_as(as = "[TryFromInto<SimplexParam<[V; THETA], V>>; PHI]")]
    pub cond_thetad_phi: [SimplexDist<V, THETAD>; PHI],
}

#[serde_as]
#[derive(Debug, serde::Deserialize)]
#[serde(bound(deserialize = "V: serde::Deserialize<'de>"))]
pub struct InitialFriendConditions<V>
where
    V: Float + UlpsEq + AddAssign,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    Open01: Distribution<V>,
{
    /// $S \Rightarrow F\Psi$
    #[serde_as(as = "[TryFromInto<SimplexParam<[V; F_PSI], V>>; S]")]
    pub cond_fpsi: [SimplexDist<V, F_PSI>; S],
    /// $FB \Rightarrow FO$
    #[serde_as(as = "[TryFromInto<SimplexParam<[V; F_O], V>>; F_B]")]
    pub cond_fo: [SimplexDist<V, F_O>; F_B],
    /// $FS \Rightarrow FB$
    #[serde_as(as = "[TryFromInto<SimplexParam<[V; F_B], V>>; F_S]")]
    pub cond_fb: [SimplexDist<V, F_B>; F_S],
    /// $FB,F\Psi \Rightarrow F\Theta$
    pub cond_ftheta: CondFThetaDist<V>,
    /// $F\Phi \Rightarrow F\Theta$
    #[serde_as(as = "[TryFromInto<SimplexParam<[V; F_THETA], V>>; F_PHI]")]
    pub cond_ftheta_fphi: [SimplexDist<V, F_THETA>; F_PHI],
}

#[serde_as]
#[derive(Debug, serde::Deserialize)]
#[serde(bound(deserialize = "V: serde::Deserialize<'de>"))]
pub struct InitialSocialConditions<V>
where
    V: Float + UlpsEq + AddAssign,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    Open01: Distribution<V>,
{
    /// $S \Rightarrow K\Psi$
    #[serde_as(as = "[TryFromInto<SimplexParam<[V; K_PSI], V>>; S]")]
    pub cond_kpsi: [SimplexDist<V, K_PSI>; S],
    /// $KB \Rightarrow KO$
    #[serde_as(as = "[TryFromInto<SimplexParam<[V; K_O], V>>; K_B]")]
    pub cond_ko: [SimplexDist<V, K_O>; K_B],
    /// $KS \Rightarrow KB$
    #[serde_as(as = "[TryFromInto<SimplexParam<[V; K_B], V>>; K_S]")]
    pub cond_kb: [SimplexDist<V, K_B>; K_S],
    /// $KB,K\Psi \Rightarrow K\Theta$
    pub cond_ktheta: CondKThetaDist<V>,
    /// $K\Phi \Rightarrow K\Theta$
    #[serde_as(as = "[TryFromInto<SimplexParam<[V; K_THETA], V>>; K_PHI]")]
    pub cond_ktheta_kphi: [SimplexDist<V, K_THETA>; K_PHI],
}

/// b'_i = b_i * rate.0 * rate.1, 1-u' = (1-u) * rate.1
#[derive(Debug, serde::Deserialize)]
pub struct RelativeParam<T> {
    pub belief: T,
    pub uncertainty: T,
}

impl<V> Distribution<RelativeParam<V>> for RelativeParam<EValue<V>>
where
    V: Float,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> RelativeParam<V> {
        RelativeParam {
            belief: self.belief.sample(rng),
            uncertainty: self.uncertainty.sample(rng),
        }
    }
}

impl<V> RelativeParam<V>
where
    V: Float + AddAssign + UlpsEq,
{
    fn to_simplex(self, w: &Simplex<V, 2>) -> Simplex<V, 2> {
        let b1 = w.b()[1];
        let u = *w.u();
        let sb = V::one() - u;
        let x = self.belief.min(sb / b1);
        let y = self.uncertainty.min(V::one() / u);
        let ud = u * y;
        let sbd = V::one() - ud;
        let bd1 = b1 / sb * x * sbd;
        Simplex::new([sbd - bd1, bd1], ud)
    }
}

impl<V> TryFrom<RelativeParam<EValueParam<V>>> for RelativeParam<EValue<V>>
where
    V: Float,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
{
    type Error = <EValue<V> as TryFrom<EValueParam<V>>>::Error;

    fn try_from(value: RelativeParam<EValueParam<V>>) -> Result<Self, Self::Error> {
        Ok(Self {
            belief: value.belief.try_into()?,
            uncertainty: value.uncertainty.try_into()?,
        })
    }
}

#[serde_as]
#[derive(Debug, serde::Deserialize)]
#[serde(bound(deserialize = "V: serde::Deserialize<'de>"))]
pub struct CondThetaDist<V>
where
    V: Float + UlpsEq + AddAssign,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    Open01: Distribution<V>,
{
    #[serde_as(as = "TryFromInto<SimplexParam<[V; THETA], V>>")]
    pub b0psi0: SimplexDist<V, THETA>,
    #[serde_as(as = "TryFromInto<SimplexParam<[V; THETA], V>>")]
    pub b1psi1: SimplexDist<V, THETA>,
    #[serde_as(as = "TryFromInto<RelativeParam<EValueParam<V>>>")]
    pub b0psi1: RelativeParam<EValue<V>>,
    #[serde_as(as = "TryFromInto<RelativeParam<EValueParam<V>>>")]
    pub b1psi0: RelativeParam<EValue<V>>,
}

#[serde_as]
#[derive(Debug, serde::Deserialize)]
#[serde(bound(deserialize = "V: serde::Deserialize<'de>"))]
pub struct CondFThetaDist<V>
where
    V: Float + UlpsEq + AddAssign,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    Open01: Distribution<V>,
{
    #[serde_as(as = "TryFromInto<SimplexParam<[V; F_THETA], V>>")]
    pub fb0fpsi0: SimplexDist<V, F_THETA>,
    #[serde_as(as = "TryFromInto<SimplexParam<[V; F_THETA], V>>")]
    pub fb1fpsi1: SimplexDist<V, F_THETA>,
    #[serde_as(as = "TryFromInto<RelativeParam<EValueParam<V>>>")]
    pub fb0fpsi1: RelativeParam<EValue<V>>,
    #[serde_as(as = "TryFromInto<RelativeParam<EValueParam<V>>>")]
    pub fb1fpsi0: RelativeParam<EValue<V>>,
}

#[serde_as]
#[derive(Debug, serde::Deserialize)]
#[serde(bound(deserialize = "V: serde::Deserialize<'de>"))]
pub struct CondKThetaDist<V>
where
    V: Float + UlpsEq + AddAssign,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    Open01: Distribution<V>,
{
    #[serde_as(as = "TryFromInto<SimplexParam<[V; K_THETA], V>>")]
    pub kb0kpsi0: SimplexDist<V, K_THETA>,
    #[serde_as(as = "TryFromInto<SimplexParam<[V; K_THETA], V>>")]
    pub kb1kpsi1: SimplexDist<V, K_THETA>,
    #[serde_as(as = "TryFromInto<RelativeParam<EValueParam<V>>>")]
    pub kb0kpsi1: RelativeParam<EValue<V>>,
    #[serde_as(as = "TryFromInto<RelativeParam<EValueParam<V>>>")]
    pub kb1kpsi0: RelativeParam<EValue<V>>,
}

impl<V> Distribution<HigherArr2<Simplex<V, THETA>, B, PSI>> for CondThetaDist<V>
where
    V: Float + AddAssign + UlpsEq,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    Open01: Distribution<V>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> HigherArr2<Simplex<V, THETA>, B, PSI> {
        let b0psi0 = self.b0psi0.sample(rng);
        let b1psi1 = self.b1psi1.sample(rng);
        let b0psi1 = self.b0psi1.sample(rng).to_simplex(&b1psi1);
        let b1psi0 = self.b1psi0.sample(rng).to_simplex(&b1psi1);
        harr2![[b0psi0, b0psi1], [b1psi0, b1psi1]]
    }
}

impl<V> Distribution<HigherArr2<Simplex<V, F_THETA>, F_B, F_PSI>> for CondFThetaDist<V>
where
    V: Float + AddAssign + UlpsEq,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    Open01: Distribution<V>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> HigherArr2<Simplex<V, F_THETA>, F_B, F_PSI> {
        let fb0fpsi0 = self.fb0fpsi0.sample(rng);
        let fb1fpsi1 = self.fb1fpsi1.sample(rng);
        let fb0fpsi1 = self.fb0fpsi1.sample(rng).to_simplex(&fb1fpsi1);
        let fb1fpsi0 = self.fb1fpsi0.sample(rng).to_simplex(&fb1fpsi1);
        harr2![[fb0fpsi0, fb0fpsi1], [fb1fpsi0, fb1fpsi1]]
    }
}

impl<V> Distribution<HigherArr2<Simplex<V, K_THETA>, K_B, K_PSI>> for CondKThetaDist<V>
where
    V: Float + AddAssign + UlpsEq,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    Open01: Distribution<V>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> HigherArr2<Simplex<V, K_THETA>, K_B, K_PSI> {
        let kb0kpsi0 = self.kb0kpsi0.sample(rng);
        let kb1kpsi1 = self.kb1kpsi1.sample(rng);
        let kb0kpsi1 = self.kb0kpsi1.sample(rng).to_simplex(&kb1kpsi1);
        let kb1kpsi0 = self.kb1kpsi0.sample(rng).to_simplex(&kb1kpsi1);
        harr2![[kb0kpsi0, kb0kpsi1], [kb1kpsi0, kb1kpsi1]]
    }
}

#[serde_as]
#[derive(Debug, serde::Deserialize)]
#[serde(bound(deserialize = "V: serde::Deserialize<'de>"))]
pub struct CondThetadDist<V>
where
    V: Float + UlpsEq + AddAssign,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    Open01: Distribution<V>,
{
    // pub none: SimplexDist<V, N>,
    // pub possible: SimplexDist<V, N>,
    // pub avoid_u_rates: [V; 3],
    /// b'_i = b_i * rate.0 * rate.1, 1-u' = (1-u) * rate.1
    // pub rates: [(V, V); 2],
    #[serde_as(as = "[TryFromInto<SimplexParam<[V; THETAD], V>>; PSI]")]
    pub a0b0: [SimplexDist<V, THETAD>; PSI],
    #[serde_as(as = "[TryFromInto<RelativeParam<EValueParam<V>>>; PSI]")]
    pub a0b1: [RelativeParam<EValue<V>>; PSI],
    #[serde_as(as = "TryFromInto<[[RelativeParam<EValueParam<V>>; PSI]; B]>")]
    pub a1: HigherArr2<RelativeParam<EValue<V>>, B, PSI>,
}

impl<V> Distribution<HigherArr3<Simplex<V, THETAD>, A, B, PSI>> for CondThetadDist<V>
where
    V: Float + AddAssign + UlpsEq,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    Open01: Distribution<V>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> HigherArr3<Simplex<V, THETAD>, B, PSI, A> {
        let [a0b00, a0b01] = array::from_fn(|i| self.a0b0[i].sample(rng));
        let [a0b10, a0b11] = array::from_fn(|i| self.a0b1[i].sample(rng).to_simplex(&a0b01));
        // a -> b -> psi
        harr3![
            [[a0b00.clone(), a0b01.clone()], [a0b10, a0b11]],
            [
                [
                    self.a1[[0, 0]].sample(rng).to_simplex(&a0b00),
                    self.a1[[0, 1]].sample(rng).to_simplex(&a0b00),
                ],
                [
                    self.a1[[1, 0]].sample(rng).to_simplex(&a0b00),
                    self.a1[[1, 1]].sample(rng).to_simplex(&a0b00),
                ]
            ]
        ]
    }
}

#[derive(Debug, Default)]
pub struct ConditionalOpinions<V: Float> {
    base: BaseConditionalOpinions<V>,
    friend: FriendConditionalOpinions<V>,
    social: SocialConditionalOpinions<V>,
}

#[derive(Debug, Default)]
pub struct BaseConditionalOpinions<V: Float> {
    /// $B \Rightarrow O$
    cond_o: [Simplex<V, O>; B],
    /// $K\Theta \Rightarrow B$
    cond_b: [Simplex<V, B>; K_THETA],
    /// $B,\Psi \Rightarrow \Theta$
    cond_theta: HigherArr2<Simplex<V, THETA>, B, PSI>,
    /// $\Phi \Rightarrow \Theta$
    cond_theta_phi: [Simplex<V, THETA>; PHI],
    /// $K\Theta \Rightarrow A$
    cond_a: [Simplex<V, A>; F_THETA],
    /// $B,\Psi,A \Rightarrow \Theta'$
    cond_thetad: HigherArr3<Simplex<V, THETAD>, A, B, PSI>,
    /// $\Phi \Rightarrow \Theta'$
    cond_thetad_phi: [Simplex<V, THETAD>; PHI],
}

#[derive(Debug, Default)]
pub struct FriendConditionalOpinions<V: Float> {
    /// $S \Rightarrow F\Psi$
    cond_fpsi: [Simplex<V, F_PSI>; S],
    /// $FB \Rightarrow FO$
    cond_fo: [Simplex<V, F_O>; F_B],
    /// $FS \Rightarrow FB$
    cond_fb: [Simplex<V, F_B>; F_S],
    /// $FB,F\Psi \Rightarrow F\Theta$
    cond_ftheta: HigherArr2<Simplex<V, F_THETA>, F_B, F_PSI>,
    /// $F\Phi \Rightarrow F\Theta$
    cond_ftheta_fphi: [Simplex<V, F_THETA>; F_PHI],
}

#[derive(Debug, Default)]
pub struct SocialConditionalOpinions<V: Float> {
    /// $S \Rightarrow K\Psi$
    cond_kpsi: [Simplex<V, K_PSI>; S],
    /// $KB \Rightarrow KO$
    cond_ko: [Simplex<V, K_O>; K_B],
    /// $KS \Rightarrow KB$
    cond_kb: [Simplex<V, K_B>; K_S],
    /// $KB,K\Psi \Rightarrow K\Theta$
    cond_ktheta: HigherArr2<Simplex<V, K_THETA>, K_B, K_PSI>,
    /// $K\Phi \Rightarrow K\Theta$
    cond_ktheta_kphi: [Simplex<V, K_THETA>; K_PHI],
}

#[derive(Debug, Default)]
pub struct Opinions<V: Float> {
    op: BaseOpinions<V>,
    sop: SocialOpinions<V>,
    fop: FriendOpinions<V>,
}

#[derive(Debug, Default)]
struct BaseOpinions<V: Float> {
    psi: Opinion1d<V, PSI>,
    phi: Opinion1d<V, PHI>,
    s: Opinion1d<V, S>,
    o: Opinion1d<V, O>,
}

#[derive(Debug, Default)]
pub struct SocialOpinions<V: Float> {
    ko: Opinion1d<V, K_O>,
    ks: Opinion1d<V, K_S>,
    kphi: Opinion1d<V, K_PHI>,
}

#[derive(Debug, Default)]
pub struct FriendOpinions<V: Float> {
    fo: Opinion1d<V, F_O>,
    fs: Opinion1d<V, F_S>,
    fphi: Opinion1d<V, F_PHI>,
}

#[derive(Debug)]
pub struct TempOpinions<V: Float> {
    theta: Opinion1d<V, THETA>,
    b: Opinion1d<V, B>,
    fpsi_ded: Opinion1d<V, F_PSI>,
}

impl<V: Float> TempOpinions<V> {
    #[inline]
    pub fn get_theta_projection(&self) -> [V; THETA]
    where
        V: NumAssign + fmt::Debug,
    {
        let p = self.theta.projection();
        log::debug!("P_TH : {:?}", p);
        p
    }
}

impl<V: Float> BaseOpinions<V> {
    fn new(&self, info: &Info<V>, trust: V) -> Self
    where
        V: UlpsEq + NumAssign,
    {
        let psi = FuseOp::Wgh.fuse(&self.psi, &info.content.psi.discount(trust));
        let phi = FuseOp::Wgh.fuse(&self.phi, &info.content.phi.discount(trust));
        let s = FuseOp::Wgh.fuse(&self.s, &info.content.s.discount(trust));
        let o = FuseOp::Wgh.fuse(&self.o, &info.content.o.discount(trust));

        Self { psi, phi, s, o }
    }

    fn compute_a(
        &self,
        ftheta: &Opinion1d<V, F_THETA>,
        conds: &BaseConditionalOpinions<V>,
        base_rates: &BaseRates<V>,
    ) -> Opinion1d<V, A>
    where
        V: Float + UlpsEq + NumAssign + Sum + Default,
    {
        ftheta
            .deduce(&conds.cond_a)
            .unwrap_or_else(|| Opinion::vacuous_with(base_rates.a))
    }

    fn compute_b(
        &self,
        ktheta: &Opinion1d<V, K_THETA>,
        conds: &BaseConditionalOpinions<V>,
        base_rates: &BaseRates<V>,
    ) -> Opinion1d<V, B>
    where
        V: Float + UlpsEq + NumAssign + Sum + Default,
    {
        let mut b = ktheta
            .deduce(&conds.cond_b)
            .unwrap_or_else(|| Opinion::vacuous_with(base_rates.b));
        let b_abd = self
            .o
            .simplex
            .abduce(&conds.cond_o, base_rates.b)
            .unwrap_or_else(|| Opinion::vacuous_with(base_rates.b));
        FuseOp::ACm.fuse_assign(&mut b, &b_abd);
        b
    }

    fn compute_theta(
        &self,
        b: &Opinion1d<V, B>,
        conds: &BaseConditionalOpinions<V>,
        base_rates: &BaseRates<V>,
    ) -> Opinion1d<V, THETA>
    where
        V: Float + UlpsEq + NumAssign + Sum + Default,
    {
        let mut theta_ded = Opinion::product2(b.as_ref(), self.psi.as_ref())
            .deduce(&conds.cond_theta)
            .unwrap_or_else(|| Opinion::vacuous_with(base_rates.theta));
        let theta_ded_2 = self
            .phi
            .deduce(&conds.cond_theta_phi)
            .unwrap_or_else(|| Opinion::vacuous_with(base_rates.theta));
        FuseOp::Wgh.fuse_assign(&mut theta_ded, &theta_ded_2);
        theta_ded
    }

    fn compute_thetad(
        &self,
        a: &Opinion1d<V, A>,
        b: &Opinion1d<V, B>,
        conds: &BaseConditionalOpinions<V>,
        base_rates: &BaseRates<V>,
    ) -> Opinion1d<V, THETA>
    where
        V: Float + UlpsEq + NumAssign + Sum + Default,
    {
        let thetad_ded_2 = self
            .phi
            .deduce(&conds.cond_thetad_phi)
            .unwrap_or_else(|| Opinion::vacuous_with(base_rates.thetad));
        let mut thetad_ded_1 = Opinion::product3(a.as_ref(), b.as_ref(), self.psi.as_ref())
            .deduce(&conds.cond_thetad)
            .unwrap_or_else(|| Opinion::vacuous_with(base_rates.thetad));
        FuseOp::Wgh.fuse_assign(&mut thetad_ded_1, &thetad_ded_2);
        thetad_ded_1
    }
}

impl<V: Float> SocialOpinions<V> {
    fn new(&self, info: &Info<V>, trust: V) -> Self
    where
        V: UlpsEq + NumAssign,
    {
        // compute friends opinions
        let ko = FuseOp::Wgh.fuse(&self.ko, &info.content.o.discount(trust));
        let ks = FuseOp::Wgh.fuse(&self.ks, &info.content.s.discount(trust));
        let kphi = FuseOp::Wgh.fuse(&self.kphi, &info.content.phi.discount(trust));

        Self { ko, ks, kphi }
    }

    fn compute_ktheta(
        &self,
        kb: &Opinion1d<V, K_B>,
        kpsi: &Opinion1d<V, K_PSI>,
        conds: &SocialConditionalOpinions<V>,
        base_rates: &SocialBaseRates<V>,
    ) -> Opinion1d<V, K_THETA>
    where
        V: UlpsEq + NumAssign + Sum + Default,
    {
        let mut ktheta = Opinion::product2(kb, kpsi)
            .deduce(&conds.cond_ktheta)
            .unwrap_or_else(|| Opinion::<_, V>::vacuous_with(base_rates.ktheta));
        let ktheta_ded_2 = self
            .kphi
            .deduce(&conds.cond_ktheta_kphi)
            .unwrap_or_else(|| Opinion::<_, V>::vacuous_with(base_rates.ktheta));
        FuseOp::Wgh.fuse_assign(&mut ktheta, &ktheta_ded_2);

        ktheta
    }

    fn compute_kb(
        &self,
        conds: &SocialConditionalOpinions<V>,
        base_rates: &SocialBaseRates<V>,
    ) -> Opinion1d<V, F_B>
    where
        V: UlpsEq + NumAssign + Sum + Default,
    {
        let mut kb = self
            .ks
            .deduce(&conds.cond_kb)
            .unwrap_or_else(|| Opinion::vacuous_with(base_rates.kb));
        let kb_abd = self
            .ko
            .simplex
            .abduce(&conds.cond_ko, base_rates.kb)
            .unwrap_or_else(|| Opinion::vacuous_with(base_rates.kb));

        // Use aleatory cummulative fusion
        FuseOp::ACm.fuse_assign(&mut kb, &kb_abd);
        kb
    }

    fn compute_kpsi(
        info: &Info<V>,
        s: &Opinion1d<V, S>,
        trust: V,
        conds: &SocialConditionalOpinions<V>,
        base_rates: &SocialBaseRates<V>,
    ) -> Opinion1d<V, F_B>
    where
        V: UlpsEq + NumAssign + Sum + fmt::Debug + Default,
    {
        let mut kpsi = s
            .deduce(&conds.cond_kpsi)
            .unwrap_or_else(|| Opinion::vacuous_with(base_rates.kpsi));
        FuseOp::Wgh.fuse_assign(&mut kpsi, &info.content.psi.discount(trust));
        kpsi
    }
}

impl<V: Float> FriendOpinions<V> {
    fn new(&self, info: &Info<V>, trust: V) -> Self
    where
        V: UlpsEq + NumAssign,
    {
        // compute friends opinions
        let fo = FuseOp::Wgh.fuse(&self.fo, &info.content.o.discount(trust));
        let fs = FuseOp::Wgh.fuse(&self.fs, &info.content.s.discount(trust));
        let fphi = FuseOp::Wgh.fuse(&self.fphi, &info.content.phi.discount(trust));

        Self { fo, fs, fphi }
    }

    fn deduce_fpsi(
        s: &Opinion1d<V, S>,
        conds: &FriendConditionalOpinions<V>,
        base_rates: &FriendBaseRates<V>,
    ) -> Opinion1d<V, F_PSI>
    where
        V: UlpsEq + NumAssign + Sum + Default,
    {
        s.deduce(&conds.cond_fpsi)
            .unwrap_or_else(|| Opinion::vacuous_with(base_rates.fpsi))
    }

    fn compute_fb(
        &self,
        conds: &FriendConditionalOpinions<V>,
        base_rates: &FriendBaseRates<V>,
    ) -> Opinion1d<V, F_B>
    where
        V: UlpsEq + NumAssign + Sum + Default + fmt::Debug,
    {
        let mut fb = self
            .fs
            .deduce(&conds.cond_fb)
            .unwrap_or_else(|| Opinion::vacuous_with(base_rates.fb));
        let fb_abd = self
            .fo
            .simplex
            .abduce(&conds.cond_fo, base_rates.fb)
            .unwrap_or_else(|| Opinion::vacuous_with(base_rates.fb));

        log::debug!("w_FB@ : {:?}", fb_abd.simplex);
        log::debug!("w_FB% : {:?}", fb.simplex);
        // Use aleatory cummulative fusion
        FuseOp::ACm.fuse_assign(&mut fb, &fb_abd);
        fb
    }

    fn compute_fpsi(info: &Info<V>, fpsi_ded: &Opinion1d<V, F_PSI>, trust: V) -> Opinion1d<V, F_PSI>
    where
        V: UlpsEq + NumAssign + Sum + Default,
    {
        FuseOp::Wgh.fuse(fpsi_ded, &info.content.psi.discount(trust))
    }

    fn compute_ftheta(
        &self,
        fb: &Opinion1d<V, F_B>,
        fpsi: &Opinion1d<V, F_PSI>,
        conds: &FriendConditionalOpinions<V>,
        base_rates: &FriendBaseRates<V>,
    ) -> Opinion1d<V, F_THETA>
    where
        V: UlpsEq + NumAssign + Sum + Default,
    {
        let mut ftheta = Opinion::product2(fb, fpsi)
            .deduce(&conds.cond_ftheta)
            .unwrap_or_else(|| Opinion::<_, V>::vacuous_with(base_rates.ftheta));
        let ftheta_ded_2 = self
            .fphi
            .deduce(&conds.cond_ftheta_fphi)
            .unwrap_or_else(|| Opinion::<_, V>::vacuous_with(base_rates.ftheta));
        FuseOp::Wgh.fuse_assign(&mut ftheta, &ftheta_ded_2);

        ftheta
    }
}

impl<V: Float> Opinions<V> {
    pub fn reset(&mut self, initial_opinions: InitialOpinions<V>, base_rates: &GlobalBaseRates<V>)
    where
        V: UlpsEq + NumAssign,
    {
        let InitialOpinions {
            base,
            friend,
            social,
        } = initial_opinions;

        self.op = (base, &base_rates.base).into();
        self.fop = (friend, &base_rates.friend).into();
        self.sop = (social, &base_rates.social).into();
    }

    pub fn new(&self, info: &Info<V>, trust: V, friend_trust: V, social_trust: V) -> Self
    where
        V: UlpsEq + NumAssign,
    {
        let op = self.op.new(info, trust);
        let sop = self.sop.new(info, friend_trust);
        let fop = self.fop.new(info, social_trust);

        Self { op, sop, fop }
    }

    pub fn compute(
        &self,
        info: &Info<V>,
        social_trust: V,
        conds: &ConditionalOpinions<V>,
        base_rates: &GlobalBaseRates<V>,
    ) -> TempOpinions<V>
    where
        V: UlpsEq + NumAssign + Sum + Default + std::fmt::Debug,
    {
        log::debug!("b_B|PTH_1: {:?}", conds.base.cond_b[1].belief);
        let fpsi_ded = FriendOpinions::deduce_fpsi(&self.op.s, &conds.friend, &base_rates.friend);
        let kb = self.sop.compute_kb(&conds.social, &base_rates.social);
        let kpsi = SocialOpinions::compute_kpsi(
            info,
            &self.op.s,
            social_trust,
            &conds.social,
            &base_rates.social,
        );
        let ktheta = self
            .sop
            .compute_ktheta(&kb, &kpsi, &conds.social, &base_rates.social);
        let b = self.op.compute_b(&ktheta, &conds.base, &base_rates.base);
        let theta = self.op.compute_theta(&b, &conds.base, &base_rates.base);

        log::debug!(" w_PSI  : {:?}", self.op.psi.simplex);
        log::debug!(" w_KPSI : {:?}", kpsi.simplex);
        log::debug!(" w_FPSI': {:?}", fpsi_ded.simplex);
        log::debug!(" w_KB   : {:?}", kb.simplex);
        log::debug!(" w_B    : {:?}", b.simplex);

        TempOpinions { theta, b, fpsi_ded }
    }

    pub fn predict(
        &self,
        temp: &TempOpinions<V>,
        info: &Info<V>,
        friend_trust: V,
        pred_friend_trust: V,
        conds: &ConditionalOpinions<V>,
        base_rates: &GlobalBaseRates<V>,
    ) -> (FriendOpinions<V>, [HigherArr2<V, A, THETAD>; 2])
    where
        V: UlpsEq + Sum + Default + fmt::Debug + NumAssign,
    {
        // current opinions
        let fb = self.fop.compute_fb(&conds.friend, &base_rates.friend);
        let fpsi = FriendOpinions::compute_fpsi(info, &temp.fpsi_ded, friend_trust);
        let ftheta = self
            .fop
            .compute_ftheta(&fb, &fpsi, &conds.friend, &base_rates.friend);
        let a = self.op.compute_a(&ftheta, &conds.base, &base_rates.base);
        let thetad = self
            .op
            .compute_thetad(&a, &temp.b, &conds.base, &base_rates.base);
        let a_thetad = Opinion::product2(a.as_ref(), thetad.as_ref());

        // predict  opinions in case of sharing
        let pred_fop = self.fop.new(info, pred_friend_trust);
        let pred_fb = pred_fop.compute_fb(&conds.friend, &base_rates.friend);
        let pred_fpsi = FriendOpinions::compute_fpsi(info, &temp.fpsi_ded, pred_friend_trust);
        let pred_ftheta =
            pred_fop.compute_ftheta(&pred_fb, &pred_fpsi, &conds.friend, &base_rates.friend);
        let pred_a = self
            .op
            .compute_a(&pred_ftheta, &conds.base, &base_rates.base);
        let pred_thetad = self
            .op
            .compute_thetad(&pred_a, &temp.b, &conds.base, &base_rates.base);
        let pred_fa_thetad = Opinion::product2(pred_ftheta.as_ref(), pred_thetad.as_ref());

        let ps = [a_thetad.projection(), pred_fa_thetad.projection()];

        log::debug!(" w_FS    : {:?}", self.fop.fs.simplex);
        log::debug!("~w_FS    : {:?}", pred_fop.fs.simplex);
        log::debug!(" w_FPSI  : {:?}", fpsi.simplex);
        log::debug!("~w_FPSI  : {:?}", pred_fpsi.simplex);
        log::debug!(" w_FB    : {:?}", fb.simplex);
        log::debug!("~w_FB    : {:?}", pred_fb.simplex);
        log::debug!(" w_FTH   : {:?}", ftheta.simplex);
        log::debug!("~w_FTH   : {:?}", pred_ftheta.simplex);
        log::debug!(" w_A     : {:?}", a.simplex);
        log::debug!("~w_A     : {:?}", pred_a.simplex);
        log::debug!(" w_TH    : {:?}", temp.theta.simplex);
        log::debug!(" w_TH'   : {:?}", thetad.simplex);
        log::debug!("~w_TH'   : {:?}", pred_thetad.simplex);
        log::debug!(" P_A,TH' : {:?}", ps[0]);
        log::debug!("~P_A,TH' : {:?}", ps[1]);

        (pred_fop, ps)
    }

    pub fn replace_pred_fop(&mut self, pred_fop: FriendOpinions<V>) {
        self.fop = pred_fop;
    }
}

impl<V> From<(InitialBaseSimplexes<V>, &BaseRates<V>)> for BaseOpinions<V>
where
    V: Float + AddAssign + UlpsEq,
{
    fn from((init, base_rates): (InitialBaseSimplexes<V>, &BaseRates<V>)) -> Self {
        let InitialBaseSimplexes { psi, phi, s, o } = init;
        Self {
            psi: (psi, base_rates.psi).into(),
            phi: (phi, base_rates.phi).into(),
            s: (s, base_rates.s).into(),
            o: (o, base_rates.o).into(),
        }
    }
}

impl<V> From<(InitialFriendSimplexes<V>, &FriendBaseRates<V>)> for FriendOpinions<V>
where
    V: Float + AddAssign + UlpsEq,
{
    fn from((init, base_rates): (InitialFriendSimplexes<V>, &FriendBaseRates<V>)) -> Self {
        let InitialFriendSimplexes { fphi, fs, fo } = init;
        Self {
            fphi: (fphi, base_rates.fphi).into(),
            fs: (fs, base_rates.fs).into(),
            fo: (fo, base_rates.fo).into(),
        }
    }
}

impl<V> From<(InitialSocialSimplexes<V>, &SocialBaseRates<V>)> for SocialOpinions<V>
where
    V: Float + AddAssign + UlpsEq,
{
    fn from((init, base_rates): (InitialSocialSimplexes<V>, &SocialBaseRates<V>)) -> Self {
        let InitialSocialSimplexes { kphi, ks, ko } = init;
        Self {
            kphi: (kphi, base_rates.kphi).into(),
            ks: (ks, base_rates.ks).into(),
            ko: (ko, base_rates.ko).into(),
        }
    }
}

impl<V: Float> ConditionalOpinions<V> {
    pub fn from_init<R: Rng>(init: &InitialConditions<V>, rng: &mut R) -> Self
    where
        V: Float + AddAssign + UlpsEq,
        Standard: Distribution<V>,
        StandardNormal: Distribution<V>,
        Exp1: Distribution<V>,
        Open01: Distribution<V>,
    {
        // self.cond_fpsi = {
        //     let u = cond_fpsi[1].uncertainty;
        //     let b1 = pi_rate * cond_fpsi[1].belief[1];
        //     cond_fpsi[1].belief[0] = V::one() - u - b1;
        //     cond_fpsi[1].belief[1] = b1;
        //     cond_fpsi
        // };
        // self.cond_kpsi = {
        //     let u = cond_kpsi[1].uncertainty;
        //     let b1 = pi_rate * cond_kpsi[1].belief[1];
        //     cond_kpsi[1].belief[0] = V::one() - u - b1;
        //     cond_kpsi[1].belief[1] = b1;
        //     cond_kpsi
        // };

        Self {
            base: BaseConditionalOpinions::from_init(&init.base, rng),
            friend: FriendConditionalOpinions::from_init(&init.friend, rng),
            social: SocialConditionalOpinions::from_init(&init.social, rng),
        }
    }
}

impl<V: Float> BaseConditionalOpinions<V> {
    pub fn from_init<R: Rng>(init: &InitialBaseConditions<V>, rng: &mut R) -> Self
    where
        V: Float + AddAssign + UlpsEq,
        Standard: Distribution<V>,
        StandardNormal: Distribution<V>,
        Exp1: Distribution<V>,
        Open01: Distribution<V>,
    {
        Self {
            cond_o: array::from_fn(|i| init.cond_o[i].sample(rng)),
            cond_b: array::from_fn(|i| init.cond_b[i].sample(rng)),
            cond_theta: init.cond_theta.sample(rng),
            cond_theta_phi: array::from_fn(|i| init.cond_theta_phi[i].sample(rng)),
            cond_a: array::from_fn(|i| init.cond_a[i].sample(rng)),
            cond_thetad: init.cond_thetad.sample(rng),
            cond_thetad_phi: array::from_fn(|i| init.cond_thetad_phi[i].sample(rng)),
        }
    }
}

impl<V: Float> FriendConditionalOpinions<V> {
    pub fn from_init<R: Rng>(init: &InitialFriendConditions<V>, rng: &mut R) -> Self
    where
        V: Float + AddAssign + UlpsEq,
        Standard: Distribution<V>,
        StandardNormal: Distribution<V>,
        Exp1: Distribution<V>,
        Open01: Distribution<V>,
    {
        Self {
            cond_fpsi: array::from_fn(|i| init.cond_fpsi[i].sample(rng)),
            cond_fo: array::from_fn(|i| init.cond_fo[i].sample(rng)),
            cond_fb: array::from_fn(|i| init.cond_fb[i].sample(rng)),
            cond_ftheta: init.cond_ftheta.sample(rng),
            cond_ftheta_fphi: array::from_fn(|i| init.cond_ftheta_fphi[i].sample(rng)),
        }
    }
}

impl<V: Float> SocialConditionalOpinions<V> {
    pub fn from_init<R: Rng>(init: &InitialSocialConditions<V>, rng: &mut R) -> Self
    where
        V: Float + AddAssign + UlpsEq,
        Standard: Distribution<V>,
        StandardNormal: Distribution<V>,
        Exp1: Distribution<V>,
        Open01: Distribution<V>,
    {
        Self {
            cond_kpsi: array::from_fn(|i| init.cond_kpsi[i].sample(rng)),
            cond_ko: array::from_fn(|i| init.cond_ko[i].sample(rng)),
            cond_kb: array::from_fn(|i| init.cond_kb[i].sample(rng)),
            cond_ktheta: init.cond_ktheta.sample(rng),
            cond_ktheta_kphi: array::from_fn(|i| init.cond_ktheta_kphi[i].sample(rng)),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::array;

    use approx::ulps_eq;
    use rand::thread_rng;
    use rand_distr::Distribution;
    use subjective_logic::{
        harr2,
        mul::{prod::HigherArr2, IndexedContainer, Simplex},
    };

    use crate::value::EValue;

    use super::{CondThetaDist, CondThetadDist, RelativeParam, SimplexDist, SimplexParam};

    #[test]
    fn test_simplex_dist_conversion() {
        let mut rng = thread_rng();
        let b = [0.1f32, 0.2];
        let u = 0.7;
        let fp = SimplexParam::Fixed(b, u);
        let d = SimplexDist::try_from(fp).unwrap();
        let s = d.sample(&mut rng);
        assert_eq!(s.belief, b);
        assert_eq!(s.uncertainty, u);

        let edp = SimplexParam::Dirichlet {
            alpha: vec![5.0f32, 5.0, 1.0],
            zeros: None,
        };
        let ed = SimplexDist::<_, 3>::try_from(edp);
        assert!(ed.is_err());
        if let Err(e) = ed {
            println!("{:}", e);
        }
        let dp = SimplexParam::Dirichlet {
            alpha: vec![7.0, 4.0, 1.0],
            zeros: None,
        };
        let d = SimplexDist::<_, 2>::try_from(dp);
        let n = 25;
        assert!(d.is_ok());
        let ss = d.unwrap().sample_iter(&mut rng).take(n).collect::<Vec<_>>();
        let mean_b0 = ss.iter().map(|s| s.b()[0]).sum::<f32>() / n as f32;
        let mean_b1 = ss.iter().map(|s| s.b()[1]).sum::<f32>() / n as f32;
        let mean_u = ss.iter().map(|s| s.u()).sum::<f32>() / n as f32;
        println!("{mean_b0:?}, {mean_b1:?}, {mean_u:?}");
        assert!(mean_b0 > mean_b1 && mean_b1 > mean_u);
    }

    #[test]
    fn test_simplex_zeros() {
        let rng = &mut thread_rng();

        let dp = SimplexParam::Dirichlet {
            alpha: vec![7.0, 4.0],
            zeros: Some(vec![2]),
        };
        let d = SimplexDist::<_, 2>::try_from(dp);
        assert!(d.is_ok());
        let s = d.unwrap().sample(rng);
        assert_eq!(*s.u(), 0.0f32);
        let dp = SimplexParam::Dirichlet {
            alpha: vec![7.0, 4.0],
            zeros: Some(vec![0]),
        };
        let d = SimplexDist::<_, 2>::try_from(dp);
        assert!(d.is_ok());
        let s = d.unwrap().sample(rng);
        assert_eq!(s.b()[0], 0.0f32);
        let dp = SimplexParam::Dirichlet {
            alpha: vec![7.0, 4.0],
            zeros: Some(vec![1]),
        };
        let d = SimplexDist::<_, 2>::try_from(dp);
        assert!(d.is_ok());
        let s = d.unwrap().sample(rng);
        assert_eq!(s.b()[1], 0.0f32);

        let dp = SimplexParam::Dirichlet {
            alpha: vec![7.0, 4.0, 1.0],
            zeros: Some(vec![1]),
        };
        let d = SimplexDist::<_, 2>::try_from(dp);
        assert!(d.is_err());
        println!("{}", d.err().unwrap());
        let dp = SimplexParam::Dirichlet {
            alpha: vec![7.0, 4.0],
            zeros: Some(vec![1, 1]),
        };
        let d = SimplexDist::<_, 3>::try_from(dp);
        assert!(d.is_err());
        println!("{}", d.err().unwrap());
        let dp = SimplexParam::Dirichlet {
            alpha: vec![7.0, 4.0],
            zeros: Some(vec![3]),
        };
        let d = SimplexDist::<_, 2>::try_from(dp);
        assert!(d.is_err());
        println!("{}", d.err().unwrap());
    }

    #[test]
    fn test_relative_param() {
        let w = Simplex::new([0.9, 0.1], 0.00);
        let wd = RelativeParam {
            belief: 1.0,
            uncertainty: 1.0,
        }
        .to_simplex(&w);
        assert_eq!(w, wd);

        let w = Simplex::new([0.0, 0.90], 0.10);
        let wd = RelativeParam {
            belief: 1.0,
            uncertainty: 1.0,
        }
        .to_simplex(&w);
        assert_eq!(w, wd);

        let w = Simplex::new([0.0, 0.90], 0.10);
        let wd = RelativeParam {
            belief: 1.0,
            uncertainty: 0.0,
        }
        .to_simplex(&w);
        assert_eq!(wd.b()[1], 1.0);

        let w = Simplex::new([0.2, 0.30], 0.50);
        let wd = RelativeParam {
            belief: 1.0,
            uncertainty: 0.0,
        }
        .to_simplex(&w);
        assert_eq!(wd.b()[0], 0.4);
        assert_eq!(wd.b()[1], 0.6);

        let w = Simplex::new([0.2, 0.30], 0.50);
        let wd = RelativeParam {
            belief: 0.0,
            uncertainty: 1.2,
        }
        .to_simplex(&w);
        assert_eq!(wd.b()[0], 0.4);
        assert_eq!(*wd.u(), 0.6);
    }

    #[test]
    fn test_cond_theta() {
        let rng = &mut thread_rng();
        let cond_dist = CondThetaDist {
            b0psi0: SimplexParam::Fixed([0.95, 0.00], 0.05).try_into().unwrap(),
            b1psi1: SimplexParam::Dirichlet {
                alpha: vec![10.5, 10.5, 9.0],
                zeros: None,
            }
            .try_into()
            .unwrap(),
            b0psi1: RelativeParam {
                belief: EValue::fixed(1.2),
                uncertainty: EValue::fixed(1.0),
            },
            b1psi0: RelativeParam {
                belief: EValue::fixed(1.5),
                uncertainty: EValue::fixed(1.0),
            },
        };
        for cond in cond_dist.sample_iter(rng).take(10) {
            let pw = &cond[[1, 1]];
            let pw1 = &cond[[0, 1]];
            let pw2 = &cond[[1, 0]];
            let k = pw.b()[1] / (1.0 - *pw.u());
            assert!(1.2 > 1.0 / k || ulps_eq!(k * 1.2, pw1.b()[1] / (1.0 - *pw1.u())));
            assert!(1.5 > 1.0 / k || ulps_eq!(k * 1.5, pw2.b()[1] / (1.0 - *pw2.u())));
        }
    }

    #[test]
    fn test_cond_thetad() {
        let rng = &mut thread_rng();
        let a0b1 = [
            RelativeParam {
                belief: EValue::fixed(1.2f32),
                uncertainty: EValue::fixed(0.98),
            },
            RelativeParam {
                belief: EValue::fixed(1.5),
                uncertainty: EValue::fixed(1.01),
            },
        ];
        let a1 = harr2![
            [
                RelativeParam {
                    belief: EValue::fixed(1.0),
                    uncertainty: EValue::fixed(1.0),
                },
                RelativeParam {
                    belief: EValue::fixed(1.0),
                    uncertainty: EValue::fixed(1.0),
                },
            ],
            [
                RelativeParam {
                    belief: EValue::fixed(1.0),
                    uncertainty: EValue::fixed(1.25),
                },
                RelativeParam {
                    belief: EValue::fixed(1.0),
                    uncertainty: EValue::fixed(1.5),
                },
            ]
        ];
        let a0b1_: [RelativeParam<_>; 2] = array::from_fn(|i| a0b1[i].sample(rng));
        let a1_: HigherArr2<RelativeParam<_>, 2, 2> = HigherArr2::from_fn(|i| a1[i].sample(rng));

        let cond_dist = CondThetadDist {
            a0b0: [
                SimplexParam::Dirichlet {
                    alpha: vec![27.0, 3.0],
                    zeros: Some(vec![1]),
                }
                .try_into()
                .unwrap(),
                SimplexParam::Dirichlet {
                    alpha: vec![10.5, 10.5, 9.0],
                    zeros: None,
                }
                .try_into()
                .unwrap(),
            ],
            a0b1,
            a1,
        };
        for cond in cond_dist.sample_iter(rng).take(10) {
            let a0b0 = [&cond[[0, 0, 0]], &cond[[0, 0, 1]]];
            let a0b1 = [&cond[[0, 1, 0]], &cond[[0, 1, 1]]];
            let a1 = [
                &cond[[1, 0, 0]],
                &cond[[1, 0, 1]],
                &cond[[1, 1, 0]],
                &cond[[1, 1, 1]],
            ];
            let pu = *a0b0[1].u();
            let k = a0b0[1].b()[1] / (1.0 - pu);
            let nu = *a0b0[0].u();

            assert_eq!(a0b0[0].b()[1], 0.0);
            assert!(*a0b0[0].u() > 0.0);
            for (rp, pw) in a0b1_.iter().zip(a0b1) {
                assert!(
                    rp.belief > 1.0 / k || ulps_eq!(k * rp.belief, pw.b()[1] / (1.0 - *pw.u()))
                );
                assert!(rp.uncertainty > 1.0 / pu || ulps_eq!(pu * rp.uncertainty, *pw.u()));
            }
            for (rp, aw) in a1_.into_iter().zip(a1) {
                assert!(rp.uncertainty > 1.0 / nu || ulps_eq!(rp.uncertainty * nu, *aw.u()));
            }
        }
    }
}
