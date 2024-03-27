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
use std::{iter::Sum, ops::AddAssign};
use subjective_logic::{
    domain::{Domain, DomainConv},
    errors::InvalidValueError,
    impl_domain,
    iter::FromFn,
    marr_d1, marr_d2,
    mul::labeled::{OpinionD1, OpinionD2, OpinionD3, SimplexD1},
    multi_array::labeled::{MArrD1, MArrD2, MArrD3},
    ops::{
        Abduction, Deduction, Discount, Fuse, FuseAssign, FuseOp, Indexes, Product2, Product3,
        Projection, Zeros,
    },
};
use tracing::{debug, span, Level};

#[derive(Debug)]
pub struct Psi;
impl_domain!(Psi = 2);

#[derive(Debug)]
pub struct Phi;
impl_domain!(Phi = 2);

#[derive(Debug)]
pub struct S;
impl_domain!(S = 2);

#[derive(Debug)]
pub struct O;
impl_domain!(O = 2);

#[derive(Debug)]
pub struct A;
impl_domain!(A = 2);

#[derive(Debug)]
pub struct B;
impl_domain!(B = 2);

#[derive(Debug)]
pub struct Theta;
impl_domain!(Theta = 2);

#[derive(Debug)]
pub struct Thetad;
impl_domain!(Thetad from Theta);

#[derive(Debug)]
pub struct FO;
impl_domain!(FO from O);

#[derive(Debug)]
pub struct FS;
impl_domain!(FS from S);

#[derive(Debug)]
pub struct FPhi;
impl_domain!(FPhi from Phi);

#[derive(Debug)]
pub struct FPsi;
impl_domain!(FPsi from Psi);

#[derive(Debug)]
pub struct FB;
impl_domain!(FB from B);

#[derive(Debug)]
pub struct FTheta;
impl_domain!(FTheta from Theta);

#[derive(Debug)]
pub struct KPsi;
impl_domain!(KPsi from Psi);

#[derive(Debug)]
pub struct KPhi;
impl_domain!(KPhi from Phi);

#[derive(Debug)]
pub struct KS;
impl_domain!(KS from S);

#[derive(Debug)]
pub struct KO;
impl_domain!(KO from O);

#[derive(Debug)]
pub struct KTheta;
impl_domain!(KTheta from Theta);

#[derive(Debug)]
pub struct KB;
impl_domain!(KB from B);

#[derive(Debug, serde::Deserialize)]
pub struct GlobalBaseRates<V> {
    pub base: BaseRates<V>,
    pub friend: FriendBaseRates<V>,
    pub social: SocialBaseRates<V>,
}

#[serde_as]
#[derive(Debug, serde::Deserialize)]
#[serde(bound(deserialize = "V: serde::Deserialize<'de>"))]
pub struct BaseRates<V> {
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    pub psi: MArrD1<Psi, V>,
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    pub phi: MArrD1<Phi, V>,
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    pub s: MArrD1<S, V>,
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    pub o: MArrD1<O, V>,
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    pub b: MArrD1<B, V>,
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    pub theta: MArrD1<Theta, V>,
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    pub a: MArrD1<A, V>,
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    pub thetad: MArrD1<Thetad, V>,
}

#[serde_as]
#[derive(Debug, serde::Deserialize)]
#[serde(bound(deserialize = "V: serde::Deserialize<'de>"))]
pub struct FriendBaseRates<V> {
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    pub fpsi: MArrD1<FPsi, V>,
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    pub fphi: MArrD1<FPhi, V>,
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    pub fs: MArrD1<FS, V>,
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    pub fo: MArrD1<FO, V>,
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    pub fb: MArrD1<FB, V>,
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    pub ftheta: MArrD1<FTheta, V>,
}

#[serde_as]
#[derive(Debug, serde::Deserialize)]
#[serde(bound(deserialize = "V: serde::Deserialize<'de>"))]
pub struct SocialBaseRates<V> {
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    pub kpsi: MArrD1<KPsi, V>,
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    pub kphi: MArrD1<KPhi, V>,
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    pub ks: MArrD1<KS, V>,
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    pub ko: MArrD1<KO, V>,
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    pub kb: MArrD1<KB, V>,
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    pub ktheta: MArrD1<KTheta, V>,
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
    #[serde_as(as = "TryFromInto<(Vec<V>, V)>")]
    pub psi: SimplexD1<Psi, V>,
    #[serde_as(as = "TryFromInto<(Vec<V>, V)>")]
    pub phi: SimplexD1<Phi, V>,
    #[serde_as(as = "TryFromInto<(Vec<V>, V)>")]
    pub s: SimplexD1<S, V>,
    #[serde_as(as = "TryFromInto<(Vec<V>, V)>")]
    pub o: SimplexD1<O, V>,
}

#[serde_as]
#[derive(Debug, serde::Deserialize, Clone)]
#[serde(bound(deserialize = "V: serde::Deserialize<'de>"))]
pub struct InitialFriendSimplexes<V: Float + AddAssign + UlpsEq> {
    #[serde_as(as = "TryFromInto<(Vec<V>, V)>")]
    pub fphi: SimplexD1<FPhi, V>,
    #[serde_as(as = "TryFromInto<(Vec<V>, V)>")]
    pub fs: SimplexD1<FS, V>,
    #[serde_as(as = "TryFromInto<(Vec<V>, V)>")]
    pub fo: SimplexD1<FO, V>,
}

#[serde_as]
#[derive(Debug, serde::Deserialize, Clone)]
#[serde(bound(deserialize = "V: serde::Deserialize<'de>"))]
pub struct InitialSocialSimplexes<V: Float + AddAssign + UlpsEq> {
    #[serde_as(as = "TryFromInto<(Vec<V>, V)>")]
    pub kphi: SimplexD1<KPhi, V>,
    #[serde_as(as = "TryFromInto<(Vec<V>, V)>")]
    pub ks: SimplexD1<KS, V>,
    #[serde_as(as = "TryFromInto<(Vec<V>, V)>")]
    pub ko: SimplexD1<KO, V>,
}

#[derive(Debug, serde::Deserialize, Clone)]
pub enum SimplexParam<V> {
    /// belief: T, uncertainty: V
    Fixed(Vec<V>, V),
    /// alpha: Vec<V>
    Dirichlet {
        alpha: Vec<V>,
        zeros: Option<Vec<usize>>,
    },
}

#[derive(Debug)]
pub enum SimplexDist<D, V>
where
    V: Float,
    D: Domain,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    Open01: Distribution<V>,
{
    Fixed(SimplexD1<D, V>),
    Dirichlet {
        dist: Dirichlet<V>,
        b_zeros: MArrD1<D, bool>,
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

impl<D, V> TryFrom<SimplexParam<V>> for SimplexDist<D, V>
where
    D: Domain<Idx = usize>,
    V: Float + UlpsEq + AddAssign,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    Open01: Distribution<V>,
{
    type Error = SimplexDistError;
    fn try_from(value: SimplexParam<V>) -> Result<Self, Self::Error> {
        match value {
            SimplexParam::Fixed(b, u) => Ok(Self::Fixed(SimplexD1::try_from((b, u))?)),
            SimplexParam::Dirichlet { alpha, zeros } => {
                let l = alpha.len();
                let zero_idxs = zeros.unwrap_or(Vec::new());
                if l + zero_idxs.len() != D::LEN + 1 {
                    return Err(SimplexDistError::LengthExceed(D::LEN + 1));
                }
                let mut b_zeros = MArrD1::default();
                let mut u_zero = false;
                for idx in zero_idxs {
                    if idx >= D::LEN + 1 {
                        return Err(SimplexDistError::ZeroIndexExceed(idx));
                    }
                    if idx == D::LEN {
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

impl<D, V> Distribution<SimplexD1<D, V>> for SimplexDist<D, V>
where
    D: Domain<Idx = usize>,
    V: Float + UlpsEq + AddAssign,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    Open01: Distribution<V>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> SimplexD1<D, V> {
        match self {
            SimplexDist::Fixed(s) => s.clone(),
            SimplexDist::Dirichlet {
                dist,
                b_zeros,
                u_zero,
            } => {
                let mut s = dist.sample(rng);
                let u = if *u_zero { V::zero() } else { s.pop().unwrap() };
                let mut b = MArrD1::zeros();

                for i in MArrD1::<D, V>::indexes().rev() {
                    if !b_zeros[i] {
                        b[i] = s.pop().unwrap();
                    }
                }
                SimplexD1::new(b, u)
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
    /// $K\Theta \Rightarrow A$
    #[serde_as(as = "TryFromInto<Vec<SimplexParam<V>>>")]
    pub cond_a: MArrD1<KTheta, SimplexDist<A, V>>,
    /// $K\Theta \Rightarrow B$
    #[serde_as(as = "TryFromInto<Vec<SimplexParam<V>>>")]
    pub cond_b: MArrD1<KTheta, SimplexDist<B, V>>,
    /// $B \Rightarrow O$
    #[serde_as(as = "TryFromInto<Vec<SimplexParam<V>>>")]
    pub cond_o: MArrD1<B, SimplexDist<O, V>>,
    /// $B,\Psi \Rightarrow \Theta$
    pub cond_theta: CondThetaDist<V>,
    /// relative paramter of $B,\Psi,A \Rightarrow \Theta'$ and $\Phi \Rightarrow \Theta'$
    #[serde_as(as = "TryFromInto<RelativeParam<EValueParam<V>>>")]
    pub rel_cond_thetad: RelativeParam<EValue<V>>,
    /// $\Phi \Rightarrow \Theta$
    #[serde_as(as = "TryFromInto<Vec<SimplexParam<V>>>")]
    pub cond_theta_phi: MArrD1<Phi, SimplexDist<Theta, V>>,
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
    #[serde_as(as = "TryFromInto<Vec<SimplexParam<V>>>")]
    pub cond_fpsi: MArrD1<S, SimplexDist<FPsi, V>>,
    /// $FB \Rightarrow FO$
    #[serde_as(as = "TryFromInto<Vec<SimplexParam<V>>>")]
    pub cond_fo: MArrD1<FB, SimplexDist<FO, V>>,
    /// $FS \Rightarrow FB$
    #[serde_as(as = "TryFromInto<Vec<SimplexParam<V>>>")]
    pub cond_fb: MArrD1<FS, SimplexDist<FB, V>>,
    /// $FB,F\Psi \Rightarrow F\Theta$
    pub cond_ftheta: CondFThetaDist<V>,
    /// $F\Phi \Rightarrow F\Theta$
    #[serde_as(as = "TryFromInto<Vec<SimplexParam<V>>>")]
    pub cond_ftheta_fphi: MArrD1<FPhi, SimplexDist<FTheta, V>>,
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
    #[serde_as(as = "TryFromInto<Vec<SimplexParam<V>>>")]
    pub cond_kpsi: MArrD1<S, SimplexDist<KPsi, V>>,
    /// $KB \Rightarrow KO$
    #[serde_as(as = "TryFromInto<Vec<SimplexParam<V>>>")]
    pub cond_ko: MArrD1<KB, SimplexDist<KO, V>>,
    /// $KS \Rightarrow KB$
    #[serde_as(as = "TryFromInto<Vec<SimplexParam<V>>>")]
    pub cond_kb: MArrD1<KS, SimplexDist<KB, V>>,
    /// $KB,K\Psi \Rightarrow K\Theta$
    pub cond_ktheta: CondKThetaDist<V>,
    /// $K\Phi \Rightarrow K\Theta$
    #[serde_as(as = "TryFromInto<Vec<SimplexParam<V>>>")]
    pub cond_ktheta_kphi: MArrD1<KPhi, SimplexDist<KTheta, V>>,
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
    fn to_simplex<D: Domain<Idx = usize>>(
        &self,
        w: &SimplexD1<D, V>,
    ) -> Result<SimplexD1<D, V>, InvalidValueError> {
        if w.is_vacuous() {
            return Ok(SimplexD1::vacuous());
        }
        let b1 = w.b()[1];
        let u = *w.u();
        let sb = V::one() - u;
        let x = self.belief.min(sb / b1);
        let y = self.uncertainty.min(V::one() / u);
        let ud = u * y;
        let sbd = V::one() - ud;
        let bd1 = b1 / sb * x * sbd;
        SimplexD1::try_new(marr_d1![sbd - bd1, bd1], ud)
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
    #[serde_as(as = "TryFromInto<SimplexParam<V>>")]
    pub b0psi0: SimplexDist<Theta, V>,
    #[serde_as(as = "TryFromInto<SimplexParam<V>>")]
    pub b1psi1: SimplexDist<Theta, V>,
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
    #[serde_as(as = "TryFromInto<SimplexParam<V>>")]
    pub fb0fpsi0: SimplexDist<FTheta, V>,
    #[serde_as(as = "TryFromInto<SimplexParam< V>>")]
    pub fb1fpsi1: SimplexDist<FTheta, V>,
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
    #[serde_as(as = "TryFromInto<SimplexParam< V>>")]
    pub kb0kpsi0: SimplexDist<KTheta, V>,
    #[serde_as(as = "TryFromInto<SimplexParam<V>>")]
    pub kb1kpsi1: SimplexDist<KTheta, V>,
    #[serde_as(as = "TryFromInto<RelativeParam<EValueParam<V>>>")]
    pub kb0kpsi1: RelativeParam<EValue<V>>,
    #[serde_as(as = "TryFromInto<RelativeParam<EValueParam<V>>>")]
    pub kb1kpsi0: RelativeParam<EValue<V>>,
}

impl<V> Distribution<MArrD2<B, Psi, SimplexD1<Theta, V>>> for CondThetaDist<V>
where
    V: Float + AddAssign + UlpsEq,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    Open01: Distribution<V>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> MArrD2<B, Psi, SimplexD1<Theta, V>> {
        let b0psi0 = self.b0psi0.sample(rng);
        let b1psi1 = self.b1psi1.sample(rng);
        let b0psi1 = self.b0psi1.sample(rng).to_simplex(&b1psi1).unwrap();
        let b1psi0 = self.b1psi0.sample(rng).to_simplex(&b1psi1).unwrap();
        marr_d2![[b0psi0, b0psi1], [b1psi0, b1psi1]]
    }
}

impl<V> Distribution<MArrD2<FB, FPsi, SimplexD1<FTheta, V>>> for CondFThetaDist<V>
where
    V: Float + AddAssign + UlpsEq,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    Open01: Distribution<V>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> MArrD2<FB, FPsi, SimplexD1<FTheta, V>> {
        let fb0fpsi0 = self.fb0fpsi0.sample(rng);
        let fb1fpsi1 = self.fb1fpsi1.sample(rng);
        let fb0fpsi1 = self.fb0fpsi1.sample(rng).to_simplex(&fb1fpsi1).unwrap();
        let fb1fpsi0 = self.fb1fpsi0.sample(rng).to_simplex(&fb1fpsi1).unwrap();
        marr_d2![[fb0fpsi0, fb0fpsi1], [fb1fpsi0, fb1fpsi1]]
    }
}

impl<V> Distribution<MArrD2<KB, KPsi, SimplexD1<KTheta, V>>> for CondKThetaDist<V>
where
    V: Float + AddAssign + UlpsEq,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    Open01: Distribution<V>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> MArrD2<KB, KPsi, SimplexD1<KTheta, V>> {
        let kb0kpsi0 = self.kb0kpsi0.sample(rng);
        let kb1kpsi1 = self.kb1kpsi1.sample(rng);
        let kb0kpsi1 = self.kb0kpsi1.sample(rng).to_simplex(&kb1kpsi1).unwrap();
        let kb1kpsi0 = self.kb1kpsi0.sample(rng).to_simplex(&kb1kpsi1).unwrap();
        marr_d2![[kb0kpsi0, kb0kpsi1], [kb1kpsi0, kb1kpsi1]]
    }
}

// #[serde_as]
// #[derive(Debug, serde::Deserialize)]
// #[serde(bound(deserialize = "V: serde::Deserialize<'de>"))]
// pub struct CondThetadDist<V>
// where
//     V: Float + UlpsEq + AddAssign,
//     Standard: Distribution<V>,
//     StandardNormal: Distribution<V>,
//     Exp1: Distribution<V>,
//     Open01: Distribution<V>,
// {
//     // #[serde_as(as = "TryFromInto<[[RelativeParam<EValueParam<V>>; PSI]; B]>")]
//     // pub a0: HigherArr2<RelativeParam<EValue<V>>, B, PSI>,
//     // #[serde_as(as = "TryFromInto<[[RelativeParam<EValueParam<V>>; PSI]; B]>")]
//     // pub a1: HigherArr2<RelativeParam<EValue<V>>, B, PSI>,
// }

// impl<V> CondThetadDist<V>
// where
//     V: Float + AddAssign + UlpsEq,
//     Standard: Distribution<V>,
//     StandardNormal: Distribution<V>,
//     Exp1: Distribution<V>,
//     Open01: Distribution<V>,
// {
//     fn sample<R: Rng + ?Sized>(
//         &self,
//         rng: &mut R,
//         cond: &HigherArr2<Simplex<V, THETA>, B, PSI>,
//     ) -> HigherArr3<Simplex<V, THETAD>, B, PSI, A> {
//         // a -> b -> psi
//         let a000 = self.a0[[0, 0]]
//             .sample(rng)
//             .to_simplex(&cond[[0, 0]])
//             .unwrap();
//         let a001 = self.a0[[0, 1]]
//             .sample(rng)
//             .to_simplex(&cond[[0, 1]])
//             .unwrap();
//         let a010 = self.a0[[1, 0]]
//             .sample(rng)
//             .to_simplex(&cond[[1, 0]])
//             .unwrap();
//         let a011 = self.a0[[1, 1]]
//             .sample(rng)
//             .to_simplex(&cond[[1, 1]])
//             .unwrap();
//         harr3![
//             [[a000.clone(), a001], [a010, a011]],
//             [[a000.clone(), a000.clone()], [a000.clone(), a000]]
//         ]
//     }
// }

#[derive(Debug, Default)]
pub struct ConditionalOpinions<V: Float> {
    base: BaseConditionalOpinions<V>,
    friend: FriendConditionalOpinions<V>,
    social: SocialConditionalOpinions<V>,
}

#[derive(Debug, Default)]
pub struct BaseConditionalOpinions<V: Float> {
    /// $B \Rightarrow O$
    cond_o: MArrD1<B, SimplexD1<O, V>>,
    /// $K\Theta \Rightarrow B$
    cond_b: MArrD1<KTheta, SimplexD1<B, V>>,
    /// $B,\Psi \Rightarrow \Theta$
    cond_theta: MArrD2<B, Psi, SimplexD1<Theta, V>>,
    /// $\Phi \Rightarrow \Theta$
    cond_theta_phi: MArrD1<Phi, SimplexD1<Theta, V>>,
    /// $K\Theta \Rightarrow A$
    cond_a: MArrD1<FTheta, SimplexD1<A, V>>,
    /// $B,\Psi,A \Rightarrow \Theta'$
    cond_thetad: MArrD3<A, B, Psi, SimplexD1<Thetad, V>>,
    /// $\Phi \Rightarrow \Theta'$
    cond_thetad_phi: MArrD1<Phi, SimplexD1<Thetad, V>>,
}

#[derive(Debug, Default)]
pub struct FriendConditionalOpinions<V: Float> {
    /// $S \Rightarrow F\Psi$
    cond_fpsi: MArrD1<S, SimplexD1<FPsi, V>>,
    /// $FB \Rightarrow FO$
    cond_fo: MArrD1<FB, SimplexD1<FO, V>>,
    /// $FS \Rightarrow FB$
    cond_fb: MArrD1<FS, SimplexD1<FB, V>>,
    /// $FB,F\Psi \Rightarrow F\Theta$
    cond_ftheta: MArrD2<FB, FPsi, SimplexD1<FTheta, V>>,
    /// $F\Phi \Rightarrow F\Theta$
    cond_ftheta_fphi: MArrD1<FPhi, SimplexD1<FTheta, V>>,
}

#[derive(Debug, Default)]
pub struct SocialConditionalOpinions<V: Float> {
    /// $S \Rightarrow K\Psi$
    cond_kpsi: MArrD1<S, SimplexD1<KPsi, V>>,
    /// $KB \Rightarrow KO$
    cond_ko: MArrD1<KB, SimplexD1<KO, V>>,
    /// $KS \Rightarrow KB$
    cond_kb: MArrD1<KS, SimplexD1<KB, V>>,
    /// $KB,K\Psi \Rightarrow K\Theta$
    cond_ktheta: MArrD2<KB, KPsi, SimplexD1<KTheta, V>>,
    /// $K\Phi \Rightarrow K\Theta$
    cond_ktheta_kphi: MArrD1<KPhi, SimplexD1<KTheta, V>>,
}

#[derive(Debug, Default)]
pub struct Opinions<V: Float> {
    op: BaseOpinions<V>,
    sop: SocialOpinions<V>,
    fop: FriendOpinions<V>,
}

#[derive(Debug, Default)]
struct BaseOpinions<V: Float> {
    psi: OpinionD1<Psi, V>,
    phi: OpinionD1<Phi, V>,
    s: OpinionD1<S, V>,
    o: OpinionD1<O, V>,
}

#[derive(Debug, Default)]
pub struct SocialOpinions<V: Float> {
    ko: OpinionD1<KO, V>,
    ks: OpinionD1<KS, V>,
    kphi: OpinionD1<KPhi, V>,
}

#[derive(Debug, Default)]
pub struct FriendOpinions<V: Float> {
    fo: OpinionD1<FO, V>,
    fs: OpinionD1<FS, V>,
    fphi: OpinionD1<FPhi, V>,
}

#[derive(Debug)]
pub struct TempOpinions<V: Float> {
    theta: OpinionD1<Theta, V>,
    b: OpinionD1<B, V>,
    fpsi_ded: OpinionD1<FPsi, V>,
}

impl<V: Float> TempOpinions<V> {
    #[inline]
    pub fn get_theta_projection(&self) -> MArrD1<Theta, V>
    where
        V: NumAssign + fmt::Debug,
    {
        self.theta.projection()
    }
}

impl<V> BaseOpinions<V>
where
    V: Float + NumAssign + Sum + fmt::Debug,
{
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
        ftheta: &OpinionD1<FTheta, V>,
        conds: &BaseConditionalOpinions<V>,
        base_rates: &BaseRates<V>,
    ) -> OpinionD1<A, V>
    where
        V: Float + UlpsEq + NumAssign + Sum + Default,
    {
        ftheta
            .deduce(&conds.cond_a)
            .unwrap_or_else(|| OpinionD1::vacuous_with(base_rates.a.clone()))
    }

    fn compute_b(
        &self,
        ktheta: &OpinionD1<KTheta, V>,
        conds: &BaseConditionalOpinions<V>,
        base_rates: &BaseRates<V>,
    ) -> OpinionD1<B, V>
    where
        V: Float + UlpsEq + NumAssign + Sum + Default,
    {
        let mut b = ktheta
            .deduce(&conds.cond_b)
            .unwrap_or_else(|| OpinionD1::vacuous_with(base_rates.b.clone()));
        let b_abd = self
            .o
            .simplex
            .abduce(&conds.cond_o, base_rates.b.clone())
            .unwrap_or_else(|| OpinionD1::vacuous_with(base_rates.b.clone()));
        FuseOp::ACm.fuse_assign(&mut b, &b_abd);
        b
    }

    fn compute_theta(
        &self,
        b: &OpinionD1<B, V>,
        conds: &BaseConditionalOpinions<V>,
        base_rates: &BaseRates<V>,
    ) -> OpinionD1<Theta, V>
    where
        V: Float + UlpsEq + NumAssign + Sum + Default,
    {
        let mut theta_ded = OpinionD2::product2(b.as_ref(), self.psi.as_ref())
            .deduce(&conds.cond_theta)
            .unwrap_or_else(|| OpinionD1::vacuous_with(base_rates.theta.clone()));
        let theta_ded_2 = self
            .phi
            .deduce(&conds.cond_theta_phi)
            .unwrap_or_else(|| OpinionD1::vacuous_with(base_rates.theta.clone()));
        FuseOp::Wgh.fuse_assign(&mut theta_ded, &theta_ded_2);
        theta_ded
    }

    fn compute_thetad(
        &self,
        a: &OpinionD1<A, V>,
        b: &OpinionD1<B, V>,
        conds: &BaseConditionalOpinions<V>,
        base_rates: &BaseRates<V>,
    ) -> OpinionD1<Thetad, V>
    where
        V: Float + UlpsEq + NumAssign + Sum + Default,
    {
        let thetad_ded_2 = self
            .phi
            .deduce(&conds.cond_thetad_phi)
            .unwrap_or_else(|| OpinionD1::vacuous_with(base_rates.thetad.clone()));
        let mut thetad_ded_1 = OpinionD3::product3(a.as_ref(), b.as_ref(), self.psi.as_ref())
            .deduce(&conds.cond_thetad)
            .unwrap_or_else(|| OpinionD1::vacuous_with(base_rates.thetad.clone()));
        FuseOp::Wgh.fuse_assign(&mut thetad_ded_1, &thetad_ded_2);
        thetad_ded_1
    }
}

impl<V: Float + fmt::Debug> SocialOpinions<V> {
    fn new(&self, info: &Info<V>, trust: V) -> Self
    where
        V: UlpsEq + NumAssign + Sum,
    {
        // compute friends opinions
        let ko = FuseOp::Wgh.fuse(&self.ko, &info.content.o.discount(trust).conv());
        let ks = FuseOp::Wgh.fuse(&self.ks, &info.content.s.discount(trust).conv());
        let kphi = FuseOp::Wgh.fuse(&self.kphi, &info.content.phi.discount(trust).conv());

        Self { ko, ks, kphi }
    }

    fn compute_ktheta(
        &self,
        kb: &OpinionD1<KB, V>,
        kpsi: &OpinionD1<KPsi, V>,
        conds: &SocialConditionalOpinions<V>,
        base_rates: &SocialBaseRates<V>,
    ) -> OpinionD1<KTheta, V>
    where
        V: UlpsEq + NumAssign + Sum + Default,
    {
        let mut ktheta = OpinionD2::product2(kb, kpsi)
            .deduce(&conds.cond_ktheta)
            .unwrap_or_else(|| OpinionD1::<_, V>::vacuous_with(base_rates.ktheta.clone()));
        let ktheta_ded_2 = self
            .kphi
            .deduce(&conds.cond_ktheta_kphi)
            .unwrap_or_else(|| OpinionD1::<_, V>::vacuous_with(base_rates.ktheta.clone()));
        FuseOp::Wgh.fuse_assign(&mut ktheta, &ktheta_ded_2);

        ktheta
    }

    fn compute_kb(
        &self,
        conds: &SocialConditionalOpinions<V>,
        base_rates: &SocialBaseRates<V>,
    ) -> OpinionD1<KB, V>
    where
        V: UlpsEq + NumAssign + Sum + Default,
    {
        let mut kb = self
            .ks
            .deduce(&conds.cond_kb)
            .unwrap_or_else(|| OpinionD1::vacuous_with(base_rates.kb.clone()));
        let kb_abd = self
            .ko
            .simplex
            .abduce(&conds.cond_ko, base_rates.kb.clone())
            .unwrap_or_else(|| OpinionD1::vacuous_with(base_rates.kb.clone()));

        // Use aleatory cummulative fusion
        FuseOp::ACm.fuse_assign(&mut kb, &kb_abd);
        kb
    }

    fn compute_kpsi(
        info: &Info<V>,
        s: &OpinionD1<S, V>,
        trust: V,
        conds: &SocialConditionalOpinions<V>,
        base_rates: &SocialBaseRates<V>,
    ) -> OpinionD1<KPsi, V>
    where
        V: UlpsEq + NumAssign + Sum + fmt::Debug + Default,
    {
        let mut kpsi = s
            .deduce(&conds.cond_kpsi)
            .unwrap_or_else(|| OpinionD1::vacuous_with(base_rates.kpsi.clone()));
        FuseOp::Wgh.fuse_assign(&mut kpsi, &info.content.psi.discount(trust).conv());
        kpsi
    }
}

impl<V: Float + fmt::Debug> FriendOpinions<V> {
    fn new(&self, info: &Info<V>, trust: V) -> Self
    where
        V: Float + UlpsEq + NumAssign + Sum + fmt::Debug,
    {
        // compute friends opinions
        let fo = FuseOp::Wgh.fuse(&self.fo, &info.content.o.discount(trust).conv());
        let fs = FuseOp::Wgh.fuse(&self.fs, &info.content.s.discount(trust).conv());
        let fphi = FuseOp::Wgh.fuse(&self.fphi, &info.content.phi.discount(trust).conv());

        debug!(target: "    FS", w = ?fs);

        Self { fo, fs, fphi }
    }

    fn deduce_fpsi(
        s: &OpinionD1<S, V>,
        conds: &FriendConditionalOpinions<V>,
        base_rates: &FriendBaseRates<V>,
    ) -> OpinionD1<FPsi, V>
    where
        V: UlpsEq + NumAssign + Sum + Default,
    {
        s.deduce(&conds.cond_fpsi)
            .unwrap_or_else(|| OpinionD1::vacuous_with(base_rates.fpsi.clone()))
    }

    fn compute_fb(
        &self,
        conds: &FriendConditionalOpinions<V>,
        base_rates: &FriendBaseRates<V>,
    ) -> OpinionD1<FB, V>
    where
        V: UlpsEq + NumAssign + Sum + Default,
    {
        let mut fb = self
            .fs
            .deduce(&conds.cond_fb)
            .unwrap_or_else(|| OpinionD1::vacuous_with(base_rates.fb.clone()));
        let fb_abd = self
            .fo
            .simplex
            .abduce(&conds.cond_fo, base_rates.fb.clone())
            .unwrap_or_else(|| OpinionD1::vacuous_with(base_rates.fb.clone()));

        debug!(target: "FB||FS", w = ?fb);
        debug!(target: "FB~|FO", w = ?fb_abd);
        // Use aleatory cummulative fusion
        FuseOp::ACm.fuse_assign(&mut fb, &fb_abd);
        fb
    }

    fn compute_fpsi(info: &Info<V>, fpsi_ded: &OpinionD1<FPsi, V>, trust: V) -> OpinionD1<FPsi, V>
    where
        V: UlpsEq + NumAssign + Sum + Default,
    {
        FuseOp::Wgh.fuse(fpsi_ded, &info.content.psi.discount(trust).conv())
    }

    fn compute_ftheta(
        &self,
        fb: &OpinionD1<FB, V>,
        fpsi: &OpinionD1<FPsi, V>,
        conds: &FriendConditionalOpinions<V>,
        base_rates: &FriendBaseRates<V>,
    ) -> OpinionD1<FTheta, V>
    where
        V: UlpsEq + NumAssign + Sum + Default,
    {
        let mut ftheta = OpinionD2::product2(fb, fpsi)
            .deduce(&conds.cond_ftheta)
            .unwrap_or_else(|| OpinionD1::<_, V>::vacuous_with(base_rates.ftheta.clone()));

        debug!(target: "FTH||0", w = ?ftheta);

        let ftheta_ded_2 = self
            .fphi
            .deduce(&conds.cond_ftheta_fphi)
            .unwrap_or_else(|| OpinionD1::<_, V>::vacuous_with(base_rates.ftheta.clone()));
        FuseOp::Wgh.fuse_assign(&mut ftheta, &ftheta_ded_2);

        ftheta
    }

    fn compute_p_a_thetad(
        &self,
        op: &BaseOpinions<V>,
        temp: &TempOpinions<V>,
        info: &Info<V>,
        friend_trust: V,
        conds: &ConditionalOpinions<V>,
        base_rates: &GlobalBaseRates<V>,
    ) -> MArrD2<A, Thetad, V>
    where
        V: UlpsEq + NumAssign + Sum + Default,
    {
        let fb = self.compute_fb(&conds.friend, &base_rates.friend);
        debug!(target: "    FB", w = ?fb);
        let fpsi = FriendOpinions::compute_fpsi(info, &temp.fpsi_ded, friend_trust);
        debug!(target: "  FPSI", w = ?fpsi);
        let ftheta = self.compute_ftheta(&fb, &fpsi, &conds.friend, &base_rates.friend);
        debug!(target: "   FTH", w = ?ftheta);
        let a = op.compute_a(&ftheta, &conds.base, &base_rates.base);
        debug!(target: "     A", w = ?a);
        let thetad = op.compute_thetad(&a, &temp.b, &conds.base, &base_rates.base);
        debug!(target: "   THd", w = ?thetad);

        let a_thetad = OpinionD2::product2(a.as_ref(), thetad.as_ref());
        let p = a_thetad.projection();
        debug!(target: " A,THd", P = ?p);
        p
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
        V: UlpsEq + NumAssign + Sum + fmt::Debug,
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

        debug!(target: "     S", w = ?self.op.s);
        debug!(target: "   PSI", w = ?self.op.psi);
        debug!(target: " FPSId", w = ?fpsi_ded);
        debug!(target: "    KB", w = ?kb);
        debug!(target: "  KPSI", w = ?kpsi);
        debug!(target: "   KTH", w = ?ktheta);
        debug!(target: "     B", w = ?b);
        debug!(target: "    TH", w = ?theta);

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
    ) -> (FriendOpinions<V>, [MArrD2<A, Thetad, V>; 2])
    where
        V: UlpsEq + Sum + Default + NumAssign + fmt::Debug,
    {
        // current opinions
        let span = span!(Level::DEBUG, "pred-curr");
        let _guard = span.enter();
        let p_a_thetad =
            self.fop
                .compute_p_a_thetad(&self.op, temp, info, friend_trust, conds, base_rates);
        drop(_guard);

        // predict  opinions in case of sharing
        let span = span!(Level::DEBUG, "pred-pred");
        let _guard = span.enter();
        let pred_fop = self.fop.new(info, pred_friend_trust);
        let p_pred_a_thetad =
            pred_fop.compute_p_a_thetad(&self.op, temp, info, pred_friend_trust, conds, base_rates);
        drop(_guard);

        let ps = [p_a_thetad, p_pred_a_thetad];
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
            psi: (psi, base_rates.psi.clone()).into(),
            phi: (phi, base_rates.phi.clone()).into(),
            s: (s, base_rates.s.clone()).into(),
            o: (o, base_rates.o.clone()).into(),
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
            fphi: (fphi, base_rates.fphi.clone()).into(),
            fs: (fs, base_rates.fs.clone()).into(),
            fo: (fo, base_rates.fo.clone()).into(),
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
            kphi: (kphi, base_rates.kphi.clone()).into(),
            ks: (ks, base_rates.ks.clone()).into(),
            ko: (ko, base_rates.ko.clone()).into(),
        }
    }
}

impl<V: Float> ConditionalOpinions<V> {
    pub fn from_init<R: Rng>(init: &InitialConditions<V>, rng: &mut R) -> Self
    where
        V: Float + AddAssign + UlpsEq + fmt::Debug,
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

        let base = BaseConditionalOpinions::from_init(&init.base, rng);
        let friend = FriendConditionalOpinions::from_init(&init.friend, rng);
        let social = SocialConditionalOpinions::from_init(&init.social, rng);

        debug!(target: " TH|0,1", w = ?base.cond_theta[(0, 1)]);
        debug!(target: " TH|1,0", w = ?base.cond_theta[(1, 0)]);
        debug!(target: " TH|1,1", w = ?base.cond_theta[(1, 1)]);
        debug!(target: "FTH|0,1", w = ?friend.cond_ftheta[(0, 1)]);
        debug!(target: "FTH|1,0", w = ?friend.cond_ftheta[(1, 0)]);
        debug!(target: "FTH|1,1", w = ?friend.cond_ftheta[(1, 1)]);
        debug!(target: "KTH|0,1", w = ?social.cond_ktheta[(0, 1)]);
        debug!(target: "KTH|1,0", w = ?social.cond_ktheta[(1, 0)]);
        debug!(target: "KTH|1,1", w = ?social.cond_ktheta[(1, 1)]);

        Self {
            base,
            friend,
            social,
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
        let cond_theta = init.cond_theta.sample(rng);
        let r = init.rel_cond_thetad.sample(rng);
        let cond_thetad = MArrD3::from_fn(|idx| match idx {
            (0, i, j) => r.to_simplex(&cond_theta[(i, j)]).unwrap().conv(),
            _ => r.to_simplex(&cond_theta[(0, 0)]).unwrap().conv(),
        });
        // init.cond_thetad.sample(rng, &cond_theta);
        let cond_theta_phi = MArrD1::from_fn(|i| init.cond_theta_phi[i].sample(rng));
        let cond_thetad_phi = MArrD1::from_fn(|i| r.to_simplex(&cond_theta_phi[i]).unwrap().conv());
        Self {
            cond_o: FromFn::from_fn(|i| init.cond_o[i].sample(rng)),
            cond_b: FromFn::from_fn(|i| init.cond_b[i].sample(rng)),
            cond_theta,
            cond_theta_phi,
            cond_a: FromFn::from_fn(|i| init.cond_a[i].sample(rng)),
            cond_thetad,
            cond_thetad_phi,
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
            cond_fpsi: MArrD1::from_fn(|i| init.cond_fpsi[i].sample(rng)),
            cond_fo: MArrD1::from_fn(|i| init.cond_fo[i].sample(rng)),
            cond_fb: MArrD1::from_fn(|i| init.cond_fb[i].sample(rng)),
            cond_ftheta: init.cond_ftheta.sample(rng),
            cond_ftheta_fphi: MArrD1::from_fn(|i| init.cond_ftheta_fphi[i].sample(rng)),
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
            cond_kpsi: MArrD1::from_fn(|i| init.cond_kpsi[i].sample(rng)),
            cond_ko: MArrD1::from_fn(|i| init.cond_ko[i].sample(rng)),
            cond_kb: MArrD1::from_fn(|i| init.cond_kb[i].sample(rng)),
            cond_ktheta: init.cond_ktheta.sample(rng),
            cond_ktheta_kphi: MArrD1::from_fn(|i| init.cond_ktheta_kphi[i].sample(rng)),
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::ulps_eq;
    use rand::thread_rng;
    use rand_distr::Distribution;
    use subjective_logic::iter::Container;
    use subjective_logic::mul::labeled::SimplexD1;
    use subjective_logic::multi_array::labeled::MArrD1;
    use subjective_logic::{domain::Domain, impl_domain, marr_d1};

    use super::{
        BaseConditionalOpinions, CondThetaDist, InitialBaseConditions, RelativeParam, SimplexDist,
        SimplexParam,
    };

    struct X;
    impl_domain!(X = 2);

    struct Y;
    impl_domain!(Y = 3);

    #[test]
    fn test_simplex_dist_conversion() {
        let mut rng = thread_rng();
        let b = vec![0.1f32, 0.2];
        let u = 0.7;
        let fp = SimplexParam::Fixed(b.clone(), u);
        let d = SimplexDist::<X, _>::try_from(fp).unwrap();
        let s = d.sample(&mut rng);
        assert_eq!(s.belief, MArrD1::try_from(b).unwrap());
        assert_eq!(s.uncertainty, u);

        let edp = SimplexParam::Dirichlet {
            alpha: vec![5.0f32, 5.0, 1.0],
            zeros: None,
        };
        let ed = SimplexDist::<Y, _>::try_from(edp);
        assert!(ed.is_err());
        if let Err(e) = ed {
            println!("{:}", e);
        }
        let dp = SimplexParam::Dirichlet {
            alpha: vec![7.0, 4.0, 1.0],
            zeros: None,
        };
        let d = SimplexDist::<X, _>::try_from(dp);
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
        let d = SimplexDist::<X, _>::try_from(dp);
        assert!(d.is_ok());
        let s = d.unwrap().sample(rng);
        assert_eq!(*s.u(), 0.0f32);
        let dp = SimplexParam::Dirichlet {
            alpha: vec![7.0, 4.0],
            zeros: Some(vec![0]),
        };
        let d = SimplexDist::<X, _>::try_from(dp);
        assert!(d.is_ok());
        let s = d.unwrap().sample(rng);
        assert_eq!(s.b()[0], 0.0f32);
        let dp = SimplexParam::Dirichlet {
            alpha: vec![7.0, 4.0],
            zeros: Some(vec![1]),
        };
        let d = SimplexDist::<X, _>::try_from(dp);
        assert!(d.is_ok());
        let s = d.unwrap().sample(rng);
        assert_eq!(s.b()[1], 0.0f32);

        let dp = SimplexParam::Dirichlet {
            alpha: vec![7.0, 4.0, 1.0],
            zeros: Some(vec![1]),
        };
        let d = SimplexDist::<X, _>::try_from(dp);
        assert!(d.is_err());
        println!("{}", d.err().unwrap());
        let dp = SimplexParam::Dirichlet {
            alpha: vec![7.0, 4.0],
            zeros: Some(vec![1, 1]),
        };
        let d = SimplexDist::<Y, _>::try_from(dp);
        assert!(d.is_err());
        println!("{}", d.err().unwrap());
        let dp = SimplexParam::Dirichlet {
            alpha: vec![7.0, 4.0],
            zeros: Some(vec![3]),
        };
        let d = SimplexDist::<X, _>::try_from(dp);
        assert!(d.is_err());
        println!("{}", d.err().unwrap());
    }

    #[test]
    fn test_relative_param() {
        let w = SimplexD1::<X, _>::new(marr_d1![0.9, 0.1], 0.00);
        let wd = RelativeParam {
            belief: 1.0,
            uncertainty: 1.0,
        }
        .to_simplex(&w)
        .unwrap();
        assert_eq!(w, wd);

        let w = SimplexD1::<X, _>::new(marr_d1![0.0, 0.90], 0.10);
        let wd = RelativeParam {
            belief: 1.0,
            uncertainty: 1.0,
        }
        .to_simplex(&w)
        .unwrap();
        assert_eq!(w, wd);

        let w = SimplexD1::<X, _>::new(marr_d1![0.0, 0.90], 0.10);
        let wd = RelativeParam {
            belief: 1.0,
            uncertainty: 0.0,
        }
        .to_simplex(&w)
        .unwrap();
        assert_eq!(wd.b()[1], 1.0);

        let w = SimplexD1::<X, _>::new(marr_d1![0.2, 0.30], 0.50);
        let wd = RelativeParam {
            belief: 1.0,
            uncertainty: 0.0,
        }
        .to_simplex(&w)
        .unwrap();
        assert_eq!(wd.b()[0], 0.4);
        assert_eq!(wd.b()[1], 0.6);

        let w = SimplexD1::<X, _>::new(marr_d1![0.2, 0.30], 0.50);
        let wd = RelativeParam {
            belief: 0.0,
            uncertainty: 1.2,
        }
        .to_simplex(&w)
        .unwrap();
        assert_eq!(wd.b()[0], 0.4);
        assert_eq!(*wd.u(), 0.6);
    }

    #[test]
    fn test_relative_vacuous() {
        let w0 = SimplexD1::<X, _>::new(marr_d1![0.0, 0.0], 1.0);
        let w1 = SimplexD1::<X, _>::new(marr_d1![0.9, 0.0], 0.1);
        let r = RelativeParam {
            belief: 1.0,
            uncertainty: 1.0,
        };
        let _rw0 = r.to_simplex(&w0).unwrap();
        let _rw1 = r.to_simplex(&w1).unwrap();
    }

    #[test]
    fn test_cond_theta() -> anyhow::Result<()> {
        let mut rng = thread_rng();
        let cond_dist = toml::from_str::<CondThetaDist<f32>>(
            r#"
            b0psi0 = { Fixed = [[0.95, 0.00], 0.05] }
            b1psi1 = { Dirichlet = { alpha = [10.5, 10.5, 9.0]} }
            b0psi1 = { belief = { base = 1.2 }, uncertainty = { base = 1.0 } }
            b1psi0 = { belief = { base = 1.5 }, uncertainty = { base = 1.0 } }
        "#,
        )?;

        let rs = [
            cond_dist.b0psi1.sample(&mut rng),
            cond_dist.b1psi0.sample(&mut rng),
        ];

        for cond in cond_dist.sample_iter(&mut rng).take(10) {
            let w = &cond[(1, 1)];
            let wds = [&cond[(0, 1)], &cond[(1, 0)]];
            let x = w.b()[1] / (1.0 - *w.u());
            for (wd, r) in wds.into_iter().zip(&rs) {
                let xd = wd.b()[1] / (1.0 - *wd.u());
                assert!(r.belief > 1.0 / x || ulps_eq!(x * r.belief, xd));
            }
        }
        Ok(())
    }

    #[test]
    fn test_base_cond() -> anyhow::Result<()> {
        let s = r#"
            psi  = [[0.0, 0.0], 1.0]
            phi  = [[0.0, 0.0], 1.0]
            s    = [[0.0, 0.0], 1.0]
            o    = [[0.0, 0.0], 1.0]
            cond_a = [
                { Fixed = [[0.95, 0.00], 0.05] },
                { Dirichlet = { alpha = [5.0, 5.0, 5.0]} },
            ]
            cond_b = [
                { Fixed = [[0.90, 0.00], 0.10] },
                { Dirichlet = { alpha = [5.0, 5.0, 5.0]} },
            ]
            cond_o = [
                { Fixed = [[1.0, 0.00], 0.00] },
                { Dirichlet = { alpha = [5.0, 5.0, 5.0]} },
            ]
            cond_theta_phi= [
                { Fixed = [[0.00, 0.0], 1.00] },
                { Dirichlet = { alpha = [5.0, 5.0, 5.0]} },
            ]
            [cond_theta]
            b0psi0 = { Fixed = [[0.95, 0.00], 0.05] }
            b1psi1 = { Dirichlet = { alpha = [5.0, 5.0, 1.0]} }
            b0psi1 = { belief = { base = 1.0 }, uncertainty = { base = 1.0, error = {dist = "Standard", low = -0.2, high = 0.2 } } }
            b1psi0 = { belief = { base = 1.0 }, uncertainty = { base = 1.0, error = {dist = "Standard", low = -0.2, high = 0.2 } } }
            [rel_cond_thetad]
            belief = { base = 1.1 }
            uncertainty = { base = 1.1 }
        "#;

        let mut rng = thread_rng();

        let init_base_cond = toml::from_str::<InitialBaseConditions<f32>>(s)?;

        let r = init_base_cond.rel_cond_thetad.sample(&mut rng);
        for _ in 0..10 {
            let base_cond = BaseConditionalOpinions::from_init(&init_base_cond, &mut rng);
            for (k, wd) in base_cond.cond_thetad.iter_with() {
                // let wd = &[k];
                let w = match k {
                    (0, i, j) => &base_cond.cond_theta[(i, j)],
                    _ => &base_cond.cond_theta[(0, 0)],
                };
                let xd = wd.b()[1] / (1.0 - *wd.u());
                let x = w.b()[1] / (1.0 - *w.u());
                assert!(r.belief > 1.0 / x || ulps_eq!(x * r.belief, xd));
            }
            for (w, wd) in base_cond
                .cond_theta_phi
                .iter()
                .zip(&base_cond.cond_thetad_phi)
            {
                if w.is_vacuous() {
                    assert!(wd.is_vacuous());
                } else {
                    let xd = wd.b()[1] / (1.0 - *wd.u());
                    let x = w.b()[1] / (1.0 - *w.u());
                    assert!(r.belief > 1.0 / x || ulps_eq!(x * r.belief, xd));
                }
            }
        }
        Ok(())
    }
}
