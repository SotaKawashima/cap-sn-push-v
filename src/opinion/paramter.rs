use approx::UlpsEq;
use num_traits::{Float, FromPrimitive};
use rand::Rng;
use rand_distr::{Dirichlet, Distribution, Exp1, Open01, Standard, StandardNormal};
use serde_with::{serde_as, TryFromInto};
use std::{iter::Sum, marker::PhantomData, ops::AddAssign};
use subjective_logic::{
    approx_ext,
    domain::{Domain, Keys},
    iter::{Container, ContainerMap},
    marr_d1,
    mul::labeled::SimplexD1,
    multi_array::labeled::MArrD1,
    ops::{Indexes, Zeros},
};

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
#[serde(bound(deserialize = "V: serde::Deserialize<'de>"))]
pub struct ConditionParams<C0, C1, D, V>
where
    C0: Domain,
    C1: Domain,
    D: Domain<Idx = usize>,
    V: Float + UlpsEq + AddAssign,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    Open01: Distribution<V>,
{
    #[serde(skip)]
    _marker_e0: PhantomData<C0>,
    #[serde(skip)]
    _marker_e1: PhantomData<C1>,
    /// $\omega_{D||c^0_0}$ and $\omega_{D||c^1_0}$
    #[serde_as(as = "TryFromInto<SimplexParam<V>>")]
    pub no_cause: SimplexDist<D, V>,
    /// $\omega_{D||c^0_1}$
    #[serde_as(as = "TryFromInto<SimplexParam<V>>")]
    pub by_cause0: SimplexDist<D, V>,
    /// $\omega_{D||c^1_1}$
    #[serde_as(as = "TryFromInto<DependentParam<D, V, SimplexParam<V>>>")]
    pub by_cause1: DependentParam<D, V, SimplexDist<D, V>>,
}

impl<C0, C1, D, V> Distribution<(MArrD1<C0, SimplexD1<D, V>>, MArrD1<C1, SimplexD1<D, V>>)>
    for ConditionParams<C0, C1, D, V>
where
    C0: Domain,
    C1: Domain,
    D: Domain<Idx = usize>,
    V: Float + AddAssign + UlpsEq + FromPrimitive + Sum + std::fmt::Debug,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    Open01: Distribution<V>,
{
    fn sample<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
    ) -> (MArrD1<C0, SimplexD1<D, V>>, MArrD1<C1, SimplexD1<D, V>>) {
        let no_cause = self.no_cause.sample(rng);
        let by_cause0 = self.by_cause0.sample(rng);
        let by_cause1 = self.by_cause1.sample(rng, &by_cause0);
        (
            marr_d1![no_cause.clone(), by_cause0],
            marr_d1![no_cause, by_cause1],
        )
    }
}

#[serde_as]
#[derive(Debug, serde::Deserialize)]
#[serde(bound(deserialize = "V: serde::Deserialize<'de>, T: serde::Deserialize<'de>"))]
pub enum DependentParam<D, V, T>
where
    D: Domain<Idx = usize>,
    V: Float + UlpsEq + AddAssign,
{
    Rel(RelativeParam<D, V>),
    Abs(T),
}

impl<D, V> TryFrom<DependentParam<D, V, SimplexParam<V>>>
    for DependentParam<D, V, SimplexDist<D, V>>
where
    D: Domain<Idx = usize>,
    V: Float + UlpsEq + AddAssign,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    Open01: Distribution<V>,
{
    type Error = <SimplexParam<V> as TryInto<SimplexDist<D, V>>>::Error;

    fn try_from(value: DependentParam<D, V, SimplexParam<V>>) -> Result<Self, Self::Error> {
        match value {
            DependentParam::Rel(rel) => Ok(Self::Rel(rel)),
            DependentParam::Abs(abs) => Ok(Self::Abs(abs.try_into()?)),
        }
    }
}

#[serde_as]
#[derive(Debug, serde::Deserialize)]
#[serde(bound(deserialize = "V: serde::Deserialize<'de>"))]
pub struct RelativeParam<D: Domain, V: Float> {
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    coef_b: MArrD1<D, V>,
    coef_u: V,
    error: V,
}

fn relative_sample<D, V, R>(
    base: &SimplexD1<D, V>,
    coef_b: &MArrD1<D, V>,
    coef_u: V,
    error: V,
    rng: &mut R,
) -> SimplexD1<D, V>
where
    D: Domain<Idx = usize> + Keys<D::Idx>,
    V: Float + UlpsEq + AddAssign + FromPrimitive + Sum + std::fmt::Debug,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    Open01: Distribution<V>,
    R: Rng + ?Sized,
{
    for c in coef_b.iter().chain([&coef_u]) {
        assert!(*c > V::zero());
    }

    let k = V::from_usize(D::LEN).unwrap() + V::one();
    let alpha = {
        let mut alpha = D::keys()
            .map(|i| base.b()[i] * coef_b[i])
            .collect::<Vec<_>>();
        alpha.push(*base.u() * coef_u);
        let mut has_zero = false;
        for a in &alpha {
            if approx_ext::is_zero(*a) {
                has_zero = true;
                break;
            }
        }
        if has_zero {
            let s = alpha.iter().cloned().sum::<V>();
            for a in &mut alpha {
                *a = (*a + error) / (s + error * k);
            }
        }
        alpha
    };
    let dir = Dirichlet::<V>::new(&alpha).unwrap();
    let mut v = dir.sample(rng);
    let u = v.pop().unwrap();
    SimplexD1::new(MArrD1::new(v), u)
}

impl<D, V> RelativeParam<D, V>
where
    D: Domain<Idx = usize> + Keys<D::Idx>,
    V: Float + UlpsEq + AddAssign + FromPrimitive + Sum + std::fmt::Debug,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    Open01: Distribution<V>,
{
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R, base: &SimplexD1<D, V>) -> SimplexD1<D, V> {
        relative_sample(base, &self.coef_b, self.coef_u, self.error, rng)
    }

    pub fn samples<R: Rng + ?Sized, T, U, TI, UI>(
        &self,
        rng: &mut R,
        base: &(T, U),
    ) -> (T::Map<SimplexD1<D, V>>, U::Map<SimplexD1<D, V>>)
    where
        TI: Domain,
        T: Container<TI::Idx, Output = SimplexD1<D, V>> + ContainerMap<TI::Idx>,
        UI: Domain,
        U: Container<UI::Idx, Output = SimplexD1<D, V>> + ContainerMap<UI::Idx>,
    {
        (
            T::map(|d| relative_sample(&base.0[d], &self.coef_b, self.coef_u, self.error, rng)),
            U::map(|d| relative_sample(&base.1[d], &self.coef_b, self.coef_u, self.error, rng)),
        )
    }
}

impl<D, V> DependentParam<D, V, SimplexDist<D, V>>
where
    D: Domain<Idx = usize> + Keys<D::Idx>,
    V: Float + UlpsEq + AddAssign + FromPrimitive + Sum + std::fmt::Debug,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    Open01: Distribution<V>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R, base: &SimplexD1<D, V>) -> SimplexD1<D, V> {
        match self {
            Self::Abs(dist) => dist.sample(rng),
            Self::Rel(rel) => rel.sample(rng, base),
        }
    }
}

impl<D, V, S> DependentParam<D, V, S>
where
    D: Domain<Idx = usize> + Keys<D::Idx>,
    V: Float + UlpsEq + AddAssign + FromPrimitive + Sum + std::fmt::Debug,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    Open01: Distribution<V>,
{
    pub fn samples<TI, UI, T, U, R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        base: &(T, U),
    ) -> (T::Map<SimplexD1<D, V>>, U::Map<SimplexD1<D, V>>)
    where
        S: Distribution<(T::Map<SimplexD1<D, V>>, U::Map<SimplexD1<D, V>>)>,
        TI: Domain,
        UI: Domain,
        T: Container<TI::Idx, Output = SimplexD1<D, V>> + ContainerMap<TI::Idx>,
        U: Container<UI::Idx, Output = SimplexD1<D, V>> + ContainerMap<UI::Idx>,
    {
        match self {
            Self::Abs(dist) => dist.sample(rng),
            Self::Rel(rel) => rel.samples::<R, T, U, TI, UI>(rng, base),
        }
    }
}
