use num_traits::{Float, ToPrimitive};
use rand::Rng;
use rand_distr::{Beta, Distribution, Exp1, Gamma, Open01, Standard, StandardNormal};
use serde::Deserialize;

#[derive(serde::Deserialize, Debug)]
pub struct RangedDist<V: Float, D> {
    pub dist: D,
    pub low: Option<V>,
    pub high: Option<V>,
}

#[derive(serde::Deserialize, Debug)]
pub enum DistParam<V: Float> {
    Beta { alpha: V, beta: V },
    Standard,
}

#[derive(Debug)]
pub enum Dist<V>
where
    V: Float,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
{
    Beta(Beta<V>),
    Standard(Standard),
}

impl<V> TryFrom<DistParam<V>> for Dist<V>
where
    V: Float,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
{
    type Error = rand_distr::BetaError;

    fn try_from(value: DistParam<V>) -> Result<Self, Self::Error> {
        let dist = match value {
            DistParam::Beta { alpha, beta } => Dist::Beta(Beta::new(alpha, beta)?),
            DistParam::Standard => Dist::Standard(Standard),
        };
        Ok(dist)
    }
}

impl<V> TryFrom<RangedDist<V, DistParam<V>>> for RangedDist<V, Dist<V>>
where
    V: Float,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
{
    type Error = <Dist<V> as TryFrom<DistParam<V>>>::Error;

    fn try_from(value: RangedDist<V, DistParam<V>>) -> Result<Self, Self::Error> {
        Ok(RangedDist {
            dist: value.dist.try_into()?,
            low: value.low,
            high: value.high,
        })
    }
}

impl<V> Distribution<V> for Dist<V>
where
    V: Float,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> V {
        match self {
            Dist::Beta(dist) => dist.sample(rng),
            Dist::Standard(dist) => dist.sample(rng),
        }
    }
}

impl<V, D> Distribution<V> for RangedDist<V, D>
where
    V: Float,
    D: Distribution<V>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> V {
        let v = self.dist.sample(rng);
        match (self.low, self.high) {
            (None, None) => v,
            (Some(low), None) => v * (V::one() - low) + low,
            (None, Some(high)) => v * high,
            (Some(low), Some(high)) => v * (high - low) + low,
        }
    }
}

#[derive(Debug, Deserialize)]
pub enum IValueParam<F> {
    Fixed(u32),
    Gamma { shape: F, scale: F },
}

#[derive(Debug)]
pub enum IValue<F>
where
    F: Float,
    StandardNormal: Distribution<F>,
    Exp1: Distribution<F>,
    Open01: Distribution<F>,
{
    Fixed(u32),
    Gamma(Gamma<F>),
}

impl<F> TryFrom<IValueParam<F>> for IValue<F>
where
    F: Float,
    StandardNormal: Distribution<F>,
    Exp1: Distribution<F>,
    Open01: Distribution<F>,
{
    type Error = rand_distr::GammaError;

    fn try_from(value: IValueParam<F>) -> Result<Self, Self::Error> {
        match value {
            IValueParam::Fixed(v) => Ok(IValue::Fixed(v)),
            IValueParam::Gamma { shape, scale } => Ok(IValue::Gamma(Gamma::new(shape, scale)?)),
        }
    }
}

impl<F> Distribution<u32> for IValue<F>
where
    F: Float + ToPrimitive,
    StandardNormal: Distribution<F>,
    Exp1: Distribution<F>,
    Open01: Distribution<F>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> u32 {
        match self {
            IValue::Fixed(v) => *v,
            IValue::Gamma(dist) => dist.sample(rng).floor().to_u32().unwrap(),
        }
    }
}
