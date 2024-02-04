use either::Either::{self, Left, Right};
use num_traits::Float;
use rand::Rng;
use rand_distr::{uniform::SampleUniform, Beta, Distribution, Open01, Standard, Uniform};

#[derive(serde::Deserialize, Debug)]
pub enum DistParam<V: Float> {
    Fixed { value: V },
    Beta { alpha: V, beta: V },
    Standard,
    Uniform { low: V, high: V },
}

#[derive(thiserror::Error, Debug)]
pub enum DistParamError {
    #[error("")]
    Infallable(#[from] std::convert::Infallible),
    #[error("{0}")]
    Beta(#[from] rand_distr::BetaError),
}

pub enum Dist<V>
where
    V: Float + SampleUniform,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
{
    Beta(Beta<V>),
    Standard(Standard),
    Uniform(Uniform<V>),
}

impl<V> Distribution<V> for Dist<V>
where
    V: Float + SampleUniform,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> V {
        match self {
            Dist::Beta(dist) => dist.sample(rng),
            Dist::Standard(dist) => dist.sample(rng),
            Dist::Uniform(dist) => dist.sample(rng),
        }
    }
}

impl<V> TryFrom<DistParam<V>> for Either<V, Dist<V>>
where
    V: Float + SampleUniform,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
{
    type Error = rand_distr::BetaError;

    fn try_from(value: DistParam<V>) -> Result<Self, Self::Error> {
        match value {
            DistParam::Fixed { value } => Ok(Left(value)),
            DistParam::Beta { alpha, beta } => {
                Beta::new(alpha, beta).map(|dist| Right(Dist::Beta(dist)))
            }
            DistParam::Standard => Ok(Right(Dist::Standard(Standard))),
            DistParam::Uniform { low, high } => Ok(Right(Dist::Uniform(Uniform::new(low, high)))),
        }
    }
}

pub fn sample<V>(dist: &Either<V, Dist<V>>, rng: &mut impl Rng) -> V
where
    V: Float + SampleUniform,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
{
    dist.as_ref()
        .map_left(|v| *v)
        .map_right(|dist| dist.sample(rng))
        .into_inner()
}
