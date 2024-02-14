use num_traits::Float;
use rand::Rng;
use rand_distr::{Beta, Distribution, Open01, Standard};

pub type ParamValue<V> = Value<V, DistParam<V>>;
pub type DistValue<V> = Value<V, Dist<V>>;

#[derive(serde::Deserialize, Debug)]
pub struct Value<V: Float, D> {
    base: V,
    error: Option<ErrorValue<V, D>>,
}

impl<V> DistValue<V>
where
    V: Float,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
{
    pub fn fixed(base: V) -> Self {
        Self { base, error: None }
    }
}

#[derive(serde::Deserialize, Debug)]
pub struct ErrorValue<V: Float, D> {
    dist: D,
    low: Option<V>,
    high: Option<V>,
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

impl<V> TryFrom<ParamValue<V>> for DistValue<V>
where
    V: Float,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
{
    type Error = rand_distr::BetaError;

    fn try_from(Value { base, error }: ParamValue<V>) -> Result<Self, Self::Error> {
        let Some(ev) = error else {
            return Ok(Value { base, error: None });
        };
        let dist = match ev.dist {
            DistParam::Beta { alpha, beta } => Dist::Beta(Beta::new(alpha, beta)?),
            DistParam::Standard => Dist::Standard(Standard),
        };
        Ok(Value {
            base,
            error: Some(ErrorValue {
                dist,
                low: ev.low,
                high: ev.high,
            }),
        })
    }
}

impl<V> Distribution<V> for DistValue<V>
where
    V: Float,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> V {
        let Some(ev) = &self.error else {
            return self.base;
        };
        let e = match &ev.dist {
            Dist::Beta(dist) => dist.sample(rng),
            Dist::Standard(dist) => dist.sample(rng),
        };
        let e = match (ev.low, ev.high) {
            (None, None) => e,
            (Some(low), None) => e * (V::one() - low),
            (None, Some(high)) => e * high,
            (Some(low), Some(high)) => e * (high - low),
        };
        self.base + e
    }
}
