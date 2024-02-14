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
            (Some(low), None) => e * (V::one() - low) + low,
            (None, Some(high)) => e * high,
            (Some(low), Some(high)) => e * (high - low) + low,
        };
        self.base + e
    }
}

#[cfg(test)]
mod tests {
    use rand::thread_rng;
    use rand_distr::Distribution;

    use super::{DistParam, DistValue, ErrorValue, Value};

    #[test]
    fn test_sample() {
        let s = Value {
            base: 1.0,
            error: Some(ErrorValue {
                dist: DistParam::<f32>::Standard,
                low: Some(-2.0),
                high: Some(3.0),
            }),
        };
        let v = DistValue::try_from(s).unwrap();
        let rng = &mut thread_rng();
        let mut min_x = f32::MAX;
        let mut max_x = f32::MIN;
        for _ in 0..1000 {
            let x = v.sample(rng) - 1.0;
            min_x = min_x.min(x);
            max_x = max_x.max(x);
        }
        println!("{min_x}, {max_x}");
        assert!(min_x >= -2.0);
        assert!(max_x <= 3.0);
    }
}
