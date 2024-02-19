use num_traits::Float;
use rand::Rng;
use rand_distr::{Distribution, Open01, Standard};

use crate::dist::{Dist, DistParam, RangedDist};

pub type EValueParam<V> = ValueWithError<V, DistParam<V>>;
pub type EValue<V> = ValueWithError<V, Dist<V>>;

#[derive(serde::Deserialize, Debug)]
pub struct ValueWithError<V: Float, D> {
    base: V,
    error: Option<RangedDist<V, D>>,
}

impl<V> EValue<V>
where
    V: Float,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
{
    pub fn fixed(base: V) -> Self {
        Self { base, error: None }
    }
}

impl<V> TryFrom<EValueParam<V>> for EValue<V>
where
    V: Float,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
{
    type Error = <RangedDist<V, Dist<V>> as TryFrom<RangedDist<V, DistParam<V>>>>::Error;

    fn try_from(value: EValueParam<V>) -> Result<Self, Self::Error> {
        Ok(ValueWithError {
            base: value.base,
            error: value.error.map(|e| e.try_into()).transpose()?,
        })
    }
}

impl<V> Distribution<V> for EValue<V>
where
    V: Float,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> V {
        let Some(ev) = &self.error else {
            return self.base;
        };
        self.base + ev.sample(rng)
    }
}

#[cfg(test)]
mod tests {
    use rand::thread_rng;
    use rand_distr::Distribution;

    use super::{DistParam, EValue, RangedDist, ValueWithError};

    #[test]
    fn test_sample() {
        let s = ValueWithError {
            base: 1.0,
            error: Some(RangedDist {
                dist: DistParam::<f32>::Standard,
                low: Some(-2.0),
                high: Some(3.0),
            }),
        };
        let v = EValue::try_from(s).unwrap();
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
