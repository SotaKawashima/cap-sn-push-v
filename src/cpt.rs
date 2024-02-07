use std::{
    iter::Sum,
    ops::{AddAssign, Deref},
};

use approx::{ulps_eq, UlpsEq};
use either::Either;
use num_traits::Float;
use rand::Rng;
use rand_distr::{uniform::SampleUniform, Distribution, Open01, Standard};
use serde_with::{serde_as, TryFromInto};
use subjective_logic::{harr2, mul::IndexedContainer};

use crate::dist::{sample, Dist, DistParam};

#[derive(Clone, Default, Debug)]
pub struct CPT<V: Float> {
    alpha: V,
    beta: V,
    lambda: V,
    gamma: V,
    delta: V,
}

impl<V> CPT<V>
where
    V: Float + UlpsEq + AddAssign + Sum,
{
    pub fn reset(&mut self, alpha: V, beta: V, lambda: V, gamma: V, delta: V) {
        self.alpha = alpha;
        self.beta = beta;
        self.lambda = lambda;
        self.gamma = gamma;
        self.delta = delta;
    }

    pub fn reset_with(&mut self, params: &CptParams<V>, rng: &mut impl Rng)
    where
        V: SampleUniform,
        Open01: Distribution<V>,
        Standard: Distribution<V>,
    {
        self.reset(
            sample(&params.alpha, rng),
            sample(&params.beta, rng),
            sample(&params.lambda, rng),
            sample(&params.gamma, rng),
            sample(&params.delta, rng),
        );
    }

    fn w(p: V, e: V) -> V {
        if ulps_eq!(p, V::one()) {
            V::one()
        } else {
            let temp = p.powf(e);
            temp / (temp + (V::one() - p).powf(e)).powf(V::one() / e)
        }
    }

    /// Computes a probability weighting function for gains
    fn positive_weight(&self, p: V) -> V {
        Self::w(p, self.gamma)
    }

    /// Computes a probability weighting function for losses
    fn negative_weight(&self, p: V) -> V {
        Self::w(p, self.delta)
    }

    /// Computes a value funciton for positive value
    fn positive_value(&self, x: V) -> V {
        x.powf(self.alpha)
    }

    /// Computes a value funciton for negative value
    fn negative_value(&self, x: V) -> V {
        -self.lambda * x.abs().powf(self.beta)
    }

    /// Computes Choquet integral of a positive function
    fn positive_valuate<P: IndexedContainer<Idx, Output = V>, Idx: Copy>(
        &self,
        positive_level_sets: &[(V, Vec<Idx>)],
        prob: &P,
    ) -> V {
        positive_level_sets
            .iter()
            .scan((V::zero(), V::zero()), |(w, acc), (o, ids)| {
                let w0 = *w;
                *acc += ids.iter().map(|i| prob[*i]).sum::<V>();
                *w = self.positive_weight(*acc);
                Some(self.positive_value(*o) * (*w - w0))
            })
            .sum::<V>()
    }

    /// Computes Choquet integral of a negative function
    fn negative_valuate<P: IndexedContainer<Idx, Output = V>, Idx: Copy>(
        &self,
        negative_level_sets: &[(V, Vec<Idx>)],
        prob: &P,
    ) -> V {
        negative_level_sets
            .iter()
            .scan((V::zero(), V::zero()), |(w, acc), (o, ids)| {
                let w0 = *w;
                *acc += ids.iter().map(|i| prob[*i]).sum::<V>();
                *w = self.negative_weight(*acc);
                Some(self.negative_value(*o) * (*w - w0))
            })
            .sum::<V>()
    }

    /// Computes CPT
    pub fn valuate<P: IndexedContainer<Idx, Output = V>, Idx: Copy>(
        &self,
        level_sets: &LevelSet<Idx, V>,
        prob: &P,
    ) -> V {
        self.positive_valuate(&level_sets.positive, prob)
            + self.negative_valuate(&level_sets.negative, prob)
    }
}

#[derive(Default, Clone, Debug)]
pub struct LevelSet<Idx, V> {
    positive: Vec<(V, Vec<Idx>)>,
    negative: Vec<(V, Vec<Idx>)>,
}

impl<Idx, V: Float> LevelSet<Idx, V> {
    pub fn new<T>(outcome: &T) -> Self
    where
        T: IndexedContainer<Idx, Output = V>,
        for<'a> &'a T: IntoIterator<Item = &'a V>,
        Idx: Copy,
    {
        let mut pos = T::keys()
            .zip(outcome)
            .filter(|(_, &o)| o > V::zero())
            .collect::<Vec<_>>();
        pos.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let mut positive = Vec::<(V, Vec<Idx>)>::new();
        for (i, &o) in pos {
            match positive.last_mut() {
                Some((o2, v)) if o == *o2 => {
                    v.push(i);
                }
                _ => {
                    positive.push((o, vec![i]));
                }
            }
        }
        let mut neg = T::keys()
            .zip(outcome)
            .filter(|(_, &o)| o < V::zero())
            .collect::<Vec<_>>();
        neg.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let mut negative = Vec::<(V, Vec<Idx>)>::new();
        for (i, &o) in neg {
            match negative.last_mut() {
                Some((o2, v)) if o == *o2 => {
                    v.push(i);
                }
                _ => {
                    negative.push((o, vec![i]));
                }
            }
        }
        Self { positive, negative }
    }
}

#[serde_as]
#[derive(serde::Deserialize)]
#[serde(bound(deserialize = "V: serde::Deserialize<'de>"))]
pub struct CptParams<V>
where
    V: Float + SampleUniform,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
{
    #[serde_as(as = "TryFromInto<DistParam<V>>")]
    pub x0_dist: Either<V, Dist<V>>,
    #[serde_as(as = "TryFromInto<DistParam<V>>")]
    pub x1_dist: Either<V, Dist<V>>,
    #[serde_as(as = "TryFromInto<DistParam<V>>")]
    pub y_dist: Either<V, Dist<V>>,
    #[serde_as(as = "TryFromInto<DistParam<V>>")]
    pub alpha: Either<V, Dist<V>>,
    #[serde_as(as = "TryFromInto<DistParam<V>>")]
    pub beta: Either<V, Dist<V>>,
    #[serde_as(as = "TryFromInto<DistParam<V>>")]
    pub lambda: Either<V, Dist<V>>,
    #[serde_as(as = "TryFromInto<DistParam<V>>")]
    pub gamma: Either<V, Dist<V>>,
    #[serde_as(as = "TryFromInto<DistParam<V>>")]
    pub delta: Either<V, Dist<V>>,
}

#[derive(Default)]
pub struct Prospect<V: Float> {
    pub selfish: [LevelSet<usize, V>; 2],
    pub sharing: [LevelSet<[usize; 2], V>; 2],
}

impl<V> Prospect<V>
where
    V: Float,
{
    pub fn reset(&mut self, x0: V, x1: V, y: V) {
        let selfish_outcome_maps = [[V::zero(), x1, V::zero()], [x0, x0, x0]];
        let sharing_outcome_maps = [
            harr2![[V::zero(), x1, V::zero()], [x0, x0, x0]],
            harr2![[y, x1 + y, y], [x0 + y, x0 + y, x0 + y]],
        ];
        let selfish = [
            LevelSet::new(&selfish_outcome_maps[0]),
            LevelSet::new(&selfish_outcome_maps[1]),
        ];
        let sharing = [
            LevelSet::new(sharing_outcome_maps[0].deref()),
            LevelSet::new(sharing_outcome_maps[1].deref()),
        ];

        self.selfish = selfish;
        self.sharing = sharing;
    }

    pub fn reset_with(&mut self, params: &CptParams<V>, rng: &mut impl Rng)
    where
        V: SampleUniform,
        Open01: Distribution<V>,
        Standard: Distribution<V>,
    {
        let x0 = sample(&params.x0_dist, rng);
        let x1 = sample(&params.x1_dist, rng);
        let y = sample(&params.y_dist, rng);
        self.reset(x0, x1, y);
    }
}

#[cfg(test)]
mod tests {
    use crate::cpt::{LevelSet, CPT};
    use approx::{assert_ulps_eq, relative_eq, ulps_eq};
    use std::ops::Deref;
    use subjective_logic::{
        harr2,
        mul::{prod::Product2, Opinion1d, OpinionBase, Projection},
    };

    fn v(o: f32) -> f32 {
        if o.is_sign_negative() {
            -2.25 * (-o).powf(0.88)
        } else {
            o.powf(0.88)
        }
    }
    fn wp(p: f32) -> f32 {
        let q = p.powf(0.61);
        q / (q + (1.0 - p).powf(0.61)).powf(1.0 / 0.61)
    }
    fn wm(p: f32) -> f32 {
        let q = p.powf(0.69);
        q / (q + (1.0 - p).powf(0.69)).powf(1.0 / 0.69)
    }

    #[test]
    fn test_cpt() {
        let outcome = [6.0, 2.0, 4.0, -3.0, -1.0, -5.0];
        let prob = [1.0 / 6.0; 6];
        let cpt = CPT {
            alpha: 0.88,
            beta: 0.88,
            lambda: 2.25,
            gamma: 0.61,
            delta: 0.69,
        };
        let ls = LevelSet::<_, f32>::new(&outcome);
        let a = cpt.valuate(&ls, &prob);
        let b = v(2.0) * (wp(1.0 / 2.0) - wp(1.0 / 3.0))
            + v(4.0) * (wp(1.0 / 3.0) - wp(1.0 / 6.0))
            + v(6.0) * (wp(1.0 / 6.0) - wp(0.0))
            + v(-5.0) * (wm(1.0 / 6.0) - wm(0.0))
            + v(-3.0) * (wm(1.0 / 3.0) - wm(1.0 / 6.0))
            + v(-1.0) * (wm(1.0 / 2.0) - wm(1.0 / 3.0));

        let c = cpt.positive_value(2.0)
            * (cpt.positive_weight(1.0 / 2.0) - cpt.positive_weight(1.0 / 3.0))
            + cpt.positive_value(4.0)
                * (cpt.positive_weight(1.0 / 3.0) - cpt.positive_weight(1.0 / 6.0))
            + cpt.positive_value(6.0) * (cpt.positive_weight(1.0 / 6.0) - cpt.positive_weight(0.0))
            + cpt.negative_value(-5.0)
                * (cpt.negative_weight(1.0 / 6.0) - cpt.negative_weight(0.0))
            + cpt.negative_value(-3.0)
                * (cpt.negative_weight(1.0 / 3.0) - cpt.negative_weight(1.0 / 6.0))
            + cpt.negative_value(-1.0)
                * (cpt.negative_weight(1.0 / 2.0) - cpt.negative_weight(1.0 / 3.0));

        assert!(ulps_eq!(a, b));
        assert!(ulps_eq!(a, c));
    }

    #[test]
    fn test_cpt_prod() {
        let outcome = harr2![[6.0, 2.0, 4.0], [-3.0, -1.0, -5.0]];
        let prob = harr2![
            [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0],
            [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0]
        ];
        let cpt = CPT {
            alpha: 0.88,
            beta: 0.88,
            lambda: 2.25,
            gamma: 0.61,
            delta: 0.69,
        };
        let ls = LevelSet::<_, f32>::new(outcome.deref());
        let a = cpt.valuate(&ls, &prob);
        let b = v(2.0) * (wp(1.0 / 2.0) - wp(1.0 / 3.0))
            + v(4.0) * (wp(1.0 / 3.0) - wp(1.0 / 6.0))
            + v(6.0) * (wp(1.0 / 6.0) - wp(0.0))
            + v(-5.0) * (wm(1.0 / 6.0) - wm(0.0))
            + v(-3.0) * (wm(1.0 / 3.0) - wm(1.0 / 6.0))
            + v(-1.0) * (wm(1.0 / 2.0) - wm(1.0 / 3.0));

        let c = cpt.positive_value(2.0)
            * (cpt.positive_weight(1.0 / 2.0) - cpt.positive_weight(1.0 / 3.0))
            + cpt.positive_value(4.0)
                * (cpt.positive_weight(1.0 / 3.0) - cpt.positive_weight(1.0 / 6.0))
            + cpt.positive_value(6.0) * (cpt.positive_weight(1.0 / 6.0) - cpt.positive_weight(0.0))
            + cpt.negative_value(-5.0)
                * (cpt.negative_weight(1.0 / 6.0) - cpt.negative_weight(0.0))
            + cpt.negative_value(-3.0)
                * (cpt.negative_weight(1.0 / 3.0) - cpt.negative_weight(1.0 / 6.0))
            + cpt.negative_value(-1.0)
                * (cpt.negative_weight(1.0 / 2.0) - cpt.negative_weight(1.0 / 3.0));

        assert!(ulps_eq!(a, b));
        assert!(ulps_eq!(a, c));
    }

    #[test]
    fn test_nan() {
        let fa = Opinion1d::<f32, 2>::new(
            [0.090361536, 0.082669996],
            0.8269686,
            [0.9991985, 0.0008016379],
        );
        let theta = Opinion1d::<f32, 3>::new(
            [0.009244099, 0.37928966, 0.41457662],
            0.19688994,
            [0.99831605, 0.00080163794, 0.0008825256],
        );
        let fatheta = OpinionBase::product2(&fa, &theta);

        println!("fa {}", fa.b().into_iter().sum::<f32>() + fa.u());
        println!("fa {}", fa.base_rate.into_iter().sum::<f32>());
        println!("th {}", theta.b().into_iter().sum::<f32>() + theta.u());
        println!("th {}", theta.base_rate.into_iter().sum::<f32>());
        println!(
            "fath {}",
            fatheta.b().into_iter().sum::<f32>() + fatheta.u()
        );
        println!("fath {}", fatheta.base_rate.into_iter().sum::<f32>());

        assert_ulps_eq!(fa.projection().into_iter().sum::<f32>(), 1.0);
        assert_ulps_eq!(theta.projection().into_iter().sum::<f32>(), 1.0);

        let p = fatheta.projection();
        let sump = p.into_iter().sum::<f32>();
        println!("{}", sump);
        // let p = HigherArr2::<f32, 2, 3>::map(|i| p[i] / sump);

        // let fad = Opinion1d::<f32, 2>::new(
        //     fa.b().map(|b| b / ),
        //     0.8269686,
        //     [0.9991985, 0.0008016379],
        // );
        // let theta = Opinion1d::<f32, 3>::new(
        //     [0.009244099, 0.37928966, 0.41457662],
        //     0.19688994,
        //     [0.99831605, 0.00080163794, 0.0008825256],
        // );
        // assert_ulps_eq!(, 1.0);

        // let p = harr2![
        //     [0.18865241, 0.34782714, 0.3801881],
        //     [0.017150123, 0.03162047, 0.03456236]
        // ];
        let x0 = -0.1;
        let x1 = -2.0;
        let y = -0.01;
        let sharing_outcome_maps = harr2![[y, x1 + y, y], [x0 + y, x0 + y, x0 + y]];
        let level_sets = LevelSet::<_, f32>::new(sharing_outcome_maps.deref());
        let cpt = CPT {
            alpha: 0.88,
            beta: 0.88,
            lambda: 2.25,
            gamma: 0.61,
            delta: 0.69,
        };
        let a = p.into_iter().sum::<f32>();
        println!("{}, {}", a, relative_eq!(a, 1.0));
        println!("{:?}", level_sets);
        // println!("{}", cpt.valuate(&level_sets, &p));
        println!("V-={}", cpt.negative_valuate(&level_sets.negative, &p));
    }
}
