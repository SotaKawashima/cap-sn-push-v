use std::{fmt, iter::Sum, ops::AddAssign};

use approx::{ulps_eq, UlpsEq};
use num_traits::Float;
use subjective_logic::{
    iter::Container,
    marr_d1, marr_d2,
    multi_array::labeled::{MArrD1, MArrD2},
};
use tracing::debug;

use crate::opinion::{Theta, Thetad, A};

#[derive(Clone, Default, Debug)]
pub struct CPT<V> {
    alpha: V,
    beta: V,
    lambda: V,
    gamma: V,
    delta: V,
}

impl<V> CPT<V> {
    pub fn reset(&mut self, alpha: V, beta: V, lambda: V, gamma: V, delta: V) {
        self.alpha = alpha;
        self.beta = beta;
        self.lambda = lambda;
        self.gamma = gamma;
        self.delta = delta;
    }

    // pub fn reset_with(&mut self, params: &CptParams<V>, rng: &mut impl Rng)
    // where
    //     V: SampleUniform,
    //     Open01: Distribution<V>,
    //     Standard: Distribution<V>,
    // {
    //     self.reset(
    //         params.alpha.sample(rng),
    //         params.beta.sample(rng),
    //         params.lambda.sample(rng),
    //         params.gamma.sample(rng),
    //         params.delta.sample(rng),
    //     );
    // }

    fn w(p: V, e: V) -> V
    where
        V: Float + UlpsEq + AddAssign + Sum,
    {
        if ulps_eq!(p, V::one()) {
            V::one()
        } else {
            let temp = p.powf(e);
            temp / (temp + (V::one() - p).powf(e)).powf(V::one() / e)
        }
    }

    /// Computes a probability weighting function for gains
    fn positive_weight(&self, p: V) -> V
    where
        V: Float + UlpsEq + AddAssign + Sum,
    {
        Self::w(p, self.gamma)
    }

    /// Computes a probability weighting function for losses
    fn negative_weight(&self, p: V) -> V
    where
        V: Float + UlpsEq + AddAssign + Sum,
    {
        Self::w(p, self.delta)
    }

    /// Computes a value funciton for positive value
    fn positive_value(&self, x: V) -> V
    where
        V: Float + UlpsEq + AddAssign + Sum,
    {
        x.powf(self.alpha)
    }

    /// Computes a value funciton for negative value
    fn negative_value(&self, x: V) -> V
    where
        V: Float + UlpsEq + AddAssign + Sum,
    {
        -self.lambda * x.abs().powf(self.beta)
    }

    /// Computes Choquet integral of a positive function
    fn positive_valuate<P: Container<Idx, Output = V>, Idx: Copy>(
        &self,
        positive_level_sets: &[(V, Vec<Idx>)],
        prob: &P,
    ) -> V
    where
        V: Float + UlpsEq + AddAssign + Sum,
    {
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
    fn negative_valuate<P: Container<Idx, Output = V>, Idx: Copy>(
        &self,
        negative_level_sets: &[(V, Vec<Idx>)],
        prob: &P,
    ) -> V
    where
        V: Float + UlpsEq + AddAssign + Sum,
    {
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
    pub fn valuate<P: Container<Idx, Output = V>, Idx: Copy>(
        &self,
        level_sets: &LevelSet<Idx, V>,
        prob: &P,
    ) -> V
    where
        V: Float + UlpsEq + AddAssign + Sum,
    {
        self.positive_valuate(&level_sets.positive, prob)
            + self.negative_valuate(&level_sets.negative, prob)
    }
}

#[derive(Clone, Debug)]
pub struct LevelSet<Idx, V> {
    positive: Vec<(V, Vec<Idx>)>,
    negative: Vec<(V, Vec<Idx>)>,
}

impl<Idx, V> Default for LevelSet<Idx, V> {
    fn default() -> Self {
        Self {
            positive: Vec::new(),
            negative: Vec::new(),
        }
    }
}

impl<Idx, V: Float> LevelSet<Idx, V> {
    pub fn new<T>(outcome: &T) -> Self
    where
        T: Container<Idx, Output = V>,
        for<'a> &'a T: IntoIterator<Item = &'a V>,
        Idx: Copy,
    {
        let mut pos = T::indexes()
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
        let mut neg = T::indexes()
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

#[derive(Default)]
pub struct Prospect<V> {
    pub selfish: [LevelSet<Theta, V>; 2],
    pub sharing: [LevelSet<(A, Thetad), V>; 2],
}

impl<V> Prospect<V> {
    pub fn reset(&mut self, x0: V, x1: V, y: V)
    where
        V: Float + fmt::Debug,
    {
        let selfish_outcome_maps: [MArrD1<Theta, V>; 2] =
            [marr_d1![V::zero(), x1], marr_d1![x0, x0]];
        let sharing_outcome_maps: [MArrD2<A, Thetad, V>; 2] = [
            marr_d2![[V::zero(), x1], [x0, x0]],
            marr_d2![[y, x1 + y], [x0 + y, x0 + y]],
        ];
        let selfish = [
            LevelSet::new(&selfish_outcome_maps[0]),
            LevelSet::new(&selfish_outcome_maps[1]),
        ];
        let sharing = [
            LevelSet::new(&sharing_outcome_maps[0]),
            LevelSet::new(&sharing_outcome_maps[1]),
        ];

        debug!(target: "X outcomes", x = ?selfish_outcome_maps);
        debug!(target: "Y outcomes", x = ?sharing_outcome_maps);
        self.selfish = selfish;
        self.sharing = sharing;
    }

    // pub fn reset_with(&mut self, loss_params: &LossParams<V>, rng: &mut impl Rng)
    // where
    //     V: SampleUniform,
    //     Open01: Distribution<V>,
    //     Standard: Distribution<V>,
    // {
    //     let x0 = loss_params.x0.sample(rng);
    //     let x1 = x0 * loss_params.x1_of_x0.sample(rng);
    //     let y = x0 * loss_params.y_of_x0.sample(rng);
    //     self.reset(x0, x1, y);
    // }
}

#[cfg(test)]
mod tests {
    use crate::decision::{LevelSet, CPT};
    use approx::ulps_eq;
    use subjective_logic::{domain::Domain, impl_domain, marr_d2};

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

    struct X;
    impl_domain!(X = 2);

    struct Y;
    impl_domain!(Y = 3);

    #[test]
    fn test_cpt_prod() {
        let outcome = marr_d2!(X, Y; [[6.0, 2.0, 4.0], [-3.0, -1.0, -5.0]]);
        let prob = marr_d2!(X, Y; [
            [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0],
            [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0]
        ]);
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
}
