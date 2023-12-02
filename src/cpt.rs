use subjective_logic::mul::IndexedContainer;

#[derive(Default, Debug)]
pub struct CPT {
    alpha: f32,
    beta: f32,
    lambda: f32,
    gamma: f32,
    delta: f32,
}

impl CPT {
    pub fn new(alpha: f32, beta: f32, lambda: f32, gamma: f32, delta: f32) -> Self {
        Self {
            alpha,
            beta,
            lambda,
            gamma,
            delta,
        }
    }

    fn w(p: f32, e: f32) -> f32 {
        let temp = p.powf(e);
        temp / (temp + (1.0 - p).powf(e)).powf(1.0 / e)
    }

    /// Computes a probability weighting function for gains
    fn positive_weight(&self, p: f32) -> f32 {
        Self::w(p, self.gamma)
    }

    /// Computes a probability weighting function for losses
    fn negative_weight(&self, p: f32) -> f32 {
        Self::w(p, self.delta)
    }

    /// Computes a value funciton for positive value
    fn positive_value(&self, x: f32) -> f32 {
        x.powf(self.alpha)
    }

    /// Computes a value funciton for negative value
    fn negative_value(&self, x: f32) -> f32 {
        -self.lambda * x.abs().powf(self.beta)
    }

    /// Computes Choquet integral of a positive function
    fn positive_valuate<P: IndexedContainer<Idx, Output = f32>, Idx: Copy>(
        &self,
        positive_level_sets: &[(f32, Vec<Idx>)],
        prob: &P,
    ) -> f32 {
        positive_level_sets
            .iter()
            .scan((0.0, 0.0), |(w, acc), (o, ids)| {
                let w0 = *w;
                *acc += ids.iter().map(|i| prob[*i]).sum::<f32>();
                *w = self.positive_weight(*acc);
                Some(self.positive_value(*o) * (*w - w0))
            })
            .sum::<f32>()
    }

    /// Computes Choquet integral of a negative function
    fn negative_valuate<P: IndexedContainer<Idx, Output = f32>, Idx: Copy>(
        &self,
        negative_level_sets: &[(f32, Vec<Idx>)],
        prob: &P,
    ) -> f32 {
        negative_level_sets
            .iter()
            .scan((0.0, 0.0), |(w, acc), (o, ids)| {
                let w0 = *w;
                *acc += ids.iter().map(|i| prob[*i]).sum::<f32>();
                *w = self.negative_weight(*acc);
                Some(self.negative_value(*o) * (*w - w0))
            })
            .sum::<f32>()
    }

    /// Computes CPT
    pub fn valuate<P: IndexedContainer<Idx, Output = f32>, Idx: Copy>(
        &self,
        level_sets: &LevelSet<Idx, f32>,
        prob: &P,
    ) -> f32 {
        self.positive_valuate(&level_sets.positive, prob)
            + self.negative_valuate(&level_sets.negative, prob)
    }
}

#[derive(Default)]
pub struct LevelSet<Idx, V> {
    positive: Vec<(V, Vec<Idx>)>,
    negative: Vec<(V, Vec<Idx>)>,
}

macro_rules! impl_level_set {
    ($ft: ty) => {
        impl<Idx> LevelSet<Idx, $ft> {
            pub fn new<T>(outcome: &T) -> Self
            where
                T: IndexedContainer<Idx, Output = $ft>,
                for<'a> &'a T: IntoIterator<Item = &'a $ft>,
                Idx: Copy,
            {
                let mut pos = T::keys()
                    .zip(outcome)
                    .filter(|(_, &o)| o > 0.0)
                    .collect::<Vec<_>>();
                pos.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                let mut positive = Vec::<($ft, Vec<Idx>)>::new();
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
                    .filter(|(_, &o)| o < 0.0)
                    .collect::<Vec<_>>();
                neg.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                let mut negative = Vec::<($ft, Vec<Idx>)>::new();
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
    };
}

impl_level_set!(f32);
impl_level_set!(f64);

#[cfg(test)]
mod tests {
    use crate::cpt::{LevelSet, CPT};
    use approx::ulps_eq;
    use std::ops::Deref;
    use subjective_logic::harr2;

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
        let cpt = CPT::new(0.88, 0.88, 2.25, 0.61, 0.69);
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
        let cpt = CPT::new(0.88, 0.88, 2.25, 0.61, 0.69);
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
}
