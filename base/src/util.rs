use rand::Rng;

pub trait Reset<T> {
    fn reset<R: Rng>(&self, value: &mut T, rng: &mut R);
}
