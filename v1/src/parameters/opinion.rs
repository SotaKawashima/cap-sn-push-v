use std::fmt::Debug;

use itertools::Itertools;
use num_traits::Float;
use rand::Rng;
use rand_distr::Dirichlet;
use rand_distr::{Distribution, Exp1, Open01, Standard, StandardNormal};
use serde_with::{serde_as, TryFromInto};
use std::marker::PhantomData;
use subjective_logic::multi_array::labeled::MArrD2;
use subjective_logic::{
    approx_ext,
    domain::{Domain, Keys},
    ops::{Indexes, Zeros},
};
use subjective_logic::{
    iter::FromFn,
    mul::labeled::{OpinionD1, SimplexD1},
    multi_array::labeled::MArrD1,
};
use tracing::debug;

use base::{
    opinion::{
        FPhi, FPsi, FixedOpinions, KPhi, KPsi, MyFloat, MyOpinions, Phi, Psi, Theta, Thetad, A, B,
        FH, FO, H, KH, KO, O,
    },
    util::Reset,
};
use input::value::{EValue, EValueParam};

#[serde_as]
#[derive(Debug, serde::Deserialize, Clone)]
#[serde(bound(deserialize = "V: serde::Deserialize<'de>"))]
struct InitialBaseRates<V> {
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    a: MArrD1<A, V>,
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    b: MArrD1<B, V>,
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    h: MArrD1<H, V>,
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    fh: MArrD1<FH, V>,
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    kh: MArrD1<KH, V>,
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    theta: MArrD1<Theta, V>,
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    thetad: MArrD1<Thetad, V>,
}

#[serde_as]
#[derive(Debug, serde::Deserialize, Clone)]
#[serde(bound(deserialize = "V: serde::Deserialize<'de>"))]
struct InitialState<V: MyFloat> {
    #[serde_as(as = "TryFromInto<(Vec<V>, V, Vec<V>)>")]
    psi: OpinionD1<Psi, V>,
    #[serde_as(as = "TryFromInto<(Vec<V>, V, Vec<V>)>")]
    phi: OpinionD1<Phi, V>,
    #[serde_as(as = "TryFromInto<(Vec<V>, V, Vec<V>)>")]
    o: OpinionD1<O, V>,
    #[serde_as(as = "TryFromInto<(Vec<V>, V, Vec<V>)>")]
    fo: OpinionD1<FO, V>,
    #[serde_as(as = "TryFromInto<(Vec<V>, V, Vec<V>)>")]
    ko: OpinionD1<KO, V>,
    #[serde_as(as = "TryFromInto<(Vec<V>, V, Vec<V>)>")]
    fpsi: OpinionD1<FPsi, V>,
    #[serde_as(as = "TryFromInto<(Vec<V>, V, Vec<V>)>")]
    fphi: OpinionD1<FPhi, V>,
    #[serde_as(as = "TryFromInto<(Vec<V>, V, Vec<V>)>")]
    kpsi: OpinionD1<KPsi, V>,
    #[serde_as(as = "TryFromInto<(Vec<V>, V, Vec<V>)>")]
    kphi: OpinionD1<KPhi, V>,
    #[serde_as(as = "TryFromInto<Vec<(Vec<V>, V)>>")]
    h_psi_if_phi1: MArrD1<Psi, SimplexD1<H, V>>,
    #[serde_as(as = "TryFromInto<Vec<(Vec<V>, V)>>")]
    h_b_if_phi1: MArrD1<B, SimplexD1<H, V>>,
    #[serde_as(as = "TryFromInto<Vec<(Vec<V>, V)>>")]
    fh_fpsi_if_fphi1: MArrD1<FPsi, SimplexD1<FH, V>>,
    #[serde_as(as = "TryFromInto<Vec<(Vec<V>, V)>>")]
    kh_kpsi_if_kphi1: MArrD1<KPsi, SimplexD1<KH, V>>,
}

#[serde_as]
#[derive(Debug, serde::Deserialize)]
#[serde(bound(deserialize = "V: serde::Deserialize<'de>"))]
struct InitialFixed<V>
where
    V: MyFloat,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    Open01: Distribution<V>,
{
    /// $H^F \implies A$
    #[serde_as(as = "TryFromInto<Vec<SimplexParam<V>>>")]
    a_fh: MArrD1<FH, SimplexDist<A, V>>,
    /// $H^K \implies B$
    #[serde_as(as = "TryFromInto<Vec<SimplexParam<V>>>")]
    b_kh: MArrD1<KH, SimplexDist<B, V>>,
    /// $B \implies O$
    #[serde_as(as = "TryFromInto<Vec<SimplexParam<V>>>")]
    o_b: MArrD1<B, SimplexDist<O, V>>,
    /// $H \implies \Theta$
    #[serde_as(as = "TryFromInto<Vec<SimplexParam<V>>>")]
    theta_h: MArrD1<H, SimplexDist<Theta, V>>,
    /// $H \implies \Theta'$
    #[serde_as(as = "TryFromInto<Vec<SimplexParam<V>>>")]
    thetad_h: MArrD1<H, SimplexDist<Thetad, V>>,
    /// parameters of conditional opinions $\Psi \implies H$ and $B \implies H$ when $\phi_0$ is true
    h_psi_b_if_phi0: ConditionParams<Psi, B, H, V>,
    // /// parameters of conditional opinions $\Psi^F \implies H^F$ when $\phi^F_0$ is true
    // #[serde_as(as = "TryFromInto<DependentParam<FH, V, Vec<SimplexParam<V>>>>")]
    // fh_fpsi_if_fphi0: DependentParam<FH, V, Vec<SimplexDist<FH, V>>>,
    // /// parameters of conditional opinions $\Psi^K \implies H^K$ when $\phi^K_0$ is true
    // #[serde_as(as = "TryFromInto<DependentParam<KH, V, Vec<SimplexParam<V>>>>")]
    // kh_kpsi_if_kphi0: DependentParam<KH, V, Vec<SimplexDist<KH, V>>>,
    #[serde_as(as = "TryFromInto<Vec<EValueParam<V>>>")]
    uncertainty_fh_fpsi_if_fphi0: MArrD1<FPsi, EValue<V>>,
    #[serde_as(as = "TryFromInto<Vec<EValueParam<V>>>")]
    uncertainty_kh_kpsi_if_kphi0: MArrD1<KPsi, EValue<V>>,
    #[serde_as(as = "TryFromInto<Vec<Vec<EValueParam<V>>>>")]
    uncertainty_fh_fo_fphi: MArrD2<FO, FPhi, EValue<V>>,
    #[serde_as(as = "TryFromInto<Vec<Vec<EValueParam<V>>>>")]
    uncertainty_kh_ko_kphi: MArrD2<KO, KPhi, EValue<V>>,
}

#[serde_as]
#[derive(Debug, serde::Deserialize)]
#[serde(bound(deserialize = "V: serde::Deserialize<'de>"))]
pub struct InitialOpinions<V>
where
    V: MyFloat,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    Open01: Distribution<V>,
{
    state: InitialState<V>,
    fixed: InitialFixed<V>,
    deduction_base_rates: InitialBaseRates<V>,
}

impl<V> Reset<MyOpinions<V>> for InitialOpinions<V>
where
    V: MyFloat,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    Open01: Distribution<V>,
{
    fn reset<R: Rng>(&self, value: &mut MyOpinions<V>, rng: &mut R) {
        let InitialState {
            psi,
            phi,
            o,
            fo,
            ko,
            fpsi,
            fphi,
            kpsi,
            kphi,
            h_psi_if_phi1,
            h_b_if_phi1,
            fh_fpsi_if_fphi1,
            kh_kpsi_if_kphi1,
        } = self.state.clone();
        value.state.reset(
            psi,
            phi,
            o,
            fo,
            ko,
            h_psi_if_phi1,
            h_b_if_phi1,
            fpsi,
            fphi,
            fh_fpsi_if_fphi1,
            kpsi,
            kphi,
            kh_kpsi_if_kphi1,
        );
        let InitialBaseRates {
            a,
            b,
            h,
            fh,
            kh,
            theta,
            thetad,
        } = self.deduction_base_rates.clone();
        value.ded.reset(
            OpinionD1::vacuous_with(h),
            OpinionD1::vacuous_with(fh),
            OpinionD1::vacuous_with(kh),
            OpinionD1::vacuous_with(a),
            OpinionD1::vacuous_with(b),
            OpinionD1::vacuous_with(theta),
            OpinionD1::vacuous_with(thetad),
        );
        self.fixed.reset(&mut value.fixed, rng);
        debug!("{:?}", self.fixed);
    }
}

impl<V> Reset<FixedOpinions<V>> for InitialFixed<V>
where
    V: MyFloat,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    Open01: Distribution<V>,
{
    fn reset<R: Rng>(&self, value: &mut FixedOpinions<V>, rng: &mut R) {
        let (h_b_if_phi0, h_psi_if_phi0) = {
            let mut x = self.h_psi_b_if_phi0.sample(rng);
            (
                MArrD1::<B, _>::new(x.pop().unwrap()),
                MArrD1::<Psi, _>::new(x.pop().unwrap()),
            )
        };
        // let fh_fpsi_if_fphi0 = {
        //     let base =
        //         MArrD1::<FPhi, SimplexD1<FH, _>>::from_fn(|i| h_psi_if_phi0[i].clone().conv());
        //     let x = self.fh_fpsi_if_fphi0.samples1(rng, base.iter());
        //     MArrD1::<FPsi, _>::new(x)
        // };
        // let kh_kpsi_if_kphi0 = {
        //     let base =
        //         MArrD1::<KPhi, SimplexD1<KH, _>>::from_fn(|i| h_psi_if_phi0[i].clone().conv());
        //     let x = self.kh_kpsi_if_kphi0.samples1(rng, base.iter());
        //     MArrD1::<KPsi, _>::new(x)
        // };

        value.reset(
            MArrD1::from_fn(|i| self.o_b[i].sample(rng)),
            MArrD1::from_fn(|i| self.b_kh[i].sample(rng)),
            MArrD1::from_fn(|i| self.a_fh[i].sample(rng)),
            MArrD1::from_fn(|i| self.theta_h[i].sample(rng)),
            MArrD1::from_fn(|i| self.thetad_h[i].sample(rng)),
            h_psi_if_phi0,
            h_b_if_phi0,
            MArrD1::from_fn(|i| self.uncertainty_fh_fpsi_if_fphi0[i].sample(rng)),
            MArrD1::from_fn(|i| self.uncertainty_kh_kpsi_if_kphi0[i].sample(rng)),
            MArrD2::from_fn(|i| self.uncertainty_fh_fo_fphi[i].sample(rng)),
            MArrD2::from_fn(|i| self.uncertainty_kh_ko_kphi[i].sample(rng)),
        )
    }
}
// fn reset(&mut self, base_rates: &InitialBaseRates<V>) {
//     *self = Self {
//         a: OpinionD1::vacuous_with(a),
//         b: OpinionD1::vacuous_with(b),
//         h: OpinionD1::vacuous_with(h),
//         fh: OpinionD1::vacuous_with(fh),
//         kh: OpinionD1::vacuous_with(kh),
//         theta: OpinionD1::vacuous_with(theta),
//         thetad: OpinionD1::vacuous_with(thetad),
//     }
// }

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
    V: MyFloat,
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
    V: MyFloat,
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
    V: MyFloat,
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

// (MArrD1<C0, SimplexD1<D, V>>, MArrD1<C1, SimplexD1<D, V>>)
impl<C0, C1, D, V> Distribution<Vec<Vec<SimplexD1<D, V>>>> for ConditionParams<C0, C1, D, V>
where
    C0: Domain,
    C1: Domain,
    D: Domain<Idx = usize>,
    V: MyFloat,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    Open01: Distribution<V>,
{
    fn sample<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        // ) -> (MArrD1<C0, SimplexD1<D, V>>, MArrD1<C1, SimplexD1<D, V>>)
    ) -> Vec<Vec<SimplexD1<D, V>>> {
        let no_cause = self.no_cause.sample(rng);
        let by_cause0 = self.by_cause0.sample(rng);
        let by_cause1 = self.by_cause1.sample(rng, &by_cause0);
        vec![vec![no_cause.clone(), by_cause0], vec![no_cause, by_cause1]]
    }
}

#[serde_as]
#[derive(Debug, serde::Deserialize)]
#[serde(bound(deserialize = "V: serde::Deserialize<'de>, T: serde::Deserialize<'de>"))]
pub enum DependentParam<D, V, T>
where
    D: Domain<Idx = usize>,
    V: MyFloat,
{
    Rel(RelativeParam<D, V>),
    Abs(T),
}

impl<D, V> TryFrom<DependentParam<D, V, SimplexParam<V>>>
    for DependentParam<D, V, SimplexDist<D, V>>
where
    D: Domain<Idx = usize>,
    V: MyFloat,
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

impl<D, V> TryFrom<DependentParam<D, V, Vec<SimplexParam<V>>>>
    for DependentParam<D, V, Vec<SimplexDist<D, V>>>
where
    D: Domain<Idx = usize>,
    V: MyFloat,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    Open01: Distribution<V>,
{
    type Error = <SimplexParam<V> as TryInto<SimplexDist<D, V>>>::Error;

    fn try_from(value: DependentParam<D, V, Vec<SimplexParam<V>>>) -> Result<Self, Self::Error> {
        match value {
            DependentParam::Rel(rel) => Ok(Self::Rel(rel)),
            DependentParam::Abs(abs) => Ok(Self::Abs(
                abs.into_iter().map(|p| p.try_into()).try_collect()?,
            )),
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
    V: MyFloat,
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

#[allow(dead_code)]
impl<D, V> RelativeParam<D, V>
where
    D: Domain<Idx = usize> + Keys<D::Idx>,
    V: MyFloat,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    Open01: Distribution<V>,
{
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R, base: &SimplexD1<D, V>) -> SimplexD1<D, V> {
        relative_sample(base, &self.coef_b, self.coef_u, self.error, rng)
    }

    pub fn samples1<'a, R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        base_iter: impl Iterator<Item = &'a SimplexD1<D, V>>,
    ) -> Vec<SimplexD1<D, V>>
    where
        V: 'a,
        D: 'a,
    {
        base_iter
            .map(|base| relative_sample(&base, &self.coef_b, self.coef_u, self.error, rng))
            .collect()
        // (
        //     T::map(|d| relative_sample(&base.0[d], &self.coef_b, self.coef_u, self.error, rng)),
        //     U::map(|d| relative_sample(&base.1[d], &self.coef_b, self.coef_u, self.error, rng)),
        // )
    }
    pub fn samples2<'a, R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        base_iters: Vec<Box<dyn Iterator<Item = &SimplexD1<D, V>> + 'a>>,
    ) -> Vec<Vec<SimplexD1<D, V>>> {
        base_iters
            .into_iter()
            .map(|iter| {
                iter.map(|base| relative_sample(&base, &self.coef_b, self.coef_u, self.error, rng))
                    .collect()
            })
            .collect()
        // (
        //     T::map(|d| relative_sample(&base.0[d], &self.coef_b, self.coef_u, self.error, rng)),
        //     U::map(|d| relative_sample(&base.1[d], &self.coef_b, self.coef_u, self.error, rng)),
        // )
    }
}

impl<D, V> DependentParam<D, V, SimplexDist<D, V>>
where
    D: Domain<Idx = usize> + Keys<D::Idx>,
    V: MyFloat,
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

#[allow(dead_code)]
impl<D, V, S> DependentParam<D, V, S>
where
    D: Domain<Idx = usize> + Keys<D::Idx>,
    V: MyFloat,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    Open01: Distribution<V>,
{
    pub fn samples2<'a, R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        base_iters: Vec<Box<dyn Iterator<Item = &SimplexD1<D, V>> + 'a>>,
    ) -> Vec<Vec<SimplexD1<D, V>>>
    // ) -> (T::Map<SimplexD1<D, V>>, U::Map<SimplexD1<D, V>>)
    where
        S: Distribution<Vec<Vec<SimplexD1<D, V>>>>,
        // (T::Map<SimplexD1<D, V>>, U::Map<SimplexD1<D, V>>)>,
        // TI: Domain,
        // UI: Domain,
        // T: Container<TI::Idx, Output = SimplexD1<D, V>> + ContainerMap<TI::Idx>,
        // U: Container<UI::Idx, Output = SimplexD1<D, V>> + ContainerMap<UI::Idx>,
    {
        match self {
            Self::Abs(dist) => dist.sample(rng),
            Self::Rel(rel) => rel.samples2(rng, base_iters),
        }
    }
}

#[allow(dead_code)]
impl<D, V> DependentParam<D, V, Vec<SimplexDist<D, V>>>
where
    D: Domain<Idx = usize>,
    V: MyFloat,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    Open01: Distribution<V>,
{
    pub fn samples1<'a, R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        base_iter: impl Iterator<Item = &'a SimplexD1<D, V>>,
    ) -> Vec<SimplexD1<D, V>>
    where
        V: 'a,
        D: 'a,
    {
        match self {
            Self::Abs(dists) => dists.iter().map(|dist| dist.sample(rng)).collect(),
            Self::Rel(rel) => rel.samples1(rng, base_iter),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs::read_to_string;

    use super::InitialOpinions;
    use super::SimplexDist;
    use subjective_logic::marr_d1;

    #[test]
    fn test_toml() -> anyhow::Result<()> {
        let initial_opinions = toml::from_str::<InitialOpinions<f32>>(&read_to_string(
            "./test/config/test_initial_opinions.toml",
        )?)?;
        assert!(matches!(
            &initial_opinions.fixed.theta_h[0],
            SimplexDist::Fixed(s) if s.b() == &marr_d1![1.0, 0.0] && s.u() == &0.0,
        ));
        assert!(matches!(
            &initial_opinions.fixed.theta_h[1],
            SimplexDist::Fixed(s) if s.b() == &marr_d1![0.0, 0.7] && s.u() == &0.3,
        ));
        assert!(initial_opinions.deduction_base_rates.a == marr_d1![0.99, 0.01]);
        Ok(())
    }
}
