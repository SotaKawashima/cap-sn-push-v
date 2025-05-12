use base::{
    decision::{Prospect, CPT},
    opinion::{
        DeducedOpinions, FPhi, FPsi, FixedOpinions, KPhi, KPsi, MyFloat, MyOpinions, Phi, Psi,
        StateOpinions, Theta, Thetad, A, B, FH, FO, H, KH, KO, O,
    },
};

use itertools::Itertools;
use rand::{seq::SliceRandom, Rng};
use rand_distr::{Beta, Distribution, Exp1, Open01, Standard, StandardNormal, Uniform};
use serde_with::{serde_as, FromInto, TryFromInto};

use subjective_logic::{
    domain::Domain,
    mul::{
        labeled::{OpinionD1, SimplexD1},
        Simplex,
    },
    multi_array::labeled::{MArrD1, MArrD2},
};

#[serde_as]
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(bound(deserialize = "V: serde::Deserialize<'de>"))]
pub struct InitialOpinions<V: MyFloat> {
    #[serde_as(as = "TryFromInto<(Vec<V>, V, Vec<V>)>")]
    psi: OpinionD1<Psi, V>,
    #[serde_as(as = "TryFromInto<(Vec<V>, V, Vec<V>)>")]
    fpsi: OpinionD1<FPsi, V>,
    #[serde_as(as = "TryFromInto<(Vec<V>, V, Vec<V>)>")]
    kpsi: OpinionD1<KPsi, V>,
    #[serde_as(as = "TryFromInto<(Vec<V>, V, Vec<V>)>")]
    phi: OpinionD1<Phi, V>,
    #[serde_as(as = "TryFromInto<(Vec<V>, V, Vec<V>)>")]
    fphi: OpinionD1<FPhi, V>,
    #[serde_as(as = "TryFromInto<(Vec<V>, V, Vec<V>)>")]
    kphi: OpinionD1<KPhi, V>,
    #[serde_as(as = "TryFromInto<(Vec<V>, V, Vec<V>)>")]
    o: OpinionD1<O, V>,
    #[serde_as(as = "TryFromInto<(Vec<V>, V, Vec<V>)>")]
    fo: OpinionD1<FO, V>,
    #[serde_as(as = "TryFromInto<(Vec<V>, V, Vec<V>)>")]
    ko: OpinionD1<KO, V>,
    #[serde_as(as = "TryFromInto<Vec<(Vec<V>, V)>>")]
    h_psi_if_phi1: MArrD1<Psi, SimplexD1<H, V>>,
    #[serde_as(as = "TryFromInto<Vec<(Vec<V>, V)>>")]
    fh_fpsi_if_fphi1: MArrD1<FPsi, SimplexD1<FH, V>>,
    #[serde_as(as = "TryFromInto<Vec<(Vec<V>, V)>>")]
    kh_kpsi_if_kphi1: MArrD1<KPsi, SimplexD1<KH, V>>,
    #[serde_as(as = "TryFromInto<Vec<(Vec<V>, V)>>")]
    h_b_if_phi1: MArrD1<B, SimplexD1<H, V>>,
}

impl<V: MyFloat> InitialOpinions<V> {
    fn reset_to(self, state: &mut StateOpinions<V>)
    where
        Standard: Distribution<V>,
        StandardNormal: Distribution<V>,
        Exp1: Distribution<V>,
        Open01: Distribution<V>,
    {
        let InitialOpinions {
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
        } = self;
        state.reset(
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
    }
}

#[serde_as]
#[derive(Debug, serde::Deserialize, Clone)]
#[serde(bound(deserialize = "V: serde::Deserialize<'de>"))]
pub struct InitialBaseRates<V> {
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

impl<V: MyFloat> InitialBaseRates<V> {
    fn reset_to(self, ded: &mut DeducedOpinions<V>) {
        let InitialBaseRates {
            a,
            b,
            h,
            fh,
            kh,
            theta,
            thetad,
        } = self;
        ded.reset(
            OpinionD1::vacuous_with(h),
            OpinionD1::vacuous_with(fh),
            OpinionD1::vacuous_with(kh),
            OpinionD1::vacuous_with(a),
            OpinionD1::vacuous_with(b),
            OpinionD1::vacuous_with(theta),
            OpinionD1::vacuous_with(thetad),
        );
    }
}

pub struct ConditionSamples<V>
where
    V: MyFloat,
    Open01: Distribution<V>,
{
    pub h_psi_if_phi0: ConditionSampler<Psi, H, V>,
    pub h_b_if_phi0: ConditionSampler<B, H, V>,
    pub o_b: ConditionSampler<B, O, V>,
    pub a_fh: ConditionSampler<FH, A, V>,
    pub b_kh: ConditionSampler<KH, B, V>,
    pub theta_h: ConditionSampler<H, Theta, V>,
    pub thetad_h: ConditionSampler<H, Thetad, V>,
}

pub enum ConditionSampler<D0, D1, V>
where
    D0: Domain,
    D1: Domain,
    V: MyFloat,
    Open01: Distribution<V>,
{
    Array(Vec<MArrD1<D0, SimplexD1<D1, V>>>),
    Random(MArrD1<D0, SimplexContainer<D1::Idx, V>>),
}

impl<D0: Domain, D1: Domain<Idx: Copy>, V: MyFloat> ConditionSampler<D0, D1, V>
where
    Open01: Distribution<V>,
{
    pub fn sample<R: Rng>(&self, rng: &mut R) -> MArrD1<D0, SimplexD1<D1, V>> {
        match self {
            ConditionSampler::Array(vec) => vec.choose(rng).unwrap().to_owned(),
            ConditionSampler::Random(marr_d1) => MArrD1::from_iter(marr_d1.into_iter().map(|c| {
                let mut acc = V::zero();
                let mut b = MArrD1::default();
                let mut u = V::default();
                for x in &c.fixed {
                    match x {
                        SimplexIndexed::B(d1, v) => {
                            acc += *v;
                            b[*d1] = *v;
                        }
                        SimplexIndexed::U(v) => {
                            acc += *v;
                            u = *v;
                        }
                    }
                }
                if let Some(x) = &c.sampler {
                    match x {
                        SimplexIndexed::B(d1, s) => {
                            let v = s.choose(rng);
                            acc += v;
                            b[*d1] = v;
                        }
                        SimplexIndexed::U(s) => {
                            let v = s.choose(rng);
                            acc += v;
                            u = v;
                        }
                    }
                }
                match &c.auto {
                    SimplexIndexed::B(d1, _) => b[*d1] = V::one() - acc,
                    SimplexIndexed::U(_) => u = V::one() - acc,
                }
                Simplex::new_unchecked(b, u)
            })),
        }
    }
}

pub struct SimplexContainer<Idx, V>
where
    V: MyFloat,
    Open01: Distribution<V>,
{
    pub sampler: Option<SimplexIndexed<Idx, Sampler<V>>>,
    pub fixed: Vec<SimplexIndexed<Idx, V>>,
    pub auto: SimplexIndexed<Idx, ()>,
}

pub enum SimplexIndexed<Idx, T> {
    B(Idx, T),
    U(T),
}

#[serde_as]
#[derive(Debug, serde::Deserialize)]
#[serde(bound(deserialize = "V: serde::Deserialize<'de>"))]
pub struct InitialStates<V: MyFloat> {
    pub initial_opinions: InitialOpinions<V>,
    pub initial_base_rates: InitialBaseRates<V>,
}

pub struct OpinionSamples<V: MyFloat>
where
    Open01: Distribution<V>,
{
    pub initial_opinions: InitialOpinions<V>,
    pub initial_base_rates: InitialBaseRates<V>,
    pub condition: ConditionSamples<V>,
    pub uncertainty: UncertaintySamples<V>,
}

pub struct UncertaintySamples<V> {
    pub fh_fpsi_if_fphi0: Vec<MArrD1<FPsi, V>>,
    pub kh_kpsi_if_kphi0: Vec<MArrD1<KPsi, V>>,
    pub fh_fphi_fo: Vec<MArrD2<FPhi, FO, V>>,
    pub kh_kphi_ko: Vec<MArrD2<KPhi, KO, V>>,
}

impl<V: MyFloat> OpinionSamples<V>
where
    Open01: Distribution<V>,
{
    pub fn reset_to<R: Rng>(&self, ops: &mut MyOpinions<V>, rng: &mut R)
    where
        Standard: Distribution<V>,
        StandardNormal: Distribution<V>,
        Exp1: Distribution<V>,
        Open01: Distribution<V>,
    {
        reset_fixed(&self.condition, &self.uncertainty, &mut ops.fixed, rng);
        self.initial_opinions.clone().reset_to(&mut ops.state);
        self.initial_base_rates.clone().reset_to(&mut ops.ded);
    }
}

fn reset_fixed<V: MyFloat, R: Rng>(
    condition: &ConditionSamples<V>,
    uncertainty: &UncertaintySamples<V>,
    fixed: &mut FixedOpinions<V>,
    rng: &mut R,
) where
    Open01: Distribution<V>,
{
    let o_b = condition.o_b.sample(rng);
    let b_kh = condition.b_kh.sample(rng);
    let a_fh = condition.a_fh.sample(rng);
    let theta_h = condition.theta_h.sample(rng);
    let thetad_h = condition.thetad_h.sample(rng);
    let h_psi_if_phi0 = condition.h_psi_if_phi0.sample(rng);
    let h_b_if_phi0 = condition.h_b_if_phi0.sample(rng);
    let uncertainty_fh_fpsi_if_fphi0 = uncertainty.fh_fpsi_if_fphi0.choose(rng).unwrap().to_owned();
    let uncertainty_kh_kpsi_if_kphi0 = uncertainty.kh_kpsi_if_kphi0.choose(rng).unwrap().to_owned();
    let uncertainty_fh_fo_fphi = uncertainty.fh_fphi_fo.choose(rng).unwrap().to_owned();
    let uncertainty_kh_ko_kphi = uncertainty.kh_kphi_ko.choose(rng).unwrap().to_owned();
    fixed.reset(
        o_b,
        b_kh,
        a_fh,
        theta_h,
        thetad_h,
        h_psi_if_phi0,
        h_b_if_phi0,
        uncertainty_fh_fpsi_if_fphi0,
        uncertainty_kh_kpsi_if_kphi0,
        uncertainty_fh_fo_fphi,
        uncertainty_kh_ko_kphi,
    );
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SamplerOption<V> {
    Single(V),
    Array(Vec<V>),
    Uniform(V, V),
    Beta(V, V),
}

pub enum Sampler<V: MyFloat>
where
    Open01: Distribution<V>,
{
    Single(V),
    Arr(Vec<V>),
    Uni(Uniform<V>),
    Beta(Beta<V>),
}

impl<V> From<SamplerOption<V>> for Sampler<V>
where
    V: MyFloat,
    Open01: Distribution<V>,
{
    fn from(value: SamplerOption<V>) -> Self {
        match value {
            SamplerOption::Single(v) => Self::Single(v),
            SamplerOption::Array(v) => Self::Arr(v),
            SamplerOption::Uniform(low, high) => Self::Uni(Uniform::new(low, high)),
            SamplerOption::Beta(alpha, beta) => Self::Beta(Beta::new(alpha, beta).unwrap()),
        }
    }
}

impl<V> Sampler<V>
where
    V: MyFloat,
    Open01: Distribution<V>,
{
    pub fn choose<R: Rng>(&self, rng: &mut R) -> V {
        match self {
            Self::Single(v) => *v,
            Self::Arr(v) => *v.choose(rng).unwrap(),
            Self::Uni(u) => u.sample(rng),
            Self::Beta(b) => b.sample(rng),
        }
    }
}

#[serde_as]
#[derive(serde::Deserialize)]
#[serde(bound(deserialize = "V: serde::Deserialize<'de>"))]
pub struct ProbabilitySamples<V>
where
    V: MyFloat,
    Open01: Distribution<V>,
{
    #[serde_as(as = "FromInto<SamplerOption<V>>")]
    pub viewing: Sampler<V>,
    #[serde_as(as = "FromInto<SamplerOption<V>>")]
    pub viewing_friend: Sampler<V>,
    #[serde_as(as = "FromInto<SamplerOption<V>>")]
    pub viewing_social: Sampler<V>,
    #[serde_as(as = "FromInto<SamplerOption<V>>")]
    pub arrival_friend: Sampler<V>,
    #[serde_as(as = "FromInto<SamplerOption<V>>")]
    pub plural_ignore_friend: Sampler<V>,
    #[serde_as(as = "FromInto<SamplerOption<V>>")]
    pub plural_ignore_social: Sampler<V>,
}

#[serde_as]
#[derive(serde::Deserialize)]
#[serde(bound(deserialize = "V: serde::Deserialize<'de>"))]
pub struct SharerTrustSamples<V>
where
    V: MyFloat,
    Open01: Distribution<V>,
{
    #[serde_as(as = "FromInto<SamplerOption<V>>")]
    pub misinfo: Sampler<V>,
    #[serde_as(as = "FromInto<SamplerOption<V>>")]
    pub correction: Sampler<V>,
    #[serde_as(as = "FromInto<SamplerOption<V>>")]
    pub obserbation: Sampler<V>,
    #[serde_as(as = "FromInto<SamplerOption<V>>")]
    pub inhibition: Sampler<V>,
}

#[derive(Debug, serde::Deserialize)]
pub enum PopSampleType<V> {
    Random(V),
    Top(V),
    Middle(V),
    Bottom(V),
}

#[derive(Debug, serde::Deserialize)]
pub struct Informing<V> {
    pub step: u32,
    pub pop_agents: V,
}

#[derive(Debug, serde::Deserialize)]
pub struct InformingParams<V> {
    /// order by step & non-duplicated
    pub max_pop_misinfo: V,
    pub misinfo: Vec<Informing<V>>,

    /// order by step & non-duplicated
    pub max_pop_correction: V,
    pub correction: Vec<Informing<V>>,

    pub max_pop_observation: V,
    pub prob_post_observation: V,
    pub max_step_pop_observation: V,

    /// order by step & non-duplicated
    pub max_pop_inhibition: PopSampleType<V>,
    pub inhibition: Vec<Informing<V>>,
}

pub struct InformationSamples<V> {
    /// also used for $M$ in correction
    pub misinfo: Vec<OpinionD1<Psi, V>>,
    pub correction: Vec<OpinionD1<Psi, V>>,
    pub observation: Vec<OpinionD1<O, V>>,
    pub inhibition: Vec<(
        OpinionD1<Phi, V>,
        MArrD1<Psi, SimplexD1<H, V>>,
        MArrD1<B, SimplexD1<H, V>>,
    )>,
}

pub struct SupportLevelTable<V> {
    /// vector index === agent_idx
    pub levels: Vec<V>,
    /// sorted in descending order by level
    pub indexes_by_level: Vec<usize>,
}

impl<V: MyFloat> SupportLevelTable<V> {
    pub fn level(&self, idx: usize) -> V {
        self.levels[idx]
    }

    /// `levels` should be ordered by agent index
    pub fn from_vec(levels: Vec<V>) -> Self {
        let indexes_by_level = levels
            .iter()
            .enumerate()
            .sorted_by(|a, b| b.1.partial_cmp(&a.1).unwrap())
            .map(|(i, _)| i)
            .collect_vec();
        Self {
            levels,
            indexes_by_level,
        }
    }

    // pub fn from_iter<I>(into_iter: I) -> Self
    // where
    //     I: IntoIterator<Item = SupportLevelRecord<V>>,
    // {
    //     let levels = into_iter.into_iter().map(|s| s.level).collect_vec();
    //     let indexes_by_level = levels
    //         .iter()
    //         .enumerate()
    //         .sorted_by(|a, b| b.1.partial_cmp(&a.1).unwrap())
    //         .map(|(i, _)| i)
    //         .collect_vec();
    //     Self {
    //         levels,
    //         indexes_by_level,
    //     }
    // }

    pub fn random<R: Rng>(&self, n: usize, rng: &mut R) -> Vec<usize> {
        self.indexes_by_level
            .choose_multiple(rng, n)
            .cloned()
            .collect()
    }

    pub fn top<R: Rng>(&self, n: usize, rng: &mut R) -> Vec<usize> {
        let mut v = self.indexes_by_level.iter().take(n).cloned().collect_vec();
        v.shuffle(rng);
        v
    }

    pub fn middle<R: Rng>(&self, n: usize, rng: &mut R) -> Vec<usize>
    where
        V: MyFloat,
    {
        macro_rules! level_of {
            ($e:expr) => {
                self.levels[self.indexes_by_level[$e]]
            };
        }
        let l = self.indexes_by_level.len();
        let c = self.indexes_by_level.len() / 2;
        let median = if l % 2 == 1 {
            level_of!(c)
        } else {
            (level_of!(c) + level_of!(c - 1)) / V::from_u32(2).unwrap()
        };

        let from = c.checked_sub(n).unwrap_or(0);
        let to = (c + n).min(l);
        let mut v = (from..to)
            .sorted_by(|&i, &j| {
                let a = (level_of!(i) - median).abs();
                let b = (level_of!(j) - median).abs();
                a.partial_cmp(&b).unwrap()
            })
            .take(n)
            .map(|i| self.indexes_by_level[i])
            .collect_vec();
        v.shuffle(rng);
        v
    }

    pub fn bottom<R: Rng>(&self, n: usize, rng: &mut R) -> Vec<usize> {
        let mut v = self
            .indexes_by_level
            .iter()
            .rev()
            .take(n)
            .cloned()
            .collect_vec();
        v.shuffle(rng);
        v
    }
}

#[derive(Debug, serde::Deserialize)]
pub struct CptRecord<V> {
    alpha: V,
    beta: V,
    gamma: V,
    delta: V,
    lambda: V,
}

pub struct CptSamples<V>(pub Vec<CptRecord<V>>);

impl<V: MyFloat> CptSamples<V> {
    pub fn reset_to<R: Rng>(&self, cpt: &mut CPT<V>, rng: &mut R) {
        let &CptRecord {
            alpha,
            beta,
            gamma,
            delta,
            lambda,
        } = self.0.choose(rng).unwrap();
        cpt.reset(alpha, beta, lambda, gamma, delta);
    }
}

#[derive(Debug, serde::Deserialize)]
pub struct ProspectRecord<V> {
    x0: V,
    x1: V,
    y: V,
}

pub struct ProspectSamples<V>(pub Vec<ProspectRecord<V>>);

impl<V: MyFloat> ProspectSamples<V> {
    pub fn reset_to<R: Rng>(&self, prospect: &mut Prospect<V>, rng: &mut R) {
        let &ProspectRecord { x0, x1, y } = self.0.choose(rng).unwrap();
        prospect.reset(x0, x1, y);
    }
}

#[cfg(test)]
mod tests {
    use super::SupportLevelTable;

    #[test]
    fn test_support_levels() {
        let levels = vec![0.5, 0.2, 0.3, 0.1, 0.6];
        let sls = SupportLevelTable::<f32>::from_vec(levels);
        assert_eq!(sls.indexes_by_level, vec![4, 0, 2, 1, 3]);
    }
}
