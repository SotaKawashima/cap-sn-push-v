use std::{
    fmt::Debug,
    fs::File,
    io,
    path::{Path, PathBuf},
};

use anyhow::{bail, ensure, Context};
use base::{
    decision::{Prospect, CPT},
    opinion::{
        DeducedOpinions, FPhi, FPsi, FixedOpinions, KPhi, KPsi, MyFloat, MyOpinions, Phi, Psi,
        StateOpinions, Theta, Thetad, A, B, FH, FO, H, KH, KO, O,
    },
};
use graph_lib::prelude::{Graph, GraphB};
use input::format::DataFormat;
use itertools::{Itertools, ProcessResults};
use rand::{seq::SliceRandom, Rng};
use rand_distr::{Beta, Distribution, Exp1, Open01, Standard, StandardNormal, Uniform};
use serde::{de::DeserializeOwned, Deserialize};
use serde_with::{serde_as, FromInto, TryFromInto};
use subjective_logic::{
    domain::{Domain, Keys},
    errors::{check_unit_interval, InvalidValueError},
    iter::FromFn,
    marr_d1, marr_d2,
    mul::{
        labeled::{OpinionD1, SimplexD1},
        Simplex,
    },
    multi_array::labeled::{MArrD1, MArrD2},
};

use crate::{exec::Exec, io::MyPath};

pub struct Config {
    agent: AgentConfig,
    strategy: StrategyConfig,
    network: NetworkConfig,
    agent_root: PathBuf,
    strategy_root: PathBuf,
    network_root: PathBuf,
}

impl Config {
    pub fn try_new<P: AsRef<Path>>(
        network_config: P,
        agent_config: P,
        strategy_config: P,
    ) -> anyhow::Result<Self> {
        let agent: AgentConfig = DataFormat::read(&agent_config)?.parse()?;
        let strategy: StrategyConfig = DataFormat::read(&strategy_config)?.parse()?;
        let network: NetworkConfig = DataFormat::read(&network_config)?.parse()?;
        Ok(Self {
            agent,
            agent_root: agent_config
                .as_ref()
                .parent()
                .unwrap_or_else(|| &(Path::new("/")))
                .to_path_buf(),
            strategy,
            strategy_root: strategy_config
                .as_ref()
                .parent()
                .unwrap_or_else(|| &(Path::new("/")))
                .to_path_buf(),
            network,
            network_root: network_config
                .as_ref()
                .parent()
                .unwrap_or_else(|| &(Path::new("/")))
                .to_path_buf(),
        })
    }

    pub fn into_exec<V>(
        self,
        enable_inhibition: bool,
        delay_selfish: u32,
    ) -> anyhow::Result<Exec<V>>
    where
        V: MyFloat + for<'a> Deserialize<'a>,
        Open01: Distribution<V>,
    {
        let Self {
            agent,
            strategy,
            network,
            agent_root,
            network_root,
            strategy_root,
        } = self;

        let graph = network.parse_graph(&network_root)?;
        let fnum_agents = V::from_usize(graph.node_count()).unwrap();
        let mean_degree = V::from_usize(graph.edge_count()).unwrap() / fnum_agents;
        let community_psi1 = network.parse_comm(&network_root)?;

        let InitialStates {
            initial_opinions,
            initial_base_rates,
        } = DataFormat::read(agent.initial_states.verified(&agent_root)?)?.parse()?;

        Ok(Exec {
            enable_inhibition,
            delay_selfish,
            graph,
            fnum_agents,
            mean_degree,
            sharer_trust: DataFormat::read(&agent.sharer_trust.verified(&agent_root)?)?.parse()?,
            opinion: OpinionSamples {
                condition: ConditionConfig::read(&agent.condition.verified(&agent_root)?)?,
                uncertainty: UncertaintyConfig::read(&agent.uncertainty.verified(&agent_root)?)?,
                initial_opinions,
                initial_base_rates,
            },
            information: strategy.information.into_samples(&strategy_root)?,
            informing: DataFormat::read(&strategy.informing.verified(strategy_root)?)?.parse()?,
            community_psi1,
            probabilies: DataFormat::read(&agent.probabilities.verified(&agent_root)?)?.parse()?,
            prospect: ProspectSamples(read_csv(&agent.prospect.verified(&agent_root)?)?),
            cpt: CptSamples(read_csv(&agent.cpt.verified(&agent_root)?)?),
        })
    }
}

#[derive(Debug, Deserialize)]
struct AgentConfig {
    probabilities: MyPath,
    sharer_trust: MyPath,
    prospect: MyPath,
    cpt: MyPath,
    initial_states: MyPath,
    condition: MyPath,   // ConditionConfig<V>,
    uncertainty: MyPath, // UncertaintyConfig,
}

#[derive(Debug, Deserialize)]
struct StrategyConfig {
    informing: MyPath,
    information: InformationConfig,
}

#[derive(Debug, Deserialize)]
pub struct InformationConfig {
    /// also used for $M$ in correction
    misinfo: MyPath,
    correction: MyPath,
    observation: MyPath,
    inhibition: MyPath,
}

impl InformationConfig {
    fn into_samples<V, P>(self, root: P) -> anyhow::Result<InformationSamples<V>>
    where
        V: MyFloat + for<'a> Deserialize<'a>,
        P: AsRef<Path>,
    {
        Ok(InformationSamples {
            misinfo: read_csv_and_then(&self.misinfo.verified(&root)?, OpinionRecord::try_into)?,
            correction: read_csv_and_then(
                &self.correction.verified(&root)?,
                OpinionRecord::try_into,
            )?,
            observation: read_csv_and_then(
                &self.observation.verified(&root)?,
                OpinionRecord::try_into,
            )?,
            inhibition: read_csv_and_then(
                &self.inhibition.verified(&root)?,
                InhibitionRecord::try_into,
            )?,
        })
    }
}

#[derive(Debug, Deserialize)]
struct NetworkConfig {
    path: PathBuf,
    graph: MyPath,
    directed: bool,
    transposed: bool,
    community: MyPath,
}

impl NetworkConfig {
    fn get_root<P: AsRef<Path>>(&self, root: P) -> PathBuf {
        root.as_ref().join(&self.path)
    }

    fn parse_graph<P: AsRef<Path>>(&self, root: P) -> Result<GraphB, io::Error> {
        let builder = graph_lib::io::ParseBuilder::new(
            File::open(self.graph.verified(self.get_root(root))?)?,
            graph_lib::io::DataFormat::EdgeList,
        );
        if self.directed {
            if self.transposed {
                Ok(GraphB::Di(builder.transpose().parse()?))
            } else {
                Ok(GraphB::Di(builder.parse()?))
            }
        } else {
            Ok(GraphB::Ud(builder.parse()?))
        }
    }

    fn parse_comm<V: MyFloat + for<'a> Deserialize<'a>, P: AsRef<Path>>(
        &self,
        root: P,
    ) -> anyhow::Result<SupportLevels<V>> {
        read_csv_with(self.community.verified(&self.get_root(root))?, |iter| {
            SupportLevels::from_iter(iter)
        })
    }
}

#[serde_as]
#[derive(Debug, Deserialize)]
#[serde(bound(deserialize = "V: serde::Deserialize<'de>"))]
struct InitialStates<V: MyFloat> {
    initial_opinions: InitialOpinions<V>,
    initial_base_rates: InitialBaseRates<V>,
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
enum ConditionParam<V> {
    Import(MyPath),
    Generate(Vec<Vec<SimplexParam<V>>>),
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
enum SimplexParam<V> {
    B(usize, SamplerOption<V>),
    U(SamplerOption<V>),
}

enum ConditionSampler<D0, D1, V>
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
    fn sample<R: Rng>(&self, rng: &mut R) -> MArrD1<D0, SimplexD1<D1, V>> {
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

struct SimplexContainer<Idx, V>
where
    V: MyFloat,
    Open01: Distribution<V>,
{
    sampler: Option<SimplexIndexed<Idx, Sampler<V>>>,
    fixed: Vec<SimplexIndexed<Idx, V>>,
    auto: SimplexIndexed<Idx, ()>,
}

enum SimplexIndexed<Idx, T> {
    B(Idx, T),
    U(T),
}

impl<V> ConditionParam<V>
where
    V: MyFloat + for<'a> Deserialize<'a>,
    Open01: Distribution<V>,
{
    fn into_sample<D0, D1, P>(self, root: P) -> anyhow::Result<ConditionSampler<D0, D1, V>>
    where
        D0: Domain<Idx: Debug> + Keys<D0::Idx>,
        D1: Domain<Idx: Debug + From<usize> + Copy> + Keys<D1::Idx>,
        P: AsRef<Path>,
    {
        match self {
            ConditionParam::Import(path) => Ok(ConditionSampler::Array(read_csv_and_then(
                path.verified(root)?,
                ConditionRecord::try_into,
            )?)),
            ConditionParam::Generate(pss) => {
                ensure!(D0::LEN == pss.len(), "few conditional opinion(s)");
                let containers = pss.into_iter().map(|ps| {
                    let mut b_check = MArrD1::<D1, bool>::from_fn(|_| false);
                    let mut u_check = false;
                    let mut sampler = None;
                    let mut fixed = Vec::new();
                    for p in ps {
                        match p {
                            SimplexParam::B(i, so) => {
                                ensure!(!b_check[i.into()], "b({i}) is duplicated");
                                b_check[i.into()] = true;
                                match so {
                                    SamplerOption::Single(v) => fixed.push(SimplexIndexed::B(i.into(), v)),
                                    _ => {
                                        ensure!(sampler.is_none(), "at most one sampler is avilable");
                                        sampler = Some(SimplexIndexed::B(i.into(), so.into()));
                                    },
                                }
                            }
                            SimplexParam::U(so) => {
                                ensure!(!u_check, "u is duplicated");
                                u_check = true;
                                match so {
                                    SamplerOption::Single(v) => fixed.push(SimplexIndexed::U(v)),
                                    _ => {
                                        ensure!(sampler.is_none(), "at most one sampler is avilable");
                                        sampler = Some(SimplexIndexed::U(so.into()));
                                    },
                                }
                            }
                        }
                    }
                    let b_remain = D1::keys()
                        .filter_map(|i| if b_check[i] { None } else { Some(i) })
                        .collect_vec();
                    let auto = match b_remain.len() {
                        0 => {
                            ensure!(!u_check, "one belief or uncertainty should be unset");
                            SimplexIndexed::U(())
                        }
                        1 => {
                            ensure!(u_check, "uncertainty should be set");
                            SimplexIndexed::B(b_remain[0], ())
                        }
                        _ => {
                            if u_check {
                                bail!(
                                    "{} belief(s) of indexes {:?} should be set",
                                    b_remain.len() - 1,
                                    b_remain
                                );
                            } else {
                                bail!("{} belief(s) or {} belief(s) and uncertainty should be set from indexes {:?}",
                                b_remain.len(), b_remain.len() - 1, b_remain);
                            }
                        }
                    };
                    Ok(SimplexContainer { sampler, fixed, auto })
                }).zip(D0::keys())
                .map(|(c, d0)| c.with_context(|| format!("at domain {d0:?}")))
                .try_collect::<_, Vec<_>, _>()?;
                // for ps in pss { }
                Ok(ConditionSampler::Random(MArrD1::from_iter(containers)))
            }
        }
    }
}

#[derive(Debug, Deserialize)]
struct ConditionConfig<V> {
    h_psi_if_phi0: ConditionParam<V>,
    h_b_if_phi0: ConditionParam<V>,
    o_b: ConditionParam<V>,
    a_fh: ConditionParam<V>,
    b_kh: ConditionParam<V>,
    theta_h: ConditionParam<V>,
    thetad_h: ConditionParam<V>,
}

impl<V: MyFloat + for<'a> Deserialize<'a>> ConditionConfig<V>
where
    Open01: Distribution<V>,
{
    fn read<P: AsRef<Path>>(root: P) -> anyhow::Result<ConditionSamples<V>> {
        let root = root.as_ref();
        let parent = root.parent().unwrap();
        let this = DataFormat::read(root)?.parse::<Self>()?;
        ConditionSamples::try_new(this, parent)
    }
}

#[derive(Debug, Deserialize)]
struct UncertaintyConfig {
    fh_fpsi_if_fphi0: MyPath,
    kh_kpsi_if_kphi0: MyPath,
    fh_fphi_fo: MyPath,
    kh_kphi_ko: MyPath,
}

impl UncertaintyConfig {
    fn read<V: MyFloat + for<'a> Deserialize<'a>, P: AsRef<Path>>(
        root: P,
    ) -> anyhow::Result<UncertaintySamples<V>> {
        let root = root.as_ref();
        let parent = root.parent().unwrap();
        let this = DataFormat::read(root)?.parse::<Self>()?;
        UncertaintySamples::try_new(this, parent)
    }
}

pub struct OpinionSamples<V: MyFloat>
where
    Open01: Distribution<V>,
{
    initial_opinions: InitialOpinions<V>,
    initial_base_rates: InitialBaseRates<V>,
    condition: ConditionSamples<V>,
    uncertainty: UncertaintySamples<V>,
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

struct ConditionSamples<V>
where
    V: MyFloat,
    Open01: Distribution<V>,
{
    h_psi_if_phi0: ConditionSampler<Psi, H, V>,
    h_b_if_phi0: ConditionSampler<B, H, V>,
    o_b: ConditionSampler<B, O, V>,
    a_fh: ConditionSampler<FH, A, V>,
    b_kh: ConditionSampler<KH, B, V>,
    theta_h: ConditionSampler<H, Theta, V>,
    thetad_h: ConditionSampler<H, Thetad, V>,
}

impl<V: MyFloat + for<'a> Deserialize<'a>> ConditionSamples<V>
where
    Open01: Distribution<V>,
{
    fn try_new<P: AsRef<Path>>(config: ConditionConfig<V>, root: P) -> anyhow::Result<Self> {
        Ok(Self {
            h_psi_if_phi0: config
                .h_psi_if_phi0
                .into_sample(&root)
                .context("h_psi_if_phi0")?,
            h_b_if_phi0: config
                .h_b_if_phi0
                .into_sample(&root)
                .context("h_b_if_phi0")?,
            o_b: config.o_b.into_sample(&root).context("o_b")?,
            a_fh: config.a_fh.into_sample(&root).context("a_fh")?,
            b_kh: config.b_kh.into_sample(&root).context("b_kh")?,
            theta_h: config.theta_h.into_sample(&root).context("theta_h")?,
            thetad_h: config.thetad_h.into_sample(&root).context("thetad_h")?,
        })
    }
}

#[derive(Debug, serde::Deserialize)]
struct ConditionRecord<V> {
    b00: V,
    b01: V,
    u0: V,
    b10: V,
    b11: V,
    u1: V,
}

impl<V, D1, D2> TryFrom<ConditionRecord<V>> for MArrD1<D1, SimplexD1<D2, V>>
where
    V: MyFloat,
    D1: Domain<Idx: Debug>,
    D2: Domain<Idx: Debug>,
{
    type Error = InvalidValueError;
    fn try_from(value: ConditionRecord<V>) -> Result<Self, Self::Error> {
        Ok(marr_d1![
            SimplexD1::try_new(marr_d1![value.b00, value.b01], value.u0)?,
            SimplexD1::try_new(marr_d1![value.b10, value.b11], value.u1)?,
        ])
    }
}

struct UncertaintySamples<V> {
    fh_fpsi_if_fphi0: Vec<MArrD1<FPsi, V>>,
    kh_kpsi_if_kphi0: Vec<MArrD1<KPsi, V>>,
    fh_fphi_fo: Vec<MArrD2<FPhi, FO, V>>,
    kh_kphi_ko: Vec<MArrD2<KPhi, KO, V>>,
}

impl<V: MyFloat + for<'a> Deserialize<'a>> UncertaintySamples<V> {
    fn try_new<P: AsRef<Path>>(config: UncertaintyConfig, root: P) -> anyhow::Result<Self> {
        Ok(Self {
            fh_fpsi_if_fphi0: read_csv_and_then(
                config.fh_fpsi_if_fphi0.verified(&root)?,
                UncertaintyD1Record::try_into,
            )?,
            kh_kpsi_if_kphi0: read_csv_and_then(
                config.kh_kpsi_if_kphi0.verified(&root)?,
                UncertaintyD1Record::try_into,
            )?,
            fh_fphi_fo: read_csv_and_then(
                config.fh_fphi_fo.verified(&root)?,
                UncertaintyD2Record::try_into,
            )?,
            kh_kphi_ko: read_csv_and_then(
                config.kh_kphi_ko.verified(&root)?,
                UncertaintyD2Record::try_into,
            )?,
        })
    }
}

#[derive(Debug, Deserialize)]
struct UncertaintyD1Record<V> {
    u0: V,
    u1: V,
}

impl<V: MyFloat, D1: Domain> TryFrom<UncertaintyD1Record<V>> for MArrD1<D1, V> {
    type Error = InvalidValueError;
    fn try_from(value: UncertaintyD1Record<V>) -> Result<Self, Self::Error> {
        check_unit_interval(value.u0, "u0")?;
        check_unit_interval(value.u1, "u1")?;
        Ok(marr_d1![value.u0, value.u1])
    }
}

#[derive(Debug, Deserialize)]
struct UncertaintyD2Record<V> {
    u00: V,
    u01: V,
    u10: V,
    u11: V,
}

impl<V: MyFloat, D1: Domain, D2: Domain> TryFrom<UncertaintyD2Record<V>> for MArrD2<D1, D2, V> {
    type Error = InvalidValueError;
    fn try_from(value: UncertaintyD2Record<V>) -> Result<Self, Self::Error> {
        check_unit_interval(value.u00, "u00")?;
        check_unit_interval(value.u01, "u01")?;
        check_unit_interval(value.u10, "u10")?;
        check_unit_interval(value.u11, "u11")?;
        Ok(marr_d2![[value.u00, value.u01], [value.u10, value.u11],])
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

#[derive(Debug, Deserialize)]
pub struct SupportLevel<V> {
    level: V,
}

pub struct SupportLevels<V> {
    /// vector index === agent_idx
    levels: Vec<V>,
    /// sorted in descending order by level
    indexes_by_level: Vec<usize>,
}

impl<V: MyFloat> SupportLevels<V> {
    pub fn level(&self, idx: usize) -> V {
        self.levels[idx]
    }

    fn from_iter<I>(into_iter: I) -> Self
    where
        I: IntoIterator<Item = SupportLevel<V>>,
    {
        let levels = into_iter.into_iter().map(|s| s.level).collect_vec();
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
#[serde(rename_all = "lowercase")]
enum SamplerOption<V> {
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

#[derive(Debug, Deserialize)]
struct OpinionRecord<V> {
    b0: V,
    b1: V,
    u: V,
    a0: V,
    a1: V,
}

impl<V: MyFloat, D1> TryFrom<OpinionRecord<V>> for OpinionD1<D1, V>
where
    D1: Domain<Idx: Debug>,
{
    type Error = InvalidValueError;
    fn try_from(value: OpinionRecord<V>) -> Result<Self, Self::Error> {
        Ok(OpinionD1::try_new(
            marr_d1![value.b0, value.b1],
            value.u,
            marr_d1![value.a0, value.a1],
        )?)
    }
}

#[derive(Debug, Deserialize)]
struct InhibitionRecord<V> {
    phi_b0: V,
    phi_b1: V,
    phi_u: V,
    phi_a0: V,
    phi_a1: V,
    psi0_b0: V,
    psi0_b1: V,
    psi0_u: V,
    psi1_b0: V,
    psi1_b1: V,
    psi1_u: V,
    b0_b0: V,
    b0_b1: V,
    b0_u: V,
    b1_b0: V,
    b1_b1: V,
    b1_u: V,
}

impl<V: MyFloat> TryFrom<InhibitionRecord<V>>
    for (
        OpinionD1<Phi, V>,
        MArrD1<Psi, SimplexD1<H, V>>,
        MArrD1<B, SimplexD1<H, V>>,
    )
{
    type Error = InvalidValueError;
    fn try_from(value: InhibitionRecord<V>) -> Result<Self, Self::Error> {
        Ok((
            OpinionD1::try_new(
                marr_d1![value.phi_b0, value.phi_b1],
                value.phi_u,
                marr_d1![value.phi_a0, value.phi_a1],
            )?,
            marr_d1![
                SimplexD1::try_new(marr_d1![value.psi0_b0, value.psi0_b1], value.psi0_u)?,
                SimplexD1::try_new(marr_d1![value.psi1_b0, value.psi1_b1], value.psi1_u)?,
            ],
            marr_d1![
                SimplexD1::try_new(marr_d1![value.b0_b0, value.b0_b1], value.b0_u)?,
                SimplexD1::try_new(marr_d1![value.b1_b0, value.b1_b1], value.b1_u)?,
            ],
        ))
    }
}

#[serde_as]
#[derive(Debug, Clone, Deserialize)]
#[serde(bound(deserialize = "V: serde::Deserialize<'de>"))]
struct InitialOpinions<V: MyFloat> {
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
#[derive(Debug, serde::Deserialize)]
pub enum Sampling<V> {
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
    pub max_pop_inhibition: Sampling<V>,
    pub inhibition: Vec<Informing<V>>,
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

#[derive(Debug, serde::Deserialize)]
struct CptRecord<V> {
    alpha: V,
    beta: V,
    gamma: V,
    delta: V,
    lambda: V,
}

pub struct CptSamples<V>(Vec<CptRecord<V>>);

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

fn read_csv_and_then<T, P, F, U, E>(path: P, f: F) -> anyhow::Result<Vec<U>>
where
    T: DeserializeOwned,
    P: AsRef<Path>,
    F: Fn(T) -> Result<U, E>,
    E: std::error::Error + Send + Sync + 'static,
{
    let file = File::open(path)?;
    let mut rdr = csv::Reader::from_reader(file);
    Ok(rdr
        .deserialize::<T>()
        .into_iter()
        .process_results(|iter| iter.map(f).try_collect())??)
}

fn read_csv_with<T, P, F, U>(path: P, processor: F) -> anyhow::Result<U>
where
    T: DeserializeOwned,
    P: AsRef<Path>,
    F: FnOnce(ProcessResults<csv::DeserializeRecordsIter<File, T>, csv::Error>) -> U,
{
    let file = File::open(path)?;
    let mut rdr = csv::Reader::from_reader(file);
    Ok(rdr.deserialize::<T>().process_results(processor)?)
}

fn read_csv<T, P>(path: P) -> anyhow::Result<Vec<T>>
where
    T: DeserializeOwned,
    P: AsRef<Path>,
{
    read_csv_with(path, |iter| iter.collect_vec())
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use base::opinion::{Phi, H};
    use rand::{rngs::SmallRng, thread_rng, SeedableRng};
    use serde::Deserialize;
    use subjective_logic::{
        marr_d1, marr_d2,
        mul::labeled::{OpinionD1, SimplexD1},
    };
    use toml::toml;

    use super::{ConditionParam, ConditionSampler, Config, SupportLevel, SupportLevels};
    use crate::{config::Sampler, exec::Exec};

    #[test]
    fn test_support_levels() {
        let data = vec![0.5, 0.2, 0.3, 0.1, 0.6];
        let sls = SupportLevels::<f32>::from_iter(data.iter().map(|&level| SupportLevel { level }));
        assert_eq!(sls.levels, data);
        assert_eq!(sls.indexes_by_level, vec![4, 0, 2, 1, 3]);
    }

    #[test]
    fn test_path() {
        let d0 = Path::new("/hoge");
        let d1 = Path::new("./fuga");
        let d2 = d0.join(d1);
        assert!(!d1.is_absolute());
        assert_ne!(d1, Path::new("fuga"));
        println!("{d2:?}");
        assert_eq!(d2, Path::new("/hoge/fuga"));
    }

    #[test]
    fn test_config() -> anyhow::Result<()> {
        let config = Config::try_new(
            "./test/network_config.toml",
            "./test/agent_config.toml",
            "./test/strategy_config.toml",
        )?;
        let exec: Exec<f32> = config.into_exec(true, 0)?;
        assert_eq!(exec.community_psi1.levels.len(), 100);
        assert!(
            exec.community_psi1.levels[exec.community_psi1.indexes_by_level[10]]
                > exec.community_psi1.levels[exec.community_psi1.indexes_by_level[90]]
        );
        assert!(
            matches!(&exec.opinion.condition.o_b, ConditionSampler::Array(arr) if arr.len() == 7)
        );
        assert!(matches!(
            &exec.opinion.condition.o_b, ConditionSampler::Array(arr) if arr[3] == marr_d1![
            SimplexD1::new(marr_d1![1.0, 0.00], 0.00),
            SimplexD1::new(marr_d1![0.0, 0.75], 0.25)
            ]
        ));
        assert_eq!(
            exec.opinion.uncertainty.fh_fpsi_if_fphi0[0],
            marr_d1![0.1, 0.1]
        );
        assert_eq!(
            exec.opinion.uncertainty.kh_kphi_ko[0],
            marr_d2![[0.1, 0.1], [0.1, 0.1]]
        );
        assert_eq!(
            exec.information.inhibition[0].0,
            OpinionD1::new(marr_d1![0.0, 1.0], 0.0, marr_d1![0.05, 0.95])
        );
        assert_eq!(
            exec.information.inhibition[0].1,
            marr_d1![
                SimplexD1::new(marr_d1![0.5, 0.0], 0.5),
                SimplexD1::new(marr_d1![0.1, 0.7], 0.2),
            ]
        );
        assert_eq!(
            exec.information.inhibition[0].2,
            marr_d1![
                SimplexD1::new(marr_d1![0.5, 0.0], 0.5),
                SimplexD1::new(marr_d1![0.1, 0.6], 0.3),
            ]
        );
        let mut rng = SmallRng::seed_from_u64(0);
        for _ in 0..10 {
            assert!(exec.sharer_trust.misinfo.choose(&mut rng) < 0.5);
        }

        assert!(matches!(
            exec.probabilies.plural_ignore_social,
            Sampler::Beta(_)
        ));
        let mut p = 0.0;
        for _ in 0..10 {
            p += exec.probabilies.plural_ignore_social.choose(&mut rng);
        }
        p /= 10.0;
        assert!((p - 0.1).abs() < 0.05);

        Ok(())
    }

    #[derive(Debug, Deserialize)]
    struct TestCondParam<V> {
        hoge: ConditionParam<V>,
    }

    #[test]
    fn test_cond_param() -> anyhow::Result<()> {
        let param: TestCondParam<f32> = toml! {
            hoge = { generate = [
                [{b = [0,{single = 0.95}]},{b = [0,{single = 0.95}]},{u = {single = 0.05}}],
                [{b = [0,{single = 0.95}]}, {u = {single = 0.05}}],
            ] }
        }
        .try_into()?;
        let r = param.hoge.into_sample::<Phi, H, _>("");
        assert!(r.is_err());
        println!("{:#?}", r.err());

        let param: TestCondParam<f32> = toml! {
            hoge = { generate = [
                [{b = [0,{single = 0.95}]},{u = {single = 0.95}},{u = {single = 0.05}}],
                [{b = [0,{single = 0.95}]}, {u = {single = 0.05}}],
            ] }
        }
        .try_into()?;
        let r = param.hoge.into_sample::<Phi, H, _>("");
        assert!(r.is_err());
        println!("{:#?}", r.err());

        let param: TestCondParam<f32> = toml! {
            hoge = { generate = [
                [{b = [0,{single = 0.95}]},{b = [1,{single = 0.95}]},{u = {single = 0.05}}],
                [{b = [0,{single = 0.95}]}, {u = {single = 0.05}}],
            ] }
        }
        .try_into()?;
        let r = param.hoge.into_sample::<Phi, H, _>("");
        assert!(r.is_err());
        println!("{:#?}", r.err());

        let param: TestCondParam<f32> = toml! {
            hoge = { generate = [
                [{b = [0,{single = 0.95}]}, {u = {single = 0.05}}],
                [{b = [0,{array = [0.0,0.1]}]}, {b = [1,{uniform = [0.80,0.90]}]}],
            ] }
        }
        .try_into()?;
        let r = param.hoge.into_sample::<Phi, H, _>("");
        assert!(r.is_err());
        println!("{:#?}", r.err());

        let param: TestCondParam<f32> = toml! {
            hoge = { generate = [
                [{u = {single = 0.05}},{b = [0,{single = 0.95}]}],
                [{b = [1,{single = 0.85}]},{b = [0,{single = 0.0}]}],
            ] }
        }
        .try_into()?;
        let s: ConditionSampler<Phi, H, f32> = param.hoge.into_sample("")?;
        let c = s.sample(&mut thread_rng());
        assert_eq!(c[Phi(0)].b()[H(0)], 0.95);
        assert_eq!(c[Phi(0)].u(), &0.05);
        assert_eq!(c[Phi(1)].b()[H(0)], 0.0);
        assert_eq!(c[Phi(1)].b()[H(1)], 0.85);

        let param: TestCondParam<f32> = toml! {
            hoge = { generate = [
                [{b = [0,{single = 0.95}]}, {u = {single = 0.05}}],
                [{b = [0,{single = 0.0}]}, {b = [1,{uniform = [0.80,0.90]}]}],
            ] }
        }
        .try_into()?;
        let s: ConditionSampler<Phi, H, f32> = param.hoge.into_sample("")?;
        for _ in 0..20 {
            let c = s.sample(&mut thread_rng());
            assert_eq!(c[Phi(0)].b()[H(0)], 0.95);
            assert_eq!(c[Phi(0)].u(), &0.05);
            assert_eq!(c[Phi(1)].b()[H(0)], 0.0);
            assert_eq!(c[Phi(1)].u() + c[Phi(1)].b()[H(1)], 1.0);
            assert!(c[Phi(1)].b()[H(1)] >= 0.80);
            assert!(c[Phi(1)].b()[H(1)] < 0.90);
        }
        Ok(())
    }
}
