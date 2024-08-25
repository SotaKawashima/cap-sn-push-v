use std::{borrow::Cow, fmt::Debug, fs::File, path::PathBuf, vec::Drain};

use base::{
    executor::{
        AgentExtTrait, AgentIdx, AgentWrapper, Executor, InfoIdx, InstanceExt, InstanceWrapper,
    },
    info::{InfoContent, InfoLabel},
    opinion::{
        AccessProb, FPhi, FPsi, KPhi, KPsi, MyFloat, Phi, Psi, Theta, Thetad, Trusts, A, B, FH, FO,
        H, KH, KO, O,
    },
    runner::{run, RuntimeParams},
    stat::FileWriters,
    util::GraphInfo,
};
use graph_lib::prelude::{Graph, GraphB};
use input::format::DataFormat;
use itertools::Itertools;
use polars_arrow::datatypes::Metadata;
use rand::{seq::IteratorRandom, seq::SliceRandom, Rng};
use rand_distr::{Distribution, Exp1, Open01, Standard, StandardNormal};
use serde::Deserialize;
use serde_with::{serde_as, TryFromInto};
use subjective_logic::{
    domain::Domain,
    marr_d1, marr_d2,
    mul::labeled::{OpinionD1, SimplexD1},
    multi_array::labeled::{MArrD1, MArrD2},
};

#[derive(clap::Parser)]
pub struct Cli {
    /// string to identify given configuration data
    identifier: String,
    /// the path of output files
    output_dir: PathBuf,
    /// the path of a runtime config file
    #[arg(short, long)]
    runtime: String,
    /// the path of a agent parameters config file
    #[arg(short, long)]
    config: String,
    /// Enable overwriting of a output file
    #[arg(short, long, default_value_t = false)]
    overwriting: bool,
    /// Compress a output file
    #[arg(short, long, default_value_t = true)]
    compressing: bool,
}

pub async fn start<V>(args: Cli) -> anyhow::Result<()>
where
    V: MyFloat + 'static,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    V::Sampler: Sync,
    for<'de> V: serde::Deserialize<'de>,
{
    let Cli {
        identifier,
        output_dir,
        runtime,
        config,
        overwriting,
        compressing,
    } = args;
    let runtime_data = DataFormat::read(&runtime)?.parse::<RuntimeParams>()?;
    let metadata = Metadata::from_iter([
        ("version".to_string(), env!("CARGO_PKG_VERSION").to_string()),
        ("runtime".to_string(), runtime),
        // ("agent_params".to_string(), agent_params),
        // ("scenario".to_string(), scenario),
        (
            "iteration_count".to_string(),
            runtime_data.iteration_count.to_string(),
        ),
    ]);
    let writers =
        FileWriters::try_new(&identifier, &output_dir, overwriting, compressing, metadata)?;
    let config = DataFormat::read(&config)?.parse()?;
    let exec = Exec::<V>::try_new(config)?;
    run::<V, _, AgentExt<V>, Instance>(writers, &runtime_data, exec, None).await
}

#[derive(Debug, Deserialize)]
#[serde(bound(deserialize = "V: serde::Deserialize<'de>"))]
struct Config<V: MyFloat> {
    graph: GraphInfo,
    initial_opinions: InitialOpinions<V>,
    initial_base_rate: InitialBaseRates<V>,
    sharer_trust_path: String,
    condition: ConditionConfig,
    uncertainty: UncertaintyConfig,
    information: InformationConfig,
    informing_path: String,
    community_psi1_path: String,
    prob_post_observation: V,
    probabilities_path: String,
    prospect_path: String,
    cpt_path: String,
}

#[derive(Debug, Deserialize)]
struct ConditionConfig {
    h_psi_if_phi0: String,
    h_b_if_phi0: String,
    o_b: String,
    a_fh: String,
    b_kh: String,
    theta_h: String,
    thetad_h: String,
}

#[derive(Debug, Deserialize)]
struct UncertaintyConfig {
    fh_fpsi_if_fphi0: String,
    fh_kpsi_if_kphi0: String,
    fh_fo_fphi: String,
    kh_ko_kphi: String,
}

#[derive(Debug, Deserialize)]
struct InformationConfig {
    /// also used for $M$ in correction
    misinfo: String,
    correction: String,
    observation: String,
    inhibition: String,
}

struct Exec<V: MyFloat> {
    graph: GraphB,
    fnum_agents: V,
    mean_degree: V,
    sharer_trust: SharerTrustSamples<V>,
    condition: ConditionSamples<V>,
    uncertainty: UncertaintySamples<V>,
    initial_opinions: InitialOpinions<V>,
    initial_base_rates: InitialBaseRates<V>,
    information: InformationSamples<V>,
    // informer: InformerParams,
    informing: InformingParams,
    /// descending ordered by level
    community_psi1: SupportLevels<V>,
    prob_post_observation: V,
    probabilies: ProbabilityParams<V>,
    prospect: ProspectSamples<V>,
    cpt: CptSamples<V>,
}

impl<V> Exec<V>
where
    V: MyFloat + for<'a> Deserialize<'a>,
{
    fn try_new(config: Config<V>) -> anyhow::Result<Self> {
        let graph: GraphB = config.graph.try_into()?;
        let fnum_agents = V::from_usize(graph.node_count()).unwrap();
        let mean_degree = V::from_usize(graph.edge_count()).unwrap() / fnum_agents;
        Ok(Self {
            graph,
            fnum_agents,
            mean_degree,
            sharer_trust: DataFormat::read(&config.sharer_trust_path)?.parse()?,
            condition: config.condition.try_into()?,
            uncertainty: config.uncertainty.try_into()?,
            initial_opinions: config.initial_opinions,
            initial_base_rates: config.initial_base_rate,
            information: config.information.try_into()?,
            informing: DataFormat::read(&config.informing_path)?.parse()?,
            community_psi1: SupportLevels::conv_from(config.community_psi1_path)?,
            prob_post_observation: config.prob_post_observation,
            probabilies: DataFormat::read(&config.probabilities_path)?.parse()?,
            prospect: ProspectSamples::conv_from(config.prospect_path)?,
            cpt: CptSamples::conv_from(config.cpt_path)?,
        })
    }
}

#[derive(Debug, Deserialize)]
struct SupportLevel<V> {
    level: V,
    agent_idx: usize,
}

struct SupportLevels<V>(Vec<SupportLevel<V>>);

impl<V> SupportLevels<V> {
    fn conv_from(path: String) -> anyhow::Result<Self>
    where
        V: for<'a> Deserialize<'a>,
    {
        let file = File::open(&path)?;
        let mut rdr = csv::Reader::from_reader(file);
        Ok(Self(rdr.deserialize().try_collect()?))
    }

    fn top(&self, n: usize) -> Vec<usize> {
        (0..n).map(|i| self.0[i].agent_idx).collect()
    }
    fn random<R: Rng>(&self, n: usize, rng: &mut R) -> Vec<usize> {
        self.0
            .choose_multiple(rng, n)
            .map(|s| s.agent_idx)
            .collect()
    }
    fn middle(&self, n: usize) -> Vec<usize>
    where
        V: MyFloat,
    {
        let l = self.0.len();
        let c = self.0.len() / 2;
        let median = if l % 2 == 1 {
            self.0[c].level
        } else {
            (self.0[c].level + self.0[c - 1].level) / V::from_u32(2).unwrap()
        };

        let from = c.checked_sub(n).unwrap_or(0);
        let to = (c + n).min(l);
        (from..to)
            .sorted_by(|&i, &j| {
                let a = (self.0[i].level - median).abs();
                let b = (self.0[j].level - median).abs();
                a.partial_cmp(&b).unwrap()
            })
            .take(n)
            .map(|i| self.0[i].agent_idx)
            .collect_vec()
    }
    fn bottom(&self, n: usize) -> Vec<usize> {
        let l = self.0.len() - 1;
        (0..n).map(|i| self.0[l - i].agent_idx).collect()
    }
}

#[derive(Debug, serde::Deserialize)]
struct SharerTrustSamples<V> {
    misinfo: Vec<V>,
    correction: Vec<V>,
    obserbation: Vec<V>,
    inhibition: Vec<V>,
}

struct ConditionSamples<V> {
    h_psi_if_phi0: Vec<MArrD1<Psi, SimplexD1<H, V>>>,
    h_b_if_phi0: Vec<MArrD1<B, SimplexD1<H, V>>>,
    o_b: Vec<MArrD1<B, SimplexD1<O, V>>>,
    a_fh: Vec<MArrD1<FH, SimplexD1<A, V>>>,
    b_kh: Vec<MArrD1<KH, SimplexD1<B, V>>>,
    theta_h: Vec<MArrD1<H, SimplexD1<Theta, V>>>,
    thetad_h: Vec<MArrD1<H, SimplexD1<Thetad, V>>>,
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

impl<V> ConditionRecord<V> {
    fn conv_from<D1, D2>(path: String) -> anyhow::Result<Vec<MArrD1<D1, SimplexD1<D2, V>>>>
    where
        D1: Domain<Idx: Debug>,
        D2: Domain<Idx: Debug>,
        V: MyFloat + for<'a> Deserialize<'a>,
    {
        let file = File::open(&path)?;
        let mut rdr = csv::Reader::from_reader(file);
        rdr.deserialize()
            .map(|res| {
                res.map_err(|e| e.into()).and_then(|rec: Self| {
                    Ok(marr_d1![
                        SimplexD1::try_new(marr_d1![rec.b00, rec.b01], rec.u0)?,
                        SimplexD1::try_new(marr_d1![rec.b10, rec.b11], rec.u1)?,
                    ])
                })
            })
            .try_collect()
    }
}

impl<V: MyFloat + for<'a> Deserialize<'a>> TryFrom<ConditionConfig> for ConditionSamples<V> {
    type Error = anyhow::Error;

    fn try_from(value: ConditionConfig) -> Result<Self, Self::Error> {
        Ok(Self {
            h_psi_if_phi0: ConditionRecord::conv_from(value.h_psi_if_phi0)?,
            h_b_if_phi0: ConditionRecord::conv_from(value.h_b_if_phi0)?,
            o_b: ConditionRecord::conv_from(value.o_b)?,
            a_fh: ConditionRecord::conv_from(value.a_fh)?,
            b_kh: ConditionRecord::conv_from(value.b_kh)?,
            theta_h: ConditionRecord::conv_from(value.theta_h)?,
            thetad_h: ConditionRecord::conv_from(value.thetad_h)?,
        })
    }
}

struct UncertaintySamples<V> {
    fh_fpsi_if_fphi0: Vec<MArrD1<FPsi, V>>,
    kh_kpsi_if_kphi0: Vec<MArrD1<KPsi, V>>,
    fh_fo_fphi: Vec<MArrD2<FO, FPhi, V>>,
    kh_ko_kphi: Vec<MArrD2<KO, KPhi, V>>,
}

#[derive(Debug, Deserialize)]
struct UncertaintyD1Record<V> {
    u0: V,
    u1: V,
}

impl<V> UncertaintyD1Record<V> {
    fn conv_from<D1>(path: String) -> anyhow::Result<Vec<MArrD1<D1, V>>>
    where
        D1: Domain<Idx: Debug>,
        V: MyFloat + for<'a> Deserialize<'a>,
    {
        let file = File::open(&path)?;
        let mut rdr = csv::Reader::from_reader(file);
        rdr.deserialize()
            .map(|res| {
                res.map_err(|e| e.into())
                    .map(|rec: Self| marr_d1![rec.u0, rec.u1])
            })
            .try_collect()
    }
}

#[derive(Debug, Deserialize)]
struct UncertaintyD2Record<V> {
    u00: V,
    u01: V,
    u10: V,
    u11: V,
}

impl<V> UncertaintyD2Record<V> {
    fn conv_from<D1, D2>(path: String) -> anyhow::Result<Vec<MArrD2<D1, D2, V>>>
    where
        D1: Domain<Idx: Debug>,
        D2: Domain<Idx: Debug>,
        V: MyFloat + for<'a> Deserialize<'a>,
    {
        let file = File::open(&path)?;
        let mut rdr = csv::Reader::from_reader(file);
        rdr.deserialize()
            .map(|res| {
                res.map_err(|e| e.into())
                    .map(|rec: Self| marr_d2![[rec.u00, rec.u01], [rec.u10, rec.u11],])
            })
            .try_collect()
    }
}

impl<V: MyFloat + for<'a> Deserialize<'a>> TryFrom<UncertaintyConfig> for UncertaintySamples<V> {
    type Error = anyhow::Error;

    fn try_from(value: UncertaintyConfig) -> Result<Self, Self::Error> {
        Ok(Self {
            fh_fpsi_if_fphi0: UncertaintyD1Record::conv_from(value.fh_fpsi_if_fphi0)?,
            kh_kpsi_if_kphi0: UncertaintyD1Record::conv_from(value.fh_kpsi_if_kphi0)?,
            fh_fo_fphi: UncertaintyD2Record::conv_from(value.fh_fo_fphi)?,
            kh_ko_kphi: UncertaintyD2Record::conv_from(value.kh_ko_kphi)?,
        })
    }
}

struct InformationSamples<V> {
    /// also used for $M$ in correction
    misinfo: Vec<OpinionD1<Psi, V>>,
    correction: Vec<OpinionD1<Psi, V>>,
    observation: Vec<OpinionD1<O, V>>,
    inhibition: Vec<(
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

impl<V> OpinionRecord<V> {
    fn conv_from<D1>(path: String) -> anyhow::Result<Vec<OpinionD1<D1, V>>>
    where
        D1: Domain<Idx: Debug>,
        V: MyFloat + for<'a> Deserialize<'a>,
    {
        let file = File::open(&path)?;
        let mut rdr = csv::Reader::from_reader(file);
        rdr.deserialize()
            .map(|res| {
                res.map_err(|e| e.into()).and_then(|rec: Self| {
                    Ok(OpinionD1::try_new(
                        marr_d1![rec.b0, rec.b1],
                        rec.u,
                        marr_d1![rec.a0, rec.a1],
                    )?)
                })
            })
            .try_collect()
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

impl<V: MyFloat + for<'a> Deserialize<'a>> InhibitionRecord<V> {
    fn conv_from(
        path: String,
    ) -> anyhow::Result<
        Vec<(
            OpinionD1<Phi, V>,
            MArrD1<Psi, SimplexD1<H, V>>,
            MArrD1<B, SimplexD1<H, V>>,
        )>,
    > {
        let file = File::open(&path)?;
        let mut rdr = csv::Reader::from_reader(file);
        rdr.deserialize()
            .map(|res| {
                res.map_err(|e| e.into()).and_then(|rec: Self| {
                    Ok((
                        OpinionD1::try_new(
                            marr_d1![rec.phi_b0, rec.phi_b1],
                            rec.phi_u,
                            marr_d1![rec.phi_a0, rec.phi_a1],
                        )?,
                        marr_d1![
                            SimplexD1::try_new(marr_d1![rec.psi0_b0, rec.psi0_b1], rec.psi0_u)?,
                            SimplexD1::try_new(marr_d1![rec.psi1_b0, rec.psi1_b1], rec.psi1_u)?,
                        ],
                        marr_d1![
                            SimplexD1::try_new(marr_d1![rec.b0_b0, rec.b0_b1], rec.b0_u)?,
                            SimplexD1::try_new(marr_d1![rec.b1_b0, rec.b1_b1], rec.b1_u)?,
                        ],
                    ))
                })
            })
            .try_collect()
    }
}

impl<V: MyFloat + for<'a> Deserialize<'a>> TryFrom<InformationConfig> for InformationSamples<V> {
    type Error = anyhow::Error;

    fn try_from(value: InformationConfig) -> Result<Self, Self::Error> {
        Ok(Self {
            misinfo: OpinionRecord::conv_from(value.misinfo)?,
            correction: OpinionRecord::conv_from(value.correction)?,
            observation: OpinionRecord::conv_from(value.observation)?,
            inhibition: InhibitionRecord::conv_from(value.inhibition)?,
        })
    }
}

#[serde_as]
#[derive(Debug, Clone, Deserialize)]
#[serde(bound(deserialize = "V: serde::Deserialize<'de>"))]
struct InitialOpinions<V: MyFloat> {
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
    #[serde_as(as = "TryFromInto<Vec<(Vec<V>, V)>>")]
    h_psi_if_phi1: MArrD1<Psi, SimplexD1<H, V>>,
    #[serde_as(as = "TryFromInto<Vec<(Vec<V>, V)>>")]
    h_b_if_phi1: MArrD1<B, SimplexD1<H, V>>,
    #[serde_as(as = "TryFromInto<(Vec<V>, V, Vec<V>)>")]
    fpsi: OpinionD1<FPsi, V>,
    #[serde_as(as = "TryFromInto<(Vec<V>, V, Vec<V>)>")]
    fphi: OpinionD1<FPhi, V>,
    #[serde_as(as = "TryFromInto<Vec<(Vec<V>, V)>>")]
    fh_fpsi_if_fphi1: MArrD1<FPsi, SimplexD1<FH, V>>,
    #[serde_as(as = "TryFromInto<(Vec<V>, V, Vec<V>)>")]
    kpsi: OpinionD1<KPsi, V>,
    #[serde_as(as = "TryFromInto<(Vec<V>, V, Vec<V>)>")]
    kphi: OpinionD1<KPhi, V>,
    #[serde_as(as = "TryFromInto<Vec<(Vec<V>, V)>>")]
    kh_kpsi_if_kphi1: MArrD1<KPsi, SimplexD1<KH, V>>,
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

#[derive(Debug, serde::Deserialize)]
enum Sampling {
    Random(usize),
    Top(usize),
    Middle(usize),
    Bottom(usize),
}

#[derive(Debug, serde::Deserialize)]
struct InformerParams {
    num_misinfo: usize,
    num_correction: usize,
    inhibition: Sampling,
}

#[derive(Debug, serde::Deserialize)]
struct Informing {
    step: u32,
    num_agents: usize,
}

#[derive(Debug, serde::Deserialize)]
struct InformingParams {
    informer: InformerParams,
    /// ordered by step & non-duplicated
    misinfo: Vec<Informing>,
    /// ordered by step & non-duplicated
    correction: Vec<Informing>,
    obs_threshold_selfish: usize,
    /// ordered by step & non-duplicated
    inhibition: Vec<Informing>,
}

#[derive(Debug, serde::Deserialize)]
struct ProbabilityParams<V> {
    viewing: Vec<V>,
    viewing_friend: Vec<V>,
    viewing_social: Vec<V>,
    arrival_friend: Vec<V>,
    plural_ignore_friend: Vec<V>,
    plural_ignore_social: Vec<V>,
}

#[derive(Debug, serde::Deserialize)]
struct ProspectRecord<V> {
    x0: V,
    x1: V,
    y: V,
}

struct ProspectSamples<V>(Vec<ProspectRecord<V>>);

impl<V: for<'a> Deserialize<'a>> ProspectSamples<V> {
    fn conv_from(path: String) -> anyhow::Result<Self> {
        let file = File::open(&path)?;
        let mut rdr = csv::Reader::from_reader(file);
        Ok(Self(rdr.deserialize().try_collect()?))
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

struct CptSamples<V>(Vec<CptRecord<V>>);

impl<V: for<'a> Deserialize<'a>> CptSamples<V> {
    fn conv_from(path: String) -> anyhow::Result<Self> {
        let file = File::open(&path)?;
        let mut rdr = csv::Reader::from_reader(file);
        Ok(Self(rdr.deserialize().try_collect()?))
    }
}

impl<V> Executor<V, AgentExt<V>, Instance> for Exec<V>
where
    V: MyFloat,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
{
    fn num_agents(&self) -> usize {
        self.graph.node_count()
    }

    fn graph(&self) -> &GraphB {
        &self.graph
    }
}

#[derive(Default)]
struct AgentExt<V> {
    trusts: InfoMap<V>,
    arrival_prob: Option<V>,
    /// (friend, social)
    viewing_probs: Option<(V, V)>,
    /// (friend, social)
    plural_ignores: Option<(V, V)>,
    visit_prob: Option<V>,
}

impl<V> AgentExt<V> {
    fn clear(&mut self) {
        self.trusts = InfoMap::new();
        self.arrival_prob = None;
        self.viewing_probs = None;
        self.plural_ignores = None;
        self.visit_prob = None;
    }

    fn get_trust<R: Rng>(&mut self, label: InfoLabel, exec: &Exec<V>, rng: &mut R) -> V
    where
        V: MyFloat,
    {
        *self.trusts.entry(label).or_insert_with(|| {
            *match label {
                InfoLabel::Misinfo => &exec.sharer_trust.misinfo,
                InfoLabel::Corrective => &exec.sharer_trust.correction,
                InfoLabel::Observed => &exec.sharer_trust.obserbation,
                InfoLabel::Inhibitive => &exec.sharer_trust.inhibition,
            }
            .choose(rng)
            .unwrap()
        })
    }

    fn get_plural_ignorances<'a, R>(&mut self, exec: &Exec<V>, rng: &mut R) -> (V, V)
    where
        V: MyFloat,
        R: Rng,
    {
        *self.plural_ignores.get_or_insert_with(|| {
            (
                *exec.probabilies.plural_ignore_friend.choose(rng).unwrap(),
                *exec.probabilies.plural_ignore_social.choose(rng).unwrap(),
            )
        })
    }

    fn viewing_probs<'a, R>(&mut self, exec: &Exec<V>, rng: &mut R) -> (V, V)
    where
        V: MyFloat,
        R: Rng,
    {
        *self.viewing_probs.get_or_insert_with(|| {
            (
                *exec.probabilies.viewing_friend.choose(rng).unwrap(),
                *exec.probabilies.viewing_social.choose(rng).unwrap(),
            )
        })
    }

    fn arrival_prob<'a, R>(&mut self, exec: &Exec<V>, rng: &mut R) -> V
    where
        V: MyFloat,
        R: Rng,
    {
        *self
            .arrival_prob
            .get_or_insert_with(|| *exec.probabilies.arrival_friend.choose(rng).unwrap())
    }
}

impl<V> AgentExtTrait<V> for AgentExt<V>
where
    V: MyFloat,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    Open01: Distribution<V>,
{
    type Exec = Exec<V>;
    type Ix = Instance;

    fn informer_trusts<'a, R: Rng>(
        &mut self,
        ins: &mut InstanceWrapper<'a, Self::Exec, V, R, Self::Ix>,
        _: InfoIdx,
    ) -> Trusts<V> {
        let trust_mis = self.get_trust(InfoLabel::Misinfo, ins.exec, &mut ins.rng);
        Trusts {
            p: V::one(),
            fp: V::one(),
            kp: V::one(),
            fm: trust_mis,
            km: trust_mis,
        }
    }

    fn informer_access_probs<'a, R: Rng>(
        &mut self,
        ins: &mut InstanceWrapper<'a, Self::Exec, V, R, Self::Ix>,
        _: InfoIdx,
    ) -> AccessProb<V> {
        let (fm, km) = self.get_plural_ignorances(ins.exec, &mut ins.rng);
        AccessProb {
            fp: V::zero(),
            kp: V::zero(),
            pred_fp: V::zero(),
            fm,
            km,
        }
    }

    fn sharer_trusts<'a, R: Rng>(
        &mut self,
        ins: &mut InstanceWrapper<'a, Self::Exec, V, R, Self::Ix>,
        info_idx: InfoIdx,
    ) -> Trusts<V> {
        let trust = self.get_trust(*ins.get_info_label(info_idx), ins.exec, &mut ins.rng);
        let trust_mis = self.get_trust(InfoLabel::Misinfo, ins.exec, &mut ins.rng);
        Trusts {
            p: trust,
            fp: trust,
            kp: trust,
            fm: trust_mis,
            km: trust_mis,
        }
    }

    fn sharer_access_probs<'a, R: Rng>(
        &mut self,
        ins: &mut InstanceWrapper<'a, Self::Exec, V, R, Self::Ix>,
        info_idx: InfoIdx,
    ) -> AccessProb<V> {
        let r = Instance::receipt_prob(ins, info_idx);
        let (fq, kq) = self.viewing_probs(ins.exec, &mut ins.rng);
        let arr = self.arrival_prob(ins.exec, &mut ins.rng);
        let (fm, km) = self.get_plural_ignorances(ins.exec, &mut ins.rng);
        AccessProb {
            fp: fq * r,
            kp: kq * r,
            pred_fp: fq * arr,
            fm,
            km,
        }
    }

    fn visit_prob<R: Rng>(&mut self, exec: &Self::Exec, rng: &mut R) -> V {
        *self
            .visit_prob
            .get_or_insert_with(|| *exec.probabilies.viewing.choose(rng).unwrap())
    }

    fn reset<R: Rng>(wrapper: &mut AgentWrapper<V, Self>, exec: &Exec<V>, rng: &mut R) {
        wrapper.ext.clear();
        wrapper.core.reset(|ops, decision| {
            let o_b = exec.condition.o_b.choose(rng).unwrap().to_owned();
            let b_kh = exec.condition.b_kh.choose(rng).unwrap().to_owned();
            let a_fh = exec.condition.a_fh.choose(rng).unwrap().to_owned();
            let theta_h = exec.condition.theta_h.choose(rng).unwrap().to_owned();
            let thetad_h = exec.condition.thetad_h.choose(rng).unwrap().to_owned();
            let h_psi_if_phi0 = exec.condition.h_psi_if_phi0.choose(rng).unwrap().to_owned();
            let h_b_if_phi0 = exec.condition.h_b_if_phi0.choose(rng).unwrap().to_owned();
            let uncertainty_fh_fpsi_if_fphi0 = exec
                .uncertainty
                .fh_fpsi_if_fphi0
                .choose(rng)
                .unwrap()
                .to_owned();
            let uncertainty_kh_kpsi_if_kphi0 = exec
                .uncertainty
                .kh_kpsi_if_kphi0
                .choose(rng)
                .unwrap()
                .to_owned();
            let uncertainty_fh_fo_fphi =
                exec.uncertainty.fh_fo_fphi.choose(rng).unwrap().to_owned();
            let uncertainty_kh_ko_kphi =
                exec.uncertainty.kh_ko_kphi.choose(rng).unwrap().to_owned();
            ops.fixed.reset(
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
            } = exec.initial_opinions.clone();
            ops.state.reset(
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
            } = exec.initial_base_rates.clone();
            ops.ded.reset(
                OpinionD1::vacuous_with(h),
                OpinionD1::vacuous_with(fh),
                OpinionD1::vacuous_with(kh),
                OpinionD1::vacuous_with(a),
                OpinionD1::vacuous_with(b),
                OpinionD1::vacuous_with(theta),
                OpinionD1::vacuous_with(thetad),
            );
            decision.reset(0, |prospect, cpt| {
                let &ProspectRecord { x0, x1, y } = exec.prospect.0.choose(rng).unwrap();
                prospect.reset(x0, x1, y);
                let &CptRecord {
                    alpha,
                    beta,
                    gamma,
                    delta,
                    lambda,
                } = exec.cpt.0.choose(rng).unwrap();
                cpt.reset(alpha, beta, lambda, gamma, delta);
            });
        });
    }
}

#[derive(Default)]
struct InfoMap<T> {
    misinfo: Option<T>,
    correction: Option<T>,
    observation: Option<T>,
    inhibition: Option<T>,
}

impl<T> InfoMap<T> {
    fn new() -> Self {
        Self {
            misinfo: None,
            correction: None,
            observation: None,
            inhibition: None,
        }
    }
    fn entry(&mut self, label: InfoLabel) -> MyEntry<'_, T> {
        let value = match label {
            InfoLabel::Misinfo => &mut self.misinfo,
            InfoLabel::Corrective => &mut self.correction,
            InfoLabel::Observed => &mut self.observation,
            InfoLabel::Inhibitive => &mut self.inhibition,
        };
        MyEntry { value }
    }
}

struct MyEntry<'a, T> {
    value: &'a mut Option<T>,
}

impl<'a, T> MyEntry<'a, T> {
    fn or_insert_with<F: FnOnce() -> T>(self, default: F) -> &'a mut T {
        self.value.get_or_insert_with(default)
    }
}

struct Instance {
    /// desc ordered by support levels
    misinfo: Informers,
    /// asc ordered by support levels
    corection: Informers,
    observation: Vec<usize>,
    /// order is determined by inhibition sampling parameter
    inhibition: Informers,
}

struct Informers {
    agents: Vec<usize>,
    next_index: usize,
}

impl Informers {
    fn new(agents: Vec<usize>) -> Self {
        Self {
            agents,
            next_index: 0,
        }
    }
}

impl Informers {
    fn pick(&mut self, t: u32, informings: &[Informing]) -> Option<Drain<'_, usize>> {
        match informings.get(self.next_index) {
            Some(&Informing { step, num_agents }) if step == t => {
                self.next_index += 1;
                Some(self.agents.drain(0..(num_agents.min(self.agents.len()))))
            }
            _ => None,
        }
    }
}

impl Instance {
    fn receipt_prob<'a, V, R>(
        ins: &InstanceWrapper<'a, Exec<V>, V, R, Self>,
        info_idx: InfoIdx,
    ) -> V
    where
        V: MyFloat,
    {
        V::one()
            - (V::one() - V::from_usize(ins.num_shared(info_idx)).unwrap() / ins.exec.fnum_agents)
                .powf(ins.exec.mean_degree)
    }
}

impl<V, R> InstanceExt<V, R, Exec<V>> for Instance
where
    V: MyFloat,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    R: Rng,
{
    fn from_exec(exec: &Exec<V>, rng: &mut R) -> Self {
        Self {
            misinfo: Informers::new(exec.community_psi1.top(exec.informing.informer.num_misinfo)),
            corection: Informers::new(
                exec.community_psi1
                    .bottom(exec.informing.informer.num_correction),
            ),
            observation: (0..exec.graph.node_count()).collect(),
            inhibition: match exec.informing.informer.inhibition {
                Sampling::Random(n) => Informers::new(exec.community_psi1.random(n, rng)),
                Sampling::Top(n) => Informers::new(exec.community_psi1.top(n)),
                Sampling::Middle(n) => Informers::new(exec.community_psi1.middle(n)),
                Sampling::Bottom(n) => Informers::new(exec.community_psi1.bottom(n)),
            },
        }
    }

    fn is_continued(&self, exec: &Exec<V>) -> bool {
        (self.misinfo.next_index < exec.informing.misinfo.len() - 1)
            && (self.corection.next_index < exec.informing.correction.len() - 1)
            && (self.inhibition.next_index < exec.informing.inhibition.len() - 1)
    }

    fn get_informers_with<'a>(
        ins: &mut InstanceWrapper<'a, Exec<V>, V, R, Self>,
        t: u32,
    ) -> Vec<(AgentIdx, InfoContent<'a, V>)> {
        let mut informers = Vec::new();
        if let Some(d) = ins.ext.misinfo.pick(t, &ins.exec.informing.misinfo) {
            for agent_idx in d {
                informers.push((
                    agent_idx.into(),
                    InfoContent::Misinfo {
                        op: Cow::Borrowed(
                            ins.exec.information.misinfo.choose(&mut ins.rng).unwrap(),
                        ),
                    },
                ));
            }
        }
        if let Some(d) = ins.ext.corection.pick(t, &ins.exec.informing.correction) {
            for agent_idx in d {
                informers.push((
                    agent_idx.into(),
                    InfoContent::Correction {
                        op: Cow::Borrowed(
                            ins.exec
                                .information
                                .correction
                                .choose(&mut ins.rng)
                                .unwrap(),
                        ),
                        misinfo: Cow::Borrowed(
                            ins.exec.information.misinfo.choose(&mut ins.rng).unwrap(),
                        ),
                    },
                ));
            }
        }
        if ins.total_num_selfish() > ins.exec.informing.obs_threshold_selfish {
            ins.ext.observation.retain(|&agent_idx| {
                if ins.exec.prob_post_observation <= ins.rng.gen() {
                    return true;
                }
                let op = ins
                    .exec
                    .information
                    .observation
                    .iter()
                    .choose(&mut ins.rng)
                    .unwrap();
                informers.push((
                    agent_idx.into(),
                    InfoContent::Observation {
                        op: Cow::Borrowed(op),
                    },
                ));
                false
            });
        }
        if let Some(d) = ins.ext.inhibition.pick(t, &ins.exec.informing.inhibition) {
            for agent_idx in d {
                let (op1, op2, op3) = ins
                    .exec
                    .information
                    .inhibition
                    .choose(&mut ins.rng)
                    .unwrap();
                informers.push((
                    agent_idx.into(),
                    InfoContent::Inhibition {
                        op1: Cow::Borrowed(op1),
                        op2: Cow::Borrowed(op2),
                        op3: Cow::Borrowed(op3),
                    },
                ));
            }
        }
        informers
    }
}
