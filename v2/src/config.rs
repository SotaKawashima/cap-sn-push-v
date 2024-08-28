use std::{
    fmt::Debug,
    fs::File,
    path::{Path, PathBuf},
};

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
use rand_distr::{Distribution, Exp1, Open01, Standard, StandardNormal};
use serde::{de::DeserializeOwned, Deserialize};
use serde_with::{serde_as, TryFromInto};
use subjective_logic::{
    domain::Domain,
    errors::check_unit_interval,
    marr_d1, marr_d2,
    mul::labeled::{OpinionD1, SimplexD1},
    multi_array::labeled::{MArrD1, MArrD2},
};

use crate::exec::Exec;

#[derive(Debug, Deserialize)]
pub struct Config {
    agent: AgentConfig,
    strategy: StrategyConfig,
    network: NetworkConfig,
}

fn join_path<P: AsRef<Path>>(path: &mut PathBuf, root: P) {
    if !path.is_absolute() {
        *path = root.as_ref().join(&path);
    }
}

impl Config {
    pub fn into_exec<V, P>(self, root: P) -> anyhow::Result<Exec<V>>
    where
        V: MyFloat + for<'a> Deserialize<'a>,
        P: AsRef<Path>,
    {
        let Self {
            mut agent,
            mut strategy,
            mut network,
        } = self;
        agent.set_root(&root);
        strategy.set_root(&root);
        network.set_root(&root);

        let graph = network.parse_graph()?;
        let fnum_agents = V::from_usize(graph.node_count()).unwrap();
        let mean_degree = V::from_usize(graph.edge_count()).unwrap() / fnum_agents;
        let community_psi1 =
            read_csv_with(network.community, |iter| SupportLevels::from_iter(iter))?;

        let InitialStates {
            initial_opinions,
            initial_base_rates,
        } = DataFormat::read(&agent.initial_states)?.parse()?;

        Ok(Exec {
            graph,
            fnum_agents,
            mean_degree,
            sharer_trust: DataFormat::read(&agent.sharer_trust)?.parse()?,
            opinion: OpinionSamples {
                condition: agent.condition.try_into()?,
                uncertainty: agent.uncertainty.try_into()?,
                initial_opinions,
                initial_base_rates,
            },
            information: strategy.information.try_into()?,
            informing: DataFormat::read(&strategy.informing)?.parse()?,
            community_psi1,
            probabilies: DataFormat::read(&agent.probabilities)?.parse()?,
            prospect: ProspectSamples(read_csv(&agent.prospect)?),
            cpt: CptSamples(read_csv(&agent.cpt)?),
        })
    }
}

#[derive(Debug, Deserialize)]
struct AgentConfig {
    probabilities: PathBuf,
    sharer_trust: PathBuf,
    prospect: PathBuf,
    cpt: PathBuf,
    initial_states: PathBuf,
    condition: ConditionConfig,
    uncertainty: UncertaintyConfig,
}

impl AgentConfig {
    fn set_root<P: AsRef<Path>>(&mut self, root: P) {
        join_path(&mut self.probabilities, &root);
        join_path(&mut self.sharer_trust, &root);
        join_path(&mut self.prospect, &root);
        join_path(&mut self.cpt, &root);
        join_path(&mut self.initial_states, &root);
        self.condition.set_root(&root);
        self.uncertainty.set_root(&root);
    }
}

#[derive(Debug, Deserialize)]
struct StrategyConfig {
    informing: PathBuf,
    information: InformationConfig,
}

impl StrategyConfig {
    fn set_root<P: AsRef<Path>>(&mut self, root: P) {
        join_path(&mut self.informing, &root);
        self.information.set_root(&root);
    }
}

#[derive(Debug, Deserialize)]
pub struct InformationConfig {
    /// also used for $M$ in correction
    misinfo: PathBuf,
    correction: PathBuf,
    observation: PathBuf,
    inhibition: PathBuf,
}

impl InformationConfig {
    fn set_root<P: AsRef<Path>>(&mut self, root: P) {
        join_path(&mut self.misinfo, &root);
        join_path(&mut self.correction, &root);
        join_path(&mut self.observation, &root);
        join_path(&mut self.inhibition, &root);
    }
}

#[derive(Debug, Deserialize)]
struct NetworkConfig {
    path: PathBuf,
    graph: PathBuf,
    directed: bool,
    transposed: bool,
    community: PathBuf,
}

impl NetworkConfig {
    fn set_root<P: AsRef<Path>>(&mut self, root: P) {
        join_path(&mut self.path, &root);
        join_path(&mut self.graph, &self.path);
        join_path(&mut self.community, &self.path);
    }

    fn parse_graph(&self) -> anyhow::Result<GraphB> {
        let builder = graph_lib::io::ParseBuilder::new(
            File::open(&self.graph)?,
            graph_lib::io::DataFormat::EdgeList,
        );
        if !self.directed {
            if self.transposed {
                Ok(GraphB::Ud(builder.transpose().parse()?))
            } else {
                Ok(GraphB::Ud(builder.parse()?))
            }
        } else {
            Ok(GraphB::Di(builder.parse()?))
        }
    }
}

#[serde_as]
#[derive(Debug, Deserialize)]
#[serde(bound(deserialize = "V: serde::Deserialize<'de>"))]
struct InitialStates<V: MyFloat> {
    initial_opinions: InitialOpinions<V>,
    initial_base_rates: InitialBaseRates<V>,
}

#[derive(Debug, Deserialize)]
struct ConditionConfig {
    h_psi_if_phi0: PathBuf,
    h_b_if_phi0: PathBuf,
    o_b: PathBuf,
    a_fh: PathBuf,
    b_kh: PathBuf,
    theta_h: PathBuf,
    thetad_h: PathBuf,
}

impl ConditionConfig {
    fn set_root<P: AsRef<Path>>(&mut self, root: P) {
        join_path(&mut self.h_psi_if_phi0, &root);
        join_path(&mut self.h_b_if_phi0, &root);
        join_path(&mut self.o_b, &root);
        join_path(&mut self.a_fh, &root);
        join_path(&mut self.b_kh, &root);
        join_path(&mut self.theta_h, &root);
        join_path(&mut self.thetad_h, &root);
    }
}

#[derive(Debug, Deserialize)]
struct UncertaintyConfig {
    fh_fpsi_if_fphi0: PathBuf,
    kh_kpsi_if_kphi0: PathBuf,
    fh_fo_fphi: PathBuf,
    kh_ko_kphi: PathBuf,
}

impl UncertaintyConfig {
    fn set_root<P: AsRef<Path>>(&mut self, root: P) {
        join_path(&mut self.fh_fpsi_if_fphi0, &root);
        join_path(&mut self.kh_kpsi_if_kphi0, &root);
        join_path(&mut self.fh_fo_fphi, &root);
        join_path(&mut self.kh_ko_kphi, &root);
    }
}

pub struct OpinionSamples<V: MyFloat> {
    initial_opinions: InitialOpinions<V>,
    initial_base_rates: InitialBaseRates<V>,
    condition: ConditionSamples<V>,
    uncertainty: UncertaintySamples<V>,
}

impl<V: MyFloat> OpinionSamples<V> {
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

impl<V, D1, D2> TryFrom<ConditionRecord<V>> for MArrD1<D1, SimplexD1<D2, V>>
where
    V: MyFloat,
    D1: Domain<Idx: Debug>,
    D2: Domain<Idx: Debug>,
{
    type Error = anyhow::Error;
    fn try_from(value: ConditionRecord<V>) -> Result<Self, Self::Error> {
        Ok(marr_d1![
            SimplexD1::try_new(marr_d1![value.b00, value.b01], value.u0)?,
            SimplexD1::try_new(marr_d1![value.b10, value.b11], value.u1)?,
        ])
    }
}

impl<V: MyFloat + for<'a> Deserialize<'a>> TryFrom<ConditionConfig> for ConditionSamples<V> {
    type Error = anyhow::Error;

    fn try_from(value: ConditionConfig) -> Result<Self, Self::Error> {
        Ok(Self {
            h_psi_if_phi0: read_csv_and_then(value.h_psi_if_phi0, ConditionRecord::try_into)?,
            h_b_if_phi0: read_csv_and_then(value.h_b_if_phi0, ConditionRecord::try_into)?,
            o_b: read_csv_and_then(value.o_b, ConditionRecord::try_into)?,
            a_fh: read_csv_and_then(value.a_fh, ConditionRecord::try_into)?,
            b_kh: read_csv_and_then(value.b_kh, ConditionRecord::try_into)?,
            theta_h: read_csv_and_then(value.theta_h, ConditionRecord::try_into)?,
            thetad_h: read_csv_and_then(value.thetad_h, ConditionRecord::try_into)?,
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

impl<V: MyFloat, D1: Domain> TryFrom<UncertaintyD1Record<V>> for MArrD1<D1, V> {
    type Error = anyhow::Error;
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
    type Error = anyhow::Error;
    fn try_from(value: UncertaintyD2Record<V>) -> Result<Self, Self::Error> {
        check_unit_interval(value.u00, "u00")?;
        check_unit_interval(value.u01, "u01")?;
        check_unit_interval(value.u10, "u10")?;
        check_unit_interval(value.u11, "u11")?;
        Ok(marr_d2![[value.u00, value.u01], [value.u10, value.u11],])
    }
}

impl<V: MyFloat + for<'a> Deserialize<'a>> TryFrom<UncertaintyConfig> for UncertaintySamples<V> {
    type Error = anyhow::Error;

    fn try_from(value: UncertaintyConfig) -> Result<Self, Self::Error> {
        Ok(Self {
            fh_fpsi_if_fphi0: read_csv_and_then(
                value.fh_fpsi_if_fphi0,
                UncertaintyD1Record::try_into,
            )?,
            kh_kpsi_if_kphi0: read_csv_and_then(
                value.kh_kpsi_if_kphi0,
                UncertaintyD1Record::try_into,
            )?,
            fh_fo_fphi: read_csv_and_then(value.fh_fo_fphi, UncertaintyD2Record::try_into)?,
            kh_ko_kphi: read_csv_and_then(value.kh_ko_kphi, UncertaintyD2Record::try_into)?,
        })
    }
}

fn reset_fixed<V: MyFloat, R: Rng>(
    condition: &ConditionSamples<V>,
    uncertainty: &UncertaintySamples<V>,
    fixed: &mut FixedOpinions<V>,
    rng: &mut R,
) {
    let o_b = condition.o_b.choose(rng).unwrap().to_owned();
    let b_kh = condition.b_kh.choose(rng).unwrap().to_owned();
    let a_fh = condition.a_fh.choose(rng).unwrap().to_owned();
    let theta_h = condition.theta_h.choose(rng).unwrap().to_owned();
    let thetad_h = condition.thetad_h.choose(rng).unwrap().to_owned();
    let h_psi_if_phi0 = condition.h_psi_if_phi0.choose(rng).unwrap().to_owned();
    let h_b_if_phi0 = condition.h_b_if_phi0.choose(rng).unwrap().to_owned();
    let uncertainty_fh_fpsi_if_fphi0 = uncertainty.fh_fpsi_if_fphi0.choose(rng).unwrap().to_owned();
    let uncertainty_kh_kpsi_if_kphi0 = uncertainty.kh_kpsi_if_kphi0.choose(rng).unwrap().to_owned();
    let uncertainty_fh_fo_fphi = uncertainty.fh_fo_fphi.choose(rng).unwrap().to_owned();
    let uncertainty_kh_ko_kphi = uncertainty.kh_ko_kphi.choose(rng).unwrap().to_owned();
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
pub struct SharerTrustSamples<V> {
    pub misinfo: Vec<V>,
    pub correction: Vec<V>,
    pub obserbation: Vec<V>,
    pub inhibition: Vec<V>,
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
    type Error = anyhow::Error;
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
    type Error = anyhow::Error;
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

impl<V: MyFloat + for<'a> Deserialize<'a>> TryFrom<InformationConfig> for InformationSamples<V> {
    type Error = anyhow::Error;

    fn try_from(value: InformationConfig) -> Result<Self, Self::Error> {
        Ok(Self {
            misinfo: read_csv_and_then(&value.misinfo, OpinionRecord::try_into)?,
            correction: read_csv_and_then(&value.correction, OpinionRecord::try_into)?,
            observation: read_csv_and_then(&value.observation, OpinionRecord::try_into)?,
            inhibition: read_csv_and_then(&value.inhibition, InhibitionRecord::try_into)?,
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

#[derive(Debug, serde::Deserialize)]
pub struct ProbabilitySamples<V> {
    pub viewing: Vec<V>,
    pub viewing_friend: Vec<V>,
    pub viewing_social: Vec<V>,
    pub arrival_friend: Vec<V>,
    pub plural_ignore_friend: Vec<V>,
    pub plural_ignore_social: Vec<V>,
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

fn read_csv_and_then<T, P, F, U>(path: P, f: F) -> anyhow::Result<Vec<U>>
where
    T: DeserializeOwned,
    P: AsRef<Path>,
    F: Fn(T) -> Result<U, anyhow::Error>,
{
    let file = File::open(path)?;
    let mut rdr = csv::Reader::from_reader(file);
    Ok(rdr.deserialize::<T>().map(|res| f(res?)).try_collect()?)
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
    use std::{fs::read_to_string, path::Path};

    use subjective_logic::{
        marr_d1, marr_d2,
        mul::labeled::{OpinionD1, SimplexD1},
    };

    use super::{Config, SupportLevel, SupportLevels};
    use crate::exec::Exec;

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
        let config_path = Path::new("./test/config.toml");
        let config: Config = toml::from_str(&read_to_string(&config_path)?)?;
        let exec: Exec<f32> = config.into_exec(config_path.parent().unwrap())?;
        assert_eq!(exec.community_psi1.levels.len(), 100);
        assert!(
            exec.community_psi1.levels[exec.community_psi1.indexes_by_level[10]]
                > exec.community_psi1.levels[exec.community_psi1.indexes_by_level[90]]
        );
        assert_eq!(exec.opinion.condition.o_b.len(), 7);
        assert_eq!(
            exec.opinion.condition.o_b[3],
            marr_d1![
                SimplexD1::new(marr_d1![1.0, 0.00], 0.00),
                SimplexD1::new(marr_d1![0.0, 0.75], 0.25)
            ]
        );
        assert_eq!(
            exec.opinion.uncertainty.fh_fpsi_if_fphi0[0],
            marr_d1![0.1, 0.1]
        );
        assert_eq!(
            exec.opinion.uncertainty.kh_ko_kphi[0],
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
        Ok(())
    }
}
