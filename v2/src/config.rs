use std::{fmt::Debug, fs::File};

use base::{
    decision::{Prospect, CPT},
    opinion::{
        DeducedOpinions, FPhi, FPsi, FixedOpinions, KPhi, KPsi, MyFloat, MyOpinions, Phi, Psi,
        StateOpinions, Theta, Thetad, A, B, FH, FO, H, KH, KO, O,
    },
    util::GraphInfo,
};
use graph_lib::prelude::{Graph, GraphB};
use input::format::DataFormat;
use itertools::Itertools;
use rand::{seq::SliceRandom, Rng};
use rand_distr::{Distribution, Exp1, Open01, Standard, StandardNormal};
use serde::Deserialize;
use serde_with::{serde_as, TryFromInto};
use subjective_logic::{
    domain::Domain,
    marr_d1, marr_d2,
    mul::labeled::{OpinionD1, SimplexD1},
    multi_array::labeled::{MArrD1, MArrD2},
};

use crate::exec::Exec;

#[derive(Debug, Deserialize)]
pub struct Config {
    agent: AgentConfig,
    strategy: StrategyConfig,
    environment: EnvironmentConfig,
}

#[derive(Debug, Deserialize)]
struct AgentConfig {
    probabilities: String,
    sharer_trust: String,
    prospect: String,
    cpt: String,
    initial_states: String,
    condition: ConditionConfig,
    uncertainty: UncertaintyConfig,
}

#[derive(Debug, Deserialize)]
struct StrategyConfig {
    informing: String,
    information: InformationConfig,
}

#[derive(Debug, Deserialize)]
struct EnvironmentConfig {
    network: String,
}

#[derive(Debug, Deserialize)]
struct NetworkConfig {
    graph: GraphInfo,
    community: String,
}

impl<V> TryFrom<Config> for Exec<V>
where
    V: MyFloat + for<'a> Deserialize<'a>,
{
    type Error = anyhow::Error;

    fn try_from(
        Config {
            agent,
            strategy,
            environment,
        }: Config,
    ) -> Result<Self, Self::Error> {
        let network: NetworkConfig = DataFormat::read(&environment.network)?.parse()?;
        let graph: GraphB = network.graph.try_into()?;
        let fnum_agents = V::from_usize(graph.node_count()).unwrap();
        let mean_degree = V::from_usize(graph.edge_count()).unwrap() / fnum_agents;
        let community_psi1 = SupportLevels::conv_from(network.community)?;

        let InitialStates {
            initial_opinions,
            initial_base_rates,
        } = DataFormat::read(&agent.initial_states)?.parse()?;

        Ok(Self {
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
            prospect: ProspectSamples::conv_from(agent.prospect)?,
            cpt: CptSamples::conv_from(agent.cpt)?,
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
    kh_kpsi_if_kphi0: String,
    fh_fo_fphi: String,
    kh_ko_kphi: String,
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
            kh_kpsi_if_kphi0: UncertaintyD1Record::conv_from(value.kh_kpsi_if_kphi0)?,
            fh_fo_fphi: UncertaintyD2Record::conv_from(value.fh_fo_fphi)?,
            kh_ko_kphi: UncertaintyD2Record::conv_from(value.kh_ko_kphi)?,
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
pub struct InformationConfig {
    /// also used for $M$ in correction
    misinfo: String,
    correction: String,
    observation: String,
    inhibition: String,
}

#[derive(Debug, Deserialize)]
pub struct SupportLevel<V> {
    level: V,
    agent_idx: usize,
}

pub struct SupportLevels<V>(Vec<SupportLevel<V>>);

impl<V> SupportLevels<V> {
    /// descending ordered by level
    pub fn conv_from(path: String) -> anyhow::Result<Self>
    where
        V: MyFloat + for<'a> Deserialize<'a>,
    {
        let file = File::open(&path)?;
        let mut rdr = csv::Reader::from_reader(file);
        let mut v: Vec<SupportLevel<V>> = rdr.deserialize().try_collect()?;
        v.sort_by(|a, b| b.level.partial_cmp(&a.level).unwrap());
        Ok(Self(v))
    }

    pub fn top(&self, n: usize) -> Vec<usize> {
        (0..n).map(|i| self.0[i].agent_idx).collect()
    }
    pub fn random<R: Rng>(&self, n: usize, rng: &mut R) -> Vec<usize> {
        self.0
            .choose_multiple(rng, n)
            .map(|s| s.agent_idx)
            .collect()
    }
    pub fn middle(&self, n: usize) -> Vec<usize>
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
    pub fn bottom(&self, n: usize) -> Vec<usize> {
        let l = self.0.len() - 1;
        (0..n).map(|i| self.0[l - i].agent_idx).collect()
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
pub enum Sampling {
    Random(usize),
    Top(usize),
    Middle(usize),
    Bottom(usize),
}

#[derive(Debug, serde::Deserialize)]
pub struct InformerParams {
    pub num_misinfo: usize,
    pub num_correction: usize,
    pub inhibition: Sampling,
}

#[derive(Debug, serde::Deserialize)]
pub struct Informing {
    pub step: u32,
    pub num_agents: usize,
}

#[derive(Debug, serde::Deserialize)]
pub struct InformingParams<V> {
    pub informer: InformerParams,
    /// ordered by step & non-duplicated
    pub misinfo: Vec<Informing>,
    /// ordered by step & non-duplicated
    pub correction: Vec<Informing>,
    pub obs_threshold_selfish: usize,
    /// ordered by step & non-duplicated
    pub inhibition: Vec<Informing>,
    pub prob_post_observation: V,
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

impl<V: for<'a> Deserialize<'a>> ProspectSamples<V> {
    pub fn conv_from(path: String) -> anyhow::Result<Self> {
        let file = File::open(&path)?;
        let mut rdr = csv::Reader::from_reader(file);
        Ok(Self(rdr.deserialize().try_collect()?))
    }
}

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

impl<V: for<'a> Deserialize<'a>> CptSamples<V> {
    pub fn conv_from(path: String) -> anyhow::Result<Self> {
        let file = File::open(&path)?;
        let mut rdr = csv::Reader::from_reader(file);
        Ok(Self(rdr.deserialize().try_collect()?))
    }
}

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

#[cfg(test)]
mod tests {
    use std::fs::read_to_string;

    use subjective_logic::{
        marr_d1, marr_d2,
        mul::labeled::{OpinionD1, SimplexD1},
    };

    use crate::exec::Exec;

    use super::Config;

    #[test]
    fn test_config() -> anyhow::Result<()> {
        let config: Config = toml::from_str(&read_to_string("./test/config.toml")?)?;
        let exec: Exec<f32> = config.try_into()?;
        assert_eq!(exec.community_psi1.0.len(), 100);
        assert!(exec.community_psi1.0[10].level > exec.community_psi1.0[90].level);
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
