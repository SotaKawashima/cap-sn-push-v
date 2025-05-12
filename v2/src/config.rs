use std::{
    fmt::Debug,
    fs::File,
    path::{Path, PathBuf},
};

use anyhow::{bail, ensure, Context};
use base::opinion::{MyFloat, Phi, Psi, B, H};
use graph_lib::prelude::{Graph, GraphB};
use input::format::DataFormat;
use itertools::Itertools;
use rand_distr::{Distribution, Open01};
use serde::Deserialize;
use subjective_logic::{
    domain::{Domain, Keys},
    errors::{check_unit_interval, InvalidValueError},
    iter::FromFn,
    marr_d1, marr_d2,
    mul::labeled::{OpinionD1, SimplexD1},
    multi_array::labeled::{MArrD1, MArrD2},
};

use crate::{
    exec::Exec,
    io::{read_csv, MyPath},
    parameter::{
        ConditionSampler, ConditionSamples, CptRecord, CptSamples, InformationSamples,
        InformingParams, InitialStates, OpinionSamples, ProbabilitySamples, ProspectRecord,
        ProspectSamples, SamplerOption, SharerTrustSamples, SimplexContainer, SimplexIndexed,
        SupportLevelTable, UncertaintySamples,
    },
};

pub struct Config<V>
where
    V: MyFloat,
    Open01: Distribution<V>,
{
    agent: MyPath<AgentConfig<V>>,
    strategy: MyPath<StrategyConfig<V>>,
    network: MyPath<NetworkConfig<V>>,
}

impl<V> Config<V>
where
    V: MyFloat + for<'a> Deserialize<'a>,
    Open01: Distribution<V>,
{
    pub fn try_new<P: AsRef<Path>>(network: P, agent: P, strategy: P) -> anyhow::Result<Self> {
        Ok(Self {
            agent: agent.as_ref().into(),
            strategy: strategy.as_ref().into(),
            network: network.as_ref().into(),
        })
    }

    pub fn into_exec(self, enable_inhibition: bool, delay_selfish: u32) -> anyhow::Result<Exec<V>>
    where
        V: MyFloat + for<'a> Deserialize<'a>,
        Open01: Distribution<V>,
    {
        let Self {
            agent,
            strategy,
            network,
        } = self;

        let DeNetworkConfig {
            graph,
            fnum_agents,
            mean_degree,
            community_psi1,
        } = network.de("")?;

        let DeAgentConfig {
            probabilities,
            sharer_trust,
            prospect,
            cpt,
            opinion,
        } = agent.de("")?;

        let DeStrategyConfig {
            informing,
            information,
        } = strategy.de("")?;

        Ok(Exec {
            enable_inhibition,
            delay_selfish,
            graph,
            fnum_agents,
            mean_degree,
            sharer_trust,
            opinion,
            information,
            informing,
            community_psi1,
            probabilities,
            prospect,
            cpt,
        })
    }
}

trait DeserializeAt<T> {
    fn de<P: AsRef<Path>>(self, at: P) -> anyhow::Result<T>;
}

#[derive(Deserialize)]
#[serde(bound(deserialize = "V: serde::Deserialize<'de>"))]
pub struct AgentConfig<V>
where
    V: MyFloat,
    Open01: Distribution<V>,
{
    probabilities: MyPath<ProbabilitySamples<V>>,
    sharer_trust: MyPath<SharerTrustSamples<V>>,
    prospect: MyPath<Vec<ProspectRecord<V>>>,
    cpt: MyPath<Vec<CptRecord<V>>>,
    initial_states: MyPath<InitialStates<V>>,
    condition: MyPath<ConditionConfig<V>>,
    uncertainty: MyPath<UncertaintyConfig<V>>,
}

struct DeAgentConfig<V>
where
    V: MyFloat,
    Open01: Distribution<V>,
{
    probabilities: ProbabilitySamples<V>,
    sharer_trust: SharerTrustSamples<V>,
    prospect: ProspectSamples<V>,
    cpt: CptSamples<V>,
    opinion: OpinionSamples<V>,
}

impl<V> DeserializeAt<DeAgentConfig<V>> for MyPath<AgentConfig<V>>
where
    V: MyFloat + for<'a> Deserialize<'a>,
    Open01: Distribution<V>,
{
    fn de<P: AsRef<Path>>(self, parent: P) -> anyhow::Result<DeAgentConfig<V>> {
        let child = self.verified_child(&parent)?;
        let agent = self.parse_at(&parent, DataFormat::de)?;

        let initial_states = agent.initial_states.parse_at(&child, DataFormat::de)?;
        Ok(DeAgentConfig {
            probabilities: agent.probabilities.parse_at(&child, DataFormat::de)?,
            sharer_trust: agent.sharer_trust.parse_at(&child, DataFormat::de)?,
            prospect: ProspectSamples(agent.prospect.parse_at(&child, read_csv)?),
            cpt: CptSamples(agent.cpt.parse_at(&child, read_csv)?),
            opinion: OpinionSamples {
                initial_opinions: initial_states.initial_opinions,
                initial_base_rates: initial_states.initial_base_rates,
                condition: agent.condition.de(&child)?,
                uncertainty: agent.uncertainty.de(&child)?,
            },
        })
    }
}

#[derive(Debug, Deserialize)]
struct StrategyConfig<V>
where
    V: MyFloat,
    Open01: Distribution<V>,
{
    informing: MyPath<InformingParams<V>>,
    information: InformationConfig<V>,
}

struct DeStrategyConfig<V> {
    informing: InformingParams<V>,
    information: InformationSamples<V>,
}

impl<V> DeserializeAt<DeStrategyConfig<V>> for MyPath<StrategyConfig<V>>
where
    V: MyFloat + for<'a> Deserialize<'a>,
    Open01: Distribution<V>,
{
    fn de<P: AsRef<Path>>(self, parent: P) -> anyhow::Result<DeStrategyConfig<V>> {
        let child = self.verified_child(&parent)?;
        let strategy = self.parse_at(&parent, DataFormat::de)?;

        Ok(DeStrategyConfig {
            informing: strategy.informing.parse_at(&child, DataFormat::de)?,
            information: strategy.information.de(&child)?,
        })
    }
}

#[derive(Debug, Deserialize)]
struct InformationConfig<V>
where
    V: MyFloat,
    Open01: Distribution<V>,
{
    /// also used for $M$ in correction
    misinfo: MyPath<Vec<OpinionRecord<V>>>,
    correction: MyPath<Vec<OpinionRecord<V>>>,
    observation: MyPath<Vec<OpinionRecord<V>>>,
    inhibition: MyPath<Vec<InhibitionRecord<V>>>,
}

impl<V> DeserializeAt<InformationSamples<V>> for InformationConfig<V>
where
    V: MyFloat + for<'a> Deserialize<'a>,
    Open01: Distribution<V>,
{
    fn de<P: AsRef<Path>>(self, at: P) -> anyhow::Result<InformationSamples<V>> {
        Ok(InformationSamples {
            misinfo: self
                .misinfo
                .parse_at(&at, read_csv)?
                .into_iter()
                .map(OpinionRecord::try_into)
                .try_collect()?,
            correction: self
                .correction
                .parse_at(&at, read_csv)?
                .into_iter()
                .map(OpinionRecord::try_into)
                .try_collect()?,
            observation: self
                .observation
                .parse_at(&at, read_csv)?
                .into_iter()
                .map(OpinionRecord::try_into)
                .try_collect()?,
            inhibition: self
                .inhibition
                .parse_at(&at, read_csv)?
                .into_iter()
                .map(InhibitionRecord::try_into)
                .try_collect()?,
        })
    }
}

#[derive(Debug, Deserialize)]
pub struct NetworkConfig<V> {
    path: PathBuf,
    graph: MyPath<GraphB>,
    directed: bool,
    transposed: bool,
    community: MyPath<Vec<SupportLevelRecord<V>>>,
}

struct DeNetworkConfig<V> {
    graph: GraphB,
    fnum_agents: V,
    mean_degree: V,
    community_psi1: SupportLevelTable<V>,
}

impl<V> DeserializeAt<DeNetworkConfig<V>> for MyPath<NetworkConfig<V>>
where
    V: MyFloat + for<'a> Deserialize<'a>,
{
    fn de<P: AsRef<Path>>(self, at: P) -> anyhow::Result<DeNetworkConfig<V>> {
        let child = self.verified_child(&at)?;
        let network = self.parse_at(&at, DataFormat::de)?;
        let child = child.join(network.path);

        let graph = network.graph.parse_at(&child, |p| {
            let builder = graph_lib::io::ParseBuilder::new(
                File::open(p)?,
                graph_lib::io::DataFormat::EdgeList,
            );
            if network.directed {
                if network.transposed {
                    Ok(GraphB::Di(builder.transpose().parse()?))
                } else {
                    Ok(GraphB::Di(builder.parse()?))
                }
            } else {
                Ok(GraphB::Ud(builder.parse()?))
            }
        })?;
        let fnum_agents = V::from_usize(graph.node_count()).unwrap();
        let mean_degree = V::from_usize(graph.edge_count()).unwrap() / fnum_agents;
        let records = network.community.parse_at(&child, read_csv)?;
        let community =
            SupportLevelTable::from_vec(records.into_iter().map(|r| r.level).collect_vec());
        Ok(DeNetworkConfig {
            graph,
            fnum_agents,
            mean_degree,
            community_psi1: community,
        })
    }
}

#[derive(Debug, serde::Deserialize)]
pub struct SupportLevelRecord<V> {
    level: V,
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
enum ConditionOption<V> {
    Import(MyPath<Vec<ConditionRecord<V>>>),
    Generate(Vec<Vec<SimplexOption<V>>>),
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
enum SimplexOption<V> {
    B(usize, SamplerOption<V>),
    U(SamplerOption<V>),
}

impl<V, D0, D1> DeserializeAt<ConditionSampler<D0, D1, V>> for ConditionOption<V>
where
    V: MyFloat + for<'a> Deserialize<'a>,
    Open01: Distribution<V>,
    D0: Domain<Idx: Debug> + Keys<D0::Idx>,
    D1: Domain<Idx: Debug + From<usize> + Copy> + Keys<D1::Idx>,
{
    fn de<P>(self, at: P) -> anyhow::Result<ConditionSampler<D0, D1, V>>
    where
        P: AsRef<Path>,
    {
        match self {
            ConditionOption::Import(c) => Ok(ConditionSampler::Array(
                c.parse_at(at, read_csv)?
                    .into_iter()
                    .map(ConditionRecord::try_into)
                    .try_collect()?,
            )),
            ConditionOption::Generate(pss) => {
                ensure!(D0::LEN == pss.len(), "few conditional opinion(s)");
                let containers = pss.into_iter().map(|ps| {
                    let mut b_check = MArrD1::<D1, bool>::from_fn(|_| false);
                    let mut u_check = false;
                    let mut sampler = None;
                    let mut fixed = Vec::new();
                    for p in ps {
                        match p {
                            SimplexOption::B(i, so) => {
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
                            SimplexOption::U(so) => {
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

                Ok(ConditionSampler::Random(MArrD1::from_iter(containers)))
            }
        }
    }
}

#[derive(Debug, Deserialize)]
struct ConditionConfig<V> {
    h_psi_if_phi0: ConditionOption<V>,
    h_b_if_phi0: ConditionOption<V>,
    o_b: ConditionOption<V>,
    a_fh: ConditionOption<V>,
    b_kh: ConditionOption<V>,
    theta_h: ConditionOption<V>,
    thetad_h: ConditionOption<V>,
}

impl<V> DeserializeAt<ConditionSamples<V>> for MyPath<ConditionConfig<V>>
where
    V: MyFloat + for<'a> Deserialize<'a>,
    Open01: Distribution<V>,
{
    fn de<P: AsRef<Path>>(self, parent: P) -> anyhow::Result<ConditionSamples<V>> {
        let child = self.verified_child(&parent)?;
        let ConditionConfig {
            h_psi_if_phi0,
            h_b_if_phi0,
            o_b,
            a_fh,
            b_kh,
            theta_h,
            thetad_h,
        } = self.parse_at(&parent, DataFormat::de)?;

        Ok(ConditionSamples {
            h_psi_if_phi0: h_psi_if_phi0.de(&child).context("h_psi_if_phi0")?,
            h_b_if_phi0: h_b_if_phi0.de(&child).context("h_b_if_phi0")?,
            o_b: o_b.de(&child).context("o_b")?,
            a_fh: a_fh.de(&child).context("a_fh")?,
            b_kh: b_kh.de(&child).context("b_kh")?,
            theta_h: theta_h.de(&child).context("theta_h")?,
            thetad_h: thetad_h.de(&child).context("thetad_h")?,
        })
    }
}

#[derive(Debug, Deserialize)]
struct UncertaintyConfig<V> {
    fh_fpsi_if_fphi0: MyPath<Vec<UncertaintyD1Record<V>>>,
    kh_kpsi_if_kphi0: MyPath<Vec<UncertaintyD1Record<V>>>,
    fh_fphi_fo: MyPath<Vec<UncertaintyD2Record<V>>>,
    kh_kphi_ko: MyPath<Vec<UncertaintyD2Record<V>>>,
}

impl<V> DeserializeAt<UncertaintySamples<V>> for MyPath<UncertaintyConfig<V>>
where
    V: MyFloat + for<'a> Deserialize<'a>,
{
    fn de<P: AsRef<Path>>(self, parent: P) -> anyhow::Result<UncertaintySamples<V>> {
        let child = self.verified_child(&parent)?;
        let UncertaintyConfig {
            fh_fpsi_if_fphi0,
            kh_kpsi_if_kphi0,
            fh_fphi_fo,
            kh_kphi_ko,
        } = self.parse_at(&parent, DataFormat::de)?;

        Ok(UncertaintySamples {
            fh_fpsi_if_fphi0: fh_fpsi_if_fphi0
                .parse_at(&child, read_csv)?
                .into_iter()
                .map(UncertaintyD1Record::try_into)
                .try_collect()?,
            kh_kpsi_if_kphi0: kh_kpsi_if_kphi0
                .parse_at(&child, read_csv)?
                .into_iter()
                .map(UncertaintyD1Record::try_into)
                .try_collect()?,
            fh_fphi_fo: fh_fphi_fo
                .parse_at(&child, read_csv)?
                .into_iter()
                .map(UncertaintyD2Record::try_into)
                .try_collect()?,
            kh_kphi_ko: kh_kphi_ko
                .parse_at(&child, read_csv)?
                .into_iter()
                .map(UncertaintyD2Record::try_into)
                .try_collect()?,
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

    use super::{ConditionOption, ConditionSampler};
    use crate::{
        config::{
            AgentConfig, DeAgentConfig, DeNetworkConfig, DeStrategyConfig, DeserializeAt,
            NetworkConfig, StrategyConfig,
        },
        io::MyPath,
        parameter::Sampler,
    };

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
    fn test_agent_config() -> anyhow::Result<()> {
        let config: MyPath<AgentConfig<f32>> = "./test/agent_config.toml".into();
        let agent: DeAgentConfig<f32> = config.de("")?;

        let mut rng = SmallRng::seed_from_u64(0);

        assert!(
            matches!(&agent.opinion.condition.o_b, ConditionSampler::Array(arr) if arr.len() == 7)
        );
        assert!(matches!(
            &agent.opinion.condition.o_b, ConditionSampler::Array(arr) if arr[3] == marr_d1![
            SimplexD1::new(marr_d1![1.0, 0.00], 0.00),
            SimplexD1::new(marr_d1![0.0, 0.75], 0.25)
            ]
        ));
        assert_eq!(
            agent.opinion.uncertainty.fh_fpsi_if_fphi0[0],
            marr_d1![0.1, 0.1]
        );
        assert_eq!(
            agent.opinion.uncertainty.kh_kphi_ko[0],
            marr_d2![[0.1, 0.1], [0.1, 0.1]]
        );

        assert!(matches!(
            agent.probabilities.plural_ignore_social,
            Sampler::Beta(_)
        ));
        let mut p = 0.0;
        for _ in 0..10 {
            p += agent.probabilities.plural_ignore_social.choose(&mut rng);
        }
        p /= 10.0;
        assert!((p - 0.1).abs() < 0.05);

        for _ in 0..10 {
            assert!(agent.sharer_trust.misinfo.choose(&mut rng) < 0.5);
        }
        Ok(())
    }

    #[test]
    fn test_network_config() -> anyhow::Result<()> {
        let config: MyPath<NetworkConfig<f32>> = "./test/network_config.toml".into();
        let network: DeNetworkConfig<f32> = config.de("")?;
        assert_eq!(network.community_psi1.levels.len(), 100);
        assert!(
            network.community_psi1.levels[network.community_psi1.indexes_by_level[10]]
                > network.community_psi1.levels[network.community_psi1.indexes_by_level[90]]
        );
        Ok(())
    }

    #[test]
    fn test_strategy_config() -> anyhow::Result<()> {
        let config: MyPath<StrategyConfig<f32>> = "./test/strategy_config.toml".into();
        let strategy: DeStrategyConfig<f32> = config.de("")?;
        assert_eq!(
            strategy.information.inhibition[0].0,
            OpinionD1::new(marr_d1![0.0, 1.0], 0.0, marr_d1![0.05, 0.95])
        );
        assert_eq!(
            strategy.information.inhibition[0].1,
            marr_d1![
                SimplexD1::new(marr_d1![0.5, 0.0], 0.5),
                SimplexD1::new(marr_d1![0.1, 0.7], 0.2),
            ]
        );
        assert_eq!(
            strategy.information.inhibition[0].2,
            marr_d1![
                SimplexD1::new(marr_d1![0.5, 0.0], 0.5),
                SimplexD1::new(marr_d1![0.1, 0.6], 0.3),
            ]
        );
        Ok(())
    }

    #[derive(Debug, Deserialize)]
    struct TestCondParam<V> {
        hoge: ConditionOption<V>,
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
        let r = DeserializeAt::<ConditionSampler<Phi, H, _>>::de(param.hoge, "");
        assert!(r.is_err());
        println!("{:#?}", r.err());

        let param: TestCondParam<f32> = toml! {
            hoge = { generate = [
                [{b = [0,{single = 0.95}]},{u = {single = 0.95}},{u = {single = 0.05}}],
                [{b = [0,{single = 0.95}]}, {u = {single = 0.05}}],
            ] }
        }
        .try_into()?;
        let r = DeserializeAt::<ConditionSampler<Phi, H, _>>::de(param.hoge, "");
        assert!(r.is_err());
        println!("{:#?}", r.err());

        let param: TestCondParam<f32> = toml! {
            hoge = { generate = [
                [{b = [0,{single = 0.95}]},{b = [1,{single = 0.95}]},{u = {single = 0.05}}],
                [{b = [0,{single = 0.95}]}, {u = {single = 0.05}}],
            ] }
        }
        .try_into()?;
        let r = DeserializeAt::<ConditionSampler<Phi, H, _>>::de(param.hoge, "");
        assert!(r.is_err());
        println!("{:#?}", r.err());

        let param: TestCondParam<f32> = toml! {
            hoge = { generate = [
                [{b = [0,{single = 0.95}]}, {u = {single = 0.05}}],
                [{b = [0,{array = [0.0,0.1]}]}, {b = [1,{uniform = [0.80,0.90]}]}],
            ] }
        }
        .try_into()?;
        let r = DeserializeAt::<ConditionSampler<Phi, H, _>>::de(param.hoge, "");
        assert!(r.is_err());
        println!("{:#?}", r.err());

        let param: TestCondParam<f32> = toml! {
            hoge = { generate = [
                [{u = {single = 0.05}},{b = [0,{single = 0.95}]}],
                [{b = [1,{single = 0.85}]},{b = [0,{single = 0.0}]}],
            ] }
        }
        .try_into()?;
        let s: ConditionSampler<Phi, H, f32> = param.hoge.de("")?;
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
        let s: ConditionSampler<Phi, H, f32> = param.hoge.de("")?;
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
