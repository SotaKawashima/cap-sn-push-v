pub mod opinion;

use std::fmt::Debug;

use base::{
    decision::{Prospect, CPT},
    info::{Info, InfoLabel},
};
use num_traits::Float;
use opinion::InitialOpinions;
use rand::Rng;
use rand_distr::{uniform::SampleUniform, Distribution, Exp1, Open01, Standard, StandardNormal};
use serde_with::{serde_as, TryFromInto};

use base::{
    agent::{Agent, Decision},
    opinion::MyFloat,
    util::Reset,
};
use input::{
    dist::{IValue, IValueParam},
    value::{EValue, EValueParam},
};

#[serde_as]
#[derive(Debug, serde::Deserialize)]
pub struct AgentParams<V>
where
    V: MyFloat,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
{
    pub initial_opinions: InitialOpinions<V>,
    #[serde_as(as = "TryFromInto<IValueParam<V>>")]
    pub delay_selfish: IValue<V>,
    loss_params: LossParams<V>,
    cpt_params: CptParams<V>,
    pub trust_params: TrustParams<V>,
    #[serde_as(as = "TryFromInto<EValueParam<V>>")]
    pub access_prob: EValue<V>,
}

impl<V> Reset<Agent<V>> for AgentParams<V>
where
    V: MyFloat,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
{
    fn reset<R: rand::Rng>(&self, value: &mut Agent<V>, rng: &mut R) {
        self.reset(&mut value.decision, rng);
        value.access_prob = self.access_prob.sample(rng);
        self.initial_opinions.reset(&mut value.ops_gen2, rng);
    }
}

impl<V> Reset<Decision<V>> for AgentParams<V>
where
    V: MyFloat,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
{
    fn reset<R: rand::Rng>(&self, value: &mut Decision<V>, rng: &mut R) {
        self.loss_params.reset(&mut value.prospect, rng);
        self.cpt_params.reset(&mut value.cpt, rng);
        value.delay_selfish = self.delay_selfish.sample(rng);
    }
}

#[serde_as]
#[derive(Debug, serde::Deserialize)]
pub struct TrustParams<V>
where
    V: Float,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
{
    #[serde_as(as = "TryFromInto<EValueParam<V>>")]
    pub friend_access_prob: EValue<V>,
    #[serde_as(as = "TryFromInto<EValueParam<V>>")]
    pub social_access_prob: EValue<V>,
    #[serde_as(as = "TryFromInto<EValueParam<V>>")]
    pub friend_arrival_prob: EValue<V>,
    pub info_trust_params: InfoTrustParams<V>,
    #[serde_as(as = "TryFromInto<EValueParam<V>>")]
    pub friend_misinfo_trust: EValue<V>,
    #[serde_as(as = "TryFromInto<EValueParam<V>>")]
    pub social_misinfo_trust: EValue<V>,
}

#[serde_as]
#[derive(Debug, serde::Deserialize)]
#[serde(bound(deserialize = "V: serde::Deserialize<'de>"))]
struct LossParams<V>
where
    V: Float,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
{
    #[serde_as(as = "TryFromInto<EValueParam<V>>")]
    pub x0: EValue<V>,
    #[serde_as(as = "TryFromInto<EValueParam<V>>")]
    pub x1_of_x0: EValue<V>,
    #[serde_as(as = "TryFromInto<EValueParam<V>>")]
    pub y_of_x0: EValue<V>,
}

impl<V> Reset<Prospect<V>> for LossParams<V>
where
    V: Float + Debug,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
{
    fn reset<R: Rng>(&self, value: &mut Prospect<V>, rng: &mut R) {
        let x0 = self.x0.sample(rng);
        let x1 = x0 * self.x1_of_x0.sample(rng);
        let y = x0 * self.y_of_x0.sample(rng);
        value.reset(x0, x1, y);
    }
}

#[serde_as]
#[derive(Debug, serde::Deserialize)]
#[serde(bound(deserialize = "V: serde::Deserialize<'de>"))]
struct CptParams<V>
where
    V: Float,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
{
    #[serde_as(as = "TryFromInto<EValueParam<V>>")]
    pub alpha: EValue<V>,
    #[serde_as(as = "TryFromInto<EValueParam<V>>")]
    pub beta: EValue<V>,
    #[serde_as(as = "TryFromInto<EValueParam<V>>")]
    pub lambda: EValue<V>,
    #[serde_as(as = "TryFromInto<EValueParam<V>>")]
    pub gamma: EValue<V>,
    #[serde_as(as = "TryFromInto<EValueParam<V>>")]
    pub delta: EValue<V>,
}

impl<V> Reset<CPT<V>> for CptParams<V>
where
    V: MyFloat,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
{
    fn reset<R: Rng>(&self, value: &mut CPT<V>, rng: &mut R) {
        value.reset(
            self.alpha.sample(rng),
            self.beta.sample(rng),
            self.lambda.sample(rng),
            self.gamma.sample(rng),
            self.delta.sample(rng),
        );
    }
}

#[serde_as]
#[derive(Debug, serde::Deserialize)]
#[serde(bound(deserialize = "V: serde::Deserialize<'de>"))]
pub struct InfoTrustParams<V>
where
    V: Float,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
{
    #[serde_as(as = "TryFromInto<EValueParam<V>>")]
    pub misinfo: EValue<V>,
    #[serde_as(as = "TryFromInto<EValueParam<V>>")]
    pub corrective: EValue<V>,
    #[serde_as(as = "TryFromInto<EValueParam<V>>")]
    pub observed: EValue<V>,
    #[serde_as(as = "TryFromInto<EValueParam<V>>")]
    pub inhibitive: EValue<V>,
}

#[allow(dead_code)]
impl<V> InfoTrustParams<V>
where
    V: MyFloat,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
{
    pub fn gen_map<R: Rng>(&self, rng: &mut R) -> impl Fn(&Info<V>) -> V
    where
        V: SampleUniform,
        Open01: Distribution<V>,
        Standard: Distribution<V>,
    {
        let misinfo = self.misinfo.sample(rng);
        let corrective = self.corrective.sample(rng);
        let observed = self.observed.sample(rng);
        let inhibitive = self.inhibitive.sample(rng);

        move |info: &Info<V>| match &info.label() {
            InfoLabel::Misinfo => misinfo,
            InfoLabel::Corrective => corrective,
            InfoLabel::Observed => observed,
            InfoLabel::Inhibitive => inhibitive,
        }
    }

    pub fn get_sampler(&self, info_label: &InfoLabel) -> &EValue<V> {
        match info_label {
            InfoLabel::Misinfo => &self.misinfo,
            InfoLabel::Corrective => &self.corrective,
            InfoLabel::Observed => &self.observed,
            InfoLabel::Inhibitive => &self.inhibitive,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs::read_to_string;

    use serde::Deserialize;

    use super::AgentParams;

    #[test]
    fn test_toml() -> anyhow::Result<()> {
        let initial_opinions = toml::from_str::<toml::Value>(&read_to_string(
            "./test/config/test_initial_opinions.toml",
        )?)?;
        let mut agent_params = toml::from_str::<toml::Value>(&read_to_string(
            "./test/config/test_agent_params.toml",
        )?)?;
        agent_params
            .as_table_mut()
            .unwrap()
            .insert("initial_opinions".to_string(), initial_opinions);

        AgentParams::<f32>::deserialize(agent_params)?;
        Ok(())
    }
}
