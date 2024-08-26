mod parameters;
mod scenario;

use std::{collections::HashMap, path::PathBuf};

use base::{
    executor::{AgentExtTrait, AgentIdx, Executor, InfoIdx, InstanceExt, InstanceWrapper},
    info::{InfoContent, InfoLabel},
    opinion::{AccessProb, MyFloat, Trusts},
    runner::{run, RuntimeParams},
    stat::FileWriters,
};
use input::format::DataFormat;

use crate::scenario::{Inform, Scenario, ScenarioParam};

use parameters::AgentParams;
use polars_arrow::datatypes::Metadata;
use rand_distr::{Distribution, Exp1, Open01, Standard, StandardNormal};

use std::collections::{BTreeMap, VecDeque};

use graph_lib::prelude::GraphB;
use rand::{seq::SliceRandom, Rng};

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
    agent_params: String,
    /// the path of a scenario config file
    #[arg(short, long)]
    scenario: String,
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
        agent_params,
        scenario,
        overwriting,
        compressing,
    } = args;
    let runtime_data = DataFormat::read(&runtime)?.parse::<RuntimeParams>()?;
    let agent_params_data = DataFormat::read(&agent_params)?.parse::<AgentParams<V>>()?;
    let scenario_data: Scenario<V> = DataFormat::read(&scenario)?
        .parse::<ScenarioParam<V>>()?
        .try_into()?;
    let metadata = Metadata::from_iter([
        ("app".to_string(), env!("CARGO_PKG_NAME").to_string()),
        ("version".to_string(), env!("CARGO_PKG_VERSION").to_string()),
        ("runtime".to_string(), runtime),
        ("agent_params".to_string(), agent_params),
        ("scenario".to_string(), scenario),
        (
            "iteration_count".to_string(),
            runtime_data.iteration_count.to_string(),
        ),
    ]);
    let writers =
        FileWriters::try_new(&identifier, &output_dir, overwriting, compressing, metadata)?;
    let exec = ExecV1::<V>::new(agent_params_data, scenario_data);
    run::<V, _, AgentExt<V>, InstanceV1<V>>(writers, &runtime_data, exec, None).await
}

struct ExecV1<V>
where
    V: MyFloat,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
{
    agent_params: AgentParams<V>,
    scenario: Scenario<V>,
}

impl<V> ExecV1<V>
where
    V: MyFloat,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
{
    fn new(agent_params: AgentParams<V>, scenario: Scenario<V>) -> Self {
        Self {
            agent_params,
            scenario,
        }
    }
}

#[derive(Default)]
struct AgentExt<V> {
    visit_prob: V,
}

impl<V> AgentExtTrait<V> for AgentExt<V>
where
    V: MyFloat,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
{
    type Exec = ExecV1<V>;
    type Ix = InstanceV1<V>;

    fn visit_prob<R: Rng>(&mut self, _: &ExecV1<V>, _: &mut R) -> V {
        self.visit_prob
    }

    fn reset_core<R: Rng>(
        ops: &mut base::opinion::MyOpinions<V>,
        decision: &mut base::agent::Decision<V>,
        exec: &Self::Exec,
        rng: &mut R,
    ) {
        exec.agent_params.initial_opinions.reset_to(ops, rng);
        let delay_selfish = exec.agent_params.delay_selfish.sample(rng);
        decision.reset(delay_selfish, |prospect, cpt| {
            exec.agent_params.loss_params.reset_to(prospect, rng);
            exec.agent_params.cpt_params.reset_to(cpt, rng);
        });
    }

    fn reset<R: Rng>(&mut self, _: usize, exec: &ExecV1<V>, rng: &mut R) {
        self.visit_prob = exec.agent_params.access_prob.sample(rng);
    }

    fn informer_trusts<'a, R>(
        &mut self,
        _: &mut InstanceWrapper<'a, Self::Exec, V, R, Self::Ix>,
        _: InfoIdx,
    ) -> Trusts<V> {
        Trusts {
            p: V::one(),
            fp: V::one(),
            kp: V::one(),
            fm: V::zero(),
            km: V::zero(),
        }
    }

    fn informer_access_probs<'a, R: Rng>(
        &mut self,
        ins: &mut InstanceWrapper<'a, Self::Exec, V, R, Self::Ix>,
        _: InfoIdx,
    ) -> AccessProb<V> {
        AccessProb {
            fp: V::zero(),
            kp: V::zero(),
            pred_fp: ins
                .exec
                .agent_params
                .trust_params
                .friend_access_prob
                .sample(&mut ins.rng)
                * ins
                    .exec
                    .agent_params
                    .trust_params
                    .friend_arrival_prob
                    .sample(&mut ins.rng),
            fm: V::one(),
            km: V::one(),
        }
    }

    fn sharer_trusts<'a, R: Rng>(
        &mut self,
        ins: &mut InstanceWrapper<'a, Self::Exec, V, R, Self::Ix>,
        info_idx: InfoIdx,
    ) -> Trusts<V> {
        let params = &ins.exec.agent_params.trust_params;
        let trust_sampler = params
            .info_trust_params
            .get_sampler(ins.get_info_label(info_idx));
        let info_trust = *ins
            .ext
            .info_trust_map
            .entry(info_idx)
            .or_insert_with(|| trust_sampler.sample(&mut ins.rng));
        let corr_misinfo_trust = *ins
            .ext
            .corr_misinfo_trust_map
            .entry(info_idx)
            .or_insert_with(|| {
                params
                    .info_trust_params
                    .get_sampler(&InfoLabel::Misinfo)
                    .sample(&mut ins.rng)
            });

        Trusts {
            p: info_trust,
            fp: info_trust,
            kp: info_trust,
            fm: corr_misinfo_trust,
            km: corr_misinfo_trust,
        }
    }

    fn sharer_access_probs<'a, R: Rng>(
        &mut self,
        ins: &mut InstanceWrapper<'a, Self::Exec, V, R, Self::Ix>,
        info_idx: InfoIdx,
    ) -> AccessProb<V> {
        let params = &ins.exec.agent_params.trust_params;
        let friend_access_prob = params.friend_access_prob.sample(&mut ins.rng);
        let social_access_prob = params.social_access_prob.sample(&mut ins.rng);
        let friend_arrival_prob = params.friend_arrival_prob.sample(&mut ins.rng);
        let misinfo_friend = params.friend_misinfo_trust.sample(&mut ins.rng);
        let misinfo_social = params.social_misinfo_trust.sample(&mut ins.rng);
        let receipt_prob = V::one()
            - (V::one()
                - V::from_usize(ins.num_shared(info_idx)).unwrap() / ins.exec.scenario.fnum_nodes)
                .powf(ins.exec.scenario.mean_degree);

        AccessProb {
            fp: friend_access_prob * receipt_prob,
            kp: social_access_prob * receipt_prob,
            fm: misinfo_friend,
            km: misinfo_social,
            pred_fp: friend_access_prob * friend_arrival_prob,
        }
    }
}

struct InstanceV1<V> {
    event_table: BTreeMap<u32, VecDeque<Inform>>,
    observable: Vec<usize>,
    info_trust_map: HashMap<InfoIdx, V>,
    corr_misinfo_trust_map: HashMap<InfoIdx, V>,
}

impl<'a, V> Executor<V, AgentExt<V>, InstanceV1<V>> for ExecV1<V>
where
    V: MyFloat,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
{
    fn num_agents(&self) -> usize {
        self.scenario.num_nodes
    }

    fn graph(&self) -> &GraphB {
        &self.scenario.graph
    }
    // fn reset<R: Rng>(&self, memory: &mut Memory<V, AgentExt<V>>, rng: &mut R) {
    //     for (idx, agent) in memory.agents.iter_mut().enumerate() {
    //         let span = span!(Level::INFO, "init", "#" = idx);
    //         let _guard = span.enter();
    //         agent.reset(&self.agent_params, rng);
    //     }
    // }
}

/*
// self.decision.reset(param, rng);
// self.access_prob = agent_params.access_prob.sample(rng);
// self.ops_gen2.reset(&agent_params.initial_opinions, rng);
// self.trust.reset(&agent_params.trust_params, rng);
// receipt_prob: V,
// trust_params: &TrustParams<V>,
// rng: &mut impl Rng,
// let trusts = self.trust.to_sharer(info, receipt_prob, trust_params, rng);
// let trusts = self.trust.to_inform();

#[derive(Default)]
struct Trust<V: Float> {
    friend_access_prob: V,
    social_access_prob: V,
    friend_arrival_prob: V,
    info_trust_map: BTreeMap<usize, V>,
    corr_misinfo_trust_map: BTreeMap<usize, V>,
    misinfo_friend: V,
    misinfo_social: V,
}

impl<V: Float> Trust<V> {
    fn reset<R: Rng>(&mut self, trust_params: &TrustParams<V>, rng: &mut R)
    where
        V: MyFloat,
        Open01: Distribution<V>,
        Standard: Distribution<V>,
        StandardNormal: Distribution<V>,
        Exp1: Distribution<V>,
    {
        self.friend_access_prob = trust_params.friend_access_prob.sample(rng);
        self.social_access_prob = trust_params.social_access_prob.sample(rng);
        self.friend_arrival_prob = trust_params.friend_arrival_prob.sample(rng);
        self.misinfo_friend = trust_params.friend_misinfo_trust.sample(rng);
        self.misinfo_social = trust_params.social_misinfo_trust.sample(rng);
        self.info_trust_map.clear();
        self.corr_misinfo_trust_map.clear();
    }

    fn to_inform(&self) -> Trusts<V>
    where
        V: MyFloat,
    {
        Trusts {
            p: V::one(),
            corr_misinfo: V::zero(),
            friend: V::zero(),
            social: V::zero(),
            pi_friend: V::one(),
            pi_social: V::one(),
            pred_friend: self.friend_access_prob * self.friend_arrival_prob,
        }
    }

    fn to_sharer<R: Rng>(
        &mut self,
        info: &Info<V>,
        receipt_prob: V,
        params: &TrustParams<V>,
        rng: &mut R,
    ) -> Trusts<V>
    where
        V: MyFloat,
        Open01: Distribution<V>,
        Standard: Distribution<V>,
        StandardNormal: Distribution<V>,
        Exp1: Distribution<V>,
    {
        let trust_sampler = params.info_trust_params.get_sampler(info.label());
        let info_trust = *self
            .info_trust_map
            .entry(info.idx)
            .or_insert_with(|| trust_sampler.sample(rng));
        let corr_misinfo_trust =
            *self
                .corr_misinfo_trust_map
                .entry(info.idx)
                .or_insert_with(|| {
                    params
                        .info_trust_params
                        .get_sampler(&InfoLabel::Misinfo)
                        .sample(rng)
                });

        Trusts {
            p: info_trust,
            corr_misinfo: corr_misinfo_trust,
            friend: self.friend_access_prob * receipt_prob,
            social: self.social_access_prob * receipt_prob,
            pi_friend: self.misinfo_friend,
            pi_social: self.misinfo_social,
            pred_friend: self.friend_access_prob * self.friend_arrival_prob,
        }
    }
}
*/

impl<V, R> InstanceExt<V, R, ExecV1<V>> for InstanceV1<V>
where
    V: MyFloat,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    R: Rng,
{
    fn from_exec(exec: &ExecV1<V>, _: &mut R) -> Self {
        Self {
            info_trust_map: HashMap::new(),
            corr_misinfo_trust_map: HashMap::new(),
            observable: Vec::from_iter(0..exec.scenario.num_nodes),
            event_table: exec.scenario.table.clone(),
        }
    }

    fn is_continued(&self, _: &ExecV1<V>) -> bool {
        !self.event_table.is_empty()
    }

    fn get_informers_with<'a>(
        ins: &mut InstanceWrapper<'a, ExecV1<V>, V, R, Self>,
        t: u32,
    ) -> Vec<(AgentIdx, InfoContent<'a, V>)> {
        let mut contents = Vec::new();
        // register observer agents
        // senders of observed info have priority over existing senders.
        if let Some(observer) = &ins.exec.scenario.observer {
            if ins.total_num_selfish() > observer.threshold {
                ins.ext.observable.retain(|&agent_idx| {
                    if observer.po <= ins.rng.gen() {
                        return true;
                    }
                    if observer.pp <= ins.rng.gen() {
                        return true;
                    }
                    contents.push((
                        agent_idx.into(),
                        (&ins.exec.scenario.info_objects[observer.observed_info_obj_idx]).into(),
                    ));
                    false
                });
            }
            if contents.len() > 1 {
                contents.shuffle(&mut ins.rng);
            }
        }

        if let Some(informms) = ins.ext.event_table.remove(&t) {
            for i in informms {
                contents.push((
                    i.agent_idx.into(),
                    (&ins.exec.scenario.info_objects[i.info_obj_idx]).into(),
                ));
            }
        }
        contents
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use polars::{
        frame::DataFrame,
        io::{ipc::IpcReader, SerReader},
        lazy::{
            dsl::col,
            frame::{LazyFrame, ScanArgsIpc},
        },
    };
    use polars_arrow::datatypes::Metadata;
    use rand::{rngs::SmallRng, Rng, SeedableRng};
    use std::{fs, sync::Arc};
    use tokio::runtime::Runtime;

    use crate::start;

    async fn exec(
        runtime_path: &str,
        agent_params_path: &str,
        scenario_path: &str,
        identifier: &str,
    ) -> anyhow::Result<()> {
        start::<f32>(crate::Cli {
            identifier: identifier.to_string(),
            output_dir: "./test".into(),
            runtime: runtime_path.to_string(),
            agent_params: agent_params_path.to_string(),
            scenario: scenario_path.to_string(),
            overwriting: true,
            compressing: true,
        })
        .await?;
        Ok(())
    }

    fn compare_two_ipcs(
        na: &str,
        rpa: &str,
        apa: &str,
        spa: &str,
        nb: &str,
        rpb: &str,
        apb: &str,
        spb: &str,
    ) -> anyhow::Result<()> {
        let labels = ["info", "agent", "pop"];
        for label in labels {
            let pa = format!("./test/{na}_{label}.arrow");
            let pb = format!("./test/{nb}_{label}.arrow");
            let mut reader_a = IpcReader::new(fs::File::open(pa.clone())?);
            let mut reader_b = IpcReader::new(fs::File::open(pb.clone())?);
            let metadata_a = &reader_a.schema()?.metadata;
            let metadata_b = &reader_b.schema()?.metadata;
            check_metadata(rpa, apa, spa, metadata_a)?;
            check_metadata(rpb, apb, spb, metadata_b)?;
            drop(reader_a);
            drop(reader_b);

            let lfa = LazyFrame::scan_ipc(
                pa,
                ScanArgsIpc {
                    memory_map: false,
                    ..Default::default()
                },
            )?;
            let lfb = LazyFrame::scan_ipc(
                pb,
                ScanArgsIpc {
                    memory_map: false,
                    ..Default::default()
                },
            )?;
            compare_arrows(lfa.collect()?, lfb.collect()?)?;
        }
        Ok(())
    }

    fn check_metadata(
        runtime_path: &str,
        agent_params_path: &str,
        scenario_path: &str,
        metadata: &Metadata,
    ) -> anyhow::Result<()> {
        assert_eq!(metadata["runtime"], runtime_path);
        assert_eq!(metadata["agent_params"], agent_params_path);
        assert_eq!(metadata["scenario"], scenario_path);
        Ok(())
    }

    fn compare_arrows(dfa: DataFrame, dfb: DataFrame) -> anyhow::Result<()> {
        let (ra, ca) = dfa.shape();
        let (rb, cb) = dfb.shape();
        assert_eq!(ra, rb);
        assert_eq!(ca, cb);

        let csa = dfa.get_column_names();
        let csb = dfb.get_column_names();
        assert_eq!(csa, csb);

        let dfa = dfa.sort(&csa, Default::default())?;
        let dfb = dfb.sort(&csb, Default::default())?;

        for &c in csa.iter().rev() {
            let sa = dfa[c].iter().collect_vec();
            let sb = dfb[c].iter().collect_vec();
            assert_eq!(sa, sb);
        }
        Ok(())
    }

    #[test]
    fn test_transposed() -> anyhow::Result<()> {
        let runtime_path = "./test/config/runtime.toml";
        let agent_params_path = "./test/config/agent_params.toml";
        let scenario_path = "./test/config/scenario.toml";
        let scenario_path_t = "./test/config/scenario-t.toml";

        let rt = Runtime::new()?;
        rt.block_on(async {
            exec(
                runtime_path,
                agent_params_path,
                scenario_path_t,
                "run_test-t",
            )
            .await
            .unwrap();
            exec(runtime_path, agent_params_path, scenario_path, "run_test")
                .await
                .unwrap();
        });

        compare_two_ipcs(
            "run_test",
            &runtime_path,
            &agent_params_path,
            &scenario_path,
            "run_test-t",
            &runtime_path,
            &agent_params_path,
            &scenario_path_t,
        )?;
        Ok(())
    }

    #[test]
    fn test_events() -> anyhow::Result<()> {
        let runtime_path = "./test/config/runtime.toml";
        let agent_params_path = "./test/config/agent_params.toml";
        let scenario_path0 = "./test/config/scenario-e0.toml";
        let scenario_path1 = "./test/config/scenario-e1.toml";

        let rt = Runtime::new()?;
        rt.block_on(async {
            exec(
                runtime_path,
                agent_params_path,
                scenario_path0,
                "run_test-e0",
            )
            .await
            .unwrap();
        });
        rt.block_on(async {
            exec(
                runtime_path,
                agent_params_path,
                scenario_path1,
                "run_test-e1",
            )
            .await
            .unwrap();
        });

        compare_two_ipcs(
            "run_test-e0",
            &runtime_path,
            &agent_params_path,
            &scenario_path0,
            "run_test-e1",
            &runtime_path,
            &agent_params_path,
            &scenario_path1,
        )?;
        Ok(())
    }

    #[test]
    fn test_infos() -> anyhow::Result<()> {
        let runtime_path = "./test/config/runtime.toml";
        let agent_params_path = "./test/config/agent_params.toml";
        let scenario_path0 = "./test/config/scenario-i.toml";

        let rt = Runtime::new()?;
        rt.block_on(async {
            exec(
                runtime_path,
                agent_params_path,
                scenario_path0,
                "run_test-sample",
            )
            .await
            .unwrap();
        });

        let lf = LazyFrame::scan_ipc(
            "./test/run_test-sample_info.arrow",
            ScanArgsIpc {
                memory_map: false,
                ..Default::default()
            },
        )?
        .filter(col("info_label").eq(3))
        .collect()?;
        assert!(lf.shape().0 > 0);
        Ok(())
    }

    struct TestEnv(usize);
    impl TestEnv {
        fn execute<R: Rng>(&mut self, _input: usize, mut rng: R) {
            for i in 0..10_000 {
                for j in 0..(10_000 + rng.gen_range(0..1000)) {
                    let _a = i + j;
                }
            }
        }
    }

    #[tokio::test]
    async fn test_manager() {
        println!("init");
        let mut resources = Vec::new();
        let n = 8;
        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        for i in 0..n {
            resources.push(Arc::new(tokio::sync::Mutex::new(TestEnv(i))));
            tx.try_send(i).unwrap();
        }

        let m = 8;
        let mut rng = SmallRng::seed_from_u64(0);
        let rngs = (0..m)
            .map(|_| SmallRng::from_rng(&mut rng))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        let mut hs = Vec::new();
        for (i, rng) in rngs.into_iter().enumerate() {
            let tx = tx.clone();
            let idx = rx.recv().await.unwrap();
            let e = resources[idx].clone();
            let h = tokio::spawn(async move {
                {
                    let mut e = e.lock().await;
                    println!("s:{}:{}", i, e.0);
                    e.execute(i, rng);
                    println!("e:{}:{}", i, e.0);
                }
                drop(e);
                tx.try_send(idx).unwrap();
            });
            hs.push(h);
        }
        println!("running");
        for h in hs {
            h.await.unwrap();
        }
        drop(rx);
        println!("done");
    }
}
