mod agent;
pub mod config;
mod decision;
mod dist;
mod executor;
mod info;
mod opinion;
mod scenario;
mod stat;
mod value;

use std::{
    io::{stdout, Write},
    path::PathBuf,
    sync::Arc,
};

use executor::{
    core::{Executor, InstanceExt, Memory},
    ExecV1,
};
use futures::future::try_join_all;
use opinion::MyFloat;
use polars_arrow::datatypes::Metadata;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use rand_distr::{Distribution, Exp1, Open01, Standard, StandardNormal};

use agent::AgentParams;
use config::{ConfigData, Runtime};
use scenario::{Scenario, ScenarioParam};
use stat::{FileWriters, Stat};
use tokio::sync::{mpsc, Mutex};

pub struct Runner<V>
where
    V: MyFloat,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
{
    runtime: ConfigData<Runtime>,
    agent_params: ConfigData<AgentParams<V>>,
    scenario: ConfigData<Scenario<V>>,
    identifier: String,
    output_dir: PathBuf,
    overwriting: bool,
    compressing: bool,
    num_cpus: usize,
}

impl<V> Runner<V>
where
    V: MyFloat,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
{
    pub fn try_new(
        runtime_path: String,
        agent_params_path: String,
        scenario_path: String,
        identifier: String,
        output_dir: PathBuf,
        overwriting: bool,
        compressing: bool,
    ) -> anyhow::Result<Self>
    where
        for<'de> V: serde::Deserialize<'de>,
    {
        Ok(Self {
            runtime: ConfigData::try_new::<Runtime>(runtime_path)?,
            agent_params: ConfigData::try_new::<AgentParams<V>>(agent_params_path)?,
            scenario: ConfigData::try_new::<ScenarioParam<V>>(scenario_path)?,
            identifier,
            output_dir,
            overwriting,
            compressing,
            num_cpus: num_cpus::get(),
        })
    }

    fn create_metadata(&self) -> Metadata {
        Metadata::from_iter([
            ("version".to_string(), env!("CARGO_PKG_VERSION").to_string()),
            ("runtime".to_string(), self.runtime.path.to_string()),
            (
                "agent_params".to_string(),
                self.agent_params.path.to_string(),
            ),
            ("scenario".to_string(), self.scenario.path.to_string()),
            (
                "iteration_count".to_string(),
                self.runtime.data.iteration_count.to_string(),
            ),
        ])
    }

    pub async fn run(self) -> anyhow::Result<()>
    where
        V: 'static,
        V::Sampler: Sync,
    {
        println!("initialising...");

        let (tx, mut rx) = mpsc::channel::<Stat>(self.num_cpus);
        let mut writers = FileWriters::try_new(
            &self.identifier,
            &self.output_dir,
            self.overwriting,
            self.compressing,
            self.create_metadata(),
        )?;

        let handle = tokio::spawn(async move {
            while let Some(stat) = rx.recv().await {
                writers.write(stat).unwrap();
            }
            writers.finish().unwrap();
        });

        let mut rng = SmallRng::seed_from_u64(self.runtime.data.seed_state);
        let rngs = (0..(self.runtime.data.iteration_count))
            .map(|_| SmallRng::from_rng(&mut rng))
            .collect::<Result<Vec<_>, _>>()?;

        let agent_params = Arc::new(self.agent_params.data);
        let scenario = Arc::new(self.scenario.data);
        let num_nodes = scenario.num_nodes;
        let exec = Arc::new(ExecV1::new(agent_params, scenario));
        let mut manager = Manager::new(self.num_cpus, |_| {
            // Env::new(agent_params.clone(), scenario.clone())
            Memory::new(num_nodes)
        });

        let mut jhs = Vec::new();
        print!("started.");
        for (num_iter, rng) in rngs.into_iter().enumerate() {
            let permit = manager.rent().await;
            let tx = tx.clone();
            jhs.push(tokio::spawn(permit.run(exec.clone(), num_iter, rng, tx)));
        }

        try_join_all(jhs).await?;
        drop(tx);
        handle.await.unwrap();
        println!("\ndone.");
        Ok(())
    }
}

struct Manager<E> {
    rx: mpsc::Receiver<usize>,
    tx: mpsc::Sender<usize>,
    resources: Vec<Arc<Mutex<E>>>,
}

impl<E> Manager<E> {
    fn new<F: Fn(usize) -> E>(permits: usize, f: F) -> Self {
        let mut resources = Vec::new();
        let (tx, rx) = mpsc::channel(permits);
        for i in 0..permits {
            let r = Arc::new(Mutex::new(f(i)));
            resources.push(r);
            tx.try_send(i).unwrap();
        }
        Self { rx, tx, resources }
    }

    async fn rent(&mut self) -> EnvPermit<E> {
        let idx = self.rx.recv().await.unwrap();
        EnvPermit {
            idx,
            env: self.resources[idx].clone(),
            tx: self.tx.clone(),
        }
    }
}

/*
struct Env<V>
where
    V: MyFloat,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
{
    agent_params: Arc<AgentParams<V>>,
    scenario: Arc<Scenario<V>>,
    agents: Vec<Agent<V>>,
}

impl<V> Env<V>
where
    V: MyFloat,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
{
    fn new(agent_params: Arc<AgentParams<V>>, scenario: Arc<Scenario<V>>) -> Self
    where
        V: Default,
    {
        let agents = (0..scenario.num_nodes)
            .map(|_| Agent::default())
            .collect::<Vec<_>>();
        Self {
            agent_params,
            scenario,
            agents,
        }
    }

    fn f0<R: Rng>(agent_params: &AgentParams<V>, rng: &mut R) -> (Trusts<V>, AccessProb<V>) {
        (
            Trusts {
                p: V::one(),
                fp: V::one(),
                kp: V::one(),
                fm: V::zero(),
                km: V::zero(),
            },
            AccessProb {
                fp: V::zero(),
                kp: V::zero(),
                pred_fp: agent_params.trust_params.friend_access_prob.sample(rng)
                    * agent_params.trust_params.friend_arrival_prob.sample(rng),
                fm: V::one(),
                km: V::one(),
            },
        )
    }

    fn f1<R: Rng>(
        scenario: &Scenario<V>,
        params: &TrustParams<V>,
        info: &Info<V>,
        rng: &mut R,
        info_trust_map: &mut BTreeMap<usize, V>,
        corr_misinfo_trust_map: &mut BTreeMap<usize, V>,
    ) -> (Trusts<V>, AccessProb<V>) {
        let friend_access_prob = params.friend_access_prob.sample(rng);
        let social_access_prob = params.social_access_prob.sample(rng);
        let friend_arrival_prob = params.friend_arrival_prob.sample(rng);
        let misinfo_friend = params.friend_misinfo_trust.sample(rng);
        let misinfo_social = params.social_misinfo_trust.sample(rng);
        let trust_sampler = params.info_trust_params.get_sampler(info.label());
        let info_trust = *info_trust_map
            .entry(info.idx)
            .or_insert_with(|| trust_sampler.sample(rng));
        let corr_misinfo_trust = *corr_misinfo_trust_map.entry(info.idx).or_insert_with(|| {
            params
                .info_trust_params
                .get_sampler(&InfoLabel::Misinfo)
                .sample(rng)
        });

        let receipt_prob = V::one()
            - (V::one() - V::from_usize(info.num_shared()).unwrap() / scenario.fnum_nodes)
                .powf(scenario.mean_degree);
        (
            Trusts {
                p: info_trust,
                fp: info_trust,
                kp: info_trust,
                fm: corr_misinfo_trust,
                km: corr_misinfo_trust,
            },
            AccessProb {
                fp: friend_access_prob * receipt_prob,
                kp: social_access_prob * receipt_prob,
                fm: misinfo_friend,
                km: misinfo_social,
                pred_fp: friend_access_prob * friend_arrival_prob,
            },
        )
    }

    fn execute<R: Rng>(&mut self, num_iter: u32, mut rng: R) -> Vec<Stat> {
        let span = span!(Level::INFO, "iter", i = num_iter);
        let _guard = span.enter();
        for (idx, agent) in self.agents.iter_mut().enumerate() {
            let span = span!(Level::INFO, "init", "#" = idx);
            let _guard = span.enter();
            agent.reset(&self.agent_params, &mut rng);
        }

        let info_objects = &self.scenario.info_objects;
        let mut event_table = self.scenario.table.clone();

        let mut infos = Vec::<Info<V>>::new();
        let mut info_data_table = BTreeMap::<InfoLabel, InfoData>::new();
        let mut agents_willing_selfish = Vec::<usize>::new();
        let mut info_stat = InfoStat::default();
        let mut agent_stat = AgentStat::default();
        let mut pop_stat = PopStat::default();
        let mut t = 0;

        // let mut receivers = Vec::new();
        // let mut senders = Vec::new();

        let mut rps = Vec::new();
        let mut info_trust_map = BTreeMap::<usize, V>::new();
        let mut corr_misinfo_trust_map = BTreeMap::<usize, V>::new();
        let mut total_num_selfish = 0;
        let mut observable = Vec::from_iter(0..self.agents.len());

        while !rps.is_empty() || !event_table.is_empty() || !agents_willing_selfish.is_empty() {
            let span = span!(Level::INFO, "step", t);
            let _guard = span.enter();

            let mut ips = Vec::new();

            macro_rules! new_info {
                ($agent_idx:expr,$info_obj_idx:expr) => {
                    let info_idx = infos.len();
                    let info = Info::new(info_idx, &info_objects[$info_obj_idx]);
                    info_data_table.entry(*info.label()).or_default();
                    infos.push(info);
                    ips.push(($agent_idx, info_idx));
                };
            }

            // register observer agents
            // senders of observed info have priority over existing senders.
            if let Some(observer) = &self.scenario.observer {
                let mut temp = Vec::new();
                observable.retain(|&agent_idx| {
                    if total_num_selfish <= observer.threshold {
                        return true;
                    }
                    if observer.po <= rng.gen() {
                        return true;
                    }
                    if observer.pp <= rng.gen() {
                        return true;
                    }
                    temp.push(agent_idx);
                    false
                });
                if temp.len() > 1 {
                    temp.shuffle(&mut rng);
                }
                for agent_idx in temp {
                    new_info!(agent_idx, observer.observed_info_obj_idx);
                }
            }

            if let Some(informms) = event_table.remove(&t) {
                for i in informms {
                    new_info!(i.agent_idx, i.info_obj_idx);
                }
            }

            let next_rps = step(
                &mut self.agents,
                &mut infos,
                &mut info_data_table,
                ips,
                rps,
                &mut agents_willing_selfish,
                |rng, _, _| Self::f0(&self.agent_params, rng),
                |rng, _, info| {
                    Self::f1(
                        &self.scenario,
                        &self.agent_params.trust_params,
                        info,
                        rng,
                        &mut info_trust_map,
                        &mut corr_misinfo_trust_map,
                    )
                },
                |a| a.access_prob(),
                &mut rng,
                &self.scenario.graph,
            );

            // update selfish states of agents
            let mut pop_data = PopData::default();
            agents_willing_selfish.retain(|&agent_idx| {
                let agent = &mut self.agents[agent_idx];
                let span = span!(Level::INFO, "Ag", "#" = agent_idx);
                let _guard = span.enter();
                if agent.progress_selfish_status() {
                    agent_stat.push_selfish(num_iter, t, agent_idx);
                    pop_data.selfish();
                    total_num_selfish += 1;
                }
                agent.is_willing_selfish()
            });
            pop_stat.push(num_iter, t, pop_data);

            for (info_label, d) in &mut info_data_table {
                info_stat.push(num_iter, t, d, info_label);
                *d = InfoData::default();
            }
            t += 1;
            rps = next_rps;
        }
        vec![info_stat.into(), agent_stat.into(), pop_stat.into()]
    }
}

fn step<V, F0, F1, H, R>(
    agents: &mut [Agent<V>],
    infos: &mut [Info<V>],
    info_data_table: &mut BTreeMap<InfoLabel, InfoData>,
    ips: Vec<(usize, usize)>,
    rps: Vec<(usize, usize)>,
    selfishes: &mut Vec<usize>,
    mut f0: F0,
    mut f1: F1,
    q: H,
    rng: &mut R,
    graph: &GraphB,
) -> Vec<(usize, usize)>
where
    V: MyFloat,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    F0: FnMut(&mut R, &Agent<V>, &Info<V>) -> (Trusts<V>, AccessProb<V>),
    F1: FnMut(&mut R, &Agent<V>, &Info<V>) -> (Trusts<V>, AccessProb<V>),
    H: Fn(&Agent<V>) -> V,
    R: Rng,
{
    let mut s = Vec::new();
    for &(agent_idx, info_idx) in &ips {
        let span = span!(Level::INFO, "IA", "#" = agent_idx);
        let _guard = span.enter();
        let info = &infos[info_idx];
        debug!(target: "recv", l = ?info.label(), "#" = info.idx);

        let agent = &mut agents[agent_idx];
        let (trusts, ap) = f0(rng, &agent, &info);
        agent.set_info_opinions(&info, trusts, ap);
        if agent.is_willing_selfish() {
            selfishes.push(agent_idx);
        }
    }

    for (agent_idx, info_idx) in rps {
        let span = span!(Level::INFO, "SA", "#" = agent_idx);
        let _guard = span.enter();

        let info = &mut infos[info_idx];
        let d = info_data_table.get_mut(info.label()).unwrap();
        d.received();
        debug!(target: "recv", l = ?info.label(), "#" = info.idx);

        let agent = &mut agents[agent_idx];

        if rng.gen::<V>() >= q(agent) {
            continue;
        }

        let (trusts, ap) = f1(rng, &agent, &info);
        let b = agent.read_info(info, trusts, ap);
        info.viewed();
        d.viewed();
        if b.sharing {
            info.shared();
            d.shared();
            s.push((agent_idx, info_idx));
        }
        if b.first_access {
            d.first_viewed();
        }
        if agent.is_willing_selfish() {
            selfishes.push(agent_idx);
        }
    }

    let mut r = ips
        .into_iter()
        .chain(s.into_iter())
        .flat_map(|(agent_idx, info_idx)| {
            graph.successors(agent_idx).map(move |bid| (*bid, info_idx))
        })
        .collect_vec();
    r.shuffle(rng);
    r
}
*/

struct EnvPermit<E> {
    idx: usize,
    tx: mpsc::Sender<usize>,
    env: Arc<Mutex<E>>,
}

impl<V> EnvPermit<Memory<V>>
where
    V: MyFloat + 'static,
{
    async fn run<Ex, X, R: Rng + Send + 'static>(
        self,
        exec: Arc<Ex>,
        num_iter: usize,
        rng: R,
        tx: mpsc::Sender<Stat>,
    ) -> anyhow::Result<()>
    where
        Open01: Distribution<V>,
        Standard: Distribution<V>,
        StandardNormal: Distribution<V>,
        Exp1: Distribution<V>,
        Ex: Executor<V, X> + Send + Sync + 'static,
        X: InstanceExt<V, R, Ex>,
        X: Send,
    {
        let env = self.env.clone();
        let handle = tokio::spawn(async move {
            if num_iter % 100 == 0 {
                println!("\n{num_iter}");
            }
            if num_iter % 10 == 0 {
                print!("|");
                stdout().flush().unwrap();
            }
            print!(".");
            let mut memory = env.lock().await;
            exec.execute::<R>(&mut memory, num_iter as u32, rng)
        });
        let ss = handle.await?;
        for s in ss {
            tx.send(s).await.unwrap();
        }
        Ok(())
    }
}

impl<E> Drop for EnvPermit<E> {
    fn drop(&mut self) {
        self.tx.try_send(self.idx).unwrap();
    }
}

// #[derive(Debug)]
// struct Receiver {
//     agent_idx: usize,
//     info_idx: usize,
// }

#[cfg(test)]
mod tests {
    use crate::Runner;
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

    async fn exec(
        runtime_path: &str,
        agent_params_path: &str,
        scenario_path: &str,
        identifier: &str,
    ) -> anyhow::Result<()> {
        let runner = Runner::<f32>::try_new(
            runtime_path.to_string(),
            agent_params_path.to_string(),
            scenario_path.to_string(),
            identifier.to_string(),
            "./test".into(),
            true,
            true,
        )?;
        runner.run().await?;
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
            assert_eq!(dfa[c], dfb[c]);
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
