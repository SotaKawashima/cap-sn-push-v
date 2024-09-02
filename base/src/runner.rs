use std::io::{stdout, Write};
use std::sync::Arc;

use futures::future::try_join_all;
use rand::rngs::SmallRng;
use rand_distr::uniform::SampleUniform;
use tokio::sync::{mpsc, Mutex};

use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Exp1, Open01, Standard, StandardNormal};

use crate::executor::AgentExtTrait;
use crate::stat::FileWriters;
use crate::{
    executor::{Executor, InstanceExt, Memory},
    opinion::MyFloat,
    stat::Stat,
};

#[derive(Debug, serde::Deserialize)]
pub struct RuntimeParams {
    pub seed_state: u64,
    pub iteration_count: u32,
}

pub async fn run<V, E, Ax, Ix>(
    mut writers: FileWriters,
    runtime: &RuntimeParams,
    exec: E,
    max_permits: Option<usize>,
) -> anyhow::Result<()>
where
    V: MyFloat + SampleUniform + 'static,
    V::Sampler: Sync + Send,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    E: Executor<V, Ax, Ix> + Send + Sync + 'static,
    Ax: AgentExtTrait<V, Exec = E, Ix = Ix> + Default + Send + 'static,
    Ix: InstanceExt<V, SmallRng, E> + Send + 'static,
{
    println!("initialising...");

    let permits = max_permits.unwrap_or(num_cpus::get());
    let (tx, mut rx) = mpsc::channel::<Stat>(permits);
    let handle = tokio::spawn(async move {
        while let Some(stat) = rx.recv().await {
            writers.write(stat).unwrap();
        }
        writers.finish().unwrap();
    });

    let mut rng = SmallRng::seed_from_u64(runtime.seed_state);
    let rngs: Vec<SmallRng> = (0..(runtime.iteration_count))
        .map(|_| SmallRng::from_rng(&mut rng))
        .collect::<Result<Vec<_>, _>>()?;

    let exec = Arc::new(exec);
    let mut manager = Manager::new(permits, |id| Memory::new(exec.as_ref(), id));

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

pub struct Manager<E> {
    pub rx: mpsc::Receiver<usize>,
    pub tx: mpsc::Sender<usize>,
    pub resources: Vec<Arc<Mutex<E>>>,
}

impl<E> Manager<E> {
    pub fn new<F: Fn(usize) -> E>(permits: usize, f: F) -> Self {
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

pub struct EnvPermit<E> {
    idx: usize,
    tx: mpsc::Sender<usize>,
    env: Arc<Mutex<E>>,
}

impl<V, Ax> EnvPermit<Memory<V, Ax>>
where
    V: MyFloat + 'static,
{
    async fn run<Ex, Ix, R: Rng + Send + 'static>(
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
        Ex: Executor<V, Ax, Ix> + Send + Sync + 'static,
        Ax: AgentExtTrait<V, Exec = Ex, Ix = Ix> + Default + Send + 'static,
        Ix: InstanceExt<V, R, Ex> + Send,
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

#[cfg(test)]
mod tests {
    use std::fs::read_to_string;

    use super::RuntimeParams;
    use serde_json::json;

    #[test]
    fn test_json_config() -> anyhow::Result<()> {
        let runtime = json!({
            "seed_state": 0,
            "num_parallel": 1,
            "iteration_count": 1,
        });
        let runtime = serde_json::from_value::<RuntimeParams>(runtime)?;
        println!("{:?}", runtime);
        Ok(())
    }

    #[test]
    fn test_toml_config() -> anyhow::Result<()> {
        let runtime = toml::from_str::<RuntimeParams>(&read_to_string("./test_runtime.toml")?)?;
        assert_eq!(runtime.seed_state, 0);
        assert_eq!(runtime.iteration_count, 1);

        Ok(())
    }
}
