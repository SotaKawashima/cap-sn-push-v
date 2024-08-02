pub mod core;

use std::{
    collections::{BTreeMap, VecDeque},
    sync::Arc,
};

use graph_lib::prelude::GraphB;
use rand::seq::SliceRandom;
use rand::Rng;
use rand_distr::{Distribution, Exp1, Open01, Standard, StandardNormal};
use tracing::{span, Level};

use crate::{
    agent::{Agent, AgentParams},
    info::{gen2::InfoContent, InfoLabel},
    opinion::{
        gen2::{AccessProb, Trusts},
        MyFloat,
    },
    scenario::{Inform, Scenario},
};

use core::{Executor, InstanceExt, InstanceWrapper, Memory};

pub struct ExecV1<V>
where
    V: MyFloat,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
{
    agent_params: Arc<AgentParams<V>>,
    scenario: Arc<Scenario<V>>,
    num_agents: usize,
}

impl<V> ExecV1<V>
where
    V: MyFloat,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
{
    pub fn new(agent_params: Arc<AgentParams<V>>, scenario: Arc<Scenario<V>>) -> Self {
        Self {
            agent_params,
            num_agents: scenario.num_nodes,
            scenario,
        }
    }
}

pub struct InstanceV1<V: MyFloat> {
    event_table: BTreeMap<u32, VecDeque<Inform>>,
    observable: Vec<usize>,
    info_trust_map: BTreeMap<usize, V>,
    corr_misinfo_trust_map: BTreeMap<usize, V>,
}

// let info_objects = &self.scenario.info_objects;
// let mut event_table = self.scenario.table.clone();
impl<'a, V: MyFloat> Executor<V, InstanceV1<V>> for ExecV1<V>
where
    V: MyFloat,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
{
    fn graph(&self) -> &GraphB {
        &self.scenario.graph
    }

    fn reset<R: Rng>(&self, memory: &mut Memory<V>, rng: &mut R) {
        for (idx, agent) in memory.agents.iter_mut().enumerate() {
            let span = span!(Level::INFO, "init", "#" = idx);
            let _guard = span.enter();
            agent.reset(&self.agent_params, rng);
        }
    }

    fn instance_ext(&self) -> InstanceV1<V> {
        InstanceV1 {
            info_trust_map: BTreeMap::<usize, V>::new(),
            corr_misinfo_trust_map: BTreeMap::<usize, V>::new(),
            observable: Vec::from_iter(0..self.num_agents),
            event_table: self.scenario.table.clone(),
        }
    }
}

// impl<'a, V, R> AgentInfo<ExecV1<'a, V, R>, V, R, InstanceV1<'a, V, R>> for AgentInfoV1
// where
//     V: MyFloat,
//     Open01: Distribution<V>,
//     Standard: Distribution<V>,
//     StandardNormal: Distribution<V>,
//     Exp1: Distribution<V>,
//     R: Rng,
// {
//     fn get_informer<'b: 'a>(
//         exe: &mut ExecV1<'a, V, R>,
//         _: &mut InstanceV1<'a, V, R>,
//         agent_idx: usize,
//         info_idx: usize,
//     ) -> (&'b Info<'b, V>, &'b mut Agent<V>, Trusts<V>, AccessProb<V>) {
//         (
//             &exe.instance.core.infos[info_idx],
//             &mut exe.agents[agent_idx],
//             Trusts {
//                 p: V::one(),
//                 fp: V::one(),
//                 kp: V::one(),
//                 fm: V::zero(),
//                 km: V::zero(),
//             },
//             AccessProb {
//                 fp: V::zero(),
//                 kp: V::zero(),
//                 pred_fp: exe
//                     .agent_params
//                     .trust_params
//                     .friend_access_prob
//                     .sample(&mut exe.instance.core.rng)
//                     * exe
//                         .agent_params
//                         .trust_params
//                         .friend_arrival_prob
//                         .sample(&mut exe.instance.core.rng),
//                 fm: V::one(),
//                 km: V::one(),
//             },
//         )
//     }

//     fn get_sharer_params<'b>(
//         exe: &mut ExecV1<'a, V, R>,
//         _: &mut InstanceV1<'a, V, R>,
//         agent_idx: usize,
//         info_idx: usize,
//     ) -> (&'b mut Agent<V>, Trusts<V>, AccessProb<V>) {
//         todo!()
//     }
// }

impl<V, R> InstanceExt<V, R, ExecV1<V>> for InstanceV1<V>
where
    V: MyFloat,
    Open01: Distribution<V>,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    R: Rng,
{
    fn is_continued(&self) -> bool {
        !self.event_table.is_empty()
    }

    fn info_contents<'a>(
        ins: &mut InstanceWrapper<'a, ExecV1<V>, V, R, Self>,
        t: u32,
    ) -> Vec<(usize, &'a InfoContent<V>)> {
        let mut contents = Vec::new();
        // register observer agents
        // senders of observed info have priority over existing senders.
        if let Some(observer) = &ins.e.scenario.observer {
            let mut temp = Vec::new();
            ins.ext.observable.retain(|&agent_idx| {
                if ins.total_num_selfish <= observer.threshold {
                    return true;
                }
                if observer.po <= ins.rng.gen() {
                    return true;
                }
                if observer.pp <= ins.rng.gen() {
                    return true;
                }
                temp.push(agent_idx);
                false
            });
            if temp.len() > 1 {
                temp.shuffle(&mut ins.rng);
            }
            for agent_idx in temp {
                // let info_idx = self .core .new_info(&e.scenario.info_objects[observer.observed_info_obj_idx]);
                contents.push((
                    agent_idx,
                    &ins.e.scenario.info_objects[observer.observed_info_obj_idx],
                ));
            }
        }

        if let Some(informms) = ins.ext.event_table.remove(&t) {
            for i in informms {
                contents.push((i.agent_idx, &ins.e.scenario.info_objects[i.info_obj_idx]));
                // let info_idx = in.new_info(&e.scenario.info_objects[i.info_obj_idx]);
                // ips.push((i.agent_idx, info_idx));
            }
        }
        contents
    }

    fn get_sharer<'a>(
        ins: &mut InstanceWrapper<'a, ExecV1<V>, V, R, Self>,
        info_idx: usize,
    ) -> (Trusts<V>, AccessProb<V>) {
        let info = &ins.infos[info_idx];
        let params = &ins.e.agent_params.trust_params;
        let rng = &mut ins.rng;
        let friend_access_prob = params.friend_access_prob.sample(rng);
        let social_access_prob = params.social_access_prob.sample(rng);
        let friend_arrival_prob = params.friend_arrival_prob.sample(rng);
        let misinfo_friend = params.friend_misinfo_trust.sample(rng);
        let misinfo_social = params.social_misinfo_trust.sample(rng);
        let trust_sampler = params.info_trust_params.get_sampler(info.label());
        let info_trust = *ins
            .ext
            .info_trust_map
            .entry(info.idx)
            .or_insert_with(|| trust_sampler.sample(rng));
        let corr_misinfo_trust = *ins
            .ext
            .corr_misinfo_trust_map
            .entry(info.idx)
            .or_insert_with(|| {
                params
                    .info_trust_params
                    .get_sampler(&InfoLabel::Misinfo)
                    .sample(rng)
            });

        let receipt_prob = V::one()
            - (V::one() - V::from_usize(info.num_shared()).unwrap() / ins.e.scenario.fnum_nodes)
                .powf(ins.e.scenario.mean_degree);
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

    fn q(&self, agent: &Agent<V>) -> V {
        agent.access_prob()
    }
    fn get_informer<'a>(
        ins: &mut InstanceWrapper<'a, ExecV1<V>, V, R, Self>,
    ) -> (Trusts<V>, AccessProb<V>) {
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
                pred_fp: ins
                    .e
                    .agent_params
                    .trust_params
                    .friend_access_prob
                    .sample(&mut ins.rng)
                    * ins
                        .e
                        .agent_params
                        .trust_params
                        .friend_arrival_prob
                        .sample(&mut ins.rng),
                fm: V::one(),
                km: V::one(),
            },
        )
    }
}

// impl<'a, V: MyFloat, R> ExecutorInstance<'a, V, R, ExecV1<'a, V, R>> for InstanceV1<'a, V, R> {
//     fn info_producers(e: &mut ExecV1<'a, V, R>, t: u32) -> Vec<(usize, usize)> {
//         let mut ips = Vec::new();
//         // register observer agents
//         // senders of observed info have priority over existing senders.
//         if let Some(observer) = &self.scenario.observer {
//             let mut temp = Vec::new();
//             self.instance.observable.retain(|&agent_idx| {
//                 if self.instance.core.total_num_selfish <= observer.threshold {
//                     return true;
//                 }
//                 if observer.po <= self.instance.core.rng.gen() {
//                     return true;
//                 }
//                 if observer.pp <= self.instance.core.rng.gen() {
//                     return true;
//                 }
//                 temp.push(agent_idx);
//                 false
//             });
//             if temp.len() > 1 {
//                 temp.shuffle(&mut self.instance.core.rng);
//             }
//             for agent_idx in temp {
//                 let info_idx = self
//                     .instance
//                     .core
//                     .new_info(&self.info_objects[observer.observed_info_obj_idx]);
//                 ips.push((agent_idx, info_idx));
//             }
//         }

//         if let Some(informms) = self.instance.event_table.remove(&t) {
//             for i in informms {
//                 let info_idx = self
//                     .instance
//                     .core
//                     .new_info(&self.info_objects[i.info_obj_idx]);
//                 ips.push((i.agent_idx, info_idx));
//             }
//         }
//         ips
//     }

//     fn new(e: &ExecV1<'a, V, R>, rng: R) -> InstanceV1<'a, V, R> {
//         let observable = Vec::from_iter(0..self.agents.len());
//         InstanceV1 {
//             core: InstanceCore::new(rng),
//             event_table: self.scenario.table.clone(),
//             observable,
//         }
//     }
//     fn is_continued(&self) -> bool {
//         todo!()
//     }

//     fn core(&mut self) -> &mut InstanceCore<'a, V, R> {
//         &mut self.core
//     }

//     fn into_core(self) -> InstanceCore<'a, V, R> {
//         self.core
//     }
// }
