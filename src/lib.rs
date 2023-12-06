pub mod agent;
pub mod cpt;
pub mod snippet;

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::ops::Deref;

use agent::{Agent, AgentOpinion, FriendOpinion, Info, InfoContent, A, PHI, PSI, THETA};
use arrow2::array::PrimitiveArray;
use arrow2::chunk::Chunk;
use arrow2::datatypes::{DataType, Field, Schema};
use arrow2::io::ipc::write::{Compression, StreamWriter, WriteOptions};
use cpt::{LevelSet, CPT};

use graph_lib::io::{DataFormat, ParseBuilder};
use graph_lib::prelude::{DiGraphB, Graph};
use rand::{rngs::SmallRng, seq::SliceRandom, SeedableRng};
use subjective_logic::{
    harr2,
    mul::{Opinion1d, Simplex},
};

struct Receipt {
    agent_idx: usize,
    force: bool,
    info_idx: usize,
}

pub fn exec_sim<R: Read, W: Write>(
    reader: R,
    writer: &mut W,
) -> Result<(), Box<dyn std::error::Error>> {
    let br_psi = [0.999, 0.001];
    let br_ppsi = [0.999, 0.001];
    let br_fpsi = [0.999, 0.001];
    let br_fppsi = [0.999, 0.001];
    let br_pa = [0.999, 0.001];
    let br_fa = [0.999, 0.001];
    let br_fpa = [0.999, 0.001];
    let br_phi = [0.999, 0.001];
    let br_fphi = [0.999, 0.001];
    let br_theta = [0.999, 0.0005, 0.0005];
    let br_ptheta = [0.999, 0.0005, 0.0005];
    let br_ftheta = [0.999, 0.0005, 0.0005];
    let br_fptheta = [0.999, 0.0005, 0.0005];

    let mut infos = [Info::new(
        0,
        InfoContent::new(
            Opinion1d::<f32, PSI>::new([0.01, 0.98], 0.01, br_psi.clone()),
            Opinion1d::<f32, PSI>::new([0.0, 0.0], 1.0, br_ppsi.clone()),
            Opinion1d::<f32, A>::new([0.0, 0.0], 1.0, br_pa.clone()),
            Opinion1d::<f32, PHI>::new([0.0, 0.0], 1.0, br_phi.clone()),
            [
                Simplex::<f32, THETA>::vacuous(),
                Simplex::<f32, THETA>::vacuous(),
            ],
        ),
    )];

    let x0 = -0.1;
    let x1 = -2.0;
    let y = -0.01;
    let selfish_outcome_maps = [[0.0, x1, 0.0], [x0, x0, x0]];
    let sharing_outcome_maps = [
        harr2![[0.0, x1, 0.0], [x0, x0, x0]],
        harr2![[y, x1 + y, y], [x0 + y, x0 + y, x0 + y]],
    ];
    let mut sender = vec![(0, 31, 0)];

    let g: DiGraphB = ParseBuilder::new(reader, DataFormat::EdgeList)
        .parse()
        .unwrap();

    let mut agents = (0..g.node_count())
        .map(|_| {
            Agent::new(
                infos.len(),
                AgentOpinion::new(br_theta, br_psi, br_phi, br_ppsi),
                FriendOpinion::new(br_fppsi, br_fpsi, br_fphi, br_fpa, br_fptheta, br_ftheta),
                CPT::new(0.88, 0.88, 2.25, 0.61, 0.69),
                [
                    LevelSet::<_, f32>::new(&selfish_outcome_maps[0]),
                    LevelSet::<_, f32>::new(&selfish_outcome_maps[1]),
                ],
                [
                    LevelSet::<_, f32>::new(sharing_outcome_maps[0].deref()),
                    LevelSet::<_, f32>::new(sharing_outcome_maps[1].deref()),
                ],
                0.9,
                0.9,
                0.9,
                vec![0.90],
                br_pa,
                br_ptheta,
                br_fa,
            )
        })
        .collect::<Vec<_>>();

    let mut received = Vec::<Receipt>::new();

    let mut rng = SmallRng::seed_from_u64(0);
    let n = g.node_count() as f32;
    let d = g.edge_count() as f32 / n; // average outdegree

    let mut col_t = Vec::new();
    let mut col_num_sharing = Vec::new();
    let mut col_num_receipt = Vec::new();

    let mut t: u32 = 0;
    while !received.is_empty() || !sender.is_empty() {
        col_t.push(t);
        if sender.first().map(|a| a.0 == t).unwrap_or(false) {
            let (_, agent_idx, info_idx) = sender.pop().unwrap();
            received.push(Receipt {
                agent_idx,
                force: true,
                info_idx,
            });
        }
        let mut num_sharing = 0;
        received.shuffle(&mut rng);
        received = received
            .into_iter()
            .filter_map(|r| {
                let a = &mut agents[r.agent_idx];
                println!("id: {}", r.agent_idx);
                let info = &mut infos[r.info_idx];
                let receipt_prob = 1.0 - (1.0 - info.num_shared() as f32 / n).powf(d);
                if !r.force {
                    if !a.try_read(&mut rng) {
                        return None;
                    }
                }
                let b = a.read_info(info, receipt_prob);
                // println!("{:?}", b);
                if !b.sharing {
                    return None;
                }
                if !r.force {
                    info.shared();
                    num_sharing += 1;
                }
                Some(
                    g.successors(r.agent_idx)
                        .unwrap()
                        .iter()
                        .map(|bid| Receipt {
                            agent_idx: *bid,
                            force: false,
                            info_idx: r.info_idx,
                        })
                        .collect::<Vec<_>>(),
                )
            })
            .flatten()
            .collect();
        col_num_sharing.push(num_sharing);
        col_num_receipt.push(received.len() as u32);
        t += 1;
    }
    let metadata = BTreeMap::from_iter([
        ("version".to_string(), env!("CARGO_PKG_VERSION").to_string()),
        ("graph".to_string(), "filmtrust".to_string()),
        ("info".to_string(), "#0: misinfo".to_string()),
        (
            "senarios".to_string(),
            "#31 post info #0 at t=0".to_string(),
        ),
    ]);

    let schema = Schema {
        fields: vec![
            Field::new("t", DataType::UInt32, false),
            Field::new("num_receipt", DataType::UInt32, false),
            Field::new("num_sharing", DataType::UInt32, false),
        ],
        metadata,
    };
    let chunk = Chunk::try_new(vec![
        PrimitiveArray::from_slice(&col_t).boxed(),
        PrimitiveArray::from_slice(&col_num_receipt).boxed(),
        PrimitiveArray::from_slice(&col_num_sharing).boxed(),
    ])?;
    let mut writer = StreamWriter::new(
        writer,
        WriteOptions {
            compression: Some(Compression::ZSTD),
        },
    );
    writer.start(&schema, None)?;
    writer.write(&chunk, None)?;
    writer.finish()?;
    Ok(())
}

pub fn test_agent() {
    let br_psi = [0.999, 0.001];
    let br_ppsi = [0.999, 0.001];
    let br_fpsi = [0.999, 0.001];
    let br_fppsi = [0.999, 0.001];
    let br_pa = [0.999, 0.001];
    let br_fa = [0.999, 0.001];
    let br_fpa = [0.999, 0.001];
    let br_phi = [0.999, 0.001];
    let br_fphi = [0.999, 0.001];
    let br_theta = [0.999, 0.0005, 0.0005];
    let br_ptheta = [0.999, 0.0005, 0.0005];
    let br_ftheta = [0.999, 0.0005, 0.0005];
    let br_fptheta = [0.999, 0.0005, 0.0005];

    let infos = [Info::new(
        0,
        InfoContent::new(
            Opinion1d::<f32, PSI>::new([0.01, 0.98], 0.01, br_psi.clone()),
            Opinion1d::<f32, PSI>::new([0.0, 0.0], 1.0, br_ppsi.clone()),
            Opinion1d::<f32, A>::new([0.0, 0.0], 1.0, br_pa.clone()),
            Opinion1d::<f32, PHI>::new([0.0, 0.0], 1.0, br_phi.clone()),
            [
                Simplex::<f32, THETA>::vacuous(),
                Simplex::<f32, THETA>::vacuous(),
            ],
        ),
    )];

    let x0 = -0.1;
    let x1 = -2.0;
    let y = -0.01;
    let selfish_outcome_maps = [[0.0, x1, 0.0], [x0, x0, x0]];
    let sharing_outcome_maps = [
        harr2![[0.0, x1, 0.0], [x0, x0, x0]],
        harr2![[y, x1 + y, y], [x0 + y, x0 + y, x0 + y]],
    ];

    let mut a = Agent::new(
        infos.len(),
        AgentOpinion::new(br_theta, br_psi, br_phi, br_ppsi),
        FriendOpinion::new(br_fppsi, br_fpsi, br_fphi, br_fpa, br_fptheta, br_ftheta),
        CPT::new(0.88, 0.88, 2.25, 0.61, 0.69),
        [
            LevelSet::<_, f32>::new(&selfish_outcome_maps[0]),
            LevelSet::<_, f32>::new(&selfish_outcome_maps[1]),
        ],
        [
            LevelSet::<_, f32>::new(sharing_outcome_maps[0].deref()),
            LevelSet::<_, f32>::new(sharing_outcome_maps[1].deref()),
        ],
        0.5,
        0.5,
        0.5,
        vec![0.90],
        br_pa,
        br_ptheta,
        br_fa,
    );

    let receipt_prob = 0.0;
    println!("{:?}", a.read_info(&infos[0], receipt_prob));
}
