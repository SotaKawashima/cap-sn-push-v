use cap_sn::agent::{Agent, Info, InfoContent, A, PHI, PSI, THETA};
use graph_lib::prelude::{DiGraphL, Graph};
use rand::{rngs::SmallRng, seq::SliceRandom, SeedableRng};
use subjective_logic::mul::{Opinion1d, Simplex};

fn main() {
    let mut g = DiGraphL::<Agent>::new();
    let mut received = Vec::<(usize, usize)>::new();

    let br_psi = [1.0, 0.0];
    let br_ppsi = [1.0, 0.0];
    let br_pa = [1.0, 0.0];
    let br_phi = [1.0, 0.0];
    let mut infos = [Info::new(
        0,
        InfoContent::new(
            Opinion1d::<f32, PSI>::new([0.9, 0.09], 0.01, br_psi.clone()),
            Opinion1d::<f32, PSI>::new([0.0, 0.0], 1.0, br_ppsi.clone()),
            Opinion1d::<f32, A>::new([0.0, 0.0], 1.0, br_pa.clone()),
            Opinion1d::<f32, PHI>::new([0.0, 0.0], 1.0, br_phi.clone()),
            [
                Simplex::<f32, THETA>::vacuous(),
                Simplex::<f32, THETA>::vacuous(),
            ],
        ),
    )];
    // step
    let mut rng = SmallRng::seed_from_u64(0);
    let n = g.node_count() as f32;
    let d: f32 = 5.0; // todo! (compute average degree)

    while !received.is_empty() {
        received.shuffle(&mut rng);
        received = received
            .into_iter()
            .filter_map(|(aid, iid)| {
                let a = &mut g.node_slice_mut()[aid];
                let info = &mut infos[iid];
                let receipt_prob = 1.0 - (1.0 - info.num_shared() as f32 / n).powf(d);
                let b = a.receive_info(info, receipt_prob);
                if b.sharing {
                    info.shared();
                    Some(
                        g.successors(aid)
                            .unwrap()
                            .iter()
                            .filter_map(|bid| Some((*bid, iid))) // todo! (consider reading probability)
                            .collect::<Vec<_>>(),
                    )
                } else {
                    None
                }
            })
            .flatten()
            .collect();
    }
}
