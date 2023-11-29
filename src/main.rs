use cap_sn::{
    agent::{Agent, AgentOpinion},
    cpt::{LevelSet, CPT},
};
use subjective_logic::mul::{Opinion1d, Simplex};

fn main() {
    // let mut agent = Agent::new(
    //     AgentOpinion::new(
    //         Opinion1d::<f32, 2>::new([0.0, 0.0], 1.0, [0.5, 0.5]),
    //         Opinion1d::<f32, 3>::new([0.0, 0.0, 0.0], 1.0, [0.01, 0.1, 0.89]),
    //         Opinion1d::<f32, 2>::new([0.0, 0.0], 1.0, [0.01, 0.99]),
    //         [
    //             Simplex::<f32, 2>::new([0.99, 0.0], 0.01),
    //             Simplex::<f32, 2>::new([0.99, 0.0], 0.01),
    //             Simplex::<f32, 2>::new([0.35, 0.35], 0.3),
    //         ],
    //         [
    //             Simplex::<f32, 2>::new([0.7, 0.0], 0.3),
    //             Simplex::<f32, 2>::new([0.7, 0.0], 0.3),
    //             Simplex::<f32, 2>::new([0.0, 0.7], 0.3),
    //         ],
    //     ),
    //     CPT::new(0.88, 0.88, 2.25, 0.61, 0.69),
    //     [
    //         LevelSet::<_, f32>::new(&[0.5, -0.5]),
    //         LevelSet::<_, f32>::new(&[0.2, -0.2]),
    //     ],
    // );

    // let mw_o = Simplex::<f32, 2>::new([0.95, 0.0], 0.05);
    // let mw_x = [
    //     Simplex::<f32, 3>::new([0.2, 0.05, 0.05], 0.7),
    //     Simplex::<f32, 3>::new([0.0, 0.0, 0.95], 0.05),
    //     Simplex::<f32, 3>::new([0.0, 0.95, 0.0], 0.05),
    // ];
    // let ips = [
    //     ("m1", InfoProcess::P0 { op_x: &mw_x[0] }),
    //     ("m2", InfoProcess::P0 { op_x: &mw_x[1] }),
    //     ("m2*", InfoProcess::P0 { op_x: &mw_x[2] }),
    //     ("m3", InfoProcess::P1 { op_o: &mw_o }),
    //     (
    //         "m4",
    //         InfoProcess::P2 {
    //             op_o: &mw_o,
    //             op_x: &mw_x[0],
    //         },
    //     ),
    //     (
    //         "m5",
    //         InfoProcess::P2 {
    //             op_o: &mw_o,
    //             op_x: &mw_x[1],
    //         },
    //     ),
    //     (
    //         "m5*",
    //         InfoProcess::P2 {
    //             op_o: &mw_o,
    //             op_x: &mw_x[2],
    //         },
    //     ),
    // ];

    // println!("-- initial state --");
    // println!("{}", agent.op);

    // for (m, ip) in ips {
    //     println!("-- i.s. -> {m} --");
    //     ip.info_process(&mut agent.op);
    //     println!("{}", agent.op);
    //     let val = agent.valuate();
    //     println!("V(f_a)={:.3},V(f_~a)={:.3}", val[0], val[1]);
    //     agent.op.reset();
    // }
}
