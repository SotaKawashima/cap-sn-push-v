use std::array;

use subjective_logic::mul::{
    op::{Abduction, Deduction, Fuse, FuseAssign, FuseOp},
    Opinion1d, Simplex,
};
use subjective_logic::mul::{OpinionRef, Projection};

use crate::cpt::{LevelSet, CPT};

pub struct Agent {
    pub op: AgentOpinion,
    cpt: CPT,
    selfish: [LevelSet<usize, f32>; 2],
}

impl Agent {
    pub fn new(op: AgentOpinion, cpt: CPT, selfish: [LevelSet<usize, f32>; 2]) -> Self {
        Self { op, cpt, selfish }
    }

    pub fn valuate(&self) -> [f32; 2] {
        let p = self.op.w_th.projection();
        array::from_fn(|i| self.cpt.valuate(&self.selfish[i], &p))
    }
}

#[derive(Debug)]
pub struct AgentOpinion {
    w_o: Opinion1d<f32, 2>,
    w_x: Opinion1d<f32, 3>,
    w_th: Opinion1d<f32, 2>,
    conds_ox: [Simplex<f32, 2>; 3],
    conds_thx: [Simplex<f32, 2>; 3],
    init_a_o: [f32; 2],
    init_a_x: [f32; 3],
    init_a_th: [f32; 2],
}

impl std::fmt::Display for AgentOpinion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fn put<const N: usize>(w: &Opinion1d<f32, N>) -> String {
            let mut str = String::new();
            for (i, v) in w.simplex.b().iter().enumerate() {
                str.push_str(&format!("b{}={:.3} ", i, v));
            }
            str.push_str(&format!("u ={:.3} ", w.u()));
            for (i, v) in w.base_rate.iter().enumerate() {
                str.push_str(&format!("a{}={:.3} ", i, v));
            }
            str.pop();
            str
        }
        writeln!(f, "w_O  {}", put(&self.w_o))?;
        writeln!(f, "w_X  {}", put(&self.w_x))?;
        writeln!(f, "w_TH {}", put(&self.w_th))?;
        write!(f, "P(w_TH)={:.3}", self.w_th.projection()[0])
    }
}

impl AgentOpinion {
    const OP: FuseOp = FuseOp::ACm;

    pub fn new(
        w_o: Opinion1d<f32, 2>,
        w_x: Opinion1d<f32, 3>,
        w_th: Opinion1d<f32, 2>,
        conds_ox: [Simplex<f32, 2>; 3],
        conds_thx: [Simplex<f32, 2>; 3],
    ) -> Self {
        Self {
            init_a_o: w_o.base_rate.clone(),
            init_a_x: w_x.base_rate.clone(),
            init_a_th: w_th.base_rate.clone(),
            w_o,
            w_x,
            w_th,
            conds_ox,
            conds_thx,
        }
    }

    pub fn reset(&mut self) {
        self.w_o.simplex = Simplex::new_unchecked([0.0, 0.0], 1.0);
        self.w_o.base_rate = self.init_a_o.clone();
        self.w_x.simplex = Simplex::new_unchecked([0.0, 0.0, 0.0], 1.0);
        self.w_x.base_rate = self.init_a_x.clone();
        self.w_th.simplex = Simplex::new_unchecked([0.0, 0.0], 1.0);
        self.w_th.base_rate = self.init_a_th.clone();
    }
}

pub enum InfoProcess<'a> {
    P0 {
        op_x: &'a Simplex<f32, 3>,
    },
    P1 {
        op_o: &'a Simplex<f32, 2>,
    },
    P2 {
        op_o: &'a Simplex<f32, 2>,
        op_x: &'a Simplex<f32, 3>,
    },
}

impl<'a> InfoProcess<'a> {
    pub fn info_process(self, op: &mut AgentOpinion) {
        match self {
            InfoProcess::P0 { op_x } => {
                let mw_x = OpinionRef::from((op_x, &op.w_x.base_rate));
                let mw_th = mw_x.deduce_with(&op.conds_thx, op.w_th.base_rate.clone());
                AgentOpinion::OP.fuse_assign(&mut op.w_x, op_x);
                AgentOpinion::OP.fuse_assign(&mut op.w_th, &mw_th);
            }
            InfoProcess::P1 { op_o } => {
                let (mw_x, _) = op_o.abduce(&op.conds_ox, op.w_x.base_rate.clone()).unwrap();
                let mw_th = mw_x.deduce_with(&op.conds_thx, op.w_th.base_rate.clone());
                AgentOpinion::OP.fuse_assign(&mut op.w_o, op_o);
                AgentOpinion::OP.fuse_assign(&mut op.w_x, &mw_x);
                AgentOpinion::OP.fuse_assign(&mut op.w_th, &mw_th);
            }
            InfoProcess::P2 { op_o, op_x } => {
                let (mw_x_ab, _) = op_o.abduce(&op.conds_ox, op.w_x.base_rate.clone()).unwrap();
                let mw_x = AgentOpinion::OP.fuse(
                    mw_x_ab.as_ref(),
                    OpinionRef::from((op_x, &op.w_x.base_rate)),
                );
                let mw_th = mw_x.deduce_with(&op.conds_thx, op.w_th.base_rate.clone());
                AgentOpinion::OP.fuse_assign(&mut op.w_o, op_o);
                AgentOpinion::OP.fuse_assign(&mut op.w_x, &mw_x);
                AgentOpinion::OP.fuse_assign(&mut op.w_th, &mw_th);
            }
        }
    }
}
