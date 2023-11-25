use approx::relative_eq;
use subjective_logic::mul::OpinionRef;
use subjective_logic::mul::{
    op::{Abduction, Deduction, Fuse, FuseAssign, FuseOp},
    Opinion1d, Simplex,
};

pub struct Agent {
    pub op: AgentOpinion,
    cpt: AgentCPT,
}

impl Agent {
    pub fn new(op: AgentOpinion, cpt: AgentCPT) -> Self {
        Self { op, cpt }
    }

    pub fn valuate(&self) -> [f32; 2] {
        self.cpt.valuate_prospects(self.op.w_th.projection()[0])
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
                AgentOpinion::OP.fuse_assign(&mut op.w_x, op_x).unwrap();
                AgentOpinion::OP.fuse_assign(&mut op.w_th, &mw_th).unwrap();
            }
            InfoProcess::P1 { op_o } => {
                let (mw_x, _) = op_o.abduce(&op.conds_ox, op.w_x.base_rate.clone()).unwrap();
                let mw_th = mw_x.deduce_with(&op.conds_thx, op.w_th.base_rate.clone());
                AgentOpinion::OP.fuse_assign(&mut op.w_o, op_o).unwrap();
                AgentOpinion::OP.fuse_assign(&mut op.w_x, &mw_x).unwrap();
                AgentOpinion::OP.fuse_assign(&mut op.w_th, &mw_th).unwrap();
            }
            InfoProcess::P2 { op_o, op_x } => {
                let (mw_x_ab, _) = op_o.abduce(&op.conds_ox, op.w_x.base_rate.clone()).unwrap();
                let mw_x = AgentOpinion::OP
                    .fuse(
                        mw_x_ab.as_ref(),
                        OpinionRef::from((op_x, &op.w_x.base_rate)),
                    )
                    .unwrap();
                let mw_th = mw_x.deduce_with(&op.conds_thx, op.w_th.base_rate.clone());
                AgentOpinion::OP.fuse_assign(&mut op.w_o, op_o).unwrap();
                AgentOpinion::OP.fuse_assign(&mut op.w_x, &mw_x).unwrap();
                AgentOpinion::OP.fuse_assign(&mut op.w_th, &mw_th).unwrap();
            }
        }
    }
}

#[derive(Debug)]
pub struct AgentCPT {
    alpha: f32,
    beta: f32,
    lambda: f32,
    gamma: f32,
    delta: f32,
    outcomes: [f32; 2],
    cost: f32,
}

impl AgentCPT {
    pub fn new(
        alpha: f32,
        beta: f32,
        lambda: f32,
        gamma: f32,
        delta: f32,
        outcomes: [f32; 2],
        cost: f32,
    ) -> Self {
        assert!(outcomes[0] < outcomes[1]);
        Self {
            alpha,
            beta,
            lambda,
            gamma,
            delta,
            outcomes,
            cost,
        }
    }

    fn w(p: f32, e: f32) -> f32 {
        let temp = p.powf(e);
        temp / (temp + (1.0 - p).powf(e)).powf(1.0 / e)
    }

    /// Computes a probability weighting function for gains
    fn weighting_p(&self, p: f32) -> f32 {
        Self::w(p, self.gamma)
    }

    /// Computes a probability weighting function for losses
    fn weighting_m(&self, p: f32) -> f32 {
        Self::w(p, self.delta)
    }

    /// Computes a value funciton
    fn value(&self, x: f32) -> f32 {
        if x.is_sign_positive() {
            x.powf(self.alpha)
        } else {
            -self.lambda * x.abs().powf(self.beta)
        }
    }

    /// Computes a CPT value of a prospect f_a
    fn valuate_prospect_a(&self) -> f32 {
        self.value(self.outcomes[1] - self.cost)
    }

    /// Computes a CPT value of a prospect f_{\bar a}
    fn valuate_prospect_na(&self, p: f32) -> f32 {
        let inv_p = 1.0 - p;
        if self.outcomes[0] > 0.0 || relative_eq!(self.outcomes[0], 0.0) {
            self.value(self.outcomes[0]) * (1.0 - self.weighting_p(inv_p))
                + self.value(self.outcomes[1]) * self.weighting_p(inv_p)
        } else if self.outcomes[1] > 0.0 || relative_eq!(self.outcomes[1], 0.0) {
            self.value(self.outcomes[0]) * self.weighting_m(p)
                + self.value(self.outcomes[1]) * self.weighting_p(inv_p)
        } else {
            self.value(self.outcomes[0]) * self.weighting_m(p)
                + self.value(self.outcomes[1]) * (1.0 - self.weighting_m(p))
        }
    }

    pub fn valuate_prospects(&self, p: f32) -> [f32; 2] {
        [self.valuate_prospect_a(), self.valuate_prospect_na(p)]
    }
}
