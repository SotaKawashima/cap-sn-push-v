use approx::relative_eq;
use subjective_logic::{
    Abduction, BOpinion, Deduction, Fusion, FusionAssign, FusionOp, MOpinion1d, MSimplex,
};

fn main() {
    let mut agent = Agent::new(
        AgentOpinion::new(
            MOpinion1d::<f32, 2>::new([0.0, 0.0], 1.0, [0.5, 0.5]),
            MOpinion1d::<f32, 3>::new([0.0, 0.0, 0.0], 1.0, [0.01, 0.1, 0.89]),
            MOpinion1d::<f32, 2>::new([0.0, 0.0], 1.0, [0.01, 0.99]),
            [
                MSimplex::<f32, 2>::new([0.99, 0.0], 0.01),
                MSimplex::<f32, 2>::new([0.99, 0.0], 0.01),
                MSimplex::<f32, 2>::new([0.35, 0.35], 0.3),
            ],
            [
                MSimplex::<f32, 2>::new([0.7, 0.0], 0.3),
                MSimplex::<f32, 2>::new([0.7, 0.0], 0.3),
                MSimplex::<f32, 2>::new([0.0, 0.7], 0.3),
            ],
        ),
        AgentCPT::new(0.88, 0.88, 2.25, 0.61, 0.69, [-1.0, 1.0], 0.9),
    );

    let mw_o = MSimplex::<f32, 2>::new([0.95, 0.0], 0.05);
    let mw_x = [
        MSimplex::<f32, 3>::new([0.2, 0.05, 0.05], 0.7),
        MSimplex::<f32, 3>::new([0.0, 0.0, 0.95], 0.05),
        MSimplex::<f32, 3>::new([0.0, 0.95, 0.0], 0.05),
    ];
    let ips = [
        ("m1", InfoProcess::P0 { op_x: &mw_x[0] }),
        ("m2", InfoProcess::P0 { op_x: &mw_x[1] }),
        ("m2*", InfoProcess::P0 { op_x: &mw_x[2] }),
        ("m3", InfoProcess::P1 { op_o: &mw_o }),
        (
            "m4",
            InfoProcess::P2 {
                op_o: &mw_o,
                op_x: &mw_x[0],
            },
        ),
        (
            "m5",
            InfoProcess::P2 {
                op_o: &mw_o,
                op_x: &mw_x[1],
            },
        ),
        (
            "m5*",
            InfoProcess::P2 {
                op_o: &mw_o,
                op_x: &mw_x[2],
            },
        ),
    ];

    println!("-- initial state --");
    println!("{}", agent.op);

    for (m, ip) in ips {
        println!("-- i.s. -> {m} --");
        agent.op.info_process(ip);
        println!("{}", agent.op);
        let val = agent.valuate();
        println!("V(f_a)={:.3},V(f_~a)={:.3}", val[0], val[1]);
        agent.op.reset();
    }
}

struct Agent {
    op: AgentOpinion,
    cpt: AgentCPT,
}

impl Agent {
    pub fn new(op: AgentOpinion, cpt: AgentCPT) -> Self {
        Self { op, cpt }
    }

    pub fn valuate(&self) -> [f32; 2] {
        self.cpt.valuate_prospects(self.op.w_th.projection(0))
    }
}

#[derive(Debug)]
struct AgentOpinion {
    w_o: MOpinion1d<f32, 2>,
    w_x: MOpinion1d<f32, 3>,
    w_th: MOpinion1d<f32, 2>,
    conds_ox: [MSimplex<f32, 2>; 3],
    conds_thx: [MSimplex<f32, 2>; 3],
    init_a_o: [f32; 2],
    init_a_x: [f32; 3],
    init_a_th: [f32; 2],
}

impl std::fmt::Display for AgentOpinion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fn put<const N: usize>(w: &MOpinion1d<f32, N>) -> String {
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
        write!(f, "P(w_TH)={:.3}", self.w_th.projection(0))
    }
}

impl AgentOpinion {
    const OP: FusionOp<f32> = FusionOp::CumulativeA(0.5);

    pub fn new(
        w_o: MOpinion1d<f32, 2>,
        w_x: MOpinion1d<f32, 3>,
        w_th: MOpinion1d<f32, 2>,
        conds_ox: [MSimplex<f32, 2>; 3],
        conds_thx: [MSimplex<f32, 2>; 3],
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
        self.w_o.simplex = MSimplex::new_unchecked([0.0, 0.0], 1.0);
        self.w_o.base_rate = self.init_a_o.clone();
        self.w_x.simplex = MSimplex::new_unchecked([0.0, 0.0, 0.0], 1.0);
        self.w_x.base_rate = self.init_a_x.clone();
        self.w_th.simplex = MSimplex::new_unchecked([0.0, 0.0], 1.0);
        self.w_th.base_rate = self.init_a_th.clone();
    }

    pub fn info_process(&mut self, ip: InfoProcess) {
        match ip {
            InfoProcess::P0 { op_x } => self.process0(op_x),
            InfoProcess::P1 { op_o } => self.process1(op_o),
            InfoProcess::P2 { op_o, op_x } => self.process2(op_o, op_x),
        }
    }

    fn process0(&mut self, ms_x: &MSimplex<f32, 3>) {
        let mw_th =
            (ms_x, &self.w_x.base_rate).deduce(&self.conds_thx, self.w_th.base_rate.clone());
        self.w_x.fusion_assign(ms_x, &Self::OP).unwrap();
        self.w_th.fusion_assign(&mw_th, &Self::OP).unwrap();
    }

    fn process1(&mut self, mw_o: &MSimplex<f32, 2>) {
        let (mw_x, _) = mw_o
            .clone()
            .abduce(&self.conds_ox, self.w_x.base_rate.clone(), None)
            .unwrap();
        let mw_th = mw_x.deduce(&self.conds_thx, self.w_th.base_rate.clone());
        self.w_o.fusion_assign(mw_o, &Self::OP).unwrap();
        self.w_x.fusion_assign(&mw_x, &Self::OP).unwrap();
        self.w_th.fusion_assign(&mw_th, &Self::OP).unwrap();
    }

    fn process2(&mut self, mw_o: &MSimplex<f32, 2>, ms_x: &MSimplex<f32, 3>) {
        let (mw_x_ab, _) = mw_o
            .clone()
            .abduce(&self.conds_ox, self.w_x.base_rate.clone(), None)
            .unwrap();
        let mw_x = mw_x_ab.cfuse_al((ms_x, &self.w_x.base_rate), 0.5).unwrap();
        let mw_th = mw_x.deduce(&self.conds_thx, self.w_th.base_rate.clone());
        self.w_o.fusion_assign(mw_o, &Self::OP).unwrap();
        self.w_x.fusion_assign(&mw_x, &Self::OP).unwrap();
        self.w_th.fusion_assign(&mw_th, &Self::OP).unwrap();
    }
}

enum InfoProcess<'a> {
    P0 {
        op_x: &'a MSimplex<f32, 3>,
    },
    P1 {
        op_o: &'a MSimplex<f32, 2>,
    },
    P2 {
        op_o: &'a MSimplex<f32, 2>,
        op_x: &'a MSimplex<f32, 3>,
    },
}

#[derive(Debug)]
struct AgentCPT {
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

pub fn deduce_unit3() {
    let w = MOpinion1d::<f32, 2>::new([0.0, 0.0], 1.0, [0.2, 0.8]);
    let a = [0.1, 0.9];
    let cnd = [
        BOpinion::<f32>::new(1.0, 0.0, 0.0, a[0]),
        BOpinion::<f32>::new(0.0, 1.0, 0.0, a[0]),
    ];
    let u = w.deduce((&cnd).into(), a[0]);

    println!("   w: {w:?}");
    println!(" ded: {:?}", u);
    println!("  pu: {:?}", u.projection());
}

pub fn deduce_unit2() {
    let ws = [
        MOpinion1d::<f32, 2>::new([0.25, 0.125], 0.625, [0.5, 0.5]),
        MOpinion1d::<f32, 2>::new([0.0, 0.0], 1.0, [0.5, 0.5]),
    ];

    let cnds = [
        [
            MOpinion1d::<f32, 2>::new([1.0, 0.0], 0.0, [0.5, 0.5]),
            MOpinion1d::<f32, 2>::new([0.0, 1.0], 0.0, [0.5, 0.5]),
        ],
        [
            MOpinion1d::<f32, 2>::new([0.0, 1.0], 0.0, [0.5, 0.5]),
            MOpinion1d::<f32, 2>::new([1.0, 0.0], 0.0, [0.5, 0.5]),
        ],
        [
            MOpinion1d::<f32, 2>::new([1.0, 0.0], 0.0, [0.5, 0.5]),
            MOpinion1d::<f32, 2>::new([0.0, 0.0], 1.0, [0.5, 0.5]),
        ],
        [
            MOpinion1d::<f32, 2>::new([1.0, 0.0], 0.0, [0.5, 0.5]),
            MOpinion1d::<f32, 2>::new([0.75, 0.25], 0.0, [0.5, 0.5]),
        ],
        [
            MOpinion1d::<f32, 2>::new([1.0, 0.0], 0.0, [0.5, 0.5]),
            MOpinion1d::<f32, 2>::new([0.75, 0.0], 0.25, [0.5, 0.5]),
        ],
        [
            MOpinion1d::<f32, 2>::new([1.0, 0.0], 0.0, [0.5, 0.5]),
            MOpinion1d::<f32, 2>::new([1.0, 0.0], 0.0, [0.5, 0.5]),
        ],
    ];

    for cnd in &cnds {
        println!("{:?}", cnd);
        for w in &ws {
            println!("   w: {w:?}");
            println!(" ded: {:?}", w.deduce(cnd, [0.5, 0.5]));
        }
    }
}

pub fn deduce_unit() {
    let a = 0.5;
    let ws = [
        // BOpinion::<f32>::new(0.5, 0.0, 0.5, a),
        BOpinion::<f32>::new(0.0, 1.0, 0.0, a),
        // BOpinion::<f32>::new(0.7, 0.1, 0.2, a),
    ];

    let cnd = [
        BOpinion::<f32>::new(1.0, 0.0, 0.0, a),
        BOpinion::<f32>::new(0.0, 0.0, 1.0, a),
    ];

    for w in &ws {
        let f = w.deduce((&cnd).into(), a);
        println!("{w}");
        println!("  {f}");
        println!("  {}", w.cfuse(&f).unwrap());
    }
}

pub fn comparison_deductions() {
    let w_2_2 = deduction_2_2(0.6, 0.2, 0.7, 0.2, 0.5);
    let w_2_3 = deduction_2_3(0.6, 0.2, 0.7, 0.2, 0.5);

    println!("{}, {}", w_2_2.projection(), w_2_2);
    println!("{}, {}", w_2_3.projection(), w_2_3);
}

fn deduction_2_2(b_a0: f32, b_a1: f32, b_xa0: f32, b_xa1: f32, a: f32) -> BOpinion<f32> {
    let wa = MOpinion1d::<f32, 2>::new([b_a0, b_a1], 1.0 - b_a0 - b_a1, [1.0 / 3.0, 2.0 / 3.0]);
    let wxa = [
        BOpinion::<f32>::new(b_xa0, 0.0, 1.0 - b_xa0, a),
        BOpinion::<f32>::new(b_xa1, 0.0, 1.0 - b_xa1, a),
    ];
    wa.deduce((&wxa).into(), a)
}

fn deduction_2_3(b_a0: f32, b_a1: f32, b_xa0: f32, b_xa1: f32, a: f32) -> BOpinion<f32> {
    let wa = MOpinion1d::<f32, 3>::new(
        [b_a0, b_a1, 0.0],
        1.0 - b_a0 - b_a1,
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
    );
    let wxa = [
        BOpinion::<f32>::new(b_xa0, 0.0, 1.0 - b_xa0, a),
        BOpinion::<f32>::new(b_xa1, 0.0, 1.0 - b_xa1, a),
        BOpinion::<f32>::new(b_xa1, 0.0, 1.0 - b_xa1, a),
        // BSL::<f32>::new(0.0, 0.0, 1.0, a),
    ];
    wa.deduce((&wxa).into(), a)
}

pub fn check_deduction() {
    let k = 3;
    let n = 2i32.pow(k);
    let r = (n as f32).recip();

    for i in 0..=(n / 2) {
        let b_a = r * (i as f32);
        for j in 0..=n {
            let b_xa = r * (j as f32);
            println!("{b_a}, {b_xa}, {}", calc_deduction(b_a, b_xa, 0.5, 1.0));
        }
    }
}

fn calc_deduction(b_a: f32, b_xa: f32, b_b: f32, b_xb: f32) -> f32 {
    let wa = MOpinion1d::<f32, 3>::new(
        [b_a, b_b, 0.0],
        1.0 - b_a - b_b,
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
    );
    let wxa = [
        BOpinion::<f32>::new(b_xa, 0.0, 1.0 - b_xa, 0.5),
        BOpinion::<f32>::new(b_xb, 0.0, 1.0 - b_xb, 0.5),
        BOpinion::<f32>::new(0.0, 0.0, 1.0, 0.5),
    ];
    (0..3).map(|i| wa.projection(i) * wxa[i].projection()).sum()
}
