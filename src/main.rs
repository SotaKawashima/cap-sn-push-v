use subjective_logic::{
    Abduction, BOpinion, Deduction, Fusion, MOpinion1d, MOpinion1dRef, MSimplex, MSimplexRef,
};

fn main() {
    let mut agent = Agent::new(
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
    );

    let mw_o = MSimplex::<f32, 2>::new([0.95, 0.0], 0.05);
    let mw_x = MSimplex::<f32, 3>::new([0.95, 0.0, 0.0], 0.05);
    println!("-- m3 --");
    agent.process3(mw_o.borrow());
    println!("{}", agent);
    agent.reset();

    println!("-- m4 --");
    agent.process4(mw_o.borrow(), mw_x.borrow());
    println!("{}", agent);
    agent.reset();

    println!("-- m5 --");
    agent.process4(
        mw_o.borrow(),
        MSimplex::<f32, 3>::new([0.0, 0.0, 0.95], 0.05).borrow(),
    );
    println!("{}", agent);
    agent.reset();

    println!("-- m5* --");
    agent.process4(
        mw_o.borrow(),
        MSimplex::<f32, 3>::new([0.0, 0.95, 0.0], 0.05).borrow(),
    );
    println!("{}", agent);
}

#[derive(Debug)]
struct Agent {
    w_o: MOpinion1d<f32, 2>,
    w_x: MOpinion1d<f32, 3>,
    w_th: MOpinion1d<f32, 2>,
    conds_ox: [MSimplex<f32, 2>; 3],
    conds_thx: [MSimplex<f32, 2>; 3],
    init_a_o: [f32; 2],
    init_a_x: [f32; 3],
    init_a_th: [f32; 2],
}

impl std::fmt::Display for Agent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "w_O  s: {:?}, a: {:?}",
            self.w_o.simplex, self.w_o.base_rate
        )?;
        writeln!(
            f,
            "w_X  s: {:?}, a: {:?}",
            self.w_x.simplex, self.w_x.base_rate
        )?;
        writeln!(
            f,
            "w_TH s: {:?}, a: {:?}",
            self.w_th.simplex, self.w_th.base_rate
        )?;
        write!(f, "P_TH: {}", self.w_th.projection(0))
    }
}

impl Agent {
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

    fn update_o(&mut self, w: MSimplexRef<f32, 2>) {
        self.w_o.simplex = self.w_o.cfuse_al(w, 0.5).unwrap();
    }

    fn update_x(&mut self, w: MOpinion1d<f32, 3>) {
        self.w_x = self.w_x.cfuse_al(&w, 0.5).unwrap();
    }

    fn update_th(&mut self, w: MOpinion1d<f32, 2>) {
        self.w_th = self.w_th.cfuse_al(&w, 0.5).unwrap();
    }

    fn process3(&mut self, mw_o: MSimplexRef<f32, 2>) {
        let (mw_x, _) = mw_o
            .clone()
            .abduce(&self.conds_ox, self.w_x.base_rate.clone(), None)
            .unwrap();
        let mw_th = mw_x.deduce(&self.conds_thx, self.w_th.base_rate.clone());
        self.update_o(mw_o);
        self.update_x(mw_x);
        self.update_th(mw_th);
    }

    fn process4(&mut self, mw_o: MSimplexRef<f32, 2>, ms_x: MSimplexRef<f32, 3>) {
        let (mw_x_ab, _) = mw_o
            .clone()
            .abduce(&self.conds_ox, self.w_x.base_rate.clone(), None)
            .unwrap();
        let mw_x = {
            let w = MOpinion1dRef::<f32, 3>::from((ms_x, &self.w_x.base_rate));
            mw_x_ab.cfuse_al(w, 0.5).unwrap()
        };
        let mw_th = mw_x.deduce(&self.conds_thx, self.w_th.base_rate.clone());
        self.update_o(mw_o);
        self.update_x(mw_x);
        self.update_th(mw_th);
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
