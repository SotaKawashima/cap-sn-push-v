use subjective_logic::bi::{BOpinion, BSimplex};
use subjective_logic::mul::{op::Deduction, Opinion1d, Projection, Simplex};

pub fn deduce_unit3() {
    let w = Opinion1d::<f32, 2>::new([0.0, 0.0], 1.0, [0.2, 0.8]);
    let cnd = [
        BSimplex::<f32>::new(1.0, 0.0, 0.0),
        BSimplex::<f32>::new(0.0, 1.0, 0.0),
    ];
    let u = w.deduce(&cnd).unwrap();

    println!("   w: {w:?}");
    println!(" ded: {:?}", u);
    println!("  pu: {:?}", u.projection());
}

pub fn deduce_unit2() {
    let ws = [
        Opinion1d::<f32, 2>::new([0.25, 0.125], 0.625, [0.5, 0.5]),
        Opinion1d::<f32, 2>::new([0.0, 0.0], 1.0, [0.5, 0.5]),
    ];

    let cnds = [
        [
            Simplex::<f32, 2>::new([1.0, 0.0], 0.0),
            Simplex::<f32, 2>::new([0.0, 1.0], 0.0),
        ],
        [
            Simplex::<f32, 2>::new([0.0, 1.0], 0.0),
            Simplex::<f32, 2>::new([1.0, 0.0], 0.0),
        ],
        [
            Simplex::<f32, 2>::new([1.0, 0.0], 0.0),
            Simplex::<f32, 2>::new([0.0, 0.0], 1.0),
        ],
        [
            Simplex::<f32, 2>::new([1.0, 0.0], 0.0),
            Simplex::<f32, 2>::new([0.75, 0.25], 0.0),
        ],
        [
            Simplex::<f32, 2>::new([1.0, 0.0], 0.0),
            Simplex::<f32, 2>::new([0.75, 0.0], 0.25),
        ],
        [
            Simplex::<f32, 2>::new([1.0, 0.0], 0.0),
            Simplex::<f32, 2>::new([1.0, 0.0], 0.0),
        ],
    ];

    for cnd in &cnds {
        println!("{:?}", cnd);
        for w in &ws {
            println!("   w: {w:?}");
            println!(" ded: {:?}", w.deduce(cnd));
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
        BSimplex::<f32>::new(1.0, 0.0, 0.0),
        BSimplex::<f32>::new(0.0, 0.0, 1.0),
    ];

    for w in &ws {
        let f = w.deduce(&cnd, a);
        println!("{w}");
        println!("  {f}");
        println!("  {}", w.cfuse(&f).unwrap());
    }
}

pub fn comparison_deductions() {
    let w_2_2 = deduction_2_2(0.6, 0.2, 0.7, 0.2);
    let w_2_3 = deduction_2_3(0.6, 0.2, 0.7, 0.2);

    println!("{}, {}", w_2_2.projection(), w_2_2);
    println!("{}, {}", w_2_3.projection(), w_2_3);
}

fn deduction_2_2(b_a0: f32, b_a1: f32, b_xa0: f32, b_xa1: f32) -> BOpinion<f32> {
    let wa = Opinion1d::<f32, 2>::new([b_a0, b_a1], 1.0 - b_a0 - b_a1, [1.0 / 3.0, 2.0 / 3.0]);
    let wxa = [
        BSimplex::<f32>::new(b_xa0, 0.0, 1.0 - b_xa0),
        BSimplex::<f32>::new(b_xa1, 0.0, 1.0 - b_xa1),
    ];
    wa.deduce(&wxa).unwrap().into()
}

fn deduction_2_3(b_a0: f32, b_a1: f32, b_xa0: f32, b_xa1: f32) -> BOpinion<f32> {
    let wa = Opinion1d::<f32, 3>::new(
        [b_a0, b_a1, 0.0],
        1.0 - b_a0 - b_a1,
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
    );
    let wxa = [
        BSimplex::<f32>::new(b_xa0, 0.0, 1.0 - b_xa0),
        BSimplex::<f32>::new(b_xa1, 0.0, 1.0 - b_xa1),
        BSimplex::<f32>::new(b_xa1, 0.0, 1.0 - b_xa1),
        // BSL::<f32>::new(0.0, 0.0, 1.0, a),
    ];
    wa.deduce(&wxa).unwrap().into()
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
    let wa = Opinion1d::<f32, 3>::new(
        [b_a, b_b, 0.0],
        1.0 - b_a - b_b,
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
    );
    let wxa = [
        BOpinion::<f32>::new(b_xa, 0.0, 1.0 - b_xa, 0.5),
        BOpinion::<f32>::new(b_xb, 0.0, 1.0 - b_xb, 0.5),
        BOpinion::<f32>::new(0.0, 0.0, 1.0, 0.5),
    ];
    (0..3)
        .map(|i| wa.projection()[i] * wxa[i].projection())
        .sum()
}
