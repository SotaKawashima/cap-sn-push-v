use subjective_logic::{BOpinion, Deduction, MOpinion1d};

fn main() {
    // deduce_unit();
    deduce_unit3();
    let w = BOpinion::<f32>::new(0.0, 1.0, 0.0, 0.5);
    let w2 = BOpinion::<f32>::new(0.5, 0.0, 0.5, 0.5);
    println!("{}", w.cfuse(&w2).unwrap());
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
            println!(" ded: {:?}", w.deduce(cnd.into(), [0.5, 0.5]));
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
