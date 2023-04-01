use subjective_logic::{MSL1d, MSLOps, SLOps, BSL};

fn main() {
    let a = 0.0;
    let b_m0a1 = 0.0;
    let m0 = [
        BSL::<f32>::new(0.5, 0.0, 1.0 - 0.5, a),
        BSL::<f32>::new(b_m0a1, 0.0, 1.0 - b_m0a1, a),
        BSL::<f32>::new(b_m0a1, 0.0, 1.0 - b_m0a1, a),
    ];

    let b_m1a1 = 0.5;
    let m1 = [
        BSL::<f32>::new(0.0, 0.9, 0.1, a),
        BSL::<f32>::new(b_m1a1, 0.0, 1.0 - b_m1a1, a),
        BSL::<f32>::new(b_m1a1, 0.0, 1.0 - b_m1a1, a),
    ];

    for i in 0..3 {
        let f = m0[i].cfus(&m1[i]).unwrap();
        println!("{f}");
    }
}

pub fn comparison_deductions() {
    let w_2_2 = deduction_2_2(0.6, 0.2, 0.7, 0.2, 0.5);
    let w_2_3 = deduction_2_3(0.6, 0.2, 0.7, 0.2, 0.5);

    println!("{}, {}", w_2_2.projection(), w_2_2);
    println!("{}, {}", w_2_3.projection(), w_2_3);
}

fn deduction_2_2(b_a0: f32, b_a1: f32, b_xa0: f32, b_xa1: f32, a: f32) -> BSL<f32> {
    let wa = MSL1d::<f32, 2>::new([b_a0, b_a1], 1.0 - b_a0 - b_a1, [1.0 / 3.0, 2.0 / 3.0]);
    let wxa = [
        BSL::<f32>::new(b_xa0, 0.0, 1.0 - b_xa0, a),
        BSL::<f32>::new(b_xa1, 0.0, 1.0 - b_xa1, a),
    ];
    wa.ded(&wxa, a)
}

fn deduction_2_3(b_a0: f32, b_a1: f32, b_xa0: f32, b_xa1: f32, a: f32) -> BSL<f32> {
    let wa = MSL1d::<f32, 3>::new(
        [b_a0, b_a1, 0.0],
        1.0 - b_a0 - b_a1,
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
    );
    let wxa = [
        BSL::<f32>::new(b_xa0, 0.0, 1.0 - b_xa0, a),
        BSL::<f32>::new(b_xa1, 0.0, 1.0 - b_xa1, a),
        BSL::<f32>::new(b_xa1, 0.0, 1.0 - b_xa1, a),
        // BSL::<f32>::new(0.0, 0.0, 1.0, a),
    ];
    wa.ded(&wxa, a)
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
    let wa = MSL1d::<f32, 3>::new(
        [b_a, b_b, 0.0],
        1.0 - b_a - b_b,
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
    );
    let wxa = [
        BSL::<f32>::new(b_xa, 0.0, 1.0 - b_xa, 0.5),
        BSL::<f32>::new(b_xb, 0.0, 1.0 - b_xb, 0.5),
        BSL::<f32>::new(0.0, 0.0, 1.0, 0.5),
    ];
    (0..3).map(|i| wa.projection(i) * wxa[i].projection()).sum()
}
