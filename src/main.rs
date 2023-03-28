use subjective_logic::{MSL1d, BSL};

fn main() {
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
    // let x = wa.ded(&wxa, 0.5);
    // x.projection()
    (0..3).map(|i| wa.projection(i) * wxa[i].projection()).sum()
}
