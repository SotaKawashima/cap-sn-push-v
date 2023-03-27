use subjective_logic::{MSL1d, MSLOps, BSL};

fn main() {
    println!("Hello, world!");
    let wxa = [
        BSL::<f32>::new(0.7, 0.0, 1.0, 1.0 / 3.0),
        BSL::<f32>::new(0.0, 0.0, 1.0, 1.0 / 3.0),
        BSL::<f32>::new(0.0, 0.0, 1.0, 1.0 / 3.0),
    ];

    let wa = MSL1d::<f32, 3>::new([1.0, 0.0, 0.0], 0.0, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);

    let x = wa.ded(&wxa, 0.5);
    println!("{}", x.projection());
}
