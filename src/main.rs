use std::fs::File;

use cap_sn::exec_sim;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let reader = File::open("./dataset/librec-filmtrust-trust/out.librec-filmtrust-trust")?;
    let mut writer = File::create("result.arrow")?;
    exec_sim(reader, &mut writer)
}
