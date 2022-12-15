use std::env;

use blstrs::Scalar;
use ec_gpu_gen::SourceBuilder;
use pairing::bn256::Bn256;

fn main() {
    env::set_var("OUT_DIR", "outdir");
    let source_builder = SourceBuilder::new().add_fft::<Scalar>();
    ec_gpu_gen::generate(&source_builder);
}
