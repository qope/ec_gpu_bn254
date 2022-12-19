use std::env;

use ec_gpu_gen::SourceBuilder;

fn main() {
    env::set_var("OUT_DIR", "outdir");
    let source_builder = SourceBuilder::new().add_multiexp::<C, F>();
}
