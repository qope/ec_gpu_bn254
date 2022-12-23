use ark_bn254::{FqParameters, Fr, FrParameters};
use ark_ff::FpParameters;
use ec_gpu_gen::SourceBuilder;
use std::env;

fn u64_to_u32(limbs: &[u64]) -> Vec<u32> {
    limbs
        .iter()
        .flat_map(|limb| vec![(limb & 0xFFFF_FFFF) as u32, (limb >> 32) as u32])
        .collect()
}

pub struct Scalar();

impl ec_gpu::GpuName for Scalar {
    fn name() -> String {
        ec_gpu::name!()
    }
}

impl ec_gpu::GpuField for Scalar {
    fn one() -> Vec<u32> {
        u64_to_u32(&<FrParameters as FpParameters>::R.0[..])
    }

    fn r2() -> Vec<u32> {
        u64_to_u32(&<FrParameters as FpParameters>::R2.0[..])
    }

    fn modulus() -> Vec<u32> {
        u64_to_u32(&<FrParameters as FpParameters>::MODULUS.0[..])
    }
}

fn main() {
    env::set_var("OUT_DIR", "outdir");
    let source_builder = SourceBuilder::new().add_fft::<Scalar>();
    ec_gpu_gen::generate(&source_builder);
}
