use std::env;

// use blstrs::Scalar;
use ec_gpu_gen::SourceBuilder;
// use pairing::bn256::Fr as Scalar;

fn u64_to_u32(limbs: &[u64]) -> Vec<u32> {
    limbs
        .iter()
        .flat_map(|limb| vec![(limb & 0xFFFF_FFFF) as u32, (limb >> 32) as u32])
        .collect()
}

pub struct Scalar();

const MODULUS: [u64; 4] = [
    0xffff_ffff_0000_0001,
    0x53bd_a402_fffe_5bfe,
    0x3339_d808_09a1_d805,
    0x73ed_a753_299d_7d48,
];

const R: [u64; 4] = [
    0x0000_0001_ffff_fffe,
    0x5884_b7fa_0003_4802,
    0x998c_4fef_ecbc_4ff5,
    0x1824_b159_acc5_056f,
];

const R2: [u64; 4] = [
    0xc999_e990_f3f2_9c6d,
    0x2b6c_edcb_8792_5c23,
    0x05d3_1496_7254_398f,
    0x0748_d9d9_9f59_ff11,
];

impl ec_gpu::GpuName for Scalar {
    fn name() -> String {
        ec_gpu::name!()
    }
}

impl ec_gpu::GpuField for Scalar {
    fn one() -> Vec<u32> {
        crate::u64_to_u32(&R[..])
    }

    fn r2() -> Vec<u32> {
        crate::u64_to_u32(&R2[..])
    }

    fn modulus() -> Vec<u32> {
        crate::u64_to_u32(&MODULUS[..])
    }
}

fn main() {
    env::set_var("OUT_DIR", "outdir");
    let source_builder = SourceBuilder::new().add_fft::<Scalar>();
    ec_gpu_gen::generate(&source_builder);
}
