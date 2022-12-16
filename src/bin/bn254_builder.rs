use std::env;

use ec_gpu_gen::SourceBuilder;

fn u64_to_u32(limbs: &[u64]) -> Vec<u32> {
    limbs
        .iter()
        .flat_map(|limb| vec![(limb & 0xFFFF_FFFF) as u32, (limb >> 32) as u32])
        .collect()
}

pub struct Scalar();

const MODULUS: [u64; 4] = [
    0x3c20_8c16_d87c_fd47,
    0x9781_6a91_6871_ca8d,
    0xb850_45b6_8181_585d,
    0x3064_4e72_e131_a029,
];

const R: [u64; 4] = [
    0xd35d_438d_c58f_0d9d,
    0x0a78_eb28_f5c7_0b3d,
    0x666e_a36f_7879_462c,
    0x0e0a_77c1_9a07_df2f,
];

const R2: [u64; 4] = [
    0xf32c_fc5b_538a_fa89,
    0xb5e7_1911_d445_01fb,
    0x47ab_1eff_0a41_7ff6,
    0x06d8_9f71_cab8_351f,
];

impl ec_gpu::GpuName for Scalar {
    fn name() -> String {
        ec_gpu::name!()
    }
}

impl ec_gpu::GpuField for Scalar {
    fn one() -> Vec<u32> {
        u64_to_u32(&R[..])
    }

    fn r2() -> Vec<u32> {
        u64_to_u32(&R2[..])
    }

    fn modulus() -> Vec<u32> {
        u64_to_u32(&MODULUS[..])
    }
}

fn main() {
    env::set_var("OUT_DIR", "outdir");
    let source_builder = SourceBuilder::new().add_fft::<Scalar>();
    ec_gpu_gen::generate(&source_builder);
}
