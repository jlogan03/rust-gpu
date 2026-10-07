// build-pass
// only-vulkan1.2
// compile-flags: -C target-feature=+Float16,+StorageBuffer16BitAccess

#![feature(f16)]

use spirv_std::spirv;

// `fmaf16` must lower to GLSL.std.450 Fma.
#[spirv(compute(threads(1)))]
pub fn main(
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] inputs: &[f16; 3],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] output: &mut f16,
) {
    *output = inputs[0].mul_add(inputs[1], inputs[2]);
}
