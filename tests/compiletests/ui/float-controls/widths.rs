// build-pass
// only-vulkan1.2
// compile-flags: -C target-feature=+Float16,+Float64,+StorageBuffer16BitAccess,+DenormPreserve,+DenormFlushToZero,+RoundingModeRTE
// compile-flags: -C llvm-args=--disassemble-globals
// normalize-stderr-test "\n\W*OpSource .*" -> ""
// normalize-stderr-test "\n\W*%\d+ = OpString .*" -> ""

#![feature(f16)]

use spirv_std::spirv;

// Reuse the explicit f16 settings and add the policy's settings for f64.
#[spirv(compute(threads(1), rust_math, denorm_preserve = 16, rounding_mode_rte = 16))]
pub fn mixed(
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] half: &mut f16,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] double: &mut f64,
) {
    *half += 1.0;
    *double += 1.0;
}

#[spirv(compute(threads(1), fast_math, denorm_flush_to_zero = 64, rounding_mode_rte = 64))]
pub fn fast_double(#[spirv(storage_buffer, descriptor_set = 0, binding = 0)] data: &mut f64) {
    *data += 1.0;
}
