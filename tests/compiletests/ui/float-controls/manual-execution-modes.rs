// build-pass
// only-vulkan1.2
// compile-flags: -C target-feature=+Float64,+SignedZeroInfNanPreserve,+RoundingModeRTZ,+DenormFlushToZero
// compile-flags: -C llvm-args=--disassemble-globals
// normalize-stderr-test "\n\W*OpSource .*" -> ""
// normalize-stderr-test "\n\W*%\d+ = OpString .*" -> ""

use spirv_std::spirv;

// Compatibility mode leaves the manually requested float environment intact.
#[spirv(compute(
    threads(1),
    compat_math,
    signed_zero_inf_nan_preserve = 64,
    rounding_mode_rtz = 64,
    denorm_flush_to_zero = 64
))]
pub fn manual(#[spirv(storage_buffer, descriptor_set = 0, binding = 0)] data: &mut f64) {
    *data += 1.0;
}
