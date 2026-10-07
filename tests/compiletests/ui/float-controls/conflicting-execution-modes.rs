// build-fail
// only-vulkan1.2

use spirv_std::spirv;

// Check both policies for conflicts at each float width (16, 32, and 64 bits).
#[spirv(compute(
    threads(1),
    rust_math,
    signed_zero_inf_nan_preserve = 32,
    rounding_mode_rtz = 64,
    denorm_flush_to_zero = 16,
    contraction_off
))]
pub fn strict_policy() {}

// The order of attributes must not affect conflict detection.
#[spirv(compute(
    threads(1),
    signed_zero_inf_nan_preserve = 64,
    rounding_mode_rtz = 16,
    denorm_preserve = 32,
    contraction_off,
    fast_math
))]
pub fn fast_policy() {}
