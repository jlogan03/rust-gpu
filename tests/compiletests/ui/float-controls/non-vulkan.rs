// build-fail
// only-spv1.3

use spirv_std::spirv;

#[spirv(fragment(fast_math))]
pub fn main() {}

#[spirv(compute(threads(1), rust_math))]
pub fn strict() {}
