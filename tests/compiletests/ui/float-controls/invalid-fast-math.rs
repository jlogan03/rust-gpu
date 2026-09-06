// build-fail

use spirv_std::spirv;

#[spirv(compute(threads(1), fast_math, fast_math))]
pub fn duplicate() {}

#[spirv(fragment(fast_math = true))]
pub fn value() {}

#[spirv(fragment(fast_math()))]
pub fn arguments() {}

#[spirv(compute(threads(1), rust_math, fast_math))]
pub fn conflicting() {}

#[spirv(fragment(rust_math, rust_math))]
pub fn duplicate_rust() {}

#[spirv(fragment(rust_math = true))]
pub fn rust_value() {}
