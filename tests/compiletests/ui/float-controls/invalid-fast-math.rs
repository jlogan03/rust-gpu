// build-fail

use spirv_std::spirv;

#[spirv(fragment(fast_math, fast_math))]
pub fn duplicate() {}

#[spirv(fragment(rust_math = true))]
pub fn value() {}

#[spirv(fragment(compat_math()))]
pub fn arguments() {}

#[spirv(fragment(rust_math, fast_math))]
pub fn conflicting() {}
