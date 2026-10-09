// build-pass
// only-vulkan1.2
// compile-flags: -C target-feature=+Float64
// compile-flags: -C llvm-args=--disassemble
// compile-flags: -C debuginfo=0
// normalize-stderr-test "\n\W*OpLine .*" -> ""
// normalize-stderr-test "\n\W*OpSource .*" -> ""
// normalize-stderr-test "\n\W*%\d+ = OpString .*" -> ""

#![feature(float_algebraic)]

use spirv_std::spirv;

// The snapshot's fast-math mask 458767 is NotNaN | NotInf | NSZ | AllowRecip |
// AllowContract | AllowReassoc | AllowTransform.
// A zero mask disables these permissions.
// Each entry point must retain its policy when it calls the same function.
// The linker must copy this helper to preserve `compat_math` behavior.
// The policy must include `f64`, which occurs only in the callee's body.
#[inline(never)]
fn shared(x: f32) -> f32 {
    (x as f64).algebraic_add(1.0) as f32
}

#[spirv(compute(threads(1), compat_math))]
pub fn legacy(#[spirv(storage_buffer, descriptor_set = 0, binding = 0)] data: &mut f32) {
    *data = shared(*data);
}

#[spirv(compute(threads(1), rust_math))]
pub fn strict(#[spirv(storage_buffer, descriptor_set = 0, binding = 0)] data: &mut f32) {
    *data = shared(*data);
}

#[spirv(compute(threads(1), fast_math))]
pub fn fast(#[spirv(storage_buffer, descriptor_set = 0, binding = 0)] data: &mut f32) {
    *data = shared(*data);
}
