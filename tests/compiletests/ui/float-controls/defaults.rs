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

#[inline(never)]
fn shared(a: f32, b: f32) -> f32 {
    (a.algebraic_add(b) + b) - a
}

#[spirv(compute(threads(1), compat_math))]
pub fn legacy(#[spirv(storage_buffer, descriptor_set = 0, binding = 0)] data: &mut [f32; 2]) {
    data[0] = shared(data[0], data[1]);
}

#[spirv(compute(threads(1)))]
pub fn strict(#[spirv(storage_buffer, descriptor_set = 0, binding = 0)] data: &mut [f32; 2]) {
    data[0] = shared(data[0], data[1]);
    data[1] = data[0].algebraic_add(data[1]);
}

#[spirv(compute(threads(1), fast_math))]
pub fn fast(#[spirv(storage_buffer, descriptor_set = 0, binding = 0)] data: &mut [f32; 2]) {
    data[0] = shared(data[0], data[1]);
}

#[spirv(compute(threads(1), rust_math))]
pub fn double(#[spirv(storage_buffer, descriptor_set = 0, binding = 0)] data: &mut [f64; 2]) {
    data[0] = (data[0] + data[1]) - data[0];
}
