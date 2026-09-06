// build-pass
// only-vulkan1.2
// compile-flags: -C llvm-args=--disassemble
// normalize-stderr-test "\n\W*OpLine .*" -> ""
// normalize-stderr-test "\n\W*OpSource .*" -> ""
// normalize-stderr-test "\n\W*%\d+ = OpString .*" -> ""

#![feature(float_algebraic, core_intrinsics)]
#![allow(internal_features)]

use spirv_std::{num_traits::Float, spirv};

#[spirv(compute(threads(1), rust_math))]
pub fn main(#[spirv(storage_buffer, descriptor_set = 0, binding = 0)] data: &mut [f32; 8]) {
    let (a, b) = (data[0], data[1]);
    data[0] = a.algebraic_add(b);
    data[1] = a.algebraic_sub(b);
    data[2] = a.algebraic_mul(b);
    data[3] = a.algebraic_div(b);
    data[4] = a.algebraic_rem(b);
    data[5] = unsafe { core::intrinsics::fadd_fast(a, b) };
    data[6] = a.mul_add(b, a);
    data[7] = a * b + a;
}
