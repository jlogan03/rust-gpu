// build-pass
// only-vulkan1.2
// compile-flags: -C llvm-args=--disassemble
// normalize-stderr-test "\n\W*OpLine .*" -> ""
// normalize-stderr-test "\n\W*OpSource .*" -> ""
// normalize-stderr-test "\n\W*%\d+ = OpString .*" -> ""

#![feature(float_algebraic, core_intrinsics)]
#![allow(internal_features)]

use core::intrinsics::fadd_fast;
use spirv_std::{glam::Vec4, num_traits::Float, spirv};

#[spirv(fragment(rust_math))]
pub fn scalar(a: f32, b: f32, output: &mut f32) {
    let x = a
        .algebraic_add(b)
        .algebraic_sub(b)
        .algebraic_mul(b)
        .algebraic_div(b)
        .algebraic_rem(b);
    let x = unsafe { fadd_fast(x, b) };
    // `mul_add` emits `Fma` (GLSL.std.450 instruction 50).
    *output = x.mul_add(b, a) * b + a;
}

// The snapshot's fast-math mask 458764 is NSZ | AllowRecip | AllowContract |
// AllowReassoc | AllowTransform.
// A mix of decorated and undecorated lanes must override this default with a zero mask.
#[spirv(fragment(fast_math))]
pub fn vectors(
    a: Vec4,
    b: Vec4,
    matching: &mut Vec4,
    differing: &mut Vec4,
    mixed: &mut Vec4,
    mixed_reverse: &mut Vec4,
    scaled: &mut Vec4,
    ordinary: &mut Vec4,
) {
    let add = f32::algebraic_add;
    let mul = f32::algebraic_mul;
    // The vector must retain exactly the permissions common to all lanes.
    *matching = Vec4::new(add(a.x, b.x), add(a.y, b.y), add(a.z, b.z), add(a.w, b.w));
    let fast_x = unsafe { fadd_fast(a.x, b.x) };
    *differing = Vec4::new(fast_x, add(a.y, b.y), add(a.z, b.z), add(a.w, b.w));
    // Either ordering of decorated and undecorated lanes must produce a zero mask.
    *mixed = Vec4::new(add(a.x, b.x), a.y + b.y, a.z + b.z, a.w + b.w);
    *mixed_reverse = Vec4::new(a.x + b.x, add(a.y, b.y), a.z + b.z, a.w + b.w);
    // `OpVectorTimesScalar` must also retain lane permissions.
    let b_x = b.x;
    *scaled = Vec4::new(mul(a.x, b_x), mul(a.y, b_x), mul(a.z, b_x), mul(a.w, b_x));
    // When no lane has a decoration, the vector must inherit the entry-point default.
    *ordinary = Vec4::new(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
