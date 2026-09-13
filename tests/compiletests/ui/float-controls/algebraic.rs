// build-pass
// only-vulkan1.2
// compile-flags: -C llvm-args=--disassemble
// normalize-stderr-test "\n\W*OpLine .*" -> ""
// normalize-stderr-test "\n\W*OpSource .*" -> ""
// normalize-stderr-test "\n\W*%\d+ = OpString .*" -> ""

#![feature(float_algebraic, core_intrinsics)]
#![allow(internal_features)]

use spirv_std::{glam::Vec4, num_traits::Float, spirv};

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

#[spirv(fragment(rust_math))]
pub fn vectors(
    a: Vec4,
    b: Vec4,
    scale: f32,
    matching: &mut Vec4,
    differing: &mut Vec4,
    mixed: &mut Vec4,
    mixed_reverse: &mut Vec4,
    scaled: &mut Vec4,
    ordinary: &mut Vec4,
) {
    // Matching flags survive on the vector OpFAdd.
    *matching = Vec4::new(
        a.x.algebraic_add(b.x),
        a.y.algebraic_add(b.y),
        a.z.algebraic_add(b.z),
        a.w.algebraic_add(b.w),
    );
    // Different explicit flags retain only their common permissions.
    *differing = Vec4::new(
        unsafe { core::intrinsics::fadd_fast(a.x, b.x) },
        a.y.algebraic_add(b.y),
        a.z.algebraic_add(b.z),
        a.w.algebraic_add(b.w),
    );
    // Either ordering of decorated and undecorated lanes must stay strict.
    *mixed = Vec4::new(a.x.algebraic_add(b.x), a.y + b.y, a.z + b.z, a.w + b.w);
    *mixed_reverse = Vec4::new(a.x + b.x, a.y.algebraic_add(b.y), a.z + b.z, a.w + b.w);
    // The OpVectorTimesScalar shortcut must preserve flags too.
    *scaled = Vec4::new(
        a.x.algebraic_mul(scale),
        a.y.algebraic_mul(scale),
        a.z.algebraic_mul(scale),
        a.w.algebraic_mul(scale),
    );
    // Wholly undecorated vectors continue to inherit the entry-point default.
    *ordinary = Vec4::new(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

#[spirv(fragment(fast_math))]
pub fn fast_vectors(
    a: Vec4,
    b: Vec4,
    scale: f32,
    mixed_result: &mut Vec4,
    scaled: &mut Vec4,
    ordinary: &mut Vec4,
) {
    // A mixed vector's conservative zero mask must not become an absent
    // decoration, which would restore the entry-point fast-math permissions.
    *mixed_result = Vec4::new(a.x.algebraic_add(b.x), a.y + b.y, a.z + b.z, a.w + b.w);
    *scaled = Vec4::new(
        a.x.algebraic_mul(scale),
        a.y.algebraic_mul(scale),
        a.z.algebraic_mul(scale),
        a.w.algebraic_mul(scale),
    );
    *ordinary = Vec4::new(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
