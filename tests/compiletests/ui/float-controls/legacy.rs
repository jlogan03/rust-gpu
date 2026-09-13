// build-pass
// only-vulkan1.2
// compile-flags: -C llvm-args=--disassemble
// compile-flags: -C debuginfo=0
// normalize-stderr-test "\n\W*OpLine .*" -> ""
// normalize-stderr-test "\n\W*OpSource .*" -> ""
// normalize-stderr-test "\n\W*%\d+ = OpString .*" -> ""

#![feature(float_algebraic, core_intrinsics)]
#![allow(internal_features)]
use spirv_std::spirv;

#[spirv(compute(threads(1)))]
pub fn legacy(#[spirv(storage_buffer, descriptor_set = 0, binding = 0)] data: &mut [f32; 2]) {
    data[0] = data[0].algebraic_add(data[1]);
    data[1] = unsafe { core::intrinsics::fmul_fast(data[0], data[1]) };
}

// Opting in without using floats must not introduce float-control requirements.
#[spirv(compute(threads(1), rust_math))]
pub fn integer(#[spirv(storage_buffer, descriptor_set = 0, binding = 0)] data: &mut u32) {
    *data += 1;
}
