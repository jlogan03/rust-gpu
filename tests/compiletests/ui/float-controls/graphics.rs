// build-pass
// only-vulkan1.2
// compile-flags: -C llvm-args=--disassemble-globals
// compile-flags: -C debuginfo=0
// normalize-stderr-test "\n\W*OpSource .*" -> ""
// normalize-stderr-test "\n\W*%\d+ = OpString .*" -> ""

use spirv_std::spirv;

// Fragment entry points must receive the same policy settings as compute entry points.
// The snapshot's fast-math mask 196620 is NSZ | AllowRecip | AllowContract | AllowReassoc.
// A zero mask disables these permissions.
#[spirv(fragment(rust_math))]
pub fn strict(#[spirv(location = 0)] input: f32, #[spirv(location = 0)] output: &mut f32) {
    *output = (input + 1.0) - input;
}

#[spirv(fragment(fast_math))]
pub fn fast(#[spirv(location = 0)] input: f32, #[spirv(location = 0)] output: &mut f32) {
    *output = (input + 1.0) - input;
}
