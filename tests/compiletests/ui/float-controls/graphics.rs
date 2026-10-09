// build-pass
// only-vulkan1.2
// compile-flags: -C llvm-args=--disassemble-globals
// compile-flags: -C debuginfo=0
// normalize-stderr-test "\n\W*OpSource .*" -> ""
// normalize-stderr-test "\n\W*%\d+ = OpString .*" -> ""

use spirv_std::spirv;

// Floating-point policies also apply to fragment entry points.
// The snapshot's fast-math mask 458767 is NotNaN | NotInf | NSZ | AllowRecip |
// AllowContract | AllowReassoc | AllowTransform.
#[spirv(fragment(fast_math))]
pub fn fast(#[spirv(location = 0)] output: &mut f32) {
    *output = 0.0;
}
