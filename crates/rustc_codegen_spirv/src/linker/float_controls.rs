//! Preserve floating-point policies through linking.
//!
//! After SPIR-T, apply entry-point defaults to every reachable float width.
//! After module splitting and DCE, remove float-control capabilities
//! that the remaining code does not need.

use rspirv::dr::{Instruction, Module, Operand};
use rspirv::spirv::{Capability, ExecutionMode, Op};
use std::collections::{BTreeMap, HashMap, HashSet};

/// Apply each entry-point policy to every float width that its code uses after SPIR-T.
pub(super) fn expand_policies(module: &mut Module) {
    let policies: Vec<_> = module
        .execution_modes
        .extract_if(.., |mode| {
            mode.operands.get(1) == Some(&Operand::ExecutionMode(ExecutionMode::FPFastMathDefault))
        })
        .collect();
    if policies.is_empty() {
        return;
    }
    // Follow value dependencies, type dependencies, and calls to find reachable float types.
    // Include floats loaded through pointers and composites, beyond the entry function's result types.
    let mut edges: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut floats = HashMap::new();
    for inst in &module.types_global_values {
        if let Some(id) = inst.result_id {
            let deps = edges.entry(id).or_default();
            deps.extend(inst.result_type);
            deps.extend(inst.operands.iter().filter_map(|op| match op {
                Operand::IdRef(id) => Some(*id),
                _ => None,
            }));
            if inst.class.opcode == Op::TypeFloat {
                floats.insert(id, inst.operands[0].unwrap_literal_bit32());
            }
        }
    }
    // A function's signature omits types used only in its body.
    // Connect each function to the types, globals, and callees that its instructions use.
    // Local values need no separate edges because the function includes their dependencies.
    for function in &module.functions {
        let deps = edges
            .entry(function.def.as_ref().unwrap().result_id.unwrap())
            .or_default();
        for inst in function.all_inst_iter() {
            deps.extend(inst.result_type);
            deps.extend(inst.operands.iter().filter_map(|op| match op {
                Operand::IdRef(id) => Some(*id),
                _ => None,
            }));
        }
    }
    let mut capabilities = Vec::new();
    for policy in policies {
        let entry = policy.operands[0].unwrap_id_ref();
        let flags = policy.operands[3].unwrap_id_ref();
        let constant = module
            .types_global_values
            .iter()
            .find(|inst| inst.result_id == Some(flags))
            .unwrap();
        // Codegen emits zero for `rust_math` and `ALGEBRAIC_MATH_FLAGS` for `fast_math`.
        let fast = constant.class.opcode != Op::ConstantNull
            && constant.operands[0].unwrap_literal_bit32() != 0;
        // Collect one scalar type for each reachable float width.
        // `BTreeMap` orders the emitted modes by width, regardless of the order of traversal.
        let mut pending = vec![entry];
        let mut visited = HashSet::new();
        let mut widths = BTreeMap::new();
        while let Some(id) = pending.pop() {
            if !visited.insert(id) {
                continue;
            }
            if let Some(&width) = floats.get(&id) {
                widths.insert(width, id);
            }
            if let Some(deps) = edges.get(&id) {
                pending.extend(deps);
            }
        }
        for (width, ty) in widths {
            let mut mode = policy.clone();
            mode.operands[2] = Operand::IdRef(ty);
            module.execution_modes.push(mode);
            // Denormal handling and rounding apply to the entry point.
            // Fast-math permissions apply to individual operations.
            // Both policies round to nearest, with ties to even (RTE).
            // `rust_math` preserves subnormals. `fast_math` flushes subnormals to zero.
            let (denorm, capability) = if fast {
                (
                    ExecutionMode::DenormFlushToZero,
                    Capability::DenormFlushToZero,
                )
            } else {
                (ExecutionMode::DenormPreserve, Capability::DenormPreserve)
            };
            capabilities.extend([capability, Capability::RoundingModeRTE]);
            for execution_mode in [denorm, ExecutionMode::RoundingModeRTE] {
                // Reuse modes from source attributes when the entry point, mode, and width match.
                let operands = vec![
                    Operand::IdRef(entry),
                    Operand::ExecutionMode(execution_mode),
                    Operand::LiteralBit32(width),
                ];
                if module
                    .execution_modes
                    .iter()
                    .any(|inst| inst.operands == operands)
                {
                    continue;
                }
                module.execution_modes.push(Instruction::new(
                    Op::ExecutionMode,
                    None,
                    None,
                    operands,
                ));
            }
        }
    }
    // Declare the requirements of the emitted modes.
    // Do not duplicate declarations in the linked module.
    if !capabilities.is_empty()
        && !module
            .extensions
            .iter()
            .any(|inst| inst.operands == [Operand::LiteralString("SPV_KHR_float_controls".into())])
    {
        module.extensions.push(Instruction::new(
            Op::Extension,
            None,
            None,
            vec![Operand::LiteralString("SPV_KHR_float_controls".into())],
        ));
    }
    for capability in capabilities {
        let operands = vec![Operand::Capability(capability)];
        if !module
            .capabilities
            .iter()
            .any(|inst| inst.operands == operands)
        {
            module
                .capabilities
                .push(Instruction::new(Op::Capability, None, None, operands));
        }
    }
}

/// After splitting and DCE, remove requirements from entry points that no longer exist.
/// This lets modules with only compatibility entry points avoid those requirements.
pub(super) fn remove_unused_capabilities(module: &mut Module) {
    let has_mode = |mode| {
        module
            .execution_modes
            .iter()
            .any(|inst| inst.operands[1] == Operand::ExecutionMode(mode))
    };
    // Retain `FloatControls2` for entry-point defaults or operation overrides.
    let controls2 = has_mode(ExecutionMode::FPFastMathDefault)
        || module.annotations.iter().any(|inst| {
            inst.operands.get(1)
                == Some(&Operand::Decoration(
                    rspirv::spirv::Decoration::FPFastMathMode,
                ))
        });
    let preserve = has_mode(ExecutionMode::DenormPreserve);
    let flush = has_mode(ExecutionMode::DenormFlushToZero);
    let rte = has_mode(ExecutionMode::RoundingModeRTE);
    let rtz = has_mode(ExecutionMode::RoundingModeRTZ);
    let preserve_special = has_mode(ExecutionMode::SignedZeroInfNanPreserve);
    module
        .capabilities
        .retain(|inst| match inst.operands[0].unwrap_capability() {
            Capability::FloatControls2 => controls2,
            Capability::DenormPreserve => preserve,
            Capability::DenormFlushToZero => flush,
            Capability::RoundingModeRTE => rte,
            Capability::RoundingModeRTZ => rtz,
            Capability::SignedZeroInfNanPreserve => preserve_special,
            _ => true,
        });
    // Consumers such as Naga reject unsupported extensions even without corresponding capabilities.
    // `SPV_KHR_float_controls` also covers the legacy RTZ and signed-zero/Inf/NaN modes.
    // Retain the extension if the module uses these modes.
    module
        .extensions
        .retain(|inst| match inst.operands[0].unwrap_literal_string() {
            "SPV_KHR_float_controls" => preserve || flush || rte || rtz || preserve_special,
            "SPV_KHR_float_controls2" => controls2,
            _ => true,
        });
}
