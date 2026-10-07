//! Preserve floating-point policies through linking.
//!
//! Save entry-point defaults outside the module before SPIR-T runs.
//! After SPIR-T, apply those defaults to every reachable float width.
//! After module splitting and DCE, remove float-control capabilities
//! that the remaining code does not need.

use rspirv::dr::{Instruction, Module, Operand};
use rspirv::spirv::{Capability, ExecutionMode, ExecutionModel, Op};
use std::collections::{BTreeMap, HashMap, HashSet};

pub(super) struct Policy {
    entry_name: String,
    execution_model: ExecutionModel,
    // Codegen emits zero for `rust_math` and `ALGEBRAIC_MATH_FLAGS` for `fast_math`.
    flags: u32,
}

// SPIR-T 0.4 cannot lower execution modes with ID operands.
// Its structurization and layout passes do not transform floating-point arithmetic.
// Save the policy with the entry-point name across these passes.
// Emit the modes before SPIRV-Tools optimizes arithmetic or validates the module.
pub(super) fn take_policies(module: &mut Module) -> Vec<Policy> {
    let mut policies = Vec::new();
    module.execution_modes.retain(|mode| {
        if mode.operands.get(1) != Some(&Operand::ExecutionMode(ExecutionMode::FPFastMathDefault)) {
            return true;
        }
        // SPIR-T can change IDs. Identify the entry point by its name and execution model.
        // Read the flag constant while the module still contains it.
        let entry = module
            .entry_points
            .iter()
            .find(|entry| entry.operands[1] == mode.operands[0])
            .unwrap();
        let constant = module
            .types_global_values
            .iter()
            .find(|inst| inst.result_id == Some(mode.operands[3].unwrap_id_ref()))
            .unwrap();
        let flags = if constant.class.opcode == Op::ConstantNull {
            0
        } else {
            constant.operands[0].unwrap_literal_bit32()
        };
        policies.push(Policy {
            entry_name: entry.operands[2].unwrap_literal_string().to_owned(),
            execution_model: entry.operands[0].unwrap_execution_model(),
            flags,
        });
        false
    });
    policies
}

/// Emit final execution modes directly from the policies saved before SPIR-T.
pub(super) fn restore_policies(module: &mut Module, policies: Vec<Policy>) {
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
    let mut builder = rspirv::dr::Builder::new_from_module(std::mem::take(module));
    let mut capabilities = Vec::new();
    for policy in policies {
        // Find the entry point's ID in the module that SPIR-T produced.
        let entry = builder
            .module_ref()
            .entry_points
            .iter()
            .find(|entry| {
                entry.operands[2].unwrap_literal_string() == policy.entry_name
                    && entry.operands[0] == Operand::ExecutionModel(policy.execution_model)
            })
            .unwrap()
            .operands[1]
            .unwrap_id_ref();
        let fast = policy.flags != 0;
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
        if widths.is_empty() {
            continue;
        }
        // `FPFastMathDefault` references a constant ID.
        // Create the constant only if there are final modes to emit.
        // Reuse the constant for all float widths of this entry point.
        let uint = builder.type_int(32, 0);
        let flags = builder.constant_bit32(uint, policy.flags);
        for (width, ty) in widths {
            builder.module_mut().execution_modes.push(Instruction::new(
                Op::ExecutionModeId,
                None,
                None,
                vec![
                    Operand::IdRef(entry),
                    Operand::ExecutionMode(ExecutionMode::FPFastMathDefault),
                    Operand::IdRef(ty),
                    Operand::IdRef(flags),
                ],
            ));
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
                if builder
                    .module_ref()
                    .execution_modes
                    .iter()
                    .any(|inst| inst.operands == operands)
                {
                    continue;
                }
                builder.module_mut().execution_modes.push(Instruction::new(
                    Op::ExecutionMode,
                    None,
                    None,
                    operands,
                ));
            }
        }
    }
    *module = builder.module();
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
