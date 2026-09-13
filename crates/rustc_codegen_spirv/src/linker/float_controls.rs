//! Preserve floating-point policies through the linker pipeline.
//!
//! Before SPIR-T, save entry-point defaults outside the module.
//! After SPIR-T, expand those defaults
//! to the reachable float widths. After module splitting and DCE, discard any
//! float-control capabilities that the surviving code no longer needs.

use rspirv::dr::{Instruction, Module, Operand};
use rspirv::spirv::{Capability, ExecutionMode, ExecutionModel, Op};
use std::collections::{BTreeMap, HashMap, HashSet};

// SPIR-T 0.4 cannot lower execution modes with ID operands. Its current
// structurization/layout passes do not transform floating-point arithmetic.
// Carry the entry policy by name across that pipeline, then materialize modes
// before SPIRV-Tools performs any arithmetic optimization or validation.
pub(super) fn take_policies(module: &mut Module) -> Vec<(String, ExecutionModel, u32)> {
    let mut policies = Vec::new();
    module.execution_modes.retain(|mode| {
        if mode.operands.get(1) != Some(&Operand::ExecutionMode(ExecutionMode::FPFastMathDefault)) {
            return true;
        }
        // SPIR-T can change IDs. Preserve the entry's name and execution model
        // as its identity, and decode the flag constant while it is still present.
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
        policies.push((
            entry.operands[2].unwrap_literal_string().to_owned(),
            entry.operands[0].unwrap_execution_model(),
            flags,
        ));
        false
    });
    policies
}

/// Emit final execution modes directly from the policies saved before SPIR-T.
pub(super) fn restore_policies(module: &mut Module, policies: Vec<(String, ExecutionModel, u32)>) {
    if policies.is_empty() {
        return;
    }
    // Follow value/type dependencies and calls, not just the entry function's
    // result types. This includes floats loaded through pointers and composites.
    let mut edges: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut floats = HashMap::new();
    for inst in module.all_inst_iter() {
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
    // A function's signature alone misses types used only in its body. Connect
    // the function to all its instructions, including stores and calls, so the
    // traversal below includes every reachable use of floating-point values.
    for function in &module.functions {
        let deps = edges
            .entry(function.def.as_ref().unwrap().result_id.unwrap())
            .or_default();
        for inst in function.all_inst_iter() {
            deps.extend(inst.result_id);
            deps.extend(inst.result_type);
            deps.extend(inst.operands.iter().filter_map(|op| match op {
                Operand::IdRef(id) => Some(*id),
                _ => None,
            }));
        }
    }
    let mut builder = rspirv::dr::Builder::new_from_module(std::mem::take(module));
    let mut capabilities = Vec::new();
    for (name, model, flags) in policies {
        // Recover the entry's current ID after SPIR-T has rewritten the module.
        let entry = builder
            .module_ref()
            .entry_points
            .iter()
            .find(|entry| {
                entry.operands[2].unwrap_literal_string() == name
                    && entry.operands[0] == Operand::ExecutionModel(model)
            })
            .unwrap()
            .operands[1]
            .unwrap_id_ref();
        let fast = flags != 0;
        // Gather one scalar type per reachable float width. BTreeMap keeps the
        // emitted modes ordered by width regardless of dependency traversal order.
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
        // FPFastMathDefault references a constant ID. Create it only when there
        // are final modes to emit, and reuse it for all widths of this entry.
        let uint = builder.type_int(32, 0);
        let flags = builder.constant_bit32(uint, flags);
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
            // Denormal handling and rounding are entry-point settings, separate
            // from operation-level fast-math permissions. Both policies use RTE;
            // rust_math preserves subnormals, while fast_math flushes them.
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
                // Matching source attributes already provide this mode. Reuse
                // them, comparing the width as well as the entry and mode.
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
    // Declare the requirements of the modes just emitted without duplicating
    // declarations already present in the linked module.
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

/// After splitting and DCE, allow compatibility-only modules to shed requirements
/// introduced by entry points that are no longer present.
pub(super) fn remove_unused_capabilities(module: &mut Module) {
    let has_mode = |mode| {
        module
            .execution_modes
            .iter()
            .any(|inst| inst.operands[1] == Operand::ExecutionMode(mode))
    };
    // Either an entry-point default or an operation override still needs
    // FloatControls2; checking execution modes alone would miss the latter.
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
    // Consumers such as Naga reject unsupported extensions even without any
    // corresponding capabilities. The original extension also covers legacy
    // RTZ and signed-zero/Inf/NaN modes, not just the modes our policies emit.
    module
        .extensions
        .retain(|inst| match inst.operands[0].unwrap_literal_string() {
            "SPV_KHR_float_controls" => preserve || flush || rte || rtz || preserve_special,
            "SPV_KHR_float_controls2" => controls2,
            _ => true,
        });
}
