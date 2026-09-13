//! Preserve floating-point policies through linking.
//!
//! Before SPIR-T, convert intrinsic markers to operation decorations.
//! After SPIR-T, apply entry-point defaults to every reachable float width.
//! After module splitting and DCE, remove float-control capabilities
//! that the remaining code does not need.

use crate::custom_decorations::{CustomDecoration, MathFlagsDecoration};
use rspirv::dr::{Instruction, Module, Operand};
use rspirv::spirv::{Capability, ExecutionMode, Op};
use std::collections::{BTreeMap, HashMap, HashSet};

/// Decorate intrinsic results for `rust_math` and `fast_math` callers before SPIR-T.
/// Copy shared helpers to preserve intrinsic behavior for compatibility callers.
pub(super) fn resolve_operation_policies(module: &mut Module) {
    use rspirv::spirv::Decoration;
    // Record each private marker's flags before removing it.
    // Preserve user-authored decorations.
    let marked: HashMap<_, _> = MathFlagsDecoration::decode_all(module)
        .map(|(id, flags)| (id, flags.decode().0))
        .collect();
    MathFlagsDecoration::remove_all(module);
    if marked.is_empty() {
        return;
    }
    let explicit_entries: HashSet<_> = module
        .execution_modes
        .iter()
        .filter(|inst| {
            inst.operands.get(1) == Some(&Operand::ExecutionMode(ExecutionMode::FPFastMathDefault))
        })
        .map(|inst| inst.operands[0].unwrap_id_ref())
        .collect();
    if explicit_entries.is_empty() {
        return;
    }
    // Find functions reachable from explicit-policy and compatibility entry points.
    // Shared helpers can belong to both sets.
    let calls: HashMap<_, Vec<_>> = module
        .functions
        .iter()
        .map(|f| {
            (
                f.def_id().unwrap(),
                f.all_inst_iter()
                    .filter(|i| i.class.opcode == Op::FunctionCall)
                    .map(|i| i.operands[0].unwrap_id_ref())
                    .collect(),
            )
        })
        .collect();
    let reachable = |edges: &HashMap<u32, Vec<u32>>, mut pending: Vec<u32>| {
        let mut visited = HashSet::new();
        while let Some(id) = pending.pop() {
            if visited.insert(id)
                && let Some(neighbors) = edges.get(&id)
            {
                pending.extend(neighbors);
            }
        }
        visited
    };
    let policy_functions = reachable(&calls, explicit_entries.iter().copied().collect());
    let legacy_functions = reachable(
        &calls,
        module
            .entry_points
            .iter()
            .map(|e| e.operands[1].unwrap_id_ref())
            .filter(|id| !explicit_entries.contains(id))
            .collect(),
    );
    // Copy only helpers that contain marked operations or reach them through other helpers.
    // Other helpers can inherit the caller's policy.
    let marked_functions = module
        .functions
        .iter()
        .filter(|f| {
            f.all_inst_iter()
                .any(|i| i.result_id.is_some_and(|id| marked.contains_key(&id)))
        })
        .map(|f| f.def_id().unwrap())
        .collect();
    // Follow callers backward from functions that contain marked operations.
    let mut callers: HashMap<u32, Vec<u32>> = HashMap::new();
    for (&caller, callees) in &calls {
        for &callee in callees {
            callers.entry(callee).or_default().push(caller);
        }
    }
    let needs_policy = reachable(&callers, marked_functions);
    // Keep the original functions for compatibility callers.
    // Assign IDs to all copies before redirecting calls between them.
    let mut remap = HashMap::new();
    let mut clones = Vec::new();
    for f in &module.functions {
        let id = f.def_id().unwrap();
        if policy_functions.contains(&id)
            && legacy_functions.contains(&id)
            && needs_policy.contains(&id)
        {
            for inst in f.all_inst_iter() {
                if let Some(id) = inst.result_id {
                    remap.insert(id, super::id(module.header.as_mut().unwrap()));
                }
            }
            clones.push(f.clone());
        }
    }
    let rewrite = |inst: &mut Instruction| {
        for id in inst
            .result_id
            .iter_mut()
            .chain(&mut inst.result_type)
            .chain(inst.operands.iter_mut().filter_map(Operand::id_ref_any_mut))
        {
            if let Some(new) = remap.get(id) {
                *id = *new;
            }
        }
    };
    // Redirect `rust_math` and `fast_math` callers to the copied helpers.
    // Read flags with the original IDs. Decorate results with the rewritten IDs.
    let mut decorate = Vec::new();
    for f in module
        .functions
        .iter_mut()
        .filter(|f| {
            let id = f.def_id().unwrap();
            policy_functions.contains(&id) && !legacy_functions.contains(&id)
        })
        .chain(clones.iter_mut())
    {
        for inst in f.all_inst_iter_mut() {
            let flags = inst.result_id.and_then(|id| marked.get(&id)).copied();
            rewrite(inst);
            if let Some(flags) = flags {
                decorate.push(Instruction::new(
                    Op::Decorate,
                    None,
                    None,
                    vec![
                        Operand::IdRef(inst.result_id.unwrap()),
                        Operand::Decoration(Decoration::FPFastMathMode),
                        Operand::FPFastMathMode(rspirv::spirv::FPFastMathMode::from_bits_retain(
                            flags,
                        )),
                    ],
                ));
            }
        }
    }
    // Copy existing decorations and debug names.
    // Give function copies distinct names for diagnostics.
    for section in [&mut module.annotations, &mut module.debug_names] {
        let extra: Vec<_> = section
            .iter()
            .filter(|i| {
                i.operands
                    .first()
                    .and_then(|o| o.id_ref_any())
                    .is_some_and(|id| remap.contains_key(&id))
            })
            .cloned()
            .map(|mut i| {
                if i.class.opcode == Op::Name
                    && calls.contains_key(&i.operands[0].unwrap_id_ref())
                    && let Operand::LiteralString(name) = &mut i.operands[1]
                {
                    name.push_str(".math");
                }
                rewrite(&mut i);
                i
            })
            .collect();
        section.extend(extra);
    }
    module.functions.extend(clones);
    module.annotations.extend(decorate);
}

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
