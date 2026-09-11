//! Preserve floating-point policies through the linker pipeline.
//!
//! Before SPIR-T, resolve intrinsic markers into operation decorations and save
//! entry-point defaults outside the module. After SPIR-T, expand those defaults
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
    // Intrinsic lowering needs to know which entry points have explicit policies,
    // so resolve it before removing their FPFastMathDefault modes.
    resolve_operation_policies(module);
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

/// Keep legacy intrinsic lowering unchanged, even when a helper is shared with
/// a rust_math or fast_math entry point. Resolve private codegen markers before SPIR-T or
/// optimizers see the module; user-authored SPIR-V decorations are untouched.
fn resolve_operation_policies(module: &mut Module) {
    use rspirv::spirv::Decoration;
    // Consume only our private markers. Whether they become real decorations
    // depends on the calling entry points; legacy-only uses stay undecorated.
    let mut marked = HashMap::new();
    module.annotations.retain(|inst| {
        if inst.operands.get(1) == Some(&Operand::Decoration(Decoration::UserSemantic))
            && let Some(Operand::LiteralString(text)) = inst.operands.get(2)
            && let Some(flags) = text.strip_prefix("rust_gpu.math_flags:")
        {
            marked.insert(
                inst.operands[0].unwrap_id_ref(),
                flags.parse::<u32>().unwrap(),
            );
            return false;
        }
        true
    });
    if marked.is_empty() {
        return;
    }
    let explicit: HashSet<_> = module
        .execution_modes
        .iter()
        .filter(|inst| {
            inst.operands.get(1) == Some(&Operand::ExecutionMode(ExecutionMode::FPFastMathDefault))
        })
        .map(|inst| inst.operands[0].unwrap_id_ref())
        .collect();
    if explicit.is_empty() {
        return;
    }
    // Partition the call graph by reachability from explicit-policy and legacy
    // entry points. A shared helper can belong to both sets.
    let calls: HashMap<_, Vec<_>> = module
        .functions
        .iter()
        .map(|f| {
            (
                f.def.as_ref().unwrap().result_id.unwrap(),
                f.all_inst_iter()
                    .filter(|i| i.class.opcode == Op::FunctionCall)
                    .map(|i| i.operands[0].unwrap_id_ref())
                    .collect(),
            )
        })
        .collect();
    let reachable = |mut pending: Vec<u32>| {
        let mut visited = HashSet::new();
        while let Some(id) = pending.pop() {
            if visited.insert(id)
                && let Some(callees) = calls.get(&id)
            {
                pending.extend(callees);
            }
        }
        visited
    };
    let opted = reachable(explicit.iter().copied().collect());
    let legacy = reachable(
        module
            .entry_points
            .iter()
            .map(|e| e.operands[1].unwrap_id_ref())
            .filter(|id| !explicit.contains(id))
            .collect(),
    );
    // Only specialize helpers that contain, or transitively call, marked ops.
    // Ordinary helpers can inherit either entry-point policy without cloning.
    let mut needs_policy: HashSet<_> = module
        .functions
        .iter()
        .filter(|f| {
            f.all_inst_iter()
                .any(|i| i.result_id.is_some_and(|id| marked.contains_key(&id)))
        })
        .map(|f| f.def.as_ref().unwrap().result_id.unwrap())
        .collect();
    // Walk backward from marked functions instead of rescanning every call.
    let mut callers: HashMap<u32, Vec<u32>> = HashMap::new();
    for (&caller, callees) in &calls {
        for &callee in callees {
            callers.entry(callee).or_default().push(caller);
        }
    }
    let mut pending: Vec<u32> = needs_policy.iter().copied().collect();
    while let Some(callee) = pending.pop() {
        if let Some(parents) = callers.get(&callee) {
            for &caller in parents {
                if needs_policy.insert(caller) {
                    pending.push(caller);
                }
            }
        }
    }
    // Keep the original shared functions for legacy callers, and allocate fresh
    // IDs for copies used by explicit-policy callers. Allocate every ID first so
    // calls between cloned functions can be redirected with the same map.
    let mut remap = HashMap::new();
    let mut clones = Vec::new();
    for f in &module.functions {
        let id = f.def.as_ref().unwrap().result_id.unwrap();
        if opted.contains(&id) && legacy.contains(&id) && needs_policy.contains(&id) {
            for inst in f.all_inst_iter() {
                if let Some(id) = inst.result_id {
                    let bound = &mut module.header.as_mut().unwrap().bound;
                    remap.insert(id, *bound);
                    *bound += 1;
                }
            }
            clones.push(f.clone());
        }
    }
    let rewrite = |inst: &mut Instruction| {
        if let Some(id) = &mut inst.result_id
            && let Some(new) = remap.get(id)
        {
            *id = *new;
        }
        if let Some(id) = &mut inst.result_type
            && let Some(new) = remap.get(id)
        {
            *id = *new;
        }
        for operand in &mut inst.operands {
            if let Operand::IdRef(id) | Operand::IdScope(id) | Operand::IdMemorySemantics(id) =
                operand
                && let Some(new) = remap.get(id)
            {
                *id = *new;
            }
        }
    };
    // Redirect explicit-policy calls to the copies and materialize intrinsic
    // flags on their arithmetic results. Read markers using the original IDs
    // before rewriting them; the emitted decorations use the final IDs.
    let mut decorate = Vec::new();
    for f in module
        .functions
        .iter_mut()
        .filter(|f| {
            let id = f.def.as_ref().unwrap().result_id.unwrap();
            opted.contains(&id) && !legacy.contains(&id)
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
    // Cloned instructions also need their existing decorations and debug names.
    // Retain the originals and distinguish cloned function names in diagnostics.
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
