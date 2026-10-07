//! SPIR-T passes related to debuginfo.

use crate::custom_insts::{self, CustomInst, CustomOp};
use rustc_data_structures::fx::FxIndexSet;
use smallvec::SmallVec;
use spirt::transform::{InnerInPlaceTransform, Transformer};
use spirt::visit::InnerVisit;
use spirt::{
    Attr, AttrSetDef, ConstKind, Context, DataInstKind, DbgSrcLoc, InternedStr, Module, Node,
    NodeKind, OrdAssertEq, Value, spv,
};

/// Replace our custom extended instruction debuginfo with standard SPIR-V ones.
//
// FIXME(eddyb) also handle `SrcLocDecoration`s (when `rspirv` isn't used on the
// SPIR-V output of `spirt::spv::lift`, as it's lossy wrt `OpLine`).
pub fn convert_custom_debuginfo_to_spv(module: &mut Module) {
    let cx = &module.cx();

    // FIXME(eddyb) reuse this collection work in some kind of "pass manager".
    let all_funcs = {
        let mut collector = super::ReachableUseCollector {
            cx,
            module,

            seen_types: FxIndexSet::default(),
            seen_consts: FxIndexSet::default(),
            seen_global_vars: FxIndexSet::default(),
            seen_funcs: FxIndexSet::default(),
        };
        for (export_key, &exportee) in &module.exports {
            export_key.inner_visit_with(&mut collector);
            exportee.inner_visit_with(&mut collector);
        }
        collector.seen_funcs
    };

    let mut transformer = CustomDebuginfoToSpv {
        cx,
        wk: &super::SpvSpecWithExtras::get().well_known,
        custom_ext_inst_set: cx.intern(&custom_insts::CUSTOM_EXT_INST_SET[..]),
    };
    for func in all_funcs {
        transformer.in_place_transform_func_decl(&mut module.funcs[func]);
    }
}

struct CustomDebuginfoToSpv<'a> {
    cx: &'a Context,
    wk: &'static super::SpvWellKnownWithExtras,

    /// Interned name for our custom "extended instruction set"
    /// (see `crate::custom_insts` for more details).
    custom_ext_inst_set: InternedStr,
}

impl Transformer for CustomDebuginfoToSpv<'_> {
    fn in_place_transform_node_def(
        &mut self,
        mut func_at_control_node: spirt::func_at::FuncAtMut<'_, Node>,
    ) {
        // HACK(eddyb) this relies on the fact that `NodeKind::Block` maps
        // to one original SPIR-V block, which may not necessarily be true, and
        // steps should be taken elsewhere to explicitly unset debuginfo, instead
        // of relying on the end of a SPIR-V block implicitly unsetting it all.
        // NOTE(eddyb) allowing debuginfo to apply *outside* of a `Block` could
        // be useful in allowing *some* structured control-flow to have debuginfo,
        // but that would likely require more work on the SPIR-T side.
        if let NodeKind::Block { mut insts } = func_at_control_node.reborrow().def().kind {
            let mut current_src_loc = None;

            // HACK(eddyb) buffering the `DataInst`s to remove from this block,
            // as iterating and modifying a list at the same time isn't supported.
            let mut insts_to_remove = SmallVec::<[_; 8]>::new();

            let mut func_at_inst_iter = func_at_control_node.reborrow().at(insts).into_iter();
            while let Some(func_at_inst) = func_at_inst_iter.next() {
                let inst = func_at_inst.position;
                let data_inst_def = func_at_inst.def();

                // FIXME(eddyb) deduplicate with `spirt_passes::diagnostics`.
                if let DataInstKind::SpvExtInst {
                    ext_set,
                    inst: ext_inst,
                } = data_inst_def.kind
                    && ext_set == self.custom_ext_inst_set
                {
                    let custom_op = CustomOp::decode(ext_inst);
                    match custom_op.with_operands(&data_inst_def.inputs) {
                        CustomInst::SetDebugSrcLoc {
                            file,
                            line_start: line,
                            line_end: _,
                            col_start: col,
                            col_end: _,
                        } => {
                            let const_kind = |v: Value| match v {
                                Value::Const(ct) => &self.cx[ct].kind,
                                _ => unreachable!(),
                            };
                            let const_str = |v: Value| match const_kind(v) {
                                &ConstKind::SpvStringLiteralForExtInst(s) => s,
                                _ => unreachable!(),
                            };
                            let const_u32 = |v: Value| match const_kind(v) {
                                ConstKind::SpvInst {
                                    spv_inst_and_const_inputs,
                                } => {
                                    let (spv_inst, _const_inputs) = &**spv_inst_and_const_inputs;
                                    assert!(spv_inst.opcode == self.wk.OpConstant);
                                    match spv_inst.imms[..] {
                                        [spv::Imm::Short(_, x)] => x,
                                        _ => unreachable!(),
                                    }
                                }
                                _ => unreachable!(),
                            };
                            let line_col = (const_u32(line), const_u32(col));
                            current_src_loc = Some(DbgSrcLoc {
                                file_path: const_str(file),
                                start_line_col: line_col,
                                end_line_col: line_col,
                                inlined_callee_name_and_call_site: None,
                            });
                            insts_to_remove.push(inst);
                            continue;
                        }
                        CustomInst::ClearDebugSrcLoc => {
                            current_src_loc = None;
                            insts_to_remove.push(inst);
                            continue;
                        }
                        CustomInst::PushInlinedCallFrame { .. }
                        | CustomInst::PopInlinedCallFrame => {
                            insts_to_remove.push(inst);
                            continue;
                        }
                        CustomInst::Abort { .. } => {
                            assert!(
                                !custom_op.is_debuginfo(),
                                "`CustomOp::{custom_op:?}` debuginfo not lowered"
                            );
                        }
                    }
                }

                // Add/remove the equivalent `Attr::DbgSrcLoc` attribute.
                // FIXME(eddyb) this could use more caching.
                data_inst_def.attrs = self.cx.intern(AttrSetDef {
                    attrs: self.cx[data_inst_def.attrs]
                        .attrs
                        .iter()
                        .filter(|attr| !matches!(attr, Attr::DbgSrcLoc(_)))
                        .cloned()
                        .chain(current_src_loc.map(|loc| Attr::DbgSrcLoc(OrdAssertEq(loc))))
                        .collect(),
                });
            }

            // Finally remove the `DataInst`s buffered for removal earlier.
            for inst in insts_to_remove {
                insts.remove(inst, func_at_control_node.data_insts);
            }
            func_at_control_node.reborrow().def().kind = NodeKind::Block { insts };
        }

        func_at_control_node.inner_in_place_transform_with(self);
    }
}
