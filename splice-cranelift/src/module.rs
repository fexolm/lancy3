//! Lower the entire kernel before selecting which CLIF bodies to inline and emit.
use crate::lower;
use cranelift_codegen::{
    Context,
    inline::{Inline, InlineCommand},
    ir::{
        self, AbiParam, FuncRef, InstructionData, UserFuncName, Value,
        types::{I32, I64},
    },
};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::JITModule;
use cranelift_module::{FuncId, Linkage, Module};
use splice_core::{Function, hir::Instr};
use std::{
    borrow::Cow,
    collections::{HashMap, HashSet, hash_map::Entry},
    sync::Arc,
};

#[cfg(test)]
mod tests;

// Only the public entry reads the argument arrays and writes its result
// word. Helpers use the host
// calling convention with scalar values and (data, length) pairs.
fn declare(
    module: &mut JITModule,
    f: &Function,
    entry: bool,
    profile_id: Option<usize>,
) -> splice_core::Result<FuncId> {
    let mut sig = module.make_signature();
    if entry {
        // Scalar words, slice descriptors and the result word.
        sig.params.extend([AbiParam::new(I64); 3]);
    } else {
        // Cranelift uses u16 indices for block parameters.
        let out = usize::from(lower::returns_in_memory(f.result()));
        if out + f.arity() + 2 * f.buffer_kinds().len() > usize::from(u16::MAX) + 1 {
            return Err(splice_core::BuildError::unsupported(
                "too many parameters for a Cranelift kernel call",
            ));
        }
        if lower::returns_in_memory(f.result()) {
            sig.params.push(AbiParam::new(I64));
        }
        sig.params.extend(
            f.params()
                .iter()
                .map(|&ty| AbiParam::new(lower::scalar_type(ty))),
        );
        for _ in f.buffer_kinds() {
            sig.params.extend([AbiParam::new(I64), AbiParam::new(I32)]);
        }
    }
    sig.returns.push(AbiParam::new(I64));
    let id = if let Some(id) = profile_id {
        let label = f.source().map_or("kernel", |source| source.name);
        module.declare_function(
            &splice_jit::symbol_name(&format!("{label}#{id}")),
            Linkage::Local,
            &sig,
        )
    } else {
        module.declare_anonymous_function(&sig)
    };
    id.map_err(|e| splice_core::BuildError::backend("cranelift", e))
}

pub(super) struct KernelModule<'a> {
    pub entry: FuncId,
    bodies: HashMap<FuncId, ir::Function>,
    pub functions: HashMap<FuncId, &'a Function>,
}

impl<'a> KernelModule<'a> {
    pub fn lower(
        module: &mut JITModule,
        root: &'a Function,
        target: &splice_core::Target,
        debug: bool,
    ) -> splice_core::Result<Self> {
        let _span =
            tracing::debug_span!(target: "splice_core::compile", "lower_cranelift_ir").entered();
        let profile = splice_jit::perf_enabled();
        let entry = declare(module, root, true, profile.then_some(0))?;
        let mut pending = vec![(root, entry)];
        let mut ids = HashMap::from([(std::ptr::from_ref(root), entry)]);
        let mut bodies = HashMap::new();
        let mut functions = HashMap::new();
        let track_sources = debug || profile;
        let mut builder_ctx = FunctionBuilderContext::new();
        let machine = lower::Machine::new(target);
        let legacy_sse = target.x86().is_some_and(|isa| !isa.avx);
        while let Some((f, id)) = pending.pop() {
            if track_sources {
                functions.insert(id, f);
            }
            let mut ctx = module.make_context();
            ctx.func.name = UserFuncName::user(0, id.as_u32());
            ctx.func.signature = module
                .declarations()
                .get_function_decl(id)
                .signature
                .clone();
            let mut imports = HashMap::new();
            for inst in f.instructions() {
                if let Instr::Call { function, .. } = inst {
                    let key = Arc::as_ptr(function);
                    let profile_id = profile.then_some(ids.len());
                    let callee = match ids.entry(key) {
                        Entry::Occupied(slot) => *slot.get(),
                        Entry::Vacant(slot) => {
                            let callee = declare(module, function, false, profile_id)?;
                            pending.push((function, callee));
                            *slot.insert(callee)
                        }
                    };
                    imports.entry(key).or_insert_with(|| {
                        let reference = module.declare_func_in_func(callee, &mut ctx.func);
                        // A surviving native call may cross JIT allocations more than 2 GiB apart.
                        ctx.func.dfg.ext_funcs[reference].colocated = false;
                        reference
                    });
                }
            }
            let debug_bytes = if debug {
                f.instructions()
                    .iter()
                    .filter_map(|inst| match inst {
                        Instr::Debug(point) => {
                            Some(point.storage_bytes(f, machine.vector_bytes).max(1))
                        }
                        _ => None,
                    })
                    .max()
                    .map(i32::try_from)
                    .transpose()
                    .map_err(|_| {
                        splice_core::BuildError::unsupported(
                            "debug frame exceeds Cranelift's stack slot limit",
                        )
                    })?
                    .map(|size| size as u32)
            } else {
                None
            };
            lower::emit(
                FunctionBuilder::new(&mut ctx.func, &mut builder_ctx),
                f,
                id == entry,
                &imports,
                machine,
                debug_bytes,
            );
            if legacy_sse {
                lower::keep_compared_loads_in_registers(&mut ctx.func);
            }
            // Context::inline requires legalized and valid callee bodies.
            ctx.legalize(module.isa())
                .map_err(|e| splice_core::BuildError::backend("cranelift", e))?;
            ctx.verify(module.isa())
                .map_err(|e| splice_core::BuildError::backend("cranelift", e))?;
            bodies.insert(id, ctx.func);
        }
        Ok(Self {
            entry,
            bodies,
            functions,
        })
    }

    pub fn emit(
        self,
        ctx: &mut Context,
        debug: bool,
        mut define: impl FnMut(FuncId, &mut Context) -> splice_core::Result<()>,
    ) -> splice_core::Result<()> {
        if debug {
            for (id, body) in self.bodies {
                ctx.func = body;
                define(id, ctx)?;
                ctx.clear();
            }
            return Ok(());
        }
        // The first inline copy can replace its standalone definition. Allow
        // one module's worth of additional copies across all shared callees.
        let remaining = self.bodies.values().map(|f| f.dfg.num_insts()).sum();
        let mut bodies = self.bodies;
        let root = bodies.remove(&self.entry).expect("lowered entry");
        let mut inliner = KernelInliner {
            bodies: &bodies,
            inlined: HashSet::new(),
            remaining,
        };
        let mut seen = HashSet::from([self.entry]);
        let mut pending = vec![(self.entry, root)];
        while let Some((id, func)) = pending.pop() {
            ctx.func = func;
            tracing::debug_span!(target: "splice_core::compile", "inline_cranelift_ir")
                .in_scope(|| ctx.inline(&mut inliner))
                .map_err(|e| splice_core::BuildError::backend("cranelift", e))?;
            for child in local_callees(&ctx.func) {
                if seen.insert(child) {
                    pending.push((child, bodies[&child].clone()));
                }
            }
            define(id, ctx)?;
            ctx.clear();
        }
        Ok(())
    }
}

fn local_callees(func: &ir::Function) -> impl Iterator<Item = FuncId> + '_ {
    func.layout
        .blocks()
        .flat_map(|b| func.layout.block_insts(b))
        .filter_map(|i| match func.dfg.insts[i] {
            InstructionData::Call { func_ref, .. } => local_function(func, func_ref),
            _ => None,
        })
}

// declare_func_in_func encodes the module FuncId in user namespace zero.
// FuncRefs themselves are local to each body and are remapped during inlining.
fn local_function(caller: &ir::Function, callee: FuncRef) -> Option<FuncId> {
    let ir::ExternalName::User(name) = caller.dfg.ext_funcs[callee].name else {
        return None;
    };
    let name = &caller.params.user_named_funcs()[name];
    (name.namespace == 0).then(|| FuncId::from_u32(name.index))
}

struct KernelInliner<'a> {
    bodies: &'a HashMap<FuncId, ir::Function>,
    inlined: HashSet<FuncId>,
    remaining: usize,
}

impl Inline for KernelInliner<'_> {
    fn inline(
        &mut self,
        caller: &ir::Function,
        _: ir::Inst,
        _: ir::Opcode,
        callee: FuncRef,
        _: &[Value],
    ) -> InlineCommand<'_> {
        let Some((id, body)) =
            local_function(caller, callee).and_then(|id| self.bodies.get_key_value(&id))
        else {
            return InlineCommand::KeepCall;
        };
        if !self.inlined.insert(*id) {
            let Some(remaining) = self.remaining.checked_sub(body.dfg.num_insts()) else {
                return InlineCommand::KeepCall;
            };
            self.remaining = remaining;
        }
        InlineCommand::Inline {
            callee: Cow::Borrowed(body),
            visit_callee: true,
        }
    }
}
