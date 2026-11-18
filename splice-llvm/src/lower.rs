//! Lower a verified Splice call graph using Inkwell's typed builder.
mod body;
mod direct;
pub(crate) use direct::populate_direct;

use body::Body;
use inkwell::AddressSpace;
use inkwell::attributes::{Attribute, AttributeLoc};
use inkwell::builder::{Builder, BuilderError};
use inkwell::context::Context;
use inkwell::module::{Linkage, Module};
use inkwell::values::{BasicMetadataValueEnum, FunctionValue, IntValue};
use splice::Function;
use splice::hir::{BufferKind, Instr};
use std::collections::HashMap;
use std::sync::Arc;

// Only internal kernel functions use the fast calling convention.
const FAST_CALL: u32 = inkwell::llvm_sys::LLVMCallConv::LLVMFastCallConv as u32;

pub(crate) fn populate<'ctx>(
    ctx: &'ctx Context,
    module: &Module<'ctx>,
    root: &Function,
) -> Result<(), String> {
    let mut functions = vec![root];
    let mut ids = HashMap::from([(std::ptr::from_ref(root), 0)]);
    let mut i = 0;
    while i < functions.len() {
        for inst in functions[i].instructions() {
            if let Instr::Call { function, .. } = inst {
                ids.entry(Arc::as_ptr(function)).or_insert_with(|| {
                    let id = functions.len();
                    functions.push(function);
                    id
                });
            }
        }
        i += 1;
    }

    let mut declarations = Vec::with_capacity(functions.len());
    for (id, f) in functions.iter().enumerate() {
        declarations.push(declare(ctx, module, f.arity(), f.buffer_kinds(), id)?);
    }
    let builder = ctx.create_builder();
    emit_entry(
        ctx,
        module,
        &builder,
        root.arity(),
        root.buffer_kinds().len(),
        declarations[0],
    )
    .map_err(|e| e.to_string())?;
    for (id, f) in functions.into_iter().enumerate() {
        Body::emit(ctx, &builder, f, declarations[id], &ids, &declarations)
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn declare<'ctx>(
    ctx: &'ctx Context,
    module: &Module<'ctx>,
    arity: usize,
    buffers: &[BufferKind],
    id: usize,
) -> Result<FunctionValue<'ctx>, String> {
    let n = buffers
        .len()
        .checked_mul(2)
        .and_then(|n| n.checked_add(arity))
        .and_then(|n| n.checked_add(1))
        .ok_or("LLVM kernel signature too large")?;
    u32::try_from(n).map_err(|_| "kernel signature exceeds LLVM's parameter count")?;
    let mut params = Vec::with_capacity(n);
    let pointer = ctx.ptr_type(AddressSpace::default());
    params.push(pointer.into()); // status
    params.resize(1 + arity, ctx.i32_type().into());
    for _ in buffers {
        params.push(pointer.into());
        params.push(ctx.i64_type().into());
    }
    let ty = ctx.i32_type().fn_type(&params, false);
    let callee = module.add_function(&format!("kernel{id}"), ty, Some(Linkage::Internal));
    callee.set_call_conventions(FAST_CALL);

    let attr = |name: &str| Attribute::get_named_enum_kind_id(name);
    let noalias = ctx.create_enum_attribute(attr("noalias"), 0);
    let readonly = ctx.create_enum_attribute(attr("readonly"), 0);
    // LLVM 21 replaced nocapture with captures(none), encoded as zero bits.
    let capture = ["nocapture", "captures"]
        .into_iter()
        .map(attr)
        .find(|&id| id != 0)
        .map(|id| ctx.create_enum_attribute(id, 0));
    callee.add_attribute(AttributeLoc::Param(0), noalias);
    for (i, kind) in buffers.iter().enumerate() {
        let slot = AttributeLoc::Param((1 + arity + i * 2) as u32);
        callee.add_attribute(
            slot,
            match kind {
                BufferKind::Read => readonly,
                BufferKind::Write => noalias,
            },
        );
        if let Some(a) = capture {
            callee.add_attribute(slot, a);
        }
    }
    Ok(callee)
}

/// The public ABI receives scalars and descriptors in arrays. Kernel-to-kernel
/// calls pass scalar values, data pointers and lengths as native parameters.
fn emit_entry<'ctx>(
    ctx: &'ctx Context,
    module: &Module<'ctx>,
    b: &Builder<'ctx>,
    arity: usize,
    buffer_count: usize,
    root: FunctionValue<'ctx>,
) -> Result<(), BuilderError> {
    let ptr = ctx.ptr_type(AddressSpace::default());
    let i32_ty = ctx.i32_type();
    let i64_ty = ctx.i64_type();
    let ty = i32_ty.fn_type(&[ptr.into(); 3], false);
    let entry = module.add_function("splice_entry", ty, None);
    b.position_at_end(ctx.append_basic_block(entry, "entry"));
    let mut params = entry.get_param_iter();
    let scalars = params.next().unwrap().into_pointer_value();
    let descriptors = params.next().unwrap().into_pointer_value();
    let status = params.next().unwrap();
    let mut args = vec![status.into()];
    for i in 0..arity {
        let index = i64_ty.const_int(i as u64, false);
        // SAFETY: Executable::call verifies the length of the scalar array.
        let address = unsafe { b.build_in_bounds_gep(i32_ty, scalars, &[index], "")? };
        args.push(b.build_load(i32_ty, address, "")?.into());
    }
    let desc = ctx.struct_type(&[ptr.into(), i64_ty.into()], false);
    for i in 0..buffer_count {
        let index = i64_ty.const_int(i as u64, false);
        // SAFETY: Executable::call validates one (data, length) descriptor per slice.
        let slot = unsafe { b.build_in_bounds_gep(desc, descriptors, &[index], "")? };
        let data = b.build_struct_gep(desc, slot, 0, "")?;
        let len = b.build_struct_gep(desc, slot, 1, "")?;
        args.push(b.build_load(ptr, data, "")?.into());
        args.push(b.build_load(i64_ty, len, "")?.into());
    }
    let result = call(b, root, &args)?;
    b.build_return(Some(&result))?;
    Ok(())
}

fn call<'ctx>(
    b: &Builder<'ctx>,
    function: FunctionValue<'ctx>,
    args: &[BasicMetadataValueEnum<'ctx>],
) -> Result<IntValue<'ctx>, BuilderError> {
    let call = b.build_call(function, args, "")?;
    call.set_call_convention(FAST_CALL);
    // All Splice functions have an i32 result by construction.
    Ok(call.try_as_basic_value().unwrap_basic().into_int_value())
}
