//! Lower Splice kernels using Inkwell's typed builder.
mod body;
mod direct;
pub(crate) use body::native as native_intrinsic;
pub(crate) use direct::{Staged, populate, populate_function};

use inkwell::AddressSpace;
use inkwell::attributes::{Attribute, AttributeLoc};
use inkwell::builder::{Builder, BuilderError};
use inkwell::context::Context;
use inkwell::module::{Linkage, Module};
use inkwell::types::{BasicMetadataTypeEnum, BasicType};
use inkwell::values::{BasicMetadataValueEnum, BasicValueEnum, FunctionValue, IntValue};
use splice_core::frontend::Signature;
use splice_core::hir::ScalarType;

// Only internal kernel functions use the fast calling convention.
const FAST_CALL: u32 = inkwell::llvm_sys::LLVMCallConv::LLVMFastCallConv as u32;

fn declare<'ctx>(
    ctx: &'ctx Context,
    module: &Module<'ctx>,
    signature: Signature<'_>,
    id: usize,
) -> splice_core::Result<FunctionValue<'ctx>> {
    let (arity, buffers) = (signature.params.len(), signature.buffers);
    let n = buffers
        .len()
        .checked_mul(2)
        .and_then(|n| n.checked_add(arity))
        .and_then(|n| n.checked_add(1))
        .ok_or_else(|| splice_core::BuildError::unsupported("LLVM kernel signature too large"))?;
    u32::try_from(n).map_err(|_| {
        splice_core::BuildError::unsupported("kernel signature exceeds LLVM's parameter count")
    })?;
    let mut params = Vec::with_capacity(n);
    let pointer = ctx.ptr_type(AddressSpace::default());
    params.push(pointer.into()); // status
    params.extend(
        signature
            .params
            .iter()
            .map(|&ty| BasicMetadataTypeEnum::from(body::scalar_type(ctx, ty))),
    );
    for _ in buffers {
        params.push(pointer.into());
        params.push(ctx.i64_type().into());
    }
    let ty = body::scalar_type(ctx, signature.result).fn_type(&params, false);
    let callee = module.add_function(&format!("kernel{id}"), ty, Some(Linkage::Internal));
    callee.set_call_conventions(FAST_CALL);

    let attr = |name: &str| Attribute::get_named_enum_kind_id(name);
    let noalias = ctx.create_enum_attribute(attr("noalias"), 0);
    let readonly = ctx.create_enum_attribute(attr("readonly"), 0);
    let nocapture = ctx.create_enum_attribute(attr("nocapture"), 0);
    // An opaque Rust operation may observe a borrowed slice's address and
    // record it in the caller's object. Preserve that possibility in LLVM IR.
    let exposes_slice_addresses = buffers
        .iter()
        .any(|kind| kind.context_signature().is_some());
    callee.add_attribute(AttributeLoc::Param(0), noalias);
    for (i, kind) in buffers.iter().enumerate() {
        let slot = AttributeLoc::Param((1 + arity + i * 2) as u32);
        callee.add_attribute(slot, if kind.writable() { noalias } else { readonly });
        if !exposes_slice_addresses {
            callee.add_attribute(slot, nocapture);
        }
    }
    Ok(callee)
}

fn permit_buffer_capture(module: &Module<'_>) {
    let kind = Attribute::get_named_enum_kind_id("nocapture");
    // A Rust operation can record a slice address even when the slice was
    // originally borrowed by an ancestor kernel. Remove the assumptions
    // from the whole generated call graph before running LLVM passes.
    for function in module.get_functions() {
        for parameter in 0..function.count_params() {
            function.remove_enum_attribute(AttributeLoc::Param(parameter), kind);
        }
    }
}

/// The public ABI receives scalars and descriptors in arrays. Kernel-to-kernel
/// calls pass scalar values, data pointers and lengths as native parameters.
fn emit_entry<'ctx>(
    ctx: &'ctx Context,
    module: &Module<'ctx>,
    b: &Builder<'ctx>,
    signature: Signature<'_>,
    root: FunctionValue<'ctx>,
) -> Result<(), BuilderError> {
    let ptr = ctx.ptr_type(AddressSpace::default());
    let i64_ty = ctx.i64_type();
    let ty = i64_ty.fn_type(&[ptr.into(); 3], false);
    let entry = module.add_function("splice_entry", ty, None);
    b.position_at_end(ctx.append_basic_block(entry, "entry"));
    let mut params = entry.get_param_iter();
    let scalars = params.next().unwrap().into_pointer_value();
    let descriptors = params.next().unwrap().into_pointer_value();
    let result_word = params.next().unwrap().into_pointer_value();
    // Kernel functions store a trap code through the status pointer.
    let i32_ty = ctx.i32_type();
    let status = b.build_alloca(i32_ty, "status")?;
    b.build_store(status, i32_ty.const_zero())?;
    let mut args = vec![status.into()];
    for (i, &ty) in signature.params.iter().enumerate() {
        let index = i64_ty.const_int(i as u64, false);
        // SAFETY: Executable::call verifies the length of the scalar array.
        let address = unsafe { b.build_in_bounds_gep(i64_ty, scalars, &[index], "")? };
        let word = b.build_load(i64_ty, address, "")?.into_int_value();
        args.push(from_word(ctx, b, word, ty)?.into());
    }
    let desc = ctx.struct_type(&[ptr.into(), i64_ty.into()], false);
    for i in 0..signature.buffers.len() {
        let index = i64_ty.const_int(i as u64, false);
        // SAFETY: Executable::call validates one (data, length) descriptor per slice.
        let slot = unsafe { b.build_in_bounds_gep(desc, descriptors, &[index], "")? };
        let data = b.build_struct_gep(desc, slot, 0, "")?;
        let len = b.build_struct_gep(desc, slot, 1, "")?;
        args.push(b.build_load(ptr, data, "")?.into());
        args.push(b.build_load(i64_ty, len, "")?.into());
    }
    let result = call(b, root, &args)?;
    b.build_store(result_word, to_word(ctx, b, result, signature.result)?)?;
    // The status of `splice_core::call::Entry`: the trap code in the upper
    // word (see `Trap::status`).
    let code = b.build_load(i32_ty, status, "")?.into_int_value();
    let code = b.build_int_z_extend(code, i64_ty, "")?;
    let status = b.build_left_shift(code, i64_ty.const_int(32, false), "")?;
    b.build_return(Some(&status))?;
    Ok(())
}

/// A scalar parameter from its 64-bit ABI word, which holds the value's
/// bits in the low bits (see `splice_core::KernelScalar`).
fn from_word<'ctx>(
    ctx: &'ctx Context,
    b: &Builder<'ctx>,
    word: IntValue<'ctx>,
    ty: ScalarType,
) -> Result<BasicValueEnum<'ctx>, BuilderError> {
    Ok(match ty {
        ScalarType::Bool => b.build_int_truncate(word, ctx.bool_type(), "")?.into(),
        ScalarType::I32 | ScalarType::U32 => b.build_int_truncate(word, ctx.i32_type(), "")?.into(),
        ScalarType::I64 | ScalarType::U64 | ScalarType::Index => word.into(),
        ScalarType::F32 => {
            let bits = b.build_int_truncate(word, ctx.i32_type(), "")?;
            b.build_bit_cast(bits, ctx.f32_type(), "")?
        }
        ScalarType::F64 => b.build_bit_cast(word, ctx.f64_type(), "")?,
    })
}

/// The 64-bit ABI word of a scalar result: its bits, zero-extended.
fn to_word<'ctx>(
    ctx: &'ctx Context,
    b: &Builder<'ctx>,
    value: BasicValueEnum<'ctx>,
    ty: ScalarType,
) -> Result<IntValue<'ctx>, BuilderError> {
    let i64_ty = ctx.i64_type();
    let bits = match ty {
        ScalarType::F32 => b
            .build_bit_cast(value, ctx.i32_type(), "")?
            .into_int_value(),
        ScalarType::F64 => b.build_bit_cast(value, i64_ty, "")?.into_int_value(),
        _ => value.into_int_value(),
    };
    if bits.get_type() == i64_ty {
        Ok(bits)
    } else {
        b.build_int_z_extend(bits, i64_ty, "")
    }
}

fn call<'ctx>(
    b: &Builder<'ctx>,
    function: FunctionValue<'ctx>,
    args: &[BasicMetadataValueEnum<'ctx>],
) -> Result<BasicValueEnum<'ctx>, BuilderError> {
    let call = b.build_call(function, args, "")?;
    call.set_call_convention(FAST_CALL);
    // Every Splice function returns a scalar.
    Ok(call.try_as_basic_value().unwrap_basic())
}
