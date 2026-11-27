use crate::hir::IrError;
use crate::hir::ssa::{Function, Op, Terminator, Type};
use std::sync::Arc;

#[test]
fn rejects_sibling_uses_and_bad_edge_arity() {
    let mut f = Function::scalar(1);
    let first = f.entry();
    let cond = f.params(first)[0];
    let then = f.create_block();
    let other = f.create_block();
    let join = f.create_block();
    let result = f.append_param(join);
    let yes = f.add_edge(then, &[]);
    let no = f.add_edge(other, &[]);
    f.set_terminator(first, Terminator::Branch { cond, yes, no });
    let then_value = f.append(then, Op::Const(3)).unwrap();
    f.jump(then, join, &[then_value]);
    f.jump(other, join, &[then_value]);
    f.set_terminator(join, Terminator::Return(result));
    assert!(
        matches!(f.verify(), Err(IrError::DoesNotDominate { value, block, .. }) if value == then_value && block == other)
    );
    f.jump(other, join, &[]);
    assert!(matches!(
        f.verify(),
        Err(IrError::EdgeArity {
            expected: 1,
            actual: 0,
            ..
        })
    ));
    f.jump(other, join, &[cond]);
    f.verify().unwrap();
    assert!(f.merge_blocks(then, join).is_err());
    f.verify().unwrap();
}

#[test]
fn validates_call_types_and_rejects_entry_backedges() {
    let mut callee = Function::new(&[Type::Ptr], Type::Ptr);
    let arg = callee.params(callee.entry())[0];
    callee.set_terminator(callee.entry(), Terminator::Return(arg));
    let mut f = Function::scalar(1);
    let entry = f.entry();
    let arg = f.params(entry)[0];
    let call = f.add_call(Arc::new(callee), &[arg]);
    assert!(matches!(
        f.append(entry, Op::Call(call)),
        Err(IrError::OperandType {
            expected: Type::Ptr,
            actual: Type::I32,
            ..
        })
    ));
    f.jump(entry, entry, &[arg]);
    assert_eq!(f.verify(), Err(IrError::EntryBackedge { from: entry }));
}

#[test]
fn pointers_keep_their_type_across_edits_edges_and_returns() {
    use Type::{I32, Ptr};
    let mut f = Function::new(&[Ptr, I32], Ptr);
    let entry = f.entry();
    let [ptr, integer] = f.params(entry).try_into().unwrap();
    let copy = f.append_inst(entry, Op::Copy(ptr)).unwrap();
    let copied = f.result(copy).unwrap();
    let join = f.create_block();
    let param = f.append_typed_param(join, Ptr);
    f.jump(entry, join, &[copied]);
    f.set_terminator(join, Terminator::Return(param));
    f.verify().unwrap();

    assert!(f.replace_op(copy, Op::Copy(integer)).is_err());
    assert!(f.replace_op(copy, Op::Load(Ptr, integer)).is_err());
    assert_eq!(*f.op(copy), Op::Copy(ptr));
    f.jump(entry, join, &[integer]);
    assert!(matches!(
        f.verify(),
        Err(IrError::EdgeArgumentType {
            expected: Ptr,
            actual: I32,
            ..
        })
    ));
    f.jump(entry, join, &[copied]);
    f.set_terminator(join, Terminator::Return(integer));
    assert_eq!(
        f.verify(),
        Err(IrError::ReturnType {
            block: join,
            expected: Ptr,
            actual: I32
        })
    );
}
#[test]
fn rust_calls_verify_operand_types() {
    unsafe extern "C" fn add(_: *mut u8, operation: u32, _: *mut u8, word: i32) -> i64 {
        i64::from((operation as i32).wrapping_add(word) as u32)
    }
    static ADD: crate::hir::RustFunction = crate::hir::RustFunction {
        name: "add",
        entry: add,
    };
    // Each operand in turn gets a value of the other type.
    for wrong in [None, Some(0), Some(1), Some(2), Some(3)] {
        let mut function = Function::scalar(2);
        let entry = function.entry();
        let [operation, word] = function.params(entry)[..] else {
            unreachable!()
        };
        let packet = function
            .append(
                entry,
                Op::CallScratch {
                    offset: 0,
                    bytes: 8,
                },
            )
            .unwrap();
        let mut args = [packet, operation, packet, word];
        if let Some(i) = wrong {
            args[i] = if i % 2 == 0 { operation } else { packet };
        }
        let result = function.append(entry, Op::RustCall(&ADD, args));
        assert_eq!(result.is_ok(), wrong.is_none());
        if let Ok(result) = result {
            function.set_terminator(entry, Terminator::Return(result));
            function.verify().unwrap();
            assert_eq!(function.interpret(&[3, 4]), Ok(7));
            let kernel = crate::Kernel::compile(&function, &crate::test_target::options()).unwrap();
            // SAFETY: the only pointer is the kernel's own call scratch.
            assert_eq!(unsafe { kernel.call_word(&[3, 4]) }, Ok(7));
        }
    }
}

#[test]
fn one_operation_serves_scalars_and_vectors_but_not_their_mix() {
    use crate::hir::{Binary, Lane};
    let vector = Type::Vector(Lane::F32);
    let mut f = Function::new(&[vector, Type::F32], vector);
    let [x, s] = f.params(f.entry())[..] else {
        unreachable!()
    };
    let sum = f.append(f.entry(), Op::Bin(Binary::Add, x, x)).unwrap();
    assert_eq!(f.value_type(sum), vector);
    assert!(matches!(
        f.append(f.entry(), Op::Bin(Binary::Add, x, s)),
        Err(IrError::OperandType { .. })
    ));
    // A vector cast keeps the lane count, so it keeps the lane width; a
    // bitcast reinterprets the register. Neither turns a vector into a scalar.
    let to_int = f
        .append(f.entry(), Op::Cast(Type::Vector(Lane::I32), x))
        .unwrap();
    assert_eq!(f.value_type(to_int), Type::Vector(Lane::I32));
    for op in [
        Op::Cast(Type::Vector(Lane::F64), x),
        Op::Cast(Type::F32, x),
        Op::Bitcast(Type::F32, x),
        Op::Cast(vector, s),
    ] {
        assert!(matches!(
            f.append(f.entry(), op),
            Err(IrError::InvalidNumeric { .. })
        ));
    }
    let bytes = f
        .append(f.entry(), Op::Bitcast(Type::Vector(Lane::U8), x))
        .unwrap();
    assert_eq!(f.value_type(bytes), Type::Vector(Lane::U8));

    for (op, to) in [
        (Op::Bitcast as fn(Type, _) -> Op, Type::I64),
        (Op::Cast, Type::U64),
    ] {
        let mut f = Function::new(&[Type::Ptr], Type::I32);
        let p = f.params(f.entry())[0];
        assert!(matches!(
            f.append(f.entry(), op(to, p)),
            Err(IrError::PointerCast { from: Type::Ptr, to: actual, .. }) if actual == to
        ));
    }
    let mut f = Function::new(&[Type::Ptr], Type::U64);
    let p = f.params(f.entry())[0];
    let address = f.append(f.entry(), Op::Bitcast(Type::U64, p)).unwrap();
    f.set_terminator(f.entry(), Terminator::Return(address));
    f.verify().unwrap();
}

/// The builder types each operation when it is appended: an ill-typed
/// operation is an error and leaves the function unchanged, and the
/// verifier applies the same rule to an operation edited in place.
#[test]
fn append_rejects_ill_typed_operations_like_the_verifier() {
    use crate::hir::Binary;
    let mut f = Function::new(&[Type::F32, Type::I32], Type::I32);
    let entry = f.entry();
    let [x, n] = f.params(entry).try_into().unwrap();
    let values = f.value_count();
    assert!(matches!(
        f.append(entry, Op::Bin(Binary::Add, x, n)),
        Err(IrError::OperandType { value, expected: Type::F32, actual: Type::I32, .. }) if value == n
    ));
    assert_eq!(f.value_count(), values);
    assert_eq!(f.instructions(entry).count(), 0);
    let sum = f.append_inst(entry, Op::Bin(Binary::Add, n, n)).unwrap();
    f.set_terminator(entry, Terminator::Return(f.result(sum).unwrap()));
    f.verify().unwrap();
    assert!(matches!(
        f.replace_op(sum, Op::Bin(Binary::Add, n, x)),
        Err(IrError::OperandType { value, .. }) if value == x
    ));
    assert!(matches!(
        f.append(entry, Op::Store(n, n)),
        Err(IrError::OperandType {
            expected: Type::Ptr,
            ..
        })
    ));
    assert!(matches!(
        f.append(entry, Op::DebugStore(0, x)),
        Err(IrError::NoResult { .. })
    ));
}

/// One literal has one representation: `Const` for `i32`, and no bits above
/// the lane for every other lane.
#[test]
fn literals_have_one_canonical_form() {
    use crate::hir::Lane;
    let mut f = Function::new(&[], Type::I32);
    let entry = f.entry();
    for (lane, bits) in [(Lane::I32, 7), (Lane::I8, 0x1ff), (Lane::U16, 1 << 16)] {
        assert!(matches!(
            f.append(entry, Op::Literal(lane, bits)),
            Err(IrError::NonCanonicalLiteral { lane: l, bits: b, .. }) if l == lane && b == bits
        ));
    }
    assert_eq!(Op::literal(Lane::I32, 0xffff_ffff), Op::Const(-1));
    assert_eq!(Op::literal(Lane::I8, 0x1ff), Op::Literal(Lane::I8, 0xff));
    let byte = f.append_inst(entry, Op::literal(Lane::I8, 0x1ff)).unwrap();
    let zero = f.append(entry, Op::Const(0)).unwrap();
    f.set_terminator(entry, Terminator::Return(zero));
    f.verify().unwrap();
    assert!(matches!(
        f.replace_op(byte, Op::Literal(Lane::I8, 0x1ff)),
        Err(IrError::NonCanonicalLiteral { .. })
    ));
}
