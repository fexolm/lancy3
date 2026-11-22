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
    let res = f.append(entry, Op::Call(call)).unwrap();
    f.set_terminator(entry, Terminator::Return(res));
    assert!(matches!(
        f.verify(),
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
    let copy = f.append_inst(entry, Op::Copy(ptr));
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
fn rust_packet_calls_verify_word_types_and_arity() {
    fn sum(args: &[i32]) -> i32 {
        args.iter().copied().fold(0, i32::wrapping_add)
    }
    static IMPORT: crate::hir::RustPacketImport = crate::hir::RustPacketImport {
        name: "sum4",
        arity: 4,
        resolve: || sum,
    };
    for (types, valid) in [
        (&[Type::I32; 4][..], true),
        (&[Type::I32; 3][..], false),
        (&[Type::I32, Type::I64, Type::I32, Type::I32][..], false),
    ] {
        let mut function = Function::new(types, Type::I32);
        let entry = function.entry();
        let args = function.params(entry).to_vec();
        let site = function.add_packet_call(&IMPORT, &args);
        let result = function.append(entry, Op::RustPacketCall(site)).unwrap();
        function.set_terminator(entry, Terminator::Return(result));
        assert_eq!(function.verify().is_ok(), valid);
        if valid {
            assert_eq!(function.interpret(&[1, 2, 3, 4]), Ok(10));
            assert_eq!(
                crate::ScalarKernel::compile(&function)
                    .unwrap()
                    .call(&[1, 2, 3, 4]),
                Ok(10)
            );
        }
    }
}
