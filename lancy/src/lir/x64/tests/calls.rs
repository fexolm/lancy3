use crate::hir::ssa::InterpretError;
use crate::hir::{Binary, Function, Op as H, Scale, Terminator, Type};
use crate::lir::x64::compile::compile;
use crate::{CallError, ScalarKernel, SignatureError, Trap};
use std::sync::Arc;

#[test]
fn a_small_callee_leaves_registers_available_to_its_caller() {
    let mut identity = Function::scalar(1);
    let b = identity.entry();
    identity.set_terminator(b, Terminator::Return(identity.params(b)[0]));

    let mut f = Function::scalar(4);
    let b = f.entry();
    let args = f.params(b).to_vec();
    let site = f.add_call(Arc::new(identity), &args[..1]);
    let mut result = f.append(b, H::Call(site)).unwrap();
    for arg in args {
        result = f.append(b, H::Bin(Binary::Add, result, arg)).unwrap();
    }
    f.set_terminator(b, Terminator::Return(result));
    let source = f.verified_module().unwrap();
    let entry = source.call_graph.root();
    let edges = source.call_graph.callees(entry).as_ptr();
    let module = compile(crate::lir::x64::Machine::host(), source, &mut |_| {}).unwrap();
    assert_eq!(module.call_graph.callees(entry).as_ptr(), edges);
    assert_eq!(
        module.functions[module.call_graph.root()]
            .allocation
            .num_spillslots,
        0
    );
    assert_eq!(
        ScalarKernel::compile(&f, &crate::test_target::options())
            .unwrap()
            .call(&[1, 2, 3, 4]),
        Ok(11)
    );
}

#[test]
fn checked_nested_access_preserves_prior_stores_and_forwards_addresses() {
    use Type::{I32, Ptr};
    let signature = [I32, I32, Ptr, I32, Ptr, I32];
    let mut leaf = Function::new(&signature, I32);
    let b = leaf.entry();
    let [i, j, src, n, dst, m] = leaf.params(b).try_into().unwrap();
    leaf.append_inst(b, H::BoundsCheck(i, n)).unwrap();
    let at = leaf.append(b, H::PtrOffset(src, i, Scale::Four)).unwrap();
    let read = leaf.append(b, H::Load(I32, at)).unwrap();
    let sum = leaf.append(b, H::Bin(Binary::Add, read, n)).unwrap();
    leaf.append_inst(b, H::BoundsCheck(j, m)).unwrap();
    let at = leaf.append(b, H::PtrOffset(dst, j, Scale::Four)).unwrap();
    leaf.append_inst(b, H::Store(at, sum)).unwrap();
    leaf.set_terminator(b, Terminator::Return(sum));

    let mut root = Function::new(&signature, I32);
    let b = root.entry();
    let [i, j, dst, m, src, n] = root.params(b).try_into().unwrap();
    let nine = root.append(b, H::Const(9)).unwrap();
    root.append_inst(b, H::Store(dst, nine)).unwrap();
    let call = root.add_call(Arc::new(leaf), &[i, j, src, n, dst, m]);
    let result = root.append(b, H::Call(call)).unwrap();
    let two = root.append(b, H::Const(2)).unwrap();
    let last = root.append(b, H::PtrOffset(dst, two, Scale::Four)).unwrap();
    root.append_inst(b, H::Store(last, result)).unwrap();
    root.set_terminator(b, Terminator::Return(result));
    let k = ScalarKernel::compile(&root, &crate::test_target::options()).unwrap();
    assert_eq!(
        k.call(&[0; 6]),
        Err(crate::CallError::Signature(
            SignatureError::PointerSignature
        ))
    );
    for (i, j) in [(-1, 1), (1, -1), (3, 0), (0, 3), (1, 1)] {
        let data = [2i32, 42, 1];
        for native in [false, true] {
            let mut out = [-77i32; 3];
            let args = [
                i as u32 as u64,
                j as u32 as u64,
                out.as_mut_ptr() as u64,
                3,
                data.as_ptr() as u64,
                3,
            ];
            // SAFETY: both arrays have three elements and all variable indices are checked.
            let result = unsafe {
                if native {
                    k.call_raw(&args).map_err(InterpretError::Trap)
                } else {
                    root.interpret_raw(&args)
                }
            };
            if (i, j) == (1, 1) {
                assert_eq!(result, Ok(45));
                assert_eq!(out, [9, 45, 45]);
            } else {
                assert_eq!(
                    result,
                    Err(InterpretError::Trap(CallError::Trap(Trap::Bounds)))
                );
                assert_eq!(out, [9, -77, -77]);
            }
        }
    }
}

#[test]
fn shared_calls_preserve_live_registers_and_spilled_arguments() {
    let mut leaf = Function::scalar(2);
    let b = leaf.entry();
    let [x, y] = leaf.params(b).try_into().unwrap();
    let v = leaf.append(b, H::Bin(Binary::Sub, x, y)).unwrap();
    leaf.set_terminator(b, Terminator::Return(v));
    let shared = Arc::new(leaf);
    let mut f = Function::scalar(6);
    let b = f.entry();
    let a = f.params(b).to_vec();
    let call = f.add_call(shared.clone(), &[a[4], a[5]]);
    let first = f.append(b, H::Call(call)).unwrap();
    let call = f.add_call(shared, &[first, a[3]]);
    let mut v = f.append(b, H::Call(call)).unwrap();
    for arg in a {
        v = f.append(b, H::Bin(Binary::Add, v, arg)).unwrap();
    }
    f.set_terminator(b, Terminator::Return(v));
    let source = f.verified_module().unwrap();
    assert_eq!(
        compile(crate::lir::x64::Machine::host(), source, &mut |_| {})
            .unwrap()
            .functions
            .len(),
        2
    );
    let k = ScalarKernel::compile(
        &f,
        &crate::test_target::options().with_inlining(crate::Inlining::Never),
    )
    .unwrap();
    for input in [[1, 2, 3, 4, 5, 6], [-1, -2, 5, i32::MIN, i32::MAX, 0]] {
        assert_eq!(k.call(&input).unwrap(), f.interpret(&input).unwrap());
    }
}
