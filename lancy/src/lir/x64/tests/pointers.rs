use crate::ScalarKernel;
use crate::hir::{Function, Op, Scale, Terminator, Type};
use std::sync::Arc;

#[test]
fn pointer_arguments_survive_spills_calls_and_block_parameters() {
    use Type::{I32, Ptr};
    // Enough live pointers to force spills and preserve registers across a nested call.
    let mut select = Function::new(&[Ptr, Ptr, I32], Ptr);
    let entry = select.entry();
    let [left, right, cond] = select.params(entry).try_into().unwrap();
    let join = select.create_block();
    let ptr = select.append_typed_param(join, Ptr);
    let yes = select.add_edge(join, &[left]);
    let no = select.add_edge(join, &[right]);
    select.set_terminator(entry, Terminator::Branch { cond, yes, no });
    select.set_terminator(join, Terminator::Return(ptr));

    let mut f = Function::new(&[Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, I32], Ptr);
    let b = f.entry();
    let [a, b1, c, d, e, output, cond] = f.params(b).try_into().unwrap();
    let call = f.add_call(Arc::new(select), &[a, e, cond]);
    let result = f.append(b, Op::Call(call)).unwrap();
    for p in [a, b1, c, d, e] {
        let value = f.append(b, Op::Const(27)).unwrap();
        f.append(b, Op::Store(p, value));
    }
    // A pointer-sized store and load at an unaligned address.
    let one = f.append(b, Op::Const(1)).unwrap();
    let at = f.append(b, Op::PtrOffset(output, one, Scale::One)).unwrap();
    f.append(b, Op::Store(at, result));
    let ptr = f.append(b, Op::Load(Ptr, at)).unwrap();
    f.set_terminator(b, Terminator::Return(ptr));

    let kernel = ScalarKernel::compile(&f).unwrap();
    for native in [false, true] {
        for condition in [0, 1] {
            let mut data = [0i32; 5];
            let mut output = [0u8; 9];
            let mut args: Vec<_> = data
                .iter_mut()
                .map(|v| std::ptr::from_mut(v) as u64)
                .collect();
            args.extend([output.as_mut_ptr() as u64, condition]);
            // SAFETY: each scalar has its own writable address and output[1..9] holds one pointer.
            let result = unsafe {
                if native {
                    kernel.call_raw(&args).map_err(|e| e.to_string())
                } else {
                    f.interpret_raw(&args)
                }
            };
            let expected = if condition == 0 { args[4] } else { args[0] };
            assert_eq!(result, Ok(expected));
            assert_eq!(
                u64::from_ne_bytes(output[1..].try_into().unwrap()),
                expected
            );
            assert_eq!(data, [27; 5]);
        }
    }
}

#[test]
fn pointer_offsets_sign_extend_indices_before_scaling() {
    let mut f = Function::new(&[Type::Ptr, Type::I32], Type::Ptr);
    let b = f.entry();
    let [base, index] = f.params(b).try_into().unwrap();
    let ptr = f
        .append(b, Op::PtrOffset(base, index, Scale::Eight))
        .unwrap();
    f.set_terminator(b, Terminator::Return(ptr));
    let kernel = ScalarKernel::compile(&f).unwrap();
    let data = [0u64; 3];
    let base = data[1..].as_ptr() as u64;
    // SAFETY: this kernel only computes an address and never dereferences it.
    for index in [-1i32, 0, 1, i32::MAX, i32::MIN] {
        let expected = base.wrapping_add_signed(i64::from(index) * 8);
        assert_eq!(
            unsafe { kernel.call_raw(&[base, index as u32 as u64]) },
            Ok(expected)
        );
    }
}
