use crate::ScalarKernel;
use crate::hir::ssa::InterpretError;
use crate::hir::{Binary, Function, Op, Scale, Terminator, Type};
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
        f.append_inst(b, Op::Store(p, value)).unwrap();
    }
    // A pointer-sized store and load at an unaligned address.
    let one = f.append(b, Op::Const(1)).unwrap();
    let at = f.append(b, Op::PtrOffset(output, one, Scale::One)).unwrap();
    f.append_inst(b, Op::Store(at, result)).unwrap();
    let ptr = f.append(b, Op::Load(Ptr, at)).unwrap();
    f.set_terminator(b, Terminator::Return(ptr));

    let kernel = ScalarKernel::compile(&f, &crate::test_target::options()).unwrap();
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
                    kernel.call_raw(&args).map_err(InterpretError::Trap)
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
    let kernel = ScalarKernel::compile(&f, &crate::test_target::options()).unwrap();
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

#[test]
fn constant_offsets_outside_disp32_keep_the_full_pointer_width() {
    for (scale, stride) in [
        (Scale::One, 1),
        (Scale::Two, 2),
        (Scale::Four, 4),
        (Scale::Eight, 8),
    ] {
        let max_disp_index = (i64::from(i32::MAX) / stride) as i32;
        let min_disp_index = (i64::from(i32::MIN) / stride) as i32;
        for index in [
            i32::MIN,
            min_disp_index.saturating_sub(1),
            min_disp_index,
            max_disp_index,
            max_disp_index.saturating_add(1),
            i32::MAX,
        ] {
            let mut f = Function::new(&[Type::Ptr], Type::Ptr);
            let b = f.entry();
            let base = f.params(b)[0];
            let i = f.append(b, Op::Const(index)).unwrap();
            let at = f.append(b, Op::PtrOffset(base, i, scale)).unwrap();
            f.set_terminator(b, Terminator::Return(at));
            let kernel = ScalarKernel::compile(&f, &crate::test_target::options()).unwrap();
            let word = 0u64;
            let base = std::ptr::from_ref(&word) as u64;
            let expected = base.wrapping_add_signed(i64::from(index) * stride);
            // SAFETY: the kernel only computes an address; it never accesses the result.
            assert_eq!(
                unsafe { kernel.call_raw(&[base]) },
                Ok(expected),
                "{scale:?}, {index}"
            );
        }
    }
}

#[test]
fn scaled_store_between_overlapping_loads_changes_exactly_four_bytes() {
    for (scale, stride) in [
        (Scale::One, 1),
        (Scale::Two, 2),
        (Scale::Four, 4),
        (Scale::Eight, 8),
    ] {
        let mut f = Function::new(&[Type::Ptr, Type::I32, Type::Ptr, Type::I32], Type::I32);
        let b = f.entry();
        let [base, index, window, value] = f.params(b).try_into().unwrap();
        let before = f.append(b, Op::Load(Type::I32, window)).unwrap();
        let at = f.append(b, Op::PtrOffset(base, index, scale)).unwrap();
        f.append_inst(b, Op::Store(at, value)).unwrap();
        let after = f.append(b, Op::Load(Type::I32, window)).unwrap();
        let change = f.append(b, Op::Bin(Binary::Sub, after, before)).unwrap();
        f.set_terminator(b, Terminator::Return(change));
        let kernel = ScalarKernel::compile(&f, &crate::test_target::options()).unwrap();

        for index in -3i32..=3 {
            let dst = (33 + index * stride) as usize;
            for delta in -4..=4 {
                let window = dst.wrapping_add_signed(delta);
                let mut bytes: [u8; 72] = std::array::from_fn(|i| (i as u8).wrapping_mul(19));
                let mut expected = bytes;
                let read =
                    |buf: &[u8]| i32::from_ne_bytes(buf[window..window + 4].try_into().unwrap());
                let before = read(&expected);
                let value = 0x8042_a5ffu32;
                expected[dst..dst + 4].copy_from_slice(&value.to_ne_bytes());
                let change = read(&expected).wrapping_sub(before);
                let ptr = bytes.as_mut_ptr();
                let args = [
                    ptr.wrapping_add(33) as u64,
                    0x7654_3210_0000_0000 | u64::from(index as u32),
                    ptr.wrapping_add(window) as u64,
                    u64::from(value),
                ];
                // SAFETY: both raw pointers derive from the live byte array. The
                // four-byte accesses start in 5..=61, and unaligned access is allowed.
                let actual = unsafe { kernel.call_raw(&args) }.map(|v| v as i32);
                assert_eq!(
                    actual,
                    Ok(change),
                    "{scale:?}, index={index}, delta={delta}"
                );
                assert_eq!(bytes, expected, "{scale:?}, index={index}, delta={delta}");
            }
        }
    }
}
