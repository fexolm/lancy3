use crate::ScalarKernel;
use crate::hir::{Binary, Function, Op, Scale, Terminator, Type};
use crate::lir::x64::{
    compile::compile,
    inst::{Address, Inst, Rhs},
};

fn selected(f: &Function) -> Vec<Inst> {
    let (m, ids) = f.verified_module().unwrap();
    let m = compile(m, &ids).unwrap();
    m.functions[m.call_graph.root()].body.insts().to_vec()
}

#[test]
fn branch_covers_a_single_use_comparison_and_retains_a_shared_mask() {
    for shared in [false, true] {
        let mut f = Function::scalar(1);
        let b = f.entry();
        let x = f.params(b)[0];
        let one = f.append(b, Op::Const(1)).unwrap();
        let cond = f.append(b, Op::Bin(Binary::Lt, x, one)).unwrap();
        let yes = f.create_block();
        let no = f.create_block();
        f.set_terminator(yes, Terminator::Return(if shared { cond } else { x }));
        f.set_terminator(no, Terminator::Return(one));
        let yes = f.add_edge(yes, &[]);
        let no = f.add_edge(no, &[]);
        f.set_terminator(b, Terminator::Branch { cond, yes, no });
        let insts = selected(&f);
        assert_eq!(
            insts
                .iter()
                .any(|i| matches!(i, Inst::BrCmp(_, _, Rhs::Imm(1)))),
            !shared
        );
        assert_eq!(insts.iter().any(|i| matches!(i, Inst::Compare(..))), shared);
        let code = ScalarKernel::compile(&f).unwrap();
        assert_eq!(code.call(&[-9]), Ok(if shared { -1 } else { -9 }));
        assert_eq!(code.call(&[2]), Ok(1));
    }
}

#[test]
fn load_covers_scaled_address_and_shift_covers_its_constant() {
    let mut f = Function::new(&[Type::Ptr, Type::I32], Type::I32);
    let b = f.entry();
    let [base, i] = f.params(b).try_into().unwrap();
    let at = f.append(b, Op::PtrOffset(base, i, Scale::Four)).unwrap();
    let v = f.append(b, Op::Load(Type::I32, at)).unwrap();
    let n = f.append(b, Op::Const(48)).unwrap();
    let shifted = f.append(b, Op::Bin(Binary::Shr, v, n)).unwrap();
    f.set_terminator(b, Terminator::Return(shifted));
    let insts = selected(&f);
    assert!(insts.iter().any(|i| matches!(
        i,
        Inst::Load(
            _,
            _,
            Address {
                index: Some((_, Scale::Four)),
                ..
            }
        )
    )));
    assert!(insts.iter().any(|i| matches!(
        i,
        Inst::Shift {
            count: Rhs::Imm(48),
            ..
        }
    )));
    assert!(
        !insts
            .iter()
            .any(|i| matches!(i, Inst::Lea(..) | Inst::Imm(..)))
    );
    let words = [-0x123456i32, 0];
    let args = [words[1..].as_ptr() as u64, u32::MAX as u64];
    let code = ScalarKernel::compile(&f).unwrap();
    // SAFETY: index -1 from the second element addresses the first live i32.
    assert_eq!(
        unsafe { code.call_raw(&args) }.unwrap() as i32,
        words[0] >> 16
    );
}
