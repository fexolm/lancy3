use splice::{BuildOptions, CallError, Codegen, Executable, Trap, hir::Instr, kernel};
use splice_text::Text;
use std::{
    ops::Index,
    sync::atomic::{AtomicUsize, Ordering},
};

#[kernel]
fn chosen(#[comptime] table: &[u16], flag: i32) -> i32 {
    if flag != 0 { table[0] as i32 } else { 0 }
}

#[kernel]
fn store_before_lookup<const N: usize>(#[comptime] table: [i32; N], out: &mut [i32]) -> i32 {
    out[0] = 7;
    let value = table[0];
    out[1] = value;
    value
}

#[kernel]
fn mandatory(#[comptime] table: &[i32], flag: i32) -> i32 {
    if flag != 0 { comptime!(table[0]) } else { 0 }
}

#[kernel]
fn runtime_offset(#[comptime] table: &[i32], index: i32) -> i32 {
    table[index as usize]
}

struct Indexed {
    count: AtomicUsize,
    value: i32,
}

impl Index<usize> for Indexed {
    type Output = i32;

    fn index(&self, _index: usize) -> &Self::Output {
        self.count.fetch_add(1, Ordering::Relaxed);
        &self.value
    }
}

#[kernel]
fn custom_index(#[comptime] plan: &Indexed) -> i32 {
    plan[0]
}

#[kernel]
fn borrow_custom_index(#[comptime] plan: &Indexed, x: i32) -> i32 {
    let value = &plan[0];
    x.wrapping_add(*value)
}

#[kernel]
fn write_borrowed_plan(#[comptime] plan: &mut [i32], value: i32) -> i32 {
    plan[0] += 1;
    value
}

fn check<B: Codegen<Output: Executable>>() {
    let mut borrowed = [7];
    assert!(
        write_borrowed_plan::build::<B>(write_borrowed_plan::BuildArgs {
            plan: &mut borrowed
        })
        .is_err()
    );
    assert!(write_borrowed_plan::ir(&mut borrowed).is_err());
    assert_eq!(borrowed, [7]);

    let options = BuildOptions::Release;
    let ir = chosen::ir_with_options(options, &[]).expect("construct IR without a plan address");
    assert!(
        ir.instructions()
            .iter()
            .any(|op| matches!(op, Instr::TrapBounds))
    );
    let direct =
        chosen::build_with_options::<B>(options, chosen::BuildArgs { table: &[] }).unwrap();
    let portable = B::compile(&ir).unwrap();
    for code in [direct, portable] {
        assert_eq!(code.call(chosen::CallArgs { flag: 0 }).unwrap(), 0);
        assert_eq!(
            code.call(chosen::CallArgs { flag: 1 }).unwrap_err(),
            CallError::Trap(Trap::Bounds)
        );
        assert_eq!(code.call(chosen::CallArgs { flag: 0 }).unwrap(), 0);
    }

    let (direct, portable) = {
        let table = [11];
        let direct =
            chosen::build_with_options::<B>(options, chosen::BuildArgs { table: &table }).unwrap();
        let portable = B::compile(&chosen::ir_with_options(options, &table).unwrap()).unwrap();
        (direct, portable)
    };
    for code in [direct, portable] {
        assert_eq!(code.call(chosen::CallArgs { flag: 1 }).unwrap(), 11);
    }

    let direct = store_before_lookup::build_with_options::<B, 0>(
        options,
        store_before_lookup::BuildArgs { table: [] },
    )
    .unwrap();
    let portable = B::compile(&store_before_lookup::ir_with_options(options, []).unwrap()).unwrap();
    for code in [direct, portable] {
        let mut out = [0, 21];
        assert!(
            code.call(store_before_lookup::CallArgs { out: &mut out })
                .is_err()
        );
        assert_eq!(out, [7, 21]);
    }

    assert!(mandatory::ir(&[]).is_err());
    assert!(mandatory::build::<B>(mandatory::BuildArgs { table: &[] }).is_err());
    assert!(runtime_offset::ir(&[3]).is_err());
    assert!(runtime_offset::build::<B>(runtime_offset::BuildArgs { table: &[3] }).is_err());
    let plan = Indexed {
        count: AtomicUsize::new(0),
        value: 9,
    };
    assert!(custom_index::ir(&plan).is_err());
    assert!(custom_index::build::<B>(custom_index::BuildArgs { plan: &plan }).is_err());
    assert!(borrow_custom_index::ir(&plan).is_err());
    assert!(
        borrow_custom_index::build::<B>(borrow_custom_index::BuildArgs { plan: &plan }).is_err()
    );
    assert_eq!(plan.count.load(Ordering::Relaxed), 0);
    assert_eq!(custom_index(&plan), 9);
    assert_eq!(plan.count.load(Ordering::Relaxed), 1);
}

#[test]
fn known_indices_only_fault_on_the_reached_execution_path() {
    splice_testkit::each_backend!(B => check::<B>());
    let ir = chosen::ir(&[]).unwrap();
    assert!(
        Text::compile(&ir)
            .unwrap()
            .as_inner()
            .contains("trap bounds")
    );
    let source = chosen::print(&[]).unwrap();
    assert!(
        source.contains("panic!(\"kernel slice index out of bounds\")"),
        "{source}"
    );
}
