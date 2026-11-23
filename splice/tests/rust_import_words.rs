use splice::{Codegen, Executable, TypedIr, kernel};
use std::sync::Mutex;

static EFFECTS: Mutex<Vec<i32>> = Mutex::new(Vec::new());

fn two_words(a: i32, b: i32) -> i32 {
    EFFECTS.lock().unwrap().push(b);
    a.wrapping_sub(b)
}

fn four_words(a: i32, positive: bool, b: i32, c: i32) -> i32 {
    EFFECTS.lock().unwrap().push(a);
    assert_ne!(c, 0, "opaque Rust failure");
    let value = a
        .wrapping_mul(100)
        .wrapping_add(b.wrapping_mul(10))
        .wrapping_add(c);
    if positive {
        value
    } else {
        value.wrapping_neg()
    }
}

#[kernel]
fn call_four(output: &mut [i32], a: i32, positive: bool, b: i32, c: i32) -> i32 {
    output[0] = 41;
    let narrow = two_words(a, b);
    let value = four_words(narrow, positive, b, c);
    output[1] = value;
    output[0].wrapping_add(value)
}

#[test]
fn a_wide_scalar_rust_import_preserves_native_stores_and_effects() {
    let ir = call_four::ir().unwrap();
    use splice::hir::Instr;
    let imports = ir
        .instructions()
        .iter()
        .filter_map(|op| match op {
            Instr::RustCall { import, args, .. } => Some((import.symbol.name, args.len())),
            Instr::RustPacketCall { import, args, .. } => Some((import.name, args.len())),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(imports, [("two_words", 2), ("four_words", 4)]);
    fn check<B: Codegen<Output: Executable>>(ir: &TypedIr<call_four::Signature>) {
        for code in [
            call_four::build::<B>(call_four::BuildArgs {}).unwrap(),
            B::compile(ir).unwrap(),
        ] {
            for (a, positive, b, c) in [(-3, true, 4, -7), (2, false, -4, 7), (12, true, 3, 0)] {
                let mut expected_out = [-1, -2];
                let expected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    call_four(&mut expected_out, a, positive, b, c)
                }))
                .ok();
                let expected_effects = std::mem::take(&mut *EFFECTS.lock().unwrap());
                let mut output = [-1, -2];
                let actual = code
                    .call(call_four::CallArgs {
                        output: &mut output,
                        a,
                        positive,
                        b,
                        c,
                    })
                    .ok();
                assert_eq!(actual, expected);
                assert_eq!(output, expected_out);
                assert_eq!(
                    std::mem::take(&mut *EFFECTS.lock().unwrap()),
                    expected_effects
                );
            }
        }
    }
    splice_testkit::each_backend!(B => check::<B>(&ir));
}
