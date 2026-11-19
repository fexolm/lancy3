#[path = "workloads/probe.rs"]
mod probe;

use probe::{fixture, lookup, reference};
use splice::Executable;
use splice_cranelift::Cranelift;
use splice_lancy::Lancy;
use splice_llvm::Llvm;

fn main() -> Result<(), String> {
    let kernel: Box<dyn Executable> = match std::env::args().nth(1).as_deref() {
        Some("llvm") => Box::new(lookup::build::<Llvm>()?),
        Some("cranelift") => Box::new(lookup::build::<Cranelift>()?),
        None | Some("lancy") => Box::new(lookup::build::<Lancy>()?),
        Some(name) => return Err(format!("unknown backend {name}")),
    };
    let (table, queries) = fixture(16, 12, 9);
    let mut slots = vec![-2; queries.len()];
    let count = kernel.call(
        &[],
        &mut [
            table.as_slice().into(),
            queries.as_slice().into(),
            slots.as_mut_slice().into(),
        ],
    )?;
    assert_eq!(count as usize, queries.len());
    assert_eq!(slots, reference(&table, &queries));
    for (key, slot) in queries.iter().zip(slots) {
        println!("key={key:11}  slot={slot:2}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use splice::{Codegen, simd::Lane};
    use splice_text::Text;

    fn check(
        kernels: &[Box<dyn Executable>],
        table: &[i32],
        queries: &[i32],
        expected: &[i32],
        valid: bool,
    ) {
        const GUARD: i32 = -19;
        let count = if valid { queries.len() as i32 } else { -1 };
        let output_len = expected.len();
        let mut guarded = vec![GUARD; output_len + 2];
        assert_eq!(
            lookup(table, queries, &mut guarded[1..1 + output_len]),
            count
        );
        assert_eq!(&guarded[1..1 + output_len], expected);
        assert_eq!((guarded[0], guarded[output_len + 1]), (GUARD, GUARD));
        for kernel in kernels {
            guarded.fill(GUARD);
            let result = kernel
                .call(
                    &[],
                    &mut [
                        table.into(),
                        queries.into(),
                        (&mut guarded[1..1 + output_len]).into(),
                    ],
                )
                .unwrap();
            assert_eq!(result, count);
            assert_eq!(&guarded[1..1 + output_len], expected);
            assert_eq!((guarded[0], guarded[output_len + 1]), (GUARD, GUARD));
        }
    }

    #[test]
    fn independent_probes_stop_on_matches_empty_slots_and_full_laps() {
        let ir = lookup::ir().unwrap();
        let kernels: [Box<dyn Executable>; 6] = [
            Box::new(lookup::build::<Lancy>().unwrap()),
            Box::new(Lancy::compile(&ir).unwrap()),
            Box::new(lookup::build::<Cranelift>().unwrap()),
            Box::new(Cranelift::compile(&ir).unwrap()),
            Box::new(lookup::build::<Llvm>().unwrap()),
            Box::new(Llvm::compile(&ir).unwrap()),
        ];
        let w = Lane::<i32>::lanes();
        // The multiplier is 1 modulo 8. Keys 7, 15, -1 and 31 all start at 7.
        let cluster = [15, -1, 0, 0, 0, 0, 0, 7];
        let chunk = [7, -1, 15, 31, 7, 0];
        for n in [0, 1, w - 1, w, w + 1, 2 * w + 3] {
            let queries: Vec<_> = chunk.iter().copied().cycle().take(n).collect();
            let expected: Vec<_> = [7, 1, 0, -1, 7, -1].into_iter().cycle().take(n).collect();
            check(&kernels, &cluster, &queries, &expected, true);
        }
        let full = [15, -1, 2, 3, 4, 5, 6, 7];
        check(
            &kernels,
            &full,
            &[15, 31, -1, 7, 0],
            &[0, -1, 1, 7, -1],
            true,
        );
        check(&kernels, &[0], &[42, 0], &[-1, -1], true);
        check(&kernels, &[42], &[42, 91], &[0, -1], true);
        check(&kernels, &[], &[42], &[-19], false);
        check(&kernels, &[1, 2, 3], &[42], &[-19], false);
        check(&kernels, &cluster, &[7, 15], &[-19], false);

        let (table, queries) = fixture(256, 192, 325);
        let expected = reference(&table, &queries);
        assert_eq!(
            expected.iter().filter(|&&s| s >= 0).count(),
            queries.len().div_ceil(2)
        );
        check(&kernels, &table, &queries, &expected, true);
    }

    #[test]
    fn lookup_ir_uses_masked_gather_and_pack_arithmetic() {
        let listing = lookup::build::<Text>().unwrap();
        assert_eq!(listing, Text::compile(&lookup::ir().unwrap()).unwrap());
        for op in [
            "simd.load_select.i32",
            "simd.mul.i32",
            "simd.gather_select.i32",
            "simd.any.i32",
            "simd.mask_and.i32",
            "simd.store_select.i32",
        ] {
            assert!(listing.contains(op), "{op} missing from {listing}");
        }
    }
}
