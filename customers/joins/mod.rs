#[path = "../../splice/examples/relational/mod.rs"]
mod relational;

use relational::{Catalog, pricing_plan, query};
use splice_lancy::Lancy;
use std::collections::HashMap;

unsafe extern "C" {
    fn pricing_cpp(offsets: *const i32, data: *const i32, workspace: *mut i32, words: usize)
    -> i32;
}

type Row = (i32, i32);

fn tables(sales: usize, products: i32) -> [Vec<Row>; 5] {
    let mut state = 41_u32;
    let mut next = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 8) as i32
    };
    let orders = (0..sales)
        .map(|_| (next() % products - 1, next() % 8))
        .collect();
    let (mut prices, mut flags, mut promos, mut adjustments) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for key in -1..products - 1 {
        prices.push((key, next() % 3_000 + 1));
        if key % 13 == 0 {
            prices.push((key, next() % 10_000 + 1));
        }
        flags.push((key, i32::from(key % 5 != 0)));
        if key % 11 != 0 {
            promos.push((key, next() % 8));
        }
        adjustments.push((key, next() % 15));
        if key % 19 == 0 {
            adjustments.push((key, -3));
        }
    }
    [orders, prices, flags, promos, adjustments]
}

// Separate value-bag oracle, preserving multiplicity and scan order.
fn expected(tables: &[Vec<Row>; 5]) -> Vec<i32> {
    let lookups = [&tables[1], &tables[2], &tables[3], &tables[4]].map(|rows| {
        let mut by_key: HashMap<i32, Vec<i32>> = HashMap::new();
        for &(key, v) in rows {
            by_key.entry(key).or_default().push(v);
        }
        by_key
    });
    let mut result = Vec::new();
    for &(key, qty) in &tables[0] {
        if qty < 2 || key < 0 {
            continue;
        }
        let [Some(prices), Some(flags), Some(promos), Some(adjustments)] =
            lookups.each_ref().map(|m| m.get(&key))
        else {
            continue;
        };
        for &price in prices {
            for &flag in flags {
                if flag == 1 {
                    for &promo in promos {
                        for &adj in adjustments {
                            let discount = flag
                                .wrapping_add(promo)
                                .wrapping_mul(adj.wrapping_mul(2).min(20));
                            result.extend([
                                key,
                                qty.wrapping_mul(price)
                                    .wrapping_add(discount)
                                    .wrapping_add(1),
                            ]);
                        }
                    }
                }
            }
        }
    }
    result
}

fn cpp(catalog: &Catalog, workspace: &mut [i32]) -> i32 {
    assert_eq!(catalog.offsets().len(), 6);
    // SAFETY: Catalog validates every offset. The five tables and writable
    // workspace are distinct, and capacity is measured in i32 words.
    unsafe {
        pricing_cpp(
            catalog.offsets().as_ptr(),
            catalog.data().as_ptr(),
            workspace.as_mut_ptr(),
            workspace.len(),
        )
    }
}

pub fn run() -> Result<(), String> {
    let plan = pricing_plan();
    let code = crate::build("joins: pricing plan to x64", || {
        query::build::<Lancy>(&plan)
    })?;
    for (sales, products) in [(513, 257), (65_537, 4096)] {
        let tables = tables(sales, products);
        let catalog = Catalog::new(&tables.each_ref().map(Vec::as_slice))?;
        let expected = expected(&tables);
        let total = catalog.data().len() / 2;
        let workspace_words = 16 * total + expected.len() * 2 + 64;
        println!(
            "joins | {sales} sales | {total} input rows | {} output rows",
            expected.len() / 2
        );
        crate::compare_with_workspace(
            "joins: pricing, per sale",
            sales,
            &expected,
            workspace_words,
            |out| assert_eq!(cpp(&catalog, out), (expected.len() / 2) as i32),
            |out| assert_eq!(catalog.run(&code, out).unwrap().len(), expected.len()),
        )?;
    }
    Ok(())
}

#[test]
fn fixed_pricing_query_handles_duplicates_extremes_and_simd_tail() {
    let tables = [
        vec![(2, 1), (1, 3), (-1, 2), (2, 4), (1, i32::MAX)],
        vec![(1, -2), (2, i32::MIN), (1, 8)],
        vec![(1, 1), (1, 0), (2, 1)],
        vec![(1, i32::MAX), (2, 4)],
        vec![(2, -6), (1, i32::MIN), (1, 10)],
    ];
    let answer = expected(&tables);
    let catalog = Catalog::new(&tables.each_ref().map(Vec::as_slice)).unwrap();
    let jit = query::build::<Lancy>(&pricing_plan()).unwrap();
    let mut work = [0; 512];
    assert_eq!(cpp(&catalog, &mut work), (answer.len() / 2) as i32);
    assert_eq!(&work[..answer.len()], answer);
    work.fill(0);
    assert_eq!(catalog.run(&jit, &mut work).unwrap(), answer);
    assert_eq!(cpp(&catalog, &mut work[..3]), -1);
}
