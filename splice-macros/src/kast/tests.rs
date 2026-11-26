use super::*;
use syn::{Stmt, parse_quote};

fn ident(name: &str) -> Ident {
    Ident::new(name, proc_macro2::Span::call_site())
}

fn local(block: &Block, index: usize) -> &Local {
    let Stmt::Local(local) = &block.stmts[index] else {
        panic!("statement {index} is not a let")
    };
    local
}

#[test]
fn short_circuit_rhs_promotes_the_local_it_conditionally_writes() {
    let block = parse_quote!({
        let mut visited = 0;
        let _hit = x > 0 && {
            visited = 1;
            true
        };
        visited
    });
    let body = Body::new(&block, &[(ident("x"), Param::Scalar { index: false })]).unwrap();
    assert!(body.runtime_local(local(&block, 0)));
}

#[test]
fn return_operands_propagate_conditional_writes() {
    let block = parse_quote!({
        let mut a = 0;
        return {
            if x > 0 {
                a = 1;
            }
            a
        };
    });
    let body = Body::new(&block, &[(ident("x"), Param::Scalar { index: false })]).unwrap();
    assert!(body.runtime_local(local(&block, 0)));
}

/// A use resolves to the declaration in scope: a runtime value assigned to
/// a shadowing local does not stage the outer one, and an inner block's
/// local does not outlive the block.
#[test]
fn assignments_stage_the_local_they_resolve_to() {
    let block = parse_quote!({
        let mut a = 0;
        let mut b = 0;
        {
            let mut a = 0;
            a = x;
            b = a;
        }
        let c = a;
        c + b
    });
    let body = Body::new(&block, &[(ident("x"), Param::Scalar { index: false })]).unwrap();
    assert!(!body.runtime_local(local(&block, 0)));
    assert!(body.runtime_local(local(&block, 1)));
    assert!(!body.runtime_local(local(&block, 3)));
}

/// A literal local is a `usize` through a `.len()` comparison, a slice
/// index or another `usize` local; an `i32` counter and a suffixed literal
/// keep their own types, and a shadowing local is a separate declaration.
#[test]
fn literal_locals_take_the_usize_type_of_their_uses() {
    let block = parse_quote!({
        let mut i = 0;
        let mut j = 0;
        let mut k = 0;
        let mut n = 0;
        let s = 0u32;
        while i < xs.len() {
            j = i + 1;
            n += xs[k];
            i += 1;
        }
        let t = 0;
        let t = t + x;
        let m = xs.len();
        let mut q = 0;
        q += m;
        let i = 0;
        (j + k) as i32 + n + t + i
    });
    let params = [
        (ident("xs"), Param::Slice),
        (ident("x"), Param::Scalar { index: false }),
    ];
    let body = Body::new(&block, &params).unwrap();
    let usize_at: Vec<_> = (0..block.stmts.len())
        .filter(|&i| matches!(block.stmts[i], Stmt::Local(_)) && body.usize_local(local(&block, i)))
        .collect();
    // i, j, k, the `xs.len()` local m and q, which is added to m.
    assert_eq!(usize_at, [0, 1, 2, 8, 9]);
}

#[test]
fn comptime_arguments_are_parsed_once_and_resolved() {
    let block: Block = parse_quote!({
        let k = 2;
        comptime!(k + 1)
    });
    let body = Body::new(&block, &[]).unwrap();
    let Stmt::Expr(Expr::Macro(mac), _) = &block.stmts[1] else {
        panic!()
    };
    let Expr::Binary(sum) = body.comptime(&mac.mac) else {
        panic!()
    };
    let Expr::Path(k) = &*sum.left else { panic!() };
    assert_eq!(
        body.decl(body.decl_of(k).unwrap()).origin,
        Origin::Let(body.ids.locals[&address(local(&block, 0))])
    );
    let malformed: Block = parse_quote!({ comptime!(let) });
    assert!(Body::new(&malformed, &[]).is_err());
}

#[test]
fn vec_elements_are_parsed_once_and_resolved() {
    let block: Block = parse_quote!({
        let k = 2;
        vec![k; 3]
    });
    let body = Body::new(&block, &[]).unwrap();
    let Stmt::Expr(Expr::Macro(mac), _) = &block.stmts[1] else {
        panic!()
    };
    let Expr::Repeat(repeat) = body.vec_elements(&mac.mac) else {
        panic!()
    };
    let Expr::Path(k) = &*repeat.expr else {
        panic!()
    };
    assert_eq!(
        body.decl(body.decl_of(k).unwrap()).origin,
        Origin::Let(body.ids.locals[&address(local(&block, 0))])
    );
}
