//! Rust call signatures and their portable instructions. Operands on the
//! left are evaluated in source order; the list on the right is IR order.
use super::{Elem, Scalar};
use proc_macro2::TokenStream;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Owner {
    Lane,
    Mask,
}
impl Owner {
    pub(super) fn of(ty: Scalar) -> (Self, Elem) {
        match ty {
            Scalar::Vector(e) => (Self::Lane, e),
            Scalar::Mask(e) => (Self::Mask, e),
            _ => unreachable!("SIMD receiver"),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Param {
    Buffer,
    Splat,
    Count,
    Offset,
    Int,
    Vector,
    Mask,
    Indices,
    Data,
}
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Output {
    Vector,
    Mask,
    Converted,
    Bool,
    Int,
    Length,
    Unit,
}

#[derive(Clone, Copy)]
pub(super) struct Signature {
    pub receiver: bool,
    pub params: &'static [Param],
    pub result: Output,
    pub lower: fn(&[TokenStream]) -> TokenStream,
}

// Each rule binds the receiver (if present), then the typed arguments.
macro_rules! rule {
    (($($this:ident)? ; $($arg:ident : $ty:ident),*) -> $out:ident, $op:expr, $regs:expr) => {
        Signature {
            receiver: !stringify!($($this)?).is_empty(),
            params: &[$(Param::$ty),*],
            result: Output::$out,
            lower: |values| emit!({
                let ($($this,)? $($arg,)*) = (#(#values,)*);
                __splice_builder.call_intrinsic(__SpliceIntrinsic::Simd(__SpliceSimd {
                    element: __splice_element, op: $op,
                }), &$regs)?
            }),
        }
    };
}

macro_rules! registry {
    ($(($owner:path, $($name:literal)|+) => $sig:expr),+ $(,)?) => {
        pub(super) fn lookup(owner: Owner, name: &str) -> Option<Signature> {
            Some(match (owner, name) {
                $(($owner, $($name)|+) => $sig,)+
                _ => return None,
            })
        }

        #[cfg(test)]
        const REGISTERED: &[(Owner, &[&str])] = &[$(($owner, &[$($name),+]),)+];
    };
}

registry! {
    (Owner::Lane, "add" | "add_assign") => {
        rule!((v; rhs: Vector) -> Vector, __SpliceOp::Add, [v, rhs])
    },
    (Owner::Lane, "sub" | "sub_assign") => {
        rule!((v; rhs: Vector) -> Vector, __SpliceOp::Sub, [v, rhs])
    },
    (Owner::Lane, "mul" | "mul_assign") => {
        rule!((v; rhs: Vector) -> Vector, __SpliceOp::Mul, [v, rhs])
    },
    (Owner::Lane, "div" | "div_assign") => {
        rule!((v; rhs: Vector) -> Vector, __SpliceOp::Div, [v, rhs])
    },
    (Owner::Lane, "bitand" | "bitand_assign") => {
        rule!((v; rhs: Vector) -> Vector, __SpliceOp::And, [v, rhs])
    },
    (Owner::Lane, "bitor" | "bitor_assign") => {
        rule!((v; rhs: Vector) -> Vector, __SpliceOp::Or, [v, rhs])
    },
    (Owner::Lane, "bitxor" | "bitxor_assign") => {
        rule!((v; rhs: Vector) -> Vector, __SpliceOp::Xor, [v, rhs])
    },
    (Owner::Lane, "simd_min") => rule!((v; rhs: Vector) -> Vector, __SpliceOp::Min, [v, rhs]),
    (Owner::Lane, "simd_max") => rule!((v; rhs: Vector) -> Vector, __SpliceOp::Max, [v, rhs]),
    (Owner::Lane, "simd_eq") => rule!((v; rhs: Vector) -> Mask, __SpliceOp::Eq, [v, rhs]),
    (Owner::Lane, "simd_ne") => rule!((v; rhs: Vector) -> Mask, __SpliceOp::Ne, [v, rhs]),
    (Owner::Lane, "simd_lt") => rule!((v; rhs: Vector) -> Mask, __SpliceOp::Lt, [v, rhs]),
    (Owner::Lane, "simd_le") => rule!((v; rhs: Vector) -> Mask, __SpliceOp::Le, [v, rhs]),
    (Owner::Lane, "simd_gt") => rule!((v; rhs: Vector) -> Mask, __SpliceOp::Gt, [v, rhs]),
    (Owner::Lane, "simd_ge") => rule!((v; rhs: Vector) -> Mask, __SpliceOp::Ge, [v, rhs]),
    (Owner::Lane, "element_width") => rule!((;) -> Length, __SpliceOp::ElementWidth, []),
    (Owner::Lane, "byte_width") => rule!((;) -> Length, __SpliceOp::ByteWidth, []),
    (Owner::Lane, "iota") => rule!((;) -> Vector, __SpliceOp::Iota, []),
    (Owner::Lane, "first_n") => rule!((; count: Offset) -> Mask, __SpliceOp::FirstN, [count]),
    (Owner::Lane, "alignment_prefix") => {
        rule!((; src: Buffer, offset: Offset) -> Length, __SpliceOp::AlignmentPrefix(src), [offset])
    },
    (Owner::Lane, "load") => {
        rule!((; src: Buffer, offset: Offset) -> Vector, __SpliceOp::Load(src), [offset])
    },
    (Owner::Lane, "load_select") => {
        rule!((; src: Buffer, offset: Offset, mask: Mask, fallback: Vector) -> Vector, __SpliceOp::LoadSelect(src), [offset, mask, fallback])
    },
    (Owner::Lane, "store") => {
        rule!((v; dst: Buffer, offset: Offset) -> Unit, __SpliceOp::Store(dst), [offset, v])
    },
    (Owner::Lane, "store_select") => {
        rule!((v; dst: Buffer, offset: Offset, mask: Mask) -> Unit, __SpliceOp::StoreSelect(dst), [offset, mask, v])
    },
    (Owner::Lane, "compress_store") => {
        rule!((v; dst: Buffer, offset: Offset, mask: Mask) -> Unit, __SpliceOp::CompressStore(dst), [offset, mask, v])
    },
    (Owner::Lane, "gather") => {
        rule!((; src: Buffer, indices: Indices) -> Vector, __SpliceOp::Gather(src), [indices])
    },
    (Owner::Lane, "gather_select") => {
        rule!((; src: Buffer, mask: Mask, indices: Indices, fallback: Vector) -> Vector, __SpliceOp::GatherSelect(src), [indices, mask, fallback])
    },
    (Owner::Lane, "scatter") => {
        rule!((v; dst: Buffer, indices: Indices) -> Unit, __SpliceOp::Scatter(dst), [indices, v])
    },
    (Owner::Lane, "scatter_select") => {
        rule!((v; dst: Buffer, mask: Mask, indices: Indices) -> Unit, __SpliceOp::ScatterSelect(dst), [indices, mask, v])
    },
    (Owner::Lane, "not") => rule!((v;) -> Vector, __SpliceOp::Not, [v]),
    (Owner::Lane, "prefix_sum") => rule!((v;) -> Vector, __SpliceOp::PrefixSum, [v]),
    (Owner::Lane, "reduce_sum") => rule!((v;) -> Int, __SpliceOp::ReduceSum, [v]),
    (Owner::Lane, "sqrt") => rule!((v;) -> Vector, __SpliceOp::Sqrt, [v]),
    (Owner::Lane, "last") => rule!((v;) -> Int, __SpliceOp::Last, [v]),
    (Owner::Lane, "shr") => rule!((v; count: Count) -> Vector, __SpliceOp::Shr(count), [v]),
    (Owner::Lane, "rotate_left") => {
        rule!((v; count: Count) -> Vector, __SpliceOp::RotateLeft(count), [v])
    },
    (Owner::Lane, "mul_add") => {
        rule!((v; mul: Vector, add: Vector) -> Vector, __SpliceOp::MulAdd, [v, mul, add])
    },
    (Owner::Lane, "compress") => {
        rule!((v; mask: Mask) -> Vector, __SpliceOp::Compress, [v, mask])
    },
    (Owner::Lane, "expand") => {
        rule!((v; mask: Mask, fallback: Vector) -> Vector, __SpliceOp::Expand, [v, mask, fallback])
    },
    (Owner::Lane, "swizzle") => {
        rule!((v; indices: Vector) -> Vector, __SpliceOp::Swizzle, [v, indices])
    },
    (Owner::Lane, "extract_wrapping") => {
        rule!((v; index: Int) -> Int, __SpliceOp::ExtractWrapping, [v, index])
    },
    (Owner::Lane, "replace_wrapping") => {
        rule!((v; index: Int, value: Int) -> Vector, __SpliceOp::ReplaceWrapping, [v, index, value])
    },
    (Owner::Lane, "cast") => rule!((v;) -> Converted, __SpliceOp::Cast(__splice_to), [v]),
    (Owner::Lane, "narrow") => {
        rule!((v; high: Vector) -> Converted, __SpliceOp::Narrow(__splice_to), [v, high])
    },
    (Owner::Lane, "widen_low") => {
        rule!((v;) -> Converted, __SpliceOp::Widen { to: __splice_to, high: false }, [v])
    },
    (Owner::Lane, "widen_high") => {
        rule!((v;) -> Converted, __SpliceOp::Widen { to: __splice_to, high: true }, [v])
    },
    (Owner::Lane, "interleave_low") => {
        rule!((v; rhs: Vector) -> Vector, __SpliceOp::Shuffle(__SpliceShuffle::InterleaveLow), [v, rhs])
    },
    (Owner::Lane, "interleave_high") => {
        rule!((v; rhs: Vector) -> Vector, __SpliceOp::Shuffle(__SpliceShuffle::InterleaveHigh), [v, rhs])
    },
    (Owner::Lane, "deinterleave_even") => {
        rule!((v; rhs: Vector) -> Vector, __SpliceOp::Shuffle(__SpliceShuffle::DeinterleaveEven), [v, rhs])
    },
    (Owner::Lane, "deinterleave_odd") => {
        rule!((v; rhs: Vector) -> Vector, __SpliceOp::Shuffle(__SpliceShuffle::DeinterleaveOdd), [v, rhs])
    },
    (Owner::Mask, "any") => rule!((m;) -> Bool, __SpliceOp::Any, [m]),
    (Owner::Mask, "all") => rule!((m;) -> Bool, __SpliceOp::All, [m]),
    (Owner::Mask, "count") => rule!((m;) -> Int, __SpliceOp::Count, [m]),
    (Owner::Mask, "select") => {
        rule!((m; yes: Data, no: Data) -> Converted, __SpliceOp::Select, [m, yes, no])
    },
    (Owner::Mask, "not") => rule!((m;) -> Mask, __SpliceOp::MaskNot, [m]),
    (Owner::Mask, "bitand") => rule!((m; rhs: Mask) -> Mask, __SpliceOp::MaskAnd, [m, rhs]),
    (Owner::Mask, "bitor") => rule!((m; rhs: Mask) -> Mask, __SpliceOp::MaskOr, [m, rhs]),
    (Owner::Mask, "bitxor") => rule!((m; rhs: Mask) -> Mask, __SpliceOp::MaskXor, [m, rhs]),
    (Owner::Lane, "splat") => Signature {
        receiver: false,
        params: &[Param::Splat],
        result: Output::Vector,
        lower: |values| {
            let splat = &values[0];
            emit!({
                let (op, scalar) = #splat;
                __splice_builder.call_intrinsic(__SpliceIntrinsic::Simd(__SpliceSimd { element: __splice_element, op }), scalar.as_slice())?
            })
        },
    },
    (Owner::Lane, "add_masked") => {
        rule!((v; mask: Mask, rhs: Vector, fallback: Vector) -> Vector, __SpliceOp::AddMasked, [v, mask, rhs, fallback])
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::{ImplItem, Item, Type};

    #[test]
    fn registry_matches_facade_declarations() {
        let source = include_str!("../../../../../splice/src/std/intrinsics/simd.rs");
        let mut declared = Vec::new();
        for item in syn::parse_file(source).unwrap().items {
            let Item::Impl(block) = item else { continue };
            let Type::Path(ty) = block.self_ty.as_ref() else {
                continue;
            };
            let name = &ty.path.segments.last().unwrap().ident;
            for method in block.items {
                let ImplItem::Macro(declaration) = method else {
                    continue;
                };
                if !declaration.mac.path.is_ident("intrinsic") {
                    continue;
                }
                let owner = match name.to_string().as_str() {
                    "Lane" => Owner::Lane,
                    "Mask" => Owner::Mask,
                    _ => panic!("unregistered intrinsic owner {name}"),
                };
                let sig: syn::Signature = declaration
                    .mac
                    .parse_body_with(|input: syn::parse::ParseStream<'_>| {
                        input.call(syn::Attribute::parse_outer)?;
                        input.parse::<syn::Visibility>()?;
                        let signature = input.parse()?;
                        input.parse::<Option<syn::Token![;]>>()?;
                        Ok(signature)
                    })
                    .unwrap();
                let name = sig.ident.to_string();
                let entry = lookup(owner, &name)
                    .unwrap_or_else(|| panic!("no lowering for {owner:?}::{name}"));
                assert_eq!(
                    entry.receiver,
                    sig.receiver().is_some(),
                    "{owner:?}::{name}"
                );
                assert_eq!(
                    entry.params.len() + usize::from(entry.receiver),
                    sig.inputs.len(),
                    "{owner:?}::{name}"
                );
                declared.push((owner, name));
            }
        }
        for &(owner, names) in REGISTERED {
            for &name in names {
                assert!(
                    declared.iter().any(|(o, n)| *o == owner && n == name),
                    "no declaration for {owner:?}::{name}"
                );
            }
        }
    }
}
