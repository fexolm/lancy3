//! Runtime slices carry an immutable buffer identity through specialization.
//! Only lengths, element accesses, and scalar computations become instructions.
use super::{Lower, Scalar, Stage, Value, err};
use proc_macro2::TokenStream;
use syn::{BinOp, Expr, ExprCast, ExprIndex, ExprMethodCall, ExprReference, Result, UnOp};

pub(super) fn primitive_method(name: &str) -> bool {
    matches!(
        name,
        "wrapping_add"
            | "wrapping_sub"
            | "wrapping_mul"
            | "wrapping_neg"
            | "checked_add"
            | "checked_sub"
            | "checked_mul"
            | "checked_div"
            | "checked_rem"
            | "saturating_add"
            | "saturating_sub"
            | "saturating_mul"
            | "min"
            | "max"
    )
}

impl Lower<'_> {
    pub(super) fn method(&self, m: &ExprMethodCall, hint: Option<Scalar>) -> Result<Value> {
        if let Expr::Index(index) = &*m.receiver
            && let Expr::Path(table) = &*index.expr
            && matches!(self.binding(table), Some(super::Binding::Columns))
        {
            if m.method != "as_slice" || !m.args.is_empty() {
                return err(m, "use as_slice::<T>() to bind a typed column");
            }
            let Some(elem) = m
                .turbofish
                .as_ref()
                .and_then(|args| super::simd::element_argument(args, self.types))
            else {
                return err(
                    m,
                    "as_slice::<T>() requires u16, u32, i32, i64, f32, or f64",
                );
            };
            let slot = self.host(self.expr(&index.index)?, &index.index)?;
            let root = self.root;
            let element = elem.tokens(root, self.types);
            return Ok(Value::new(
                Stage::Buffer(super::BufferType { elem, write: false }),
                emit!(#root::frontend::scalar::column(#table, #slot, #element)?),
            ));
        }
        let data_type = if matches!(
            m.method.to_string().as_str(),
            "store" | "store_select" | "scatter" | "scatter_select"
        ) && let Some(dst) = m.args.first()
            && let Stage::Buffer(buffer) = self.expr(dst)?.stage
        {
            Some(Scalar::Vector(buffer.elem))
        } else {
            None
        };
        let receiver = self.expr_with_hint(&m.receiver, data_type)?;

        if let Stage::Reg(t @ (Scalar::Vector(_) | Scalar::Mask(_))) = receiver.stage {
            return self.simd_method(t, receiver, m, hint);
        }
        if (m.method == "len" || m.method == "is_empty")
            && m.args.is_empty()
            && matches!(receiver.stage, Stage::Buffer(_))
        {
            let length = emit!({let buffer=#receiver; __splice_builder.buffer_len(buffer)});
            return Ok(if m.method == "len" {
                Value::new(Stage::Reg(Scalar::Length), length)
            } else {
                Value::new(
                    Stage::Reg(Scalar::Bool),
                    self.bin("Eq", length, emit!(__splice_builder.constant(0))),
                )
            });
        }
        if receiver.stage.is_host() {
            let args = m
                .args
                .iter()
                .map(|arg| self.expr(arg))
                .collect::<Result<Vec<_>>>()?;
            if args.iter().all(|arg| arg.stage.is_host()) {
                let root = self.root;
                let name = &m.method;
                let raw_name = name.to_string();
                if matches!(raw_name.as_str(), "len" | "is_empty") && args.is_empty() {
                    let empty = (name == "is_empty").then(|| emit!(== 0));
                    return Ok(Value::new(
                        Stage::Host,
                        emit!({
                            use #root::frontend::evaluation::{ReadLength as _, Operand};
                            let receiver = &(#receiver);
                            (&Operand::of(receiver)).length(receiver)? #empty
                        }),
                    ));
                }
                if primitive_method(&raw_name) {
                    return Ok(Value::new(
                        Stage::Host,
                        emit!({
                            use #root::frontend::evaluation::{CheckPrimitive as _, Operand};
                            let value = #receiver;
                            (&Operand::of(&value)).check()?;
                            value.#name(#({let arg = #args; (&Operand::of(&arg)).check()?; arg}),*)
                        }),
                    ));
                }
                let expression = Expr::MethodCall(m.clone());
                return Ok(Value::new(Stage::Host, self.required(&expression, hint)?));
            }
        }
        if matches!(
            receiver.stage,
            Stage::Reg(Scalar::Int) | Stage::Word(Scalar::Int)
        ) {
            let method = &m.method;
            if method == "wrapping_neg" && m.args.is_empty() {
                let value = self.reg(receiver, Scalar::Int, &m.receiver)?;
                return Ok(Value::new(
                    Stage::Reg(Scalar::Int),
                    self.bin("Sub", emit!(__splice_builder.constant(0)), value),
                ));
            }
            if matches!(
                method.to_string().as_str(),
                "saturating_add" | "saturating_sub" | "saturating_mul"
            ) && let Some(right) = m.args.first()
                && m.args.len() == 1
            {
                let left = self.reg(receiver, Scalar::Int, &m.receiver)?;
                let right = self.operand(right, Scalar::Int)?;
                let item = syn::parse_quote!(::core::primitive::i32::#method);
                let operation = self.import_call(&item, &[left, right]);
                return Ok(Value::new(Stage::Reg(Scalar::Int), operation));
            }
        }
        let binary = match m.method.to_string().as_str() {
            "min" => Some("Min"),
            "max" => Some("Max"),
            "wrapping_add" => Some("Add"),
            "wrapping_sub" => Some("Sub"),
            "wrapping_mul" => Some("Mul"),
            _ => None,
        };
        if let Some(op) = binary
            && m.args.len() == 1
        {
            let rhs = &m.args[0];
            let left = self.reg(receiver, Scalar::Int, &m.receiver)?;
            let right = self.operand(rhs, Scalar::Int)?;
            return Ok(Value::new(
                Stage::Reg(Scalar::Int),
                self.bin(op, left, right),
            ));
        }
        if receiver.stage.is_host() {
            return err(m, "a host method cannot consume runtime values");
        }
        err(
            m,
            "expected slice.len/is_empty, i32.min/max, or i32.wrapping_add/sub/mul",
        )
    }

    pub(super) fn cast(&self, cast: &ExprCast) -> Result<Value> {
        if Scalar::from_type(&cast.ty, self.types) == Some(Scalar::Int) {
            let mut operand = cast.expr.as_ref();
            loop {
                operand = match operand {
                    Expr::Paren(expr) => &expr.expr,
                    Expr::Group(expr) => &expr.expr,
                    _ => break,
                };
            }
            if let Expr::Call(call) = operand
                && matches!(call.func.as_ref(), Expr::Path(path) if crate::scalar::methods::trait_path(path).is_some())
            {
                return self.call(call, Some(Scalar::Int), Some(&cast.ty));
            }
        }
        let value = self.expr(&cast.expr).or_else(|error| {
            if super::simd::uses_lane(&cast.expr) {
                return Err(error);
            }
            self.host_raw(&cast.expr)
                .map(|code| Value::new(Stage::Host, code))
                .map_err(|_| error)
        })?;
        if value.stage.is_host() {
            let ty = &cast.ty;
            return Ok(Value::new(Stage::Host, emit!((#value) as #ty)));
        }
        let Some(kind @ (Scalar::Int | Scalar::Index)) = Scalar::from_type(&cast.ty, self.types)
        else {
            return err(
                &cast.ty,
                "runtime casts support i32 and usize slice indices",
            );
        };
        let code = match value.stage {
            Stage::Reg(scalar @ (Scalar::Index | Scalar::Length | Scalar::Int))
            | Stage::Word(scalar @ (Scalar::Index | Scalar::Length | Scalar::Int)) => {
                self.reg(value, scalar, cast)?
            }
            Stage::Reg(Scalar::Bool) | Stage::Word(Scalar::Bool) => {
                let bits = self.reg(value, Scalar::Bool, cast)?;
                self.bin("And", bits, emit!(__splice_builder.constant(1)))
            }
            _ => return err(cast, "expected an integer or boolean to cast"),
        };
        Ok(Value::new(Stage::Reg(kind), code))
    }

    pub(super) fn reference(&self, reference: &ExprReference) -> Result<Value> {
        // Explicit reborrows (&*input and &mut *output) keep the same descriptor.
        let target = match &*reference.expr {
            Expr::Unary(x) if matches!(x.op, UnOp::Deref(_)) => &*x.expr,
            expr => expr,
        };
        let value = self.expr(target)?;
        let writable = reference.mutability.is_some();
        match value.stage {
            Stage::Buffer(actual) if !writable || actual.write => Ok(Value::new(
                Stage::Buffer(super::BufferType {
                    write: writable,
                    ..actual
                }),
                value.code,
            )),
            _ => err(reference, "only a runtime slice can be reborrowed here"),
        }
    }

    fn index_operands(&self, index: &ExprIndex) -> Result<(bool, TokenStream, TokenStream)> {
        let slice = self.expr(&index.expr)?;
        let Stage::Buffer(writable) = slice.stage else {
            return err(
                &index.expr,
                "index a runtime &[i32] or &mut [i32] parameter",
            );
        };
        if writable.elem != super::Elem::I32 {
            return err(index, "use Lane::load or Lane::store for typed SIMD slices");
        }
        let value = self.expr(&index.index)?;
        // Every host index above i32::MAX is out of bounds for a runtime
        // slice. Keep the fault on the executed path, even for a literal.
        let offset = if matches!(value.stage, Stage::Host | Stage::Literal(Scalar::Int)) {
            emit!({let i: usize = #value; __splice_builder.constant(i32::try_from(i).unwrap_or(-1))})
        } else {
            self.reg(value, Scalar::Index, &index.index)?
        };
        Ok((writable.write, slice.code, offset))
    }

    pub(super) fn index(&self, index: &ExprIndex) -> Result<Value> {
        let receiver = self.expr(&index.expr)?;
        if receiver.stage.is_host() {
            let offset = self.expr(&index.index)?;
            if !offset.stage.is_host() {
                let root = self.root;
                return Ok(Value::new(
                    Stage::Reg(Scalar::Int),
                    emit!(
                        ::core::result::Result::Err::<#root::hir::Reg, #root::BuildError>(#root::BuildError::staging("runtime index requires a runtime slice"))?
                    ),
                ));
            }
            let read = self.known_index(receiver.code, offset.code);
            let fail = self.bounds_failure();
            return Ok(Value::new(
                Stage::Host,
                emit!({
                    let ::core::option::Option::Some(value) = (#read) else { #fail };
                    value
                }),
            ));
        }
        let (_, slice, offset) = self.index_operands(index)?;

        Ok(Value::new(
            Stage::Reg(Scalar::Int),
            emit!({
                let (buffer, __splice_index) = (#slice, #offset);
                __splice_builder.load(buffer, __splice_index)
            }),
        ))
    }

    pub(super) fn bounds_failure(&self) -> TokenStream {
        if let Some(label) = &self.source_fault {
            let root = self.root;
            emit! {
                __splice_source.transfer = ::core::option::Option::Some(#root::frontend::source::Transfer::Return);
                break #label r#"panic!("kernel slice index out of bounds")"#.to_owned();
            }
        } else {
            let region = self.region();
            emit! { __splice_builder.trap_bounds()?; break #region; }
        }
    }

    pub(super) fn known_index(&self, receiver: TokenStream, offset: TokenStream) -> TokenStream {
        let read = self.known_index_reference(receiver, offset);
        emit!((#read)?.copied())
    }

    pub(super) fn known_index_reference(
        &self,
        receiver: TokenStream,
        offset: TokenStream,
    ) -> TokenStream {
        let root = self.root;
        emit!({
            use #root::frontend::evaluation::{Operand, ReadIndex as _};
            let receiver = &(#receiver);
            let index = #offset;
            (&Operand::of(receiver)).read_index(receiver, index, |values, index| &values[index])
        })
    }

    pub(super) fn store(
        &self,
        index: &ExprIndex,
        rhs: &Expr,
        op: Option<(&str, &BinOp)>,
    ) -> Result<Value> {
        let (writable, slice, offset) = self.index_operands(index)?;
        if !writable {
            return err(index, "writing a kernel slice requires &mut [i32]");
        }
        let value = self.operand(rhs, Scalar::Int)?;

        let combine = op.map(|(op, _)| {
            let combined = self.operator_bin(
                op,
                emit!(__splice_builder.load(buffer, __splice_index)),
                emit!(__splice_value),
            );
            emit!(let __splice_value=#combined;)
        });
        // Rust evaluates the RHS before the place for assignment and primitive
        // compound assignment. The index expression may itself mutate memory.
        Ok(Value::unit(emit!({
            let __splice_value = #value;
            let (buffer, __splice_index) = (#slice, #offset);
            #combine
            __splice_builder.store(buffer, __splice_index, __splice_value);
        })))
    }
}
