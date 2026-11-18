use super::{Binary, Buffer, BufferKind, Function, FunctionBuilder};
use std::sync::Arc;

impl Function {
    /// Map a scalar row function across columns. Each scalar parameter becomes
    /// an input slice. The last slice receives one value per output element.
    /// Returns the number of rows written; a short input traps at its first
    /// missing element with previously written rows preserved.
    pub fn into_batch(self) -> Result<Self, String> {
        if !self.buffer_kinds().is_empty() {
            return Err("batch row parameters must be scalar i32 values".into());
        }
        let inputs = self.arity();
        let output = Buffer(u32::try_from(inputs).map_err(|_| "too many batch inputs")?);
        let mut kinds = vec![BufferKind::Read; inputs];
        kinds.push(BufferKind::Write);
        let mut b = FunctionBuilder::with_buffers(0, kinds);
        let end = b.buffer_len(output);
        let one = b.constant(1);
        let row = b.constant(0);
        let head = b.label();
        let body = b.label();
        let done = b.label();
        b.mark(head);
        let has_row = b.binary(Binary::Lt, row, end);
        b.branch(has_row, body, done);
        b.mark(body);
        let args: Vec<_> = (0..output.0).map(|i| b.load(Buffer(i), row)).collect();
        let row_function = Arc::new(self);
        let value = b.call_shared(Arc::clone(&row_function), &args);
        b.store(output, row, value);
        let next = b.binary(Binary::Add, row, one);
        b.assign(row, next);
        b.jump(head);
        b.mark(done);
        b.ret(row);
        let mut batch = b.finish()?;
        batch.batch_row = Some(row_function);
        Ok(batch)
    }
}
