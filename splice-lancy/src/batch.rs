use lancy::{NativeKernel, Program};
use splice::{BufferArg, BufferKind, validate_buffers};

pub(super) struct Batch {
    native: NativeKernel,
    signature: Box<[BufferKind]>,
}

impl Batch {
    pub(super) fn compile(projection: &Program, signature: &[BufferKind]) -> Result<Self, String> {
        Ok(Self {
            native: NativeKernel::specialize(projection, &[])?,
            signature: signature.into(),
        })
    }

    pub(super) fn call(&self, args: &[i32], buffers: &mut [BufferArg<'_>]) -> Result<i32, String> {
        if !args.is_empty() {
            return Err(format!("expected 0 scalar arguments, got {}", args.len()));
        }
        validate_buffers(&self.signature, buffers)?;
        let Some((BufferArg::Write(out), inputs)) = buffers.split_last_mut() else {
            unreachable!("a verified batch signature ends with a writable output");
        };

        // into_batch loads every input in order before calling the row kernel.
        // This includes unused row arguments. A short input fails immediately
        // after the common prefix, even if the optimized projection omits it.
        let rows = inputs.iter().fold(out.len(), |n, input| n.min(input.len()));
        let slices = inputs.iter().map(|input| &input.as_slice()[..rows]);
        let mut local = [&[][..]; 16];
        let wide;
        let columns = if inputs.len() <= local.len() {
            for (dst, slice) in local.iter_mut().zip(slices) {
                *dst = slice;
            }
            &local[..inputs.len()]
        } else {
            wide = slices.collect::<Vec<_>>();
            &wide
        };
        let written = self.native.run(columns, &mut out[..rows])?;
        if written != rows {
            return Err("unguarded native batch stopped before the end of its input".into());
        }
        if rows < out.len() {
            return Err("kernel slice index out of bounds".into());
        }
        Ok(rows as i32)
    }
}
