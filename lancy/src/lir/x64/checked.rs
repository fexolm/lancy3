//! Checked arithmetic and side exits shared by arithmetic and slice access.
use super::encode::{Code, EAX, ECX};
use super::frame::{Frame, Home};
use crate::hir::Binary;

#[derive(Default)]
pub(super) struct Failures {
    pub(super) bounds: Vec<usize>,
    zero: Vec<usize>,
    overflow: Vec<usize>,
}

impl Code {
    pub(super) fn divide(
        &mut self,
        op: Binary,
        dst: Home,
        lhs: Home,
        rhs: Home,
        failures: &mut Failures,
    ) -> Result<(), String> {
        self.load(EAX, lhs);
        self.load(ECX, rhs);
        // IDIV would raise #DE for either case. Check even a discarded result.
        self.put(&[0x85, 0xc9]); // test ecx,ecx
        failures.zero.push(self.rel32(&[0x0f, 0x84]));
        self.put(&[0x83, 0xf9, 0xff]); // cmp ecx,-1
        let skip = self.rel32(&[0x0f, 0x85]);
        self.put(&[0x3d]); // cmp eax,INT_MIN
        self.put(&i32::MIN.to_le_bytes());
        failures.overflow.push(self.rel32(&[0x0f, 0x84]));
        self.patch(skip, self.bytes.len())?;
        // IDIV needs EDX:EAX. XMM0 is volatile on Windows and SysV.
        self.put(&[0x66, 0x48, 0x0f, 0x6e, 0xc2]); // movq xmm0,rdx
        self.put(&[0x99, 0xf7, 0xf9]); // cdq; idiv ecx
        if op == Binary::Rem {
            self.put(&[0x89, 0xd0]); // mov eax,edx
        }
        self.put(&[0x66, 0x48, 0x0f, 0x7e, 0xc2]); // movq rdx,xmm0
        self.store(dst, EAX);
        Ok(())
    }

    pub(super) fn failure_exits(
        &mut self,
        frame: &Frame,
        failures: Failures,
    ) -> Result<(), String> {
        for (status, jumps) in [
            (1, failures.bounds),
            (2, failures.zero),
            (3, failures.overflow),
        ] {
            if !jumps.is_empty() {
                let exit = self.bytes.len();
                self.read_pointer(EAX, frame.status_slot);
                self.put(&[0xc7, 0x00, status, 0, 0, 0]); // *status_ptr = status
                self.put(&[0x31, 0xc0, 0xc3]); // xor eax,eax; ret directly to Rust
                for jump in jumps {
                    self.patch(jump, exit)?;
                }
            }
        }
        Ok(())
    }
}
