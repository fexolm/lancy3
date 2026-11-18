//! Checked accesses to caller-owned i32 slices. RDX always points to the
//! current frame; RAX and RCX are scratch on both supported x64 ABIs.

use super::encode::{Code, EAX, ECX, RDX};
use super::frame::{Frame, Home};
use crate::hir::Buffer;

impl Code {
    // Pointer slots are two i32 words. Frame bases are only four-byte
    // aligned; x86-64 MOV permits an unaligned 64-bit pointer load or store.
    pub(super) fn read_pointer(&mut self, reg: u8, slot: usize) {
        self.put(&[0x48]);
        self.memory_op(&[0x8b], reg, RDX, slot);
    }

    fn copy_pointer(&mut self, from: usize, to: usize) {
        self.read_pointer(EAX, from);
        self.put(&[0x48]);
        self.memory_op(&[0x89], EAX, RDX, to);
    }

    pub(super) fn pass_header(&mut self, caller: &Frame, child: &Frame, buffers: &[Buffer]) {
        self.copy_pointer(caller.status_slot, caller.slots + child.status_slot);
        for (i, buffer) in buffers.iter().enumerate() {
            self.copy_pointer(
                caller.buffer_slot(buffer.index()),
                caller.slots + child.buffer_slot(i),
            );
        }
    }

    pub(super) fn buffer_len(&mut self, descriptor: usize, dst: Home) {
        if dst != Home::Dead {
            self.read_pointer(ECX, descriptor);
            self.put(&[0x8b, 0x41, 8]); // mov eax,[rcx+8]: validated len <= i32::MAX
            self.store(dst, EAX);
        }
    }

    fn check_index(&mut self, descriptor: usize, index: Home) -> usize {
        self.read_pointer(ECX, descriptor);
        self.load(EAX, index); // zero-extends to RAX
        self.put(&[0x3b, 0x41, 8]); // cmp eax,[rcx+8]
        self.rel32(&[0x0f, 0x83]) // jae: also rejects every negative i32
    }

    pub(super) fn load_element(&mut self, descriptor: usize, index: Home, dst: Home) -> usize {
        let failure = self.check_index(descriptor, index);
        // A discarded load still has observable bounds-error behavior.
        if dst != Home::Dead {
            self.put(&[0x48, 0x8b, 0x09]); // mov rcx,[rcx]: data
            self.put(&[0x8b, 0x04, 0x81]); // mov eax,[rcx+rax*4]
            self.store(dst, EAX);
        }
        failure
    }

    pub(super) fn store_element(&mut self, descriptor: usize, index: Home, value: Home) -> usize {
        let failure = self.check_index(descriptor, index);
        self.put(&[0x48, 0x8b, 0x09]); // mov rcx,[rcx]: data
        match value {
            Home::Register(r) => {
                if r >= 8 {
                    self.put(&[0x44]); // REX.R
                }
                self.put(&[0x89, 0x04 | ((r & 7) << 3), 0x81]);
            }
            Home::Immediate(n) => {
                self.put(&[0xc7, 0x04, 0x81]); // mov dword [rcx+rax*4],imm32
                self.put(&n.to_le_bytes());
            }
            Home::Slot(_) => {
                self.put(&[0x48, 0x8d, 0x0c, 0x81]); // lea rcx,[rcx+rax*4]
                self.load(EAX, value);
                self.put(&[0x89, 0x01]); // mov [rcx],eax
            }
            Home::Dead => unreachable!(),
        }
        failure
    }
}
