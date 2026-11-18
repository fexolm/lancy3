//! Encoding of scalar integer instructions and the internal call convention.

use super::frame::{Frame, Home, LINK_SLOTS};
use crate::hir::Binary;

pub(super) const EAX: u8 = 0;
pub(super) const ECX: u8 = 1;
pub(super) const RDX: u8 = 2;
pub(super) const ENDBR64: [u8; 4] = [0xf3, 0x0f, 0x1e, 0xfa];

#[derive(Default)]
pub(super) struct Code {
    pub(super) bytes: Vec<u8>,
    pub(super) calls: Vec<(usize, usize)>,
}

impl Code {
    pub(super) fn put(&mut self, bytes: &[u8]) {
        self.bytes.extend_from_slice(bytes);
    }

    pub(super) fn rel32(&mut self, op: &[u8]) -> usize {
        self.put(op);
        let offset = self.bytes.len();
        self.put(&[0; 4]);
        offset
    }

    pub(super) fn patch(&mut self, offset: usize, target: usize) -> Result<(), String> {
        let rel = i32::try_from(target as i64 - offset as i64 - 4)
            .map_err(|_| "scalar jump exceeds x64 rel32 range")?;
        self.bytes[offset..offset + 4].copy_from_slice(&rel.to_le_bytes());
        Ok(())
    }

    fn rex(&mut self, reg: u8, rm: u8) {
        let bits = ((reg >> 3) << 2) | (rm >> 3);
        if bits != 0 {
            self.bytes.push(0x40 | bits);
        }
    }

    pub(super) fn register_op(&mut self, opcode: &[u8], reg: u8, rm: u8) {
        self.rex(reg, rm);
        self.put(opcode);
        self.bytes.push(0xc0 | ((reg & 7) << 3) | (rm & 7));
    }

    pub(super) fn memory_op(&mut self, opcode: &[u8], reg: u8, base: u8, slot: usize) {
        // The graph's workspace and input-size checks bound this displacement.
        // Only rcx (argument table) and rdx (current frame) are base registers.
        let offset = (slot * 4) as i32;
        self.rex(reg, base);
        self.put(opcode);
        let modrm = ((reg & 7) << 3) | base;
        match offset {
            0 => self.bytes.push(modrm),
            1..=127 => self.put(&[0x40 | modrm, offset as u8]),
            _ => {
                self.bytes.push(0x80 | modrm);
                self.put(&offset.to_le_bytes());
            }
        }
    }

    pub(super) fn load(&mut self, reg: u8, source: Home) {
        match source {
            Home::Register(other) if other != reg => self.register_op(&[0x8b], reg, other),
            Home::Register(_) => {}
            Home::Slot(slot) => self.memory_op(&[0x8b], reg, RDX, slot as usize),
        }
    }

    pub(super) fn store(&mut self, dest: Home, reg: u8) {
        match dest {
            Home::Register(other) => self.load(other, Home::Register(reg)),
            Home::Slot(slot) => self.memory_op(&[0x89], reg, RDX, slot as usize),
        }
    }

    pub(super) fn assign(&mut self, dest: Home, source: Home) {
        if dest == source {
            return;
        }
        match (dest, source) {
            (Home::Register(reg), _) => self.load(reg, source),
            (_, Home::Register(reg)) => self.store(dest, reg),
            _ => {
                self.load(EAX, source);
                self.store(dest, EAX);
            }
        }
    }

    fn alu_immediate(&mut self, extension: u8, reg: u8, imm: i32) {
        if let Ok(short) = i8::try_from(imm) {
            self.register_op(&[0x83], extension, reg);
            self.bytes.push(short as u8);
        } else {
            self.register_op(&[0x81], extension, reg);
            self.put(&imm.to_le_bytes());
        }
    }

    pub(super) fn constant(&mut self, dst: Home, n: i32) {
        self.put(&[0xb8]); // mov eax,imm32
        self.put(&n.to_le_bytes());
        self.store(dst, EAX);
    }

    pub(super) fn binary(&mut self, op: Binary, dst: Home, lhs: Home, rhs: Home) {
        self.load(EAX, lhs);
        self.load(ECX, rhs);
        match op {
            Binary::Add => self.put(&[0x01, 0xc8]), // add eax,ecx
            Binary::Sub => self.put(&[0x29, 0xc8]),
            Binary::And => self.put(&[0x21, 0xc8]),
            Binary::Or => self.put(&[0x09, 0xc8]),
            Binary::Xor => self.put(&[0x31, 0xc8]),
            Binary::Mul => self.put(&[0x0f, 0xaf, 0xc1]), // imul eax,ecx
            Binary::Shl => self.put(&[0xd3, 0xe0]),       // shl eax,cl
            Binary::Shr => self.put(&[0xd3, 0xf8]),       // sar eax,cl
            Binary::Min | Binary::Max => {
                self.put(&[0x39, 0xc8]); // cmp eax,ecx
                let cc = if op == Binary::Min { 0x4f } else { 0x4c };
                self.put(&[0x0f, cc, 0xc1]); // cmovg/cmovl eax,ecx
            }
            Binary::Div | Binary::Rem => unreachable!("checked division uses IDIV"),
            cmp => {
                self.put(&[0x39, 0xc8]);
                let cc = comparison(cmp);
                // setcc al; movzx eax,al; neg eax -> boolean mask.
                self.put(&[0x0f, 0x90 | cc, 0xc0, 0x0f, 0xb6, 0xc0, 0xf7, 0xd8]);
            }
        }
        self.store(dst, EAX);
    }

    pub(super) fn return_value(&mut self, value: Home, root: bool) {
        self.load(EAX, value);
        self.put(if root { &[0xc3] } else { &[0xff, 0x22] });
    }

    fn frame_adjust(&mut self, slots: usize, subtract: bool) {
        if slots > 0 {
            self.bytes.push(0x48); // 64-bit address
            self.alu_immediate(if subtract { 5 } else { 0 }, RDX, (slots * 4) as i32);
        }
    }

    pub(super) fn call(
        &mut self,
        caller: &Frame,
        child_id: usize,
        dst: Home,
        args: &[Home],
    ) -> Result<(), String> {
        for (i, arg) in args.iter().enumerate() {
            self.assign(Home::Slot((caller.slots + LINK_SLOTS + i) as u32), *arg);
        }
        for i in 0..4u8 {
            let (reg, slot) = (8 + i, caller.save_base + i as usize);
            self.store(Home::Slot(slot as u32), reg);
        }
        self.frame_adjust(caller.slots, false);
        let next = self.rel32(&[0x48, 0x8d, 0x05]); // lea rax,[rip+continuation]
        self.put(&[0x48, 0x89, 0x02]); // mov [rdx],rax
        let target = self.rel32(&[0xe9]);
        self.calls.push((target, child_id));
        self.patch(next, self.bytes.len())?;
        self.put(&ENDBR64); // indirect jump target of the callee return
        self.frame_adjust(caller.slots, true);
        for i in 0..4u8 {
            let (reg, slot) = (8 + i, caller.save_base + i as usize);
            self.load(reg, Home::Slot(slot as u32));
        }
        self.store(dst, EAX);
        Ok(())
    }
}

fn comparison(op: Binary) -> u8 {
    match op {
        Binary::Eq => 4,
        Binary::Ne => 5,
        Binary::Lt => 12,
        Binary::Le => 14,
        Binary::Gt => 15,
        Binary::Ge => 13,
        _ => unreachable!("comparison opcode expected"),
    }
}
