//! Encoding of scalar integer instructions and the internal call convention.

use super::frame::{Frame, Home, LINK_SLOTS};
use crate::hir::{Binary, Reg};
use crate::lir::scalar::ReturnValue;

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
            Home::Slot(slot) => self.memory_op(&[0x8b], reg, RDX, slot),
            Home::Immediate(value) => {
                self.rex(0, reg);
                self.bytes.push(0xb8 + (reg & 7));
                self.put(&value.to_le_bytes());
            }
            Home::Dead => unreachable!("validated IR cannot read an unused register"),
        }
    }

    pub(super) fn store(&mut self, dest: Home, reg: u8) {
        match dest {
            Home::Register(other) => self.load(other, Home::Register(reg)),
            Home::Slot(slot) => self.memory_op(&[0x89], reg, RDX, slot),
            Home::Dead | Home::Immediate(_) => {}
        }
    }

    pub(super) fn assign(&mut self, dest: Home, source: Home) {
        if dest == source || matches!(dest, Home::Dead | Home::Immediate(_)) {
            return;
        }
        match (dest, source) {
            (Home::Register(reg), _) => self.load(reg, source),
            (_, Home::Register(reg)) => self.store(dest, reg),
            (Home::Slot(slot), Home::Immediate(value)) => {
                self.memory_op(&[0xc7], 0, RDX, slot);
                self.put(&value.to_le_bytes());
            }
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

    fn alu(&mut self, opcode: u8, immediate_extension: u8, reg: u8, rhs: Home) {
        match rhs {
            Home::Immediate(n) => self.alu_immediate(immediate_extension, reg, n),
            Home::Register(r) => self.register_op(&[opcode], reg, r),
            Home::Slot(s) => self.memory_op(&[opcode], reg, RDX, s),
            Home::Dead => unreachable!(),
        }
    }

    pub(super) fn cmp(&mut self, lhs: Home, rhs: Home) {
        let reg = match lhs {
            Home::Register(r) => r,
            _ => {
                self.load(EAX, lhs);
                EAX
            }
        };
        self.alu(0x3b, 7, reg, rhs);
    }

    pub(super) fn test_mask(&mut self, source: Home, mask: i32) {
        // For small masks a byte TEST observes the same selected bits.
        let byte = u8::try_from(mask).ok();
        let opcode = if byte.is_some() { 0xf6 } else { 0xf7 };
        match source {
            Home::Register(r) => self.register_op(&[opcode], 0, r),
            Home::Slot(slot) => self.memory_op(&[opcode], 0, RDX, slot),
            _ => {
                self.load(EAX, source);
                self.register_op(&[opcode], 0, EAX);
            }
        }
        if let Some(byte) = byte {
            self.bytes.push(byte);
        } else {
            self.put(&mask.to_le_bytes());
        }
    }

    pub(super) fn binary(&mut self, op: Binary, dst: Home, lhs: Home, rhs: Home) {
        if dst == Home::Dead {
            return;
        }
        if let Some(cc) = comparison(op) {
            self.cmp(lhs, rhs);
            // setcc al; movzx eax,al; neg eax -> mask with true == -1.
            self.put(&[0x0f, 0x90 | cc, 0xc0, 0x0f, 0xb6, 0xc0, 0xf7, 0xd8]);
            self.store(dst, EAX);
            return;
        }
        let acc = match dst {
            Home::Register(r) if dst != rhs || lhs == rhs => r,
            _ => EAX,
        };
        self.load(acc, lhs);
        match op {
            Binary::Add => self.alu(0x03, 0, acc, rhs),
            Binary::Sub => self.alu(0x2b, 5, acc, rhs),
            Binary::And => self.alu(0x23, 4, acc, rhs),
            Binary::Or => self.alu(0x0b, 1, acc, rhs),
            Binary::Xor => self.alu(0x33, 6, acc, rhs),
            Binary::Min | Binary::Max => {
                let opcode = if op == Binary::Min { 0x4f } else { 0x4c }; // cmovg / cmovl
                self.alu(0x3b, 7, acc, rhs); // cmp acc,rhs
                match rhs {
                    Home::Register(r) => self.register_op(&[0x0f, opcode], acc, r),
                    Home::Slot(s) => self.memory_op(&[0x0f, opcode], acc, RDX, s),
                    Home::Immediate(_) => {
                        self.load(ECX, rhs); // MOV preserves the comparison flags.
                        self.register_op(&[0x0f, opcode], acc, ECX);
                    }
                    Home::Dead => unreachable!(),
                }
            }
            Binary::Mul => match rhs {
                Home::Immediate(n) => {
                    if let Ok(short) = i8::try_from(n) {
                        self.register_op(&[0x6b], acc, acc);
                        self.bytes.push(short as u8);
                    } else {
                        self.register_op(&[0x69], acc, acc);
                        self.put(&n.to_le_bytes());
                    }
                }
                Home::Register(r) => self.register_op(&[0x0f, 0xaf], acc, r),
                Home::Slot(s) => self.memory_op(&[0x0f, 0xaf], acc, RDX, s),
                Home::Dead => unreachable!(),
            },
            Binary::Shl | Binary::Shr => {
                let extension = if matches!(op, Binary::Shl) { 4 } else { 7 };
                if let Home::Immediate(n) = rhs {
                    self.register_op(&[0xc1], extension, acc);
                    self.bytes.push((n & 31) as u8);
                } else {
                    self.load(ECX, rhs);
                    self.register_op(&[0xd3], extension, acc);
                }
            }
            _ => unreachable!(),
        }
        self.store(dst, acc);
    }

    pub(super) fn return_value(&mut self, value: ReturnValue, frame: &Frame, root: bool) {
        self.load(
            EAX,
            match value {
                ReturnValue::Reg(reg) => frame.homes[reg.0 as usize],
                ReturnValue::Constant(n) => Home::Immediate(n),
            },
        );
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
        child: &Frame,
        dst: Home,
        args: &[Reg],
    ) -> Result<(), String> {
        for (i, arg) in args.iter().enumerate() {
            if child.homes[i] != Home::Dead {
                self.assign(
                    Home::Slot(caller.slots + LINK_SLOTS + i),
                    caller.homes[arg.0 as usize],
                );
            }
        }
        for &(reg, slot) in &caller.saves {
            self.store(Home::Slot(slot), reg);
        }
        self.frame_adjust(caller.slots, false);
        let next = self.rel32(&[0x48, 0x8d, 0x05]); // lea rax,[rip+continuation]
        self.put(&[0x48, 0x89, 0x02]); // mov [rdx],rax
        let target = self.rel32(&[0xe9]);
        self.calls.push((target, child_id));
        self.patch(next, self.bytes.len())?;
        self.put(&ENDBR64); // indirect jump target of the callee return
        self.frame_adjust(caller.slots, true);
        for &(reg, slot) in &caller.saves {
            self.load(reg, Home::Slot(slot));
        }
        self.store(dst, EAX);
        Ok(())
    }
}

pub(super) fn comparison(op: Binary) -> Option<u8> {
    Some(match op {
        Binary::Eq => 4,
        Binary::Ne => 5,
        Binary::Lt => 12,
        Binary::Le => 14,
        Binary::Gt => 15,
        Binary::Ge => 13,
        _ => return None,
    })
}
