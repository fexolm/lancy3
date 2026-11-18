//! Instruction bytes, relative displacements and moves between machine locations.
use super::super::frame::{EAX, ECX, Frame, Home, LINK_SLOTS, RDX};
use crate::CompileError;
use crate::hir::{Binary, Scale, Type};
use crate::lir::FunctionId;

pub(super) const ENDBR64: [u8; 4] = [0xf3, 0x0f, 0x1e, 0xfa];

#[derive(Default)]
pub(super) struct Code {
    pub(super) bytes: Vec<u8>,
    pub(super) calls: Vec<(usize, FunctionId)>,
}

#[derive(Default)]
pub(super) struct Failures {
    pub(super) bounds: Vec<usize>,
    zero: Vec<usize>,
    overflow: Vec<usize>,
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

    pub(super) fn patch(&mut self, offset: usize, target: usize) -> Result<(), CompileError> {
        let rel = i32::try_from(target as i64 - (offset + 4) as i64)
            .map_err(|_| CompileError::BranchOutOfRange { offset, target })?;
        self.bytes[offset..offset + 4].copy_from_slice(&rel.to_le_bytes());
        Ok(())
    }

    fn rex(&mut self, reg: u8, rm: u8) {
        let bits = ((reg >> 3) << 2) | (rm >> 3);
        self.bytes.push(0x48 | bits);
    }

    fn register_op(&mut self, opcode: &[u8], reg: u8, rm: u8) {
        self.rex(reg, rm);
        self.put(opcode);
        self.bytes.push(0xc0 | ((reg & 7) << 3) | (rm & 7));
    }

    pub(super) fn memory_op(&mut self, opcode: &[u8], reg: u8, base: u8, slot: usize) {
        // The graph's workspace and input-size checks bound this displacement.
        // Only rcx (argument table) and rdx (current frame) are base registers.
        let offset = (slot * 8) as i32;
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

    pub(super) fn pass_status(&mut self, caller: &Frame, child: &Frame) {
        self.memory_op(&[0x8b], EAX, RDX, caller.status_slot);
        self.memory_op(&[0x89], EAX, RDX, caller.slots + child.status_slot);
    }

    pub(super) fn bounds_check(&mut self, index: Home, len: Home) -> usize {
        self.load(EAX, index);
        self.load(ECX, len);
        self.put(&[0x39, 0xc8]); // cmp eax,ecx
        self.rel32(&[0x0f, 0x83]) // jae
    }

    pub(super) fn ptr_offset(&mut self, dst: Home, base: Home, index: Home, scale: Scale) {
        self.load(EAX, base);
        self.load(ECX, index);
        self.put(&[0x48, 0x63, 0xc9]); // movsxd rcx,ecx
        self.put(&[0x48, 0x8d, 0x04, ((scale as u8) << 6) | 0x08]); // lea rax,[rax+rcx*scale]
        self.store(dst, EAX);
    }

    pub(super) fn load_at(&mut self, dst: Home, ty: Type, address: Home) {
        self.load(ECX, address);
        if ty == Type::Ptr {
            self.put(&[0x48]);
        }
        self.put(&[0x8b, 0x01]); // mov eax/rax,[rcx]
        self.store(dst, EAX);
    }

    pub(super) fn store_at(&mut self, ty: Type, address: Home, value: Home) {
        self.load(ECX, address);
        self.load(EAX, value);
        if ty == Type::Ptr {
            self.put(&[0x48]);
        }
        self.put(&[0x89, 0x01]); // mov [rcx],eax/rax
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

    pub(super) fn return_value(&mut self, value: Home, root: bool) {
        self.load(EAX, value);
        self.put(if root { &[0xc3] } else { &[0xff, 0x22] });
    }

    fn frame_adjust(&mut self, slots: usize, subtract: bool) {
        if slots > 0 {
            self.alu_immediate(if subtract { 5 } else { 0 }, RDX, (slots * 8) as i32);
        }
    }

    pub(super) fn call(
        &mut self,
        caller: &Frame,
        child_id: FunctionId,
        dst: Home,
        args: &[Home],
    ) -> Result<(), CompileError> {
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

    pub(super) fn divide(
        &mut self,
        op: Binary,
        dst: Home,
        lhs: Home,
        rhs: Home,
        failures: &mut Failures,
    ) -> Result<(), CompileError> {
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
    ) -> Result<(), CompileError> {
        for (status, jumps) in [
            (1, failures.bounds),
            (2, failures.zero),
            (3, failures.overflow),
        ] {
            if !jumps.is_empty() {
                let exit = self.bytes.len();
                self.memory_op(&[0x8b], EAX, RDX, frame.status_slot);
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
