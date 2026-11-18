//! Emit native column loops with SSE2 operations and exact guard exits.

mod regalloc;

use crate::hir::{Assumption, Program};
use crate::lir::batch::{Action, Batch, Binary, Operand, Step};
use std::collections::HashMap;

struct Code {
    bytes: Vec<u8>,
    literals: Vec<(usize, i32)>,
    exits: Vec<usize>,
}

impl Code {
    fn put(&mut self, bytes: &[u8]) {
        self.bytes.extend_from_slice(bytes);
    }

    fn hole(&mut self) -> usize {
        let pos = self.bytes.len();
        self.put(&[0; 4]);
        pos
    }

    fn patch(&mut self, pos: usize, target: usize) -> Result<(), String> {
        let delta = i32::try_from(target as i64 - (pos as i64 + 4))
            .map_err(|_| "x64 code exceeds RIP-relative addressing range")?;
        self.bytes[pos..pos + 4].copy_from_slice(&delta.to_le_bytes());
        Ok(())
    }

    fn branch(&mut self, condition: u8) -> usize {
        self.put(&[0x0f, condition]);
        self.hole()
    }

    fn exit_if(&mut self, condition: u8) {
        let pos = self.branch(condition);
        self.exits.push(pos);
    }

    fn spill_address(&mut self, reg: u8, slot: u32) {
        let offset = slot * 16;
        let rm = (reg << 3) | 3; // [r11 + offset], REX.B emitted by simd()
        match offset {
            0 => self.put(&[rm]),
            1..=127 => self.put(&[0x40 | rm, offset as u8]),
            _ => {
                self.put(&[0x80 | rm]);
                self.put(&offset.to_le_bytes());
            }
        }
    }

    // Legacy 66 0F SSE2 instruction, XMM register and register/memory operand.
    fn simd(&mut self, opcode: u8, reg: u8, source: Operand) {
        self.put(&[0x66]);
        if matches!(source, Operand::Spill(_)) {
            self.put(&[0x41]);
        }
        self.put(&[0x0f, opcode]);
        match source {
            Operand::VectorReg(rm) => self.put(&[0xc0 | (reg << 3) | rm]),
            Operand::Spill(slot) => self.spill_address(reg, slot),
            Operand::Immediate(n) => {
                self.put(&[(reg << 3) | 5]);
                let pos = self.hole();
                self.literals.push((pos, n));
            }
        }
    }

    fn load(&mut self, reg: u8, source: Operand) {
        if source != Operand::VectorReg(reg) {
            self.simd(0x6f, reg, source); // movdqa
        }
    }

    fn save(&mut self, dest: Operand, reg: u8) {
        match dest {
            Operand::VectorReg(out) => self.load(out, Operand::VectorReg(reg)),
            Operand::Spill(_) => self.simd(0x7f, reg, dest),
            Operand::Immediate(_) => unreachable!("literal destination"),
        }
    }

    fn column_ptr(&mut self, reg: u8, column: u16) {
        // mov reg, [rcx + column*8]
        let offset = u32::from(column) * 8;
        self.put(&[0x48 | ((reg >> 3) << 2), 0x8b]);
        let rm = ((reg & 7) << 3) | 1;
        match offset {
            0 => self.put(&[rm]),
            1..=127 => self.put(&[0x40 | rm, offset as u8]),
            _ => {
                self.put(&[0x80 | rm]);
                self.put(&offset.to_le_bytes());
            }
        }
    }

    fn memory(&mut self, vector: bool, write: bool, xmm: u8, base: u8) {
        // movdqu/movd xmm, [base + rax*4], or its store encoding.
        self.put(&[if vector { 0xf3 } else { 0x66 }]);
        if base >= 8 {
            self.put(&[0x41]);
        }
        let opcode = match (vector, write) {
            (true, false) => 0x6f,
            (true, true) => 0x7f,
            (false, false) => 0x6e,
            (false, true) => 0x7e,
        };
        self.put(&[0x0f, opcode, (xmm << 3) | 4, 0x80 | (base & 7)]);
    }

    fn mul4(&mut self, left: Operand, right: Operand) {
        // Specialization frequently turns a column into a uniform integer.
        // For 2^k and 2^k+1 a shift and optional add replace the shuffle path.
        let constant = match (left, right) {
            (Operand::Immediate(factor), input) | (input, Operand::Immediate(factor)) => {
                Some((factor as u32, input))
            }
            _ => None,
        };
        if let Some((factor, input)) = constant {
            let extra = factor & 1;
            let base = factor - extra;
            if base.is_power_of_two() {
                self.load(0, input);
                self.put(&[0x66, 0x0f, 0x72, 0xf0, base.trailing_zeros() as u8]); // pslld xmm0,k
                if extra != 0 {
                    self.simd(0xfe, 0, input); // paddd
                }
                return;
            }
        }
        // PMULUDQ computes lanes 0 and 2. Low dwords are identical for
        // signed and unsigned multiplication modulo 2^32.
        self.load(0, left);
        self.load(2, right);
        self.simd(0x70, 1, Operand::VectorReg(0)); // pshufd 1,0,[1,1,3,3]
        self.put(&[0xf5]);
        self.simd(0xf4, 0, Operand::VectorReg(2)); // even products
        self.simd(0x70, 2, Operand::VectorReg(2));
        self.put(&[0xf5]);
        self.simd(0xf4, 1, Operand::VectorReg(2)); // odd products
        self.put(&[0x0f, 0xc6, 0xc1, 0x88]); // shufps: [even0,even2,odd1,odd3]
        self.simd(0x70, 0, Operand::VectorReg(0));
        self.put(&[0xd8]); // [even0,odd1,even2,odd3]
    }

    fn binary(
        &mut self,
        dest: Operand,
        kind: Binary,
        mut left: Operand,
        mut right: Operand,
        vector: bool,
    ) {
        if vector && matches!(kind, Binary::Mul) {
            self.mul4(left, right);
            self.save(dest, 0);
            return;
        }
        let opcode = match kind {
            Binary::Add => 0xfe,
            Binary::Sub => 0xfa,
            Binary::Mul => 0xf4,
            Binary::Eq => 0x76,
            Binary::Gt => 0x66,
            Binary::And => 0xdb,
            Binary::Or => 0xeb,
            Binary::Xor => 0xef,
        };
        if dest == right && !matches!(kind, Binary::Sub | Binary::Gt) {
            std::mem::swap(&mut left, &mut right);
        }
        // Legacy SSE clobbers its first operand. Protect a distinct RHS
        // whose register was reallocated to the result.
        let reg = match dest {
            Operand::VectorReg(r) if dest != right || left == right => r,
            _ => 0,
        };
        self.load(reg, left);
        self.simd(opcode, reg, right);
        self.save(dest, reg);
    }

    fn step(&mut self, step: &Step, vector: bool, pointers: &Pointers) {
        let dest = step.dest;
        match step.action {
            Action::Load(column) => {
                let base = pointers.base(self, column);
                let reg = if let Operand::VectorReg(r) = dest {
                    r
                } else {
                    0
                };
                self.memory(vector, false, reg, base);
                self.save(dest, reg);
            }
            Action::Binary(kind, a, b) => self.binary(dest, kind, a, b, vector),
            Action::Select(mask, yes, no) => {
                self.load(0, mask);
                self.load(1, Operand::VectorReg(0));
                self.simd(0xdb, 0, yes); // pand
                self.simd(0xdf, 1, no); // pandn
                self.simd(0xeb, 0, Operand::VectorReg(1));
                self.save(dest, 0);
            }
        }
    }

    fn body(&mut self, lir: &Batch, guards: &[Assumption], vector: bool, pointers: &Pointers) {
        for guard in guards {
            let base = pointers.base(self, guard.column);
            if vector {
                self.memory(true, false, 0, base);
                self.simd(0x76, 0, Operand::Immediate(guard.value));
                self.put(&[0x44, 0x0f, 0x50, 0xd0]); // movmskps r10d,xmm0
                self.put(&[0x41, 0x83, 0xfa, 0x0f]); // cmp r10d,15
            } else {
                if base >= 8 {
                    self.put(&[0x41]);
                }
                self.put(&[0x81, 0x3c, 0x80 | (base & 7)]); // cmp dword [base+rax*4],imm32
                self.put(&guard.value.to_le_bytes());
            }
            self.exit_if(0x85); // jne
        }
        for step in &lir.steps {
            self.step(step, vector, pointers);
        }
        let reg = if let Operand::VectorReg(r) = lir.result {
            r
        } else {
            self.load(0, lir.result);
            0
        };
        self.memory(vector, true, reg, 2); // [rdx + rax*4]
    }

    fn finish(mut self) -> Result<Vec<u8>, String> {
        // All pool entries have four identical lanes and 16-byte alignment.
        if !self.literals.is_empty() {
            let end = self.bytes.len().next_multiple_of(16);
            self.bytes.resize(end, 0);
        }
        let mut locations = HashMap::new();
        for (pos, n) in std::mem::take(&mut self.literals) {
            let target = *locations.entry(n).or_insert_with(|| {
                let target = self.bytes.len();
                for _ in 0..4 {
                    self.put(&n.to_le_bytes());
                }
                target
            });
            self.patch(pos, target)?;
        }
        Ok(self.bytes)
    }
}

// Cache two distinct input pointers in registers if the whole kernel fits.
// With more columns RCX continues to point to the table and R10 is scratch.
struct Pointers {
    first: Option<u16>,
    second: Option<u16>,
}

impl Pointers {
    fn prepare(code: &mut Code, lir: &Batch, guards: &[Assumption]) -> Self {
        let mut columns = lir
            .steps
            .iter()
            .filter_map(|step| match step.action {
                Action::Load(col) => Some(col),
                _ => None,
            })
            .chain(guards.iter().map(|g| g.column));
        let first = columns.next();
        let mut others = columns.filter(|col| Some(*col) != first);
        let second = others
            .next()
            .filter(|second| others.all(|col| col == *second));
        if let Some(col) = first {
            code.column_ptr(9, col);
        }
        if let Some(col) = second {
            code.column_ptr(1, col);
        }
        Self { first, second }
    }

    fn base(&self, code: &mut Code, column: u16) -> u8 {
        if Some(column) == self.first {
            return 9;
        }
        if Some(column) == self.second {
            return 1;
        }
        code.column_ptr(10, column);
        10
    }
}

/// The host maps the returned bytes at a 16-byte aligned executable address
/// and passes the indicated number of aligned 16-byte spill slots as argument 4.
pub(crate) fn emit(
    program: &Program,
    guards: &[Assumption],
    vectorize: bool,
) -> Result<(Vec<u8>, usize), String> {
    if guards
        .iter()
        .any(|g| usize::from(g.column) >= program.input_count())
    {
        return Err("x64 guard refers to an unknown input".into());
    }
    let lir = regalloc::lower(program, vectorize)?;
    let mut code = Code {
        bytes: Vec::new(),
        literals: Vec::new(),
        exits: Vec::new(),
    };
    code.put(&[0xf3, 0x0f, 0x1e, 0xfa]); // endbr64
    if lir.spill_slots != 0 {
        // Host-owned scratch keeps this a leaf with a fixed RSP on both ABIs.
        #[cfg(windows)]
        code.put(&[0x4d, 0x89, 0xcb]); // mov r11,r9
        #[cfg(not(windows))]
        code.put(&[0x49, 0x89, 0xcb]); // mov r11,rcx
    }
    #[cfg(not(windows))]
    {
        code.put(&[0x49, 0x89, 0xd0]); // r8 = len
        code.put(&[0x48, 0x89, 0xf2]); // rdx = output
        code.put(&[0x48, 0x89, 0xf9]); // rcx = columns
    }
    code.put(&[0x31, 0xc0]); // xor eax,eax: committed row count
    code.put(&[0x4d, 0x85, 0xc0]); // test r8,r8
    code.exit_if(0x84);
    let pointers = Pointers::prepare(&mut code, &lir, guards);
    if vectorize {
        code.put(&[0x49, 0x83, 0xf8, 4]); // cmp r8,4
        let skip = code.branch(0x82); // jb
        code.put(&[0x49, 0x83, 0xe8, 3]); // exclusive upper bound for a complete vector
        let start = code.bytes.len();
        code.body(&lir, guards, true, &pointers);
        code.put(&[0x48, 0x83, 0xc0, 4]); // add rax,4
        code.put(&[0x4c, 0x39, 0xc0]); // cmp rax,r8
        let back = code.branch(0x82); // jb
        code.patch(back, start)?;
        code.put(&[0x49, 0x83, 0xc0, 3]); // restore original len
        code.put(&[0x4c, 0x39, 0xc0]); // cmp rax,r8
        code.exit_if(0x84);
        code.patch(skip, code.bytes.len())?;
    }
    let start = code.bytes.len();
    code.body(&lir, guards, false, &pointers);
    code.put(&[0x48, 0xff, 0xc0]); // inc rax
    code.put(&[0x4c, 0x39, 0xc0]); // cmp rax,r8
    let back = code.branch(0x82);
    code.patch(back, start)?;
    let end = code.bytes.len();
    for pos in std::mem::take(&mut code.exits) {
        code.patch(pos, end)?;
    }
    code.put(&[0xc3]); // ret
    Ok((code.finish()?, lir.spill_slots))
}
