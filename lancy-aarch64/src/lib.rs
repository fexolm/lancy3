//! AArch64 instruction encoding into a byte buffer supplied by the caller.
//!
//! The caller selects an exact instruction form and supplies its operands.
//! PC-relative operands use target addresses, so a JIT can emit a branch
//! once it knows the destination or patch it after assigning block addresses.
//!
//! ```
//! use lancy_aarch64::{Code, Encoder, Operand, Register};
//!
//! let mut buffer = [0; 4];
//! let mut encoder = Encoder::new(0x1000, &mut buffer)?;
//! encoder.push(
//!     Code::Add_64_addsub_imm,
//!     &[Operand::reg(Register::X0), Operand::reg(Register::X1), Operand::imm(7)],
//! )?;
//! assert_eq!(encoder.bytes(), &[0x20, 0x1c, 0x00, 0x91]);
//! # Ok::<(), lancy_aarch64::Error>(())
//! ```
#![no_std]
#![forbid(unsafe_code)]

pub use decoderant_a64::{
    Arrangement, Code, ConditionCode, EncodeError, ExtendKind, Feature, FeatureSet, Instruction,
    MemoryMode, Operand, PredQual, Register, ShiftKind,
};

const WORD_BYTES: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("AArch64 code address {0:#x} is not 4-byte aligned")]
    UnalignedAddress(u64),
    #[error("AArch64 code buffer extends beyond the address space")]
    AddressOverflow,
    #[error("AArch64 code buffer has no space for another instruction")]
    BufferFull,
    #[error("byte offset {0} does not identify an emitted AArch64 instruction")]
    InvalidPatch(usize),
    #[error(transparent)]
    Encoding(#[from] EncodeError),
}

/// A stream of little-endian AArch64 instructions in caller-owned storage.
///
/// The code generator is responsible for selecting instructions available on
/// its target. See `Code::op_code().is_available(&features)`.
pub struct Encoder<'a> {
    start_pc: u64,
    buffer: &'a mut [u8],
    len: usize,
}

impl<'a> Encoder<'a> {
    /// `start_pc` is the target address of the first instruction.
    /// The buffer itself can have any alignment.
    pub fn new(start_pc: u64, buffer: &'a mut [u8]) -> Result<Self, Error> {
        if !start_pc.is_multiple_of(WORD_BYTES as u64) {
            return Err(Error::UnalignedAddress(start_pc));
        }
        let capacity = buffer.len() / WORD_BYTES * WORD_BYTES;
        start_pc
            .checked_add(capacity as u64)
            .ok_or(Error::AddressOverflow)?;
        Ok(Self {
            start_pc,
            buffer,
            len: 0,
        })
    }

    /// Target address of the next instruction.
    pub fn pc(&self) -> u64 {
        self.start_pc + self.len as u64
    }

    pub fn bytes(&self) -> &[u8] {
        &self.buffer[..self.len]
    }

    /// Append an instruction and return its byte offset.
    pub fn push(&mut self, code: Code, operands: &[Operand]) -> Result<usize, Error> {
        self.push_with_suffix(code, 0, operands)
    }

    /// Append an instruction with a suffix from its encoding's suffix table.
    pub fn push_with_suffix(
        &mut self,
        code: Code,
        suffix: u8,
        operands: &[Operand],
    ) -> Result<usize, Error> {
        self.append(|pc| word(code, suffix, operands, pc))
    }

    /// Encode a previously validated instruction at the current PC.
    /// A PC-relative operand retains its absolute target.
    pub fn push_instruction(&mut self, instruction: &Instruction) -> Result<usize, Error> {
        self.append(|pc| Ok(decoderant_a64::encode(instruction, pc)?.to_le_bytes()))
    }

    fn append(
        &mut self,
        encode: impl FnOnce(u64) -> Result<[u8; WORD_BYTES], EncodeError>,
    ) -> Result<usize, Error> {
        let offset = self.len;
        let pc = self.pc();
        let out = self
            .buffer
            .get_mut(offset..)
            .and_then(|tail| tail.first_chunk_mut::<WORD_BYTES>())
            .ok_or(Error::BufferFull)?;
        *out = encode(pc)?;
        self.len += WORD_BYTES;
        Ok(offset)
    }

    /// Replace an emitted instruction at the same PC. On error the bytes
    /// and cursor are unchanged.
    pub fn patch(&mut self, offset: usize, code: Code, operands: &[Operand]) -> Result<(), Error> {
        self.patch_with_suffix(offset, code, 0, operands)
    }

    /// Replace an emitted instruction with an encoding that has a suffix.
    pub fn patch_with_suffix(
        &mut self,
        offset: usize,
        code: Code,
        suffix: u8,
        operands: &[Operand],
    ) -> Result<(), Error> {
        if offset >= self.len || !offset.is_multiple_of(WORD_BYTES) {
            return Err(Error::InvalidPatch(offset));
        }
        let pc = self.start_pc + offset as u64;
        self.buffer[offset..offset + WORD_BYTES]
            .copy_from_slice(&word(code, suffix, operands, pc)?);
        Ok(())
    }
}

fn word(
    code: Code,
    suffix: u8,
    operands: &[Operand],
    pc: u64,
) -> Result<[u8; WORD_BYTES], EncodeError> {
    // The checked constructor already encodes the instruction. Its word is
    // ready to write without a mnemonic search or a second encoding pass.
    Ok(Instruction::with_suffix(code, suffix, operands, pc)?
        .decoded_word()
        .to_le_bytes())
}
