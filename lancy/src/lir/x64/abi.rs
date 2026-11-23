//! The two x64 calling conventions as data. Generated code follows the
//! target's convention at its entry, at Rust calls and at math library calls.
use regalloc2::{MachineEnv, PReg, PRegSet, RegClass};
use splice_target::{CallingConvention, X86Features};

const fn gpr(enc: usize) -> PReg {
    PReg::new(enc, RegClass::Int)
}

pub(super) const fn gprs(encodings: &[usize]) -> PRegSet {
    let mut set = PRegSet::empty();
    let mut i = 0;
    while i < encodings.len() {
        set.add(gpr(encodings[i]));
        i += 1;
    }
    set
}

pub(super) const fn vectors(mut start: usize, end: usize) -> PRegSet {
    let mut regs = PRegSet::empty();
    while start < end {
        if start != SCRATCH {
            regs.add(PReg::new(start, RegClass::Vector));
        }
        start += 1;
    }
    regs
}

/// The vector register reserved for moves and inside pseudo-ops. Win64
/// preserves XMM6..15, so a volatile register serves both conventions.
pub(super) const SCRATCH: usize = 5;

// K0 cannot predicate EVEX, and K1 is reserved for destructive masks and moves.
pub(super) const MASKS: PRegSet = {
    let mut regs = PRegSet::empty();
    let mut i = 2;
    while i <= 7 {
        regs.add(PReg::new(i, RegClass::Float));
        i += 1;
    }
    regs
};

pub(crate) struct Abi {
    /// Integer argument registers, in order.
    pub args: [PReg; 4],
    /// General registers a callee may overwrite.
    pub volatile: PRegSet,
    /// Allocatable vector registers a callee may overwrite.
    pub volatile_vectors: PRegSet,
    /// Registers a callee preserves. For vectors only the low 128 bits.
    pub saved: PRegSet,
    /// Bytes a caller reserves above the return address for the callee.
    pub shadow: i32,
    env: MachineEnv,
}

/// `saved_vectors` is the first callee-saved vector register; 16 when none is.
const fn abi(
    args: [PReg; 4],
    volatile: &[usize],
    saved_gprs: &[usize],
    saved_vectors: usize,
) -> Abi {
    let mut saved = vectors(saved_vectors, 16);
    let mut i = 0;
    while i < saved_gprs.len() {
        saved.add(gpr(saved_gprs[i]));
        i += 1;
    }
    let volatile = gprs(volatile);
    let volatile_vectors = vectors(0, saved_vectors);
    Abi {
        args,
        volatile,
        volatile_vectors,
        saved,
        shadow: if saved_vectors < 16 { 32 } else { 0 },
        env: MachineEnv {
            preferred_regs_by_class: [volatile, MASKS, volatile_vectors],
            non_preferred_regs_by_class: [
                gprs(saved_gprs),
                PRegSet::empty(),
                vectors(saved_vectors, 16),
            ],
            scratch_by_class: [
                Some(gpr(10)),
                Some(PReg::new(1, RegClass::Float)),
                Some(PReg::new(SCRATCH, RegClass::Vector)),
            ],
            fixed_stack_slots: Vec::new(),
        },
    }
}

pub(crate) static SYSV: Abi = abi(
    [gpr(7), gpr(6), gpr(2), gpr(1)],
    &[0, 1, 2, 6, 7, 8, 9],
    &[3, 5, 12, 13, 14, 15],
    16,
);

pub(crate) static WIN64: Abi = abi(
    [gpr(1), gpr(2), gpr(8), gpr(9)],
    &[0, 1, 2, 8, 9],
    &[3, 5, 6, 7, 12, 13, 14, 15],
    6,
);

impl Abi {
    pub fn env(&'static self) -> &'static MachineEnv {
        &self.env
    }

    pub fn is_win64(&self) -> bool {
        std::ptr::eq(self, &WIN64)
    }

    /// Whether `extern "C"` functions of this process use another convention.
    pub fn is_foreign(&self) -> bool {
        self.is_win64() != cfg!(windows)
    }
}

/// The x64 extensions and vector width one compilation selects for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Cpu {
    pub bytes: usize,
    pub isa: X86Features,
}

impl Cpu {
    pub fn new(bytes: usize, isa: X86Features) -> Self {
        // Every target with wide vectors has the instructions for them; see
        // `splice_target::Target::x86_64`.
        debug_assert!(bytes == 16 || isa.avx2 && (bytes == 32 || isa.avx512f && bytes == 64));
        Self { bytes, isa }
    }
}

/// What x64 operands and clobbers depend on.
#[derive(Clone, Copy)]
pub(crate) struct Machine {
    pub cpu: Cpu,
    pub abi: &'static Abi,
}

impl Machine {
    pub fn new(target: &splice_target::Target) -> Self {
        let isa = target
            .x86()
            .expect("x64 code generation for an x86-64 target");
        Self {
            cpu: Cpu::new(target.vector_bytes(), isa),
            abi: match target.calling_convention() {
                CallingConvention::Win64 => &WIN64,
                _ => &SYSV,
            },
        }
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
impl Cpu {
    /// The test target's extensions and width.
    pub fn host() -> Self {
        Machine::host().cpu
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
impl Machine {
    pub fn host() -> Self {
        Self::new(&crate::test_target::target())
    }
}
