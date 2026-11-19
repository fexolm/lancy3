#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Bank {
    X,
    W,
    V,
    Z,
    P,
}

/// An architectural register, including the distinct SP and zero roles of register 31.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Register {
    pub(crate) bank: Bank,
    pub(crate) number: u8,
    pub(crate) sp: bool,
}

macro_rules! registers {
    ($bank:ident; $($name:ident = $number:literal),+ $(,)?) => {
        $(pub const $name: Self = Self { bank: Bank::$bank, number: $number, sp: false };)+
    };
}

impl Register {
    registers!(X;
        X0 = 0, X1 = 1, X2 = 2, X3 = 3, X4 = 4, X5 = 5, X6 = 6, X7 = 7,
        X8 = 8, X9 = 9, X10 = 10, X11 = 11, X12 = 12, X13 = 13, X14 = 14, X15 = 15,
        X16 = 16, X17 = 17, X18 = 18, X19 = 19, X20 = 20, X21 = 21, X22 = 22, X23 = 23,
        X24 = 24, X25 = 25, X26 = 26, X27 = 27, X28 = 28, X29 = 29, X30 = 30, XZR = 31,
    );
    registers!(W;
        W0 = 0, W1 = 1, W2 = 2, W3 = 3, W4 = 4, W5 = 5, W6 = 6, W7 = 7,
        W8 = 8, W9 = 9, W10 = 10, W11 = 11, W12 = 12, W13 = 13, W14 = 14, W15 = 15,
        W16 = 16, W17 = 17, W18 = 18, W19 = 19, W20 = 20, W21 = 21, W22 = 22, W23 = 23,
        W24 = 24, W25 = 25, W26 = 26, W27 = 27, W28 = 28, W29 = 29, W30 = 30, WZR = 31,
    );
    registers!(V;
        V0 = 0, V1 = 1, V2 = 2, V3 = 3, V4 = 4, V5 = 5, V6 = 6, V7 = 7,
        V8 = 8, V9 = 9, V10 = 10, V11 = 11, V12 = 12, V13 = 13, V14 = 14, V15 = 15,
        V16 = 16, V17 = 17, V18 = 18, V19 = 19, V20 = 20, V21 = 21, V22 = 22, V23 = 23,
        V24 = 24, V25 = 25, V26 = 26, V27 = 27, V28 = 28, V29 = 29, V30 = 30, V31 = 31,
    );
    registers!(Z;
        Z0 = 0, Z1 = 1, Z2 = 2, Z3 = 3, Z4 = 4, Z5 = 5, Z6 = 6, Z7 = 7,
        Z8 = 8, Z9 = 9, Z10 = 10, Z11 = 11, Z12 = 12, Z13 = 13, Z14 = 14, Z15 = 15,
        Z16 = 16, Z17 = 17, Z18 = 18, Z19 = 19, Z20 = 20, Z21 = 21, Z22 = 22, Z23 = 23,
        Z24 = 24, Z25 = 25, Z26 = 26, Z27 = 27, Z28 = 28, Z29 = 29, Z30 = 30, Z31 = 31,
    );
    registers!(P;
        P0 = 0, P1 = 1, P2 = 2, P3 = 3, P4 = 4, P5 = 5, P6 = 6, P7 = 7,
        P8 = 8, P9 = 9, P10 = 10, P11 = 11, P12 = 12, P13 = 13, P14 = 14, P15 = 15,
    );
    pub const SP: Self = Self {
        sp: true,
        ..Self::XZR
    };
    pub const WSP: Self = Self {
        sp: true,
        ..Self::WZR
    };

    fn numbered(bank: Bank, number: u8, end: u8) -> Option<Self> {
        (number < end).then_some(Self {
            bank,
            number,
            sp: false,
        })
    }
    pub fn x(n: u8) -> Option<Self> {
        Self::numbered(Bank::X, n, 32)
    }
    pub fn w(n: u8) -> Option<Self> {
        Self::numbered(Bank::W, n, 32)
    }
    pub fn v(n: u8) -> Option<Self> {
        Self::numbered(Bank::V, n, 32)
    }
    pub fn z(n: u8) -> Option<Self> {
        Self::numbered(Bank::Z, n, 32)
    }
    pub fn p(n: u8) -> Option<Self> {
        Self::numbered(Bank::P, n, 16)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arrangement {
    B8,
    B16,
    H4,
    H8,
    S2,
    S4,
    D1,
    D2,
    B,
    H,
    S,
    D,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ShiftKind {
    Lsl,
    Lsr,
    Asr,
    Ror,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ConditionCode {
    Eq,
    Ne,
    Cs,
    Cc,
    Mi,
    Pl,
    Vs,
    Vc,
    Hi,
    Ls,
    Ge,
    Lt,
    Gt,
    Le,
    Al,
    Nv,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PredQual {
    None,
    Zero,
    Merge,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operand {
    Register(Register),
    Vector(Register, Arrangement),
    Immediate(u64),
    ShiftedImmediate(u64, u8),
    ShiftedRegister(Register, ShiftKind, u8),
    Memory(Register, i64),
    MemoryVl(Register, i64),
    Predicate(Register, PredQual),
    Label(u64),
    Condition(ConditionCode),
    Named(u32),
}

impl Operand {
    pub const fn reg(r: Register) -> Self {
        Self::Register(r)
    }
    pub const fn vec(r: Register, a: Arrangement) -> Self {
        Self::Vector(r, a)
    }
    pub const fn imm(v: u64) -> Self {
        Self::Immediate(v)
    }
    pub const fn imm_lsl(v: u64, shift: u8) -> Self {
        Self::ShiftedImmediate(v, shift)
    }
    pub const fn shifted(r: Register, shift: ShiftKind, amount: u8) -> Self {
        Self::ShiftedRegister(r, shift, amount)
    }
    pub const fn mem(r: Register) -> Self {
        Self::Memory(r, 0)
    }
    pub const fn mem_off(r: Register, offset: i64) -> Self {
        Self::Memory(r, offset)
    }
    pub const fn mem_mul_vl(r: Register, offset: i64) -> Self {
        Self::MemoryVl(r, offset)
    }
    pub const fn pred(r: Register, qualifier: PredQual) -> Self {
        Self::Predicate(r, qualifier)
    }
    pub const fn label(v: u64) -> Self {
        Self::Label(v)
    }
    pub const fn cond(v: ConditionCode) -> Self {
        Self::Condition(v)
    }
    pub const fn named(v: u32) -> Self {
        Self::Named(v)
    }
}
