//! Multi-instruction packed operations. Additional live intermediates are
//! explicit regalloc temporaries. Constants in installed kernels are RIP-relative.
use super::*;
mod float;
mod integer;
mod rearrange;

type Result = std::result::Result<(), IcedError>;
#[derive(Clone, Copy, Debug)]
pub(super) enum Recipe {
    Halves,
    Int(integer::Rule),
    Float(float::Rule),
    Rearrange(rearrange::Rule),
}
impl Recipe {
    pub fn select(m: Math, c: Features) -> Option<Self> {
        integer::Rule::select(m, c)
            .map(Self::Int)
            .or_else(|| float::Rule::select(m, c).map(Self::Float))
            .or_else(|| rearrange::Rule::select(m, c).map(Self::Rearrange))
            .or_else(|| {
                let lane = m.input.lane()?;
                if lane.float()
                    || lane.bytes() > 8
                    || !(c.bytes == 32 && !c.avx2
                        || c.bytes == 64 && !c.bw && lane.bytes() < 4 && c.avx2)
                {
                    return None;
                }
                match m.action {
                    // Logical operations already have full-width AVX forms.
                    Action::Binary(B::And | B::Or | B::Xor) | Action::Unary(Unary::Not) => None,
                    Action::Unary(_) | Action::Binary(_) => half(m, c).map(|_| Self::Halves),
                    _ => None,
                }
            })
    }
    pub fn temporaries(self, m: Math, c: Features) -> usize {
        match self {
            Self::Halves => {
                half(m, c).unwrap().temporaries()
                    + match m.action {
                        Action::Unary(_) => 2,
                        _ => 3,
                    }
            }
            Self::Int(r) => r.temps(m, c),
            Self::Float(r) => r.temps(m, c),
            Self::Rearrange(r) => r.temps(m, c),
        }
    }
    pub fn emit(
        self,
        a: &mut CodeAssembler,
        m: Math,
        c: Features,
        regs: &[usize],
        temps: &[usize],
        pool: Option<&mut emit::Constants>,
    ) -> Result {
        let mut v = Vector {
            a,
            width: c.bytes,
            cpu: c,
            pool,
        };
        match self {
            Self::Halves => emit_halves(&mut v, m, regs, temps),
            Self::Int(r) => r.emit(&mut v, m, regs, temps),
            Self::Float(r) => r.emit(&mut v, m, regs, temps),
            Self::Rearrange(r) => r.emit(&mut v, m, regs, temps),
        }
    }
}

fn half(m: Math, c: Features) -> Option<Native> {
    Native::select(
        m,
        Features {
            bytes: c.bytes / 2,
            ..c
        },
    )
}

fn emit_halves(v: &mut Vector, m: Math, r: &[usize], temps: &[usize]) -> Result {
    let half = half(m, v.cpu).expect("selected packed half");
    let (d, inputs) = r.split_last().unwrap();
    let (held, inner) = temps.split_at(inputs.len() + 1);
    let high = held[inputs.len()];
    let emit_part = |v: &mut Vector, inputs: &[usize], dest| {
        let mut regs = [0; 3];
        regs[..inputs.len()].copy_from_slice(inputs);
        if half.tied() && regs[0] != dest {
            copy(v.a, dest, regs[0], v.width / 2)?;
            regs[0] = dest;
        }
        regs[inputs.len()] = dest;
        half.emit_allocated(v.a, &regs[..inputs.len() + 1], inner, v.pool.as_deref_mut())
    };
    for (&src, &tmp) in inputs.iter().zip(held) {
        if v.width == 32 {
            v.a.vextractf128(XMM[tmp], YMM[src], 1)?;
        } else {
            v.a.vextracti64x4(YMM[tmp], ZMM[src], 1)?;
        }
    }
    emit_part(v, &held[..inputs.len()], high)?;
    emit_part(v, inputs, *d)?;
    if v.width == 32 {
        v.a.vinsertf128(YMM[*d], YMM[*d], XMM[high], 1)
    } else {
        v.a.vinserti64x4(ZMM[*d], ZMM[*d], YMM[high], 1)
    }
}

struct Vector<'a> {
    a: &'a mut CodeAssembler,
    width: usize,
    cpu: Features,
    pool: Option<&'a mut emit::Constants>,
}

// SSE2 encodes a destructive first operand. Preserve any live second operand
// when an intermediate reuses it; the allocator reserves SCRATCH at every site.
macro_rules! binary {
    ($v:expr,$d:expr,$x:expr,$y:expr,$s:ident,$vex:ident,$evex:ident) => {{
        let (d, x, y) = ($d, $x, $y);
        match $v.width {
            16 => {
                let tmp = if d == y && x != y {
                    crate::lir::x64::inst::SCRATCH
                } else {
                    d
                };
                $v.mov(tmp, x)?;
                $v.a.$s(XMM[tmp], XMM[y])?;
                $v.mov(d, tmp)
            }
            32 => $v.a.$vex(YMM[d], YMM[x], YMM[y]),
            _ => $v.a.$evex(ZMM[d], ZMM[x], ZMM[y]),
        }
    }};
}
use binary;
macro_rules! immediate {
    ($v:expr,$d:expr,$x:expr,$i:expr,$s:ident,$op:ident) => {{
        let (d, x, i) = ($d, $x, $i);
        match $v.width {
            16 => {
                $v.mov(d, x)?;
                $v.a.$s(XMM[d], i)
            }
            32 => $v.a.$op(YMM[d], YMM[x], i),
            _ => $v.a.$op(ZMM[d], ZMM[x], i),
        }
    }};
}
use immediate;
impl Vector<'_> {
    fn mov(&mut self, d: usize, x: usize) -> Result {
        if d == x {
            return Ok(());
        }
        copy(self.a, d, x, self.width)
    }
    fn and(&mut self, d: usize, x: usize, y: usize) -> Result {
        binary!(self, d, x, y, andps, vandps, vpandd)
    }
    fn andnot(&mut self, d: usize, x: usize, y: usize) -> Result {
        binary!(self, d, x, y, andnps, vandnps, vpandnd)
    }
    fn or(&mut self, d: usize, x: usize, y: usize) -> Result {
        binary!(self, d, x, y, orps, vorps, vpord)
    }
    fn xor(&mut self, d: usize, x: usize, y: usize) -> Result {
        binary!(self, d, x, y, xorps, vxorps, vpxord)
    }
    fn zero(&mut self, d: usize) -> Result {
        self.xor(d, d, d)
    }
    fn ones(&mut self, d: usize) -> Result {
        match self.width {
            64 => self.a.vpternlogd(ZMM[d], ZMM[d], ZMM[d], 0xff),
            32 if self.cpu.avx2 => self.a.vpcmpeqd(YMM[d], YMM[d], YMM[d]),
            32 => {
                self.a.pcmpeqd(XMM[d], XMM[d])?;
                self.a.vinsertf128(YMM[d], YMM[d], XMM[d], 1)
            }
            _ => self.a.pcmpeqd(XMM[d], XMM[d]),
        }
    }
    fn invert(&mut self, d: usize, x: usize) -> Result {
        if self.width == 64 {
            return self.a.vpternlogd(ZMM[d], ZMM[x], ZMM[x], 0x33);
        }
        let t = crate::lir::x64::inst::SCRATCH;
        self.ones(t)?;
        self.xor(d, x, t)
    }
    fn splat(&mut self, d: usize, size: usize, bits: u64) -> Result {
        let mut bytes = [0; 64];
        for slot in bytes[..self.width].chunks_exact_mut(size) {
            slot.copy_from_slice(&bits.to_le_bytes()[..size]);
        }
        let word = u64::from_le_bytes(bytes[..8].try_into().unwrap());
        if word == 0 {
            return self.zero(d);
        }
        if word == u64::MAX {
            return self.ones(d);
        }
        if let Some(pool) = self.pool.as_deref_mut() {
            let label = if let Some((label, ..)) = pool
                .iter()
                .find(|(_, b, n)| *n == self.width && *b == bytes)
            {
                *label
            } else {
                let label = self.a.create_label();
                pool.push((label, bytes, self.width));
                label
            };
            return load(self.a, d, ptr(label), self.width);
        }
        self.a.mov(r10, word)?;
        self.a.movq(XMM[d], r10)?;
        match self.width {
            64 => self.a.vpbroadcastq(ZMM[d], XMM[d]),
            32 if self.cpu.avx2 => self.a.vpbroadcastq(YMM[d], XMM[d]),
            32 => {
                self.a.punpcklqdq(XMM[d], XMM[d])?;
                self.a.vinsertf128(YMM[d], YMM[d], XMM[d], 1)
            }
            _ => self.a.punpcklqdq(XMM[d], XMM[d]),
        }
    }
    fn sign_mask(&mut self, size: usize, src: usize) -> Result {
        mask_from_signs(self.a, self.cpu, size, src)
    }
    fn materialize_mask(&mut self, d: usize, size: usize) -> Result {
        match size {
            1 => self.a.vpmovm2b(ZMM[d], k1),
            2 => self.a.vpmovm2w(ZMM[d], k1),
            4 => self.a.vpternlogd(ZMM[d].k1().z(), ZMM[d], ZMM[d], 255),
            _ => self.a.vpternlogq(ZMM[d].k1().z(), ZMM[d], ZMM[d], 255),
        }
    }
}
