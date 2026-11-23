//! Per-block definite-initialization summaries for the streaming builder.
//! No opcodes, instruction lists or values to replay are retained.
use splice_core::hir::{Label, Reg};
use std::collections::{HashSet, VecDeque};

#[derive(Clone, Copy, Default)]
pub(super) enum Exit {
    #[default]
    Open,
    Return,
    Jump(Label),
    Branch(Label, Label),
}

#[derive(Default)]
struct Block {
    marked: bool,
    referenced: bool,
    defined: HashSet<Reg>,
    incoming_uses: HashSet<Reg>,
    exit: Exit,
}

pub(super) struct Flow {
    blocks: Vec<Block>,
    pub current: Option<usize>,
}

fn index(label: Label) -> usize {
    label.0 as usize + 1
}

impl Flow {
    pub fn new() -> Self {
        Self {
            blocks: vec![Block {
                marked: true,
                ..Default::default()
            }],
            current: Some(0),
        }
    }
    pub fn add_label(&mut self) {
        self.blocks.push(Block::default());
    }
    pub fn reference(&mut self, label: Label) -> splice_core::Result<()> {
        self.blocks
            .get_mut(index(label))
            .ok_or_else(|| splice_core::BuildError::verify("invalid LLVM label"))?
            .referenced = true;
        Ok(())
    }
    pub fn begin(&mut self, label: Label) -> splice_core::Result<()> {
        let block = self
            .blocks
            .get_mut(index(label))
            .ok_or_else(|| splice_core::BuildError::verify("invalid LLVM label"))?;
        if std::mem::replace(&mut block.marked, true) {
            return Err(splice_core::BuildError::verify("duplicate LLVM label"));
        }
        self.current = Some(index(label));
        Ok(())
    }
    pub fn operands(&mut self, reads: &[Reg], output: Option<Reg>) {
        if let Some(i) = self.current {
            let b = &mut self.blocks[i];
            for &r in reads {
                if !b.defined.contains(&r) {
                    b.incoming_uses.insert(r);
                }
            }
            b.defined.extend(output);
        }
    }
    pub fn close(&mut self, exit: Exit) {
        if let Some(i) = self.current.take() {
            self.blocks[i].exit = exit;
        }
    }
    pub fn verify(&self, arity: usize, regs: usize) -> splice_core::Result<()> {
        if self.blocks.iter().any(|b| b.referenced && !b.marked) {
            return Err(splice_core::BuildError::verify("undefined LLVM label"));
        }
        let mut entry = vec![0u64; regs.div_ceil(64)];
        for i in 0..arity {
            entry[i / 64] |= 1 << (i % 64);
        }
        let mut inputs = vec![None; self.blocks.len()];
        inputs[0] = Some(entry);
        let mut queue = VecDeque::from([0]);
        while let Some(i) = queue.pop_front() {
            let mut bits = inputs[i].as_ref().unwrap().clone();
            let b = &self.blocks[i];
            for &Reg(r) in &b.incoming_uses {
                if bits[r as usize / 64] & (1 << (r % 64)) == 0 {
                    return Err(splice_core::BuildError::verify(format!(
                        "uninitialized register {r} in LLVM kernel"
                    )));
                }
            }
            for &Reg(r) in &b.defined {
                bits[r as usize / 64] |= 1 << (r % 64);
            }
            let mut propagate = |l: Label| {
                let target = index(l);
                if let Some(state) = &mut inputs[target] {
                    let mut changed = false;
                    for (old, new) in state.iter_mut().zip(&bits) {
                        let common = *old & new;
                        changed |= common != *old;
                        *old = common;
                    }
                    if changed {
                        queue.push_back(target);
                    }
                } else {
                    inputs[target] = Some(bits.clone());
                    queue.push_back(target);
                }
            };
            match b.exit {
                Exit::Open => {
                    return Err(splice_core::BuildError::verify(
                        "reachable LLVM kernel path falls off the end",
                    ));
                }
                Exit::Return => {}
                Exit::Jump(target) => propagate(target),
                Exit::Branch(yes, no) => {
                    propagate(yes);
                    propagate(no);
                }
            }
        }
        Ok(())
    }
}
