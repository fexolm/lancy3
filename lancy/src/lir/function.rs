//! A function in SSA form after target instruction selection.
use super::TargetInst;
use crate::CompileError;
use regalloc2::{Block, Inst, InstRange, Operand, PRegSet, RegClass, VReg};
use std::ops::Range;

struct BlockData {
    insts: InstRange,
    params: Range<usize>,
    edges: Range<usize>,
    loop_depth: usize,
}

pub(super) struct Function<I: TargetInst> {
    machine: I::Machine,
    insts: Vec<I>,
    blocks: Vec<BlockData>,
    predecessors: Vec<Vec<Block>>,
    successors: Vec<Block>,
    edge_args: Vec<Range<usize>>,
    backedges: Vec<bool>,
    // Block parameters and edge arguments share one dense pool.
    block_values: Vec<VReg>,
    operands: Vec<Operand>,
    operand_ends: Vec<usize>,
    vregs: usize,
}

impl<I: TargetInst> Function<I> {
    pub fn insts(&self) -> &[I] {
        &self.insts
    }

    pub fn machine(&self) -> &I::Machine {
        &self.machine
    }

    pub fn new(machine: I::Machine, blocks: usize, vregs: usize) -> Result<Self, CompileError> {
        if vregs > super::MAX_VALUES || blocks > i32::MAX as usize {
            return Err(CompileError::Unsupported(crate::Limit::Registers));
        }
        Ok(Self {
            machine,
            insts: Vec::new(),
            blocks: Vec::with_capacity(blocks),
            predecessors: vec![Vec::new(); blocks],
            successors: Vec::new(),
            edge_args: Vec::new(),
            backedges: Vec::new(),
            block_values: Vec::new(),
            operands: Vec::new(),
            operand_ends: vec![0],
            vregs,
        })
    }

    pub fn temp(&mut self, class: RegClass) -> Result<VReg, CompileError> {
        if self.vregs == super::MAX_VALUES {
            return Err(CompileError::Unsupported(crate::Limit::Registers));
        }
        let reg = VReg::new(self.vregs, class);
        self.vregs += 1;
        Ok(reg)
    }

    /// Append an instruction to the open block: the one the next
    /// [`Self::end_block`] closes. Blocks are numbered in the order they close.
    pub fn push(&mut self, inst: I) {
        inst.operands(&self.machine, &mut self.operands);
        self.operand_ends.push(self.operands.len());
        self.insts.push(inst);
    }

    pub(super) fn reorder(
        &mut self,
        start: usize,
        order: &[usize],
        scratch: &mut Vec<I>,
        operands: &mut Vec<Operand>,
    ) {
        let end = start + order.len();
        scratch.clear();
        scratch.extend(order.iter().map(|&i| self.insts[start + i]));
        self.insts[start..end].copy_from_slice(scratch);

        // A permutation leaves the total operand count and other blocks' offsets intact.
        let mut at = self.operand_ends[start];
        for i in start..end {
            operands.clear();
            self.insts[i].operands(&self.machine, operands);
            self.operands[at..at + operands.len()].copy_from_slice(operands);
            at += operands.len();
            self.operand_ends[i + 1] = at;
        }
    }

    pub fn loop_depth(&self, block: Block) -> usize {
        self.blocks[block.index()].loop_depth
    }

    /// True for a backedge or the branch arm leading to its copy block.
    pub fn leads_to_backedge(&self, block: Block, edge: usize) -> bool {
        self.backedges[self.blocks[block.index()].edges.start + edge]
    }

    /// Add a successor edge of the open block, in the order of its branch's
    /// targets.
    pub fn edge(&mut self, target: Block, args: impl IntoIterator<Item = VReg>, backedge: bool) {
        self.predecessors[target.index()].push(Block::new(self.blocks.len()));
        self.successors.push(target);
        let start = self.block_values.len();
        self.block_values.extend(args);
        self.edge_args.push(start..self.block_values.len());
        self.backedges.push(backedge);
    }

    pub fn end_block(
        &mut self,
        params: impl IntoIterator<Item = VReg>,
        loop_depth: usize,
    ) -> Result<(), CompileError> {
        // regalloc2 packs a before/after bit into its 32-bit program points.
        if self.insts.len() > i32::MAX as usize || self.operands.len() > u32::MAX as usize {
            return Err(CompileError::Unsupported(crate::Limit::Registers));
        }
        let first = self
            .blocks
            .last()
            .map_or(Inst::new(0), |b| b.insts.last().next());
        let edge_start = self.blocks.last().map_or(0, |b| b.edges.end);
        if cfg!(debug_assertions) {
            // The open block ends with its only control transfer, and only a
            // branch has edges.
            let insts = &self.insts[first.index()..];
            let last = insts.last().expect("a block ends with a branch or return");
            assert!(last.is_branch() || last.is_ret(), "a block ends with a branch or return");
            assert!(
                insts[..insts.len() - 1].iter().all(|i| !i.is_branch() && !i.is_ret()),
                "a control transfer ends its block"
            );
            assert!(
                last.is_branch() || self.successors.len() == edge_start,
                "a return has no edges"
            );
        }
        let start = self.block_values.len();
        self.block_values.extend(params);
        self.blocks.push(BlockData {
            insts: InstRange::new(first, Inst::new(self.insts.len())),
            params: start..self.block_values.len(),
            edges: edge_start..self.successors.len(),
            loop_depth,
        });
        Ok(())
    }
}

impl<I: TargetInst> regalloc2::Function for Function<I> {
    fn num_insts(&self) -> usize {
        self.insts.len()
    }
    fn num_blocks(&self) -> usize {
        self.blocks.len()
    }
    fn entry_block(&self) -> Block {
        Block::new(0)
    }
    fn block_insns(&self, b: Block) -> InstRange {
        self.blocks[b.index()].insts
    }
    fn block_succs(&self, b: Block) -> &[Block] {
        &self.successors[self.blocks[b.index()].edges.clone()]
    }
    fn block_preds(&self, b: Block) -> &[Block] {
        &self.predecessors[b.index()]
    }
    fn block_params(&self, b: Block) -> &[VReg] {
        &self.block_values[self.blocks[b.index()].params.clone()]
    }
    fn is_ret(&self, i: Inst) -> bool {
        self.insts[i.index()].is_ret()
    }
    fn is_branch(&self, i: Inst) -> bool {
        self.insts[i.index()].is_branch()
    }
    fn branch_blockparams(&self, b: Block, _: Inst, succ: usize) -> &[VReg] {
        let edge = self.blocks[b.index()].edges.start + succ;
        &self.block_values[self.edge_args[edge].clone()]
    }
    fn inst_operands(&self, i: Inst) -> &[Operand] {
        &self.operands[self.operand_ends[i.index()]..self.operand_ends[i.index() + 1]]
    }
    fn inst_clobbers(&self, i: Inst) -> PRegSet {
        self.insts[i.index()].clobbers(&self.machine)
    }
    fn num_vregs(&self) -> usize {
        self.vregs
    }
    fn spillslot_size(&self, class: RegClass) -> usize {
        I::spillslot_size(&self.machine, class)
    }
}
