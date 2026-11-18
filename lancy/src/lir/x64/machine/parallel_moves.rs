use crate::lir::instruction::Op;
use crate::lir::x64::frame::{ECX, Home};

pub(super) fn schedule(moves: &mut Vec<(Home, Home)>, ops: &mut Vec<Op<Home>>) {
    while !moves.is_empty() {
        if let Some(i) = moves
            .iter()
            .position(|(dst, _)| !moves.iter().any(|(_, src)| src == dst))
        {
            let (dst, src) = moves.swap_remove(i);
            ops.push(Op::Move(dst, src));
        } else {
            // All remaining moves participate in, or feed, a cycle. Preserve
            // an old destination before it is overwritten. Slot-to-slot MOV
            // uses EAX in the encoder, so ECX remains intact until consumed.
            let old = moves[0].0;
            let temp = Home::Register(ECX);
            ops.push(Op::Move(temp, old));
            for (_, src) in moves.iter_mut() {
                if *src == old {
                    *src = temp;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parallel_swap_supports_registers_slots_and_fanout() {
        let (a, b, c, d) = (
            Home::Register(8),
            Home::Slot(0),
            Home::Slot(1),
            Home::Register(9),
        );
        let mut ops = Vec::new();
        schedule(&mut vec![(a, b), (b, c), (c, a), (d, a)], &mut ops);
        let mut values = [(a, 1), (b, 2), (c, 3), (d, 4), (Home::Register(ECX), 0)];
        for op in ops {
            let Op::Move(dst, src) = op else {
                unreachable!()
            };
            let x = values.iter().find(|v| v.0 == src).unwrap().1;
            values.iter_mut().find(|v| v.0 == dst).unwrap().1 = x;
        }
        assert_eq!(values[..4], [(a, 2), (b, 3), (c, 1), (d, 1)]);
    }
}
