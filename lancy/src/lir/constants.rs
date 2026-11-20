use crate::{hir::Lane, support::float16::Format};

/// Encode the lanes of a vector index constant in address order.
pub(crate) fn iota_bytes(lane: Lane, width: usize) -> [u8; 64] {
    let mut bytes = [0; 64];
    for (i, element) in bytes[..width].chunks_exact_mut(lane.bytes()).enumerate() {
        let value = match lane {
            Lane::F16 | Lane::BF16 => u128::from(Format::for_lane(lane).encode(i as f64, 0.)),
            Lane::F32 => u128::from((i as f32).to_bits()),
            Lane::F64 => u128::from((i as f64).to_bits()),
            _ => i as u128,
        };
        element.copy_from_slice(&value.to_le_bytes()[..lane.bytes()]);
    }
    bytes
}
