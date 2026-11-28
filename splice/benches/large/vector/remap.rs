//! Nearest-neighbor geometric remapping of three planar 16-bit color channels.
use splice::{kernel, std::intrinsics::simd::Lane};

const CHANNELS: [i32; 3] = [0, 1, 2];

/// Each of three consecutive map planes contains absolute offsets into the
/// source tile. The destination has the same planar layout as the map.
/// A gather fault leaves the packets already written in the destination.
#[kernel]
pub(crate) fn remap_rgb16(source: &[u16], map: &[u16], out: &mut [u16]) -> i32 {
    let count = map.len() as i32;
    let channels = CHANNELS.len() as i32;
    if count % channels != 0 || out.len() < map.len() {
        return -1;
    }
    let plane = count / channels;
    let width = Lane::<u16>::element_width() as i32;
    let mut x = 0;
    while plane - x >= width {
        for &channel in CHANNELS.iter() {
            let at = (channel * plane + x) as usize;
            let index = Lane::load(map, at);
            Lane::gather(source, index).store(out, at);
        }
        x += width;
    }
    if x < plane {
        let valid = Lane::<u16>::first_n((plane - x) as usize);
        let zero = Lane::<u16>::splat(0);
        for &channel in CHANNELS.iter() {
            let at = (channel * plane + x) as usize;
            let index = Lane::load_select(map, at, valid, zero);
            Lane::gather_select(source, valid, index, zero).store_select(out, at, valid);
        }
    }
    count
}
