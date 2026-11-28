//! Cache-blocked row-major i32 matrix multiplication.
//! Each lane accumulates one output column across the current depth panel.
use splice::{kernel, std::intrinsics::simd::Lane};

#[kernel]
#[allow(clippy::too_many_arguments)]
fn row_vector(
    #[comptime] full: bool,
    row: i32,
    col: i32,
    inner: i32,
    cols: i32,
    depth_start: i32,
    depth_end: i32,
    count: i32,
    a: &[i32],
    b: &[i32],
    out: &mut [i32],
) -> i32 {
    let offset = (row * cols + col) as usize;
    let tail = Lane::<i32>::first_n(count as usize);
    let zero = Lane::<i32>::splat(0);
    let mut sum = if depth_start == 0 {
        zero
    } else if full {
        Lane::load(out, offset)
    } else {
        Lane::load_select(out, offset, tail, zero)
    };
    for k in depth_start..depth_end {
        let left = Lane::splat(a[(row * inner + k) as usize]);
        let right_offset = (k * cols + col) as usize;
        let right = if full {
            Lane::load(b, right_offset)
        } else {
            Lane::load_select(b, right_offset, tail, zero)
        };
        sum += left * right;
    }
    if full {
        sum.store(out, offset);
    } else {
        sum.store_select(out, offset, tail);
    }
    count
}

/// Write A[rows, inner] × B[inner, cols] to C[rows, cols], with wrapping i32 arithmetic.
/// Returns the number of output elements, or -1 before any write if the shape is invalid.
#[kernel]
pub fn matmul(rows: i32, inner: i32, cols: i32, a: &[i32], b: &[i32], out: &mut [i32]) -> i32 {
    if rows < 0 || inner < 0 || cols < 0 {
        return -1;
    }
    // Division checks both buffer capacity and representability of linear indices.
    if rows > 0 && (inner > a.len() as i32 / rows || cols > out.len() as i32 / rows) {
        return -1;
    }
    if inner > 0 && cols > b.len() as i32 / inner {
        return -1;
    }
    let elements = rows * cols;
    if elements == 0 {
        return 0;
    }
    if inner == 0 {
        for i in 0..elements {
            out[i as usize] = 0;
        }
        return elements;
    }

    let width = Lane::<i32>::element_width() as i32;
    let mut row_start = 0;
    while row_start < rows {
        let row_end = row_start + (rows - row_start).min(16);
        let mut col_start = 0;
        while col_start < cols {
            let col_end = col_start + (cols - col_start).min(4 * width);
            let mut depth_start = 0;
            while depth_start < inner {
                let depth_end = depth_start + (inner - depth_start).min(32);
                // All rows in this tile reuse the same depth × column panel of B.
                for row in row_start..row_end {
                    let mut col = col_start;
                    while col_end - col >= width {
                        row_vector(
                            true,
                            row,
                            col,
                            inner,
                            cols,
                            depth_start,
                            depth_end,
                            width,
                            a,
                            b,
                            out,
                        );
                        col += width;
                    }
                    if col < col_end {
                        row_vector(
                            false,
                            row,
                            col,
                            inner,
                            cols,
                            depth_start,
                            depth_end,
                            col_end - col,
                            a,
                            b,
                            out,
                        );
                    }
                }
                depth_start = depth_end;
            }
            col_start = col_end;
        }
        row_start = row_end;
    }
    elements
}

/// Independent scalar dot products for the examples, benchmarks and tests.
pub fn reference(rows: usize, inner: usize, cols: usize, a: &[i32], b: &[i32]) -> Vec<i32> {
    let mut product = vec![0i32; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            let mut sum = 0i32;
            for k in 0..inner {
                sum = sum.wrapping_add(a[row * inner + k].wrapping_mul(b[k * cols + col]));
            }
            product[row * cols + col] = sum;
        }
    }
    product
}
