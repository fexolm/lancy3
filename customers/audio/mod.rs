use lancy_jit::{NativeKernel, kernel};

#[derive(Clone, Copy)]
struct Patch {
    gain: i32,
    rise: i32,
    fall: i32,
}

const PATCH: Patch = Patch {
    gain: 2,
    rise: 1536,
    fall: 768,
};

unsafe extern "C" {
    fn audio_q15_cpp(
        input: *const i32,
        output: *mut i32,
        state: *mut i32,
        frames: usize,
        voices: usize,
    );
}

fn process_cpp(input: &[i32], output: &mut [i32], state: &mut [i32]) {
    assert_eq!(output.len(), input.len());
    if input.is_empty() {
        return;
    }
    let voices = state.len();
    assert!(voices != 0 && input.len().is_multiple_of(voices));
    // SAFETY: The input, output and writable state slices are disjoint. Shape
    // checks bound frame * voice indexing. All fixtures contain Q15 values.
    unsafe {
        audio_q15_cpp(
            input.as_ptr(),
            output.as_mut_ptr(),
            state.as_mut_ptr(),
            input.len() / voices,
            voices,
        );
    }
}

// Signed Q15 in i32 lanes. One frame can evaluate every voice independently.
#[kernel]
fn voice_step(#[comptime] patch: Patch, sample: i32, previous: i32) -> i32 {
    let driven = patch.gain * sample;
    let clipped_lo = if driven < -32768 { -32768 } else { driven };
    let target = if clipped_lo > 32767 {
        32767
    } else {
        clipped_lo
    };
    let low = previous - patch.fall;
    let high = previous + patch.rise;
    let limited = if target < low { low } else { target };
    if limited > high { high } else { limited }
}

fn compile_patch(patch: Patch) -> Result<NativeKernel, String> {
    let program = voice_step::build(patch)?
        .vectorize()?
        .ok_or("audio voice step requires SIMD")?;
    NativeKernel::specialize(&program, &[])
}

fn process(
    code: &NativeKernel,
    input: &[i32],
    output: &mut [i32],
    state: &mut [i32],
) -> Result<(), String> {
    let voices = state.len();
    if output.len() != input.len() || (voices != 0 && !input.len().is_multiple_of(voices)) {
        return Err("audio buffer shape does not match the voice count".into());
    }
    if input.is_empty() {
        return Ok(());
    }
    if voices == 0 {
        return Err("a nonempty audio buffer needs at least one voice".into());
    }
    let mut previous: &[i32] = state;
    for (frame, destination) in input
        .chunks_exact(voices)
        .zip(output.chunks_exact_mut(voices))
    {
        if code.run(&[frame, previous], destination)? != voices {
            return Err("unguarded audio kernel stopped mid-frame".into());
        }
        // The output of this frame is directly borrowed as the next frame's
        // feedback. Only one copy of the last frame is needed per callback.
        previous = destination;
    }
    state.copy_from_slice(&output[output.len() - voices..]);
    Ok(())
}

fn oracle(patch: Patch, input: &[i32], output: &mut [i32], state: &mut [i32]) {
    let voices = state.len();
    for (i, (&sample, out)) in input.iter().zip(output).enumerate() {
        let previous = &mut state[i % voices];
        let target = (patch.gain * sample).clamp(-32768, 32767);
        *previous = target.clamp(*previous - patch.fall, *previous + patch.rise);
        *out = *previous;
    }
}

// Combined ramps, rail plateaus and a repeatable bipolar excitation. Rail
// plateaus are long enough to reach both saturation limits from a cold start.
fn signal(frames: usize, voices: usize) -> Vec<i32> {
    let mut seed: u32 = 0xa7f4_39c1;
    (0..frames * voices)
        .map(|i| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let t = i / voices;
            let v = i % voices;
            match (t / 24 + v) % 5 {
                0 => 32767,
                1 => -32768,
                2 => ((t * 97 + v * 401) % 8192) as i32 - 4096,
                _ => (seed >> 16) as i32 - 32768,
            }
        })
        .collect()
}

fn initial_state(voices: usize) -> Vec<i32> {
    (0..voices)
        .map(|v| [-32768, 32000, 32767, 0, -1024][v % 5])
        .collect()
}

fn verify(code: &NativeKernel) -> Result<(), String> {
    let quiet = Patch {
        gain: 1,
        rise: 256,
        fall: 512,
    };
    let changed = compile_patch(quiet)?;
    for voices in (1..=17).chain([31, 64]) {
        let frames = 257;
        let input = signal(frames, voices);
        let initial = initial_state(voices);
        let mut expected_state = initial.clone();
        let mut expected = vec![0; input.len()];
        oracle(PATCH, &input, &mut expected, &mut expected_state);
        let mut cpp_state = initial.clone();
        let mut ir_state = initial.clone();
        let mut cpp = vec![0; input.len()];
        let mut ir = vec![0; input.len()];
        let mut start = 0;
        // Includes a zero-frame callback and non-power-of-two callback sizes.
        for end in [0, 1, 66, 129, frames] {
            let range = start * voices..end * voices;
            process(
                code,
                &input[range.clone()],
                &mut ir[range.clone()],
                &mut ir_state,
            )?;
            process_cpp(&input[range.clone()], &mut cpp[range], &mut cpp_state);
            start = end;
        }
        if ir != expected
            || cpp != expected
            || cpp_state != expected_state
            || ir_state != expected_state
        {
            return Err(format!(
                "audio output or carried state mismatch at {voices} voices"
            ));
        }

        // Swap the compiled parameter set between callbacks while preserving
        // the feedback state of all voices. This is deliberately off the timer.
        oracle(quiet, &input, &mut expected, &mut expected_state);
        process(&changed, &input, &mut ir, &mut ir_state)?;
        if ir != expected || ir_state != expected_state {
            return Err("audio state discontinuity after patch change".into());
        }
    }
    Ok(())
}

pub fn run() -> Result<(), String> {
    let code = crate::build("audio/Q15 patch", || compile_patch(PATCH))?;
    verify(&code)?;
    for (voices, frames) in [(1, 64), (31, 128), (64, 64)] {
        let input = signal(frames, voices);
        let initial = initial_state(voices);
        let mut cpp_state = initial.clone();
        let mut ir_state = initial.clone();
        let mut expected_state = initial.clone();
        let mut expected = vec![0; input.len()];
        oracle(PATCH, &input, &mut expected, &mut expected_state);
        let label = format!("audio/{voices} voices x {frames} frames");
        crate::compare(
            &label,
            frames * voices,
            &expected,
            |dst| {
                cpp_state.copy_from_slice(&initial);
                process_cpp(&input, dst, &mut cpp_state);
            },
            |dst| {
                ir_state.copy_from_slice(&initial);
                process(&code, &input, dst, &mut ir_state).expect("valid fixed-point callback");
            },
        )?;
    }
    Ok(())
}

#[test]
fn callbacks_preserve_q15_samples_and_state() {
    verify(&compile_patch(PATCH).unwrap()).unwrap();
}
