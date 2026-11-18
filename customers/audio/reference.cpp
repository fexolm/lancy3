#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <emmintrin.h>

namespace {

// External buffers contain signed Q15 amplitudes in int32_t slots. All three
// buffers are disjoint. Frames are contiguous groups of independent voices.
template<std::size_t Lanes>
inline __m128i load_q15(const std::int32_t* src) {
    const auto low = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
    if constexpr (Lanes == 8) {
        const auto high = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 4));
        return _mm_packs_epi32(low, high);
    } else {
        return _mm_packs_epi32(low, low);
    }
}

template<std::size_t Lanes>
inline void store_q15(std::int32_t* dst, __m128i value) {
    const auto sign = _mm_srai_epi16(value, 15);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), _mm_unpacklo_epi16(value, sign));
    if constexpr (Lanes == 8) {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 4), _mm_unpackhi_epi16(value, sign));
    }
}

template<std::size_t Lanes>
inline void group(const std::int32_t* input, std::int32_t* output, std::int32_t* state,
                  std::size_t frames, std::size_t stride, __m128i previous) {
    const auto rise = _mm_set1_epi16(1536);
    const auto fall = _mm_set1_epi16(768);
    for (std::size_t frame = 0; frame < frames; ++frame) {
        const auto offset = frame * stride;
        const auto x = load_q15<Lanes>(input + offset);
        const auto target = _mm_adds_epi16(x, x);
        const auto ceiling = _mm_adds_epi16(previous, rise);
        const auto floor = _mm_subs_epi16(previous, fall);
        previous = _mm_min_epi16(ceiling, _mm_max_epi16(floor, target));
        store_q15<Lanes>(output + offset, previous);
    }
    store_q15<Lanes>(state, previous);
}

template<std::size_t Lanes>
void vector_block(const std::int32_t* input, std::int32_t* output, std::int32_t* state,
                  std::size_t frames, std::size_t voices) {
    // End the last vector exactly at the end of each frame. It may overlap
    // the previous group, so snapshot its feedback before updating any voice.
    const std::size_t last = voices - Lanes;
    const auto initial = load_q15<Lanes>(state + last);
    for (std::size_t v = 0; v < last; v += Lanes) {
        group<Lanes>(input + v, output + v, state + v, frames, voices, load_q15<Lanes>(state + v));
    }
    group<Lanes>(input + last, output + last, state + last, frames, voices, initial);
}

}  // namespace

extern "C" void audio_q15_cpp(const std::int32_t* input, std::int32_t* output,
                              std::int32_t* state, std::size_t frames, std::size_t voices) {
    if (frames == 0 || voices == 0) return;
    if (voices >= 8) {
        return vector_block<8>(input, output, state, frames, voices);
    }
    if (voices >= 4) {
        return vector_block<4>(input, output, state, frames, voices);
    }
    for (std::size_t voice = 0; voice < voices; ++voice) {
        auto previous = state[voice];
        for (std::size_t t = 0; t < frames; ++t) {
            const auto offset = t * voices + voice;
            const auto target = std::clamp(2 * input[offset], -32768, 32767);
            previous = std::clamp(target, previous - 768, previous + 1536);
            output[offset] = previous;
        }
        state[voice] = previous;
    }
}
