#include <cstddef>
#include <cstdint>
#include <immintrin.h>

namespace {
constexpr std::int32_t paid = 2;
constexpr std::int32_t minimum_exclusive = 2;

template <bool Nullable, bool Clustered>
void project(const std::int32_t* status, const std::int32_t* quantity,
             const std::int32_t* amount, const std::uint8_t* bitmap,
             std::int32_t* out, std::size_t count) {
    std::size_t row = 0;
#if defined(__AVX2__)
    const auto lane_bits = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
    for (; count - row >= 8; row += 8) {
        const auto states = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(status + row));
        auto mask = _mm256_cmpeq_epi32(states, _mm256_set1_epi32(paid));
        if (!Clustered || !_mm256_testz_si256(mask, mask)) {
            const auto qty = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(quantity + row));
            mask = _mm256_and_si256(mask, _mm256_cmpgt_epi32(qty, _mm256_set1_epi32(minimum_exclusive)));
            if constexpr (Nullable) {
                const auto bits = _mm256_set1_epi32(bitmap[row / 8]);
                const auto valid = _mm256_cmpeq_epi32(_mm256_and_si256(bits, lane_bits), lane_bits);
                mask = _mm256_and_si256(mask, valid);
            }
            const auto values = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(amount + row));
            mask = _mm256_and_si256(mask, values);
        }
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + row), mask);
    }
#else
    const auto lane_bits = _mm_setr_epi32(1, 2, 4, 8);
    for (; count - row >= 4; row += 4) {
        const auto states = _mm_loadu_si128(reinterpret_cast<const __m128i*>(status + row));
        auto mask = _mm_cmpeq_epi32(states, _mm_set1_epi32(paid));
        if (!Clustered || _mm_movemask_epi8(mask) != 0) {
            const auto qty = _mm_loadu_si128(reinterpret_cast<const __m128i*>(quantity + row));
            mask = _mm_and_si128(mask, _mm_cmpgt_epi32(qty, _mm_set1_epi32(minimum_exclusive)));
            if constexpr (Nullable) {
                const auto bits = _mm_set1_epi32(bitmap[row / 8] >> (row % 8));
                const auto valid = _mm_cmpeq_epi32(_mm_and_si128(bits, lane_bits), lane_bits);
                mask = _mm_and_si128(mask, valid);
            }
            const auto values = _mm_loadu_si128(reinterpret_cast<const __m128i*>(amount + row));
            mask = _mm_and_si128(mask, values);
        }
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out + row), mask);
    }
#endif
    for (; row < count; ++row) {
        const bool valid = !Nullable || (bitmap[row / 8] & (1u << (row % 8)));
        out[row] = valid && status[row] == paid && quantity[row] > minimum_exclusive ? amount[row] : 0;
    }
}
}

// A null bitmap pointer means the producer guarantees every quantity is valid.
extern "C" void sql_cpp_project(const std::int32_t* status, const std::int32_t* quantity,
                                const std::int32_t* amount, const std::uint8_t* bitmap,
                                std::int32_t* out, std::size_t count, bool clustered) {
    if (bitmap) {
        project<true, false>(status, quantity, amount, bitmap, out, count);
    } else if (clustered) {
        project<false, true>(status, quantity, amount, bitmap, out, count);
    } else {
        project<false, false>(status, quantity, amount, bitmap, out, count);
    }
}
