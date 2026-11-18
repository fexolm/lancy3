// V1 little-endian telemetry record (8 bytes): flags:u8, channel:u8,
// sample:i16, zero_point:i16, sequence:u16. This customer needs channel 7.
#include <cstddef>
#include <cstdint>
#include <immintrin.h>
#include <limits>

namespace {
constexpr std::size_t record_size = 8;
constexpr std::int32_t missing = std::numeric_limits<std::int32_t>::min();
constexpr int header_mask = 0xff01;
constexpr int selected = (7 << 8) | 1;

inline int signed_le16(const std::uint8_t* p) {
    const int bits = p[0] | (int(p[1]) << 8);
    return (bits ^ 0x8000) - 0x8000;
}

inline std::int32_t decode_one(const std::uint8_t* record) {
    if ((record[0] & 1) == 0 || record[1] != 7) {
        return missing;
    }
    return (signed_le16(record + 2) - signed_le16(record + 4)) * 3;
}

inline __m128i decode_four(const std::uint8_t* records) {
    const auto a = _mm_castsi128_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(records)));
    const auto b = _mm_castsi128_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(records + 16)));
    const auto headers = _mm_castps_si128(_mm_shuffle_ps(a, b, _MM_SHUFFLE(2, 0, 2, 0)));
    const auto trailers = _mm_castps_si128(_mm_shuffle_ps(a, b, _MM_SHUFFLE(3, 1, 3, 1)));
    const auto sample = _mm_srai_epi32(headers, 16);
    const auto zero_point = _mm_srai_epi32(_mm_slli_epi32(trailers, 16), 16);
    const auto delta = _mm_sub_epi32(sample, zero_point);
    const auto scaled = _mm_add_epi32(_mm_slli_epi32(delta, 1), delta);
    const auto valid = _mm_cmpeq_epi32(_mm_and_si128(headers, _mm_set1_epi32(header_mask)),
                                       _mm_set1_epi32(selected));
    return _mm_or_si128(_mm_and_si128(valid, scaled),
                        _mm_andnot_si128(valid, _mm_set1_epi32(missing)));
}

#if defined(__AVX2__)
inline __m256i decode_eight(const std::uint8_t* records) {
    const auto a = _mm256_castsi256_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(records)));
    const auto b = _mm256_castsi256_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(records + 32)));
    const auto headers = _mm256_castps_si256(_mm256_shuffle_ps(a, b, _MM_SHUFFLE(2, 0, 2, 0)));
    const auto trailers = _mm256_castps_si256(_mm256_shuffle_ps(a, b, _MM_SHUFFLE(3, 1, 3, 1)));
    const auto sample = _mm256_srai_epi32(headers, 16);
    const auto zero_point = _mm256_srai_epi32(_mm256_slli_epi32(trailers, 16), 16);
    const auto delta = _mm256_sub_epi32(sample, zero_point);
    const auto scaled = _mm256_add_epi32(_mm256_slli_epi32(delta, 1), delta);
    const auto valid = _mm256_cmpeq_epi32(_mm256_and_si256(headers, _mm256_set1_epi32(header_mask)),
                                          _mm256_set1_epi32(selected));
    const auto result = _mm256_blendv_epi8(_mm256_set1_epi32(missing), scaled, valid);
    // The two lane-local shuffles produced record pairs 0,2,1,3.
    return _mm256_permute4x64_epi64(result, _MM_SHUFFLE(3, 1, 2, 0));
}
#endif
}  // namespace

extern "C" bool decoder_batch(const std::uint8_t* bytes, std::size_t len,
                              std::int32_t* out, std::size_t count) {
    if (len % record_size != 0 || len / record_size != count) {
        return false;
    }
    std::size_t i = 0;
#if defined(__AVX2__)
    for (; count - i >= 8; i += 8) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), decode_eight(bytes + i * record_size));
    }
#endif
    for (; count - i >= 4; i += 4) {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out + i), decode_four(bytes + i * record_size));
    }
    for (; i < count; ++i) {
        out[i] = decode_one(bytes + i * record_size);
    }
    return true;
}
