#include <cstddef>
#include <cstdint>
#include <emmintrin.h>

namespace {
using I = std::int32_t;

// The deployed policy is known when the C++ is built. The IR version receives
// the same two thresholds when the tenant installs the policy.
constexpr I urgent_severity = 6;
constexpr I retry_limit = 5;
constexpr I priority = 1, regulated = 2, duplicate = 4, hold = 8;
constexpr I discard = 0, normal = 1, expedited = 2;
constexpr I dead_letter = 3, quarantine = 4, audit = 5;

inline I route_one(I severity, I attempts, I flags) noexcept {
    if (flags & hold) return quarantine;
    if (attempts >= retry_limit) return dead_letter;
    if ((flags & duplicate) && severity < urgent_severity) return discard;
    if (severity >= urgent_severity || ((flags & priority) && attempts < 2))
        return expedited;
    return (flags & regulated) ? audit : normal;
}

inline __m128i pick(__m128i mask, I value, __m128i otherwise) noexcept {
    return _mm_or_si128(_mm_and_si128(mask, _mm_set1_epi32(value)),
                        _mm_andnot_si128(mask, otherwise));
}

inline __m128i has(__m128i flags, I bit) noexcept {
    const auto bit_vector = _mm_set1_epi32(bit);
    return _mm_cmpeq_epi32(_mm_and_si128(flags, bit_vector), bit_vector);
}

inline __m128i route_four(__m128i severity, __m128i attempts, __m128i flags) noexcept {
    const auto critical = _mm_cmpgt_epi32(severity, _mm_set1_epi32(urgent_severity - 1));
    const auto first_attempts = _mm_cmpgt_epi32(_mm_set1_epi32(2), attempts);
    const auto urgent = _mm_or_si128(critical, _mm_and_si128(has(flags, priority), first_attempts));
    const auto low_duplicate = _mm_andnot_si128(critical, has(flags, duplicate));
    const auto expired = _mm_cmpgt_epi32(attempts, _mm_set1_epi32(retry_limit - 1));
    auto result = pick(has(flags, regulated), audit, _mm_set1_epi32(normal));
    result = pick(urgent, expedited, result);
    result = _mm_andnot_si128(low_duplicate, result);
    result = pick(expired, dead_letter, result);
    return pick(has(flags, hold), quarantine, result);
}
} // namespace

extern "C" std::int32_t rules_cpp_one(std::int32_t severity, std::int32_t attempts,
                                     std::int32_t flags) noexcept {
    return route_one(severity, attempts, flags);
}

extern "C" void rules_cpp_batch(const std::int32_t* severity, const std::int32_t* attempts,
                                const std::int32_t* flags, std::int32_t* out,
                                std::size_t n) noexcept {
    std::size_t i = 0;
    for (; n - i >= 4; i += 4) {
        const auto vectors = route_four(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(severity + i)),
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(attempts + i)),
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(flags + i)));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out + i), vectors);
    }
    for (; i < n; ++i) out[i] = route_one(severity[i], attempts[i], flags[i]);
}
