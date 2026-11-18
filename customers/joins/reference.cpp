#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>

namespace {
using i32 = std::int32_t;
using u32 = std::uint32_t;

i32 bits(u32 value) {
    i32 result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

struct Table {
    const i32* rows;
    std::size_t count;
};

// Hash chains visit duplicate keys in input order. Four independent indexes
// let the fixed plan fuse the join pipeline and write final rows directly.
struct Index {
    Table table;
    i32* heads;
    i32* links;
    std::size_t mask;

    std::size_t bucket(i32 key) const {
        const auto value = static_cast<u32>(key);
        return (value ^ (value >> 16)) & mask;
    }

    bool build(Table source, i32* scratch, std::size_t& remaining) {
        std::size_t buckets = 1;
        while (buckets < source.count) buckets *= 2;
        if (buckets + source.count > remaining) return false;
        remaining -= buckets + source.count;
        table = source;
        heads = scratch + remaining;
        links = heads + buckets;
        mask = buckets - 1;
        std::fill_n(heads, buckets, -1);
        for (std::size_t row = source.count; row-- > 0;) {
            auto& head = heads[bucket(source.rows[row * 2])];
            links[row] = head;
            head = static_cast<i32>(row);
        }
        return true;
    }

    template<class Visit>
    void matches(i32 key, Visit visit) const {
        for (auto row = heads[bucket(key)]; row >= 0; row = links[row]) {
            if (table.rows[row * 2] == key) visit(table.rows[row * 2 + 1]);
        }
    }
};
}

extern "C" std::int32_t pricing_cpp(const i32* offsets, const i32* data,
                                   i32* workspace, std::size_t capacity) {
    auto table = [&](int id) {
        return Table{data + offsets[id], static_cast<std::size_t>(offsets[id + 1] - offsets[id]) / 2};
    };
    Index price, eligible, promo, adjustment;
    if (!price.build(table(1), workspace, capacity) ||
        !eligible.build(table(2), workspace, capacity) ||
        !promo.build(table(3), workspace, capacity) ||
        !adjustment.build(table(4), workspace, capacity)) return -1;

    std::size_t used = 0;
    bool failed = false;
    const auto sales = table(0);
    auto sale = [&](std::size_t row) {
        const auto key = sales.rows[row * 2];
        const auto quantity = static_cast<u32>(sales.rows[row * 2 + 1]);
        if (key < 0) return;
        price.matches(key, [&](i32 p) {
            const auto revenue = quantity * static_cast<u32>(p);
            eligible.matches(key, [&](i32 flag) {
                if (flag != 1) return;
                promo.matches(key, [&](i32 bonus) {
                    adjustment.matches(key, [&](i32 adj) {
                        if (capacity - used < 2) {
                            failed = true;
                            return;
                        }
                        auto capped = std::min(bits(static_cast<u32>(adj) * 2), 20);
                        auto discount = (1u + static_cast<u32>(bonus)) * static_cast<u32>(capped);
                        workspace[used++] = key;
                        workspace[used++] = bits(revenue + discount + 1u);
                    });
                });
            });
        });
    };
    std::size_t row = 0;
    for (; sales.count - row >= 4; row += 4) {
        auto first = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sales.rows + 2 * row));
        auto second = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sales.rows + 2 * row + 4));
        auto amounts = _mm_shuffle_ps(_mm_castsi128_ps(first), _mm_castsi128_ps(second), _MM_SHUFFLE(3, 1, 3, 1));
        auto keep = _mm_cmpgt_epi32(_mm_castps_si128(amounts), _mm_set1_epi32(1));
        auto mask = _mm_movemask_ps(_mm_castsi128_ps(keep));
        for (int lane = 0; lane < 4; ++lane) {
            if (mask & (1 << lane)) sale(row + lane);
        }
    }
    for (; row < sales.count; ++row) {
        if (sales.rows[row * 2 + 1] >= 2) sale(row);
    }
    return failed ? -1 : static_cast<i32>(used / 2);
}
