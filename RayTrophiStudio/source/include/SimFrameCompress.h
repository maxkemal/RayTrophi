/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          SimFrameCompress.h
* Author:        Kemal Demirtas
* License:       MIT
* =========================================================================
*
* Tile-sparse, half-precision storage for the timeline's grid-domain scrub
* cache.
*
* ★★★ WHY THIS EXISTS. The RAM frame cache stored a DEEP COPY of the whole
* FluidGrid per frame: density, temperature, fuel, interaction, pressure,
* divergence and all three MAC velocity arrays. MEASURED 2026-09-20 on a
* 160x248x160 nuclear domain that is 57.2M floats = 218 MiB PER FRAME, so a
* hundred cached frames wanted 21.3 GiB and the bake had to be cut short.
*
* Three things were wrong, and each is worth keeping written down:
*
* ★ The app already knew better. SimCache's DISK format (SimCache.cpp) writes
*   exactly four arrays - density, temperature, fuel, interaction - and its own
*   header says velocity is "left zero-sized-safe". The same program therefore
*   already treats five of those nine arrays as unnecessary for playback; the
*   RAM path simply never inherited that.
*
* ★ The fields are almost entirely empty. The measured plume occupied 8.3% of
*   its domain at its largest. Storing the other 91.7% densely is most of the
*   cost, and no amount of shaving the field LIST fixes that.
*
* ★★★ And the budget was counted in FRAMES (kMaxCachedSimFrames = 600), which
*   only bounds memory if a frame has a fixed size. It does not: at 64^3 a
*   frame is ~4 MiB and the cap means 2.4 GiB; at 160x248x160 the same cap
*   means 137 GiB. A cap that scales with the thing it is supposed to limit is
*   not a cap. Byte accounting lives in `bytes()` below for exactly this.
*
* Format: the volume is cut into 8^3 tiles. A tile holding nothing above
* `kTileEpsilon` stores NO data at all, just a sentinel index; the rest store
* 512 IEEE-754 halves. There is no entropy coding - the win is all in the
* skipped tiles and the 2x from half, which together took the measured frame
* from 218 MiB to roughly 5 MiB.
*
* ★ Half precision is a DELIBERATE loss and it is bounded: these fields are
* normalized (density ~0-3, temperature 0-10), where half carries ~3 decimal
* digits. It is a scrub cache and a render source, not a checkpoint you can
* resume a bit-exact simulation from - which is why velocity is treated
* separately; see `has_velocity`.
*/

#ifndef RAYTROPHI_SIM_FRAME_COMPRESS_H
#define RAYTROPHI_SIM_FRAME_COMPRESS_H

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>

namespace RayTrophiSim {
namespace SimFrameCompress {

// 8^3 = 512 cells. Small enough that a thin plume skips most of the volume,
// large enough that the per-tile index stays a rounding error (one int32 per
// 512 cells is 0.05 bytes per cell).
constexpr int kTileDim = 8;
constexpr int kTileCells = kTileDim * kTileDim * kTileDim;
// Below this a cell is stored as exactly zero. Deliberately well under the
// 0.01 density threshold the plume measurement uses, so compression can never
// be what makes a cloud look smaller than it measured.
constexpr float kTileEpsilon = 1.0e-4f;
constexpr int32_t kEmptyTile = -1;

inline uint16_t floatToHalf(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = bits & 0x007FFFFFu;
    if (exp >= 31) {
        // Overflow or inf/nan. Saturate rather than produce a NaN a shader
        // would carry into the image.
        const bool is_nan = ((bits & 0x7F800000u) == 0x7F800000u) && (mant != 0u);
        return static_cast<uint16_t>(sign | 0x7BFFu | (is_nan ? 0u : 0u));
    }
    if (exp <= 0) {
        // Subnormal or underflow to zero.
        if (exp < -10) return static_cast<uint16_t>(sign);
        mant |= 0x00800000u;
        const uint32_t shift = static_cast<uint32_t>(14 - exp);
        const uint32_t half_mant = mant >> shift;
        // Round to nearest even on the dropped bits.
        const uint32_t round_bit = 1u << (shift - 1u);
        uint32_t value = half_mant;
        if ((mant & (round_bit - 1u)) != 0u || (mant & round_bit) != 0u) {
            if ((mant & round_bit) && ((mant & (round_bit - 1u)) || (half_mant & 1u))) ++value;
        }
        return static_cast<uint16_t>(sign | value);
    }
    uint32_t value = sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13);
    // Round to nearest even.
    if ((mant & 0x1FFFu) > 0x1000u ||
        ((mant & 0x1FFFu) == 0x1000u && ((mant >> 13) & 1u))) {
        ++value;
    }
    return static_cast<uint16_t>(value);
}

inline float halfToFloat(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x03FFu;
    uint32_t bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int e = -14;
            uint32_t m = mant;
            while ((m & 0x0400u) == 0u) { m <<= 1; --e; }
            m &= 0x03FFu;
            bits = sign | (static_cast<uint32_t>(e + 127) << 23) | (m << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

// One scalar field over one lattice. `dim_*` are the field's OWN dimensions,
// not the cell lattice's: a MAC velocity component is one wider on its axis,
// and storing the cell dims here instead would silently truncate a face array.
struct Field {
    int dim_x = 0, dim_y = 0, dim_z = 0;
    int tiles_x = 0, tiles_y = 0, tiles_z = 0;
    std::vector<int32_t> tile_index;   // per tile: start in `data`, or kEmptyTile
    std::vector<uint16_t> data;        // kTileCells halves per stored tile

    bool empty() const { return tile_index.empty(); }
    std::size_t bytes() const {
        return tile_index.size() * sizeof(int32_t) + data.size() * sizeof(uint16_t);
    }
    void clear() {
        dim_x = dim_y = dim_z = 0;
        tiles_x = tiles_y = tiles_z = 0;
        std::vector<int32_t>().swap(tile_index);
        std::vector<uint16_t>().swap(data);
    }
};

// Pack `src` (dim_x*dim_y*dim_z floats, x-fastest) into `out`. An empty or
// wrongly sized source produces an EMPTY field, which decompresses back to an
// empty vector - the caller's "this channel is off" state, preserved.
inline void compress(const std::vector<float>& src,
                     int dim_x, int dim_y, int dim_z,
                     Field& out) {
    out.clear();
    const std::size_t expected =
        static_cast<std::size_t>(dim_x) *
        static_cast<std::size_t>(dim_y) *
        static_cast<std::size_t>(dim_z);
    if (expected == 0u || src.size() != expected) return;

    out.dim_x = dim_x; out.dim_y = dim_y; out.dim_z = dim_z;
    out.tiles_x = (dim_x + kTileDim - 1) / kTileDim;
    out.tiles_y = (dim_y + kTileDim - 1) / kTileDim;
    out.tiles_z = (dim_z + kTileDim - 1) / kTileDim;
    const std::size_t tile_count =
        static_cast<std::size_t>(out.tiles_x) *
        static_cast<std::size_t>(out.tiles_y) *
        static_cast<std::size_t>(out.tiles_z);
    out.tile_index.assign(tile_count, kEmptyTile);

    const std::size_t plane = static_cast<std::size_t>(dim_x) * static_cast<std::size_t>(dim_y);
    std::size_t stored = 0;
    // Two passes: count first so `data` is allocated once. A push_back per
    // stored tile would reallocate a multi-megabyte buffer repeatedly, which on
    // a dense frame costs more than the compression saves.
    for (int tz = 0; tz < out.tiles_z; ++tz)
      for (int ty = 0; ty < out.tiles_y; ++ty)
        for (int tx = 0; tx < out.tiles_x; ++tx) {
            const int z0 = tz * kTileDim, z1 = (std::min)(z0 + kTileDim, dim_z);
            const int y0 = ty * kTileDim, y1 = (std::min)(y0 + kTileDim, dim_y);
            const int x0 = tx * kTileDim, x1 = (std::min)(x0 + kTileDim, dim_x);
            bool occupied = false;
            for (int z = z0; z < z1 && !occupied; ++z)
              for (int y = y0; y < y1 && !occupied; ++y) {
                const std::size_t row = static_cast<std::size_t>(z) * plane +
                                        static_cast<std::size_t>(y) * static_cast<std::size_t>(dim_x);
                for (int x = x0; x < x1; ++x) {
                    if (std::fabs(src[row + static_cast<std::size_t>(x)]) > kTileEpsilon) {
                        occupied = true; break;
                    }
                }
              }
            if (occupied) {
                const std::size_t ti =
                    static_cast<std::size_t>(tx) +
                    static_cast<std::size_t>(ty) * static_cast<std::size_t>(out.tiles_x) +
                    static_cast<std::size_t>(tz) * static_cast<std::size_t>(out.tiles_x) *
                                                   static_cast<std::size_t>(out.tiles_y);
                out.tile_index[ti] = static_cast<int32_t>(stored);
                ++stored;
            }
        }

    out.data.assign(stored * static_cast<std::size_t>(kTileCells), 0u);
    for (int tz = 0; tz < out.tiles_z; ++tz)
      for (int ty = 0; ty < out.tiles_y; ++ty)
        for (int tx = 0; tx < out.tiles_x; ++tx) {
            const std::size_t ti =
                static_cast<std::size_t>(tx) +
                static_cast<std::size_t>(ty) * static_cast<std::size_t>(out.tiles_x) +
                static_cast<std::size_t>(tz) * static_cast<std::size_t>(out.tiles_x) *
                                               static_cast<std::size_t>(out.tiles_y);
            const int32_t slot = out.tile_index[ti];
            if (slot == kEmptyTile) continue;
            uint16_t* dst = out.data.data() + static_cast<std::size_t>(slot) * kTileCells;
            const int z0 = tz * kTileDim, z1 = (std::min)(z0 + kTileDim, dim_z);
            const int y0 = ty * kTileDim, y1 = (std::min)(y0 + kTileDim, dim_y);
            const int x0 = tx * kTileDim, x1 = (std::min)(x0 + kTileDim, dim_x);
            for (int z = z0; z < z1; ++z)
              for (int y = y0; y < y1; ++y) {
                const std::size_t row = static_cast<std::size_t>(z) * plane +
                                        static_cast<std::size_t>(y) * static_cast<std::size_t>(dim_x);
                const int lz = z - z0, ly = y - y0;
                for (int x = x0; x < x1; ++x) {
                    const int lx = x - x0;
                    dst[lx + ly * kTileDim + lz * kTileDim * kTileDim] =
                        floatToHalf(src[row + static_cast<std::size_t>(x)]);
                }
              }
        }
}

// Unpack into `dst`. An empty field clears `dst` rather than zero-filling it:
// "this channel was never stored" and "this channel is all zeros" are different
// states downstream, and collapsing them would turn a disabled channel into an
// enabled one full of zeros.
inline void decompress(const Field& src, std::vector<float>& dst) {
    if (src.empty()) { std::vector<float>().swap(dst); return; }
    const std::size_t count =
        static_cast<std::size_t>(src.dim_x) *
        static_cast<std::size_t>(src.dim_y) *
        static_cast<std::size_t>(src.dim_z);
    dst.assign(count, 0.0f);
    const std::size_t plane =
        static_cast<std::size_t>(src.dim_x) * static_cast<std::size_t>(src.dim_y);
    for (int tz = 0; tz < src.tiles_z; ++tz)
      for (int ty = 0; ty < src.tiles_y; ++ty)
        for (int tx = 0; tx < src.tiles_x; ++tx) {
            const std::size_t ti =
                static_cast<std::size_t>(tx) +
                static_cast<std::size_t>(ty) * static_cast<std::size_t>(src.tiles_x) +
                static_cast<std::size_t>(tz) * static_cast<std::size_t>(src.tiles_x) *
                                               static_cast<std::size_t>(src.tiles_y);
            const int32_t slot = src.tile_index[ti];
            if (slot == kEmptyTile) continue;
            const uint16_t* s = src.data.data() + static_cast<std::size_t>(slot) * kTileCells;
            const int z0 = tz * kTileDim, z1 = (std::min)(z0 + kTileDim, src.dim_z);
            const int y0 = ty * kTileDim, y1 = (std::min)(y0 + kTileDim, src.dim_y);
            const int x0 = tx * kTileDim, x1 = (std::min)(x0 + kTileDim, src.dim_x);
            for (int z = z0; z < z1; ++z)
              for (int y = y0; y < y1; ++y) {
                const std::size_t row = static_cast<std::size_t>(z) * plane +
                                        static_cast<std::size_t>(y) * static_cast<std::size_t>(src.dim_x);
                const int lz = z - z0, ly = y - y0;
                for (int x = x0; x < x1; ++x) {
                    const int lx = x - x0;
                    dst[row + static_cast<std::size_t>(x)] =
                        halfToFloat(s[lx + ly * kTileDim + lz * kTileDim * kTileDim]);
                }
              }
        }
}

// Resize to the right length and zero it, for a field that was deliberately
// NOT stored (velocity on a non-keyframe). ★ Distinct from decompress()' empty
// case on purpose: an absent velocity must come back as a correctly sized field
// of zeros, the way SimCache's disk reader already does it, because a
// zero-LENGTH velocity array means something else entirely to the solver.
inline void zeroFill(std::vector<float>& dst, std::size_t count) {
    dst.assign(count, 0.0f);
}

} // namespace SimFrameCompress
} // namespace RayTrophiSim

#endif // RAYTROPHI_SIM_FRAME_COMPRESS_H
