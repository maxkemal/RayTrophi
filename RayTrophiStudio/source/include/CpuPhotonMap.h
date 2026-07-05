#pragma once
// =============================================================================
// CpuPhotonMap.h — CPU photon caustic grid (Vulkan RT parity).
// =============================================================================
// Mirrors shaders/photon_grid.glsl 1:1 so the CPU reference render converges to
// the same caustic as Vulkan RT:
//   - same 5-uint cell layout (R,G,B fixed-point *256, photon count, owner key)
//   - same PCG avalanche hash CHAINS (multiplicative-XOR hashes are symmetric
//     under (x,z) -> (-x,-z) in two's complement and rendered a point-mirrored
//     ghost caustic — see photon_grid.glsl for the derivation)
//   - same stochastic-rounding fixed-point splat + CAS slot ownership
//   - same 3x3x3 cone-kernel gather and πR²/3 normalization
// The grid ACCUMULATES across progressive passes (cleared on accumulation
// reset); readers divide by the pass count (frameSeed + 1).
// =============================================================================
#include <atomic>
#include <cstdint>
#include <memory>
#include <cmath>
#include <algorithm>
#include "Vec3.h"

class CpuPhotonGrid {
public:
    static constexpr uint32_t kTableSize = 1u << 20;   // cells, power of two (~20 MB)

    float    cellSize  = 0.05f;
    uint32_t frameSeed = 0;   // photon pass index; readers divide by frameSeed+1
    uint32_t passCount = 0;   // photon batches traced since the last clear
    uint64_t stateHash = 0;   // lights + caster-bounds signature: restart on change

    void ensureAllocated() {
        if (!m_cells) m_cells.reset(new std::atomic<uint32_t>[(size_t)kTableSize * 5]);
    }
    bool ready() const { return m_cells != nullptr; }
    void clear() {
        if (!m_cells) return;
        const size_t n = (size_t)kTableSize * 5;
        for (size_t i = 0; i < n; ++i) m_cells[i].store(0u, std::memory_order_relaxed);
    }

    static uint32_t scramble(uint32_t v) {
        uint32_t s = v * 747796405u + 2891336453u;
        uint32_t w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
        return (w >> 22u) ^ w;
    }

    struct CellCoord { int32_t x, y, z; };
    CellCoord cellCoord(const Vec3& p) const {
        const float inv = 1.0f / std::max(cellSize, 1e-4f);
        return { (int32_t)std::floor(p.x * inv),
                 (int32_t)std::floor(p.y * inv),
                 (int32_t)std::floor(p.z * inv) };
    }
    static uint32_t slotOf(const CellCoord& c) {
        uint32_t s = scramble((uint32_t)c.x ^ 0x68bc21ebu);
        s = scramble(s ^ (uint32_t)c.y);
        s = scramble(s ^ (uint32_t)c.z);
        return s & (kTableSize - 1u);
    }
    static uint32_t keyOf(const CellCoord& c) {
        uint32_t s = scramble((uint32_t)c.x ^ 0x2c1b3c6du);
        s = scramble(s ^ (uint32_t)c.y);
        s = scramble(s ^ (uint32_t)c.z);
        return s | 1u;   // 0 = empty slot
    }

    // Fixed-point splat with STOCHASTIC ROUNDING (plain truncation drops all
    // deposits below 1/256 — the common case since power ∝ 1/N) and CAS slot
    // ownership: a photon that loses the slot to another cell is DROPPED
    // instead of depositing aliased energy at an unrelated position.
    void splat(const Vec3& p, const Vec3& power, uint32_t& rng) {
        const CellCoord c = cellCoord(p);
        const size_t idx = (size_t)slotOf(c) * 5;
        const uint32_t key = keyOf(c);
        uint32_t expected = 0u;
        if (!m_cells[idx + 4].compare_exchange_strong(expected, key, std::memory_order_relaxed) &&
            expected != key) {
            return;   // slot owned by another cell
        }
        const float v[3] = {
            std::clamp((float)power.x, 0.0f, 4.0e6f) * 256.0f,
            std::clamp((float)power.y, 0.0f, 4.0e6f) * 256.0f,
            std::clamp((float)power.z, 0.0f, 4.0e6f) * 256.0f
        };
        for (int ch = 0; ch < 3; ++ch) {
            uint32_t q = (uint32_t)v[ch];
            rng = scramble(rng);
            if ((float)rng * (1.0f / 4294967296.0f) < (v[ch] - (float)q)) ++q;
            if (q) m_cells[idx + ch].fetch_add(q, std::memory_order_relaxed);
        }
        m_cells[idx + 3].fetch_add(1u, std::memory_order_relaxed);
    }

    // 3x3x3 cone-kernel gather (w = 1 - d/R, R = 1.6 cells). Returns the
    // ACCUMULATED irradiance sum — the caller divides by (frameSeed + 1).
    Vec3 gather(const Vec3& p) const {
        const float cs = std::max(cellSize, 1e-4f);
        const float inv = 1.0f / cs;
        const float rx = p.x * inv, ry = p.y * inv, rz = p.z * inv;   // grid-space
        const int32_t cx = (int32_t)std::floor(rx);
        const int32_t cy = (int32_t)std::floor(ry);
        const int32_t cz = (int32_t)std::floor(rz);
        const float R = 1.6f;   // kernel radius in cells
        float sr = 0.0f, sg = 0.0f, sb = 0.0f;
        for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            const CellCoord c{ cx + dx, cy + dy, cz + dz };
            const float ddx = (float)c.x + 0.5f - rx;
            const float ddy = (float)c.y + 0.5f - ry;
            const float ddz = (float)c.z + 0.5f - rz;
            const float w = 1.0f - std::sqrt(ddx * ddx + ddy * ddy + ddz * ddz) / R;
            if (w <= 0.0f) continue;
            const size_t idx = (size_t)slotOf(c) * 5;
            if (m_cells[idx + 4].load(std::memory_order_relaxed) != keyOf(c)) continue;
            sr += (float)m_cells[idx + 0].load(std::memory_order_relaxed) * w;
            sg += (float)m_cells[idx + 1].load(std::memory_order_relaxed) * w;
            sb += (float)m_cells[idx + 2].load(std::memory_order_relaxed) * w;
        }
        const float Rw = R * cs;
        const float norm = (1.0f / 256.0f) * 3.0f / (3.14159265f * Rw * Rw);
        return Vec3(sr * norm, sg * norm, sb * norm);
    }

private:
    std::unique_ptr<std::atomic<uint32_t>[]> m_cells;
};
