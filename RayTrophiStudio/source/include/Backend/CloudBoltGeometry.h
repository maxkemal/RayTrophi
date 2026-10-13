#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace Backend {

inline uint32_t lightningHash(uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    return value ^ (value >> 16);
}

inline float lightningRandom(uint32_t& seed) {
    seed = lightningHash(seed + 0x9e3779b9u);
    return static_cast<float>(seed & 0x00ffffffu) / 16777216.0f;
}

constexpr int kCloudBoltSegments = 80;
constexpr int kCloudBoltMainSegments = 32;

struct CloudBoltSegmentGPU {
    float a[4];       // xyz, channel weight
    float b[4];       // xyz, leader birth progress
    float growth[4];  // completion progress, parent index (-1 origin), depth, tip weight
};
static_assert(sizeof(CloudBoltSegmentGPU) == 48, "CloudBoltSegmentGPU std430 layout");

struct BoltP3 {
    float x, y, z;
};

inline BoltP3 boltAdd(BoltP3 a, BoltP3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline BoltP3 boltSub(BoltP3 a, BoltP3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

inline BoltP3 boltMul(BoltP3 a, float scale) {
    return {a.x * scale, a.y * scale, a.z * scale};
}

inline float boltDot(BoltP3 a, BoltP3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline float boltLen(BoltP3 a) {
    return std::sqrt(boltDot(a, a));
}
inline BoltP3 boltCross(BoltP3 a, BoltP3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline BoltP3 boltNorm(BoltP3 a) {
    const float length = boltLen(a);
    return length > 1e-6f ? boltMul(a, 1.0f / length) : BoltP3{0.0f, -1.0f, 0.0f};
}

inline void boltDisplacedPath(BoltP3 a, BoltP3 b, int levels, float amplitude,
                              float verticalScale, uint32_t& seed, BoltP3* out) {
    const int n = 1 << levels;
    out[0] = a;
    out[n] = b;
    const BoltP3 direction = boltNorm(boltSub(b, a));
    float scale = amplitude * boltLen(boltSub(b, a));
    for (int step = n; step > 1; step >>= 1) {
        for (int k = 0; k < n; k += step) {
            BoltP3 offset{lightningRandom(seed) - 0.5f,
                (lightningRandom(seed) - 0.5f) * verticalScale,
                lightningRandom(seed) - 0.5f};
            offset = boltSub(offset, boltMul(direction, boltDot(offset, direction)));
            out[k + step / 2] = boltAdd(boltMul(boltAdd(out[k], out[k + step]), 0.5f),
                                       boltMul(offset, 2.0f * scale));
        }
        scale *= 0.56f;
    }
}

inline BoltP3 boltForkDirection(BoltP3 tangent, float spread, uint32_t& seed) {
    const BoltP3 reference = std::abs(tangent.y) < 0.9f
        ? BoltP3{0.0f, 1.0f, 0.0f} : BoltP3{1.0f, 0.0f, 0.0f};
    const BoltP3 side = boltNorm(boltCross(tangent, reference));
    const BoltP3 other = boltCross(tangent, side);
    const float angle = lightningRandom(seed) * 6.28318530718f;
    const BoltP3 radial = boltAdd(boltMul(side, std::cos(angle)),
                                 boltMul(other, std::sin(angle)));
    return boltNorm(boltAdd(boltMul(tangent, 1.0f), boltMul(radial, spread)));
}

// Deterministic spatial tree. Birth/completion use arc-length travel time;
// every child waits for its parent's endpoint. Time never changes topology.
// clearAirBelowY: scene y of the cloud base at the strike; main-channel points
// below it are in clear air, where a ground strike's branches are seen.
inline int buildCloudBolt(BoltP3 start, BoltP3 end, bool ground, bool upward, bool vertical,
                          float branching, float spiderSpan, float clearAirBelowY, uint32_t seed,
                          CloudBoltSegmentGPU* out, BoltP3& centroid) {
    int count = 0;
    float centroidWeight = 0.0f;
    centroid = {0.0f, 0.0f, 0.0f};
    auto emit = [&](BoltP3 a, BoltP3 b, float weight, float tipWeight,
                    float birth, float completion, int parent, int depth) {
        if (count >= kCloudBoltSegments) {
            return -1;
        }
        const int index = count++;
        auto& segment = out[index];
        segment = {{a.x, a.y, a.z, weight}, {b.x, b.y, b.z, birth},
                   {completion, static_cast<float>(parent), static_cast<float>(depth), tipWeight}};
        const float lengthWeight = boltLen(boltSub(b, a)) * weight;
        centroid = boltAdd(centroid, boltMul(boltAdd(a, b), 0.5f * lengthWeight));
        centroidWeight += lengthWeight;
        return index;
    };
    const float verticalScale = ground || vertical ? 1.0f : 0.3f;
    const float strikeY = (std::min)(start.y, end.y);
    const float branchFloorY = strikeY + (std::max)(0.04f * std::fabs(start.y - end.y), 15.0f);
    BoltP3 main[kCloudBoltMainSegments + 1];
    boltDisplacedPath(start, end, 5, 0.11f, verticalScale, seed, main);
    float arrival[kCloudBoltMainSegments + 1] = {};
    for (int k = 0; k < kCloudBoltMainSegments; ++k) {
        arrival[k + 1] = arrival[k] + boltLen(boltSub(main[k + 1], main[k]));
    }
    const float travelLength = (std::max)(arrival[kCloudBoltMainSegments], 1.0f);
    for (float& value : arrival) {
        value /= travelLength;
    }
    for (int k = 0; k < kCloudBoltMainSegments; ++k) {
        emit(main[k], main[k + 1], 1.0f, 1.0f, arrival[k], arrival[k + 1], k - 1, 0);
    }

    struct Root {
        int point;
        bool spider;
    };
    Root roots[8];
    int rootCount = 0;
    if (branching > 0.0f && (ground || vertical)) {
        const int cloudEnd = ground && upward ? kCloudBoltMainSegments : 0;
        roots[rootCount++] = {cloudEnd, true};
        if (vertical || branching > 0.5f) {
            roots[rootCount++] = {vertical ? kCloudBoltMainSegments :
                (upward ? kCloudBoltMainSegments - 1 : 1), true};
        }
    }
    const int forks = static_cast<int>(branching * 6.0f + 0.5f);
    // A ground strike's main channel is mostly inside the cloud (it starts a
    // third of the column up): forks spread over the whole channel started
    // there and none showed below the base (2026-10-13). Three in four take a
    // root in the clear-air part when it has one.
    int clearLo = kCloudBoltMainSegments, clearHi = -1;
    if (ground) {
        for (int k = 1; k < kCloudBoltMainSegments - 1; ++k) {
            if (main[k].y < clearAirBelowY) {
                clearLo = (std::min)(clearLo, k);
                clearHi = (std::max)(clearHi, k);
            }
        }
    }
    for (int fork = 0; fork < forks && rootCount < 8; ++fork) {
        int point = 3 + static_cast<int>(lightningRandom(seed) * 24.0f);
        if (clearHi > clearLo && lightningRandom(seed) < 0.75f) {
            point = clearLo + static_cast<int>(lightningRandom(seed) *
                                               static_cast<float>(clearHi - clearLo + 1));
            point = (std::min)(point, clearHi);
        }
        roots[rootCount++] = {point, !ground && !vertical};
    }
    auto chain = [&](BoltP3 origin, BoltP3 direction, float length, int levels,
                     float weight, float birth, int parent, int depth, float heightScale,
                     BoltP3* points, int* indices) {
        boltDisplacedPath(origin, boltAdd(origin, boltMul(direction, length)),
                          levels, 0.18f, heightScale, seed, points);
        const int pieces = 1 << levels;
        float pathLength = 0.0f;
        // Unconnected branches die out in the air above the strike. Clamped
        // to the strike altitude, their tips lay flat around the attachment
        // point: on a peak, many branches seemed to strike one spot
        // (2026-10-13). A branch now stops where it crosses this floor.
        bool stopped = false;
        for (int k = 1; k <= pieces; ++k) {
            if (stopped) {
                points[k] = points[k - 1];
            } else if (ground && points[k].y < branchFloorY) {
                const float drop = points[k - 1].y - points[k].y;
                const float u = drop > 1e-3f
                    ? (std::clamp)((points[k - 1].y - branchFloorY) / drop, 0.0f, 1.0f) : 0.0f;
                points[k] = boltAdd(points[k - 1], boltMul(boltSub(points[k], points[k - 1]), u));
                stopped = true;
            }
            pathLength += boltLen(boltSub(points[k], points[k - 1]));
        }
        // Terminal cloud branches can form after connection, but must finish
        // before the event's final fade even at the longest authored leader phase.
        const float timePerMeter = (std::min)(1.15f / travelLength,
            (1.35f - birth) / (std::max)(pathLength, 1e-6f));
        float clock = birth;
        for (int k = 0; k < pieces; ++k) {
            const float finish = clock + boltLen(boltSub(points[k + 1], points[k])) *
                timePerMeter;
            // A piece past the stop point has no length: no light, no dot.
            const float live = boltLen(boltSub(points[k + 1], points[k])) > 0.01f ? 1.0f : 0.0f;
            const float taper = 1.0f - 0.55f * static_cast<float>(k) / pieces;
            indices[k] = emit(points[k], points[k + 1], weight * taper * live,
                weight * (1.0f - 0.55f * static_cast<float>(k + 1) / pieces) * live,
                clock, finish, parent, depth);
            if (indices[k] < 0) {
                return false;
            }
            parent = indices[k];
            clock = finish;
        }
        return true;
    };
    for (int root = 0; root < rootCount; ++root) {
        const int point = roots[root].point;
        const int parent = point - 1;
        const BoltP3 origin = main[point];
        BoltP3 direction;
        float length;
        if (roots[root].spider) {
            const float angle = lightningRandom(seed) * 6.28318530718f;
            direction = boltNorm({std::cos(angle), (lightningRandom(seed) - 0.5f) * 0.16f,
                                  std::sin(angle)});
            length = (std::min)(spiderSpan * (0.2f + 0.2f * lightningRandom(seed)),
                               travelLength * 0.24f);
        } else {
            const BoltP3 tangent = boltNorm(boltSub(main[point + 1], main[point]));
            direction = boltForkDirection(tangent, 0.55f + lightningRandom(seed) * 0.9f, seed);
            length = travelLength * (0.08f + 0.12f * lightningRandom(seed));
        }
        BoltP3 branch[5];
        int branchIndices[4];
        const float heightScale = roots[root].spider ? 0.3f : 1.0f;
        if (!chain(origin, direction, length, 2, 0.58f, arrival[point], parent, 1,
                   heightScale, branch, branchIndices)) {
            break;
        }
        if (branching > 0.5f && root < 4) {
            BoltP3 twig[3];
            int twigIndices[2];
            const auto& anchor = out[branchIndices[1]];
            const BoltP3 twigDirection = boltForkDirection(direction, 0.9f, seed);
            if (!chain(branch[2], twigDirection, length * 0.42f, 1, 0.28f,
                       anchor.growth[0], branchIndices[1], 2, heightScale, twig, twigIndices)) {
                break;
            }
            if (branching > 0.8f && root < 2) {
                BoltP3 twiglet[3];
                int twigletIndices[2];
                chain(twig[1], boltForkDirection(twigDirection, 1.1f, seed), length * 0.18f,
                      1, 0.15f, out[twigIndices[0]].growth[0], twigIndices[0], 3,
                      heightScale, twiglet, twigletIndices);
            }
        }
    }
    if (centroidWeight > 0.0f) {
        centroid = boltMul(centroid, 1.0f / centroidWeight);
    }
    return count;
}

} // namespace Backend
