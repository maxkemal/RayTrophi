/**
 * @file CurlNoise.cpp
 * @brief Implementation of procedural noise functions
 */

#include "CurlNoise.h"
#include <cmath>
#include <algorithm>
#include <random>

namespace Physics {
namespace Noise {

// ═══════════════════════════════════════════════════════════════════════════════
// PERMUTATION TABLE (Ken Perlin's improved noise)
// ═══════════════════════════════════════════════════════════════════════════════

// Default permutation table
static std::array<int, 512> s_perm;
static bool s_initialized = false;

// 3D gradient vectors
const std::array<Vec3, 16> gradients3D = {{
    Vec3( 1,  1,  0), Vec3(-1,  1,  0), Vec3( 1, -1,  0), Vec3(-1, -1,  0),
    Vec3( 1,  0,  1), Vec3(-1,  0,  1), Vec3( 1,  0, -1), Vec3(-1,  0, -1),
    Vec3( 0,  1,  1), Vec3( 0, -1,  1), Vec3( 0,  1, -1), Vec3( 0, -1, -1),
    Vec3( 1,  1,  0), Vec3(-1,  1,  0), Vec3( 0, -1,  1), Vec3( 0, -1, -1)
}};

void initializeWithSeed(int seed) {
    std::array<int, 256> p;
    for (int i = 0; i < 256; ++i) p[i] = i;
    
    std::mt19937 rng(seed);
    std::shuffle(p.begin(), p.end(), rng);
    
    for (int i = 0; i < 256; ++i) {
        s_perm[i] = s_perm[i + 256] = p[i];
    }
    s_initialized = true;
}

static void ensureInitialized() {
    if (!s_initialized) {
        initializeWithSeed(0);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

Vec3 gradient3D(int ix, int iy, int iz, int seed) {
    ensureInitialized();
    // Hash the coordinates
    int idx = s_perm[(s_perm[(s_perm[(ix + seed) & 255] + iy) & 255] + iz) & 255] & 15;
    return gradients3D[idx];
}

// ═══════════════════════════════════════════════════════════════════════════════
// PERLIN NOISE
// ═══════════════════════════════════════════════════════════════════════════════

float perlin2D(float x, float y, int seed) {
    ensureInitialized();
    
    int ix = (int)std::floor(x);
    int iy = (int)std::floor(y);
    
    float fx = x - ix;
    float fy = y - iy;
    
    float u = smootherstep(fx);
    float v = smootherstep(fy);
    
    // Get gradients at corners (using 2D subset)
    auto grad = [&](int i, int j) -> float {
        int idx = s_perm[(s_perm[(i + seed) & 255] + j) & 255] & 7;
        float gx = (idx & 1) ? 1.0f : -1.0f;
        float gy = (idx & 2) ? 1.0f : -1.0f;
        float dx = fx - (float)(i - ix);
        float dy = fy - (float)(j - iy);
        return gx * dx + gy * dy;
    };
    
    float n00 = grad(ix, iy);
    float n10 = grad(ix + 1, iy);
    float n01 = grad(ix, iy + 1);
    float n11 = grad(ix + 1, iy + 1);
    
    float nx0 = lerp(n00, n10, u);
    float nx1 = lerp(n01, n11, u);
    
    return lerp(nx0, nx1, v);
}

float perlin3D(float x, float y, float z, int seed) {
    ensureInitialized();
    
    int ix = (int)std::floor(x);
    int iy = (int)std::floor(y);
    int iz = (int)std::floor(z);
    
    float fx = x - ix;
    float fy = y - iy;
    float fz = z - iz;
    
    float u = smootherstep(fx);
    float v = smootherstep(fy);
    float w = smootherstep(fz);
    
    // Get gradients and dot products at corners
    auto grad = [&](int i, int j, int k) -> float {
        Vec3 g = gradient3D(i, j, k, seed);
        float dx = fx - (float)(i - ix);
        float dy = fy - (float)(j - iy);
        float dz = fz - (float)(k - iz);
        return g.x * dx + g.y * dy + g.z * dz;
    };
    
    float n000 = grad(ix, iy, iz);
    float n100 = grad(ix + 1, iy, iz);
    float n010 = grad(ix, iy + 1, iz);
    float n110 = grad(ix + 1, iy + 1, iz);
    float n001 = grad(ix, iy, iz + 1);
    float n101 = grad(ix + 1, iy, iz + 1);
    float n011 = grad(ix, iy + 1, iz + 1);
    float n111 = grad(ix + 1, iy + 1, iz + 1);
    
    float nx00 = lerp(n000, n100, u);
    float nx10 = lerp(n010, n110, u);
    float nx01 = lerp(n001, n101, u);
    float nx11 = lerp(n011, n111, u);
    
    float nxy0 = lerp(nx00, nx10, v);
    float nxy1 = lerp(nx01, nx11, v);
    
    return lerp(nxy0, nxy1, w);
}

// ═══════════════════════════════════════════════════════════════════════════════
// SIMPLEX NOISE
// ═══════════════════════════════════════════════════════════════════════════════

// Skew factors for simplex grid
static const float F2 = 0.3660254037844386f;  // (sqrt(3) - 1) / 2
static const float G2 = 0.21132486540518713f; // (3 - sqrt(3)) / 6
static const float F3 = 1.0f / 3.0f;
static const float G3 = 1.0f / 6.0f;
static const float F4 = 0.309016994374947f;   // (sqrt(5) - 1) / 4
static const float G4 = 0.138196601125011f;   // (5 - sqrt(5)) / 20

float simplex2D(float x, float y, int seed) {
    ensureInitialized();
    
    float s = (x + y) * F2;
    int i = (int)std::floor(x + s);
    int j = (int)std::floor(y + s);
    
    float t = (i + j) * G2;
    float X0 = i - t;
    float Y0 = j - t;
    float x0 = x - X0;
    float y0 = y - Y0;
    
    int i1, j1;
    if (x0 > y0) { i1 = 1; j1 = 0; }
    else { i1 = 0; j1 = 1; }
    
    float x1 = x0 - i1 + G2;
    float y1 = y0 - j1 + G2;
    float x2 = x0 - 1.0f + 2.0f * G2;
    float y2 = y0 - 1.0f + 2.0f * G2;
    
    auto contribution = [&](float cx, float cy, int gi, int gj) -> float {
        float t = 0.5f - cx*cx - cy*cy;
        if (t < 0) return 0.0f;
        int idx = s_perm[(s_perm[(gi + seed) & 255] + gj) & 255] & 7;
        float gx = (idx & 1) ? 1.0f : -1.0f;
        float gy = (idx & 2) ? 1.0f : -1.0f;
        t *= t;
        return t * t * (gx * cx + gy * cy);
    };
    
    float n0 = contribution(x0, y0, i, j);
    float n1 = contribution(x1, y1, i + i1, j + j1);
    float n2 = contribution(x2, y2, i + 1, j + 1);
    
    return 70.0f * (n0 + n1 + n2);
}

float simplex3D(float x, float y, float z, int seed) {
    ensureInitialized();
    
    float s = (x + y + z) * F3;
    int i = (int)std::floor(x + s);
    int j = (int)std::floor(y + s);
    int k = (int)std::floor(z + s);
    
    float t = (i + j + k) * G3;
    float X0 = i - t, Y0 = j - t, Z0 = k - t;
    float x0 = x - X0, y0 = y - Y0, z0 = z - Z0;
    
    int i1, j1, k1, i2, j2, k2;
    if (x0 >= y0) {
        if (y0 >= z0) { i1=1; j1=0; k1=0; i2=1; j2=1; k2=0; }
        else if (x0 >= z0) { i1=1; j1=0; k1=0; i2=1; j2=0; k2=1; }
        else { i1=0; j1=0; k1=1; i2=1; j2=0; k2=1; }
    } else {
        if (y0 < z0) { i1=0; j1=0; k1=1; i2=0; j2=1; k2=1; }
        else if (x0 < z0) { i1=0; j1=1; k1=0; i2=0; j2=1; k2=1; }
        else { i1=0; j1=1; k1=0; i2=1; j2=1; k2=0; }
    }
    
    float x1 = x0 - i1 + G3, y1 = y0 - j1 + G3, z1 = z0 - k1 + G3;
    float x2 = x0 - i2 + 2.0f*G3, y2 = y0 - j2 + 2.0f*G3, z2 = z0 - k2 + 2.0f*G3;
    float x3 = x0 - 1.0f + 3.0f*G3, y3 = y0 - 1.0f + 3.0f*G3, z3 = z0 - 1.0f + 3.0f*G3;
    
    auto contribution = [&](float cx, float cy, float cz, int gi, int gj, int gk) -> float {
        float t = 0.6f - cx*cx - cy*cy - cz*cz;
        if (t < 0) return 0.0f;
        Vec3 g = gradient3D(gi, gj, gk, seed);
        t *= t;
        return t * t * (g.x*cx + g.y*cy + g.z*cz);
    };
    
    float n0 = contribution(x0, y0, z0, i, j, k);
    float n1 = contribution(x1, y1, z1, i+i1, j+j1, k+k1);
    float n2 = contribution(x2, y2, z2, i+i2, j+j2, k+k2);
    float n3 = contribution(x3, y3, z3, i+1, j+1, k+1);
    
    return 32.0f * (n0 + n1 + n2 + n3);
}

float simplex4D(float x, float y, float z, float w, int seed) {
    // Simplified 4D simplex (uses 3D + time warping)
    float t = w * 0.7f;
    return simplex3D(x + t * 0.1f, y + t * 0.2f, z + t * 0.15f, seed);
}

// ═══════════════════════════════════════════════════════════════════════════════
// FRACTAL NOISE (FBM)
// ═══════════════════════════════════════════════════════════════════════════════

float fbm2D(float x, float y, int octaves, float frequency,
            float lacunarity, float persistence, int seed) {
    float sum = 0.0f;
    float amplitude = 1.0f;
    float max_amplitude = 0.0f;
    
    for (int i = 0; i < octaves; ++i) {
        sum += perlin2D(x * frequency, y * frequency, seed + i) * amplitude;
        max_amplitude += amplitude;
        frequency *= lacunarity;
        amplitude *= persistence;
    }
    
    return sum / max_amplitude;
}

float fbm3D(const Vec3& p, int octaves, float frequency,
            float lacunarity, float persistence, int seed) {
    float sum = 0.0f;
    float amplitude = 1.0f;
    float max_amplitude = 0.0f;
    Vec3 pos = p * frequency;
    
    for (int i = 0; i < octaves; ++i) {
        sum += perlin3D(pos, seed + i) * amplitude;
        max_amplitude += amplitude;
        pos = pos * lacunarity;
        amplitude *= persistence;
    }
    
    return sum / max_amplitude;
}

float fbm3D_animated(const Vec3& p, float time, int octaves, float frequency,
                     float lacunarity, float persistence, int seed) {
    float sum = 0.0f;
    float amplitude = 1.0f;
    float max_amplitude = 0.0f;
    Vec3 pos = p * frequency;
    float t = time;
    
    for (int i = 0; i < octaves; ++i) {
        sum += simplex4D(pos.x, pos.y, pos.z, t, seed + i) * amplitude;
        max_amplitude += amplitude;
        pos = pos * lacunarity;
        t *= lacunarity;
        amplitude *= persistence;
    }
    
    return sum / max_amplitude;
}

// ═══════════════════════════════════════════════════════════════════════════════
// TURBULENCE
// ═══════════════════════════════════════════════════════════════════════════════

float turbulence3D(const Vec3& p, int octaves, float frequency,
                   float lacunarity, float persistence, int seed) {
    float sum = 0.0f;
    float amplitude = 1.0f;
    float max_amplitude = 0.0f;
    Vec3 pos = p * frequency;
    
    for (int i = 0; i < octaves; ++i) {
        sum += std::abs(perlin3D(pos, seed + i)) * amplitude;
        max_amplitude += amplitude;
        pos = pos * lacunarity;
        amplitude *= persistence;
    }
    
    return sum / max_amplitude;
}

float turbulence3D_animated(const Vec3& p, float time, int octaves, float frequency,
                            float lacunarity, float persistence, int seed) {
    float sum = 0.0f;
    float amplitude = 1.0f;
    float max_amplitude = 0.0f;
    Vec3 pos = p * frequency;
    float t = time;
    
    for (int i = 0; i < octaves; ++i) {
        sum += std::abs(simplex4D(pos.x, pos.y, pos.z, t, seed + i)) * amplitude;
        max_amplitude += amplitude;
        pos = pos * lacunarity;
        t *= lacunarity;
        amplitude *= persistence;
    }
    
    return sum / max_amplitude;
}

// ═══════════════════════════════════════════════════════════════════════════════
// CURL NOISE
// ═══════════════════════════════════════════════════════════════════════════════

Vec3 curl3D(const Vec3& p, float frequency, int seed) {
    const float eps = 0.0001f;
    Vec3 pf = p * frequency;
    
    // Sample noise at offset positions for partial derivatives
    float n_px = perlin3D(pf.x + eps, pf.y, pf.z, seed);
    float n_mx = perlin3D(pf.x - eps, pf.y, pf.z, seed);
    float n_py = perlin3D(pf.x, pf.y + eps, pf.z, seed);
    float n_my = perlin3D(pf.x, pf.y - eps, pf.z, seed);
    float n_pz = perlin3D(pf.x, pf.y, pf.z + eps, seed);
    float n_mz = perlin3D(pf.x, pf.y, pf.z - eps, seed);
    
    // Second set of samples with different seed for 3D curl
    float n2_px = perlin3D(pf.x + eps, pf.y, pf.z, seed + 1);
    float n2_mx = perlin3D(pf.x - eps, pf.y, pf.z, seed + 1);
    float n2_py = perlin3D(pf.x, pf.y + eps, pf.z, seed + 1);
    float n2_my = perlin3D(pf.x, pf.y - eps, pf.z, seed + 1);
    float n2_pz = perlin3D(pf.x, pf.y, pf.z + eps, seed + 1);
    float n2_mz = perlin3D(pf.x, pf.y, pf.z - eps, seed + 1);
    
    float n3_px = perlin3D(pf.x + eps, pf.y, pf.z, seed + 2);
    float n3_mx = perlin3D(pf.x - eps, pf.y, pf.z, seed + 2);
    float n3_py = perlin3D(pf.x, pf.y + eps, pf.z, seed + 2);
    float n3_my = perlin3D(pf.x, pf.y - eps, pf.z, seed + 2);
    float n3_pz = perlin3D(pf.x, pf.y, pf.z + eps, seed + 2);
    float n3_mz = perlin3D(pf.x, pf.y, pf.z - eps, seed + 2);
    
    // Compute partial derivatives
    float dn_dy = (n_py - n_my) / (2.0f * eps);
    float dn_dz = (n_pz - n_mz) / (2.0f * eps);
    float dn2_dx = (n2_px - n2_mx) / (2.0f * eps);
    float dn2_dz = (n2_pz - n2_mz) / (2.0f * eps);
    float dn3_dx = (n3_px - n3_mx) / (2.0f * eps);
    float dn3_dy = (n3_py - n3_my) / (2.0f * eps);
    
    // Curl = cross product of gradient
    // curl_x = dN3/dy - dN2/dz
    // curl_y = dN1/dz - dN3/dx
    // curl_z = dN2/dx - dN1/dy
    return Vec3(
        dn3_dy - dn2_dz,
        dn_dz - dn3_dx,
        dn2_dx - dn_dy
    );
}

Vec3 curl3D_animated(const Vec3& p, float time, float frequency, 
                     float speed, int seed) {
    // Offset the noise position based on time
    Vec3 animated_p = p + Vec3(time * speed * 0.7f, time * speed * 0.3f, time * speed * 0.5f);
    return curl3D(animated_p, frequency, seed);
}

Vec3 curlFBM(const Vec3& p, int octaves, float frequency,
             float lacunarity, float persistence, int seed) {
    Vec3 sum(0, 0, 0);
    float amplitude = 1.0f;
    float max_amplitude = 0.0f;
    float freq = frequency;
    
    for (int i = 0; i < octaves; ++i) {
        Vec3 c = curl3D(p, freq, seed + i * 3);
        sum = sum + c * amplitude;
        max_amplitude += amplitude;
        freq *= lacunarity;
        amplitude *= persistence;
    }
    
    return sum / max_amplitude;
}

Vec3 curlFBM_animated(const Vec3& p, float time, int octaves, float frequency,
                      float lacunarity, float persistence, 
                      float speed, int seed) {
    Vec3 sum(0, 0, 0);
    float amplitude = 1.0f;
    float max_amplitude = 0.0f;
    float freq = frequency;
    float t = time;
    
    for (int i = 0; i < octaves; ++i) {
        Vec3 c = curl3D_animated(p, t, freq, speed, seed + i * 3);
        sum = sum + c * amplitude;
        max_amplitude += amplitude;
        freq *= lacunarity;
        t *= lacunarity;
        amplitude *= persistence;
    }
    
    return sum / max_amplitude;
}

// ═══════════════════════════════════════════════════════════════════════════════
// SPECIALIZED NOISE
// ═══════════════════════════════════════════════════════════════════════════════

float voronoi3D(const Vec3& p, int seed) {
    int ix = (int)std::floor(p.x);
    int iy = (int)std::floor(p.y);
    int iz = (int)std::floor(p.z);
    
    float min_dist = 1e10f;
    
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dz = -1; dz <= 1; ++dz) {
                int cx = ix + dx;
                int cy = iy + dy;
                int cz = iz + dz;
                
                // Random point in cell
                float rx = hashFloat(cx, cy, cz, seed);
                float ry = hashFloat(cx, cy, cz, seed + 1);
                float rz = hashFloat(cx, cy, cz, seed + 2);
                
                Vec3 cell_point = Vec3(cx + rx, cy + ry, cz + rz);
                float dist = (p - cell_point).length();
                min_dist = std::min(min_dist, dist);
            }
        }
    }
    
    return min_dist;
}

float voronoiCrackle(const Vec3& p, int seed) {
    int ix = (int)std::floor(p.x);
    int iy = (int)std::floor(p.y);
    int iz = (int)std::floor(p.z);
    
    float d1 = 1e10f, d2 = 1e10f;
    
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dz = -1; dz <= 1; ++dz) {
                int cx = ix + dx;
                int cy = iy + dy;
                int cz = iz + dz;
                
                float rx = hashFloat(cx, cy, cz, seed);
                float ry = hashFloat(cx, cy, cz, seed + 1);
                float rz = hashFloat(cx, cy, cz, seed + 2);
                
                Vec3 cell_point = Vec3(cx + rx, cy + ry, cz + rz);
                float dist = (p - cell_point).length();
                
                if (dist < d1) {
                    d2 = d1;
                    d1 = dist;
                } else if (dist < d2) {
                    d2 = dist;
                }
            }
        }
    }
    
    return d2 - d1;
}

float ridgeNoise3D(const Vec3& p, int octaves, float frequency,
                   float lacunarity, float offset, int seed) {
    float sum = 0.0f;
    float amplitude = 1.0f;
    float weight = 1.0f;
    Vec3 pos = p * frequency;
    
    for (int i = 0; i < octaves; ++i) {
        float n = perlin3D(pos, seed + i);
        n = offset - std::abs(n);  // Ridge transformation
        n = n * n;
        n *= weight;
        weight = std::clamp(n * 2.0f, 0.0f, 1.0f);
        sum += n * amplitude;
        pos = pos * lacunarity;
        amplitude *= 0.5f;
    }
    
    return sum;
}

float billowNoise3D(const Vec3& p, int octaves, float frequency,
                    float lacunarity, float persistence, int seed) {
    float sum = 0.0f;
    float amplitude = 1.0f;
    float max_amplitude = 0.0f;
    Vec3 pos = p * frequency;
    
    for (int i = 0; i < octaves; ++i) {
        float n = std::abs(perlin3D(pos, seed + i)) * 2.0f - 1.0f;
        sum += n * amplitude;
        max_amplitude += amplitude;
        pos = pos * lacunarity;
        amplitude *= persistence;
    }
    
    return sum / max_amplitude;
}

// ═══════════════════════════════════════════════════════════════════════════════
// DOMAIN WARPING
// ═══════════════════════════════════════════════════════════════════════════════

Vec3 domainWarp(const Vec3& p, float warp_strength, float frequency, int seed) {
    float wx = fbm3D(p, 3, frequency, 2.0f, 0.5f, seed);
    float wy = fbm3D(p, 3, frequency, 2.0f, 0.5f, seed + 100);
    float wz = fbm3D(p, 3, frequency, 2.0f, 0.5f, seed + 200);
    
    return p + Vec3(wx, wy, wz) * warp_strength;
}

Vec3 domainWarp_animated(const Vec3& p, float time, float warp_strength,
                         float frequency, float speed, int seed) {
    float wx = fbm3D_animated(p, time * speed, 3, frequency, 2.0f, 0.5f, seed);
    float wy = fbm3D_animated(p, time * speed, 3, frequency, 2.0f, 0.5f, seed + 100);
    float wz = fbm3D_animated(p, time * speed, 3, frequency, 2.0f, 0.5f, seed + 200);
    
    return p + Vec3(wx, wy, wz) * warp_strength;
}

} // namespace Noise
} // namespace Physics
