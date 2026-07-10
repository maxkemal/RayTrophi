// =============================================================================
// photon_grid.glsl — Photon caustic hash grid (Faz 2 / Dilim 1).
// =============================================================================
// World-anchored uniform hash grid in a single SSBO at set 0, binding 19.
// Written by photon.rgen (light-side tracing, atomic splat), read by raygen.rgen
// (debug visualization) and later closesthit.rchit (caustic gather, Dilim 2).
//
// Layout mirrors VulkanRT::VkPhotonGridHeader (VulkanBackend.h) byte-for-byte:
// header = 64 bytes, then cells[] as flat uints (5 per cell: R,G,B fixed-point
// *256, photon count, owner key). std430; vec4s first so scalar/std430 agree.
// =============================================================================
#ifndef PHOTON_GRID_GLSL
#define PHOTON_GRID_GLSL

struct PhotonGridHeader {
    vec4 originCell;    // xyz = grid world origin (0 in Dilim 1), w = cell size (world units)
    vec4 emitCenter;    // xyz = photon emission target centre, w = target radius
    uint tableSize;     // cell count, power of two (hash mask = tableSize-1)
    uint photonCount;   // photons launched this frame
    uint frameSeed;     // per-frame RNG decorrelation
    uint debugMode;     // 0 = off, 1 = debug viz at primary hit, 2 = gather into shading (Dilim 2)
    uint lightIndex;    // emitting light (Dilim 1: one light per frame)
    uint lightCountReal;// actual scene light count (PC lightCount is zeroed during the pass)
    float energyScale;  // photon power multiplier (calibration knob, Dilim 2)
    float debugExposure;// debug visualization gain
};

layout(set = 0, binding = 19, std430) buffer PhotonGridBuf {
    PhotonGridHeader h;
    uint cells[];       // tableSize * 5 uints (R, G, B, count, owner key)
} photonGrid;

ivec3 photonGridCellCoord(vec3 p) {
    float cs = max(photonGrid.h.originCell.w, 1e-4);
    return ivec3(floor((p - photonGrid.h.originCell.xyz) / cs));
}

// Pure PCG-style avalanche (same mixer family as photonGridRandState, stateless).
uint photonGridScramble(uint v) {
    uint s = v * 747796405u + 2891336453u;
    uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

// Cell hashes are PCG CHAINS, not multiplicative-XOR (Teschner). The classic
// `x*A ^ y*B ^ z*C` with odd constants is SYMMETRIC under (x,z) -> (-x,-z):
// two's complement gives u(-p)^u(-q) = (p-1)^(q-1), which equals p^q whenever
// x and z share their trailing-zero count — ~1/3 of ALL cells, independent of
// the constants chosen. The slot hash and the owner key both had that form, so
// they collided TOGETHER and the ownership check passed: the caustic rendered a
// structured ghost copy point-mirrored through the grid origin (an X-mirrored
// pattern that tracked the object in -Z/+Z and ignored occluders — light can't
// do that; a wall test proved it was read-side aliasing). The avalanche chain
// has no such algebraic structure (verified: 0 mirror collisions over 51k cells).
uint photonGridHashCoord(ivec3 c) {
    uint s = photonGridScramble(uint(c.x) ^ 0x68bc21ebu);
    s = photonGridScramble(s ^ uint(c.y));
    s = photonGridScramble(s ^ uint(c.z));
    return s & (photonGrid.h.tableSize - 1u);
}

// Independent (different salt) hash of the cell coords: the cell's ownership
// key (0 = empty slot, so force nonzero). A splat claims an empty slot with a
// CAS; a photon that loses the slot to another cell is DROPPED (tiny energy
// loss) instead of depositing aliased energy, and readers verify ownership
// before trusting a slot — random collisions otherwise show up as colored
// speckles far from the caustic.
uint photonGridCellKey(ivec3 c) {
    uint s = photonGridScramble(uint(c.x) ^ 0x2c1b3c6du);
    s = photonGridScramble(s ^ uint(c.y));
    s = photonGridScramble(s ^ uint(c.z));
    return s | 1u;
}

// ── FAZ 2V: VOLUME photon grid (binding 20) ─────────────────────────────────
// Second, COARSER grid for volumetric caustics: photons deposit power along
// their flight segments through participating media (fog/dust); the camera
// march reads it back as in-scattered radiance. Same 5-uint cell layout and
// hash scheme. Header reuse: originCell.w = vol cell size; debugMode 0 = off /
// 1 = deposit enabled; energyScale = σs (scatter coefficient knob);
// frameSeed/tableSize as usual; other fields unused.
layout(set = 0, binding = 20, std430) buffer PhotonVolGridBuf {
    PhotonGridHeader h;
    uint cells[];
} photonVolGrid;

ivec3 photonVolCellCoord(vec3 p) {
    float cs = max(photonVolGrid.h.originCell.w, 1e-4);
    return ivec3(floor((p - photonVolGrid.h.originCell.xyz) / cs));
}
uint photonVolSlot(ivec3 c) {
    uint s = photonGridScramble(uint(c.x) ^ 0x68bc21ebu);
    s = photonGridScramble(s ^ uint(c.y));
    s = photonGridScramble(s ^ uint(c.z));
    return s & (photonVolGrid.h.tableSize - 1u);
}

// ── FAZ 2 DEBUG: photon DIRECTION grid (binding 22) ─────────────────────────
// Parallel to the volume grid (same slot indexing/ownership): per slot 4 ints,
// xyz = Σ(direction·256) fixed-point signed, w = deposit count. Written by
// photon.rgen ONLY when the vol header debugMode == 2 (Photon Directions view
// armed — zero atomic cost otherwise), read by raygen view 5. Average FLOW
// direction per cell; magnitude carries no meaning (readers normalize).
layout(set = 0, binding = 22, std430) buffer PhotonVolDirBuf {
    int d[];
} photonVolDir;

void photonVolDirSplat(vec3 p, vec3 dir) {
    ivec3 c = photonVolCellCoord(p);
    uint slot = photonVolSlot(c);
    // Same ownership rule as the energy splat — never blend directions from
    // an aliased cell.
    if (photonVolGrid.cells[slot * 5u + 4u] != photonGridCellKey(c)) return;
    uint idx = slot * 4u;
    ivec3 q = ivec3(round(dir * 256.0));
    if (q.x != 0) atomicAdd(photonVolDir.d[idx + 0u], q.x);
    if (q.y != 0) atomicAdd(photonVolDir.d[idx + 1u], q.y);
    if (q.z != 0) atomicAdd(photonVolDir.d[idx + 2u], q.z);
    atomicAdd(photonVolDir.d[idx + 3u], 1);
}

// Average photon flow direction at p (normalized; 0 when the cell is empty).
vec3 photonVolDirRead(vec3 p) {
    ivec3 c = photonVolCellCoord(p);
    uint slot = photonVolSlot(c);
    if (photonVolGrid.cells[slot * 5u + 4u] != photonGridCellKey(c)) return vec3(0.0);
    uint idx = slot * 4u;
    vec3 v = vec3(float(photonVolDir.d[idx + 0u]),
                  float(photonVolDir.d[idx + 1u]),
                  float(photonVolDir.d[idx + 2u]));
    return (dot(v, v) > 1e-6) ? normalize(v) : vec3(0.0);
}

uint photonGridRandState(inout uint s) {
    s = s * 747796405u + 2891336453u;
    uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

// Fixed-point energy splat: 1.0 == 256, with STOCHASTIC ROUNDING. Plain uint()
// truncation silently dropped photons whose per-channel power was below 1/256
// (the common case with a large photon budget: power ∝ 1/N) — only absurd
// energy values produced anything, and near the truncation edge the glass
// tint quantised channels differently (R clipped first → green/blue blocks).
// Rounding up with probability fract(v) keeps the EXPECTED deposit exact at
// any power level. Clamped so a single hot photon cannot wrap the counter.
void photonGridSplat(vec3 p, vec3 power, inout uint seed) {
    ivec3 c = photonGridCellCoord(p);
    uint idx = photonGridHashCoord(c) * 5u;
    uint key = photonGridCellKey(c);
    uint owner = atomicCompSwap(photonGrid.cells[idx + 4u], 0u, key);
    if (owner != 0u && owner != key) return;   // slot owned by another cell — drop, don't alias
    vec3 v = clamp(power, vec3(0.0), vec3(4.0e6)) * 256.0;
    uvec3 q = uvec3(v);
    if (float(photonGridRandState(seed)) * (1.0 / 4294967296.0) < fract(v.r)) q.r++;
    if (float(photonGridRandState(seed)) * (1.0 / 4294967296.0) < fract(v.g)) q.g++;
    if (float(photonGridRandState(seed)) * (1.0 / 4294967296.0) < fract(v.b)) q.b++;
    if (q.r > 0u) atomicAdd(photonGrid.cells[idx + 0u], q.r);
    if (q.g > 0u) atomicAdd(photonGrid.cells[idx + 1u], q.g);
    if (q.b > 0u) atomicAdd(photonGrid.cells[idx + 2u], q.b);
    atomicAdd(photonGrid.cells[idx + 3u], 1u);
}

// Volume splat: same stochastic-rounding fixed-point scheme as the surface
// grid (key check via photonGridCellKey — salts differ per FUNCTION, tables
// are separate buffers so sharing the salt constants is fine).
void photonVolSplat(vec3 p, vec3 power, inout uint seed) {
    ivec3 c = photonVolCellCoord(p);
    uint idx = photonVolSlot(c) * 5u;
    uint key = photonGridCellKey(c);
    uint owner = atomicCompSwap(photonVolGrid.cells[idx + 4u], 0u, key);
    if (owner != 0u && owner != key) return;
    vec3 v = clamp(power, vec3(0.0), vec3(4.0e6)) * 256.0;
    uvec3 q = uvec3(v);
    if (float(photonGridRandState(seed)) * (1.0 / 4294967296.0) < fract(v.r)) q.r++;
    if (float(photonGridRandState(seed)) * (1.0 / 4294967296.0) < fract(v.g)) q.g++;
    if (float(photonGridRandState(seed)) * (1.0 / 4294967296.0) < fract(v.b)) q.b++;
    if (q.r > 0u) atomicAdd(photonVolGrid.cells[idx + 0u], q.r);
    if (q.g > 0u) atomicAdd(photonVolGrid.cells[idx + 1u], q.g);
    if (q.b > 0u) atomicAdd(photonVolGrid.cells[idx + 2u], q.b);
    atomicAdd(photonVolGrid.cells[idx + 3u], 1u);
}

// Single-cell volume read, normalized to ENERGY DENSITY (per cell volume).
// Caller divides by the pass count (frameSeed+1).
vec3 photonVolReadCell(vec3 p) {
    ivec3 c = photonVolCellCoord(p);
    uint idx = photonVolSlot(c) * 5u;
    if (photonVolGrid.cells[idx + 4u] != photonGridCellKey(c)) return vec3(0.0);
    float cs = max(photonVolGrid.h.originCell.w, 1e-4);
    return vec3(float(photonVolGrid.cells[idx + 0u]),
                float(photonVolGrid.cells[idx + 1u]),
                float(photonVolGrid.cells[idx + 2u]))
         * ((1.0 / 256.0) / (cs * cs * cs));
}

// Trilinear-filtered volume read (Dilim V2 polish), same density normalization.
// The nearest-cell read showed every occupied cell as a discrete speck in the
// camera march; interpolating the 8 surrounding cells (cell CENTRES, hence the
// -0.5 offset) turns the blocks into smooth gradients. 8 hash lookups/sample.
vec3 photonVolReadTrilinear(vec3 p) {
    float cs = max(photonVolGrid.h.originCell.w, 1e-4);
    vec3 rel = (p - photonVolGrid.h.originCell.xyz) / cs - 0.5;
    ivec3 base = ivec3(floor(rel));
    vec3 f = rel - vec3(base);
    vec3 sum = vec3(0.0);
    for (int i = 0; i < 8; ++i) {
        ivec3 o = ivec3(i & 1, (i >> 1) & 1, (i >> 2) & 1);
        vec3 w3 = mix(vec3(1.0) - f, f, vec3(o));
        float w = w3.x * w3.y * w3.z;
        if (w <= 1e-4) continue;
        ivec3 c = base + o;
        uint idx = photonVolSlot(c) * 5u;
        if (photonVolGrid.cells[idx + 4u] != photonGridCellKey(c)) continue;
        sum += vec3(float(photonVolGrid.cells[idx + 0u]),
                    float(photonVolGrid.cells[idx + 1u]),
                    float(photonVolGrid.cells[idx + 2u])) * w;
    }
    return sum * ((1.0 / 256.0) / (cs * cs * cs));
}

// Raw single-cell energy read (debug viz, mode 1).
vec3 photonGridReadCell(vec3 p) {
    ivec3 c = photonGridCellCoord(p);
    uint idx = photonGridHashCoord(c) * 5u;
    if (photonGrid.cells[idx + 4u] != photonGridCellKey(c)) return vec3(0.0);
    return vec3(float(photonGrid.cells[idx + 0u]),
                float(photonGrid.cells[idx + 1u]),
                float(photonGrid.cells[idx + 2u])) * (1.0 / 256.0);
}

// 3x3x3 density estimation (Dilim 2, mode 2) with a SMOOTH CONE KERNEL:
// each of the 27 neighbor cells contributes its photon power weighted by
// w = 1 - d/R (d = distance from the cell centre to the shading point,
// R = 1.6*cs). This turns the raw per-cell blocks into round, overlapping
// splats — the "pixelated caustic" look at coarse cell sizes came from the
// unweighted box sum. Normalized as a 2D kernel density estimate on the
// receiving surface: ∫(1-d/R) dA over the disk = πR²/3, so brightness stays
// consistent across cell sizes. Returns caustic IRRADIANCE per accumulation
// pass — the caller divides by the pass count (frameSeed+1). Flat-surface
// footprint approximation; normal-aware footprints are Dilim 3 polish.
vec3 photonGridGather(vec3 p) {
    float cs = max(photonGrid.h.originCell.w, 1e-4);
    vec3 rel = (p - photonGrid.h.originCell.xyz) / cs;   // grid-space position
    ivec3 c0 = ivec3(floor(rel));
    float R = 1.6;                                       // kernel radius in CELLS
    vec3 sum = vec3(0.0);
    for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
        ivec3 c = c0 + ivec3(dx, dy, dz);
        float d = length(vec3(c) + 0.5 - rel);           // cell centre distance, in cells
        float w = 1.0 - d / R;
        if (w <= 0.0) continue;
        uint idx = photonGridHashCoord(c) * 5u;
        if (photonGrid.cells[idx + 4u] != photonGridCellKey(c)) continue;
        sum += vec3(float(photonGrid.cells[idx + 0u]),
                    float(photonGrid.cells[idx + 1u]),
                    float(photonGrid.cells[idx + 2u])) * w;
    }
    float Rw = R * cs;                                   // kernel radius, world units
    return sum * ((1.0 / 256.0) * 3.0 / (3.14159265 * Rw * Rw));
}

#endif // PHOTON_GRID_GLSL
