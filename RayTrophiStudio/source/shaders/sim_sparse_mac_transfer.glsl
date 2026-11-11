#include "sim_dispatch.glsl"
#include "sim_sparse_mac_address.glsl"
layout(local_size_x = 256) in;
layout(push_constant) uniform PC {
    int nx; int ny; int nz; int particle_count; int component;
    float origin_x; float origin_y; float origin_z; float voxel_size;
} pc;
layout(set = 0, binding = 0) readonly buffer Positions { float positions[]; };
layout(set = 0, binding = 1) buffer DenseVelocity { float dense_velocity[]; };
layout(set = 0, binding = 2) buffer DenseWeight { float dense_weight[]; };
layout(set = 0, binding = 3) buffer CompactVelocity { float compact_velocity[]; };
layout(set = 0, binding = 4) buffer CompactWeight { float compact_weight[]; };
layout(set = 0, binding = 5) buffer TileMap { uint tile_map[]; };
layout(set = 0, binding = 6) buffer TileList { uint tile_list[]; };
layout(set = 0, binding = 7) buffer FlipBaseline { float flip_baseline[]; };

void main() {
    ivec3 cells = ivec3(pc.nx, pc.ny, pc.nz);
#ifdef RT_MAC_CLEAR
    uvec3 dims = sparseMacTileDims(cells);
    uint count = dims.x * dims.y * dims.z;
    uint lane = simLane256(count);
    if (lane >= count) return;
    tile_map[lane] = 0u;
    if (lane == 0u) tile_list[0] = 0u;
#elif defined(RT_MAC_MARK)
    uint lane = simLane256(uint(pc.particle_count));
    if (lane >= uint(pc.particle_count)) return;
    vec3 p = vec3(positions[lane * 3u], positions[lane * 3u + 1u],
                  positions[lane * 3u + 2u]);
    vec3 g = (p - vec3(pc.origin_x, pc.origin_y, pc.origin_z)) * (1.0 / pc.voxel_size);
    if (any(isnan(g)) || any(isinf(g))) return;
    for (int component = 0; component < 3; ++component) {
        vec3 offset = vec3(0.5);
        offset[component] = 0.0;
        vec3 face_g = g - offset;
        if (any(lessThan(face_g, vec3(-2.0))) ||
            any(greaterThan(face_g, vec3(cells) + vec3(2.0)))) continue;
        ivec3 base = ivec3(floor(face_g - vec3(0.5)));
        ivec3 maximum = cells - ivec3(1);
        maximum[component] += 1;
        // The 3x3x3 face stencil, clipped to the grid. The tile key is
        // monotone in each face coordinate (owner clamp, then /8), so the
        // tiles it touches are exactly the box between the tiles of its two
        // corners: usually one tile, at most 8. It used to CAS all 27 faces
        // per component (81 contended atomics per particle on a handful of
        // keys; 19 ms per call at 1M parcels, 2026-11-11).
        ivec3 lo = max(base, ivec3(0));
        ivec3 hi = min(base + ivec3(2), maximum);
        if (any(greaterThan(lo, hi))) continue;
        ivec3 tile_lo = lo, tile_hi = hi;
        tile_lo[component] = min(tile_lo[component], cells[component] - 1);
        tile_hi[component] = min(tile_hi[component], cells[component] - 1);
        tile_lo /= 8;
        tile_hi /= 8;
        uvec3 dims = sparseMacTileDims(cells);
        for (int tz = tile_lo.z; tz <= tile_hi.z; ++tz)
        for (int ty = tile_lo.y; ty <= tile_hi.y; ++ty)
        for (int tx = tile_lo.x; tx <= tile_hi.x; ++tx) {
            uint key = uint(tx) + dims.x * (uint(ty) + dims.y * uint(tz));
            // Plain read first: a tile only goes 0 -> marked within a pass,
            // so a stale 0 merely falls through to the CAS.
            if (tile_map[key] != 0u) continue;
            if (atomicCompSwap(tile_map[key], 0u, 0xffffffffu) == 0u) {
                uint slot = atomicAdd(tile_list[0], 1u);
                tile_list[slot + 1u] = key;
                atomicExchange(tile_map[key], slot + 1u);
            }
        }
    }
#elif defined(RT_MAC_PUBLISH)
    uvec3 dims = uvec3(cells);
    dims[pc.component] += 1u;
    uint count = dims.x * dims.y * dims.z;
    uint lane = simLane256(count);
    if (lane >= count) return;
    ivec3 face = ivec3(lane % dims.x, (lane / dims.x) % dims.y,
                       lane / (dims.x * dims.y));
    uint slot = tile_map[sparseMacTileKey(face, pc.component, cells)];
    float velocity = 0.0, weight = 0.0;
    if (slot != 0u) {
        uint address = (slot - 1u) * 576u + sparseMacLocal(face, pc.component, cells);
        velocity = compact_velocity[address];
        weight = compact_weight[address];
    }
    // Explicitly clear absent tiles: stale dense publication may be from the
    // previous substep or from another model. Contact consumes physical weights.
    dense_velocity[lane] = velocity;
    dense_weight[lane] = weight;
#else
    uint count = tile_list[0] * 576u;
    uint lane = simLane256(count);
    if (lane >= count) return;
#ifdef RT_MAC_RESET
    compact_velocity[lane] = 0.0;
    compact_weight[lane] = 0.0;
    flip_baseline[lane] = 0.0;
#elif defined(RT_MAC_NORMALIZE)
    float weight = compact_weight[lane];
    compact_velocity[lane] = weight > 1e-8 ? compact_velocity[lane] / weight : 0.0;
#elif defined(RT_MAC_CAPTURE)
    uint key = tile_list[lane / 576u + 1u];
    ivec3 face = sparseMacFace(lane, pc.component, key, cells);
    flip_baseline[lane] = sparseMacOwned(face, pc.component, key, cells)
        ? dense_velocity[sparseMacDenseIndex(face, pc.component, cells)] : 0.0;
#elif defined(RT_MAC_CAPTURE_COMPACT)
    // Compact pages are the canonical field (no dense publication): the FLIP
    // baseline is a page copy. Padding values are 0 in both.
    flip_baseline[lane] = compact_velocity[lane];
#elif defined(RT_MAC_GATHER)
    uint key = tile_list[lane / 576u + 1u];
    ivec3 face = sparseMacFace(lane, pc.component, key, cells);
    compact_velocity[lane] = sparseMacOwned(face, pc.component, key, cells)
        ? dense_velocity[sparseMacDenseIndex(face, pc.component, cells)] : 0.0;
#endif
#endif
}
