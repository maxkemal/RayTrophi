#ifndef RT_MAC_LANE
#define RT_MAC_LANE
// One MAC velocity storage contract for dense and compact (sparse tile) kernels
// (docs/dev/MATTER_SPARSE_S1_SIVI_GPU.md). Requires a push block `pc` with
// nx/ny/nz. With RT_SPARSE_MAC the caller declares, before this include,
//   readonly buffer { uint mac_tile_map[]; }   (slot + 1, 0 = background)
//   readonly buffer { uint mac_tile_list[]; }  ([0] = active tiles, then keys)
// and binds the compact 576-value pages in place of the dense face arrays.
#include "sim_sparse_mac_address.glsl"

const uint MAC_ABSENT = 0xffffffffu;

ivec3 macCells() {
    return ivec3(pc.nx, pc.ny, pc.nz);
}

// Storage index of face (i, j, k) of `component`. MAC_ABSENT outside the
// domain, or when its tile holds no page: reads then see the background 0,
// writes are skipped (the dense field is 0 there as well, see the note).
uint macAddress(int component, int i, int j, int k) {
    ivec3 cells = macCells();
    ivec3 face = ivec3(i, j, k);
    ivec3 maximum = cells - ivec3(1);
    maximum[component] += 1;
    if (any(lessThan(face, ivec3(0))) || any(greaterThan(face, maximum))) {
        return MAC_ABSENT;
    }
#ifdef RT_SPARSE_MAC
    uint slot = mac_tile_map[sparseMacTileKey(face, component, cells)];
    if (slot == 0u) {
        return MAC_ABSENT;
    }
    return (slot - 1u) * 576u + sparseMacLocal(face, component, cells);
#else
    return sparseMacDenseIndex(face, component, cells);
#endif
}

// Lanes a face kernel must cover: every dense face of the largest component,
// or every compact page value of the resident tiles.
uint macLaneCount() {
#ifdef RT_SPARSE_MAC
    return mac_tile_list[0] * 576u;
#else
    ivec3 cells = macCells();
    uint x = uint(cells.x + 1) * uint(cells.y) * uint(cells.z);
    uint y = uint(cells.x) * uint(cells.y + 1) * uint(cells.z);
    uint z = uint(cells.x) * uint(cells.y) * uint(cells.z + 1);
    return max(x, max(y, z));
#endif
}

// The face of `component` this lane owns and its storage index. False when
// the lane has none (past that component's dense count, or a compact padding
// value: a clipped tile or the extra face row of an interior tile).
bool macLaneFace(uint lane, int component, out ivec3 face, out uint address) {
    ivec3 cells = macCells();
    face = ivec3(0);
    address = MAC_ABSENT;
#ifdef RT_SPARSE_MAC
    if (lane >= mac_tile_list[0] * 576u) {
        return false;
    }
    uint key = mac_tile_list[lane / 576u + 1u];
    face = sparseMacFace(lane, component, key, cells);
    address = lane;
    return sparseMacOwned(face, component, key, cells);
#else
    uvec3 dims = uvec3(cells);
    dims[component] += 1u;
    if (lane >= dims.x * dims.y * dims.z) {
        return false;
    }
    face = ivec3(lane % dims.x, (lane / dims.x) % dims.y, lane / (dims.x * dims.y));
    address = lane;
    return true;
#endif
}
#endif
