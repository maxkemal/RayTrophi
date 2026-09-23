#ifndef RT_SURFACE_SDF_SHADOW_QUERY
#define RT_SURFACE_SDF_SHADOW_QUERY

// Native SurfaceSDF occlusion for RayFusion shadow rays. This reads the same
// 624-byte VkVolumeInstance table and NanoVDB addresses as the surface pass.
layout(set = 0, binding = 6, std430) readonly buffer RfVolumeRawBuffer {
    uint rfVolumeWords[];
};

const uint RF_VOLUME_STRIDE_WORDS = 156u;

uint rfVolWord(uint volumeIndex, uint byteOffset) {
    return rfVolumeWords[volumeIndex * RF_VOLUME_STRIDE_WORDS +
                         (byteOffset >> 2u)];
}

float rfVolFloat(uint volumeIndex, uint byteOffset) {
    return uintBitsToFloat(rfVolWord(volumeIndex, byteOffset));
}

uint64_t rfVolAddress(uint volumeIndex, uint byteOffset) {
    uvec2 pair = uvec2(rfVolWord(volumeIndex, byteOffset),
                       rfVolWord(volumeIndex, byteOffset + 4u));
    return uint64_t(pair.x) | (uint64_t(pair.y) << 32);
}

vec3 rfVolVec3(uint volumeIndex, uint byteOffset) {
    return vec3(rfVolFloat(volumeIndex, byteOffset),
                rfVolFloat(volumeIndex, byteOffset + 4u),
                rfVolFloat(volumeIndex, byteOffset + 8u));
}

vec3 rfVolTransformPoint(uint vi, uint base, vec3 p) {
    return vec3(
        rfVolFloat(vi, base + 0u) * p.x + rfVolFloat(vi, base + 4u) * p.y +
            rfVolFloat(vi, base + 8u) * p.z + rfVolFloat(vi, base + 12u),
        rfVolFloat(vi, base + 16u) * p.x + rfVolFloat(vi, base + 20u) * p.y +
            rfVolFloat(vi, base + 24u) * p.z + rfVolFloat(vi, base + 28u),
        rfVolFloat(vi, base + 32u) * p.x + rfVolFloat(vi, base + 36u) * p.y +
            rfVolFloat(vi, base + 40u) * p.z + rfVolFloat(vi, base + 44u));
}

vec3 rfVolTransformVector(uint vi, uint base, vec3 p) {
    return vec3(
        rfVolFloat(vi, base + 0u) * p.x + rfVolFloat(vi, base + 4u) * p.y +
            rfVolFloat(vi, base + 8u) * p.z,
        rfVolFloat(vi, base + 16u) * p.x + rfVolFloat(vi, base + 20u) * p.y +
            rfVolFloat(vi, base + 24u) * p.z,
        rfVolFloat(vi, base + 32u) * p.x + rfVolFloat(vi, base + 36u) * p.y +
            rfVolFloat(vi, base + 40u) * p.z);
}

bool rfVolInterval(uint vi, vec3 origin, vec3 direction,
                   out float nearT, out float farT) {
    vec3 localOrigin = rfVolTransformPoint(vi, 184u, origin);
    vec3 localDirection = rfVolTransformVector(vi, 184u, direction);
    vec3 safeDirection = vec3(
        abs(localDirection.x) > 1e-8
            ? localDirection.x : (localDirection.x < 0.0 ? -1e-8 : 1e-8),
        abs(localDirection.y) > 1e-8
            ? localDirection.y : (localDirection.y < 0.0 ? -1e-8 : 1e-8),
        abs(localDirection.z) > 1e-8
            ? localDirection.z : (localDirection.z < 0.0 ? -1e-8 : 1e-8));
    vec3 t0 = (rfVolVec3(vi, 48u) - localOrigin) / safeDirection;
    vec3 t1 = (rfVolVec3(vi, 60u) - localOrigin) / safeDirection;
    vec3 lo = min(t0, t1);
    vec3 hi = max(t0, t1);
    nearT = max(max(lo.x, lo.y), lo.z);
    farT = min(min(hi.x, hi.y), hi.z);
    return farT > max(nearT, 0.0);
}

#define PNANOVDB_GLSL
#define PNANOVDB_BUF_CUSTOM
struct pnanovdb_buf_t { uint64_t address; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer RfNanoVDBBlock {
    uint data[];
};
uint pnanovdb_buf_read_uint32(pnanovdb_buf_t buf, uint byteOffset) {
    RfNanoVDBBlock block = RfNanoVDBBlock(buf.address);
    return block.data[byteOffset >> 2u];
}
uvec2 pnanovdb_buf_read_uint64(pnanovdb_buf_t buf, uint byteOffset) {
    RfNanoVDBBlock block = RfNanoVDBBlock(buf.address);
    uint index = byteOffset >> 2u;
    return uvec2(block.data[index], block.data[index + 1u]);
}
void pnanovdb_buf_write_uint32(pnanovdb_buf_t buf, uint byteOffset, uint value) {}
void pnanovdb_buf_write_uint64(pnanovdb_buf_t buf, uint byteOffset, uvec2 value) {}
#include "PNanoVDB.h"

float rfSampleGrid(pnanovdb_buf_t buf, pnanovdb_map_handle_t mapHandle,
                   inout pnanovdb_readaccessor_t accessor, vec3 position) {
    pnanovdb_vec3_t world = pnanovdb_vec3_uniform(0.0);
    world.x = position.x;
    world.y = position.y;
    world.z = position.z;
    pnanovdb_vec3_t indexPosition =
        pnanovdb_map_apply_inverse(buf, mapHandle, world);
    vec3 q = vec3(indexPosition.x, indexPosition.y, indexPosition.z);
    vec3 base = floor(q);
    vec3 fraction = fract(q);
    float density[8];
    for (int corner = 0; corner < 8; ++corner) {
        pnanovdb_coord_t coordinate;
        coordinate.x = int(base.x) + ((corner & 1) != 0 ? 1 : 0);
        coordinate.y = int(base.y) + ((corner & 2) != 0 ? 1 : 0);
        coordinate.z = int(base.z) + ((corner & 4) != 0 ? 1 : 0);
        pnanovdb_address_t address = pnanovdb_readaccessor_get_value_address(
            PNANOVDB_GRID_TYPE_FLOAT, buf, accessor, coordinate);
        density[corner] = pnanovdb_read_float(buf, address);
    }
    return mix(
        mix(mix(density[0], density[1], fraction.x),
            mix(density[2], density[3], fraction.x), fraction.y),
        mix(mix(density[4], density[5], fraction.x),
            mix(density[6], density[7], fraction.x), fraction.y),
        fraction.z);
}

float rfSampleSurfaceSdf(uint vi, pnanovdb_buf_t nanoBuffer,
                         pnanovdb_map_handle_t mapHandle,
                         inout pnanovdb_readaccessor_t accessor,
                         vec3 worldPosition) {
    vec3 local = rfVolTransformPoint(vi, 184u, worldPosition);
    if (any(lessThan(local, rfVolVec3(vi, 48u))) ||
        any(greaterThan(local, rfVolVec3(vi, 60u)))) {
        return 0.0;
    }
    return rfSampleGrid(nanoBuffer, mapHandle, accessor,
                        local - rfVolVec3(vi, 416u));
}

bool rfSurfaceSdfOccludes(vec3 origin, vec3 direction, float rayMin,
                          float rayMax, uint volumeCount,
                          uint materialCount) {
    uint count = min(volumeCount, 16u);
    for (uint vi = 0u; vi < count; ++vi) {
        if (rfVolWord(vi, 172u) == 0u || rfVolWord(vi, 168u) != 2u ||
            int(rfVolWord(vi, 428u)) != 4 ||
            rfVolAddress(vi, 232u) == uint64_t(0)) {
            continue;
        }
        float materialSlot = rfVolFloat(vi, 556u);
        if (materialSlot >= 1.0 &&
            uint(materialSlot - 1.0) < materialCount) {
            Material material = materials[uint(materialSlot - 1.0)];
            if (material.opacity < 0.1 || material.transmission > 0.5) {
                continue;
            }
        }
        float boxNear;
        float boxFar;
        if (!rfVolInterval(vi, origin, direction, boxNear, boxFar)) continue;
        float beginT = max(boxNear, rayMin);
        float endT = min(boxFar, rayMax);
        if (endT <= beginT) continue;

        pnanovdb_buf_t nanoBuffer;
        nanoBuffer.address = rfVolAddress(vi, 232u);
        pnanovdb_grid_handle_t gridHandle;
        gridHandle.address.byte_offset = 0u;
        pnanovdb_tree_handle_t treeHandle =
            pnanovdb_grid_get_tree(nanoBuffer, gridHandle);
        pnanovdb_root_handle_t rootHandle =
            pnanovdb_tree_get_root(nanoBuffer, treeHandle);
        pnanovdb_map_handle_t mapHandle =
            pnanovdb_grid_get_map(nanoBuffer, gridHandle);
        pnanovdb_readaccessor_t accessor;
        pnanovdb_readaccessor_init(accessor, rootHandle);

        const int maxSteps = 192;
        float voxelSize = max(rfVolFloat(vi, 176u), 1e-4);
        float stepSize = max(voxelSize * 0.65,
                             (endT - beginT) / float(maxSteps));
        int steps = min(int(ceil((endT - beginT) / stepSize)) + 1,
                        maxSteps + 1);
        for (int step = 0; step < steps; ++step) {
            float distance = min(beginT + float(step) * stepSize, endT);
            if (rfSampleSurfaceSdf(
                    vi, nanoBuffer, mapHandle, accessor,
                    origin + direction * distance) >= 0.5) {
                return true;
            }
            if (distance >= endT) break;
        }
    }
    return false;
}

#endif
