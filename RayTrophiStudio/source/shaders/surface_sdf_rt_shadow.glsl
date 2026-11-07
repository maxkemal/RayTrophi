#ifndef RT_SURFACE_SDF_RT_SHADOW
#define RT_SURFACE_SDF_RT_SHADOW

#include "surface_sdf_march_policy.glsl"

#ifndef RT_SDF_SHADOW_SAMPLE
#define RT_SDF_SHADOW_SAMPLE ch_sampleNanoVDBAcc
#endif

// Surface membership matches RayFusion. Sample the raw surface proxy: fog
// density remapping, strength and shadow-step settings must not affect it.
bool rtSurfaceSdfOccludes(VkVolumeInstance vol, vec3 origin, vec3 direction,
                         float rayMin, float rayMax) {
    if (vol.volume_type != 2 || vol.vdb_grid_address == uint64_t(0)) return false;
    if (vol.iso_material_index >= 1.0) {
        uint slot = uint(vol.iso_material_index - 1.0);
        if (slot < uint(materials.m.length())) {
            Material material = materials.m[slot];
            if (material.opacity < 0.1 || material.transmission > 0.5) return false;
        }
    }

    vec3 localOrigin;
    vec3 localDirection;
    for (int axis = 0; axis < 3; ++axis) {
        int base = axis * 4;
        vec3 row = vec3(vol.inv_transform[base], vol.inv_transform[base + 1],
                        vol.inv_transform[base + 2]);
        localOrigin[axis] = dot(row, origin) + vol.inv_transform[base + 3];
        localDirection[axis] = dot(row, direction);
    }
    vec3 lo = vec3(vol.aabb_min[0], vol.aabb_min[1], vol.aabb_min[2]);
    vec3 hi = vec3(vol.aabb_max[0], vol.aabb_max[1], vol.aabb_max[2]);
    float beginT = rayMin;
    float endT = rayMax;
    for (int axis = 0; axis < 3; ++axis) {
        if (abs(localDirection[axis]) < 1e-8) {
            if (localOrigin[axis] < lo[axis] || localOrigin[axis] > hi[axis]) {
                return false;
            }
            continue;
        }
        float a = (lo[axis] - localOrigin[axis]) / localDirection[axis];
        float b = (hi[axis] - localOrigin[axis]) / localDirection[axis];
        beginT = max(beginT, min(a, b));
        endT = min(endT, max(a, b));
    }
    if (endT <= beginT) return false;

    pnanovdb_buf_t nanoBuffer;
    nanoBuffer.address = vol.vdb_grid_address;
    pnanovdb_grid_handle_t grid;
    grid.address.byte_offset = 0u;
    pnanovdb_tree_handle_t tree = pnanovdb_grid_get_tree(nanoBuffer, grid);
    pnanovdb_root_handle_t root = pnanovdb_tree_get_root(nanoBuffer, tree);
    pnanovdb_map_handle_t mapHandle = pnanovdb_grid_get_map(nanoBuffer, grid);
    pnanovdb_readaccessor_t accessor;
    pnanovdb_readaccessor_init(accessor, root);
    float voxel = max(vol.voxel_size, 1e-4);
    int steps;
    float stepSize = surfaceSdfMarchStep(endT - beginT, voxel, voxel * 0.45,
                                        192, steps);
    vec3 pivot = vec3(vol.pivot_offset[0], vol.pivot_offset[1], vol.pivot_offset[2]);
    for (int step = 0; step < steps; ++step) {
        float t = min(beginT + float(step) * stepSize, endT);
        vec3 position = localOrigin + localDirection * t - pivot;
        if (RT_SDF_SHADOW_SAMPLE(nanoBuffer, mapHandle, accessor, position) >= 0.5) {
            return true;
        }
        if (t >= endT) break;
    }
    return false;
}

float rtSurfaceSdfShadowTransmittance(vec3 origin, vec3 direction, float maxDist) {
    int count = clamp(int(cam.pad0), 0, 16);
    for (int vi = 0; vi < count; ++vi) {
        VkVolumeInstance vol = volumes.v[vi];
        if (vol.is_active != 0 && vol.source_type == 4 &&
            rtSurfaceSdfOccludes(vol, origin, direction, SHADOW_TMIN, maxDist)) {
            return 0.0;
        }
    }
    return 1.0;
}

#undef RT_SDF_SHADOW_SAMPLE
#endif
