#ifndef SURFACE_DENSE_SAMPLING_GLSL
#define SURFACE_DENSE_SAMPLING_GLSL

// Live GPU surface (docs/dev/GPU_YUZEY_LEVEL_SET.md, Y2): a liquid/snow surface
// whose density proxy was built on the simulation compute device and is read
// here straight through its device address. volume_type 4 + source_type 4.
//
// ★ Placed through the uvw_* fields, not through _ext_reserved like live dense
// gas: a surface already owns every _ext_reserved slot (IOR, roughness, foam,
// noise), and the GPU route builds the proxy on the SAME sim grid as the
// material-coordinate field, so its world origin, cell size and dims are those
// fields by construction (FluidSurfaceGpu.cpp; the host refuses the live route
// for a refined multiplier). World space, like sampleMaterialCoord: the live
// domain volume is axis aligned and its grid origin is published in world units.
//
// Requires VkVolumeInstance to be declared before inclusion.

layout(buffer_reference, std430, buffer_reference_align = 4)
readonly buffer SurfaceDenseProxyGrid {
    float values[];
};

bool isLiveDenseSurface(VkVolumeInstance vol) {
    return vol.volume_type == 4 && vol.source_type == 4 && vol.vdb_grid_address != 0;
}

// Cell-centred trilinear sample of the 0..1 proxy (0.5 = surface). Outside the
// grid it is 0 (air), the same value a NanoVDB background gives.
float sampleDenseSurfaceProxy(VkVolumeInstance vol, vec3 worldPos) {
    if (vol.vdb_grid_address == 0 || !(vol.uvw_voxel > 0.0)) return 0.0;
    ivec3 res = ivec3(int(vol.uvw_dim[0] + 0.5),
                      int(vol.uvw_dim[1] + 0.5),
                      int(vol.uvw_dim[2] + 0.5));
    if (any(lessThanEqual(res, ivec3(0)))) return 0.0;
    vec3 origin = vec3(vol.uvw_origin[0], vol.uvw_origin[1], vol.uvw_origin[2]);
    vec3 gridPos = (worldPos - origin) / vol.uvw_voxel - vec3(0.5);
    if (any(lessThan(gridPos, vec3(-0.5))) ||
        any(greaterThan(gridPos, vec3(res) - vec3(0.5)))) {
        return 0.0;
    }
    ivec3 p0 = clamp(ivec3(floor(gridPos)), ivec3(0), res - ivec3(1));
    ivec3 p1 = min(p0 + ivec3(1), res - ivec3(1));
    vec3 f = clamp(gridPos - vec3(p0), vec3(0.0), vec3(1.0));
    SurfaceDenseProxyGrid g = SurfaceDenseProxyGrid(vol.vdb_grid_address);
    int xy = res.x * res.y;
    float c000 = g.values[p0.x + p0.y * res.x + p0.z * xy];
    float c100 = g.values[p1.x + p0.y * res.x + p0.z * xy];
    float c010 = g.values[p0.x + p1.y * res.x + p0.z * xy];
    float c110 = g.values[p1.x + p1.y * res.x + p0.z * xy];
    float c001 = g.values[p0.x + p0.y * res.x + p1.z * xy];
    float c101 = g.values[p1.x + p0.y * res.x + p1.z * xy];
    float c011 = g.values[p0.x + p1.y * res.x + p1.z * xy];
    float c111 = g.values[p1.x + p1.y * res.x + p1.z * xy];
    return mix(mix(mix(c000, c100, f.x), mix(c010, c110, f.x), f.y),
               mix(mix(c001, c101, f.x), mix(c011, c111, f.x), f.y), f.z);
}

#endif
