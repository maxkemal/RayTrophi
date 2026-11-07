// GPU surface reconstruction (docs/dev/GPU_YUZEY_LEVEL_SET.md, Y1).
// Device twin of FluidLevelSet.cpp (buildLevelSet) and FluidMaterialCoordinates.cpp
// (buildMaterialCoordinateGrid). Every kernel shares this 128-byte block; the host
// mirror is SurfaceGpuConstants in FluidSurfaceGpu.cpp and the two must move together.
//
// ★ Parity is the acceptance test (fluid.surface_parity), so the weights, radii,
// clamps and bin rules below are copied from the CPU builder, not re-derived.

layout(push_constant) uniform SurfacePC {
    ivec4 grid;      // output grid nx, ny, nz, particle_count
    ivec4 bins;      // bin nx, ny, nz, bin_count
    vec4  origin;    // grid origin xyz (world), inverse bin width
    vec4  cell;      // output voxel, sim voxel, kernel R^2, kernel R
    vec4  centre;    // smoothing radius^2, lambda, particle radius, surface offset
    vec4  band;      // far value, 0.5/gradient width, displacement limit, unused
    ivec4 reach;     // bin reach, neighbour minimum, smoothing stride, mode
    ivec4 sim;       // sim grid nx, ny, nz (eligibility bounds), unused
} pc;

const uint kSurfaceInvalid = 0xffffffffu;

uint binIndex(ivec3 b) {
    return uint(b.x + pc.bins.x * (b.y + pc.bins.y * b.z));
}

ivec3 binCoordinate(vec3 world) {
    return ivec3(floor((world - pc.origin.xyz) * pc.origin.w));
}
