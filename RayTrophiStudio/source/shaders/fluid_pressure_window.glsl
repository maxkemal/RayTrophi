// A padded lane maps to the full-grid sentinel. Reduction shaders must still
// execute every workgroup barrier; they contribute zero for that sentinel.
int fluidPressureCellIndex() {
#ifdef RT_FLUID_WINDOW
    uint lane = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * gl_NumWorkGroups.x * 256u;
    uint count = uint(pc.extent_x) * uint(pc.extent_y) * uint(pc.extent_z);
    if (lane >= count) {
        return pc.nx * pc.ny * pc.nz;
    }
    uint i = uint(pc.begin_x) + lane % uint(pc.extent_x);
    uint j = uint(pc.begin_y) + (lane / uint(pc.extent_x)) % uint(pc.extent_y);
    uint k = uint(pc.begin_z) + lane / (uint(pc.extent_x) * uint(pc.extent_y));
    return int(i + uint(pc.nx) * (j + uint(pc.ny) * k));
#else
    uint lane = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * gl_NumWorkGroups.x * 256u;
    uint count = uint(pc.nx) * uint(pc.ny) * uint(pc.nz);
    return lane >= count ? int(count) : int(lane);
#endif
}

uint fluidPressureGroupIndex() {
    return gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x;
}

bool fluidPressureReductionGroupValid() {
#ifdef RT_FLUID_WINDOW
    uint count = uint(pc.extent_x) * uint(pc.extent_y) * uint(pc.extent_z);
#else
    uint count = uint(pc.nx) * uint(pc.ny) * uint(pc.nz);
#endif
    return fluidPressureGroupIndex() < (count + 255u) / 256u;
}
