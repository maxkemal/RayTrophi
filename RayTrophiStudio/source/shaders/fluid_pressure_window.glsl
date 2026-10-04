// A padded lane maps to the full-grid sentinel. Reduction shaders must still
// execute every workgroup barrier; they contribute zero for that sentinel.
int fluidPressureCellIndex() {
#ifdef RT_FLUID_WINDOW
    uint lane = gl_GlobalInvocationID.x;
    uint count = uint(pc.extent_x) * uint(pc.extent_y) * uint(pc.extent_z);
    if (lane >= count) {
        return pc.nx * pc.ny * pc.nz;
    }
    uint i = uint(pc.begin_x) + lane % uint(pc.extent_x);
    uint j = uint(pc.begin_y) + (lane / uint(pc.extent_x)) % uint(pc.extent_y);
    uint k = uint(pc.begin_z) + lane / (uint(pc.extent_x) * uint(pc.extent_y));
    return int(i + uint(pc.nx) * (j + uint(pc.ny) * k));
#else
    return int(gl_GlobalInvocationID.x);
#endif
}
