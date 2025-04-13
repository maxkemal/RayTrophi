extern "C" __global__ void __raygen__rg() {
    // Example ray origin and direction
    float3 ray_origin = make_float3(0.0f, 0.0f, 0.0f);
    float3 ray_direction = make_float3(0.0f, 0.0f, -1.0f);

    // Define tmin and tmax
    float tmin = 0.0f;
    float tmax = 1.0e16f;

    // Trace the ray
    optixTrace(
        pipeline,
        ray_origin,
        ray_direction,
        tmin,
        tmax,
        0, // Ray type (customize as needed)
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_NONE,
        0, // SBT offset
        1, // SBT stride
        0, // Miss program index
        0, // Hit program index
        0  // Any hit program index
    );
}
