// A SurfaceSDF AABB is a search domain, not an opaque surface. If it wins
// traversal before an overlapping gas AABB, integrate the foreground gas
// through the existing gas branch instead of jumping over it in the SDF walk.
// Requires volumeRayInterval and nearestSurfaceSDFCrossing: the selection and
// the subsequent gas handoff use the same authored isosurface field.
void selectForegroundGas(vec3 origin, vec3 direction, uint count,
                         uint traversalMask, inout uint index,
                         inout VkVolumeInstance volume,
                         inout float nearT, inout float farT,
                         inout bool cameraInside) {
    // The next trace after gas -> SDF explicitly excludes gas. Ignoring that
    // input would select the gas again here, producing a zero-progress loop.
    if (volume.source_type != 4 || (traversalMask & 0x02u) == 0u) return;

    uint selected = index;
    float selectedNear = farT;
    float selectedFar = farT;
    bool selectedInside = cameraInside;
    const float rayMinimum = max(gl_RayTminEXT, 0.001);
    for (uint candidate = 0u; candidate < min(count, 16u); ++candidate) {
        if (candidate == index) continue;
        VkVolumeInstance gas = volumes.v[candidate];
        if (gas.is_active == 0 || gas.source_type == 4 ||
            gas.source_type == 3 || gas.volume_type == 3) continue;
        float gasNear, gasFar;
        if (!volumeRayInterval(gas, origin, direction, gasNear, gasFar)) continue;
        bool insideGas = gasNear <= 0.0;
        gasNear = max(gasNear, rayMinimum);
        // Closest-hit gl_RayTmaxEXT aliases gl_HitTEXT (the winning liquid
        // box entry), not the original ray limit. Clamping to it here would
        // discard precisely the foreground gas segment we need to integrate.
        if (gasFar <= max(gasNear, nearT) || gasNear >= selectedNear) continue;

        // A real liquid boundary before the gas still wins. Search from the
        // original liquid entry, not from gasNear, or it could be skipped.
        float surfaceT = nearestSurfaceSDFCrossing(
            origin, direction, nearT, min(farT, gasFar), candidate, count);
        if (surfaceT > 0.0 && surfaceT <= gasNear + 0.003) continue;

        selected = candidate;
        selectedNear = gasNear;
        selectedFar = gasFar;
        selectedInside = insideGas;
    }
    if (selected == index) return;
    index = selected;
    volume = volumes.v[selected];
    nearT = selectedNear;
    farT = selectedFar;
    cameraInside = selectedInside;
}
