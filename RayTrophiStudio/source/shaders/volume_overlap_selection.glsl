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

// ═══════════════════════════════════════════════════════════════════════════
// GAS / FOG OVERLAP SEGMENTS
// ═══════════════════════════════════════════════════════════════════════════
// The gas march integrates ONE volume per closest-hit. Two gas-like volumes
// sharing space (a liquid's fog view inside the gas domain of the same burning
// spill — both boxes start at the domain origin) therefore lost whichever one
// did not win BVH traversal: the winner marched its whole box and the ray
// resumed at ITS exit, jumping over every part of the other volume inside it.
// Seen as "RT draws the gas outside the fog, but not inside it"; RayFusion
// blends each box separately and was correct.
//
// Extinction of coexisting media is additive, so the fix is to integrate the
// overlap as ONE medium, not to march the boxes one after another (sequential
// marching attenuates the second box's light by all of the first, including
// the part behind it). The ray is cut into segments at every entry/exit of a
// gas-like volume; inside a segment the member set is constant:
//   - one member  : ordinary march.
//   - two members : the higher-priority one is the PRIMARY (its emission and
//                   material program run as before), the other is a COMPANION
//                   whose density adds extinction and scattering at each step
//                   and whose own light march multiplies every shadow term
//                   (T_a * T_b is exact for summed media).
// Membership is computed from the full candidate set, not from the traversal
// winner, so the result does not depend on BVH order.
//
// Limits (stated, not hidden): a third simultaneous member is ignored; the
// companion's emission and material program are not evaluated, which is why
// the emissive / programmed volume is always chosen as primary.
bool gasOverlapCandidate(VkVolumeInstance v) {
    return v.is_active != 0 && v.source_type != 4 &&
           v.source_type != 3 && v.volume_type != 3;
}

int gasOverlapPriority(VkVolumeInstance v) {
    return (v.emission_mode >= 1 ? 2 : 0) + (v._reserved[1] > 0.5 ? 1 : 0);
}

bool gasOverlapOutranks(int prioA, uint idxA, int prioB, uint idxB) {
    return prioA > prioB || (prioA == prioB && idxA < idxB);
}

// Returns the companion slot, or -1. Narrows [nearT, farT] to the segment that
// starts at nearT and may switch the primary volume. rawSpan is widened to the
// longest member box chord so the coincident-face trap test at the end of the
// march keeps judging BOX traversal, not the (legitimately short) segment.
int resolveGasOverlapSegment(vec3 origin, vec3 direction, uint count,
                             inout uint index, inout VkVolumeInstance volume,
                             inout float nearT, inout float farT,
                             inout float rawSpan) {
    if (!gasOverlapCandidate(volume)) return -1;
    const float SEG_EPS = 1e-4;
    const float rayMinimum = max(gl_RayTminEXT, 0.001);
    float segStart = nearT;
    float segEnd = farT;
    uint primary = index;
    int primaryPrio = gasOverlapPriority(volume);
    int companion = -1;
    int companionPrio = -1;
    for (uint candidate = 0u; candidate < min(count, 16u); ++candidate) {
        if (candidate == index) continue;
        VkVolumeInstance gas = volumes.v[candidate];
        if (!gasOverlapCandidate(gas)) continue;
        float gasNear, gasFar;
        if (!volumeRayInterval(gas, origin, direction, gasNear, gasFar)) continue;
        gasNear = max(gasNear, rayMinimum);
        if (gasFar <= segStart + SEG_EPS) continue;          // behind the segment
        if (gasNear > segStart + SEG_EPS) {                  // enters later: cut there
            segEnd = min(segEnd, gasNear);
            continue;
        }
        segEnd = min(segEnd, gasFar);                        // member: cut at its exit
        rawSpan = max(rawSpan, gasFar - gasNear);
        int prio = gasOverlapPriority(gas);
        if (gasOverlapOutranks(prio, candidate, primaryPrio, primary)) {
            companion = int(primary);
            companionPrio = primaryPrio;
            primary = candidate;
            primaryPrio = prio;
        } else if (companion < 0 ||
                   gasOverlapOutranks(prio, candidate, companionPrio, uint(companion))) {
            companion = int(candidate);
            companionPrio = prio;
        }
    }
    if (primary != index) {
        index = primary;
        volume = volumes.v[primary];
    }
    farT = max(segEnd, segStart);
    return companion;
}

// Shadow transmittance through the companion medium. Multiplying it onto the
// primary's light march gives the summed medium's transmittance exactly.
float companionLightMarch(bool hasCompanion, VkVolumeInstance comp,
                          vec3 pos, vec3 dir, float limit, float minStep,
                          pnanovdb_buf_t buf, pnanovdb_map_handle_t mapH,
                          inout pnanovdb_readaccessor_t acc,
                          float shadowStrengthOverride) {
    if (!hasCompanion) return 1.0;
    float dist = min(limit, max(minStep, volumeExitDistance(comp, pos, dir)));
    return lightMarchAcc(comp, pos, dir, dist, buf, mapH, acc, shadowStrengthOverride);
}
