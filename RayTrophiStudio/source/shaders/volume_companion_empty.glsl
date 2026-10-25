// Included in the common two-medium camera loop. Both fields must prove the
// entire skipped prefix empty; testing only the selected primary loses vapor
// where Snow is empty (and vice versa).
if (hasCompanion && nanoIndexPath && compNanoIndexPath &&
    vol._uvw_pad[0] > 0.5 && comp._uvw_pad[0] > 0.5 &&
    volumeNanoRemap(vol, 1e-6) <= max(vol._reserved[0], 0.0) &&
    volumeNanoRemap(comp, 1e-6) <= max(comp._reserved[0], 0.0)) {
    vec3 primaryLocal = primaryNanoRay.localOrigin + primaryNanoRay.localDirection * t;
    vec3 companionLocal = companionNanoRay.localOrigin + companionNanoRay.localDirection * t;
    // Do not extrapolate a topology proof across an instance's clipping box.
    bool insideBoth = all(greaterThanEqual(primaryLocal, vol.aabb_min)) &&
        all(lessThanEqual(primaryLocal, vol.aabb_max)) &&
        all(greaterThanEqual(companionLocal, comp.aabb_min)) &&
        all(lessThanEqual(companionLocal, comp.aabb_max));
    if (insideBoth) {
        uint primaryKind = 0u;
        uint companionKind = 0u;
        float primarySpan = nanoEmptyIndexStep(vol,
            primaryNanoRay.indexOrigin + primaryNanoRay.indexDirection * t,
            primaryNanoRay.indexDirection, baseStep, vdbBuf, vdbAcc, primaryKind);
        float companionSpan = nanoEmptyIndexStep(comp,
            companionNanoRay.indexOrigin + companionNanoRay.indexDirection * t,
            companionNanoRay.indexDirection, baseStep, compBuf, compAcc, companionKind);
        float jointSpan = min(min(primarySpan, companionSpan), tFar - t);
        if (jointSpan > baseStep * 1.01) {
            int skippedSegments = max(1, int(floor(jointSpan / baseStep)));
            measuredEmptySegments += uint(skippedSegments);
            if (primaryKind == 2u || companionKind == 2u)
                measuredDensityLeafSegments += uint(skippedSegments);
            else
                measuredTopologySegments += uint(skippedSegments);
            // Keep the original jittered quadrature lattice and step budget.
            step += skippedSegments;
            t = tNear + (float(step) + rayJitter) * baseStep;
            continue;
        }
    }
}
