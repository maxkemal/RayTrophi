float lightMarchAcc(
    VkVolumeInstance vol,
    vec3 pos,
    vec3 lightDir,
    float maxDist,
    pnanovdb_buf_t buf,
    pnanovdb_map_handle_t mapH,
    inout pnanovdb_readaccessor_t acc,
    float shadowStrengthOverride)
{
    if (vol.shadow_steps <= 0) return 1.0;
    if (maxDist <= 1e-4) return 1.0;

    vec3 shadowAbsColor = max(vol.absorption_color, vec3(0.0));
    float shadowAbsPeak = max(
        shadowAbsColor.r, max(shadowAbsColor.g, shadowAbsColor.b));
    vec3 shadowAbsWeights = shadowAbsPeak > 1e-5
        ? shadowAbsColor / shadowAbsPeak : vec3(1.0);
    float sigma_t = vol.scatter_coefficient +
        vol.absorption_coefficient *
        dot(shadowAbsWeights, vec3(0.2126, 0.7152, 0.0722));
    if (sigma_t <= EPSILON) return 1.0;

    int reqSteps = clamp(vol.shadow_steps, 1, 64);
    uint shadowMatIndex = vol._reserved[1] > 0.5 ? uint(vol._reserved[1] - 1.0) : 0u;
    bool allowSparseTraversal = buf.address != 0 &&
        (vol._reserved[1] <= 0.5 || matProgramOffset(shadowMatIndex) == MATPROG_NONE);
    // Same Volume-Graph rule as the sparse path: a program may synthesize
    // density where the source field is empty, so the block skip is only valid
    // for unmodified density.
    bool allowDenseBlockSkip = vol.majorant_address != 0 &&
        (vol._reserved[1] <= 0.5 || matProgramOffset(shadowMatIndex) == MATPROG_NONE);
    float denseBlockCutoff = (vol._reserved[0] > 0.0) ? vol._reserved[0] : 0.0;
    // maxDist is the chord from this scatter point to the volume boundary.
    // Integrate the whole chord; half-length marching over-lights the far side
    // of dense gas and erases the rolling self-shadow detail.
    float marchLength = maxDist;
    bool indexPath = vol.volume_type == 2 && buf.address != 0 &&
        vol._ext_reserved[7] < 0.5 && allowSparseTraversal;
    VolumeNanoRay indexRay;
    if (indexPath) {
        indexRay = volumeNanoRay(vol, pos, lightDir, buf, mapH);
    }
    float midpointDensity = indexPath
        ? volumeNanoDensity(vol, indexRay, 0.5 * marchLength, buf, acc)
        : sampleDensityAcc(vol, pos + lightDir * (0.5 * marchLength), buf, mapH, acc);
    float tauHint = max(midpointDensity, 0.0) * sigma_t * marchLength;
    float stepScale = clamp(sqrt(max(tauHint, 0.0)), 0.20, 1.0);
    // Dense gas needs eight buffer loads per trilinear density lookup. Twelve
    // jittered samples over the exact exit chord retain soft self-shadow detail
    // while avoiding the 16-sample worst case on every cached shadow update.
    int shadowSampleCap = (vol.volume_type == 4 && vol.source_type == 5) ? 12 : 16;
    int steps = clamp(
        int(ceil(float(min(reqSteps, shadowSampleCap)) * stepScale)),
        min(3, reqSteps),
        min(reqSteps, shadowSampleCap));
    float stepSize = marchLength / float(max(steps, 1));
    stepSize = max(stepSize, 1e-5);
    float jitter = fract(sin(dot(pos, vec3(12.9898, 78.233, 37.719)) +
                             dot(lightDir, vec3(39.346, 11.135, 83.155))) * 43758.5453);

    float s_trans = 0.0;
    float distanceAlongRay = min((jitter + 0.5) * stepSize, marchLength);
    int densitySamples = 1; // includes midpoint density used for adaptive step count
    int traversalIters = 0;
    int maxTraversalIters = steps * 4 + 8;
    while (distanceAlongRay < marchLength &&
           densitySamples < steps &&
           traversalIters < maxTraversalIters) {
        vec3 samplePos = pos + lightDir * distanceAlongRay;
        // Live dense gas: same block-majorant skip the primary march uses. The
        // shadow chord crosses the domain just like a camera ray does, so an
        // empty region cost a full sample budget here too — and this march runs
        // once per light per (strided) step, so it is the more expensive of the
        // two places to walk through nothing.
        if (vol.volume_type == 4 && vol.source_type == 5 && allowDenseBlockSkip) {
            float denseSkip = denseGasEmptyBlockStep(
                vol, samplePos, lightDir, denseBlockCutoff);
            if (denseSkip > stepSize * 1.01) {
                distanceAlongRay += min(denseSkip, marchLength - distanceAlongRay);
                traversalIters++;
                continue;
            }
        }
        if (allowSparseTraversal) {
            uint shadowSkipKind = 0u;
            float sparseStep = indexPath
                ? nanoEmptyIndexStep(vol, indexRay.indexOrigin +
                    indexRay.indexDirection * distanceAlongRay,
                    indexRay.indexDirection, stepSize, buf, acc, shadowSkipKind)
                : nanoEmptyTileStep(vol, samplePos, lightDir, stepSize,
                    buf, mapH, acc, shadowSkipKind);
            if (sparseStep > stepSize * 1.01) {
                // Keep the original quadrature lattice after an optically
                // empty leaf. Only known-zero samples disappear.
                float advance = indexPath
                    ? max(1.0, floor(sparseStep / stepSize)) * stepSize
                    : sparseStep;
                distanceAlongRay += min(advance, marchLength - distanceAlongRay);
                traversalIters++;
                continue;
            }
        }

        float d = indexPath
            ? volumeNanoDensity(vol, indexRay, distanceAlongRay, buf, acc)
            : sampleDensityAcc(vol, samplePos, buf, mapH, acc);
        s_trans += d * sigma_t * stepSize;
        if (s_trans > 10.0) break;
        distanceAlongRay += stepSize;
        densitySamples++;
        traversalIters++;
    }
    volumeRecordShadowSamples(uint(densitySamples));

    float phys_trans = exp(-s_trans);

    float shadowStrength = (shadowStrengthOverride >= 0.0)
        ? clamp(shadowStrengthOverride, 0.0, 1.0)
        : clamp(vol.shadow_strength * 0.92, 0.0, 1.0);
    return 1.0 - shadowStrength * (1.0 - phys_trans);
}
