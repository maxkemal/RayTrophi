const uint RW_SOLID_PROBE = 0xC17D5EEDu;   // the volume solid-probe sentinel (shadow_anyhit)

// Per-walk trace tallies for volumeRecordWalk (reset at each walk's start).
uint rwProbeTraces = 0u;
uint rwShadowTraces = 0u;
uint rwShadowSkipped = 0u;

// Distance to the first solid along a ray, or -1. Same mask as the march's
// solid probe: everything but the two volume bits.
float rwSolidDistance(vec3 origin, vec3 dir, float tMax) {
    shadowPayload = vec4(0.0, 0.0, 0.0, uintBitsToFloat(RW_SOLID_PROBE));
    traceRayEXT(topLevelAS,
                gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT |
                gl_RayFlagsNoOpaqueEXT,
                0xF5, 0, 1, 1, origin, 1e-4, dir, tMax, 1);
    return shadowPayload.w < 0.5 ? shadowPayload.x : -1.0;
}

// ── Index-space flights (NanoVDB) ──────────────────────────────────────────
// Measured 2026-10-06 (walk_* counters, snow scene, 16 spp): cutting the
// solid-probe traces by 64% and the shadow samples by 56% left the render time
// unchanged, while a coarser flight step cut it by 26%. The cost per flight
// STEP was the uncounted work around each sample: the empty-tile query, the
// leaf-majorant query and the trilinear sample each redid world -> local ->
// NanoVDB index (inv_transform + map inverse), three times a step. A flight is
// a straight line and that map is affine, so the index position is o + d*t:
// computed once per flight, every step is one tree query and one trilinear
// fetch. Same skips, same majorant rule, same density transform as
// sampleDensityAcc; dense gas and volumes with material density noise (which
// needs the local position) keep the generic world-space loop below.

// Index-space ray: idx(t) = o + d * t, t in world units along the world ray.
struct RwIdxRay { vec3 o; vec3 d; };

RwIdxRay rwIndexRay(VkVolumeInstance vol, vec3 worldPos, vec3 worldDir,
                    pnanovdb_buf_t buf, pnanovdb_map_handle_t mapH) {
    vec3 lp;
    lp.x = vol.inv_transform[0]*worldPos.x + vol.inv_transform[1]*worldPos.y
         + vol.inv_transform[2]*worldPos.z + vol.inv_transform[3] - vol.pivot_offset[0];
    lp.y = vol.inv_transform[4]*worldPos.x + vol.inv_transform[5]*worldPos.y
         + vol.inv_transform[6]*worldPos.z + vol.inv_transform[7] - vol.pivot_offset[1];
    lp.z = vol.inv_transform[8]*worldPos.x + vol.inv_transform[9]*worldPos.y
         + vol.inv_transform[10]*worldPos.z + vol.inv_transform[11] - vol.pivot_offset[2];
    vec3 wp1 = worldPos + worldDir;
    vec3 lp1;
    lp1.x = vol.inv_transform[0]*wp1.x + vol.inv_transform[1]*wp1.y
          + vol.inv_transform[2]*wp1.z + vol.inv_transform[3] - vol.pivot_offset[0];
    lp1.y = vol.inv_transform[4]*wp1.x + vol.inv_transform[5]*wp1.y
          + vol.inv_transform[6]*wp1.z + vol.inv_transform[7] - vol.pivot_offset[1];
    lp1.z = vol.inv_transform[8]*wp1.x + vol.inv_transform[9]*wp1.y
          + vol.inv_transform[10]*wp1.z + vol.inv_transform[11] - vol.pivot_offset[2];
    pnanovdb_vec3_t a = pnanovdb_vec3_uniform(0.0);
    a.x = lp.x; a.y = lp.y; a.z = lp.z;
    pnanovdb_vec3_t b = pnanovdb_vec3_uniform(0.0);
    b.x = lp1.x; b.y = lp1.y; b.z = lp1.z;
    pnanovdb_vec3_t ia = pnanovdb_map_apply_inverse(buf, mapH, a);
    pnanovdb_vec3_t ib = pnanovdb_map_apply_inverse(buf, mapH, b);
    RwIdxRay r;
    r.o = vec3(ia.x, ia.y, ia.z);
    r.d = vec3(ib.x - ia.x, ib.y - ia.y, ib.z - ia.z);   // index units per world unit
    return r;
}

// sampleDensityAccMode for volume_type 2 without material noise, from an index
// position: trilinear over voxels floor(idx) .. floor(idx)+1 (same footprint as
// sampleNanoVDBFloatTrilinearAcc), then remap, cutoff and its feather, then the
// multiplier. Must stay identical to that function's optical-control tail.
float rwIdxDensity(VkVolumeInstance vol, pnanovdb_buf_t buf,
                   inout pnanovdb_readaccessor_t acc, vec3 idx, float remapScale) {
    vec3 p0 = floor(idx);
    vec3 f = idx - p0;
    float c[8];
    for (int i = 0; i < 8; ++i) {
        pnanovdb_coord_t ijk;
        ijk.x = int(p0.x) + ((i & 1) != 0 ? 1 : 0);
        ijk.y = int(p0.y) + ((i & 2) != 0 ? 1 : 0);
        ijk.z = int(p0.z) + ((i & 4) != 0 ? 1 : 0);
        c[i] = pnanovdb_read_float(buf, pnanovdb_readaccessor_get_value_address(
            PNANOVDB_GRID_TYPE_FLOAT, buf, acc, ijk));
    }
    float d = mix(mix(mix(c[0], c[1], f.x), mix(c[2], c[3], f.x), f.y),
                  mix(mix(c[4], c[5], f.x), mix(c[6], c[7], f.x), f.y), f.z);
    if (isnan(d) || isinf(d)) return 0.0;
    float r = volumeDensityCurve(max((d - vol.density_remap_low) * remapScale, 0.0), vol.density_curve);
    float cutoff = (vol._reserved[0] > 0.0) ? vol._reserved[0] : 0.0;
    if (r <= cutoff) return 0.0;
    float fade = cutoff > 0.0 ? smoothstep(cutoff, cutoff * 2.0, r) : 1.0;
    return r * vol.density_multiplier * fade;
}

// One tree query classifies the region at idx along idxDir:
//   0 = take a regular step (active tile, or a leaf's last voxel slab whose
//       trilinear footprint reads the next leaf)
//   1 = inactive tile; span = world distance to its exit
//   2 = inside a leaf's safe box [L0, L0+7]; span = distance to its exit and
//       leafMax its max (createNanoGrid's default statistics)
int rwIdxRegion(pnanovdb_buf_t buf, inout pnanovdb_readaccessor_t acc, vec3 idx,
                vec3 idxDir, out float span, out float leafMax) {
    span = 0.0;
    leafMax = 0.0;
    pnanovdb_coord_t ijk;
    ijk.x = int(floor(idx.x)); ijk.y = int(floor(idx.y)); ijk.z = int(floor(idx.z));
    uint dim = pnanovdb_readaccessor_get_dim(PNANOVDB_GRID_TYPE_FLOAT, buf, acc, ijk);
    vec3 boxMin, boxMax;
    if (dim > 1u) {
        if (pnanovdb_readaccessor_is_active(PNANOVDB_GRID_TYPE_FLOAT, buf, acc, ijk)) return 0;
        boxMin = floor(idx / float(dim)) * float(dim);
        boxMax = boxMin + vec3(float(dim));
    } else {
        boxMin = floor(idx / 8.0) * 8.0;
        boxMax = boxMin + vec3(7.0);
        if (any(greaterThan(idx, boxMax))) return 0;
    }
    vec3 boundary = mix(boxMin, boxMax, greaterThan(idxDir, vec3(0.0)));
    vec3 axisT = vec3(1e30);
    if (abs(idxDir.x) > 1e-8) axisT.x = (boundary.x - idx.x) / idxDir.x;
    if (abs(idxDir.y) > 1e-8) axisT.y = (boundary.y - idx.y) / idxDir.y;
    if (abs(idxDir.z) > 1e-8) axisT.z = (boundary.z - idx.z) / idxDir.z;
    span = min(axisT.x, min(axisT.y, axisT.z));
    if (dim > 1u) return 1;
    if (!(span > 1e-5)) return 0;
    leafMax = pnanovdb_read_float(buf, pnanovdb_leaf_get_max_address(
        PNANOVDB_GRID_TYPE_FLOAT, buf, acc.leaf));
    return 2;
}

bool rwIdxPath(VkVolumeInstance vol, pnanovdb_buf_t buf) {
    return buf.address != 0 && vol.volume_type == 2 && vol._ext_reserved[7] < 0.5;
}

// Free flight in index space: regular tracking with leaf-majorant delta
// tracking in sparse leaves. Mixing the two is exact: delta segments sample
// their own exponential, the regular segments share one exponential target
// over their summed optical depth, and the survival of the whole flight is the
// product of both.
bool rwFreeFlightIdx(VkVolumeInstance vol, vec3 o, vec3 d, float tMax, float sigmaPerDensity,
                     float stepLen, pnanovdb_buf_t buf, pnanovdb_map_handle_t mapH,
                     inout pnanovdb_readaccessor_t acc, inout uint densitySamples,
                     out float tHit) {
    tHit = tMax;
    RwIdxRay ray = rwIndexRay(vol, o, d, buf, mapH);
    float remapScale = 1.0 / max(vol.density_remap_high - vol.density_remap_low, EPSILON);
    float target = -log(max(1.0 - rnd(payload.seed), 1e-12));
    float tau = 0.0;
    float t = 0.0;
    for (int it = 0; it < 4096 && t < tMax; ++it) {
        float span, leafMax;
        int region = rwIdxRegion(buf, acc, ray.o + ray.d * t, ray.d, span, leafMax);
        if (region == 1 && span > stepLen * 1.01) {
            // Land a voxel before the tile's exit so trilinear sees density
            // across the active boundary (nanoEmptyTileStep's rule).
            t += max(stepLen, span - max(vol.voxel_size, stepLen));
            continue;
        }
        if (region == 2) {
            float sigmaBar = volumeDensityCurve(max((leafMax - vol.density_remap_low) * remapScale, 0.0), vol.density_curve) *
                             vol.density_multiplier * sigmaPerDensity * 1.0001;
            // ★ Delta tracking costs ~sigmaBar per unit length, regular steps
            // 1/stepLen. In a DENSE leaf the max sits far above the typical
            // voxel (measured: max density 8.8 over a ~1 body), every real
            // collision pays ~max/typical null samples. Only sparse leaves,
            // where sigmaBar * stepLen < 1, are cheaper this way.
            if (sigmaBar * stepLen < 1.0) {
                float segEnd = min(t + span, tMax);
                bool crossed = true;
                if (sigmaBar > 0.0) {
                    crossed = false;
                    for (int k = 0; k < 64; ++k) {
                        float tNext = t - log(max(1.0 - rnd(payload.seed), 1e-12)) / sigmaBar;
                        if (tNext >= segEnd) { crossed = true; break; }
                        t = tNext;
                        float s = rwIdxDensity(vol, buf, acc, ray.o + ray.d * t, remapScale) *
                                  sigmaPerDensity;
                        densitySamples++;
                        if (rnd(payload.seed) * sigmaBar < s) { tHit = t; return true; }
                    }
                }
                if (crossed) {
                    if (segEnd >= tMax) break;
                    t = min(segEnd + 1e-4 * stepLen, tMax);
                    continue;
                }
                // ★★ Budget spent inside the leaf: CONTINUE FROM t, never jump
                // to segEnd. An earlier version jumped, and with spiky 4x fog
                // (sigmaBar ~1e4/m) every leaf interior was skipped: snow
                // survived only in each leaf's last voxel slab, a honeycomb.
                // The regular step below finishes the leaf without bias.
            }
        }
        float dt = min(stepLen, tMax - t);
        float sigma = rwIdxDensity(vol, buf, acc, ray.o + ray.d * (t + 0.5 * dt), remapScale) *
                      sigmaPerDensity;
        densitySamples++;
        float dTau = sigma * dt;
        if (sigma > 0.0 && tau + dTau >= target) {
            tHit = t + (target - tau) / sigma;
            return true;
        }
        tau += dTau;
        t += dt;
    }
    return false;
}

// Regular-tracking free flight from `o` along `d` over [0, tMax). True with the
// collision distance when a real collision happens before tMax. NanoVDB fog
// goes through rwFreeFlightIdx; this world-space loop serves dense gas and
// volumes whose material noise needs the local position.
bool rwFreeFlight(VkVolumeInstance vol, vec3 o, vec3 d, float tMax, float sigmaPerDensity,
                  float stepLen, pnanovdb_buf_t buf, pnanovdb_map_handle_t mapH,
                  inout pnanovdb_readaccessor_t acc, inout uint densitySamples,
                  out float tHit) {
    if (rwIdxPath(vol, buf))
        return rwFreeFlightIdx(vol, o, d, tMax, sigmaPerDensity, stepLen, buf, mapH, acc,
                               densitySamples, tHit);
    tHit = tMax;
    float target = -log(max(1.0 - rnd(payload.seed), 1e-12));
    float tau = 0.0;
    float t = 0.0;
    bool denseSkip = vol.volume_type == 4 && vol.source_type == 5 && vol.majorant_address != 0;
    float cutoff = (vol._reserved[0] > 0.0) ? vol._reserved[0] : 0.0;
    for (int it = 0; it < 4096 && t < tMax; ++it) {
        vec3 p = o + d * t;
        if (denseSkip) {
            float skip = denseGasEmptyBlockStep(vol, p, d, cutoff);
            if (skip > stepLen * 1.01) { t += skip; continue; }
        }
        if (buf.address != 0) {
            uint kind = 0u;
            float skip = nanoEmptyTileStep(vol, p, d, stepLen, buf, mapH, acc, kind);
            if (skip > stepLen * 1.01) { t += skip; continue; }
        }
        float dt = min(stepLen, tMax - t);
        float sigma = sampleDensityAcc(vol, o + d * (t + 0.5 * dt), buf, mapH, acc) *
                      sigmaPerDensity;
        densitySamples++;
        float dTau = sigma * dt;
        if (sigma > 0.0 && tau + dTau >= target) {
            tHit = t + (target - tau) / sigma;
            return true;
        }
        tau += dTau;
        t += dt;
    }
    return false;
}

// Transmittance along a shadow chord with an explicit extinction per unit
// density (lightMarchAcc derives its own from the shader coefficients, which
// is the wrong medium once the walk has switched to similarity). shadow_steps
// jittered segments over the chord (min 2), the same empty-space skips as the
// flights; the jittered optical-depth estimate is unbiased, its exp() slightly
// biased bright, like the march's.
//
// ★ Measured 2026-10-05 on the snow scene: the walk's chords were 347M shadow
// samples at 16 spp because every event marched a fixed 16 points and ignored
// the shader's shadow_steps. The count is now the author's knob, and a chord
// stops once the light is below 0.25% (tau > 6). NanoVDB fog marches the
// chord in index space like the flights.
float rwTransmittance(VkVolumeInstance vol, vec3 p, vec3 d, float maxDist,
                      float sigmaPerDensity, float stepLen, pnanovdb_buf_t buf,
                      pnanovdb_map_handle_t mapH, inout pnanovdb_readaccessor_t acc,
                      inout uint shadowSamples) {
    if (maxDist <= 1e-5 || sigmaPerDensity <= 0.0) return 1.0;
    // shadow_steps 0 means "no self-shadow" to the march; a walk without it
    // would light the inside of the body, so the walk keeps at least 2.
    float samples = float(clamp(vol.shadow_steps, 2, 64));
    float dt = max(maxDist / samples, stepLen);
    float tau = 0.0;
    float t = rnd(payload.seed) * dt;
    if (rwIdxPath(vol, buf)) {
        RwIdxRay ray = rwIndexRay(vol, p, d, buf, mapH);
        float remapScale = 1.0 / max(vol.density_remap_high - vol.density_remap_low, EPSILON);
        for (int it = 0; it < 64 && t < maxDist; ++it) {
            vec3 idx = ray.o + ray.d * t;
            float span, leafMax;
            if (rwIdxRegion(buf, acc, idx, ray.d, span, leafMax) == 1 && span > dt * 1.01) {
                t += max(dt, span - max(vol.voxel_size, dt));
                continue;
            }
            tau += rwIdxDensity(vol, buf, acc, idx, remapScale) * sigmaPerDensity * dt;
            shadowSamples++;
            if (tau > 6.0) return 0.0;
            t += dt;
        }
        return exp(-tau);
    }
    bool denseSkip = vol.volume_type == 4 && vol.source_type == 5 && vol.majorant_address != 0;
    float cutoff = (vol._reserved[0] > 0.0) ? vol._reserved[0] : 0.0;
    for (int it = 0; it < 64 && t < maxDist; ++it) {
        vec3 q = p + d * t;
        if (denseSkip) {
            float skip = denseGasEmptyBlockStep(vol, q, d, cutoff);
            if (skip > dt * 1.01) { t += skip; continue; }
        }
        if (buf.address != 0) {
            uint kind = 0u;
            float skip = nanoEmptyTileStep(vol, q, d, dt, buf, mapH, acc, kind);
            if (skip > dt * 1.01) { t += skip; continue; }
        }
        tau += sampleDensityAcc(vol, q, buf, mapH, acc) * sigmaPerDensity * dt;
        shadowSamples++;
        if (tau > 6.0) return 0.0;
        t += dt;
    }
    return exp(-tau);
}

// Next-event estimate at a scattering point: one scene light (uniform pick,
// weight N) through the volume's own transmittance AND scene geometry. `w` is
// the direction the walk arrived in.
//
// ★ No Nishita-sun term, on purpose. The surface shader disabled its own
// (closesthit.rchit: "direct sun contribution must come from scene Directional
// lights only"); the sun disk is reached by escaping rays through the miss
// shader. Adding it here counted the sun TWICE on the lit side — measured as
// an over-bright, over-scattered sunlit snow on 2026-10-05.
//
// The geometry shadow ray is traced only after the in-volume transmittance
// says light can still arrive: most events of a dense walk are deep inside
// the medium, where that transmittance is ~0 and the trace would buy nothing.
//
// `reduced`: the walk has switched to the similarity medium (see
// volumeRandomWalk), so the phase is isotropic and the transmittance toward
// the light uses the reduced extinction — the same medium the flights see.
vec3 rwDirectLight(VkVolumeInstance vol, vec3 p, vec3 w, bool reduced,
                   float sigmaPerDensity, float stepLen, pnanovdb_buf_t buf,
                   pnanovdb_map_handle_t mapH, inout pnanovdb_readaccessor_t acc,
                   inout uint shadowSamples) {
    vec3 L = vec3(0.0);
    if (cam.lightCount > 0u) {
        int n = int(cam.lightCount);
        int li = clamp(int(floor(rnd(payload.seed) * float(n))), 0, n - 1);
        LightData light = lights.l[li];
        int lightType = int(light.position.w + 0.5);
        float ru = 0.5, rv = 0.5;
        if (lightType == 2) { ru = rnd(payload.seed); rv = rnd(payload.seed); }
        vec3 wi; float dist; float att; bool invSq;
        if (volSampleLight(light, p, ru, rv, wi, dist, att, invSq)) {
            if (invSq) att /= max(dist * dist, 1e-4);
            float phase = reduced
                ? 1.0 / (4.0 * PI)
                : dualLobeHG(dot(w, wi), vol.scatter_anisotropy,
                             vol.scatter_anisotropy_back, vol.scatter_lobe_mix);
            if (att * phase > 0.0) {
                float tr = rwTransmittance(vol, p, wi, min(dist, volumeExitDistance(vol, p, wi)),
                                           sigmaPerDensity, stepLen, buf, mapH, acc,
                                           shadowSamples);
                if (tr > 1e-3) {
                    rwShadowTraces++;
                    if (rwSolidDistance(p, wi, dist) < 0.0)
                        L += float(n) * light.color.rgb * light.color.a * att * phase * tr;
                } else {
                    rwShadowSkipped++;
                }
            }
        }
    }
    return L;
}

// Runs the walk and fills the payload. The caller has already clamped
// [tNear, tFar] to the first solid (solidAhead) and returns right after.
void volumeRandomWalk(VkVolumeInstance vol, vec3 rayOrigin, vec3 rayDir,
                      float tNear, float tFar, bool solidAhead, float selectedSpan,
                      pnanovdb_buf_t buf, pnanovdb_map_handle_t mapH,
                      inout pnanovdb_readaccessor_t acc) {
    float sigmaPerDensity = max(vol.scatter_coefficient, 0.0) +
                            max(vol.absorption_coefficient, 0.0);
    vec3 albedo = sigmaPerDensity > EPSILON
        ? clamp(vol.scatter_color, vec3(0.0), vec3(1.0)) *
          (max(vol.scatter_coefficient, 0.0) / sigmaPerDensity)
        : vec3(0.0);
    // The march's own step (Step multiplier x voxel, VolumeShader::quality),
    // not a hard-coded half voxel. Flights and shadow chords cost ~distance /
    // step, and at 4x fog resolution half a FOG voxel made the walk 2x slower
    // than at 1x for the same scene (measured 15.3 s vs 7.6 s, 2026-10-05) —
    // the multiplier is the user's quality/speed knob for both paths. Never
    // finer than a quarter voxel, or a tiny multiplier would stall the walk.
    float stepLen = max(vol.step_size, max(vol.voxel_size * 0.25, 1e-4));
    int maxEvents = clamp(int(vol.random_walk_max_events + 0.5), 8, 512);
    float exitEpsilon = max(0.002, vol.voxel_size * 0.25);
    uint densitySamples = 0u;
    bool primarySegment = (payload.primaryMeta & PL_PRIMARY_DONE) == 0u;

    // ── First flight: along the incoming ray, inside the clamped interval ──
    vec3 entry = rayOrigin + rayDir * tNear;
    float tHit;
    bool collided = sigmaPerDensity > EPSILON &&
        rwFreeFlight(vol, entry, rayDir, max(tFar - tNear, 0.0), sigmaPerDensity,
                     stepLen, buf, mapH, acc, densitySamples, tHit);
    if (!collided) {
        // Straight through: identical continuation to the march's no-scatter
        // exit (and its solid handoff), with transmittance 1 by construction.
        payload.radiance = vec3(0.0);
        payload.scatterDir = rayDir;
        payload.scattered = true;
        if (solidAhead) {
            payload.scatterOrigin = rayOrigin;
            payload.skipGasVolumes = true;
            volumeRecordGasHandoff();
        } else {
            payload.scatterOrigin = rayOrigin + rayDir * (tFar + exitEpsilon);
            payload.bounceType = BOUNCE_TRANSPARENT;
            payload.skipGasVolumes = selectedSpan < max(0.01, vol.voxel_size * 0.5);
        }
        volumeRecordRay(densitySamples, 0u, 0u, 0u, VOLUME_MARCH_COMPLETED);
        return;
    }

    if (primarySegment) {
        // Denoiser AOVs: the first real collision is this pixel's surface.
        payload.primaryARG  = packHalf2x16(vol.scatter_color.rg);
        payload.primaryABT  = packHalf2x16(vec2(vol.scatter_color.b, 0.0));
        payload.primaryNrm  = floatBitsToUint(tNear + tHit);
        payload.primaryMeta = (payload.primaryMeta & PL_DISP_MASK)
                            | PL_PRIMARY_DONE | PL_PRIMARY_VOLUME
                            | PL_PRIMARY_VOLUME_DEPTH | PL_MATID_MASK;
    }

    // ── Similarity switch ──────────────────────────────────────────────────
    // A forward-peaked phase (snow g ~ 0.8..0.99) barely turns the path at
    // each event, so the walk spends ~1/(1-g) events per real change of
    // direction. Deep in the medium only the DIFFUSED result matters, and the
    // similarity relation gives it exactly: sigma_s' = sigma_s (1 - g) with an
    // isotropic phase. The first `exact` events (and always the camera's own
    // flight, so the silhouette never moves) keep the true phase and sigma;
    // after that the walk continues in the reduced medium, flights AND shadow
    // chords alike. g is the mean cosine of the dual lobe. Only applied for a
    // forward-scattering medium; exact >= maxEvents keeps the walk exact.
    float gMean = mix(vol.scatter_anisotropy_back, vol.scatter_anisotropy, vol.scatter_lobe_mix);
    int exactEvents = clamp(int(vol.random_walk_exact_events + 0.5), 1, 512);
    bool canReduce = gMean > 0.05;
    float sigmaReduced = max(vol.scatter_coefficient, 0.0) * (1.0 - gMean) +
                         max(vol.absorption_coefficient, 0.0);
    vec3 albedoReduced = sigmaReduced > EPSILON
        ? clamp(vol.scatter_color, vec3(0.0), vec3(1.0)) *
          (max(vol.scatter_coefficient, 0.0) * (1.0 - gMean) / sigmaReduced)
        : vec3(0.0);
    bool reduced = false;
    uint shadowSamples = 0u;
    uint events = 0u;
    rwProbeTraces = 0u;
    rwShadowTraces = 0u;
    rwShadowSkipped = 0u;

    vec3 p = entry + rayDir * tHit;
    vec3 w = rayDir;
    // The camera flight was already clamped to the first solid, so its
    // collision point is a clean start for the deferred probe.
    vec3 lastProbe = p;
    float probeEps = max(2.0 * vol.voxel_size, 0.005);
    vec3 T = vec3(1.0);
    vec3 radiance = vec3(0.0);
    for (int ev = 0; ev < maxEvents; ++ev) {
        // Real collision at p: it scatters with weight albedo, sees the lights,
        // and leaves in a phase-sampled direction (phase/pdf = 1).
        float sigmaNow = reduced ? sigmaReduced : sigmaPerDensity;
        events++;
        T *= reduced ? albedoReduced : albedo;
        radiance += T * rwDirectLight(vol, p, w, reduced, sigmaNow, stepLen, buf, mapH, acc,
                                      shadowSamples);
        if (ev >= 4) {
            float q = min(max(T.r, max(T.g, T.b)), 0.98);
            if (rnd(payload.seed) >= q) { T = vec3(0.0); break; }
            T /= q;
        }
        if (reduced) {
            w = sampleHG(w, 0.0, payload.seed);
        } else {
            float g = (rnd(payload.seed) < vol.scatter_lobe_mix)
                      ? vol.scatter_anisotropy : vol.scatter_anisotropy_back;
            w = sampleHG(w, g, payload.seed);
        }
        // Switch BEFORE the next flight, so that flight already sees the
        // medium its endpoint's event will be weighted in.
        if (!reduced && canReduce && ev + 1 >= exactEvents) {
            reduced = true;
            sigmaNow = sigmaReduced;
        }

        float exitDist = volumeExitDistance(vol, p, w);
        float t;
        bool hit = sigmaNow > EPSILON &&
                   rwFreeFlight(vol, p, w, exitDist, sigmaNow, stepLen, buf, mapH,
                                acc, densitySamples, t);
        if (!hit) t = exitDist;

        // ── Deferred solid probe ───────────────────────────────────────────
        // Measured 2026-10-06 (walk_* counters, snow scene, 16 spp): one probe
        // trace per event was 49M traces at ~70 ns each, ~65% of the render,
        // while the flights it guarded were ~4 mm. A trace costs about as much
        // as 50 density samples. So probe the STRAIGHT SEGMENT from the last
        // probed point only once the walk has moved probeEps away: a path that
        // crossed a surface (ground plane, collider) and is still past it
        // crosses that segment too. The price: a path may dip up to probeEps
        // into a solid and come back unseen; its NEE there is occluded by the
        // solid itself, so it adds no light. The exit flight is always probed.
        bool intoSolid = false;
        vec3 solidFrom = p;
        vec3 solidDir = w;
        if (hit) {
            vec3 next = p + w * t;
            vec3 seg = next - lastProbe;
            float segLen = length(seg);
            if (segLen > probeEps) {
                vec3 segDir = seg / segLen;
                rwProbeTraces++;
                float sd = rwSolidDistance(lastProbe, segDir, segLen + 0.012);
                if (sd >= 0.0 && sd - 0.01 < segLen) {
                    intoSolid = true; solidFrom = lastProbe; solidDir = segDir;
                } else {
                    lastProbe = next;
                }
            }
            if (!intoSolid) {
                p = next;
                continue;
            }
        } else {
            // Leaving the box: the unprobed stretch first, then the exit flight.
            vec3 seg = p - lastProbe;
            float segLen = length(seg);
            if (segLen > 1e-5) {
                vec3 segDir = seg / segLen;
                rwProbeTraces++;
                float sd = rwSolidDistance(lastProbe, segDir, segLen + 0.012);
                if (sd >= 0.0 && sd - 0.01 < segLen) {
                    intoSolid = true; solidFrom = lastProbe; solidDir = segDir;
                }
            }
            if (!intoSolid) {
                rwProbeTraces++;
                float sd = rwSolidDistance(p, w, exitDist + 0.012);
                if (sd >= 0.0 && sd - 0.01 < exitDist) {
                    intoSolid = true; solidFrom = p; solidDir = w;
                }
            }
        }
        // Escaped: into a solid inside the box, or out of the box.
        payload.radiance = radiance;
        payload.attenuation *= T;
        payload.scatterDir = intoSolid ? solidDir : w;
        payload.scattered = true;
        payload.bounceType = BOUNCE_DIFFUSE;
        if (intoSolid) {
            // Raygen re-traces from here with the gas bit masked off and
            // lands on the solid along solidDir.
            payload.scatterOrigin = solidFrom;
            payload.skipGasVolumes = true;
        } else {
            payload.scatterOrigin = p + w * (exitDist + exitEpsilon);
            payload.skipGasVolumes = false;
        }
        volumeRecordShadowSamples(shadowSamples);
        volumeRecordWalk(events, rwProbeTraces, rwShadowTraces, rwShadowSkipped, false);
        volumeRecordRay(densitySamples, 0u, 0u, 0u, VOLUME_MARCH_COMPLETED);
        return;
    }
    // Absorbed, Russian-rouletted, or out of events: the path ends here with
    // the light it gathered. Hitting maxEvents loses the remaining energy, so a
    // walk that keeps hitting it reads as a darker medium — raise the budget.
    payload.radiance = radiance;
    payload.attenuation = vec3(0.0);
    payload.scattered = false;
    volumeRecordShadowSamples(shadowSamples);
    volumeRecordWalk(events, rwProbeTraces, rwShadowTraces, rwShadowSkipped,
                     events >= uint(maxEvents));
    volumeRecordRay(densitySamples, 0u, 0u, 0u, VOLUME_MARCH_EXTINCTION);
}

