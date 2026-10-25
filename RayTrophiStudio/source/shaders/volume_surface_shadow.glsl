// Surface-side participating-medium shadows, shared by mesh/splat BSDF lighting.
//
// lightIndex names the light the ray goes to (-1 = Nishita sun). It exists for
// the volume light cache (volume_light_cache.glsl): the cached integral toward
// THAT light, read where the shadow ray enters the volume, is the same chord
// this march integrates - the cell was measured along the same line. The cache
// is never guessed from the direction; a light it does not model, or a cell not
// yet filled by the volume shader, takes the march below unchanged.
float computeVolumeShadowTransmittance(vec3 shadowOrigin, vec3 lightDir, float maxDist,
                                       int lightIndex) {
    int volCount = int(cam.pad0);
    if (volCount <= 0) return 1.0;
    if (maxDist <= 1e-4) return 1.0;
    const float EPS = 1e-6;
    float transmittance = 1.0;
    uint measuredShadowSamples = 0u;
    for (int vi = 0; vi < min(volCount, 16); vi++) {
        VkVolumeInstance vol = volumes.v[vi];
        if (vol.is_active == 0) continue;
        if (vol.volume_type == 3 || vol.source_type == 3) continue;
        // SurfaceSDF's 0..1 proxy represents a surface, not participating fog.
        // Match RayFusion's volume-shadow membership: never use its Fog knobs
        // to turn a water/solid reconstruction into an extra scattering medium.
        if (vol.source_type == 4) continue;
        if (vol.shadow_steps <= 0 || vol.shadow_strength <= 0.0) continue;
        float sigma_t = vol.scatter_coefficient + vol.absorption_coefficient;
        if (sigma_t < EPS || vol.density_multiplier < EPS) continue;

        vec3 lo, ld;
        lo.x = vol.inv_transform[0] * shadowOrigin.x +
            vol.inv_transform[1] * shadowOrigin.y +
            vol.inv_transform[2] * shadowOrigin.z + vol.inv_transform[3];
        lo.y = vol.inv_transform[4] * shadowOrigin.x +
            vol.inv_transform[5] * shadowOrigin.y +
            vol.inv_transform[6] * shadowOrigin.z + vol.inv_transform[7];
        lo.z = vol.inv_transform[8] * shadowOrigin.x +
            vol.inv_transform[9] * shadowOrigin.y +
            vol.inv_transform[10] * shadowOrigin.z + vol.inv_transform[11];
        ld.x = vol.inv_transform[0] * lightDir.x +
            vol.inv_transform[1] * lightDir.y +
            vol.inv_transform[2] * lightDir.z;
        ld.y = vol.inv_transform[4] * lightDir.x +
            vol.inv_transform[5] * lightDir.y +
            vol.inv_transform[6] * lightDir.z;
        ld.z = vol.inv_transform[8] * lightDir.x +
            vol.inv_transform[9] * lightDir.y +
            vol.inv_transform[10] * lightDir.z;

        vec3 inv;
        inv.x = abs(ld.x) > EPS ? 1.0 / ld.x : (ld.x >= 0.0 ? 1e7 : -1e7);
        inv.y = abs(ld.y) > EPS ? 1.0 / ld.y : (ld.y >= 0.0 ? 1e7 : -1e7);
        inv.z = abs(ld.z) > EPS ? 1.0 / ld.z : (ld.z >= 0.0 ? 1e7 : -1e7);

        vec3 bmin = vec3(vol.aabb_min[0], vol.aabb_min[1], vol.aabb_min[2]);
        vec3 bmax = vec3(vol.aabb_max[0], vol.aabb_max[1], vol.aabb_max[2]);
        vec3 t0 = (bmin - lo) * inv;
        vec3 t1 = (bmax - lo) * inv;
        vec3 tS = min(t0, t1);
        vec3 tL = max(t0, t1);
        float tNL = max(max(tS.x, tS.y), max(tS.z, 0.0));
        float tFL = min(min(tL.x, tL.y), tL.z);
        if (tNL >= tFL) continue;

        float tNW = max(tNL, 0.001);
        float tFW = min(tFL, maxDist);
        if (tFW <= tNW) continue;

        int reqSteps = clamp(vol.shadow_steps, 1, 64);
        float segLen = tFW - tNW;
        if (segLen <= 1e-5) continue;
        float cachedIntegral;
        if (volumeLightCacheRead(uint(vi), vol, lightIndex,
                                 shadowOrigin + lightDir * tNW, cachedIntegral)) {
            float strengthCached = clamp(vol.shadow_strength, 0.0, 1.0);
            transmittance *= (1.0 - strengthCached *
                              (1.0 - exp(-min(cachedIntegral * sigma_t, 10.0))));
            if (transmittance <= 0.0) break;
            continue;
        }
        // Initialize only after all membership, setting and interval rejects.
        // A missed/disabled field must not touch NanoVDB headers or tree data.
        pnanovdb_buf_t vdbBuf;
        vdbBuf.address = uint64_t(0);
        pnanovdb_map_handle_t vdbMapH;
        vdbMapH.address.byte_offset = 0u;
        pnanovdb_readaccessor_t vdbAcc;
        bool vdbReady = vol.volume_type == 2 && vol.vdb_grid_address != uint64_t(0);
        if (vdbReady) {
            vdbBuf.address = vol.vdb_grid_address;
            pnanovdb_grid_handle_t gridH;
            gridH.address.byte_offset = 0u;
            pnanovdb_tree_handle_t treeH = pnanovdb_grid_get_tree(vdbBuf, gridH);
            pnanovdb_root_handle_t rootH = pnanovdb_tree_get_root(vdbBuf, treeH);
            vdbMapH = pnanovdb_grid_get_map(vdbBuf, gridH);
            pnanovdb_readaccessor_init(vdbAcc, rootH);
        }
        float dMid = ch_volDensity(vol, shadowOrigin + lightDir * (tNW + 0.5 * segLen),
                                   vdbBuf, vdbMapH, vdbAcc, vdbReady);
        measuredShadowSamples++;
        float tauHint = max(0.0, dMid) * sigma_t * segLen;
        if (tauHint <= 0.02) continue;
        float stepScale = clamp(sqrt(tauHint), 0.25, 1.0);
        int steps = int(ceil(float(reqSteps) * stepScale));
        steps = clamp(steps, min(3, reqSteps), min(reqSteps, 16));
        float stepW = segLen / float(steps);
        stepW = max(stepW, 1e-5);
        float jitter = fract(sin(dot(shadowOrigin + lightDir * float(vi + 1),
            vec3(12.9898, 78.233, 37.719))) * 43758.5453);
        float opticalDepth = 0.0;
        for (int s = 0; s < steps; s++) {
            vec3 sp = shadowOrigin +
                lightDir * (tNW + (float(s) + jitter + 0.5) * stepW);
            float d = ch_volDensity(vol, sp, vdbBuf, vdbMapH, vdbAcc, vdbReady);
            measuredShadowSamples++;
            opticalDepth += d * sigma_t * stepW;
            if (opticalDepth > 10.0) break;
        }

        float physTrans = exp(-opticalDepth);
        float strength = clamp(vol.shadow_strength, 0.0, 1.0);
        transmittance *= (1.0 - strength * (1.0 - physTrans));
        if (transmittance <= 0.0) break;
    }
    if (measuredShadowSamples != 0u) volumeRecordShadowSamples(measuredShadowSamples);
    return transmittance;
}
