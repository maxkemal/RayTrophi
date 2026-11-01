    // ══════════════════════════════════════════════════════════════════════════
    // Ordered gas -> liquid handoff for overlapping burning-fluid domains.
    // Ordinary VDB/cloud volumes keep the original fast path.
    float layeredSurfaceT = -1.0;
    // ★Gate on "there is something in front of me", NOT on how the gas happens to be
    // stored. This used to require source_type == 5 (live dense GPU gas), which silently
    // excluded the SAME gas domain whenever it was replayed from a cache: restoring a
    // frame calls setGridDomainStates(), which clears gpu_resident_fields_valid, so the
    // domain publishes as a HOST NanoVDB volume instead. With the handoff gated off, the
    // gas marched its whole AABB and terminated the segment, and the SurfaceSDF volume
    // overlapping it was never reached — "the gas renders, the water surface is gone",
    // reproducible on cached playback only.
    // ★★An earlier attempt widened this to `source_type == 2` and changed NOTHING, because
    // 2 is not a value source_type can hold. The producer is VolumetricRenderer.cpp:505-508
    // — isosurface → 4, procedural → 3, everything else → 0 — and VulkanBackend.cpp:20204
    // then promotes live dense gas to 5. `2` is the VOLUME_TYPE of a NanoVDB volume, a
    // different field; that mixup is why the cached-playback bug survived the "fix".
    // ★OptiX is the oracle here and it settles the question: use_live_dense_gpu requires
    // !use_optix (scene_data.h:2356), so under OptiX the gas is ALWAYS host NanoVDB — and
    // OptiX renders this exact scene correctly, live and cached. Host-NanoVDB gas over a
    // liquid surface is therefore a configuration that MUST work.
    // So state the gate as what it means: hand off unless I AM the liquid (4) or a
    // procedural cloud (3). nearestSurfaceSDFCrossing skips every candidate that is not a
    // SurfaceSDF, so with no liquid in the scene this is a ≤16-slot early-out loop. An
    // ordinary VDB genuinely overlapping a water surface WANTS this ordering too.
    if (vol.source_type != 4 && vol.source_type != 3 && volCount > 1u) {
        volumeRecordArbiterGateOpen();
        layeredSurfaceT = nearestSurfaceSDFCrossing(
            rayOrigin, rayDir, tNear, tFar, volIdx, volCount);
        if (layeredSurfaceT > 0.0) {
            if (layeredSurfaceT <= tNear + 0.003) {
                payload.radiance = vec3(0.0);
                payload.attenuation = vec3(1.0);
                payload.scatterOrigin =
                    rayOrigin + rayDir * max(tNear, layeredSurfaceT - 0.001);
                payload.scatterDir = rayDir;
                payload.scattered = true;
                payload.skipGasVolumes = true;
                payload.bounceType = BOUNCE_TRANSPARENT;
                volumeRecordLayeredHandoff();
                return;
            }
            tFar = min(tFar, layeredSurfaceT - 0.001);
        }
    }

    // SOLID SURFACE DETECTION inside the volume AABB
    // If a solid triangle exists between tNear and tFar, we must stop the march
    // just before it and signal raygen to fire the next bounce with
    // gl_RayFlagsSkipAABBEXT so the triangle closesthit fires correctly.
    //
    // Strategy: one nearest-hit distance probe. The dedicated payload sentinel makes
    // shadow_anyhit return gl_HitTEXT directly; the former 1 + 5/6 binary-search
    // traces multiplied catastrophically when a gas domain touched the ground.
    // ══════════════════════════════════════════════════════════════════════════
    float solidT = -1.0;  // -1 = no solid found
    {
        // Performance gate:
        // In optically thick segments, inner solid surfaces are effectively invisible.
        // Skip costly triangle probes in those cases.
        bool needSolidProbe = true;
        float marchDistProbe = max(tFar - tNear, 0.0);
        float sigmaTCoeff = max(vol.scatter_coefficient + vol.absorption_coefficient, 0.0);
        // ★ Never gate the CAMERA segment. When the gate is wrong here the cost is
        // not a dimmer surface, it is a MISSING one: solidT stays -1, the march is
        // never clamped, the triangle's closesthit never fires, and the viewer
        // looks straight through the wall to whatever stands behind it. The
        // estimate below cannot be made safe for this case either — it multiplies
        // the sampled density by the FULL box traversal, so a large domain drives
        // tauEst up regardless of how close to the entry the surface actually sits,
        // and a surface a few centimetres inside a big smoke box is declared
        // invisible. One probe per primary ray is a bounded, predictable cost; on
        // secondary bounces a genuinely thick medium really does hide what is
        // behind it, so the gate keeps earning its keep there.
        bool primarySegment = (payload.primaryMeta & PL_PRIMARY_DONE) == 0u;
        if (!primarySegment && !hasCompanion && sigmaTCoeff > EPSILON && marchDistProbe > 1e-4) {
            float tA = tNear + min(0.05 * marchDistProbe, 0.25);
            float tB = tNear + 0.5 * marchDistProbe;
            float dA = sampleDensityAcc(vol, rayOrigin + rayDir * tA, vdbBuf, vdbMapH, vdbAcc);
            float dB = sampleDensityAcc(vol, rayOrigin + rayDir * tB, vdbBuf, vdbMapH, vdbAcc);
            // ★★ CONSERVATIVE, not representative.
            //
            // Skipping this probe does not merely omit a cheap detail: solidT
            // stays -1, so tFar is never clamped to the surface, the march runs
            // the full AABB INCLUDING the part behind the solid, and the
            // scatter continuation that lets the triangle's closesthit fire
            // never happens. The surface is not dimmed — it is never shaded at
            // all, and the volume closes over it as an opaque blob.
            //
            // So the estimate must err toward PROBING. Averaging two taps let a
            // flame's dense core speak for the whole ray: raise scatter or
            // absorption a little, tauEst crosses the gate, and every surface
            // behind that part of the fire vanishes at once — the reported
            // "past a threshold it goes fully opaque", and only in the areas
            // that happen to have geometry behind them. Take the MINIMUM
            // instead: a ray is only declared hopeless when it is thick
            // everywhere it was sampled, not merely thick somewhere.
            float dMin = max(0.0, min(dA, dB));
            float tauEst = dMin * sigmaTCoeff * marchDistProbe;
            float transEst = exp(-tauEst);
            // And the gate itself was far too loose. At 3% transmittance a lit
            // floor behind the fire is plainly visible, so dropping it was a
            // visible error, not an invisible optimisation. Only skip once the
            // surface could contribute less than a few thousandths.
            float probeThreshold = cameraInsideVolume ? 0.006 : 0.003;
            needSolidProbe = (transEst > probeThreshold);
        }

        if (needSolidProbe) {
            const uint PROBE_FLAGS = gl_RayFlagsSkipClosestHitShaderEXT
                                   | gl_RayFlagsNoOpaqueEXT;
            // Exclude the two VOLUME bits (0x02 gas/fog, 0x08 SurfaceSDF) and
            // keep everything else, including transient simulation particles
            // (0x04). Same set as the fluid branch's SOLID_MASK above.
            //
            // ★★★ 0x04 WAS EXCLUDED HERE AND THAT IS WHY SPLAT SPHERES STILL
            // VANISHED INSIDE A GAS DOMAIN after the fluid branch was fixed.
            // The postmortem at SOLID_MASK lists three symptoms of the excluded
            // bit and fixed only the fluid branch, deferring this one on cost
            // grounds — "the GAS march's 1+6 nested solid-location probes".
            // That cost argument had already expired: the 1+5/6 binary search
            // was replaced by the SINGLE any-hit distance probe below (see the
            // strategy note above, written when a gas domain touching the ground
            // multiplied those traces catastrophically). Widening a mask on one
            // terminate-on-first-hit probe adds no probes at all.
            //
            // ★ Symptom this restores: "invisible in EMPTY space". With no solid
            // behind them the probe returned nothing, solidT stayed -1, and the
            // no-scatter branch at the end of the march teleports the ray to
            // tFar + 0.002 — straight past every sphere in the box. The spheres
            // only ever survived when a real surface happened to stand behind
            // them and dragged them into the handoff.
            //
            // ★★ This works ONLY together with RT_MASK_VOLUME_HANDOFF. The probe
            // finding the splat is half of it; the handoff re-trace must also
            // keep 0x04 alive, or the splat is found and then erased on the very
            // next trace. Either change alone leaves the bug on screen, which is
            // why fixing the handoff first did not visibly help.
            const uint PROBE_MASK  = 0xF5;

            // Nearest accepted solid in [tNear, tFar].
            const uint VOLUME_SOLID_PROBE = 0xC17D5EEDu;
            shadowPayload = vec4(0.0, 0.0, 0.0, uintBitsToFloat(VOLUME_SOLID_PROBE));
            traceRayEXT(topLevelAS, PROBE_FLAGS, PROBE_MASK, 0, 1, 1,
                        rayOrigin, max(1e-4, tNear - 0.002), rayDir, tFar + 0.002, 1);
            volumeRecordSolidProbe(shadowPayload.z > 0.5);
            if (shadowPayload.z > 0.5) {
                solidT = shadowPayload.x;
                // Integrate to the solid; handoff masking guarantees progress, not a gap.
                tFar = min(tFar, solidT);
                // If the solid is essentially at or before the entry point, skip AABBs on next bounce
                if (tFar <= tNear) {
                    payload.radiance = vec3(0.0);
                    // No step-back epsilon — see the note at the first solid handoff.
                    payload.scatterOrigin = rayOrigin;
                    payload.scatterDir = rayDir;
                    payload.scattered = true;
                    // Gas march handoff: clear the gas bit only, so a coincident
                    // liquid SurfaceSDF (0x08) still gets its chance. See the
                    // note at the end-of-march handoff — clearing both bits here
                    // is what blacked out the domain in SDF mode.
                    payload.skipGasVolumes = true;
                    return;
                }
            }
        }
    }

    bool generalMixture = (hasCompanion &&
        matterFogOverlap(rayOrigin, rayDir, tNear, volCount) &&
        matterFogNeedsGeneralMixture(volIdx, companionIdx, volCount)) ||
        (vol.random_walk_enabled > 0.5 &&
            (vol.emission_mode != 0 || matterFogJointWalk(volIdx, volCount)));
    vec3 cachedRadiance, cachedTransmission;
    bool segmentCached = volumeCachedSegment(volIdx, companionIdx, volCount, generalMixture,
        rayOrigin, rayDir, tNear, tFar, vdbBuf, vdbMapH, vdbAcc,
        compBuf, compMapH, compAcc, cachedRadiance, cachedTransmission);
    if (generalMixture) {
        shadeFogMixture(rayOrigin, rayDir, tNear, tFar,
            solidT >= 0.0 || layeredSurfaceT > 0.0, volIdx, volCount,
            segmentCached, cachedRadiance, cachedTransmission);
        return;
    }

    // Path-traced multiple scattering replaces the march below when the volume
    // asks for it. Emitters took the mixture walk above, which gathers emission
    // along every flight. The march stays for Volume Graph programs (they may
    // rewrite density per sample, which the walk's flights and shadows would
    // not see), an overlapping companion medium, and a layered SDF surface
    // inside the box.
    {
        bool rwProgram = vol._reserved[1] > 0.5 &&
            matProgramOffset(uint(vol._reserved[1] - 1.0)) != MATPROG_NONE;
        if (vol.random_walk_enabled > 0.5 && !hasCompanion && !rwProgram &&
            layeredSurfaceT <= 0.0) {
            volumeRandomWalk(vol, rayOrigin, rayDir, tNear, tFar, solidT >= 0.0,
                             selectedVolumeSpan, vdbBuf, vdbMapH, vdbAcc);
            return;
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // RAY MARCH through volume (Regular stepping with jitter)
    // Matches OptiX volumetric ray march approach
    // ══════════════════════════════════════════════════════════════════════════
    // ── March LOD (docs/dev/RT_GAS_SECONDARY_MARCH.md §8) ───────────────────
    // worldData.w.volumeMarchLod, packed by Backend::packVolumeMarchLod:
    // bits 0-3 camera light stride-1, 4-7 bounce light stride-1, 8-11 bounce
    // step scale-1, 12-20 bounce step cap (0 = authored). A zero word is the
    // legacy march. Measured 2026-10-25 (splat inside gas): in-scatter is half
    // the march cost and bounce rays hold 45% of the density samples.
    uint marchLod = worldData.w.volumeMarchLod;
    int lightStride = volumeCameraSegment
        ? int(marchLod & 15u) + 1 : int((marchLod >> 4) & 15u) + 1;
    float lodStepScale = volumeCameraSegment ? 1.0 : float(((marchLod >> 8) & 15u) + 1u);
    int lodStepCap = volumeCameraSegment
        ? int((marchLod >> 22) & 511u) : int((marchLod >> 12) & 511u);
    // Bit 21: bounce rays read the radiance cache (§13).
    float stepSize = max(vol.step_size, 1e-4) * lodStepScale;
    float marchDist = tFar - tNear;
    const int hardStepCeiling = 2048;
    int authoredStepBudget = max(vol.max_steps, 1);
    int effectiveStepBudget = min(hardStepCeiling, authoredStepBudget);
    if (lodStepCap > 0) effectiveStepBudget = min(effectiveStepBudget, lodStepCap);
    int requiredSteps = max(1, int(ceil(marchDist / stepSize)));
    int maxSteps = min(requiredSteps, effectiveStepBudget);
    // Canonical Vulkan baseline: divide the complete interval into a fixed,
    // authored number of segments. Optical integration must never shorten the
    // geometric advance and strand the ray before tFar.
    float baseStep = marchDist / float(max(maxSteps, 1));
    
    float sigma_s_coeff = vol.scatter_coefficient;
    float sigma_a_coeff = vol.absorption_coefficient;
    uint volumeMatIndex = vol._reserved[1] > 0.5 ? uint(vol._reserved[1] - 1.0) : 0u;
    uint volumeProgram = vol._reserved[1] > 0.5 ? matProgramOffset(volumeMatIndex) : MATPROG_NONE;
    // Preserve the legacy fast path: a temperature NanoVDB lookup is eight
    // trilinear tree samples and must not run for an ordinary non-emissive cloud.
    bool needsTemperature = (vol.emission_mode >= 2) || (volumeProgram != MATPROG_NONE);

    // Companion medium constants (ray-constant; see resolveGasOverlapSegment).
    // Base coefficients only: the companion's material program is not run.
    vec3 compAbsColor = max(comp.absorption_color, vec3(0.0));
    float compAbsPeak = max(compAbsColor.r, max(compAbsColor.g, compAbsColor.b));
    vec3 compAbsWeights = compAbsPeak > 1e-5 ? compAbsColor / compAbsPeak : vec3(1.0);
    
    vec3  accumulated_radiance = vec3(0.0);
    vec3  transmittance = vec3(1.0);
    float opticalDepthWeightedT = 0.0;
    float opticalDepthWeight = 0.0;
    // Deterministic analytical single-scattering is accumulated below. Keep
    // the legacy continuation branch structurally present but disabled: a
    // second stochastic HG estimator would double-count the same event.
    bool didScatter = false;
    float scatterT = tFar;
    vec3 ambientSky = sampleSkyAmbient(rayDir);
    int shadowStride = clamp(vol.shadow_stride, 1, 16);
    // Keep this scalarized. Some NVIDIA Vulkan RT compiler versions crash
    // during pipeline creation on dynamically indexed local bool arrays in a
    // closest-hit shader.
    float cachedSceneShadow0 = 1.0;
    float cachedSceneShadow1 = 1.0;
    bool cachedSceneShadow0Valid = false;
    bool cachedSceneShadow1Valid = false;
    float cachedSunShadow = 1.0;
    bool cachedSunShadowValid = false;
    float cachedAmbientShadow = 1.0;
    bool cachedAmbientShadowValid = false;
    // In-scatter held between light-stride evaluations (-1 = none yet).
    vec3 heldInscatter = vec3(0.0);
    int heldInscatterStep = -1;

    // ── Scene-light subset for this ray ───────────────────────────────────────
    // At most two lights are evaluated per volume sample. The old code took
    // lights 0 and 1 by index while still weighting by lightCount/2, so from the
    // third light onward the scene was lit by the WRONG lights at inflated
    // energy — a five-light rig scaled lights 0-1 by 2.5x and never sampled the
    // other three at all.
    //
    // Two distinct lights are drawn uniformly WITHOUT replacement, which gives
    // each light probability 2/N and therefore weight N/2 — the same constant as
    // before, now attached to an estimator that is actually unbiased.
    //
    // The draw is RAY-CONSTANT, not per step, and that is deliberate: the
    // shadow-transmittance cache below is keyed on the slot (0/1) and reused for
    // up to `shadow_stride` steps. Re-drawing per step would silently pair one
    // light's radiance with another light's cached visibility.
    int   sampledLightCount = int(min(cam.lightCount, 2u));
    int   sampledLight0 = 0;
    int   sampledLight1 = 0;
    float lightWeight = 1.0;
    if (cam.lightCount > 0u) {
        int nLights = int(cam.lightCount);
        sampledLight0 = clamp(int(floor(rnd(payload.seed) * float(nLights))), 0, nLights - 1);
        if (nLights > 1) {
            // Offset draw over the remaining N-1 lights keeps the pair distinct
            // without a rejection loop.
            int offset = clamp(int(floor(rnd(payload.seed) * float(nLights - 1))), 0, nLights - 2);
            sampledLight1 = (sampledLight0 + 1 + offset) % nLights;
        }
        lightWeight = float(nLights) / float(sampledLightCount);
    }
    // Must match sampleSkyAmbient(): this is the representative direction of
    // the sky hemisphere whose radiance is used as the ambient source.
    vec3 ambientLightDir = normalize(vec3(0.0, 1.0, 0.0) * 0.55 + rayDir * 0.45);

    // Exactly the rejection threshold sampleDensityAcc applies to the remapped
    // density. The block skip must use the SAME test, or it discards space the
    // sampler would have shaded.
    float denseSkipCutoff = (vol._reserved[0] > 0.0) ? vol._reserved[0] : 0.0;

    // One source-space proposal, shared by all lighting updates on this ray.
    vec3 cachedInternalLight = vec3(0.0);
    bool cachedInternalLightValid = false;

    // Jitter first sample to reduce banding. Mix in ray state so horizon rays do
    // not share coherent step planes across neighboring pixels.
    float rayJitter = fract(sin(dot(rayOrigin + rayDir * 17.0, vec3(12.9898, 78.233, 37.719)) + float(payload.seed) * 0.000173) * 43758.5453);
    VolumePhaseCache phaseCache = volumePhaseCache();
    bool nanoIndexPath = vol.volume_type == 2 && vdbBuf.address != 0 &&
        vol._ext_reserved[7] < 0.5 && volumeProgram == MATPROG_NONE &&
        !(volumeEmissionIndependent(vol._retired_cloud[0]) && vol.density_multiplier <= 0.0);
    VolumeNanoRay primaryNanoRay;
    if (nanoIndexPath) {
        primaryNanoRay = volumeNanoRay(vol, rayOrigin, rayDir, vdbBuf, vdbMapH);
    }
    bool compNanoIndexPath = hasCompanion && volumeNanoIndexEligible(comp, compBuf);
    VolumeNanoRay companionNanoRay;
    if (compNanoIndexPath) {
        companionNanoRay = volumeNanoRay(comp, rayOrigin, rayDir, compBuf, compMapH);
    }
    // One transform and a live accessor per ray; sampleTemperature rebuilds both
    // and descends the tree for each of eight corners at every step.
    bool temperaturePrepared = false;
    VolumeNanoRay temperatureRay;
    pnanovdb_buf_t temperatureBuf;
    pnanovdb_readaccessor_t temperatureAcc;
    if (nanoIndexPath && !segmentCached) {
        mfPrepareTemperature(vol, rayOrigin, rayDir, temperaturePrepared,
            temperatureRay, temperatureBuf, temperatureAcc);
    }
    float t = tNear + rayJitter * baseStep;
    int step = 0;
    uint measuredDensitySamples = 0u;
    uint measuredEmptySegments = 0u;
    uint measuredTopologySegments = 0u;
    uint measuredDensityLeafSegments = 0u;
    // Bounce rays: cross the box on radiance cache lookups (§13). No material
    // program, not the procedural sky cloud. A companion medium is read from
    // its own slot and combined per step (volume_radiance_cache.glsl; the
    // companion's program is not run by the full march either). A missing
    // vertex leaves radiance and transmittance untouched and the march below
    // runs as before.
    if (segmentCached) {
        accumulated_radiance = cachedRadiance;
        transmittance = cachedTransmission;
        t = tFar;
    }
    while (t < tFar && step < maxSteps) {
        vec3  samplePos = rayOrigin + rayDir * t;

        // Live dense gas: block-majorant skip. Same safety rule as the NanoVDB
        // branch below — a Volume Graph may synthesize density where the source
        // field has none, so skipping is only valid for unmodified density.
        // Both skips test the PRIMARY field only; a companion may hold density
        // exactly where the primary is empty, so they are off in an overlap.
        if (!hasCompanion && vol.volume_type == 4 && vol.source_type == 5 &&
            volumeProgram == MATPROG_NONE) {
            float denseSkip = denseGasEmptyBlockStep(
                vol, samplePos, rayDir, denseSkipCutoff);
            if (denseSkip > baseStep * 1.01) {
                int skippedSegments = max(1, int(floor(denseSkip / baseStep)));
                measuredEmptySegments += uint(skippedSegments);
                measuredTopologySegments += uint(skippedSegments);
                step += skippedSegments;
                t = tNear + (float(step) + rayJitter) * baseStep;
                continue;
            }
        }

        // A Volume Graph may synthesize density in inactive source voxels.
        // Hierarchy skipping is therefore safe only for legacy/baked density.
        if (!hasCompanion && vdbBuf.address != 0 && volumeProgram == MATPROG_NONE) {
            uint skipKind = 0u;
            float sparseStep = nanoIndexPath
                ? nanoEmptyIndexStep(vol, primaryNanoRay.indexOrigin +
                    primaryNanoRay.indexDirection * t, primaryNanoRay.indexDirection,
                    baseStep, vdbBuf, vdbAcc, skipKind)
                : nanoEmptyTileStep(vol, samplePos, rayDir, baseStep,
                    vdbBuf, vdbMapH, vdbAcc, skipKind);
            if (sparseStep > baseStep * 1.01) {
                int skippedSegments = max(1, int(floor(sparseStep / baseStep)));
                measuredEmptySegments += uint(skippedSegments);
                if (skipKind == 2u)
                    measuredDensityLeafSegments += uint(skippedSegments);
                else
                    measuredTopologySegments += uint(skippedSegments);
                step += skippedSegments;
                t = tNear + (float(step) + rayJitter) * baseStep;
                continue;
            }
        }
        
        #include "volume_companion_empty.glsl"
        float emissionDensity;
        float density = nanoIndexPath
            ? volumeNanoDensityChannels(vol, primaryNanoRay, t, vdbBuf, vdbAcc, emissionDensity)
            : volumeEmissionChannelsAcc(vol, samplePos, vdbBuf, vdbMapH, vdbAcc, emissionDensity);
        measuredDensitySamples++;
        #include "volume_rt_empty_sample.glsl"
        float temperature = !needsTemperature ? 0.0 : (temperaturePrepared
            ? mfPreparedTemperature(temperatureRay, temperatureBuf, temperatureAcc, t)
            : sampleTemperature(vol, samplePos));
        vec3 stepScatterColor = vol.scatter_color;
        vec3 stepAbsorptionColor = vol.absorption_color;
        vec3 stepEmissionColor = vol.emission_color;
        float stepEmissionStrength = vol.emission_intensity;
        float stepAnisotropy = vol.scatter_anisotropy;
        float stepMultiScatter = vol.scatter_multi;
        if (volumeProgram != MATPROG_NONE) {
            float emptyAttribs[MP_ATTRIB_SLOTS];
            for (int ai = 0; ai < MP_ATTRIB_SLOTS; ++ai) emptyAttribs[ai] = 0.0;
            vec3 localPos;
            localPos.x = vol.inv_transform[0]*samplePos.x + vol.inv_transform[1]*samplePos.y + vol.inv_transform[2]*samplePos.z + vol.inv_transform[3];
            localPos.y = vol.inv_transform[4]*samplePos.x + vol.inv_transform[5]*samplePos.y + vol.inv_transform[6]*samplePos.z + vol.inv_transform[7];
            localPos.z = vol.inv_transform[8]*samplePos.x + vol.inv_transform[9]*samplePos.y + vol.inv_transform[10]*samplePos.z + vol.inv_transform[11];
            vec3 normalizedLocalPos = clamp(
                (localPos - vol.aabb_min) / max(vol.aabb_max - vol.aabb_min, vec3(1e-6)),
                vec3(0.0), vec3(1.0));
            MatProgOut vp = evalMaterialProgram(
                volumeProgram, vec2(0.0), samplePos, -rayDir, 0.5, vec3(0.0),
                emptyAttribs, localPos, -rayDir,
                density, temperature,
                (temperature > 0.0) ? clamp(temperature / max(vol.max_temperature, 1.0), 0.0, 1.0) : 0.0,
                0.0, vec3(0.0), samplePos,
                vol.scatter_color, vol.emission_color, max(vol.voxel_size, 1e-6), normalizedLocalPos,
                cam.waterTime);
            if ((vp.volumeWritten & (1u << 0)) != 0u) density = max(vp.volumeDensity, 0.0);
            if ((vp.volumeWritten & (1u << 1)) != 0u) stepScatterColor = max(vp.volumeScatterColor, vec3(0.0));
            if ((vp.volumeWritten & (1u << 2)) != 0u) sigma_s_coeff = max(vp.volumeScatterStrength, 0.0);
            if ((vp.volumeWritten & (1u << 3)) != 0u) stepAbsorptionColor = max(vp.volumeAbsorptionColor, vec3(0.0));
            if ((vp.volumeWritten & (1u << 4)) != 0u) sigma_a_coeff = max(vp.volumeAbsorptionStrength, 0.0);
            if ((vp.volumeWritten & (1u << 5)) != 0u) stepEmissionColor = max(vp.volumeEmissionColor, vec3(0.0));
            if ((vp.volumeWritten & (1u << 6)) != 0u) stepEmissionStrength = max(vp.volumeEmissionStrength, 0.0);
            if ((vp.volumeWritten & (1u << 7)) != 0u) stepAnisotropy = clamp(vp.volumeAnisotropy, -0.99, 0.99);
            if ((vp.volumeWritten & (1u << 8)) != 0u) stepMultiScatter = clamp(vp.volumeMultiScatter, 0.0, 1.0);
        }
        if (volumeProgram != MATPROG_NONE) emissionDensity = density;
        float opticalDensity = volumeEmissionOpticalDensity(density,
            vol.density_multiplier, vol._retired_cloud[0]) *
            volumeSootScale(vol, samplePos, emissionDensity, temperature);
        float sigma_s_local = opticalDensity * sigma_s_coeff;
        // Companion medium: extinction adds, scattering colour and phase are
        // mixed by each medium's share of the scattering coefficient.
        float compDensity = 0.0;
        float compSigmaS = 0.0;
        if (hasCompanion) {
            compDensity = compNanoIndexPath
                ? volumeNanoDensity(comp, companionNanoRay, t, compBuf, compAcc)
                : sampleDensityAcc(comp, samplePos, compBuf, compMapH, compAcc);
            compSigmaS = compDensity * comp.scatter_coefficient;
            float scatterSum = sigma_s_local + compSigmaS;
            if (compSigmaS > 0.0 && scatterSum > EPSILON) {
                float compShare = compSigmaS / scatterSum;
                stepScatterColor = mix(stepScatterColor, comp.scatter_color, compShare);
                stepAnisotropy = mix(stepAnisotropy, comp.scatter_anisotropy, compShare);
                stepMultiScatter = mix(stepMultiScatter, comp.scatter_multi, compShare);
            }
        }
        sigma_s_local += compSigmaS;
        // RGB absorption. Black historically meant "neutral absorption" in
        // existing assets, so preserve it as an achromatic fallback. Otherwise
        // Absorption Color identifies the wavelengths removed by the medium.
        vec3 authoredAbsorption = max(stepAbsorptionColor, vec3(0.0));
        float absorptionPeak = max(
            authoredAbsorption.r,
            max(authoredAbsorption.g, authoredAbsorption.b));
        vec3 absorptionWeights = absorptionPeak > 1e-5
            ? authoredAbsorption / absorptionPeak
            : vec3(1.0);
        vec3 sigma_a_local = opticalDensity * sigma_a_coeff * absorptionWeights
                           + compDensity * comp.absorption_coefficient * compAbsWeights;
        vec3 sigma_t_rgb = vec3(sigma_s_local) + sigma_a_local;
        float sigma_t_local = dot(
            sigma_t_rgb, vec3(0.2126, 0.7152, 0.0722));
        // Let low-density regions survive based on scattering optical depth, not
        // raw density alone. This means increasing scatter can actually keep thin
        // edges visible instead of them being discarded by a fixed density gate.
        if (sigma_t_local <= EPSILON && !volumeEmissionIndependent(vol._retired_cloud[0])) {
            step++;
            t = tNear + (float(step) + rayJitter) * baseStep;
            continue;
        }
        
        // Fixed coverage segment. Beer-Lambert remains analytic over the full
        // segment. Two variable strides were tried and removed (2026-10-25,
        // RT_GAS_SECONDARY_MARCH.md §12, §14):
        //   * from the density at the segment START: entering dense gas from a
        //     thin fringe it skipped the onset - more samples, brighter edges;
        //   * from the transmittance already spent (the cloud's deepGrow): in
        //     dense gas T falls from 0.35 to 0.001 within a few steps, so it
        //     saved no samples and still darkened the fireball cap.
        // A future attempt must bound the stride by a majorant.
        float dt = min(baseStep, tFar - t);
        if (dt <= 1e-6) break;

        // Current extinction
        vec3 extinction = sigma_t_rgb * dt;
        vec3 sampleTransmittance = exp(-extinction);
        
        // ── Multi-scatter transmittance blend (matches OptiX) ──
        // Blends single-scatter (Beer's law) with a softer 0.25x extinction approximation
        // to model multiple scattering. When scatter_multi > 0, volumes appear brighter
        // and more translucent — matching the OptiX renderer output.
        // Extinction remains Beer-Lambert; multiple scattering redistributes
        // direction instead of making the same medium transmit extra energy.
        vec3 one_minus_sampleT = vec3(1.0) - sampleTransmittance;
        // Representative volume distance for aerial perspective. Weight each
        // sample by the camera-path opacity it actually contributes, not by the
        // procedural AABB entry/exit. Sparse fringe therefore cannot move the
        // apparent atmospheric endpoint to the domain wall.
        float stepOpacityWeight = dot(
            transmittance * one_minus_sampleT,
            vec3(0.2126, 0.7152, 0.0722));
        opticalDepthWeightedT +=
            (t + 0.5 * dt) * max(stepOpacityWeight, 0.0);
        opticalDepthWeight += max(stepOpacityWeight, 0.0);
        
        // ── Volume Emission ──
        // Mode 0 = none, 1 = plain color, 2 = blackbody/color-ramp via temperature grid.
        // Energy-stable integration: multiply by one_minus_sampleT (bounded by 1) instead of dt.
        vec3 emis = vec3(0.0);
        if (vol.emission_mode == 1) {
            emis = volumeEmissionSource(stepEmissionColor, 1.0, emissionDensity, 1.0,
                stepEmissionStrength, vol._retired_cloud, false, vol.density_multiplier) *
                volumeEmissionMaskFactor(emissionDensity, volumeEmissionSamplingMultiplier(
                    vol.density_multiplier, vol._retired_cloud[0]), vol.emission_mask);
        } else if (vol.emission_mode >= 2) {
            emis = volumeEmissionAt(vol, samplePos, emissionDensity, temperature);
        }

        // ── In-Scattering (Direct Lighting) ──
        if (sigma_s_local > 0.0) {
            // Light stride (§8): in-scatter on every k-th sample, held in
            // between; density, opacity and transmittance stay per sample. A
            // gap of k or more (an empty-space skip) re-evaluates, so a held
            // value never crosses a skipped region.
            bool evaluateLight = heldInscatterStep < 0 ||
                (step % lightStride) == 0 || step - heldInscatterStep >= lightStride;
            vec3 inscatter = heldInscatter;
            if (evaluateLight) {
                inscatter = vec3(0.0);
            
                // Sample scene lights for object volumes. For the procedural sky cloud,
                // Nishita already provides the sun/sky block below; sampling the default
                // directional light here doubles the light march cost and shifts parity
                // away from the OptiX sky-cloud path.
                bool useSceneLights = !(worldData.w.mode == 2 && (vol.volume_type == 3 || vol.source_type == 3));
                if (useSceneLights && cam.lightCount > 0u) {
                    for (int ls = 0; ls < sampledLightCount; ls++) {
                        int li = (ls == 0) ? sampledLight0 : sampledLight1;
                        LightData light = lights.l[li];
                        int lightType = int(light.position.w + 0.5);

                        vec3  lightDir;
                        float lightDist;
                        float lightAtten = 1.0;
                        bool  lightInverseSquare = true;
                        // Only an area light consumes randoms: point/spot/directional
                        // are delta sources, and drawing for them would churn the RNG
                        // stream every step for nothing.
                        //
                        // The area sample varies per step while the shadow-march
                        // result is cached across `shadow_stride` steps, so a cached
                        // transmittance can belong to a slightly different point on
                        // the light. The offset is bounded by the light's own extent
                        // and decorrelates the noise; pinning the sample per ray
                        // instead would trade that for visible banding.
                        float ru = 0.5, rv = 0.5;
                        if (lightType == 2) {
                            ru = rnd(payload.seed);
                            rv = rnd(payload.seed);
                        }
                        if (!volSampleLight(light, samplePos, ru, rv,
                                            lightDir, lightDist, lightAtten,
                                            lightInverseSquare)) {
                            continue;   // outside the cone / behind an area light
                        }
                        if (lightInverseSquare) {
                            // Analytic 1/r^2 across the whole segment instead of a
                            // point sample at its start. Without this the width and
                            // brightness of a light shaft track the step size rather
                            // than the light.
                            //
                            // Area lights sample a point on the shape, so their
                            // closest-approach geometry is that sample's, not the
                            // light origin's — reconstruct the sampled position from
                            // the direction and distance already returned.
                            vec3 sampledLightPos = samplePos + lightDir * lightDist;
                            lightAtten *= volSegmentInvSqAverage(
                                rayOrigin, rayDir, sampledLightPos, t, t + dt);
                        }

                        // Phase function evaluation
                        float cosTheta = dot(rayDir, lightDir);
                        float phase = volumeCachedPhase(phaseCache, ls,
                            lightType == 1 && !hasCompanion && volumeProgram == MATPROG_NONE,
                            cosTheta, stepAnisotropy,
                            vol.scatter_anisotropy_back, vol.scatter_lobe_mix);
                        vec3 scatteringAlbedoRGB =
                            vec3(sigma_s_local) / max(sigma_t_rgb, vec3(EPSILON));
                        float scatteringAlbedo = dot(
                            scatteringAlbedoRGB, vec3(0.2126, 0.7152, 0.0722));
                        phase = mix(
                            phase, 1.0 / (4.0 * PI),
                            clamp(stepMultiScatter * scatteringAlbedo, 0.0, 1.0));
                    
                        // Light march transmittance through volume toward light.
                        // Geometry occlusion is NOT checked per-step (matching OptiX raymarch_volumetric_object
                        // behavior): solid objects are handled by TLAS traversal order, not per-sample rays.
                        // This prevents hard black shadows from solids inside the volume.
                        float shadowTr = 1.0;
                        float cachedLightIntegral;
                        float cachedCompLightIntegral = 0.0;
                        // Volume light cache: read every step, no stride - a lookup
                        // costs less than the strided march it replaces. A companion
                        // medium reads ITS OWN slot, exactly as companionLightMarch
                        // marches its own grid; the two transmittances multiply.
                        if (sigma_t_local * dt > 0.02 && vol.shadow_steps > 0 &&
                            volumeLightCacheFetch(volIdx, vol, li, samplePos,
                                                  vdbBuf, vdbMapH, vdbAcc, cachedLightIntegral) &&
                            (!hasCompanion || comp.shadow_steps <= 0 ||
                             volumeLightCacheFetch(uint(companionIdx), comp, li, samplePos,
                                                   compBuf, compMapH, compAcc, cachedCompLightIntegral))) {
                            shadowTr = volumeShadowFromIntegral(vol, cachedLightIntegral) *
                                (hasCompanion ? volumeShadowFromIntegral(comp, cachedCompLightIntegral) : 1.0);
                        } else if (sigma_t_local * dt > 0.02) {
                            bool shadowValid = (ls == 0)
                                ? cachedSceneShadow0Valid : cachedSceneShadow1Valid;
                            bool updateShadow = !shadowValid || volumeShadowDue(step, shadowStride, lightStride);
                            if (updateShadow) {
                                float shadowMaxDist = min(
                                    lightDist,
                                    max(baseStep, volumeExitDistance(vol, samplePos, lightDir)));
                                float evaluatedShadow = lightMarchAcc(
                                    vol, samplePos, lightDir, shadowMaxDist,
                                    vdbBuf, vdbMapH, vdbAcc, -1.0)
                                    * companionLightMarch(
                                        hasCompanion, comp, samplePos, lightDir, lightDist,
                                        baseStep, compBuf, compMapH, compAcc, -1.0);
                                volumeRememberSunShadow(lightType, vol, hasCompanion, comp,
                                    samplePos, lightDir, lightDist, shadowMaxDist, baseStep,
                                    step, evaluatedShadow);
                                if (ls == 0) {
                                    cachedSceneShadow0 = evaluatedShadow;
                                    cachedSceneShadow0Valid = true;
                                } else {
                                    cachedSceneShadow1 = evaluatedShadow;
                                    cachedSceneShadow1Valid = true;
                                }
                            }
                            shadowTr = (ls == 0) ? cachedSceneShadow0 : cachedSceneShadow1;
                        }
                    
                        vec3 lightColor = light.color.rgb * light.color.a;
                        inscatter += lightWeight * lightColor * lightAtten * phase * shadowTr
                                   * stepScatterColor * scatteringAlbedoRGB;
                    }
                }
            
                // Sun/sky light contribution (if Nishita sky active)
                if (worldData.w.mode == 2) {
                    vec3 sunDir = normalize(worldData.w.sunDir);
                    float cosSun = dot(rayDir, sunDir);
                    float sunPhase = volumeCachedPhase(phaseCache, 2,
                        !hasCompanion && volumeProgram == MATPROG_NONE,
                        cosSun, stepAnisotropy,
                        vol.scatter_anisotropy_back, vol.scatter_lobe_mix);
                    vec3 scatteringAlbedoRGB =
                        vec3(sigma_s_local) / max(sigma_t_rgb, vec3(EPSILON));
                    float scatteringAlbedo = dot(
                        scatteringAlbedoRGB, vec3(0.2126, 0.7152, 0.0722));
                    sunPhase = mix(
                        sunPhase, 1.0 / (4.0 * PI),
                        clamp(stepMultiScatter * scatteringAlbedo, 0.0, 1.0));
                
                    float sunShadowTr = 1.0;
                    float cachedSunIntegral;
                    float cachedCompSunIntegral = 0.0;
                    if (sigma_t_local * dt > 0.02 && vol.shadow_steps > 0 &&
                        volumeLightCacheFetch(volIdx, vol, -1, samplePos,
                                              vdbBuf, vdbMapH, vdbAcc, cachedSunIntegral) &&
                        (!hasCompanion || comp.shadow_steps <= 0 ||
                         volumeLightCacheFetch(uint(companionIdx), comp, -1, samplePos,
                                               compBuf, compMapH, compAcc, cachedCompSunIntegral))) {
                        sunShadowTr = volumeShadowFromIntegral(vol, cachedSunIntegral) *
                            (hasCompanion ? volumeShadowFromIntegral(comp, cachedCompSunIntegral) : 1.0);
                    } else if (sigma_t_local * dt > 0.02) {
                        bool updateSunShadow = !cachedSunShadowValid || volumeShadowDue(step, shadowStride, lightStride);
                        if (updateSunShadow) {
                            float sunShadowMaxDist = max(
                                baseStep, volumeExitDistance(vol, samplePos, sunDir));
                            if (!volumeReuseSunShadow(samplePos, sunDir, sunShadowMaxDist,
                                    step, cachedSunShadow)) {
                                cachedSunShadow = lightMarchAcc(
                                    vol, samplePos, sunDir, sunShadowMaxDist,
                                    vdbBuf, vdbMapH, vdbAcc, -1.0)
                                    * companionLightMarch(
                                        hasCompanion, comp, samplePos, sunDir, 1e30,
                                        baseStep, compBuf, compMapH, compAcc, -1.0);
                            }
                            cachedSunShadowValid = true;
                        }
                        sunShadowTr = cachedSunShadow;
                    }
                
                    vec3 sunLi = sampleTransmittanceLUT(samplePos, sunDir) * worldData.w.sunColor * worldData.w.sunIntensity;
                    inscatter += sunLi * sunPhase * sunShadowTr
                               * stepScatterColor * scatteringAlbedoRGB;
                }
            
                // 3. Sky/Ambient lighting (closer to CPU world.evaluate(up) behavior)
                vec3 ambientAlbedo =
                    vec3(sigma_s_local) / max(sigma_t_rgb, vec3(EPSILON));
                float ambientAlbedoLuma = dot(
                    ambientAlbedo, vec3(0.2126, 0.7152, 0.0722));
                // Ambient is incident radiance too: letting it reach every dense
                // sample unattenuated fills the entire cloud with a flat white
                // source and erases density/depth cues. Use a cached representative
                // sky-hemisphere march so cores darken while sky-facing lobes remain
                // lit. A minimum stride of four bounds the Vulkan RT cost.
                // Hemispherical ambient changes slowly along a primary ray. Eight
                // samples of reuse removes a major nested NanoVDB-march cost without
                // altering its Beer-Lambert visibility.
                int ambientShadowStride = max(shadowStride, 8);
                bool updateAmbientShadow =
                    !cachedAmbientShadowValid || volumeShadowDue(step, ambientShadowStride, lightStride);
                if (updateAmbientShadow && sigma_t_local * dt > 0.01) {
                    float ambientShadowMaxDist = max(
                        baseStep,
                        volumeExitDistance(vol, samplePos, ambientLightDir));
                    cachedAmbientShadow = lightMarchAcc(
                        vol, samplePos, ambientLightDir, ambientShadowMaxDist,
                        vdbBuf, vdbMapH, vdbAcc, 1.0)
                        * companionLightMarch(
                            hasCompanion, comp, samplePos, ambientLightDir, 1e30,
                            baseStep, compBuf, compMapH, compAcc, 1.0);
                    cachedAmbientShadowValid = true;
                }
                inscatter += ambientSky * stepScatterColor * ambientAlbedo
                           * cachedAmbientShadow;

                // Bounded higher-order sky scattering approximation. Recycle only
                // a fraction of the ambient energy blocked on the representative
                // sky path; unlike emission this vanishes when external sky light
                // vanishes, remains density/depth dependent, and never exceeds the
                // original unoccluded ambient budget:
                //   visibility = T_sky + multi * albedo * P(2+) * (1-T_sky) <= 1.
                float multiSkyVisibility = volumeHigherOrderSkyVisibility(
                    cachedAmbientShadow, stepMultiScatter, ambientAlbedoLuma);
                inscatter += ambientSky * stepScatterColor * ambientAlbedo
                           * multiSkyVisibility;
            
                // CPU parity integration:
                // step_color = source * (1 - step_transmittance)
                // accumulated += step_color * current_transparency
                inscatter += volumeCachedInternalLighting(vol, samplePos, rayDir,
                    !hasCompanion && volumeProgram == MATPROG_NONE, step,
                    cachedInternalLight, cachedInternalLightValid) * stepScatterColor *
                    vec3(sigma_s_local) / max(sigma_t_rgb, vec3(EPSILON));
                inscatter = clampVolumeRadiance(inscatter, 8.0);
                heldInscatter = inscatter;
                heldInscatterStep = step;
            }
            accumulated_radiance += transmittance * volumeEmissionIntegrate(
                inscatter + emis, emis, sigma_t_rgb, dt, vol._retired_cloud[0]);
        } else if (any(greaterThan(emis, vec3(0.0)))) {
            // Emission-only medium segment
            accumulated_radiance += transmittance * volumeEmissionIntegrate(
                emis, emis, sigma_t_rgb, dt, vol._retired_cloud[0]);
        }
        
        // Update transmittance
        transmittance *= sampleTransmittance;
        
        // ── Stochastic Scatter Event (Delta Tracking) ──
        // Probability of scattering at this step
        // No stochastic volume continuation: the direct single-scattering
        // estimator above already accounts for redirected radiance.
        
        // Early termination if transmittance is negligible
        if (max(transmittance.r, max(transmittance.g, transmittance.b)) < 0.001) {
            break;
        }
        step++;
        t = tNear + (float(step) + rayJitter) * baseStep;
    }
    
    // ══════════════════════════════════════════════════════════════════════════
    // OUTPUT — Set payload for path tracer integration
    // ══════════════════════════════════════════════════════════════════════════
    
    // Sanitize against NaN/Inf. A single non-finite sample (e.g. a degenerate
    // ray when two coincident volumes — foam fog + fluid iso surface — share the
    // domain AABB) otherwise poisons the accumulation buffer permanently → the
    // whole domain goes black until accumulation resets. Drop the bad sample.
    if (any(isnan(accumulated_radiance)) || any(isinf(accumulated_radiance)))
        accumulated_radiance = vec3(0.0);
    if (any(isnan(transmittance)) || any(isinf(transmittance)) ||
        any(lessThan(transmittance, vec3(0.0))))
        transmittance = vec3(1.0);
    float transmittanceLuma = dot(
        transmittance, vec3(0.2126, 0.7152, 0.0722));

    // Accumulated in-scattered radiance
    payload.radiance  = accumulated_radiance;
    payload.skipVolumeAABBs = false; // default; overridden below if solid found
    // ── Solid surface found inside the volume: hand off to closesthit ──
    // March was already clamped to solidT. Now position the scatter ray
    // just before the solid surface and tell raygen to skip AABBs next
    // bounce (gl_RayFlagsSkipAABBEXT) so the triangle closesthit fires.
    if (solidT >= 0.0) {
        volumeRecordRay(
            measuredDensitySamples,
            measuredEmptySegments,
            measuredTopologySegments,
            measuredDensityLeafSegments,
            VOLUME_MARCH_COMPLETED);
        payload.attenuation  *= transmittance;
        // No step-back epsilon — see the note at the first solid handoff. This was
        // the site the camera ray actually goes through for a gas domain.
        payload.scatterOrigin = rayOrigin;
        payload.scatterDir    = rayDir;
        payload.scattered     = true;
        // ★★★ GAS handoff clears ONLY the gas bit (0xFD), not both volume bits.
        //
        // Measured 2026-08-16 with the probe counters: in SurfaceSDF mode this
        // path fires on 6.6% of volume rays (the solid's screen coverage) and
        // the handoff mask was RT_MASK_VOLUME_HANDOFF (0xF5), which also clears
        // 0x08 — the liquid's own SurfaceSDF. So every ray that found a solid
        // inside a coincident gas+liquid domain erased the LIQUID on the
        // re-trace and shaded whatever stood behind it. The domain went black in
        // SDF mode and looked fine in splat mode, because splats are 0x04 and
        // 0x04 survives that mask. That is exactly the reported symptom.
        //
        // skipGasVolumes already exists for precisely this handoff ("gas segment
        // -> coincident SurfaceSDF") and keeps 0x08 alive. Reaching this line
        // means the iso branch did not claim the ray, i.e. this is the gas/fog
        // march, so clearing the gas bit alone is both correct and enough to
        // prevent re-hitting this same box.
        //
        // ★ No ping-pong: if the re-trace lands on the SDF, THAT branch hands
        // off with RT_MASK_VOLUME_HANDOFF, which clears both bits and terminates
        // the chain. gas -> sdf -> solid, three traces, bounded.
        payload.skipGasVolumes = true;
        volumeRecordGasHandoff();
        return;
    }

    float volumeContribution = max(max(accumulated_radiance.r, accumulated_radiance.g), accumulated_radiance.b);
    float volumeOpacity = 1.0 - transmittanceLuma;
    // Keep ultra-thin fog tails out of the primary auxiliary buffers. They were
    // being treated as first-hit geometry, which overemphasized weak density
    // regions in Vulkan RT denoiser output compared to OptiX.
    bool primaryVolumeInteraction =
        volumeOpacity > 0.04 || volumeContribution > 5e-4;
    if ((payload.primaryMeta & PL_PRIMARY_DONE) == 0u && primaryVolumeInteraction) {
        float representativeVolumeT = opticalDepthWeight > 1e-8
            ? opticalDepthWeightedT / opticalDepthWeight
            : 0.5 * (tNear + tFar);
        payload.primaryARG  = packHalf2x16(vol.scatter_color.rg);
        payload.primaryABT  = packHalf2x16(vec2(vol.scatter_color.b, transmittanceLuma));
        // PL_PRIMARY_VOLUME reinterprets primaryNrm as float bits containing
        // the optical-depth centroid distance. Raygen reconstructs the volume
        // AOV normal as -primaryRayDir without increasing payload size.
        payload.primaryNrm  = floatBitsToUint(representativeVolumeT);
        // Material id stays 0xFFFF: volumes have no scene material index.
        payload.primaryMeta = (payload.primaryMeta & PL_DISP_MASK)
                            | PL_PRIMARY_DONE | PL_PRIMARY_VOLUME
                            | PL_PRIMARY_VOLUME_DEPTH | PL_MATID_MASK;
    }

    if (didScatter && transmittanceLuma < 0.99) {
        // Scatter event — continue path with new direction
        vec3 scatterPos = rayOrigin + rayDir * scatterT;
        
        // Choose lobe for direction sampling
        float g = (rnd(payload.seed) < vol.scatter_lobe_mix) 
                  ? vol.scatter_anisotropy 
                  : vol.scatter_anisotropy_back;
        vec3 newDir = sampleHG(rayDir, g, payload.seed);
        
        payload.scatterOrigin = scatterPos;
        payload.scatterDir    = newDir;
        payload.attenuation  *= vol.scatter_color * transmittance;
        payload.scattered     = true;
    } else {
        // No scatter — attenuate throughput by volume transmittance
        payload.attenuation *= transmittance;

        // Set scattered = true with original direction to let the ray continue through
        // Ensure forward progress to avoid re-hitting the same boundary when camera is inside.
        if (layeredSurfaceT <= 0.0) volumeRecordTeleport();
        // ★★★★★ THE ADVANCE MUST OUTRUN THE GAP BETWEEN COINCIDENT BOXES.
        //
        // This was a hardcoded 0.002 world units. Two domains sharing a volume
        // (the normal way a burning liquid is authored: one fluid box, one gas
        // box, same extents) put their faces in the SAME PLANE, or a few
        // thousandths apart. Measured on the reproducing scene:
        //     fluid X,Z -2.5      .. 2.5        gas X,Z -2.5 .. 2.5   (identical)
        //     fluid Y   -2.5      .. 2.5        gas Y   -2.4970703125 ..
        // The 0.002 step is SMALLER than that 0.0029 offset, so a ray leaving one
        // box landed inside/on the face of the other, hit it again, advanced 0.002
        // again... never making progress. skipGasVolumes is false on this path, so
        // no mask stops the re-entry either.
        //
        // ★★ It reads as BLACK, not as a hang: raygen's free-pass budget runs out
        // with no light gathered. Raising maxBounces does NOTHING — a loop with
        // zero forward progress is invariant to budget, which is exactly why the
        // 64-bounce test came back unchanged and was misread as exoneration.
        //
        // ★★★ Measured, same scene, counters reset between reads:
        //     coincident faces : volume_rays 1,306,255,937  density/ray 0.0001
        //     boxes separated  : volume_rays   111,898,134  density/ray 0.9832
        // 11.7x the volume traces, each doing NO work. The cost explosion and the
        // black band are ONE root, not two.
        //
        // ★ A solid inside the domain "fixed" it because the solid probe hands the
        // ray to real geometry and it escapes the loop; splat mode "fixed" it
        // because there is no second coincident box. Both are escapes, not cures.
        //
        // The voxel is the smallest distance this volume can represent, so a
        // quarter voxel past the exit cannot skip any representable content, and
        // it clears a coincident-face gap by an order of magnitude.
        float exitEpsilon = max(0.002, vol.voxel_size * 0.25);
        payload.scatterOrigin = layeredSurfaceT > 0.0
            ? rayOrigin + rayDir * max(tNear, layeredSurfaceT - 0.0005)
            : rayOrigin + rayDir * (tFar + exitEpsilon);
        payload.scatterDir    = rayDir;
        payload.scattered     = (transmittanceLuma > 0.01); // Stop if fully absorbed

        // ★★★★★ FORWARD PROGRESS IS GUARANTEED BY THE MASK, NOT BY A DISTANCE.
        //
        // The exit epsilon above is a RACE: it advances a fixed distance against
        // a gap whose size nobody controls. Two domains authored with the same
        // extents put their faces in one plane; widening the epsilon from 0.002
        // to a quarter voxel only MOVED the trapped band further in, because some
        // pair of surfaces is always closer than whatever constant is chosen.
        // Measured with fire present and the epsilon already in the build:
        //     healthy (boxes apart) : volume_rays 111,898,134  density/ray 0.9832
        //     trapped (same extents): volume_rays 419,661,625  density/ray 0.0196
        // 3.75x the entries, 50x less work each — the ray keeps being handed back
        // into a box it has already marched, burning raygen's free passes until
        // the path dies with no light gathered. That is the black band, and the
        // same re-entries are the pathtrace cost explosion. One root.
        //
        // ★★ A budget cannot fix a loop whose progress is zero, which is why the
        // 64-bounce test came back unchanged. Only refusing the re-entry fixes it.
        //
        // ★ Gated on the segment being effectively TRANSPARENT, and that gate is
        // what keeps this safe. Skipping gas the ray has already established it
        // cannot see is a visual no-op; the flag is consumed by the very next
        // trace and cleared at the top of this shader, so a genuinely dense second
        // gas volume loses at most one pass and only when the first one was clear.
        // Any segment that actually absorbed or scattered keeps the old behaviour
        // untouched.
        bool transparentSegment =
            (volumeOpacity <= 0.04 && volumeContribution <= 5e-4);

        // ★★★ THE TRAP'S SIGNATURE IS A DEGENERATE SPAN, NOT A THIN GAS.
        //
        // The first version of this mask fired on `transparentSegment` alone, and
        // that is far broader than the fault it was written for. A flame's edges,
        // wispy smoke and a dying fire are all LEGITIMATELY transparent, so the
        // mask suppressed gas for one trace after almost every thin segment and
        // the gas lost its multi-segment accumulation. Visible as: inside an
        // overlapping fluid domain the gas collapsed to a thin shell at the box
        // plane, with a different look above and below it — the overlap is where
        // the arbiter clamps tFar, so that is where the mis-gate bit hardest.
        //
        // ★ A trapped ray and a thin gas are opposite in TRAVERSAL, which is what
        // separates them cleanly. Trapped: the ray was pushed to a coincident
        // box's exit face, so re-entering leaves less than a voxel to cross.
        // Thin gas: the ray crosses the WHOLE box and simply finds little density.
        // So use the raw box interval from the intersection shader — tNear/tFar
        // here may already be clamped by the arbiter or the solid probe, which
        // would make a healthy full traversal look degenerate.
        float rawSpan = selectedVolumeSpan;
        bool degenerateSpan = rawSpan < max(0.01, vol.voxel_size * 0.5);

        // Requiring BOTH is strictly narrower than the trap: a trapped segment is
        // degenerate AND does no work. Nothing that marches a real span is
        // affected, so legitimate gas keeps every segment it had before.
        payload.skipGasVolumes =
            payload.scattered &&
            ((layeredSurfaceT > 0.0) ||
             (degenerateSpan && transparentSegment && layeredSurfaceT <= 0.0));
        // Direction is unchanged on this path (didScatter is structurally off),
        // so the continuation is never a bounce: BOUNCE_MEDIUM_PASS is free in
        // raygen's bounce budget. Transparent segments keep BOUNCE_TRANSPARENT.
        payload.bounceType = transparentSegment ? BOUNCE_TRANSPARENT : BOUNCE_MEDIUM_PASS;
        // How many medium crossings a path makes. One per box crossed is
        // healthy; many per path is the short-hop re-entry the intersection
        // clamp used to cause.
        if (payload.scattered && payload.bounceType == BOUNCE_MEDIUM_PASS)
            volumeRecordMediumPass();
    }
    uint marchOutcome = transmittanceLuma <= 0.01
        ? VOLUME_MARCH_EXTINCTION
        : ((step >= maxSteps && t < tFar)
            ? VOLUME_MARCH_STEP_BUDGET
            : VOLUME_MARCH_COMPLETED);
    volumeRecordRay(
        measuredDensitySamples,
        measuredEmptySegments,
        measuredTopologySegments,
        measuredDensityLeafSegments,
        marchOutcome);
