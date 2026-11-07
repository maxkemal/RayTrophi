
        const float ISO_THRESH = 0.5;
        // IOR comes from the bound fluid material (_ext_reserved[0]); fall back
        // to water if unset. Drives both refraction bending and Fresnel.
        float IOR_WATER = (vol._ext_reserved[0] > 1.0) ? vol._ext_reserved[0] : 1.33;

        // Walk quality. The step must satisfy TWO constraints:
        //   1. fine enough for surface accuracy (voxel-relative, step_size),
        //   2. coarse enough that `max_steps` actually reach tFar — otherwise
        //      the walk stops partway and the far side of the fluid is skipped
        //      (back cells "pass"). We take the MAX of the fine step and the
        //      cover step (marchLen / cap), so a large domain coarsens
        //      gracefully instead of being truncated. Raise max_steps for both
        //      fine AND complete on big domains.
        float marchLen = max(0.0, tFar - tNear);
        int   isoCap = clamp(vol.max_steps, 32, 2048);
        float fineStep = clamp(vol.step_size, vol.voxel_size * 0.1, vol.voxel_size * 0.5);
        float coverStep = marchLen / float(isoCap);
        float step = max(0.001, max(fineStep, coverStep));
        int   maxSteps = min(int(marchLen / step) + 2, isoCap + 2);

        float t      = tNear;
        float startD = sampleIsoField(vol, rayOrigin + rayDir * t, vdbBuf, vdbMapH, vdbAcc);
        // ★Side test with hysteresis — the black-band candidate.
        // A bare `startD > ISO_THRESH` is a coin flip whenever the walk BEGINS on
        // the boundary, and on Vulkan that happens by construction: the gas
        // closest-hit hands the ray over by restarting it a hair before the
        // crossing it just found by bisection, so the first sample sits within
        // rounding distance of 0.5. Land one ulp on the wrong side and the surface
        // is shaded as an EXIT — the entry event never happens and Beer-Lambert is
        // applied over the whole remaining depth, i.e. a correctly placed but
        // pitch-black surface, flickering per pixel.
        // OptiX uses the same naive test and stays clean precisely because it never
        // restarts a ray at a boundary: one call walks the ordered volume list from
        // the camera's t_enter. So this ambiguity is structurally Vulkan-only.
        // "Inside" now means robustly inside (interior reads 1.0, ≫ 0.55); a start
        // sitting ON the boundary is treated as just-outside-about-to-enter, which
        // is exactly what a handoff means. prevD is pulled below the threshold too,
        // otherwise the first step can't register the entering crossing and the
        // surface would silently vanish instead.
        const float ISO_HYST = 0.05;
        bool  startInside = startD > ISO_THRESH + ISO_HYST;
        float prevD  = startInside ? startD : min(startD, ISO_THRESH - 1e-4);
        float hitT   = -1.0;

        #include "surface_sdf_main_search.glsl"

        // ── Whitewater foam composited in FRONT of the surface ──────────────
        // Foam rides this same volume's TEMPERATURE channel (scaled by
        // FOAM_TEMP_SCALE = 10000 on upload; see
        // SceneData::syncSimulationRenderVolumes). One volume on the domain AABB
        // means no coincident-volume case for the integrator to drop — this is
        // the fix for the SDF+foam black cube. March it as bright white single-
        // scatter in front of the iso surface; everything behind is dimmed by
        // the foam transmittance. NO self-shadow: self-shadowing is what blacked
        // out the old separate fog volume.
        vec3  foam_inscatter = vec3(0.0);
        float foam_T = 1.0;
        if (vol.vdb_temp_address != 0) {
            const float FOAM_TEMP_SCALE = 10000.0;
            // Opacity rides _ext_reserved[6] (foam_shader density × scatter,
            // packed at sync); fall back to a sane default if unset.
            float foamOptical = vol._ext_reserved[6] > 1e-3 ? vol._ext_reserved[6] : 8.0;
            // Foam tint rides _ext_reserved[3..5] (foam_shader scattering colour);
            // default to a faint cool white.
            vec3 foamAlbedo = vec3(vol._ext_reserved[3], vol._ext_reserved[4], vol._ext_reserved[5]);
            if (dot(foamAlbedo, vec3(1.0)) < 1e-3) foamAlbedo = vec3(0.95, 0.97, 1.0);

            float foamEnd = (hitT < 0.0) ? tFar : hitT;
            float foamLen = foamEnd - tNear;
            if (foamLen > 1e-5) {
                int fsteps = int(foamLen / max(vol.voxel_size, foamLen / 48.0)) + 1;
                fsteps = min(fsteps, 64);
                float fstep = foamLen / float(fsteps);
                vec3  skyAmb  = sampleSkyAmbient(rayDir);
                vec3  sunDirF = normalize(worldData.w.sunDir);
                float jitf    = rnd(payload.seed);
                // ── Ray-constant shading terms (HOISTED out of the per-step loop) ──
                // view·sun, the HG phase and the thin-film cos() are identical for
                // every foam sample on this ray → evaluate them ONCE, not per step.
                float cosVS      = dot(rayDir, sunDirF);                      // view·sun
                float foamPhase  = henyeyGreenstein(cosVS, 0.35);            // mild forward lobe
                vec3  silver     = mix(vec3(1.0), 0.5 + 0.5 * cos(cosVS * 9.0 + vec3(0.0, 2.0, 4.0)), 0.15);
                // Foam shares the volume's Edge Cutoff (_reserved[0]) so the faint
                // low-density foam fringe is clipped like the water density — otherwise
                // it leaves a grey haze the density cutoff can't reach.
                float foamCutoff = max(vol._reserved[0], 1e-4);
                for (int fs = 0; fs < fsteps && foam_T > 0.01; ++fs) {
                    float ft   = tNear + (float(fs) + jitf) * fstep;
                    vec3  fpos = rayOrigin + rayDir * ft;
                    float fd   = sampleTemperature(vol, fpos) * (1.0 / FOAM_TEMP_SCALE);
                    if (fd <= foamCutoff) continue;
                    float a    = 1.0 - exp(-fd * foamOptical * fstep);
                    // Soft edge falloff over [cutoff, 2·cutoff] → clean fade, not a ring.
                    a *= smoothstep(foamCutoff, foamCutoff * 2.0, fd);

                    // ── Production whitewater shading ──────────────────────────
                    // Foam is a bright, high-albedo single-scatter medium. Two cues
                    // sell the look: a FORWARD-SCATTER silver lining (HG phase toward
                    // the light, so backlit crests glow) and a POWDER term (gentle
                    // edge-dark / core-bright self-occlusion). Still scene-lit (lights
                    // + nishita sun + sky ambient) and NO self-shadow, so it never
                    // goes black in scene-light-lit setups.
                    float powder    = mix(0.75, 1.0, gpu_powder_effect(fd, cosVS)); // gentle edge-dark (cosVS/foamPhase hoisted)
                    vec3 Lf = vec3(0.0);
                    if (cam.lightCount > 0u) {
                        int li = clamp(int(floor(rnd(payload.seed) * float(cam.lightCount))),
                                       0, int(cam.lightCount) - 1);
                        LightData lt = lights.l[li];
                        int   lty = int(lt.position.w + 0.5);
                        vec3  ldir; float latten = 1.0;
                        if (lty == 1) {
                            ldir = normalize(lt.direction.xyz);
                        } else {
                            vec3 toL = lt.position.xyz - fpos;
                            float dL = length(toL);
                            ldir = (dL > EPSILON) ? (toL / dL) : vec3(0.0, 1.0, 0.0);
                            if (dL > EPSILON) latten = 1.0 / (dL * dL);
                        }
                        float lph = henyeyGreenstein(dot(rayDir, ldir), 0.35);  // per-light silver lining
                        Lf += float(cam.lightCount) * lt.color.rgb * lt.color.a * latten
                            * (1.0 + lph * 3.0) * powder;
                    }
                    // Bubble-aggregate upgrade: thin-film silver tint (hoisted above).
                    if (worldData.w.mode == 2) {
                        Lf += sampleTransmittanceLUT(fpos, sunDirF)
                            * worldData.w.sunColor * worldData.w.sunIntensity
                            * (1.0 + foamPhase * 4.0) * powder * silver;        // backlit crest glow
                    } else {
                        Lf += worldData.w.sunColor * worldData.w.sunIntensity
                            * (foamPhase * 2.0) * powder * silver;              // sun glow w/o atmosphere LUT
                    }
                    Lf += skyAmb;                       // ambient fill (never fully dark)
                    // MULTIPLE SCATTERING: foam is a packed mass of air bubbles; its
                    // white comes from light diffusing across many interfaces, so thick
                    // foam reads as a bright FILLED white, not a thin wisp. Near-
                    // isotropic boost that saturates with local density.
                    float ms = 1.0 - exp(-fd * 6.0);
                    Lf += 1.2 * ms * (skyAmb + 0.4 * worldData.w.sunColor * worldData.w.sunIntensity);

                    foam_inscatter += foam_T * a * (foamAlbedo * Lf);
                    foam_T *= (1.0 - a);
                }
            }
        }

        // ── Solid geometry inside the domain ────────────────────────────────
        // The volume AABB is hit immediately when the ray is inside it, so the
        // iso branch would refract straight to the water surface and SKIP any
        // solid triangle sitting between the ray origin and that surface (or,
        // on a miss, anywhere in the domain). Probe for a solid in the relevant
        // span; if one is closer, stop and let it render (skipVolumeAABBs) instead of
        // refracting past it. Tints by the water depth in front of it.
        {
            float probeFar = (hitT < 0.0) ? tFar : hitT;
            const uint SOLID_FLAGS = gl_RayFlagsSkipClosestHitShaderEXT
                                   | gl_RayFlagsNoOpaqueEXT;
            // Exclude gas/fog AABBs (0x02) and SurfaceSDF AABBs (0x08), and
            // KEEP transient simulation particles (0x04).
            //
            // ★★★ 0x04 USED TO BE EXCLUDED HERE AND THAT MADE SPLAT SPHERES
            // DISAPPEAR. The exclusion was written when a fluid domain drew
            // EITHER spheres OR an isosurface, so no sphere could ever be inside
            // a liquid volume's AABB. Per-substance representation broke that
            // assumption: one domain now draws some substances as spheres and
            // reconstructs the rest into the isosurface, so the spheres live
            // INSIDE this volume's box. Invisible to this probe, they were
            // handled by whichever branch came next:
            //   • a real surface behind them  -> hand-off fires for THAT surface,
            //     the re-trace skips the AABBs, and the spheres draw correctly.
            //     ("renders fine in front of a solid")
            //   • nothing behind them          -> the miss branch below teleports
            //     the ray to tFar, straight PAST the spheres. ("invisible in
            //     empty space")
            //   • liquid pooled behind them    -> the iso branch shades the water
            //     and the spheres never get a chance. ("shows the dielectric")
            // One excluded bit, three different-looking symptoms, and the only
            // configuration that looked healthy was the one with no spheres at
            // all. ★★ The comment claiming "particles remain visible to primary
            // rays" stayed true and stopped being the whole story: a primary ray
            // that enters a volume never gets back to the particles on its own.
            //
            // ★ The cost argument still holds where it was made: the GAS march's
            // 1+6 nested solid-location probes still exclude 0x04. This is the
            // fluid-surface branch (vol.source_type == 4) and this is ONE
            // terminate-on-first-hit probe per volume hit.
            const uint SOLID_MASK  = 0xF5;
            const uint VOLUME_SOLID_PROBE = 0xC17D5EEDu;
            shadowPayload = vec4(0.0, 0.0, 0.0, uintBitsToFloat(VOLUME_SOLID_PROBE));
            traceRayEXT(topLevelAS, SOLID_FLAGS, SOLID_MASK, 0, 1, 1,
                        rayOrigin, max(1e-4, tNear - 0.002), rayDir, probeFar + 0.002, 1);
            volumeRecordSolidProbe(shadowPayload.z > 0.5);
            if (shadowPayload.z > 0.5) {
                float solidT = shadowPayload.x;
                // Absorption for the water the ray crossed before the solid.
                if (startInside) {
                    float depth = max(0.0, solidT - tNear);
                    payload.attenuation *= exp(-vol.absorption_color * vol.absorption_coefficient * depth);
                    payload.attenuation *= vol.scatter_color;
                }
                // Foam floating in front of the solid still shows; dim the solid
                // behind it by the foam transmittance.
                payload.radiance    += foam_inscatter;
                payload.attenuation *= foam_T;
                // Continue straight to the solid; skip volume AABBs so the
                // triangle closesthit fires on the next trace.
                // ★ Re-trace from the ORIGINAL origin, with no step-back epsilon.
                // A fixed 0.01 offset is an absolute length applied to a distance
                // that scales with the scene: where it fails to land in FRONT of
                // the surface, the re-traced ray strikes the BACK face, closesthit
                // face-forwards the normal, and the wall shades as if lit from
                // inside — flipped normals in the debug view, black in beauty, and
                // only ever inside an active volume. No epsilon is needed: the ray
                // reached this volume because nothing was closer than the box, and
                // the handoff mask removes the box, so the nearest hit from the
                // same origin is that surface at its true distance and facing.
                //
                // ★ That last sentence used to say "IS that surface", and it was
                // the bug. It only holds for geometry the probe above could SEE.
                // With gl_RayFlagsSkipAABBEXT the re-trace also erased everything
                // the probe was blind to, so anything nearer than solidT was
                // dropped twice. The mask now clears only the volume bits
                // (RT_MASK_VOLUME_HANDOFF): if a splat sphere really is closer, it
                // wins the re-trace on its own and no extra probe is paid for it.
                payload.scatterOrigin = rayOrigin;
                payload.scatterDir    = rayDir;
                payload.skipVolumeAABBs     = true;
                payload.scattered     = true;
                payload.bounceType    = 0u;
                payload.volumeTraversalMask = plSolidHandoffWord(payload.volumeTraversalMask, solidT);
                return;
            }
        }

        if (hitT < 0.0) {
            // Ray missed the iso-surface AND no solid — pass through, but any
            // airborne foam/spray in this segment still adds light + occlusion.
            payload.radiance    += foam_inscatter;
            payload.attenuation *= foam_T;
            payload.scatterOrigin = rayOrigin + rayDir * (tFar + 0.002);
            payload.scatterDir    = rayDir;
            payload.scattered     = true;
            payload.bounceType    = BOUNCE_TRANSPARENT;
            return;
        }

        vec3 hitPos = rayOrigin + rayDir * hitT;

        // Central-difference gradient on the density field for the surface
        // normal. h = voxel_size keeps the stencil aligned with the proxy
        // band thickness (band ≈ 0.5*voxel by default), so the gradient is
        // numerically well-conditioned.
        float h = max(0.001, vol.voxel_size);
        float sxp = sampleIsoField(vol, hitPos + vec3(h, 0.0, 0.0), vdbBuf, vdbMapH, vdbAcc);
        float sxm = sampleIsoField(vol, hitPos - vec3(h, 0.0, 0.0), vdbBuf, vdbMapH, vdbAcc);
        float syp = sampleIsoField(vol, hitPos + vec3(0.0, h, 0.0), vdbBuf, vdbMapH, vdbAcc);
        float sym = sampleIsoField(vol, hitPos - vec3(0.0, h, 0.0), vdbBuf, vdbMapH, vdbAcc);
        float szp = sampleIsoField(vol, hitPos + vec3(0.0, 0.0, h), vdbBuf, vdbMapH, vdbAcc);
        float szm = sampleIsoField(vol, hitPos - vec3(0.0, 0.0, h), vdbBuf, vdbMapH, vdbAcc);
        vec3 grad = vec3(sxp - sxm, syp - sym, szp - szm);
        float gradLen = length(grad);

        // Foam / whitewater: SDF Laplacian = surface curvature. High |curvature|
        // = wave crest / breaking edge / splash -> whiten. Reuses the 6 gradient
        // samples + the centre (≈iso), so it's nearly free.
        float foam_strength = clamp(vol._ext_reserved[2], 0.0, 1.0);
        if (foam_strength > 1e-3) {
            float dc = sampleIsoField(vol, hitPos, vdbBuf, vdbMapH, vdbAcc);
            float lap = abs((sxp + sxm + syp + sym + szp + szm) - 6.0 * dc);
            float foam = foam_strength * smoothstep(0.15, 0.7, lap);
            // Bright white whitewater, lit by the current throughput.
            payload.radiance += payload.attenuation * foam * vec3(0.9);
        }
        // ★ A degenerate COARSE gradient is not proof that the crossing is fake.
        // The ray march already measured a threshold sign change in this interval;
        // retry the local derivative before choosing a conservative fallback.
        // Density intentionally moves the 0.5 crossing through the proxy band
        // and makes the rendered body fuller. Near a domain-side wall or a thin
        // accumulated sheet, the one-voxel stencil can put both taps on the same
        // plateau even though the march above measured a real crossing. Retry
        // only that failed normal at quarter- and sixteenth-voxel scales. Normal
        // hits pay nothing extra; a difficult side wall pays at most 12 samples.
        for (int retry = 0; retry < 2 && gradLen <= 1e-4; ++retry) {
            h *= 0.25;
            sxp = sampleIsoField(vol, hitPos + vec3(h, 0.0, 0.0), vdbBuf, vdbMapH, vdbAcc);
            sxm = sampleIsoField(vol, hitPos - vec3(h, 0.0, 0.0), vdbBuf, vdbMapH, vdbAcc);
            syp = sampleIsoField(vol, hitPos + vec3(0.0, h, 0.0), vdbBuf, vdbMapH, vdbAcc);
            sym = sampleIsoField(vol, hitPos - vec3(0.0, h, 0.0), vdbBuf, vdbMapH, vdbAcc);
            szp = sampleIsoField(vol, hitPos + vec3(0.0, 0.0, h), vdbBuf, vdbMapH, vdbAcc);
            szm = sampleIsoField(vol, hitPos - vec3(0.0, 0.0, h), vdbBuf, vdbMapH, vdbAcc);
            grad = vec3(sxp - sxm, syp - sym, szp - szm);
            gradLen = length(grad);
        }
        // If the sparse-grid quantisation defeats both finer stencils, the
        // crossing direction still tells us the density derivative's sign:
        // outside->inside rises along the ray, inside->outside falls. Preserve
        // that measured entry/exit orientation instead of deleting the BSDF
        // surface with a transparent pass-through.
        if (gradLen <= 1e-4) {
            grad = startInside ? -rayDir : rayDir;
            gradLen = 1.0;
        }
        // Density increases TOWARD fluid interior, so -gradient points OUT of
        // the surface (toward the less-dense / air side).
        vec3 N = normalize(-grad);

        // ── Rough dielectric event (Fresnel importance-sampled). ────────────
        // Orient the geometric normal against the incoming ray.
        if (dot(rayDir, N) > 0.0) N = -N;

        // GGX roughness: jitter the normal inside a microfacet lobe so both the
        // reflection AND the refraction blur with surface_roughness
        // (_ext_reserved[1]). 0 = mirror-smooth still water.
        float roughness = clamp(vol._ext_reserved[1], 0.0, 1.0);

        // -- Scene material on the liquid surface -------------------------
        // vol.iso_material_index is the 1-based scene material the domain binds
        // for its surface. Deliberately NOT _reserved[1]: that slot means "run
        // the volume material VM with this program" and is gated on the graph
        // driving volume closure slots, so a plain Principled material resolves
        // to none there and would never have reached this branch.
        // When one is bound, the isosurface is shaded by the SAME lobe
        // selection a triangle gets: clearcoat, metal, transmission and the
        // random-walk SSS. That is what makes molten glass, lava, mud and
        // chocolate ordinary material settings instead of shader special
        // cases -- this branch used to be a bare Fresnel + Beer-Lambert
        // dielectric, which is why a dense material could only ever read as
        // dark glass.
        // No material bound -> fall through to that original path, so scenes
        // authored before this change render exactly as they did.
        // ── Which material(s) THIS point is made of ──────────────────────────
        // With a composition field published, the domain's single material is
        // only the fallback: each point resolves to the substances actually
        // present there. Without one, isoMatSlot is exactly vol.iso_material_index
        // and everything below is unchanged.
        float isoMatSlot = vol.iso_material_index;
        float isoMatSlotB = isoMatSlot;
        float isoMixB = 0.0;
        sampleComposition(vol, hitPos, isoMatSlot, isoMatSlotB, isoMixB);

        // Publish a real primary-surface record. Without DONE, raygen treated
        // this visible SDF as sky for motion/history even though its AOV colour
        // and normal were written below. The surface has no triangle motion ID.
        if ((payload.primaryMeta & PL_PRIMARY_DONE) == 0u) {
            float primarySlot = isoMixB >= 0.5 ? isoMatSlotB : isoMatSlot;
            uint primaryMaterial = primarySlot > 0.5
                ? uint(max(primarySlot, 1.0) - 1.0) : PL_MATID_MASK;
            payload.primaryMeta = (payload.primaryMeta & PL_DISP_MASK) |
                PL_PRIMARY_DONE | (primaryMaterial & PL_MATID_MASK);
            payload.primaryARG = packHalf2x16(vol.scatter_color.rg);
            payload.primaryABT = packHalf2x16(vec2(vol.scatter_color.b, 1.0));
            payload.primaryNrm = plPackNormal(N);
        }

        if (isoMatSlot > 0.5 || isoMatSlotB > 0.5) {
            // ★ A slot of 0 means "the built-in dielectric" for THAT substance,
            // which is a real authoring choice — one substance a full Principled
            // BSDF, another plain refractive liquid, in the same body. Clamped
            // to a valid index here and given zero weight below, so the mixture
            // degrades toward the material that IS bound instead of indexing
            // material -1 and reading whatever precedes the table.
            uint isoMatIdx  = uint(max(isoMatSlot,  1.0) - 1.0);
            uint isoMatIdxB = uint(max(isoMatSlotB, 1.0) - 1.0);
            float wB = (isoMatSlotB > 0.5) ? isoMixB : 0.0;
            if (isoMatSlot <= 0.5) wB = 1.0;   // A is the dielectric; B carries it

            Material    im  = materials.m[isoMatIdx];
            MaterialExt imx = materialsExt.m[isoMatIdx];

            // ═══════════════════════════════════════════════════════════════
            // BLEND THE PARAMETERS, SHADE ONCE.
            // ═══════════════════════════════════════════════════════════════
            // ★★★ Not "shade twice and mix the results". Principled's parameter
            // space is built to be interpolated, and one shading call keeps the
            // lobe selection, the SSS walk and the refraction event SINGLE — a
            // point in a mixture refracts once, through one interface, with an
            // intermediate IOR. Mixing two shaded results would give two
            // refractions and two entry events at one surface point, which reads
            // as a ghosted double image exactly where the substances meet.
            //
            // ★ Only the fields this branch actually consumes are blended.
            // Blending texture IDs would be meaningless (an index is a name, not
            // a quantity), so the dominant material's textures win: at wB > 0.5
            // the pair is swapped below instead.
            if (wB > 0.001) {
                if (wB > 0.5) {
                    // Dominant side owns the texture bindings and the flags.
                    // No temporary needed: the old A is re-fetched by index on
                    // the line below, not read out of `im`.
                    im  = materials.m[isoMatIdxB];
                    imx = materialsExt.m[isoMatIdxB];
                    isoMatIdx = isoMatIdxB;
                    wB = 1.0 - wB;
                    isoMatIdxB = uint(max(isoMatSlot, 1.0) - 1.0);
                }
                Material other = materials.m[isoMatIdxB];
                im.albedo_r     = mix(im.albedo_r,     other.albedo_r,     wB);
                im.albedo_g     = mix(im.albedo_g,     other.albedo_g,     wB);
                im.albedo_b     = mix(im.albedo_b,     other.albedo_b,     wB);
                im.roughness    = mix(im.roughness,    other.roughness,    wB);
                im.metallic     = mix(im.metallic,     other.metallic,     wB);
                im.transmission = mix(im.transmission, other.transmission, wB);
                im.ior          = mix(im.ior,          other.ior,          wB);
                im.opacity      = mix(im.opacity,      other.opacity,      wB);
                im.emission_r   = mix(im.emission_r,   other.emission_r,   wB);
                im.emission_g   = mix(im.emission_g,   other.emission_g,   wB);
                im.emission_b   = mix(im.emission_b,   other.emission_b,   wB);
                im.emission_strength =
                    mix(im.emission_strength, other.emission_strength, wB);
                im.clearcoat  = mix(im.clearcoat,  other.clearcoat,  wB);
                im.clearcoat_roughness =
                    mix(im.clearcoat_roughness, other.clearcoat_roughness, wB);
                im.subsurface_amount =
                    mix(im.subsurface_amount, other.subsurface_amount, wB);
                im.translucent = mix(im.translucent, other.translucent, wB);
                im.specular    = mix(im.specular,    other.specular,    wB);
                im.normal_strength =
                    mix(im.normal_strength, other.normal_strength, wB);
            }

            // ★ ONE exit push for EVERY lobe below — see the long note at the
            // bottom of this block. Hoisted here because the thin-shell and
            // resin branches return early and need it just as much as the
            // Principled path does; a ULP offset re-hits the same surface.
            float exitPush = max(0.003, vol.voxel_size);

            // ── GEOMETRIC vs SHADING normal ──────────────────────────────────
            // Ng is the level-set gradient: the real orientation of the surface.
            // N becomes the SHADING normal once a normal map perturbs it.
            //
            // ★★ THE SPLIT IS NOT COSMETIC. Everything that reasons about WHERE
            // the surface is must keep using Ng — above all seatOutsideBand,
            // which decides from dot(dir, N) whether a scattered ray is heading
            // out of the body or into it. Feed it a perturbed normal and that
            // test flips near grazing angles: an outgoing ray gets classified as
            // inward, takes the probe path, and either self-occludes or steps
            // through a thin wall. The result is light leaking through liquid
            // exactly where the bump detail is strongest, which reads as a
            // normal-map artefact and is really a band-logic one.
            //
            // Shading uses N; band, refraction geometry and ray re-seating use Ng.
            vec3 Ng = N;
            // ★ ONE projection draw for this hit, shared by the normal map and
            // every texture channel below. Drawn here, before the first use, so
            // there is exactly one place it can be got wrong. Independent draws
            // per channel would sample albedo from one plane and roughness from
            // another at the same surface point — which converges to a blend of
            // two different materials and looks like a plausible material rather
            // than like a bug.
            float triXi = rnd(payload.seed);
            if (im.normal_tex > 0u) {
                vec2 nmScale = vec2(abs(im.uv_scale_x) > 1e-6 ? im.uv_scale_x : 1.0,
                                    abs(im.uv_scale_y) > 1e-6 ? im.uv_scale_y : 1.0);
                // Perturbed HERE, before SurfaceSample is filled and before the
                // direct-light block reads N. Doing it later would light the
                // surface with the flat normal and bounce off the bumped one —
                // "the normal map only affects reflections", which is a report
                // that sends you looking at the map instead of at the ordering.
                N = triplanarNormal(im.normal_tex,
                                    materialAnchor(vol, hitPos), Ng,
                                    nmScale, vec2(im.uv_offset_x, im.uv_offset_y),
                                    max(im.normal_strength, 0.0), triXi);
            }

            // Depth absorption and whitewater are properties of the LIQUID
            // BODY, not of the surface lobe, so they apply whichever branch is
            // taken below. Hoisted out of the Principled path for that reason.
            if (startInside) {
                float depth = max(0.0, hitT - tNear);
                payload.attenuation *= exp(-(vol.absorption_color * vol.absorption_coefficient) * depth);
            }
            payload.radiance    += foam_inscatter;
            payload.attenuation *= foam_T;

            // ── THIN-SHELL FILM (bubble) ─────────────────────────────────────
            // Soap foam / champagne head sitting ON the liquid. A thin shell is
            // entered and exited parallel, so there is no net refraction: the
            // ray either Fresnel-reflects off the film (bright silver rim,
            // strongest at grazing) or passes STRAIGHT through. Mirrors the
            // triangle branch in closesthit.rchit — same fields, same look, so
            // a bubble material reads identically on a mesh and on a liquid.
            //
            // Dispatched BEFORE the Principled fill because on the triangle
            // side it is likewise a separate type branch, not a lobe: there is
            // no roughness/metallic/SSS to build for a film.
            if ((im.flags & MAT_FLAG_BUBBLE) != 0u) {
                float cosTb = min(abs(dot(rayDir, N)), 1.0);
                float bio   = (imx.bubble_ior > 1.0001) ? imx.bubble_ior : 1.33;
                float r0b   = (1.0 - bio) / (1.0 + bio); r0b = r0b * r0b;
                float fresb = r0b + (1.0 - r0b) * pow(1.0 - cosTb, 5.0);
                vec3 bDir, bAtt;
                bool bPassThrough = false;
                if (rnd(payload.seed) < fresb) {
                    bDir = reflect(rayDir, N);              // bright Fresnel rim
                    if (imx.bubble_film > 1e-3) {
                        float opd = imx.bubble_film * (1.0 / max(cosTb, 0.15));
                        bAtt = vec3(0.55 + 0.45 * cos(opd * TWO_PI),
                                    0.55 + 0.45 * cos(opd * TWO_PI + 2.0944),
                                    0.55 + 0.45 * cos(opd * TWO_PI + 4.1888));
                    } else {
                        bAtt = vec3(1.0);
                    }
                } else {
                    bDir = rayDir;                          // straight through
                    bAtt = vec3(0.85) + 0.15 * vec3(im.albedo_r, im.albedo_g, im.albedo_b);
                    bPassThrough = true;
                }
                // ★ Both lobes leave along bDir, pushed clear of the level-set
                // band. The triangle version offsets a few ULPs off the plane;
                // here that restarts INSIDE the band — and for the pass-through
                // lobe (bDir == rayDir) that means the ray immediately re-hits
                // the same film and never leaves the surface at all.
                // ★ The push must not STEP OVER real geometry.
                //
                // exitPush is scaled to the level-set band, i.e. to voxel_size —
                // centimetres at production settings. The pass-through lobe
                // travels straight INTO whatever sits behind the film, so any
                // solid closer than one voxel is simply jumped past: the ray
                // restarts beyond it and the surface behind never renders. That
                // is the reported "thin shell skips the surface behind", and it
                // is specific to this lobe — the reflection lobe leaves the
                // surface outward, and the Principled path's own scatter lobes
                // do too.
                //
                // So probe just the push segment first. It is a few centimetres
                // long and terminates on first hit, so it costs far less than
                // the march that produced this hit. If a solid is inside it,
                // hand the ray over exactly the way the entry-side solid probe
                // above does: continue from the hit point with the volume AABBs
                // skipped, so the triangle closest-hit fires on that surface at
                // its true distance and true facing.
                bool bHandOff = false;
                payload.scatterDir    = bDir;
                payload.scatterOrigin = seatOutsideBand(hitPos, bDir, Ng, exitPush, bHandOff);
                payload.skipVolumeAABBs     = bHandOff;
                payload.attenuation  *= bAtt;
                payload.scattered     = true;
                // ★ The two lobes must be tagged DIFFERENTLY, and this is what
                // produced the black patches on the shell.
                //
                // A straight-through crossing is not a scattering event — the
                // ray leaves along the direction it arrived on. raygen only
                // treats a pass as free when the direction is unchanged AND
                // either the attenuation is exactly 1 or the tag is
                // BOUNCE_TRANSPARENT (isTransparentPass, raygen.rgen). The film
                // tints by 0.85..1.0, so the attenuation test can never pass —
                // tagged SPECULAR, every crossing spent a full GI bounce.
                //
                // A mesh bubble survives that: two crossings. A foam shell on an
                // isosurface does not — the level-set band is crossed many times
                // over, the budget runs out before the ray reaches the ground
                // behind, and raygen breaks the loop. Those paths return only
                // what they had accumulated, which is why the holes appeared
                // exactly where the shell is thickest and why the surface behind
                // was never reached.
                //
                // The reflection lobe DOES redirect the ray, so it stays a real
                // specular bounce and keeps costing one.
                payload.bounceType    = bPassThrough ? BOUNCE_TRANSPARENT
                                                     : BOUNCE_SPECULAR;
                // NOTE: the triangle branch clears payload.radiance here. Do NOT
                // copy that — this shader ACCUMULATES radiance (+=), and the
                // foam in-scatter for this hit was already added above.
                payload.primaryARG = packHalf2x16(vec2(im.albedo_r, im.albedo_g));
                payload.primaryABT = packHalf2x16(vec2(0.0, 1.0));   // aerial parity
                payload.primaryNrm = plPackNormal(N);
                return;
            }

            SurfaceSample ss = defaultSurfaceSample();
            ss.P      = hitPos;
            ss.N      = N;
            ss.rayDir = rayDir;
            ss.albedo    = vec3(im.albedo_r, im.albedo_g, im.albedo_b);
            // The MATERIAL's roughness wins. Binding a material means the
            // liquid is shaded by that material, full stop — having one of its
            // parameters silently overridden by a panel elsewhere is the kind
            // of split authority that makes a control look broken. The domain's
            // Surface Roughness governs the built-in dielectric below, and the
            // panel greys it out while a material is bound so the split is
            // visible rather than inferred.
            // This is the material's SCALAR roughness; the tri-planar block
            // below OVERRIDES it when a roughness texture is bound. (This note
            // used to claim a roughness texture could not reach an isosurface
            // at all. That stopped being true one screen further down, and a
            // comment asserting a capability does not exist is worse than no
            // comment — nobody goes looking for a feature they have been told
            // is impossible.)
            ss.roughness = clamp(im.roughness, 0.0, 1.0);
            ss.metallic  = clamp(im.metallic, 0.0, 1.0);
            ss.specular  = im.specular;
            ss.clearcoat          = clamp(im.clearcoat, 0.0, 1.0);
            ss.clearcoatRoughness = clamp(im.clearcoat_roughness, 0.001, 1.0);
            ss.clearcoatIridescence   = imx.clearcoat_iridescence;
            ss.clearcoatFilmThickness = imx.clearcoat_film_thickness;
            ss.translucent        = clamp(im.translucent, 0.0, 1.0);
            ss.subsurfaceAmount   = clamp(im.subsurface_amount, 0.0, 1.0);
            ss.subsurfaceColor    = max(vec3(imx.subsurface_r, imx.subsurface_g, imx.subsurface_b), vec3(0.001));
            ss.subsurfaceRadius   = max(vec3(imx.subsurface_radius_r, imx.subsurface_radius_g, imx.subsurface_radius_b), vec3(0.001));
            ss.subsurfaceScale    = max(imx.subsurface_scale, 0.001);
            ss.subsurfaceAnisotropy = clamp(imx.subsurface_anisotropy, -0.99, 0.99);
            ss.subsurfaceIor        = max(imx.subsurface_ior, 1.001);
            ss.subsurfaceMethod     = (imx.sss_method > 0.5) ? 1u : 0u;
            ss.subsurfaceMaxSteps   = int(imx.sss_walk_max_steps + 0.5);

            // ── Tri-planar textures ──────────────────────────────────────────
            // This is what lets a texture reach the liquid at all. Before it,
            // only scalar material values did, and assigning an image to a
            // fluid surface silently did nothing.
            //
            // Roughness/metallic go through the SHARED packed-ORM policy, so a
            // texture authored for a mesh reads the same channel here. Writing
            // a second channel rule for the isosurface would eventually pick a
            // different one and shade the liquid as metal for no visible reason.
            {
                // uv_scale doubles as world units per tile here; 0 would
                // collapse every sample onto one texel, so treat it as unset.
                vec2 texScale = vec2(abs(im.uv_scale_x) > 1e-6 ? im.uv_scale_x : 1.0,
                                     abs(im.uv_scale_y) > 1e-6 ? im.uv_scale_y : 1.0);
                vec2 texOffset = vec2(im.uv_offset_x, im.uv_offset_y);

                // ★ THE ANCHOR. Everything below projects from `texAnchor`
                // rather than from the world hit position, so the pattern is
                // attached to the liquid instead of to the room. Sampled once
                // and shared by all four maps: two maps resolving the anchor
                // separately would be two chances for them to disagree, and a
                // normal map registered against a different coordinate than its
                // albedo is a class of wrongness that reads as "the lighting is
                // off" rather than as a coordinate bug.
                vec3 texAnchor = materialAnchor(vol, hitPos);
                // Blend weights stay on the WORLD normal. They only decide which
                // of the three projections dominates, and deriving a material-
                // space normal would need the coordinate field's Jacobian for a
                // difference that is invisible while the mapping is near
                // identity — which is the regime this is useful in. It IS the
                // reason a violently stretched region can show a seam shift; see
                // the stretch limit noted on FluidParticles::uvw.

                if (im.albedo_tex > 0u) {
                    // Replaces, matching the surface path (the texture is the
                    // base colour there, not a tint on it).
                    ss.albedo = triplanarTexel(im.albedo_tex, texAnchor, N, texScale, texOffset, triXi).rgb;
                }
                if (im.roughness_tex > 0u) {
                    ss.roughness = clamp(samplePackedRoughness(
                        triplanarTexel(im.roughness_tex, texAnchor, N, texScale, texOffset, triXi),
                        0.0, im.flags), 0.0, 1.0);
                }
                if (im.metallic_tex > 0u) {
                    ss.metallic = clamp(samplePackedMetallic(
                        triplanarTexel(im.metallic_tex, texAnchor, N, texScale, texOffset, triXi),
                        im.flags), 0.0, 1.0);
                }
                // Normal maps are handled EARLIER, where Ng/N are split — not
                // here with the other maps. They have to be: this block runs
                // after SurfaceSample is filled and after the direct-light
                // block, both of which read the shading normal, so perturbing
                // it here would light the surface flat and bounce it bumped.
            }

            // ── EMISSION ─────────────────────────────────────────────────────
            // SurfaceSample is a REFLECTION parameter set: it has no slot for
            // emission, transmission or opacity. So a bound material's Emission
            // never reached the liquid — the branch simply never read the field.
            // Reported as "emission worked yesterday and stopped": binding a
            // material moves shading into this branch, and this branch had no
            // emission, so the glow disappeared the moment a material was bound.
            //
            // Added as LOCAL radiance, matching the foam in-scatter convention
            // right above: raygen multiplies payload.radiance by the throughput
            // it already carries, so pre-multiplying here would square it.
            //
            // The thin-shell branch above deliberately returns before this, the
            // same way the triangle branch does — an emissive bubble material
            // would wash the film out.
            {
                vec3 isoEmission = vec3(im.emission_r, im.emission_g, im.emission_b)
                                 * max(im.emission_strength, 0.0);
                if (im.emission_tex > 0u) {
                    // Emission texture is authoritative for colour, exactly as on
                    // the surface path; strength stays the material's.
                    vec2 emScale = vec2(abs(im.uv_scale_x) > 1e-6 ? im.uv_scale_x : 1.0,
                                        abs(im.uv_scale_y) > 1e-6 ? im.uv_scale_y : 1.0);
                    // Same anchor as the maps above — resolved again here
                    // because this block sits outside their scope, NOT because
                    // it is a different coordinate.
                    isoEmission = triplanarTexel(im.emission_tex,
                                                 materialAnchor(vol, hitPos), N,
                                                 emScale, vec2(im.uv_offset_x, im.uv_offset_y),
                                                 triXi).rgb
                                * max(im.emission_strength, 0.0);
                }
                payload.radiance += isoEmission;
            }

            // ── Glass / resin split, drawn ONCE ──────────────────────────────
            // Declared here because BOTH branches below read it and the draw
            // must be the same one: rolling separately would let a material
            // take the glass lobe and the resin coat in the same hit.
            //
            // The rule is the triangle's, kept verbatim: a material carrying
            // BOTH transmission and interior depth is an amber/jade body, not a
            // coated one. Picking the coat for it would silently repaint every
            // existing scene with such a material bound to a liquid.
            // Resin coat absorption, hoisted so the NEE block below can apply it
            // to DIRECT light too. The triangle path does exactly this
            // (`resinActive/resinDensity/resinExt` carried into its NEE); without
            // it a thick tinted coat would dim the indirect bounce and leave the
            // direct term at full strength, i.e. the coat would look thinner the
            // brighter the key light is.
            bool  neeResinActive  = false;
            vec3  neeResinExt     = vec3(0.0);
            float neeResinDensity = 0.0;

            // ★★★ TWO DIFFERENT TRANSMISSION VALUES, AND THEY MUST STAY APART.
            //
            // `isoTrans` SELECTS A LOBE here. On the triangle path the identically
            // named local does NOT — SurfaceSample carries no transmission field,
            // so there it only ever reaches evaluate_brdf_gl. That asymmetry is
            // invisible when reading the two files side by side and it is exactly
            // what a line copied from one into the other walks into.
            //
            // Copying the mesh's `opacity < 0.99 -> transmission = 1 - opacity`
            // into the value below did precisely that: every material authored
            // with opacity even slightly under 1 — which is most SKIN, WAX, JADE,
            // MARBLE and MILK, i.e. exactly the subsurface materials — started
            // taking the GLASS lobe and returning through scatterGlass before
            // scatterPrincipled (where the subsurface lobe lives) ever ran.
            //
            // Two symptoms, one cause, and neither of them names it:
            //   - "SSS broke"          — the subsurface lobe is never reached.
            //   - "textures stopped
            //      having any effect"  — scatterGlass ignores albedo/roughness
            //                           detail, so the tri-planar maps that DID
            //                           land stop showing.
            // Both come back together when the lobe selector is restored.
            float isoTransAuthored = clamp(im.transmission, 0.0, 1.0);

            // LOBE SELECTION: authored transmission ONLY. An opaque material must
            // never be routed into refraction because someone dialled its opacity
            // to 0.95 — that is a coverage statement, not "this is glass".
            bool  takeGlassLobe = (isoTransAuthored > 0.01) &&
                                  (rnd(payload.seed) < isoTransAuthored);

            // SHADING ONLY: scalar opacity thins the diffuse lobe in the BRDF
            // evaluation, which is all the mesh's copy of this rule actually
            // does. The isosurface read im.opacity nowhere before, so a
            // semi-transparent material still looked fully solid on a liquid —
            // that part of the fix stands, it just may not touch lobe choice.
            //
            // The TEXTURE half of opacity is elsewhere entirely: it cut geometry
            // back in isoAlphaOffset.
            float isoTrans = isoTransAuthored;
            if (im.opacity < 0.99 && im.metallic < 0.1 && isoTransAuthored < 0.01) {
                isoTrans = clamp(1.0 - im.opacity, 0.0, 1.0);
            }

            // ── TRANSMISSION (real refraction) ───────────────────────────────
            // ★ This is the lobe whose absence made "transmission does nothing"
            // on the liquid. scatterPrincipled is reflection-only, so binding a
            // material actually made the surface LESS capable than leaving it
            // unbound: the built-in dielectric below refracts, the material path
            // did not. Nothing about the data was wrong.
            //
            // Uses the SAME scatterGlass the mesh uses — it already lives in the
            // shared bsdf_scatter module — so IOR, roughness, dispersion and the
            // resin-tinted interior behave identically on a mesh and on a liquid.
            //
            // ★ frontFace comes from startInside, which is SDF-INHERITED data:
            // the march already established which side of the level set the ray
            // began on. A mesh has to read a winding order for this; here it is
            // a measurement, and it is the reason the lobe can be trusted on a
            // surface that has no consistent orientation of its own.
            if (takeGlassLobe) {
                float isoIor = (im.ior > 1.0001) ? im.ior : 1.33;
                // Same absence the triangle glass lobe had: scatterGlass returns
                // without NEE, and scene lights carry no geometry in the TLAS, so
                // the mirror lobe of a transmissive material bound to a liquid had
                // no light to find. The NEE block further down never runs on this
                // draw (we return below), which is precisely why it went unnoticed.
                // Front face only, and the band's own exit push owns the origin.
                if (!startInside) {
                    addDielectricDirectLighting(hitPos, Ng, Ng, N, rayDir,
                                                isoIor, ss.roughness,
                                                vec3(0.0), 0.0, exitPush,
                                                payload.seed);
                }
                scatterGlass(hitPos,
                             /*macroNormal  */ Ng,  // level-set gradient: the real surface
                             /*shadingNormal*/ N,   // normal-mapped, if a map is bound
                             /*frontFace    */ !startInside,
                             rayDir, ss.albedo, isoIor, ss.roughness,
                             imx.transmission_density,
                             clamp(vec3(imx.resin_color_r, imx.resin_color_g, imx.resin_color_b),
                                   vec3(0.0), vec3(1.0)),
                             im.dispersion, payload.seed);
                // ★ Same band re-seat every other lobe here needs — and this one
                // NEEDS the probe: the refracted direction continues INTO the
                // body, so on a thin wall (the side of a pool) a blind voxel
                // push clears the liquid and the surface behind it together.
                bool gHandOff = false;
                payload.scatterOrigin =
                    seatOutsideBand(hitPos, payload.scatterDir, Ng, exitPush, gHandOff);
                payload.skipVolumeAABBs = gHandOff;
                payload.primaryARG = packHalf2x16(ss.albedo.rg);
                payload.primaryABT = packHalf2x16(vec2(ss.albedo.b, 1.0));
                payload.primaryNrm = plPackNormal(N);
                return;
            }

            // ── RESIN COAT ───────────────────────────────────────────────────
            // A refractive ABSORBING layer over an opaque base: lacquered/candied
            // surfaces, honey and syrup skins, anything with suspended interior
            // structure. Fresnel-splits the surface — the reflection lobe is the
            // glossy coat, everything else reaches the base, which is tinted by
            // the coat absorption and then shaded as an ordinary opaque surface.
            //
            // ★ The coat thickness is AUTHORED (transmission_density), NOT
            // measured from the field. This is a skin ON the surface, not the
            // body of the liquid — the body's own depth absorption already ran
            // above, and folding the two together would double-count. That is
            // also why this matches the triangle path's thickness model exactly
            // rather than using the (better) real hitT - tNear distance: same
            // parameter, same look, on a mesh and on a liquid alike.
            //
            // The glass/resin split is drawn once, above — on the draws where
            // the glass lobe won we already returned through scatterGlass, so
            // reaching here means this hit is the coat.
            if (imx.transmission_density > 1e-4 && !takeGlassLobe) {
                float effIor = max((im.ior > 0.01) ? im.ior : 1.45, 1.45);
                float cosTr  = clamp(dot(-rayDir, N), 0.0, 1.0);
                float fresr  = schlickFresnel(cosTr, effIor);
                // Coat gloss is the resin LAYER's own roughness, independent of
                // the base (which is forced rough below).
                float resinRough = clamp(imx.resin_roughness, 0.0, 1.0);
                if (rnd(payload.seed) < fresr) {
                    vec3 Vr = -rayDir;
                    vec3 refl;
                    if (resinRough < 0.02) {
                        refl = reflect(rayDir, N);
                    } else {
                        // ggxSampleVNDF returns the REFLECTED direction directly.
                        float alphaR = max(resinRough * resinRough, 1e-4);
                        refl = ggxSampleVNDF(N, Vr, alphaR, rnd(payload.seed), rnd(payload.seed));
                        if (dot(refl, N) <= 0.0) refl = reflect(rayDir, N);
                    }
                    refl = normalize(refl);
                    payload.scatterDir    = refl;
                    // Outward-going, so the helper's probe is skipped and this
                    // is just the band push — routed through it anyway so every
                    // lobe in this branch obeys ONE rule. Divergence between
                    // these sites is what produced the thin-wall bug.
                    bool rHandOff = false;
                    payload.scatterOrigin =
                        seatOutsideBand(hitPos, refl, Ng, exitPush, rHandOff);
                    payload.skipVolumeAABBs = rHandOff;
                    payload.scattered     = true;
                    payload.bounceType    = BOUNCE_RESIN;   // raygen's resin budget
                    payload.primaryARG = packHalf2x16(ss.albedo.rg);
                    payload.primaryABT = packHalf2x16(vec2(ss.albedo.b, 1.0));
                    payload.primaryNrm = plPackNormal(N);
                    return;
                }
                // Reached the base. Per-channel absorption ∝ (1 - tint), so a
                // warm tint passes its own hue and swallows the complement
                // instead of darkening everything.
                vec3  ct    = clamp(vec3(imx.resin_color_r, imx.resin_color_g, imx.resin_color_b),
                                    vec3(0.0), vec3(1.0));
                float ctMax = max(ct.r, max(ct.g, ct.b));
                vec3  ext   = (vec3(1.0) - ct) * 1.35 + vec3(0.22 * (1.0 - ctMax));
                neeResinActive  = true;
                neeResinExt     = ext;
                neeResinDensity = imx.transmission_density;
                float cosVr = max(abs(cosTr), 0.25);

                vec3 Tdir = refract(rayDir, N, 1.0 / effIor);
                if (dot(Tdir, Tdir) < 1e-6) Tdir = rayDir;   // TIR fallback
                Tdir = normalize(Tdir);

                // NOTE: the triangle path also parallax-offsets the base ALBEDO
                // TEXTURE lookup along the refracted lateral travel. There is no
                // texture to offset here — the isosurface has no UVs — so the
                // base albedo is the material's scalar Base Color.
                bool resinHasInclusions = (imx.resin_inclusion > 0.001 ||
                                           imx.resin_dirt > 0.001 ||
                                           imx.resin_shard > 0.001);
                if (resinHasInclusions) {
                    // One light direction to shade the interior specks by (no
                    // shadow rays; the march is purely procedural). This uses
                    // volSampleLight — the surface shader's pick_smart_light_gl
                    // does NOT exist in this stage.
                    vec3 resinLightDir = N;
                    if (cam.lightCount > 0u) {
                        uint li = min(uint(rnd(payload.seed) * float(cam.lightCount)),
                                      cam.lightCount - 1u);
                        vec3 wi_; float d_; float a_; bool invSq_;
                        if (volSampleLight(lights.l[li], hitPos,
                                           rnd(payload.seed), rnd(payload.seed),
                                           wi_, d_, a_, invSq_) && dot(wi_, wi_) > 1e-8) {
                            resinLightDir = normalize(wi_);
                        }
                    }
                    // ANCHOR. The triangle path offers object space so the
                    // interior travels with the mesh. There is no object here —
                    // the analogue is the MATERIAL coordinate, which is what the
                    // liquid has instead of an object: the inclusions are now
                    // carried by the fluid rather than left standing in the tank
                    // while it pours past them.
                    //
                    // ★ The ORIGIN goes through the shared anchor, so the resin
                    // interior sits in whatever space the domain selected —
                    // material, domain-local or world — and cannot end up in a
                    // different one from the albedo painted over it.
                    //
                    // ★★ MAT_FLAG_RESIN_OBJ_SPACE IS DELIBERATELY NOT APPLIED TO
                    // THE ORIGIN HERE, and that is a behaviour change worth being
                    // explicit about. That flag means "object space" — on a mesh,
                    // the object is the mesh. On a liquid there is no object, and
                    // the analogue is the volume's own transform, which is
                    // exactly what COORD_DOMAIN already does inside the anchor.
                    // Applying both would run the inverse transform TWICE and
                    // push the inclusions into a space that is nobody's: still
                    // stable, still smooth, just wrong by a whole transform —
                    // and it would look like a scale bug in the inclusion size
                    // rather than like a double transform. On the isosurface the
                    // domain's Coordinate Space is the single authority; the flag
                    // keeps its full meaning on the triangle path, where it is
                    // the only thing that can express it.
                    //
                    // ★★★ The DIRECTIONS still take the flag's rotation. They
                    // cannot go through the anchor — transforming a direction
                    // properly needs the field's Jacobian — and a march direction
                    // that disagrees with its origin's space does not fail
                    // visibly: it makes the inclusions subtly elongate along the
                    // flow, which reads as "that is what resin looks like".
                    // Origin in the anchored space with directions in the
                    // volume's rotation is the honest approximation: exact
                    // wherever the mapping is near-rigid, and it degrades by
                    // SHEARING the pattern rather than tearing it.
                    vec3 mOrg = materialAnchor(vol, hitPos);
                    vec3 mDir = Tdir, mLit = resinLightDir;
                    if ((im.flags & MAT_FLAG_RESIN_OBJ_SPACE) != 0u) {
                        // Directions drop the translation column.
                        mat3 volRot = mat3(vol.inv_transform[0], vol.inv_transform[4], vol.inv_transform[8],
                                           vol.inv_transform[1], vol.inv_transform[5], vol.inv_transform[9],
                                           vol.inv_transform[2], vol.inv_transform[6], vol.inv_transform[10]);
                        mDir = normalize(volRot * Tdir);
                        mLit = normalize(volRot * resinLightDir);
                    }
                    ResinMarch rm = resinMarchInterior(
                        mOrg, mDir, imx.transmission_density, ext,
                        imx.resin_inclusion, imx.resin_dirt,
                        vec3(imx.resin_dirt_color_r, imx.resin_dirt_color_g, imx.resin_dirt_color_b),
                        imx.resin_shard, imx.resin_shard_hue,
                        clamp(ct * 0.5 + vec3(0.45), 0.0, 1.0),   // dust tint from coat colour
                        mLit,
                        max(imx.resin_inclusion_scale, 0.01),
                        uint(imx.dust_style + 0.5),
                        vec3(imx.dust_color_a_r, imx.dust_color_a_g, imx.dust_color_a_b),
                        vec3(imx.dust_color_b_r, imx.dust_color_b_g, imx.dust_color_b_b),
                        uint(imx.shard_shape + 0.5), payload.seed);
                    if (rm.dirtHit) {
                        ss.albedo = rm.dirtAlbedo;   // terminate on the speck
                    } else {
                        ss.albedo = mix(ss.albedo * rm.absorb, rm.dustTint * rm.absorb, rm.dustCover);
                        ss.albedo = clamp(ss.albedo + rm.shardGlow + rm.dustGlow + vec3(rm.sparkle),
                                          0.0, 1.0);
                    }
                } else {
                    float pathLen = 2.0 * imx.transmission_density / cosVr;
                    ss.albedo *= exp(-pathLen * ext);
                }
                // The coat already took the specular, so the base is rough and
                // non-metallic — same as the triangle path.
                //
                // The base is rough and non-metallic, same as the triangle path.
                // It DOES get direct lighting now — see the NEE block below,
                // which applies the coat absorption to it as well.
                ss.roughness = 1.0;
                ss.metallic  = 0.0;
            }

            // ═══════════════════════════════════════════════════════════════
            // DIRECT LIGHTING (NEE) — the reason a liquid used to render darker
            // and far noisier than a mesh carrying the SAME material
            // ═══════════════════════════════════════════════════════════════
            // Until now this branch sampled a bounce direction and returned, so
            // the surface received light ONLY when a random BSDF sample happened
            // to land on a light. For anything but a huge, close source that is a
            // tiny solid angle, so most samples contributed nothing (dark) and
            // the rare hits carried enormous energy (noisy). The mesh path never
            // had that problem because it samples the light explicitly.
            //
            // ★ The give-away in a bug report is the pair: "dimmer AND grainier,
            // same material, same distance". Energy loss alone would be dim and
            // clean; a bad lobe would be bright and wrong. Dim + noisy together
            // is the signature of a missing direct-light estimator.
            //
            // ★★ Uses the SHARED evaluators — evaluate_brdf_gl, pdf_brdf_gl,
            // compute_light_pdf_gl, the same light picker — not a private copy.
            // That is the point: the liquid and the mesh are now lit by one
            // estimator, so they cannot drift apart under future edits the way
            // two look-alike implementations would.
            //
            // ★★★ SHADOW RAYS USE MASK 0x01 (triangles only), matching the mesh.
            // The liquid's own AABB is on 0x08, so it is invisible to the shadow
            // ray and A LIQUID DOES NOT SHADOW ITSELF in the direct term. This is
            // deliberate for now, not an oversight: a shadow ray leaving this
            // surface starts inside the half-voxel level-set band and would
            // self-occlude immediately, which needs the same seatOutsideBand
            // treatment the scatter lobes get plus a march for the thickness. The
            // visible consequence is that a deep pool of an OPAQUE liquid (molten
            // metal, chocolate) is lit slightly too evenly on its underside.
            // Clear water, which is what this path mostly renders, is unaffected.
            //
            // Skipped entirely when the glass lobe was taken — that path returned
            // long before here — and skipped for the thin-shell film, which also
            // returns early. Both are specular: NEE contributes nothing to a
            // delta lobe and would just cost a shadow ray.
            {
                float pdf_select = 0.0;
                int lightIdx = pick_smart_light_gl(uvec2(0), hitPos, pdf_select);
                if (lightIdx >= 0) {
                    vec3 wi; float dist; float lightAtten;
                    if (sample_light_direction_gl(lights.l[lightIdx], hitPos,
                                                  rnd(payload.seed), rnd(payload.seed),
                                                  wi, dist, lightAtten) &&
                        dot(wi, wi) > 1e-8) {
                        wi = normalize(wi);
                        float NdotL = max(dot(N, wi), 0.0);
                        if (NdotL > 1e-6) {
                            // Outward-going by construction (NdotL > 0), so the
                            // band probe inside the helper returns immediately —
                            // this costs nothing but keeps the ONE exit-push rule
                            // rather than inventing a second epsilon here.
                            bool sHandOff = false;
                            vec3 shadowOrigin =
                                seatOutsideBand(hitPos, wi, Ng, exitPush, sHandOff);

                            shadowPayload = vec4(1.0, 1.0, 1.0, 0.0);
                            float tmax = max(dist - 1e-3, 1e-3);
                            traceRayEXT(topLevelAS,
                                        gl_RayFlagsTerminateOnFirstHitEXT
                                      | gl_RayFlagsSkipClosestHitShaderEXT,
                                        RT_MASK_DIRECT_SHADOW, 0, 1, 1,
                                        shadowOrigin, 1e-4, wi, tmax, 1);
                            vec3 vis = (shadowPayload.w > 0.5) ? shadowPayload.rgb : vec3(0.0);

                            if (any(greaterThan(vis, vec3(1e-4)))) {
                                vec3 V    = normalize(-rayDir);
                                vec3 brdf = evaluate_brdf_gl(N, V, wi, ss.albedo,
                                                             ss.roughness, ss.metallic,
                                                             ss.specular, isoTrans);
                                vec3 Li = lights.l[lightIdx].color.rgb
                                        * lights.l[lightIdx].color.a * lightAtten;

                                int  ltype  = int(lights.l[lightIdx].position.w + 0.5);
                                bool isDelta = (ltype == 0 || ltype == 1);

                                vec3 contrib;
                                if (isDelta) {
                                    // A delta light has no solid angle for the BSDF
                                    // to have sampled, so there is nothing to
                                    // balance against — MIS weight is 1 by
                                    // definition. Applying a heuristic here would
                                    // halve point and sun lighting for no reason.
                                    contrib = brdf * Li * NdotL / max(pdf_select, 1e-6);
                                } else {
                                    float pdfL = compute_light_pdf_gl(lights.l[lightIdx], dist, 1.0)
                                               * pdf_select;
                                    float pdfB = pdf_brdf_gl(N, V, wi, ss.roughness);
                                    contrib = brdf * Li * NdotL
                                            * power_heuristic(pdfL, pdfB)
                                            / max(pdfL, 1e-6);
                                }

                                if (neeResinActive) {
                                    // Slanted ENTRY path through the coat, exactly
                                    // as the triangle path does it.
                                    float cosL = max(NdotL, 0.05);
                                    contrib *= exp(-(neeResinDensity / cosL) * neeResinExt);
                                }

                                // Firefly clamp, same shape as the mesh's. A single
                                // sample landing on a tiny bright light with a
                                // near-zero pdf otherwise writes a permanent white
                                // dot into the accumulator.
                                contrib = max(contrib, vec3(0.0));
                                contrib = min(contrib, vec3(1e4));
                                if (any(isnan(contrib)) || any(isinf(contrib))) contrib = vec3(0.0);

                                payload.radiance += contrib * vis;
                            }
                        }
                    }
                }
            }

            g_sssIsoEnabled = true;
            g_sssIsoVolume = volIdx;
            scatterPrincipled(ss, payload.seed);
            g_sssIsoEnabled = false;

            // ★ Re-seat the scattered ray OUTSIDE the level-set band.
            //
            // Every lobe leaves via offset_ray(), whose offset is a few float
            // ULPs — correct for a triangle, where the surface is a plane of
            // zero thickness. This surface is not: it is an isosurface inside a
            // proxy band about half a voxel thick, i.e. ~25 mm at the 5 cm
            // voxels the fluid actually runs at. A ULP-offset ray restarts
            // INSIDE that band, immediately re-hits the same surface, and
            // scatters again — and again.
            //
            // That single fact explains three separate symptoms:
            //   - metal turns hard BLACK: each re-hit multiplies throughput by
            //     the metal albedo, and a metal has no diffuse escape;
            //   - clearcoat piles up glare: each re-hit is another draw of the
            //     stochastic layer lottery, with its 1/p compensation;
            //   - it is WORST on overlapping droplets and layered sheets, where
            //     re-entry finds yet another crossing instead of open air.
            // The fall-through dielectric below never showed it because it
            // pushes 3 mm along the outgoing direction and mostly transmits.
            //
            // Push along the direction of travel, so it works for a reflection
            // and a transmission alike, and scale it to the field the band is
            // built from rather than to a hand-tuned millimetre.
            // (exitPush is declared at the top of this block — the thin-shell
            // and resin branches above return early and need the same push.)
            //
            // Goes through the shared seat helper as well: scatterPrincipled's
            // translucent and subsurface lobes can also leave INTO the body, and
            // there the blind push has the same thin-wall problem. Reflections
            // cost nothing extra — the helper returns immediately for them.
            // A subsurface walk LEFT the body somewhere else: seat there, with
            // the exit's own normal. Seating at hitPos (the old code) kept the
            // exit direction but teleported it back to the entry point.
            bool pHandOff = false;
            payload.scatterOrigin = g_sssExited
                ? seatOutsideBand(g_sssExitPos, payload.scatterDir, g_sssExitN, exitPush, pHandOff)
                : seatOutsideBand(hitPos, payload.scatterDir, Ng, exitPush, pHandOff);
            payload.skipVolumeAABBs = pHandOff;

            payload.primaryARG = packHalf2x16(ss.albedo.rg);
            payload.primaryABT = packHalf2x16(vec2(ss.albedo.b, 1.0));
            payload.primaryNrm = plPackNormal(N);
            return;
        }

        if (roughness > 1e-3) {
            float a = roughness * roughness;
            float u1 = rnd(payload.seed);
            float u2 = rnd(payload.seed);
            float phi = TWO_PI * u1;
            float cosT = sqrt(max(0.0, (1.0 - u2) / (1.0 + (a * a - 1.0) * u2)));
            float sinT = sqrt(max(0.0, 1.0 - cosT * cosT));
            vec3 hT = vec3(sinT * cos(phi), sinT * sin(phi), cosT);
            vec3 up = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
            vec3 T = normalize(cross(up, N));
            vec3 B = cross(N, T);
            vec3 Np = normalize(hT.x * T + hT.y * B + hT.z * N);
            if (dot(rayDir, Np) < 0.0) N = Np;  // keep facing the ray
        }

        // Beer-Lambert over the segment the ray just traversed INSIDE the fluid
        // (only when it started inside — i.e. this is an exit / internal event).
        // Pure water is clear; the blue comes from this depth absorption.
        if (startInside) {
            float depth = max(0.0, hitT - tNear);
            vec3 sigmaA = vol.absorption_color * vol.absorption_coefficient;
            payload.attenuation *= exp(-sigmaA * depth);
        }

        // Fresnel, importance-sampled: reflect (scene/sky) vs refract (through).
        // Picking the branch by Fresnel probability means the per-branch weight
        // is 1 — no (1-fresnel)/fresnel multiply, far less noise than splitting.
        float eta = startInside ? IOR_WATER : (1.0 / IOR_WATER);
        float cosTheta = clamp(abs(dot(rayDir, N)), 0.0, 1.0);
        float r0 = (1.0 - IOR_WATER) / (1.0 + IOR_WATER);
        r0 = r0 * r0;
        float fresnel = r0 + (1.0 - r0) * pow(1.0 - cosTheta, 5.0);

        vec3 refrDir = refract(rayDir, N, eta);
        bool tir = dot(refrDir, refrDir) < 1e-6;
        vec3 outDir;
        if (tir || rnd(payload.seed) < fresnel) {
            // Reflection — traces the scene/sky next bounce. No water cast.
            outDir = reflect(rayDir, N);
        } else {
            // Refraction — through the water; mild surface cast.
            outDir = normalize(refrDir);
            payload.attenuation *= vol.scatter_color;
        }

        // Whitewater foam in front of the surface: add its in-scatter now and
        // dim the redirected lobe (water/sky behind) by the foam transmittance.
        payload.radiance    += foam_inscatter;
        payload.attenuation *= foam_T;

        payload.scatterOrigin       = hitPos + outDir * 0.003;
        payload.scatterDir          = outDir;
        payload.scattered           = true;
        payload.primaryARG          = packHalf2x16(vol.scatter_color.rg);
        payload.primaryABT          = packHalf2x16(vec2(vol.scatter_color.b, 1.0));
        payload.primaryNrm          = plPackNormal(N);
        payload.bounceType          = 0u;
        return;
