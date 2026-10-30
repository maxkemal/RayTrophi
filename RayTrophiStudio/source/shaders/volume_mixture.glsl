// Joint integration for overlapping fog. The ordinary single-volume path stays
// in volume_closesthit. All contributors are evaluated at the same position;
// there is no primary/companion material truncation in this path.
const int MF_CAPACITY = 16;
uint mfCount;
uint mfSamples;
uint mfEventOrdinal;
uint mfCacheIndexA;
uint mfCacheIndexB;
VolumeNanoRay mfCameraRayA;
VolumeNanoRay mfCameraRayB;
vec2 mfCameraIntervalA;
vec2 mfCameraIntervalB;
float mfCameraT;
pnanovdb_buf_t mfCacheBufferA;
pnanovdb_buf_t mfCacheBufferB;
pnanovdb_map_handle_t mfCacheMapA;
pnanovdb_map_handle_t mfCacheMapB;
pnanovdb_readaccessor_t mfCacheAccessorA;
pnanovdb_readaccessor_t mfCacheAccessorB;

uint mfShadowSamples;
bool mfInShadow;
bool mfShadowIndexPath;
uint mfShadowIndex;
VolumeNanoRay mfShadowRay;
float mfShadowT;
bool mfWalking;
int mfLightingStep;
int mfLightCacheStep;
int mfSkyCacheStep;
int mfShadowStride;
int mfLightIndex;
float mfLightU;
float mfLightV;
vec3 mfLightVisibility;
vec3 mfSkyVisibility;
vec3 mfCameraSkyAmbient;

#include "volume_mixture_cell_cache.glsl"
#include "volume_mixture_shared_state.glsl"


struct MfTerms {
    vec3 extinction;
    vec3 scattering;
    vec3 emission;
    float scatterRate;
    float anisotropy;
    float multiScatter;
    float reduced;
};
MfTerms mfPickedTerms;
uint mfPickedIndex;
uint mfEmptySegments;


float mfPeak(vec3 value) {
    return max(value.x, max(value.y, value.z));
}

bool matterFogOverlap(vec3 origin, vec3 direction, float nearT, uint count) {
    for (uint i = 0u; i < min(count, uint(MF_CAPACITY)); ++i) {
        VkVolumeInstance v = volumes.v[i];
        if (v.source_type != 6 || !gasOverlapCandidate(v)) continue;
        float lo, hi;
        if (volumeRayInterval(v, origin, direction, lo, hi) &&
            lo <= nearT + 1e-4 && hi > nearT + 1e-4) return true;
    }
    return false;
}

bool matterFogJointWalk(uint selected, uint count) {
    bool matter = false;
    bool other = false;
    for (uint i = 0u; i < min(count, uint(MF_CAPACITY)); ++i) {
        VkVolumeInstance v = volumes.v[i];
        if (v.is_active == 0) continue;
        matter = matter || (v.source_type == 6 && gasOverlapCandidate(v));
        other = other || (i != selected &&
            (gasOverlapCandidate(v) || v.source_type == 4));
    }
    return matter && other;
}

// The established companion march already evaluates two non-emissive media
// with a common phase/sampling policy. Keep that optimized path when its exact
// representational conditions hold; use the general integrator otherwise.
bool matterFogNeedsGeneralMixture(uint selected, int companion, uint count) {
    if (companion < 0) return false;
    uint members = 0u;
    for (uint i = 0u; i < min(count, uint(MF_CAPACITY)); ++i) {
        if (gasOverlapCandidate(volumes.v[i])) members++;
    }
    if (members != 2u) return true;
    VkVolumeInstance a = volumes.v[selected];
    VkVolumeInstance b = volumes.v[uint(companion)];
    // The companion fast path has a fixed midpoint lattice. An adaptive
    // request must not silently fall back to it when two materials match.
    if (a.density_curve[9] > 0.5 || b.density_curve[9] > 0.5) return true;
    if (a.random_walk_enabled > 0.5 || b.random_walk_enabled > 0.5 ||
        a.emission_mode != 0 || b.emission_mode != 0) return true;
    if ((a._reserved[1] > 0.5 &&
            matProgramOffset(uint(a._reserved[1] - 1.0)) != MATPROG_NONE) ||
        (b._reserved[1] > 0.5 &&
            matProgramOffset(uint(b._reserved[1] - 1.0)) != MATPROG_NONE)) return true;
    return abs(a.scatter_anisotropy - b.scatter_anisotropy) > 1e-6 ||
        abs(a.scatter_anisotropy_back - b.scatter_anisotropy_back) > 1e-6 ||
        abs(a.scatter_lobe_mix - b.scatter_lobe_mix) > 1e-6 ||
        abs(a.step_size - b.step_size) > 1e-6 || a.max_steps != b.max_steps;
}

bool mfContains(VkVolumeInstance v, vec3 position) {
    float lo, hi;
    return gasOverlapCandidate(v) &&
        volumeRayInterval(v, position, vec3(1.0, 0.0, 0.0), lo, hi) && lo <= 0.0 && hi > 0.0;
}

// Fixed scalar caches cover the common two-medium scene without the dynamic
// aggregate arrays that caused shaderc failures. Further media use a scalar
// fallback, preserving the full supported membership set.
void mfInitializeNano(uint64_t gridAddress, out pnanovdb_buf_t fieldBuffer,
    out pnanovdb_map_handle_t fieldMap, out pnanovdb_readaccessor_t fieldAccessor) {
    fieldBuffer.address = gridAddress;
    fieldMap.address.byte_offset = 0u;
    pnanovdb_root_handle_t fieldRoot;
    fieldRoot.address.byte_offset = 0u;
    if (fieldBuffer.address != uint64_t(0)) {
        pnanovdb_grid_handle_t fieldGrid;
        fieldGrid.address.byte_offset = 0u;
        fieldRoot = pnanovdb_tree_get_root(fieldBuffer, pnanovdb_grid_get_tree(fieldBuffer, fieldGrid));
        fieldMap = pnanovdb_grid_get_map(fieldBuffer, fieldGrid);
    }
    pnanovdb_readaccessor_init(fieldAccessor, fieldRoot);
}

float mfDensity(uint index, vec3 position, bool cameraFast) {
    if (cameraFast && mfSharedDensityReady) {
        if (index == mfCacheIndexA) return mfSharedDensity.x;
        if (index == mfCacheIndexB) return mfSharedDensity.y;
    }
    if (mfInShadow && mfShadowIndexPath && index == mfShadowIndex) {
        if (index == mfCacheIndexA)
            return volumeNanoDensity(volumes.v[index], mfShadowRay, mfShadowT,
                mfCacheBufferA, mfCacheAccessorA);
        if (index == mfCacheIndexB)
            return volumeNanoDensity(volumes.v[index], mfShadowRay, mfShadowT,
                mfCacheBufferB, mfCacheAccessorB);
    }
    if (cameraFast && index == mfCacheIndexA) {
        return volumeNanoDensity(volumes.v[index], mfCameraRayA, mfCameraT,
            mfCacheBufferA, mfCacheAccessorA);
    }
    if (cameraFast && index == mfCacheIndexB) {
        return volumeNanoDensity(volumes.v[index], mfCameraRayB, mfCameraT,
            mfCacheBufferB, mfCacheAccessorB);
    }
    if (index == mfCacheIndexA) {
        return sampleDensityAcc(volumes.v[index], position,
            mfCacheBufferA, mfCacheMapA, mfCacheAccessorA);
    }
    if (index == mfCacheIndexB) {
        return sampleDensityAcc(volumes.v[index], position,
            mfCacheBufferB, mfCacheMapB, mfCacheAccessorB);
    }
    pnanovdb_buf_t fieldBuffer;
    pnanovdb_map_handle_t fieldMap;
    pnanovdb_readaccessor_t fieldAccessor;
    uint64_t gridAddress = volumes.v[index].volume_type == 2
        ? volumes.v[index].vdb_grid_address : uint64_t(0);
    mfInitializeNano(gridAddress, fieldBuffer, fieldMap, fieldAccessor);
    return sampleDensityAcc(volumes.v[index], position, fieldBuffer, fieldMap, fieldAccessor);
}

#include "volume_mixture_optical_terms.glsl"

#include "volume_mixture_material_sample.glsl"

MfTerms mfSum(vec3 position, vec3 direction, bool includeEmission) {
    MfTerms sum = MfTerms(vec3(0.0), vec3(0.0), vec3(0.0), 0.0, 0.0, 0.0, 0.0);
    mfPickedTerms = sum;
    mfPickedIndex = 0u;
    bool sharedGather = mfSharedWalk && mfSharedDensityReady && !mfWalking && !mfInShadow;
    uint count = sharedGather ? 2u : mfCount;
    for (uint slot = 0u; slot < count; ++slot) {
        uint i = sharedGather ? (slot == 0u ? mfCacheIndexA : mfCacheIndexB) : slot;
        MfTerms term;
        if (sharedGather) {
            float density = slot == 0u ? mfSharedDensity.x : mfSharedDensity.y;
            term = mfSharedEvaluate(i, position, density, includeEmission);
        } else {
            term = mfEvaluate(i, position, direction, includeEmission);
        }
        sum.extinction += term.extinction;
        sum.scattering += term.scattering;
        sum.emission += term.emission;
        sum.scatterRate += term.scatterRate;
        // Weighted reservoir: retain one scattering contributor during the
        // coefficient gather itself. No second NanoVDB/material gather.
        if (term.scatterRate > 0.0 &&
            rnd(payload.seed) * sum.scatterRate < term.scatterRate) {
            mfPickedTerms = term;
            mfPickedIndex = i;
        }
    }
    return sum;
}

#include "volume_mixture_adaptive.glsl"
#include "volume_mixture_shared_walk.glsl"

float mfEmptyStep(vec3 position, vec3 direction, float baseStep) {
    if (mfSharedWalk) {
        return mfSharedEmptyStep(baseStep);
    }
    float safe = 1e30;
    for (uint i = 0u; i < mfCount; ++i) {
        VkVolumeInstance v = volumes.v[i];
        if (!gasOverlapCandidate(v)) continue;
        if (v._uvw_pad[0] < 0.5 || v._ext_reserved[7] >= 0.5 ||
            volumeNanoRemap(v, 0.0) > max(v._reserved[0], 0.0)) return baseStep;
        // A graph can create density in source-empty cells. Unknown/dense
        // producers conservatively retain ordinary stepping.
        if (v._reserved[1] > 0.5 &&
            matProgramOffset(uint(v._reserved[1] - 1.0)) != MATPROG_NONE) return baseStep;
        float lo, hi;
        if (i == mfCacheIndexA) {
            lo = mfCameraIntervalA.x - mfCameraT;
            hi = mfCameraIntervalA.y - mfCameraT;
        } else if (i == mfCacheIndexB) {
            lo = mfCameraIntervalB.x - mfCameraT;
            hi = mfCameraIntervalB.y - mfCameraT;
        } else if (!volumeRayInterval(v, position, direction, lo, hi)) {
            continue;
        }
        if (hi <= 0.0) continue;
        if (lo > 0.0) {
            safe = min(safe, lo);
            continue;
        }
        if (v.volume_type != 2 || v.vdb_grid_address == uint64_t(0)) return baseStep;
        uint kind = 0u;
        float span;
        if (i == mfCacheIndexA) {
            span = nanoEmptyIndexStep(v,
                mfCameraRayA.indexOrigin + mfCameraRayA.indexDirection * mfCameraT,
                mfCameraRayA.indexDirection, baseStep, mfCacheBufferA, mfCacheAccessorA, kind);
        } else if (i == mfCacheIndexB) {
            span = nanoEmptyIndexStep(v,
                mfCameraRayB.indexOrigin + mfCameraRayB.indexDirection * mfCameraT,
                mfCameraRayB.indexDirection, baseStep, mfCacheBufferB, mfCacheAccessorB, kind);
        } else {
            pnanovdb_buf_t fieldBuffer;
            pnanovdb_map_handle_t fieldMap;
            pnanovdb_readaccessor_t fieldAccessor;
            mfInitializeNano(v.volume_type == 2 ? v.vdb_grid_address : uint64_t(0),
                fieldBuffer, fieldMap, fieldAccessor);
            span = nanoEmptyTileStep(v, position, direction, baseStep,
                fieldBuffer, fieldMap, fieldAccessor, kind);
        }
        safe = min(safe, min(span, hi));
        if (safe <= baseStep) return baseStep;
    }
    return max(safe, baseStep);
}

float mfExit(vec3 position, vec3 direction) {
    float distance = 0.0;
    for (uint i = 0u; i < mfCount; ++i) {
        if (!gasOverlapCandidate(volumes.v[i])) continue;
        float lo, hi;
        if (volumeRayInterval(volumes.v[i], position, direction, lo, hi) && hi > 0.0)
            distance = max(distance, hi);
    }
    return distance;
}

// Only skip density that the existing sampler already maps to zero. Restrict
// this proof to the live sparse producer, whose inactive values/background
// are bounded. Programs and noise can populate otherwise empty source cells.
float mfShadowEmptyStep(uint index, VkVolumeInstance v, float baseStep) {
    if (!mfShadowIndexPath || v._uvw_pad[0] < 0.5 ||
        v._ext_reserved[7] >= 0.5 ||
        volumeNanoRemap(v, 1e-6) > max(v._reserved[0], 0.0)) return baseStep;
    if (v._reserved[1] > 0.5 &&
        matProgramOffset(uint(v._reserved[1] - 1.0)) != MATPROG_NONE) return baseStep;
    vec3 indexPosition = mfShadowRay.indexOrigin + mfShadowRay.indexDirection * mfShadowT;
    uint kind = 0u;
    if (index == mfCacheIndexA) {
        return nanoEmptyIndexStep(v, indexPosition, mfShadowRay.indexDirection,
            baseStep, mfCacheBufferA, mfCacheAccessorA, kind);
    }
    if (index == mfCacheIndexB) {
        return nanoEmptyIndexStep(v, indexPosition, mfShadowRay.indexDirection,
            baseStep, mfCacheBufferB, mfCacheAccessorB, kind);
    }
    return baseStep;
}

// Each material's authored shadow budget controls its own chord. Camera
// stepping is not a shadow budget. The visibility is shared by all scattering
// contributors rather than marching every medium again for every contributor.
vec3 mfShadow(vec3 position, vec3 direction, float distance) {
    vec3 visibility = vec3(1.0);
    mfInShadow = true;
    for (uint index = 0u; index < mfCount; ++index) {
        VkVolumeInstance v = volumes.v[index];
        if (!gasOverlapCandidate(v) || (!mfWalking && v.shadow_steps <= 0)) continue;
        float lo, hi;
        if (!volumeRayInterval(v, position, direction, lo, hi)) continue;
        lo = max(lo, 0.0);
        hi = min(hi, distance);
        if (hi <= lo) continue;
        int steps = clamp(v.shadow_steps, mfWalking ? 2 : 1, 64);
        float dt = (hi - lo) / float(steps);
        // Prepare this straight light chord once, rather than transforming
        // every quadrature point through world/local/NanoVDB space again.
        // Density programs and noise retain their original world-space path.
        mfShadowIndex = index;
        mfShadowIndexPath = false;
        if (index == mfCacheIndexA && volumeNanoIndexEligible(v, mfCacheBufferA)) {
            mfShadowRay = volumeNanoRay(v, position, direction, mfCacheBufferA, mfCacheMapA);
            mfShadowIndexPath = true;
        } else if (index == mfCacheIndexB && volumeNanoIndexEligible(v, mfCacheBufferB)) {
            mfShadowRay = volumeNanoRay(v, position, direction, mfCacheBufferB, mfCacheMapB);
            mfShadowIndexPath = true;
        }
        // Prepared NanoVDB chords have no density/material program or noise.
        // Colors, emission and phase terms are unused by shadows; prepare the
        // exact extinction coefficient outside the midpoint loop.
        float shadowScatter = 0.0;
        float shadowAbsorption = 0.0;
        vec3 shadowAbsorptionWeights = vec3(1.0);
        if (mfShadowIndexPath) {
            float scatterCoefficient = max(v.scatter_coefficient, 0.0);
            float meanG = mix(v.scatter_anisotropy_back,
                v.scatter_anisotropy, v.scatter_lobe_mix);
            if (v.random_walk_enabled > 0.5 && meanG > 0.05 &&
                mfEventOrdinal >= uint(clamp(int(v.random_walk_exact_events), 1, 512))) {
                scatterCoefficient *= 1.0 - meanG;
            }
            vec3 absorption = max(v.absorption_color, vec3(0.0));
            absorption = mfPeak(absorption) > 1e-5
                ? absorption / mfPeak(absorption) : vec3(1.0);
            shadowScatter = scatterCoefficient;
            shadowAbsorption = max(v.absorption_coefficient, 0.0);
            shadowAbsorptionWeights = absorption;
        }
        vec3 opticalDepth = vec3(0.0);
        for (int step = 0; step < steps; ++step) {
            mfShadowT = lo + (float(step) + 0.5) * dt;
            float emptySpan = min(mfShadowEmptyStep(index, v, dt), hi - mfShadowT);
            if (emptySpan > dt * 1.01) {
                // Preserve every surviving midpoint and its original weight.
                // The loop increment resumes on the same authored lattice.
                int skip = max(1, int(floor(emptySpan / dt)));
                step += skip - 1;
                continue;
            }
            vec3 samplePosition = position + direction *
                mfShadowT;
            if (mfShadowIndexPath) {
                float density = mfDensity(index, samplePosition, false);
                mfShadowSamples++;
                float positiveDensity = max(density, 0.0);
                // Match the reference operation order for colored extinction.
                float scatterRate = positiveDensity * shadowScatter;
                vec3 extinction = vec3(scatterRate) +
                    positiveDensity * shadowAbsorption * shadowAbsorptionWeights;
                opticalDepth += extinction * dt;
            } else {
                opticalDepth += mfShadowExtinction(index, samplePosition, direction) * dt;
            }
            if (min(opticalDepth.x, min(opticalDepth.y, opticalDepth.z)) > 16.0) break;
        }
        vec3 physical = exp(-opticalDepth);
        float strength = mfWalking ? 1.0 : clamp(v.shadow_strength * 0.92, 0.0, 1.0);
        visibility *= mix(vec3(1.0), physical, strength);
    }
    mfInShadow = false;
    return visibility;
}

#include "volume_mixture_light_cache.glsl"

// Only the random walk tests geometry per shadow query. The camera march does
// not, like the ordinary march (volume_fog_shading.glsl): a solid inside gas must
// not paint a hard black shadow, and a trace here was the mixture's dearest
// uncounted work - roughly 30 of 106 ms in the 2026-10-30 two-gas ablation.
// Refractive SDF is not an opaque blocker: this mask matches the fog light mask.
bool mfGeometryBlocks(vec3 position, vec3 direction, float distance) {
    if (!mfWalking) {
        return false;
    }
    rwShadowTraces++;
    return rwSolidDistance(position, direction, distance) >= 0.0;
}

vec3 mfDirect(uint contributor, MfTerms term, vec3 position, vec3 direction,
    bool approximateSky, float opticalStep) {
    vec3 result = vec3(0.0);
    if (approximateSky) {
        vec3 ambientDirection = normalize(vec3(0.0, 1.0, 0.0) * 0.55 + direction * 0.45);
        // Same 0.01 gate as the ordinary march: an optically thin sample keeps
        // the visibility it holds instead of buying another sky march.
        if (mfWalking || (mfLightingStep - mfSkyCacheStep >= max(mfShadowStride, 8) &&
                opticalStep > 0.01)) {
            float distance = max(mfExit(position, ambientDirection), 0.001);
            mfSkyVisibility = mfGeometryBlocks(position, ambientDirection, distance)
                ? vec3(0.0) : mfShadow(position, ambientDirection, distance);
            mfSkyCacheStep = mfLightingStep;
        }
        float skyT = dot(mfSkyVisibility, vec3(0.2126, 0.7152, 0.0722));
        float albedo = dot(term.scattering / max(term.extinction, vec3(EPSILON)),
            vec3(0.2126, 0.7152, 0.0722));
        float higherOrder = volumeHigherOrderSkyVisibility(skyT, term.multiScatter, albedo);
        result = (mfWalking ? sampleSkyAmbient(direction) : mfCameraSkyAmbient) *
            (mfSkyVisibility + vec3(higherOrder));
    }
    if (cam.lightCount == 0u) return result;
    LightData light = lights.l[mfLightIndex];
    vec3 wi;
    float distance, attenuation;
    bool inverseSquare;
    if (!volSampleLight(light, position, mfLightU, mfLightV,
            wi, distance, attenuation, inverseSquare)) return result;
    if (inverseSquare) attenuation /= max(distance * distance, 1e-4);
    VkVolumeInstance v = volumes.v[contributor];
    float phase = dualLobeHG(dot(direction, wi), term.anisotropy,
        term.reduced > 0.5 ? 0.0 : v.scatter_anisotropy_back, v.scatter_lobe_mix);
    if (mfWalking || mfLightingStep - mfLightCacheStep >= mfShadowStride) {
        mfLightVisibility = mfGeometryBlocks(position, wi, distance)
            ? vec3(0.0) : mfDirectLightShadow(position, wi, distance);
        mfLightCacheStep = mfLightingStep;
    }
    return result + float(cam.lightCount) * light.color.rgb * light.color.a *
        attenuation * phase * mfLightVisibility;
}

// Piecewise-constant exponential flight. A scalar proposal equal to the largest
// RGB extinction preserves colored absorption through the likelihood ratio.
// The same coefficients drive survival, emission and collision selection.
bool mfFlight(vec3 origin, vec3 direction, float limit, float stepLength,
    inout vec3 throughput, inout vec3 radiance, out float hitDistance,
    out MfTerms hitTerms, out vec3 coefficientPosition) {
    float opticalTarget = -log(max(1.0 - rnd(payload.seed), 1e-7));
    int steps = clamp(int(ceil(limit / stepLength)), 1, 2048);
    float baseStep = limit / float(steps);
    hitTerms = MfTerms(vec3(0.0), vec3(0.0), vec3(0.0), 0.0, 0.0, 0.0, 0.0);
    coefficientPosition = origin;
    float start = 0.0;
    for (int i = 0; i < MF_ADAPTIVE_LIMIT && start < limit; ++i) {
        float dt = mfBudgetedSpan(origin, direction, start, limit - start, baseStep, i);
        coefficientPosition = origin + direction * (start + 0.5 * dt);
        MfTerms term = mfSum(coefficientPosition, direction, true);
        float proposal = mfPeak(term.extinction);
        bool collision = proposal > EPSILON && opticalTarget < proposal * dt;
        float length = collision ? opticalTarget / proposal : dt;
        vec3 ratio = vec3(proposal) - term.extinction;
        vec3 integral;
        for (int channel = 0; channel < 3; ++channel) {
            integral[channel] = abs(ratio[channel]) > 1e-6
                ? (exp(ratio[channel] * length) - 1.0) / ratio[channel] : length;
        }
        radiance += throughput * term.emission * integral;
        throughput *= exp(ratio * length);
        if (collision) {
            hitDistance = start + length;
            // Return the same piecewise sample used to draw the collision.
            hitTerms = term;
            return true;
        }
        opticalTarget -= proposal * dt;
        start += dt;
    }
    hitDistance = limit;
    return false;
}

void mfContinue(vec3 radiance, vec3 throughput, vec3 origin, vec3 direction,
    bool boundary, bool scattered, uint events) {
    payload.radiance = radiance;
    payload.attenuation *= throughput;
    payload.scatterOrigin = origin;
    payload.scatterDir = direction;
    payload.scattered = mfPeak(throughput) > 0.0;
    payload.skipVolumeAABBs = false;
    payload.skipGasVolumes = boundary;
    payload.bounceType = scattered ? BOUNCE_DIFFUSE : BOUNCE_TRANSPARENT;
    if (boundary) volumeRecordGasHandoff();
    volumeRecordShadowSamples(mfShadowSamples);
    volumeRecordRay(mfSamples, mfEmptySegments, 0u, 0u,
        mfStepBudgetHit ? VOLUME_MARCH_STEP_BUDGET : VOLUME_MARCH_COMPLETED);
    if (scattered) volumeRecordWalk(events, rwProbeTraces, rwShadowTraces, 0u, false);
}

void mfPrimary(vec3 color, float distance, float transmission) {
    if ((payload.primaryMeta & PL_PRIMARY_DONE) != 0u) return;
    payload.primaryARG = packHalf2x16(color.rg);
    payload.primaryABT = packHalf2x16(vec2(color.b, transmission));
    payload.primaryNrm = floatBitsToUint(distance);
    payload.primaryMeta = (payload.primaryMeta & PL_DISP_MASK) | PL_PRIMARY_DONE |
        PL_PRIMARY_VOLUME | PL_PRIMARY_VOLUME_DEPTH | PL_MATID_MASK;
}

void shadeFogMixture(vec3 origin, vec3 direction, float nearT, float farT,
    bool boundary, uint selected, uint count, bool cached,
    vec3 cachedRadiance, vec3 cachedTransmission) {
    mfCount = min(count, uint(MF_CAPACITY));
    mfCacheIndexA = 0xffffffffu;
    mfCacheIndexB = 0xffffffffu;
    mfResetCellBounds();
    mfSharedWalk = false;
    mfSharedDensityReady = false;
    mfSamples = 0u;
    mfEventOrdinal = 0u;
    mfShadowSamples = 0u;
    mfEmptySegments = 0u;
    mfStepBudgetHit = false;
    mfInShadow = false;
    mfWalking = false;
    mfLightingStep = 0;
    mfLightCacheStep = -10000;
    mfSkyCacheStep = -10000;
    mfShadowStride = 16;
    if (cached) {
        mfContinue(cachedRadiance, cachedTransmission, boundary ? origin :
            origin + direction * (farT + 0.001), direction, boundary, false, 0u);
        return;
    }
    mfLightIndex = cam.lightCount > 0u
        ? min(int(rnd(payload.seed) * float(cam.lightCount)), int(cam.lightCount) - 1) : 0;
    mfLightU = rnd(payload.seed);
    mfLightV = rnd(payload.seed);
    mfLightVisibility = vec3(1.0);
    mfSkyVisibility = vec3(1.0);
    rwShadowTraces = 0u;
    rwProbeTraces = 0u;
    float stepLength = 1e30;
    int maximumEvents = 1;
    bool randomWalk = false;
    for (uint i = 0u; i < mfCount; ++i) {
        VkVolumeInstance v = volumes.v[i];
        if (!gasOverlapCandidate(v)) continue;
        if (mfCacheIndexA == 0xffffffffu) {
            mfCacheIndexA = i;
            mfInitializeNano(v.volume_type == 2 ? v.vdb_grid_address : uint64_t(0),
                mfCacheBufferA, mfCacheMapA, mfCacheAccessorA);
            if (v.volume_type == 2 && v.vdb_grid_address != uint64_t(0)) {
                mfCameraRayA = volumeNanoRay(v, origin, direction,
                    mfCacheBufferA, mfCacheMapA);
            }
            float loA, hiA;
            bool intersectsA = volumeRayInterval(v, origin, direction, loA, hiA);
            mfCameraIntervalA = intersectsA ? vec2(loA, hiA) : vec2(-1.0);
        } else if (mfCacheIndexB == 0xffffffffu) {
            mfCacheIndexB = i;
            mfInitializeNano(v.volume_type == 2 ? v.vdb_grid_address : uint64_t(0),
                mfCacheBufferB, mfCacheMapB, mfCacheAccessorB);
            if (v.volume_type == 2 && v.vdb_grid_address != uint64_t(0)) {
                mfCameraRayB = volumeNanoRay(v, origin, direction,
                    mfCacheBufferB, mfCacheMapB);
            }
            float loB, hiB;
            bool intersectsB = volumeRayInterval(v, origin, direction, loB, hiB);
            mfCameraIntervalB = intersectsB ? vec2(loB, hiB) : vec2(-1.0);
        }
        mfShadowStride = min(mfShadowStride, clamp(v.shadow_stride, 1, 16));
        stepLength = min(stepLength, max(v.step_size, max(v.voxel_size * 0.25, 1e-4)));
        if (v.random_walk_enabled > 0.5) {
            randomWalk = true;
            maximumEvents = max(maximumEvents, clamp(int(v.random_walk_max_events), 8, 512));
        }
    }
    vec3 throughput = vec3(1.0);
    vec3 radiance = vec3(0.0);
    vec3 entry = origin + direction * nearT;
    // The caller already clamped this interval to opaque/SDF boundaries.
    // Do not issue the same camera solid probe a second time.
    mfWalking = randomWalk;
    mfPrepareDirectLightCache();
    mfConfigureSharedWalk(origin, direction);
    if (!randomWalk) {
        // A straight camera segment has one view direction, so one sky lookup.
        mfCameraSkyAmbient = sampleSkyAmbient(direction);
        int stepBudget = mfCameraStepBudget(origin, direction, nearT, farT);
        int steps = clamp(int(ceil((farT - nearT) / stepLength)), 1, stepBudget);
        float baseStep = (farT - nearT) / float(steps);
        float opacityWeight = 0.0;
        float depthWeight = 0.0;
        vec3 colorWeight = vec3(0.0);
        float marchT = nearT;
        int sampledSteps = 0;
        for (int step = 0; step < MF_ADAPTIVE_LIMIT && marchT < farT &&
                sampledSteps < stepBudget; ++step) {
            float remaining = farT - marchT;
            mfCameraT = marchT;
            float emptySpan = min(mfEmptyStep(origin + direction * marchT,
                direction, baseStep), remaining);
            if (emptySpan > baseStep * 1.01 && step < MF_ADAPTIVE_LIMIT - 1) {
                mfEmptySegments += uint(max(1.0, floor(emptySpan / baseStep)));
                marchT += emptySpan;
                continue;
            }
            float dt = mfCameraBudgetedSpan(origin, direction, marchT, remaining,
                baseStep, step, stepBudget - sampledSteps);
            mfCameraT = marchT + 0.5 * dt;
            mfPrepareSharedDensity(mfCameraT);
            vec3 position = origin + direction * mfCameraT;
            MfTerms sum = mfSum(position, direction, true);
            sampledSteps++;
            vec3 source = sum.emission;
            mfLightingStep = step;
            if (sum.scatterRate > EPSILON && mfPickedTerms.scatterRate > 0.0) {
                source += mfPickedTerms.scattering *
                    (sum.scatterRate / mfPickedTerms.scatterRate) *
                    mfDirect(mfPickedIndex, mfPickedTerms, position, direction, true,
                        dot(sum.extinction, vec3(0.2126, 0.7152, 0.0722)) * baseStep);
            }
            vec3 integral;
            for (int channel = 0; channel < 3; ++channel)
                integral[channel] = sum.extinction[channel] > 1e-6
                    ? (1.0 - exp(-sum.extinction[channel] * dt)) / sum.extinction[channel] : dt;
            radiance += throughput * source * integral;
            vec3 sampleT = exp(-sum.extinction * dt);
            float opacity = dot(throughput * (vec3(1.0) - sampleT),
                vec3(0.2126, 0.7152, 0.0722));
            opacityWeight += opacity;
            depthWeight += opacity * mfCameraT;
            colorWeight += opacity * sum.scattering / max(sum.scatterRate, EPSILON);
            throughput *= sampleT;
            marchT += dt;
            // The ordinary march's cutoff. 1e-6 doubled the optical depth walked
            // inside opaque gas for nothing the camera can resolve.
            if (mfPeak(throughput) < 0.001) break;
        }
        // A boundary handoff leaves the primary AOV for the actual SDF/solid,
        // as the ordinary gas path does. Thin transparent fog is not a surface.
        if (!boundary && opacityWeight > EPSILON &&
            (opacityWeight > 0.04 || mfPeak(radiance) > 5e-4))
            mfPrimary(colorWeight / opacityWeight, depthWeight / opacityWeight,
                dot(throughput, vec3(0.2126, 0.7152, 0.0722)));
        mfContinue(radiance, throughput, boundary ? origin :
            origin + direction * (farT + 0.001), direction, boundary, false, 0u);
        return;
    }
    vec3 position = entry;
    float limit = max(farT - nearT, 0.0);
    uint events = 0u;
    for (int event = 0; event < maximumEvents; ++event) {
        float distance;
        MfTerms sum;
        vec3 coefficientPosition;
        bool collision = mfFlight(position, direction, limit, stepLength,
            throughput, radiance, distance, sum, coefficientPosition);
        if (!collision) {
            mfContinue(radiance, throughput, boundary ? position :
                position + direction * (distance + 0.001), direction,
                boundary, events > 0u, events);
            return;
        }
        events++;
        position += direction * distance;
        if (sum.scatterRate <= EPSILON) break;
        // mfFlight's final gather retained the contributor drawn from the
        // same piecewise optical sample that generated this collision.
        uint contributor = mfPickedIndex;
        MfTerms term = mfPickedTerms;
        throughput *= (term.scatterRate > EPSILON ? term.scattering / term.scatterRate : vec3(0.0)) *
            (sum.scatterRate / max(mfPeak(sum.extinction), EPSILON));
        mfPrimary(term.scattering / max(term.scatterRate, EPSILON), nearT + distance, 0.0);
        VkVolumeInstance v = volumes.v[contributor];
        radiance += throughput * mfDirect(contributor, term, position, direction,
            v.random_walk_enabled < 0.5, 1.0);
        if (v.random_walk_enabled < 0.5) break;
        if (event >= 4) {
            float probability = min(mfPeak(throughput), 0.98);
            if (rnd(payload.seed) >= probability) break;
            throughput /= max(probability, 1e-6);
        }
        float anisotropy = term.reduced > 0.5 ? 0.0 : (rnd(payload.seed) < v.scatter_lobe_mix
            ? term.anisotropy : v.scatter_anisotropy_back);
        direction = sampleHG(direction, anisotropy, payload.seed);
        mfEventOrdinal++;
        limit = mfExit(position, direction);
        float solid = rwSolidDistance(position, direction, limit);
        rwProbeTraces++;
        float surface = nearestSurfaceSDFCrossing(position, direction, 0.001,
            limit, contributor, mfCount);
        boundary = solid >= 0.0 || surface > 0.0;
        if (solid >= 0.0) limit = min(limit, max(solid - 0.001, 0.0));
        if (surface > 0.0) limit = min(limit, max(surface - 0.001, 0.0));
    }
    mfContinue(radiance, vec3(0.0), position, direction, false, true,
        events);
}
