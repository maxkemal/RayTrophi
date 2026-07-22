#ifndef RAYTROPHI_WATER_V3_GLSL
#define RAYTROPHI_WATER_V3_GLSL

// Water V3 is deliberately independent from the generic Principled material
// vocabulary. The host may still pack authoring values through MaterialExt for
// now, but all water evaluation consumes this explicit contract.
struct WaterV3Hydrology {
    vec2  flowDirection;
    float depth;
    float bankProximity;
    float speed;
    float discharge;
    float froude;
    float foamPotential;
    float alongDistance;
    float crossDistance;
    float width;
};

struct WaterV3SurfaceSample {
    vec3  macroNormalTS;
    vec3  shadingNormalTS;
    float foamProduction;
    float depth;
    float bankProximity;
    float speed;
    float froude;
};

WaterV3Hydrology waterV3DecodeHydrology(vec4 a, vec4 b, vec4 c) {
    WaterV3Hydrology h;
    h.flowDirection = a.xy;
    h.depth = max(a.z, 0.0);
    h.bankProximity = clamp(a.w, 0.0, 1.0);
    h.speed = max(b.x, 0.0);
    h.discharge = max(b.y, 0.0);
    h.froude = max(b.z, 0.0);
    h.foamPotential = clamp(b.w, 0.0, 1.0);
    h.alongDistance = max(c.x, 0.0);
    h.width = max(c.z, 0.0);
    h.crossDistance = (clamp(c.y, 0.0, 1.0) - 0.5) * h.width;
    return h;
}

// Resolved wave bands are composed as slopes, never as a sum of unit normals.
// This keeps macro and capillary amplitudes independent and stable at low
// dielectric roughness.
vec3 waterV3ComposeSlopeNormals(vec3 macroNormalTS, vec3 detailNormalTS) {
    float macroY = max(macroNormalTS.y, 0.05);
    float detailY = max(detailNormalTS.y, 0.05);
    vec2 slope = macroNormalTS.xz / macroY + detailNormalTS.xz / detailY;
    return normalize(vec3(slope.x, 1.0, slope.y));
}

vec3 waterV3TangentToWorld(vec3 tangentNormal, vec3 tangent, vec3 normal, vec3 bitangent) {
    return normalize(tangent * tangentNormal.x +
                     normal * tangentNormal.y +
                     bitangent * tangentNormal.z);
}

float waterV3RapidResponse(float froude) {
    return smoothstep(0.45, 1.10, max(froude, 0.0));
}

float waterV3HydraulicFoam(WaterV3Hydrology h) {
    float rapid = waterV3RapidResponse(h.froude);
    float energeticFlow = rapid * smoothstep(0.15, 1.25, h.speed);
    float shallowResponse = 1.0 - smoothstep(0.35, 2.5, h.depth);
    return clamp(max(h.foamPotential,
                     energeticFlow * mix(0.22, 0.72, shallowResponse)), 0.0, 1.0);
}

// Analytic river spectrum in ribbon tangent space (U downstream, V across).
// Every band contributes its exact slope. There are no finite-difference noise
// cells here, so close-up highlights remain continuous across mesh triangles.
void waterV3EvaluateRiverSpectrum(vec2 riverUV, float time,
                                  WaterV3Hydrology h,
                                  float speedMultiplier,
                                  float strengthMultiplier,
                                  float frequencyMultiplier,
                                  out vec3 normalTS,
                                  out float foamProduction) {
    float speed = max(speedMultiplier, 0.05);
    float baseFrequency = max(frequencyMultiplier, 0.05);
    float strength = max(strengthMultiplier, 0.0);
    float rapid = waterV3RapidResponse(h.froude);
    float depthDamping = mix(0.62, 1.0, 1.0 - smoothstep(0.15, 3.0, h.depth));

    float bank = h.bankProximity;
    float openChannel = 1.0 - 0.72 * bank;
    float crossChannel = riverUV.y;

    // kU, kV, temporal multiplier, relative amplitude and phase offset.
    const vec4 bands[6] = vec4[6](
        vec4(0.38,  0.20, 0.48, 0.32),
        vec4(0.92, -0.55, 0.86, 0.25),
        vec4(1.73,  1.35, 1.18, 0.18),
        vec4(2.85, -2.10, 1.55, 0.12),
        vec4(4.60,  3.40, 2.05, 0.08),
        vec4(7.20, -4.80, 2.65, 0.05));
    const float phaseOffset[6] = float[6](0.0, 1.37, 2.41, 4.12, 0.73, 3.28);

    vec2 slope = vec2(0.0);
    float crestEnergy = 0.0;
    for (int i = 0; i < 6; ++i) {
        vec4 b = bands[i];
        float highBand = float(i) / 5.0;
        float rapidGain = mix(1.0, 1.0 + rapid * 1.35, highBand);
        float amplitude = b.w * strength * depthDamping * openChannel * rapidGain;
        float kU = b.x * baseFrequency;
        float kV = b.y;
        float phase = riverUV.x * kU + crossChannel * kV
                    - time * speed * b.z + phaseOffset[i];
        float c = cos(phase);
        float s = sin(phase);
        slope += vec2(kU, kV) * (amplitude * c);

        // Only compact, energetic crests generate whitewater. Slow long waves
        // shape the reflection but do not paint the whole river white.
        float crest = smoothstep(0.42, 0.92, s * 0.5 + 0.5);
        crestEnergy += crest * amplitude * mix(0.18, 1.0, highBand);
    }

    // Banks create a restrained transverse response without lifting the
    // carrier geometry or changing the river spline itself.
    slope.y += sign(crossChannel) * bank * strength * (0.08 + 0.16 * rapid);
    normalTS = normalize(vec3(-slope.x, 1.0, -slope.y));

    float slopeEnergy = smoothstep(0.08, 0.65, length(slope));
    float analyticCrests = smoothstep(0.025, 0.22, crestEnergy) * slopeEnergy;
    foamProduction = clamp(max(analyticCrests * mix(0.25, 1.0, rapid),
                               waterV3HydraulicFoam(h)), 0.0, 1.0);
}

// Continuous capillary bands for rivers. River surfaces must not use a
// lattice/value-noise gradient: even C1 interpolation exposes its cell grid in
// near-perfect dielectric reflections. These phases and derivatives are
// analytic, advect downstream and remain continuous across ribbon triangles.
void waterV3EvaluateRiverCapillary(vec2 riverUV, float time,
                                   float flowSpeed,
                                   float strength,
                                   float authoredScale,
                                   out vec3 normalTS,
                                   out float breakup) {
    float scale = clamp(sqrt(max(authoredScale, 0.01)), 0.35, 6.0);
    float speed = max(flowSpeed, 0.05);
    float gain = clamp(strength, 0.0, 0.35);
    vec2 q = riverUV;

    vec2 slope = vec2(0.0);
    float pattern = 0.0;
    const vec4 ripples[4] = vec4[4](
        vec4(1.00,  0.18, 1.00, 0.38),
        vec4(1.62, -0.72, 1.31, 0.27),
        vec4(2.45,  1.10, 1.76, 0.21),
        vec4(3.70, -1.65, 2.24, 0.14));
    const float offsets[4] = float[4](0.31, 2.07, 4.38, 1.16);
    for (int i = 0; i < 4; ++i) {
        vec4 r = ripples[i];
        vec2 k = vec2(r.x * scale, r.y);
        float phase = dot(q, k) - time * speed * r.z + offsets[i];
        // r.w is a slope weight, not a displacement amplitude; increasing
        // feature scale does not make the normal field explode.
        slope += normalize(k) * (cos(phase) * r.w * gain);
        pattern += (sin(phase) * 0.5 + 0.5) * r.w;
    }
    normalTS = normalize(vec3(-slope.x, 1.0, -slope.y));
    breakup = clamp(pattern, 0.0, 1.0);
}

// Converts physical foam production into a temporally coherent coverage mask.
// Breakup may move with the flow, but changing a slider does not invalidate any
// geometry and the threshold is smooth rather than a binary frame-to-frame cut.
float waterV3FoamCoverage(float production, float breakup, float threshold) {
    float coherentSignal = clamp(production * mix(0.72, 1.18, breakup), 0.0, 1.0);
    float edge = clamp(threshold, 0.02, 0.92);
    return smoothstep(edge * 0.72, min(edge + 0.28, 1.0), coherentSignal);
}

// Structured coverage: the cell field carves filaments and holes out of the
// mask (additive breakup) instead of merely dimming it, so patch borders are
// ragged rather than smooth gradient blobs. Production still gates everything:
// where nothing is produced, no pattern value can conjure foam.
float waterV3FoamCoverageStructured(float production, float cell, float threshold) {
    float edge = clamp(threshold, 0.02, 0.92);
    float signal = clamp(production + (cell - 0.5) * (0.42 + 0.30 * production), 0.0, 1.0);
    return smoothstep(edge * 0.72, min(edge + 0.28, 1.0), signal)
         * smoothstep(0.0, 0.06, production);
}

#endif
