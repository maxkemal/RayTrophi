#ifndef VOLUME_MIXTURE_OPTICAL_TERMS_GLSL
#define VOLUME_MIXTURE_OPTICAL_TERMS_GLSL

// Only graph-writable optical values travel between sampling and response.
// Transforms, ramps and acceleration arrays stay in the canonical SSBO.
struct MfOpticalResponse {
    vec3 scatterColor;
    vec3 absorptionColor;
    vec3 emissionColor;
    float scatterCoefficient;
    float absorptionCoefficient;
    float emissionStrength;
    float anisotropy;
    float multiScatter;
};

MfOpticalResponse mfReadOpticalResponse(uint index) {
    MfOpticalResponse response;
    response.scatterColor = volumes.v[index].scatter_color;
    response.absorptionColor = volumes.v[index].absorption_color;
    response.emissionColor = volumes.v[index].emission_color;
    response.scatterCoefficient = max(volumes.v[index].scatter_coefficient, 0.0);
    response.absorptionCoefficient = max(volumes.v[index].absorption_coefficient, 0.0);
    response.emissionStrength = max(volumes.v[index].emission_intensity, 0.0);
    response.anisotropy = volumes.v[index].scatter_anisotropy;
    response.multiScatter = clamp(volumes.v[index].scatter_multi, 0.0, 1.0);
    return response;
}

vec3 mfEmissionSource(uint index, vec3 emissionColor, float emissionStrength,
    vec3 position, float density, float temperature) {
    vec3 source = vec3(0.0);
    if (volumes.v[index].emission_mode == 1) {
        source = volumeEmissionSource(emissionColor, 1.0, density, 1.0,
            emissionStrength, volumes.v[index]._retired_cloud, false);
    } else if (volumes.v[index].emission_mode >= 2) {
        // Thermal mode uses the authored ramp/blackbody strength. Graph emission
        // color/intensity never affected volumeEmissionAt's thermal branch.
        source = volumeEmissionAt(volumes.v[index], position, density, temperature);
    }
    return source;
}

// Shared by camera and extinction-only shadows. No emission/temperature tree.
MfTerms mfExtinctionTerms(uint index, MfOpticalResponse response, float density) {
    MfTerms term = MfTerms(vec3(0.0), vec3(0.0), vec3(0.0), 0.0, 0.0, 0.0, 0.0);
    float scatterCoefficient = response.scatterCoefficient;
    term.anisotropy = response.anisotropy;
    term.multiScatter = response.multiScatter;
    float meanG = mix(volumes.v[index].scatter_anisotropy_back,
        term.anisotropy, volumes.v[index].scatter_lobe_mix);
    if (volumes.v[index].random_walk_enabled > 0.5 && meanG > 0.05 &&
        mfEventOrdinal >= uint(clamp(int(volumes.v[index].random_walk_exact_events), 1, 512))) {
        scatterCoefficient *= 1.0 - meanG;
        term.anisotropy = 0.0;
        term.reduced = 1.0;
    }
    term.scatterRate = max(density, 0.0) * scatterCoefficient;
    term.scattering = term.scatterRate * clamp(response.scatterColor, vec3(0.0), vec3(1.0));
    vec3 absorption = max(response.absorptionColor, vec3(0.0));
    absorption = mfPeak(absorption) > 1e-5 ? absorption / mfPeak(absorption) : vec3(1.0);
    term.extinction = vec3(term.scatterRate) +
        max(density, 0.0) * response.absorptionCoefficient * absorption;
    return term;
}

// Canonical optical response shared by graph evaluation and aligned cell walks.
MfTerms mfOpticalTerms(uint index, MfOpticalResponse response, vec3 position,
    float density, float temperature, bool includeEmission) {
    MfTerms term = mfExtinctionTerms(index, response, density);
    vec3 source = includeEmission ? mfEmissionSource(index, response.emissionColor,
        response.emissionStrength, position, density, temperature) : vec3(0.0);
    term.emission = source * term.extinction;
    return term;
}

// Unit-density terms of the aligned pair. A shared walk has no graph, no walk
// event and one ray, so extinction is linear in density and the material reads
// and the absorption normalization need to happen only once per ray.
MfTerms mfSharedUnitA;
MfTerms mfSharedUnitB;

MfTerms mfSharedEvaluate(uint index, vec3 position, float density, bool includeEmission) {
    mfSamples++;
    if (density <= 0.0) {
        return MfTerms(vec3(0.0), vec3(0.0), vec3(0.0), 0.0, 0.0, 0.0, 0.0);
    }
    MfTerms term = index == mfCacheIndexA ? mfSharedUnitA : mfSharedUnitB;
    term.extinction *= density;
    term.scattering *= density;
    term.scatterRate *= density;
    if (includeEmission) {
        float temperature = volumes.v[index].emission_mode >= 2
            ? mfSharedTemperature(index, position) : 0.0;
        vec3 source = mfEmissionSource(index, volumes.v[index].emission_color,
            max(volumes.v[index].emission_intensity, 0.0), position, density, temperature);
        term.emission = source * term.extinction;
    }
    return term;
}

#endif
