#ifndef VOLUME_MIXTURE_MATERIAL_SAMPLE_GLSL
#define VOLUME_MIXTURE_MATERIAL_SAMPLE_GLSL

// Sampling and graph evaluation are shared by camera and shadow consumers.
// Graphs still run on zero input density and can create extinction there.
bool mfSampleMaterial(uint index, vec3 position, vec3 direction, bool includeEmission,
    out MfOpticalResponse response, out float density, out float temperature,
    out float emissionDensity) {
    response = mfReadOpticalResponse(index);
    density = 0.0;
    emissionDensity = 0.0;
    temperature = 0.0;
    if (!gasOverlapCandidate(volumes.v[index])) {
        return false;
    }
    uint program = volumes.v[index]._reserved[1] > 0.5
        ? matProgramOffset(uint(volumes.v[index]._reserved[1] - 1.0)) : MATPROG_NONE;
    bool cameraFast = !mfWalking && !mfInShadow && program == MATPROG_NONE &&
        volumes.v[index].volume_type == 2 && volumes.v[index].vdb_grid_address != uint64_t(0) &&
        volumes.v[index]._ext_reserved[7] < 0.5 &&
        (index == mfCacheIndexA || index == mfCacheIndexB);
    // Prepared shadow midpoints are already clipped to this field's bounds.
    bool shadowFast = mfInShadow && mfShadowIndexPath && index == mfShadowIndex;
    if (!cameraFast && !shadowFast && !mfContains(volumes.v[index], position)) {
        return false;
    }
    if (includeEmission && program == MATPROG_NONE &&
        volumeEmissionIndependent(volumes.v[index]._retired_cloud[0])) {
        density = mfDensityChannels(index, position, cameraFast, emissionDensity);
    } else {
        density = mfDensity(index, position, cameraFast);
        emissionDensity = density;
    }
    if (mfInShadow) {
        mfShadowSamples++;
    } else {
        mfSamples++;
    }
    if (density <= 0.0 && emissionDensity <= 0.0 && program == MATPROG_NONE) {
        return false;
    }
    // A shadow graph can read temperature even though it does not emit light.
    if ((includeEmission && volumes.v[index].emission_mode >= 2) || program != MATPROG_NONE) {
        temperature = sampleTemperature(volumes.v[index], position);
    }
    if (program == MATPROG_NONE) {
        return true;
    }
    float attributes[MP_ATTRIB_SLOTS];
    for (int i = 0; i < MP_ATTRIB_SLOTS; ++i) {
        attributes[i] = 0.0;
    }
    vec3 local;
    local.x = dot(vec3(volumes.v[index].inv_transform[0], volumes.v[index].inv_transform[1],
        volumes.v[index].inv_transform[2]), position) + volumes.v[index].inv_transform[3];
    local.y = dot(vec3(volumes.v[index].inv_transform[4], volumes.v[index].inv_transform[5],
        volumes.v[index].inv_transform[6]), position) + volumes.v[index].inv_transform[7];
    local.z = dot(vec3(volumes.v[index].inv_transform[8], volumes.v[index].inv_transform[9],
        volumes.v[index].inv_transform[10]), position) + volumes.v[index].inv_transform[11];
    vec3 normalized = clamp((local - volumes.v[index].aabb_min) /
        max(volumes.v[index].aabb_max - volumes.v[index].aabb_min, vec3(1e-6)),
        vec3(0.0), vec3(1.0));
    MatProgOut material = evalMaterialProgram(program, vec2(0.0), position,
        -direction, 0.5, vec3(0.0), attributes, local, -direction, density,
        temperature, temperature > 0.0
            ? clamp(temperature / max(volumes.v[index].max_temperature, 1.0), 0.0, 1.0) : 0.0,
        0.0, vec3(0.0), position, response.scatterColor, response.emissionColor,
        max(volumes.v[index].voxel_size, 1e-6), normalized, cam.waterTime);
    if ((material.volumeWritten & 1u) != 0u) {
        density = max(material.volumeDensity, 0.0);
        emissionDensity = density;
    }
    if ((material.volumeWritten & 2u) != 0u) {
        response.scatterColor = material.volumeScatterColor;
    }
    if ((material.volumeWritten & 4u) != 0u) {
        response.scatterCoefficient = max(material.volumeScatterStrength, 0.0);
    }
    if ((material.volumeWritten & 8u) != 0u) {
        response.absorptionColor = material.volumeAbsorptionColor;
    }
    if ((material.volumeWritten & 16u) != 0u) {
        response.absorptionCoefficient = max(material.volumeAbsorptionStrength, 0.0);
    }
    if ((material.volumeWritten & 32u) != 0u) {
        response.emissionColor = material.volumeEmissionColor;
    }
    if ((material.volumeWritten & 64u) != 0u) {
        response.emissionStrength = max(material.volumeEmissionStrength, 0.0);
    }
    if ((material.volumeWritten & 128u) != 0u) {
        response.anisotropy = clamp(material.volumeAnisotropy, -0.99, 0.99);
    }
    if ((material.volumeWritten & 256u) != 0u) {
        response.multiScatter = clamp(material.volumeMultiScatter, 0.0, 1.0);
    }
    return true;
}

MfTerms mfEvaluate(uint index, vec3 position, vec3 direction, bool includeEmission) {
    MfOpticalResponse response;
    float density, temperature, emissionDensity;
    if (!mfSampleMaterial(index, position, direction, includeEmission,
            response, density, temperature, emissionDensity)) {
        return MfTerms(vec3(0.0), vec3(0.0), vec3(0.0), 0.0, 0.0, 0.0, 0.0);
    }
    return mfOpticalTerms(index, response, position, density, emissionDensity,
        temperature, includeEmission);
}

vec3 mfShadowExtinction(uint index, vec3 position, vec3 direction) {
    MfOpticalResponse response;
    float density, temperature, emissionDensity;
    if (!mfSampleMaterial(index, position, direction, false, response, density, temperature, emissionDensity)) {
        return vec3(0.0);
    }
    // No edge from this consumer to emission/ramp/blackbody evaluation.
    MfTerms term = mfExtinctionTerms(index, response, density);
    return term.extinction;
}

#endif
