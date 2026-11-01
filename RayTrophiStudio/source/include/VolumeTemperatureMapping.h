#ifndef RT_VOLUME_TEMPERATURE_MAPPING_H
#define RT_VOLUME_TEMPERATURE_MAPPING_H

// Shared scalar contract for C++, CUDA and GLSL. Units: 0=legacy inference,
// 1=Kelvin, 2=normalized authored range. Known simulation fields resolve to 1
// before publication; only unidentified imported fields retain legacy inference.
#if defined(__CUDACC__)
#define RT_TEMP_FN __host__ __device__ inline
#elif defined(__cplusplus)
#define RT_TEMP_FN inline
#else
#define RT_TEMP_FN
#endif

struct VolumeTemperatureSample {
    float kelvin;
    float normalized;
    bool valid;
};

RT_TEMP_FN int volumeResolveTemperatureUnits(int authored, bool sourceKelvin) {
    return authored == 0 && sourceKelvin ? 1 : authored;
}

RT_TEMP_FN float volumeTemperatureClamp(float value, float low, float high) {
    return value < low ? low : (value > high ? high : value);
}

RT_TEMP_FN VolumeTemperatureSample volumeMapTemperature(
    float temperature, float density, float low, float high, int units, bool hasField) {
    VolumeTemperatureSample result;
    result.kelvin = 0.0f;
    result.normalized = 0.0f;
    result.valid = false;
    // Reject NaN and infinities without renderer-specific math dependencies.
    if (!(temperature >= -3.402823e38f && temperature <= 3.402823e38f) ||
        !(low >= 0.0f && high > low && high <= 3.402823e38f) || units < 0 || units > 2) {
        return result;
    }
    if (units != 0 && (!hasField || temperature <= 0.0f)) {
        return result;
    }
    if (units == 0 && temperature <= 0.0f) {
        temperature = density;
    }
    if (!(temperature >= 0.0f && temperature <= 3.402823e38f) || high <= low) {
        return result;
    }
    if (units == 1 && temperature <= low) {
        return result;
    }
    bool kelvin = units == 1 || (units == 0 && temperature > 20.0f);
    result.kelvin = kelvin ? volumeTemperatureClamp(temperature, low, high)
        : low + volumeTemperatureClamp(temperature, 0.0f, 1.0f) * (high - low);
    float width = high - low;
    result.normalized = volumeTemperatureClamp(
        (result.kelvin - low) / (width > 1.0f ? width : 1.0f), 0.0f, 1.0f);
    result.valid = true;
    return result;
}

#undef RT_TEMP_FN
#endif
