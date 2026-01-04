#include "CloudManager.h"
#include "fft_ocean.cuh"
#include "scene_data.h" // For SCENE_LOG macros

CloudManager::~CloudManager() {
    cleanup();
}

void CloudManager::cleanup() {
    if (fft_state) {
        FFTOceanState* state = static_cast<FFTOceanState*>(fft_state);
        cleanupFFTOcean(state);
        delete state;
        fft_state = nullptr;
    }
    initialized = false;
}

void CloudManager::update(float dt, const NishitaSkyParams& params) {
    if (!params.cloud_use_fft) {
        if (initialized) cleanup(); // Free resources if disabled
        return;
    }

    if (!fft_state) {
        fft_state = new FFTOceanState();
    }

    FFTOceanState* state = static_cast<FFTOceanState*>(fft_state);

    // Config for Clouds
    // We repurpose ocean FFT for clouds. 
    // - Ocean Size -> Cloud Scale
    // - Wind Speed -> Cloud movement speed
    FFTOceanParams fft_params;
    fft_params.resolution = 256; // Standard resolution is enough for clouds
    fft_params.ocean_size = 2000.0f; // Large scale for sky
    fft_params.wind_speed = 20.0f;   // Strong winds aloft
    fft_params.wind_direction = 0.0f; 
    fft_params.choppiness = 1.0f;
    fft_params.amplitude = 1.0f;     // Full range
    fft_params.time_scale = 1.0f;

    // Use params from World if available (optional)
    // For now hardcoded or derived from existing cloud params
    // Future: Add specific FFT params to Nishita struct if needed

    if (!initialized || state->current_resolution != fft_params.resolution) {
        if (initFFTOcean(state, &fft_params)) {
             initialized = true;
        }
    }

    // Update global time
    static float global_time = 0.0f;
    global_time += dt;

    updateFFTOcean(state, &fft_params, global_time);
}

cudaTextureObject_t CloudManager::getCloudFFTTexture() const {
    if (fft_state && initialized) {
        FFTOceanState* state = static_cast<FFTOceanState*>(fft_state);
        return state->tex_height;
    }
    return 0;
}
