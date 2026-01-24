/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          CloudManager.h
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once
#include <cuda_runtime.h>
#include "World.h"
#include "fft_ocean.cuh" // Make implementation visible
#include "scene_data.h"

// Link against CUDA libraries (required for direct include)
#pragma comment(lib, "cufft.lib")
#pragma comment(lib, "cudart.lib")

class CloudManager {
public:
    static CloudManager& getInstance() {
        static CloudManager instance;
        return instance;
    }

    // Update cloud simulation (runs FFT if needed)
    void update(float dt, const NishitaSkyParams& params) {
        if (!params.cloud_use_fft) {
            if (initialized) cleanup(); // Free resources if disabled
            return;
        }

        if (!fft_state) {
            fft_state = new FFTOceanState();
        }

        FFTOceanState* state = static_cast<FFTOceanState*>(fft_state);

        // Config for Clouds
        FFTOceanParams fft_params;
        fft_params.resolution = 256; // Standard resolution
        fft_params.ocean_size = 2000.0f; // Large scale
        fft_params.wind_speed = 20.0f;   // Strong winds aloft
        fft_params.wind_direction = 0.0f; 
        fft_params.choppiness = 1.0f;
        fft_params.amplitude = 1.0f;     
        fft_params.time_scale = 1.0f;

        if (!initialized || state->current_resolution != fft_params.resolution) {
            if (initFFTOcean(state, &fft_params)) {
                 initialized = true;
            }
        }

        static float global_time = 0.0f;
        global_time += dt;

        updateFFTOcean(state, &fft_params, global_time);
    }

    // Get the generated FFT texture for cloud shaping
    cudaTextureObject_t getCloudFFTTexture() const {
        if (fft_state && initialized) {
            FFTOceanState* state = static_cast<FFTOceanState*>(fft_state);
            return state->tex_height;
        }
        return 0;
    }
    
    // Clean up resources
    void cleanup() {
        if (fft_state) {
            FFTOceanState* state = static_cast<FFTOceanState*>(fft_state);
            cleanupFFTOcean(state);
            delete state;
            fft_state = nullptr;
        }
        initialized = false;
    }

private:
    CloudManager() = default;
    
    ~CloudManager() {
        cleanup();
    }

    // Internal FFT state
    void* fft_state = nullptr; 
    bool initialized = false;
};

