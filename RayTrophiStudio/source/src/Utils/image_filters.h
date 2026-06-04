#pragma once
#include <vector>

namespace ImageFilters {
    // Separable Gaussian blur for float single-channel images
    void gaussian_blur_f(const std::vector<float>& src, int w, int h, std::vector<float>& dst, float sigma, int radius = -1);

    // Frequency separation: lowpass + detail transfer
    // - src is input normalized float [0,1]
    // - dst is output normalized float [0,1]
    // - sigma controls lowpass strength; detail_strength in [0,1] multiplies detail layer
    void frequency_separation(const std::vector<float>& src, int w, int h, std::vector<float>& dst, float sigma = 1.0f, float detail_strength = 0.85f);
}
