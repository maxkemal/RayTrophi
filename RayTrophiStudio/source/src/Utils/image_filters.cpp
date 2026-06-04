#include "image_filters.h"
#include <cmath>
#include <algorithm>

namespace {
    static void make_gaussian_kernel(float sigma, std::vector<float>& kernel, int& radius) {
        if (sigma <= 0.0f) sigma = 1.0f;
        if (radius <= 0) radius = (int)std::ceil(3.0f * sigma);
        kernel.resize(radius * 2 + 1);
        float sum = 0.0f;
        for (int i = -radius; i <= radius; ++i) {
            float x = (float)i;
            kernel[i + radius] = std::exp(-(x*x) / (2.0f * sigma * sigma));
            sum += kernel[i + radius];
        }
        for (size_t i = 0; i < kernel.size(); ++i) kernel[i] /= sum;
    }
}

namespace ImageFilters {

void gaussian_blur_f(const std::vector<float>& src, int w, int h, std::vector<float>& dst, float sigma, int radius) {
    if (w <= 0 || h <= 0) return;
    if (dst.size() != (size_t)w * h) dst.assign((size_t)w * h, 0.0f);

    std::vector<float> kernel;
    make_gaussian_kernel(sigma, kernel, radius);
    int r = (int)kernel.size() / 2;

    // Horizontal pass
    std::vector<float> temp((size_t)w * h, 0.0f);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            double sum = 0.0;
            for (int k = -r; k <= r; ++k) {
                int sx = x + k;
                if (sx < 0) sx = 0;
                if (sx >= w) sx = w - 1;
                sum += kernel[k + r] * src[y * w + sx];
            }
            temp[y * w + x] = (float)sum;
        }
    }

    // Vertical pass
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            double sum = 0.0;
            for (int k = -r; k <= r; ++k) {
                int sy = y + k;
                if (sy < 0) sy = 0;
                if (sy >= h) sy = h - 1;
                sum += kernel[k + r] * temp[sy * w + x];
            }
            dst[y * w + x] = (float)sum;
        }
    }
}

void frequency_separation(const std::vector<float>& src, int w, int h, std::vector<float>& dst, float sigma, float detail_strength) {
    if (w <= 0 || h <= 0) return;
    dst.resize((size_t)w * h);
    std::vector<float> low((size_t)w * h);
    gaussian_blur_f(src, w, h, low, sigma);
    for (int i = 0; i < w * h; ++i) {
        float detail = src[i] - low[i];
        float recomb = low[i] + detail * detail_strength;
        dst[i] = std::clamp(recomb, 0.0f, 1.0f);
    }
}

} // namespace ImageFilters
