#include "image_resample.h"
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <algorithm>
#include <vector>

namespace {
    static inline double sinc(double x) {
        if (x == 0.0) return 1.0;
        x *= M_PI;
        return sin(x) / x;
    }

    // Lanczos kernel with parameter a
    static inline double lanczos_kernel(double x, int a) {
        x = fabs(x);
        if (x >= a) return 0.0;
        return sinc(x) * sinc(x / (double)a);
    }
}

namespace ImageResample {

void lanczos_resample_u8(const uint8_t* src, int srcW, int srcH, uint8_t* dst, int dstW, int dstH, int a) {
    if (!src || !dst || srcW <= 0 || srcH <= 0 || dstW <= 0 || dstH <= 0) return;
    // Horizontal pass -> temp float buffer
    std::vector<float> temp((size_t)dstW * srcH, 0.0f);

    double scaleX = (double)dstW / (double)srcW;
    double invScaleX = 1.0 / scaleX;

    for (int y = 0; y < srcH; ++y) {
        for (int dx = 0; dx < dstW; ++dx) {
            // map dst pixel center to src space
            double srcCenter = (dx + 0.5) * invScaleX - 0.5;
            int left = (int)floor(srcCenter - a + 1);
            int right = (int)ceil(srcCenter + a - 1);
            left = std::max(left, 0);
            right = std::min(right, srcW - 1);
            double sum = 0.0;
            double wsum = 0.0;
            for (int sx = left; sx <= right; ++sx) {
                double w = lanczos_kernel(srcCenter - (double)sx, a);
                sum += w * (double)src[y * srcW + sx];
                wsum += w;
            }
            float val = 0.0f;
            if (wsum > 1e-9) val = (float)(sum / wsum);
            temp[y * dstW + dx] = val;
        }
    }

    // Vertical pass
    for (int dy = 0; dy < dstH; ++dy) {
        double srcCenterY = (dy + 0.5) / ((double)dstH / (double)srcH) - 0.5;
        int top = (int)floor(srcCenterY - a + 1);
        int bottom = (int)ceil(srcCenterY + a - 1);
        top = std::max(top, 0);
        bottom = std::min(bottom, srcH - 1);
        for (int dx = 0; dx < dstW; ++dx) {
            double sum = 0.0;
            double wsum = 0.0;
            for (int sy = top; sy <= bottom; ++sy) {
                double w = lanczos_kernel(srcCenterY - (double)sy, a);
                sum += w * (double)temp[sy * dstW + dx];
                wsum += w;
            }
            double v = 0.0;
            if (wsum > 1e-9) v = sum / wsum;
            int iv = (int)std::round(v);
            if (iv < 0) iv = 0; if (iv > 255) iv = 255;
            dst[dy * dstW + dx] = (uint8_t)iv;
        }
    }
}

void lanczos_resample_u16(const uint16_t* src, int srcW, int srcH, uint16_t* dst, int dstW, int dstH, int a) {
    if (!src || !dst || srcW <= 0 || srcH <= 0 || dstW <= 0 || dstH <= 0) return;
    std::vector<double> temp((size_t)dstW * srcH, 0.0);

    double scaleX = (double)dstW / (double)srcW;
    double invScaleX = 1.0 / scaleX;

    for (int y = 0; y < srcH; ++y) {
        for (int dx = 0; dx < dstW; ++dx) {
            double srcCenter = (dx + 0.5) * invScaleX - 0.5;
            int left = (int)floor(srcCenter - a + 1);
            int right = (int)ceil(srcCenter + a - 1);
            left = std::max(left, 0);
            right = std::min(right, srcW - 1);
            double sum = 0.0;
            double wsum = 0.0;
            for (int sx = left; sx <= right; ++sx) {
                double w = lanczos_kernel(srcCenter - (double)sx, a);
                sum += w * (double)src[y * srcW + sx];
                wsum += w;
            }
            double val = 0.0;
            if (wsum > 1e-9) val = sum / wsum;
            temp[y * dstW + dx] = val;
        }
    }

    for (int dy = 0; dy < dstH; ++dy) {
        double srcCenterY = (dy + 0.5) / ((double)dstH / (double)srcH) - 0.5;
        int top = (int)floor(srcCenterY - a + 1);
        int bottom = (int)ceil(srcCenterY + a - 1);
        top = std::max(top, 0);
        bottom = std::min(bottom, srcH - 1);
        for (int dx = 0; dx < dstW; ++dx) {
            double sum = 0.0;
            double wsum = 0.0;
            for (int sy = top; sy <= bottom; ++sy) {
                double w = lanczos_kernel(srcCenterY - (double)sy, a);
                sum += w * temp[sy * dstW + dx];
                wsum += w;
            }
            double v = 0.0;
            if (wsum > 1e-9) v = sum / wsum;
            long iv = (long)std::llround(v);
            if (iv < 0) iv = 0; if (iv > 65535) iv = 65535;
            dst[dy * dstW + dx] = (uint16_t)iv;
        }
    }
}

} // namespace ImageResample
