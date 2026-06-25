#pragma once

#include <cstdint>

// Simple separable Lanczos resampler for single-channel images.
// Supports 8-bit and 16-bit input.

namespace ImageResample {
    // resample single-channel 8-bit
    void lanczos_resample_u8(const uint8_t* src, int srcW, int srcH, uint8_t* dst, int dstW, int dstH, int a = 3);
    // resample single-channel 16-bit
    void lanczos_resample_u16(const uint16_t* src, int srcW, int srcH, uint16_t* dst, int dstW, int dstH, int a = 3);
}
