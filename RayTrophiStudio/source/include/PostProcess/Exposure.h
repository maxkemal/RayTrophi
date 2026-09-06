#pragma once
#include <array>
#include <cstdint>
#include <string>

namespace rtpost {
// Relative scene-linear exposure: the renderer does not use cd/m2.
struct ExposureSettings {
    int mode = 0; // 0 manual EV, 1 physical camera, 2 histogram
    float ev = 0.0f;
    float min_ev = -12.0f, max_ev = 12.0f;
    float low_percent = 2.0f, high_percent = 98.0f;
    float key = 0.18f;
    float speed_up = 3.0f, speed_down = 1.0f; // exponential rate, 1/seconds
    float center_weight = 0.0f;
    bool locked = false;
    float locked_ev = 0.0f;
};
constexpr unsigned HistogramBins = 256;
constexpr float HistogramMin = -16.0f, HistogramMax = 16.0f;
struct MeterResult {
    std::array<uint32_t, HistogramBins> bins{};
    uint64_t generation = 0;
};
struct ExposureTelemetry {
    float target_ev = 0, applied_ev = 0, metered_luminance = 0;
    bool valid = false;
    uint64_t generation = 1;
    std::string source = "waiting for HDR";
    std::array<uint32_t, HistogramBins> bins{};
};
bool validateExposure(const ExposureSettings&, std::string& error);
float histogramTarget(const MeterResult&, const ExposureSettings&, float& luminance, bool& valid);
float adaptExposure(float current, float target, float dt, const ExposureSettings&);
ExposureSettings meterSettings();
ExposureTelemetry exposureTelemetry();
uint64_t meterGeneration();
bool meterEnabled();
void submitMeter(const MeterResult&, const char* source);
void resetExposure();
// Changes runtime settings without mutating the user's manual exposure multiplier.
void updateExposure(const ExposureSettings&, float dt, bool freeze);
float exposureMultiplier();
void meterCpu(const float* data, int width, int height, unsigned strideFloats);
}
