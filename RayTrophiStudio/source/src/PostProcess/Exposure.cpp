#include "PostProcess/Exposure.h"
#include <algorithm>
#include <cmath>
#include <mutex>

namespace rtpost {
namespace {
std::mutex mutex;
ExposureSettings settings;
ExposureTelemetry telemetry;
bool frozen = false;
}
bool validateExposure(const ExposureSettings& s, std::string& error) {
    const float values[] = {s.ev,s.min_ev,s.max_ev,s.low_percent,s.high_percent,s.key,
        s.speed_up,s.speed_down,s.center_weight,s.locked_ev};
    for (float v : values) if (!std::isfinite(v)) { error = "exposure values must be finite"; return false; }
    if (s.mode < 0 || s.mode > 2) error = "unknown exposure mode";
    else if (s.ev < -24 || s.ev > 24 || s.min_ev < -24 || s.max_ev > 24 || s.min_ev > s.max_ev)
        error = "EV must be in [-24,24], with min_ev <= max_ev";
    else if (s.low_percent < 0 || s.high_percent > 100 || s.low_percent >= s.high_percent)
        error = "percentiles require 0 <= low_percent < high_percent <= 100";
    else if (s.key < 0.001f || s.key > 1) error = "key must be in [0.001,1]";
    else if (s.speed_up <= 0 || s.speed_down <= 0 || s.speed_up > 20 || s.speed_down > 20)
        error = "adaptation rates must be in (0,20] per second";
    else if (s.center_weight < 0 || s.center_weight > 1) error = "center_weight must be in [0,1]";
    else if (s.locked_ev < -24 || s.locked_ev > 24) error = "locked_ev must be in [-24,24]";
    else { error.clear(); return true; }
    return false;
}
float histogramTarget(const MeterResult& h, const ExposureSettings& s, float& lum, bool& valid) {
    double total = 0; for (auto n : h.bins) total += n;
    valid = total > 0; lum = 0;
    if (!valid) return 0;
    const double low = total * s.low_percent / 100, high = total * s.high_percent / 100;
    double cursor = 0, weight = 0, sum = 0;
    for (unsigned i = 0; i < HistogramBins; ++i) {
        const double end = cursor + h.bins[i];
        const double n = (std::max)(0.0, (std::min)(end, high) - (std::max)(cursor, low));
        sum += n * (HistogramMin + (i + 0.5) * (HistogramMax-HistogramMin)/HistogramBins);
        weight += n; cursor = end;
    }
    valid = weight > 0;
    if (!valid) return 0;
    lum = static_cast<float>(std::exp2(sum / weight));
    return std::clamp(std::log2(s.key / lum), s.min_ev, s.max_ev);
}
float adaptExposure(float current, float target, float dt, const ExposureSettings& s) {
    if (!std::isfinite(dt) || dt <= 0) return current;
    const float rate = target < current ? s.speed_up : s.speed_down;
    return current + (target-current) * -std::expm1(-rate * (std::min)(dt, 1.0f));
}
ExposureSettings meterSettings() { std::lock_guard<std::mutex> l(mutex); return settings; }
ExposureTelemetry exposureTelemetry() { std::lock_guard<std::mutex> l(mutex); return telemetry; }
uint64_t meterGeneration() { std::lock_guard<std::mutex> l(mutex); return telemetry.generation; }
bool meterEnabled() { std::lock_guard<std::mutex> l(mutex); return settings.mode == 2 && !settings.locked && !frozen; }
void resetExposure() {
    std::lock_guard<std::mutex> l(mutex);
    const auto generation = telemetry.generation + 1;
    telemetry = {}; telemetry.generation = generation;
}
void submitMeter(const MeterResult& h, const char* source) {
    std::lock_guard<std::mutex> l(mutex);
    if (h.generation != telemetry.generation || settings.mode != 2 || settings.locked || frozen) return;
    bool valid; float lum;
    const float target = histogramTarget(h, settings, lum, valid);
    if (!valid) return; // An empty/invalid frame cannot turn the viewport white.
    telemetry.target_ev = target; telemetry.metered_luminance = lum;
    telemetry.valid = true; telemetry.source = source; telemetry.bins = h.bins;
}
void updateExposure(const ExposureSettings& s, float dt, bool freeze) {
    std::lock_guard<std::mutex> l(mutex);
    if (s.mode != settings.mode || s.center_weight != settings.center_weight ||
        s.low_percent != settings.low_percent || s.high_percent != settings.high_percent || s.key != settings.key) {
        const auto generation = telemetry.generation + 1;
        const float applied = s.mode==settings.mode ? telemetry.applied_ev : 0.0f;
        telemetry = {}; telemetry.generation = generation; telemetry.applied_ev=applied;
    }
    settings = s; frozen = freeze;
    telemetry.target_ev = std::clamp(telemetry.target_ev, s.min_ev, s.max_ev);
    if (s.locked) telemetry.applied_ev = s.locked_ev;
    else if (!freeze && telemetry.valid) telemetry.applied_ev = std::clamp(adaptExposure(telemetry.applied_ev, telemetry.target_ev, dt, s),s.min_ev,s.max_ev);
}
float exposureMultiplier() {
    std::lock_guard<std::mutex> l(mutex);
    return std::exp2(settings.ev + (settings.mode == 2 ? telemetry.applied_ev : 0.0f));
}
void meterCpu(const float* data, int width, int height, unsigned stride) {
    if (!data || width <= 0 || height <= 0 || stride < 3 || !meterEnabled()) return;
    MeterResult h; h.generation = meterGeneration(); const auto s = meterSettings();
    const int nx = (std::min)(width, 128), ny = (std::min)(height, 128);
    for (int j=0;j<ny;++j) for(int i=0;i<nx;++i) {
        const int x = (2*i+1)*width/(2*nx), y = (2*j+1)*height/(2*ny);
        const float* p = data + (static_cast<size_t>(y)*width+x)*stride;
        if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2])) continue;
        const float lum = .2126f*(std::max)(p[0],0.0f)+.7152f*(std::max)(p[1],0.0f)+.0722f*(std::max)(p[2],0.0f);
        if (lum <= 1e-8f) continue;
        const int bin = std::clamp(int((std::log2(lum)-HistogramMin)*8.0f),0,255);
        const float u = (i+.5f)/nx*2-1, v = (j+.5f)/ny*2-1;
        const float central = (std::max)(0.0f,1.0f-.5f*(u*u+v*v));
        h.bins[bin] += uint32_t(1+255*((1-s.center_weight)+s.center_weight*central*central));
    }
    submitMeter(h, "CPU HDR histogram");
}
}
