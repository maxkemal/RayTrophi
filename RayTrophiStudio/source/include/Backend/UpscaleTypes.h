#pragma once

#include <cstdint>
#include <string>

namespace VulkanRT {
enum class UpscaleMode { Off, Fsr, Auto };
enum class UpscaleQuality { Quality = 1, Balanced, Performance, UltraPerformance };

struct UpscaleSettings {
    UpscaleMode mode = UpscaleMode::Off;
    UpscaleQuality quality = UpscaleQuality::Quality;
    float sharpness = 0.2f;
    // Clustered raygen history rejections (motion AOV w) also drive FSR's
    // reactive mask. Content moving without a matching vector (sim particles)
    // otherwise left ghost trails under FSR while SVGF alone was clean.
    bool rejection_reactive = true;
};

struct UpscaleStats {
    bool available = false;
    bool dll_loaded = false;
    bool active = false;
    bool moving = false;
    bool bridge_active = false;
    bool gpu_timing_available = false;
    uint32_t render_w = 0, render_h = 0, display_w = 0, display_h = 0;
    int jitter_phase_count = 0;
    float jitter_x = 0.0f, jitter_y = 0.0f;
    float effective_scale = 1.0f;
    double gpu_ms = 0.0;
    double svgf_filter_ms = 0.0;   // 0 unless input_provider is "svgf"
    // Who produced FSR's colour input: "svgf" (denoised HDR) or "none" (the
    // raw accumulation, noisy on path-traced scenes). Last dispatch's provider,
    // or the configured one before any dispatch. Any future provider (per-frame
    // OIDN, ...) composes into raw under VulkanRawGuard and reports here.
    std::string input_provider = "none";
    uint64_t dispatches = 0, history_resets = 0, failures = 0, rejected_slots = 0;
    uint64_t epoch = 0;
    int sample = 0;
    std::string ffx_version;
    std::string reason = "not initialized";
    std::string inactive_reason = "mode is off";
};
} // namespace VulkanRT
