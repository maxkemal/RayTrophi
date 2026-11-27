#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>

namespace Paint {

// Runtime-adaptive cadence and budget for GPU wet paint.
// Discrete desktop GPUs keep the 60 Hz reference cadence, 120 Hz stamping,
// and uncapped event/coverage rates.
// Low-power GPUs (Android mobile GPUs AND Windows/Linux integrated GPUs such as
// AMD A10/R7 or Intel UHD) automatically use 15 Hz / 10 Hz adaptive stepping,
// 30 Hz stamping, capped per-tick events, and 1 Hz periodic coverage.
struct WetPaintMobileBudget {
    static constexpr double kDesktopStepSeconds = 1.0 / 60.0;
    static constexpr double kDesktopStampIntervalSeconds = 1.0 / 120.0;
    static constexpr double kDesktopMaxStepDtSeconds = 2.5 / 60.0;

    static constexpr double kLowPowerStepSeconds = 1.0 / 15.0;
    static constexpr double kLowPowerHeavyStepSeconds = 1.0 / 10.0;
    static constexpr double kLowPowerStampIntervalSeconds = 1.0 / 30.0;
    static constexpr uint32_t kLowPowerHeavyActiveTileThreshold = 256u;
    static constexpr double kLowPowerMaxStepDtSeconds = 6.0 / 60.0;
    static constexpr uint32_t kLowPowerMaxEventsPerTick = 512u;
    static constexpr double kLowPowerMinCoverageIntervalSeconds = 1.0;

#ifdef __ANDROID__
    static constexpr double kStepSeconds = kLowPowerStepSeconds;
    static constexpr double kStampIntervalSeconds = kLowPowerStampIntervalSeconds;
    static constexpr double kMinCoverageIntervalSeconds = kLowPowerMinCoverageIntervalSeconds;
#else
    static constexpr double kStepSeconds = kDesktopStepSeconds;
    static constexpr double kStampIntervalSeconds = kDesktopStampIntervalSeconds;
    static constexpr double kMinCoverageIntervalSeconds = 0.0;
#endif

    static std::atomic<bool>& lowPowerGpuFlag() {
#ifdef __ANDROID__
        static std::atomic<bool> flag{true};
#else
        static std::atomic<bool> flag{false};
#endif
        return flag;
    }

    static void setLowPowerGpu(bool enabled) {
#ifdef __ANDROID__
        (void)enabled;
        lowPowerGpuFlag().store(true, std::memory_order_release);
#else
        lowPowerGpuFlag().store(enabled, std::memory_order_release);
#endif
    }

    static bool isLowPowerGpu() {
        return lowPowerGpuFlag().load(std::memory_order_acquire);
    }

    static double stepIntervalSeconds(uint32_t active_tiles) {
        if (isLowPowerGpu()) {
            return active_tiles > kLowPowerHeavyActiveTileThreshold
                ? kLowPowerHeavyStepSeconds
                : kLowPowerStepSeconds;
        }
        return kDesktopStepSeconds;
    }

    static double stampIntervalSeconds() {
        return isLowPowerGpu() ? kLowPowerStampIntervalSeconds : kDesktopStampIntervalSeconds;
    }

    static uint32_t clampMaxEventsPerTick(uint32_t requested) {
        uint32_t clamped = std::max<uint32_t>(1u, requested);
        if (isLowPowerGpu()) {
            clamped = std::min<uint32_t>(clamped, kLowPowerMaxEventsPerTick);
        }
        return clamped;
    }

    static float computeStepDt(double step_clock) {
        const double max_dt = isLowPowerGpu() ? kLowPowerMaxStepDtSeconds : kDesktopMaxStepDtSeconds;
        const double base_dt = isLowPowerGpu() ? kLowPowerStepSeconds : kDesktopStepSeconds;
        const double dt = step_clock > 0.0 ? std::min(step_clock, max_dt) : base_dt;
        return static_cast<float>(dt);
    }

    static bool allowPeriodicCoverage(bool requested, double& coverage_clock, double step_dt) {
        coverage_clock += step_dt;
        if (!requested) {
            return false;
        }
        if (isLowPowerGpu()) {
            if (coverage_clock < kLowPowerMinCoverageIntervalSeconds) {
                return false;
            }
            coverage_clock = 0.0;
        }
        return true;
    }
};

} // namespace Paint
