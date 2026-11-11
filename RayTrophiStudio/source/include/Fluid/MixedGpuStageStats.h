#pragma once

#include "SimulationCompute.h"
#include <array>
#include <chrono>

namespace RayTrophiSim::Fluid {

enum class MixedGpuStage : std::size_t {
    Setup, Forces, Occupancy, LiquidTransfer, MpmTransfer, Flip,
    Viscosity, Pressure, Contact, Gather, Advection, Publication, Count
};
inline constexpr std::array<const char*, static_cast<std::size_t>(MixedGpuStage::Count)>
    mixedGpuStageNames = {"setup", "forces", "occupancy", "liquid_transfer", "mpm_transfer",
        "flip_snapshot", "viscosity", "pressure", "contact", "gather", "advection",
        "publication"};

struct MixedGpuStageEntry {
    double host_wall_ms = 0.0;
    uint32_t visits = 0;
    SimulationComputeContext::TransferStats transfer;
};

struct MixedGpuStageStats {
    bool measured = false;
    int common_substeps = 0;
    int grid_steps = 0;
    // Liquid transfers + projections this frame (== grid_steps unless the
    // liquid runs on its own clock: always, when liquid and MPM share the
    // domain -- the synchronous path was removed 2026-11-11).
    int liquid_steps = 0;
    bool liquid_multirate = false;
    std::array<MixedGpuStageEntry, mixedGpuStageNames.size()> stages;
};

inline void addMixedTransferStats(SimulationComputeContext::TransferStats& to,
                                 const SimulationComputeContext::TransferStats& from) {
    to.measured = to.measured || from.measured;
#define RT_MIXED_ADD(field) to.field += from.field
    RT_MIXED_ADD(upload_bytes);
    RT_MIXED_ADD(download_bytes);
    RT_MIXED_ADD(upload_calls);
    RT_MIXED_ADD(download_calls);
    RT_MIXED_ADD(dispatch_calls);
    RT_MIXED_ADD(batch_end_calls);
    RT_MIXED_ADD(synchronize_calls);
    RT_MIXED_ADD(upload_call_ms);
    RT_MIXED_ADD(download_call_ms);
    RT_MIXED_ADD(dispatch_call_ms);
    RT_MIXED_ADD(batch_end_ms);
    RT_MIXED_ADD(synchronize_ms);
#undef RT_MIXED_ADD
}

// Host wall time includes submission and readback waits, not GPU kernel time.
// Switching stages never synchronizes. Nested transfer probes are restored and
// their counters accumulated into the surrounding frame probe exactly once.
class MixedGpuStageProbe {
public:
    MixedGpuStageProbe(SimulationComputeContext* compute, MixedGpuStageStats& stats)
        : compute_(compute), stats_(stats), start_(Clock::now()) {
        stats_ = {};
        stats_.measured = true;
        stats_.stages[0].visits = 1;
        stats_.stages[0].transfer.measured = true;
        if (compute_) {
            previous_ = compute_->setTransferProbe(&stats_.stages[0].transfer);
        }
    }
    ~MixedGpuStageProbe() { finish(); }
    MixedGpuStageProbe(const MixedGpuStageProbe&) = delete;
    MixedGpuStageProbe& operator=(const MixedGpuStageProbe&) = delete;

    void select(MixedGpuStage stage) {
        if (finished_) {
            return;
        }
        closeInterval();
        active_ = static_cast<std::size_t>(stage);
        auto& entry = stats_.stages[active_];
        ++entry.visits;
        entry.transfer.measured = true;
        if (compute_) {
            compute_->setTransferProbe(&entry.transfer);
        }
    }
    void finish() {
        if (finished_) {
            return;
        }
        closeInterval();
        if (compute_) {
            compute_->setTransferProbe(previous_);
        }
        if (previous_) {
            for (const auto& entry : stats_.stages) {
                addMixedTransferStats(*previous_, entry.transfer);
            }
        }
        finished_ = true;
    }

private:
    using Clock = std::chrono::steady_clock;
    void closeInterval() {
        const auto now = Clock::now();
        stats_.stages[active_].host_wall_ms +=
            std::chrono::duration<double, std::milli>(now - start_).count();
        start_ = now;
    }
    SimulationComputeContext* compute_;
    MixedGpuStageStats& stats_;
    SimulationComputeContext::TransferStats* previous_ = nullptr;
    Clock::time_point start_;
    std::size_t active_ = 0;
    bool finished_ = false;
};

} // namespace RayTrophiSim::Fluid
