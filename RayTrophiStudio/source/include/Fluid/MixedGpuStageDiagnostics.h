#pragma once
#include "Fluid/MixedGpuStageStats.h"
#include "json.hpp"

namespace RayTrophiSim::Fluid {

inline nlohmann::json mixedGpuStagesJson(const MixedGpuStageStats& stats) {
    nlohmann::json out{{"measured", stats.measured},
        {"timing_kind", "host_wall_including_waits"},
        {"common_substeps", stats.common_substeps}, {"grid_steps", stats.grid_steps},
        {"liquid_steps", stats.liquid_steps}, {"liquid_multirate", stats.liquid_multirate}};
    out["stages"] = nlohmann::json::object();
    if (!stats.measured) {
        return out;
    }
    for (std::size_t i = 0; i < stats.stages.size(); ++i) {
        const auto& entry = stats.stages[i];
        const auto& t = entry.transfer;
        out["stages"][mixedGpuStageNames[i]] = {
            {"host_wall_ms", entry.host_wall_ms}, {"visits", entry.visits},
            {"dispatch_calls", t.dispatch_calls}, {"batch_end_calls", t.batch_end_calls},
            {"synchronize_calls", t.synchronize_calls}, {"batch_end_ms", t.batch_end_ms},
            {"synchronize_ms", t.synchronize_ms}, {"upload_bytes", t.upload_bytes},
            {"download_bytes", t.download_bytes}, {"upload_calls", t.upload_calls},
            {"download_calls", t.download_calls}, {"dispatch_call_ms", t.dispatch_call_ms}};
    }
    return out;
}

} // namespace RayTrophiSim::Fluid
