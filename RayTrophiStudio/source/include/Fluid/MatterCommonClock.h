#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <functional>
#include <limits>
#include <string>

namespace RayTrophiSim::Fluid {

using MatterCommonStep =
    std::function<bool(uint32_t, uint32_t, float, std::string&)>;

struct MatterCommonClockHooks {
    uint32_t minimum_substeps = 0;
    uint32_t authored_max_substeps = 0;
    std::function<bool(std::size_t, std::string&)> begin;
    MatterCommonStep forces;
    MatterCommonStep contact;
};

// The DEM runtime prepares once, lends a device step to the continuum clock,
// and publishes once after the clock has advanced every owner successfully.
struct MatterGrainCommonDriver {
    std::function<bool(uint32_t, const MatterCommonStep&, std::string&)> run;
};

inline bool resolveMatterCommonSubsteps(double continuum_request,
    const MatterCommonClockHooks* hooks, uint32_t& substeps, std::string& error) {
    if (!std::isfinite(continuum_request) || continuum_request < 0.0) {
        error = "Matter common clock received a nonfinite or negative CFL request";
        return false;
    }
    double requested = std::max(1.0, std::ceil(continuum_request));
    if (hooks) {
        requested = std::max(requested, double(hooks->minimum_substeps));
        // DEM ping-pong ends in its canonical bank, without a state copy.
        requested = 2.0 * std::ceil(requested / 2.0);
    }
    if (!std::isfinite(requested) ||
        requested > double(std::numeric_limits<int>::max() - 1)) {
        error = "Matter common clock exceeds the representable substep index";
        return false;
    }
    if (hooks && hooks->authored_max_substeps &&
        requested > hooks->authored_max_substeps) {
        error = "Matter common clock needs " + std::to_string(uint32_t(requested)) +
            " substeps; raise the domain's authored grain max_substeps (" +
            std::to_string(hooks->authored_max_substeps) + ")";
        return false;
    }
    substeps = static_cast<uint32_t>(requested);
    return true;
}

} // namespace RayTrophiSim::Fluid
