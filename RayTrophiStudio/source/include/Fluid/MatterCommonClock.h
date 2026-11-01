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
    uint32_t contact_substeps = 0;
    std::function<bool(std::size_t, std::string&)> begin;
    MatterCommonStep forces;
    MatterCommonStep contact;
};

// The DEM runtime prepares once, lends a device step to the continuum clock,
// and publishes once after the clock has advanced every owner successfully.
struct MatterGrainCommonDriver {
    std::function<bool(uint32_t, const MatterCommonStep&, std::string&)> run;
};

// Equal continuum intervals contain ceil(contact/continuum) DEM ticks each.
// Their rounded product must obey the authored ceiling as well. Advance only
// across quotient boundaries: increasing continuum while per stays unchanged
// cannot make a product that is already too large fit the ceiling.
inline bool fitMatterContactSubsteps(uint32_t contact, uint32_t limit,
                                    uint32_t& continuum, std::string& error) {
    if (contact == 0) {
        error.clear();
        return true;
    }
    if (continuum == 0 || (continuum & 1u) || continuum > limit ||
        (contact & 1u) || contact > limit) {
        error = "Matter contact clock needs an even request within the substep limit";
        return false;
    }
    uint64_t candidate = continuum;
    while (candidate <= limit) {
        const uint64_t per = (uint64_t(contact) + candidate - 1u) / candidate;
        if (candidate * per <= limit) {
            continuum = static_cast<uint32_t>(candidate);
            error.clear();
            return true;
        }
        // per==1 already fits because candidate<=limit. The next quotient's
        // first even continuum count preserves both CFL and bank-0 publication.
        candidate = (uint64_t(contact) + per - 2u) / (per - 1u);
        candidate += candidate & 1u;
    }
    error = "Matter contact clock cannot fit the authored substep limit";
    return false;
}

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
        // The grains subcycle inside the continuum (no minimum is passed), so
        // this is the continuum's own CFL or elastic wave bound (liquid
        // speed, MPM stiffness), not the grains.
        error = "Matter common clock: the liquid/MPM continuum needs " +
            std::to_string(uint32_t(requested)) + " substeps per frame (its own CFL or "
            "elastic wave bound) > the Contact substep limit " +
            std::to_string(hooks->authored_max_substeps) + "; raise the limit";
        return false;
    }
    uint32_t continuum = static_cast<uint32_t>(requested);
    if (hooks) {
        const uint32_t limit = hooks->authored_max_substeps
            ? std::min(hooks->authored_max_substeps,
                       static_cast<uint32_t>(std::numeric_limits<int>::max() - 1))
            : static_cast<uint32_t>(std::numeric_limits<int>::max() - 1);
        if (!fitMatterContactSubsteps(hooks->contact_substeps, limit, continuum, error)) {
            return false;
        }
    }
    substeps = continuum;
    error.clear();
    return true;
}

} // namespace RayTrophiSim::Fluid
