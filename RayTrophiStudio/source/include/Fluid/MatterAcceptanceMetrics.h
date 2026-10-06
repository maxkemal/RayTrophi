#pragma once

#include <json.hpp>

namespace RayTrophiSim::Fluid {
class FluidParticles;

// Read-only canonical SoA measurements. Shape is not an angle-of-repose estimate.
nlohmann::json inspectMatterAcceptanceMetrics(const FluidParticles& particles,
                                              bool legacy_granular,
                                              float appearance_full_saturation = 1.0f);
} // namespace RayTrophiSim::Fluid
