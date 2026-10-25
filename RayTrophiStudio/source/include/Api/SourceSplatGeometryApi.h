#pragma once

#include "Fluid/SourceSplatGeometry.h"
#include <memory>

class SceneCommand;

namespace RayTrophiSim {
struct SimulationGridDomainDesc;
}

namespace rtapi {
struct FluidDomainInfo;
void finalizeSourceSplatProxyStats(const RayTrophiSim::SimulationGridDomainDesc& domain,
    FluidDomainInfo& info);
struct Result;
Result validateSourceSplatGeometry(const RayTrophiSim::Fluid::SourceSplatGeometry& geometry);
// Shared command factory lets compound authoring operations retain one undo step.
Result makeSourceSplatGeometryCommand(const std::string& source,
    const RayTrophiSim::Fluid::SourceSplatGeometry& geometry,
    std::unique_ptr<SceneCommand>& command);
Result setSourceSplatGeometry(const std::string& source,
    const RayTrophiSim::Fluid::SourceSplatGeometry& geometry);
Result getSubstanceSplatGeometry(const std::string& domain, const std::string& substance,
    RayTrophiSim::Fluid::SourceSplatGeometry& geometry);
Result setSubstanceSplatGeometry(const std::string& domain, const std::string& substance,
    const RayTrophiSim::Fluid::SourceSplatGeometry& geometry);
}
