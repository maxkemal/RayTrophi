#pragma once

#include "Fluid/FluidParticleLabels.h"
#include "json.hpp"

namespace pybind11 { class dict; }

nlohmann::json fluidLabelsToJson(const RayTrophiSim::Fluid::ParticleLabelReport& report);
pybind11::dict fluidLabelsToPython(const RayTrophiSim::Fluid::ParticleLabelReport& report);
