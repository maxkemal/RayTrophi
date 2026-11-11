#pragma once

#include "MatterConstitutive.h"
#include "json.hpp"
#include <limits>

namespace RayTrophiSim {
struct SubstanceProfile;
namespace Fluid {

// Preserve the existing fusion switching point; this is not an enthalpy solver.
constexpr float kFusionSolidThreshold = 0.5f;

struct MatterConstitutiveState {
    MatterConstitutiveModel model = MatterConstitutiveModel::Fluid;
    const char* phase = "unknown";
    const char* phase_basis = "unknown";
    const char* basis = "domain_fallback";
    bool fusion_fraction_used = false;
};

bool hasMatterSolidSkeleton(const SubstanceProfile* profile);

// Canonical existing birth/fusion selection. Missing temperature/fusion values
// remain unknown; fallback model behavior is preserved for legacy particles.
MatterConstitutiveState resolveMatterConstitutiveState(const SubstanceProfile* profile,
    bool granular_fallback, float kelvin,
    float solid_fraction = std::numeric_limits<float>::quiet_NaN());

// Describes implemented particle mechanics, not an object's surface/rigid-body
// behavior. In particular, supported pinned freezing is not deformable MPM.
nlohmann::json describeMatterPhaseMechanics(const SubstanceProfile& profile);

} // namespace Fluid
} // namespace RayTrophiSim
