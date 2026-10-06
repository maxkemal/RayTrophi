#include "Fluid/MatterGranularLoad.h"

#include <algorithm>
#include <cmath>

namespace RayTrophiSim::Fluid {
Granular::LoadMeasurement measureMatterGranularLoad(const FluidParticles& particles,
    bool legacy_granular, float gravity, float density) {
    Granular::LoadMeasurement result;
    bool have_extent = false;
    float lowest = 0.0f;
    float highest = 0.0f;
    float worst_squared = 0.0f;
    for (std::size_t i = 0; i < particles.size(); ++i) {
        auto model = i < particles.constitutive_model.size()
            ? static_cast<MatterConstitutiveModel>(particles.constitutive_model[i])
            : MatterConstitutiveModel::Auto;
        if (model == MatterConstitutiveModel::Auto) {
            model = legacy_granular ? MatterConstitutiveModel::Granular
                : MatterConstitutiveModel::Fluid;
        }
        if (model != MatterConstitutiveModel::Granular) {
            continue;
        }
        const float y = particles.position[i].y;
        if (std::isfinite(y)) {
            lowest = have_extent ? std::min(lowest, y) : y;
            highest = have_extent ? std::max(highest, y) : y;
            have_extent = true;
        }
        if (i < particles.affine.size()) {
            const auto& c = particles.affine[i];
            const float squared = c.col0.x * c.col0.x + c.col0.y * c.col0.y +
                c.col0.z * c.col0.z + c.col1.x * c.col1.x + c.col1.y * c.col1.y +
                c.col1.z * c.col1.z + c.col2.x * c.col2.x + c.col2.y * c.col2.y +
                c.col2.z * c.col2.z;
            if (std::isfinite(squared)) {
                worst_squared = std::max(worst_squared, squared);
            }
        }
        if (i < particles.granular_softening.size()) {
            const float softening = particles.granular_softening[i];
            if (std::isfinite(softening)) {
                result.softening_min = std::min(result.softening_min,
                    std::clamp(softening, 0.0f, 1.0f));
                if (softening < 0.999f) {
                    ++result.softened_particles;
                }
            }
        }
    }
    result.strain_rate = std::sqrt(worst_squared);
    result.column_height = have_extent ? std::max(highest - lowest, 0.0f) : 0.0f;
    result.overburden_pressure = result.column_height * std::abs(gravity) *
        std::max(density, 1.0f);
    return result;
}
} // namespace RayTrophiSim::Fluid
