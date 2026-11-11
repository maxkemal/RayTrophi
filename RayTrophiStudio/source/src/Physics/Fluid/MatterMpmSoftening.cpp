#include "Fluid/MatterMpmSoftening.h"
#include "Fluid/APICFluidSolver.h"
#include "Fluid/GranularConstitutive.h"
#include "Fluid/MatterSubstanceState.h"
#include "Fluid/SubstanceTag.h"

#include <limits>
#include <vector>

namespace RayTrophiSim::Fluid {

std::size_t updateMatterMpmSoftening(FluidParticles& particles, const APICSolverParams& params) {
    std::vector<uint8_t> mask;
    std::size_t count = 0;
    for (std::size_t index = 0; index < particles.size(); ++index) {
        const uint32_t tag = index < particles.substance_tag.size()
            ? particles.substance_tag[index] : kSubstanceUntagged;
        auto model = index < particles.constitutive_model.size()
            ? static_cast<MatterConstitutiveModel>(particles.constitutive_model[index])
            : MatterConstitutiveModel::Auto;
        if (model == MatterConstitutiveModel::Auto) {
            const float kelvin = index < particles.temperature.size()
                ? particles.temperature[index] : std::numeric_limits<float>::quiet_NaN();
            model = substanceBirthModel(tag, params, kelvin);
        }
        if ((model == MatterConstitutiveModel::Granular ||
             model == MatterConstitutiveModel::Elastic) &&
            substanceTransportOwner(tag, model, params) == MatterTransportOwner::Mpm) {
            if (mask.empty()) {
                mask.resize(particles.size(), 0u);
            }
            mask[index] = 1u;
            ++count;
        }
    }
    if (count == 0) {
        return 0;
    }
    particles.ensureGranularStateSize();
    Granular::SofteningParams softening;
    softening.softening_temperature = params.granular_softening_temperature;
    softening.softening_range = params.granular_softening_range;
    softening.tack_peak = params.granular_tack_peak;
    softening.residual_strength = params.granular_residual_strength;
    Granular::updateSoftening(particles, softening, &mask);
    return count;
}

} // namespace RayTrophiSim::Fluid
