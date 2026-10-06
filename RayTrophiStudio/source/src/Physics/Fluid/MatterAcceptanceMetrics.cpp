#include "Fluid/MatterAcceptanceMetrics.h"
#include "Fluid/FluidParticles.h"
#include "Fluid/MatterWetResponse.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace RayTrophiSim::Fluid {
namespace {
struct SampleTotals {
    std::size_t count = 0;
    double dry_mass = 0.0;
    double pore_mass = 0.0;
    double capacity = 0.0;
    double kinetic_energy = 0.0;
    double radial_second_moment = 0.0;
    std::array<double, 3> center_sum{};
    std::array<float, 3> lower{};
    std::array<float, 3> upper{};

    void add(const Vec3& position, const Vec3& velocity, double dry, double pore,
             double pore_capacity) {
        const std::array<float, 3> point{position.x, position.y, position.z};
        for (std::size_t axis = 0; axis < point.size(); ++axis) {
            lower[axis] = count ? std::min(lower[axis], point[axis]) : point[axis];
            upper[axis] = count ? std::max(upper[axis], point[axis]) : point[axis];
            center_sum[axis] += dry * point[axis];
        }
        radial_second_moment += dry * (double(position.x) * position.x +
            double(position.z) * position.z);
        kinetic_energy += 0.5 * (dry + pore) * (double(velocity.x) * velocity.x +
            double(velocity.y) * velocity.y + double(velocity.z) * velocity.z);
        dry_mass += dry;
        pore_mass += pore;
        capacity += pore_capacity;
        ++count;
    }

    nlohmann::json json() const {
        nlohmann::json result{{"particles", count}, {"dry_mass_kg", dry_mass},
            {"pore_water_kg", pore_mass}, {"capacity_kg", capacity},
            {"mean_saturation", capacity > 0.0 ? pore_mass / capacity : 0.0},
            {"kinetic_energy_j", kinetic_energy}, {"bounds_min", nullptr},
            {"bounds_max", nullptr}, {"dry_center_of_mass", nullptr},
            {"horizontal_rms_radius_m", nullptr}};
        if (count) {
            result["bounds_min"] = lower;
            result["bounds_max"] = upper;
        }
        if (dry_mass > 0.0) {
            std::array<double, 3> center{};
            for (std::size_t axis = 0; axis < center.size(); ++axis) {
                center[axis] = center_sum[axis] / dry_mass;
            }
            result["dry_center_of_mass"] = center;
            result["horizontal_rms_radius_m"] = std::sqrt(std::max(0.0,
                radial_second_moment / dry_mass - center[0] * center[0] -
                center[2] * center[2]));
        }
        return result;
    }
};
} // namespace

nlohmann::json inspectMatterAcceptanceMetrics(const FluidParticles& particles,
                                              bool legacy_granular,
                                              float appearance_full_saturation) {
    if (!std::isfinite(appearance_full_saturation) ||
        appearance_full_saturation < 0.001f || appearance_full_saturation > 1.0f) {
        throw std::invalid_argument("Matter appearance saturation must be in [0.001,1]");
    }
    const auto count = particles.size();
    if (particles.velocity.size() != count || particles.constitutive_model.size() != count ||
        particles.rest_mass_kg.size() != count || particles.mass_fraction.size() != count ||
        particles.pore_water_mass_kg.size() != count || particles.pore_capacity_kg.size() != count) {
        throw std::invalid_argument("Matter acceptance sidecar cardinality is invalid");
    }
    SampleTotals granular;
    std::array<SampleTotals, 8> bands;
    std::size_t exactly_dry = 0;
    for (std::size_t i = 0; i < count; ++i) {
        auto model = static_cast<MatterConstitutiveModel>(particles.constitutive_model[i]);
        if (model == MatterConstitutiveModel::Auto) {
            model = legacy_granular ? MatterConstitutiveModel::Granular
                : MatterConstitutiveModel::Fluid;
        }
        if (model != MatterConstitutiveModel::Granular) {
            continue;
        }
        const auto& position = particles.position[i];
        const auto& velocity = particles.velocity[i];
        const double rest_mass = particles.rest_mass_kg[i];
        const double fraction = particles.mass_fraction[i];
        const double pore = particles.pore_water_mass_kg[i];
        const double capacity = particles.pore_capacity_kg[i];
        if (!std::isfinite(position.x) || !std::isfinite(position.y) ||
            !std::isfinite(position.z) || !std::isfinite(velocity.x) ||
            !std::isfinite(velocity.y) || !std::isfinite(velocity.z) ||
            !std::isfinite(rest_mass) || !std::isfinite(fraction) ||
            !std::isfinite(pore) || !std::isfinite(capacity) || rest_mass < 0.0 ||
            fraction < 0.0 || pore < 0.0 || capacity < 0.0 ||
            pore > capacity * 1.00001 || (pore > 0.0 && capacity <= 0.0)) {
            throw std::invalid_argument("Matter acceptance granular state is invalid");
        }
        const double dry = rest_mass * fraction;
        const auto band = matterWetAppearanceBand(matterParticleSaturation(particles, i),
            appearance_full_saturation);
        granular.add(position, velocity, dry, pore, capacity);
        bands[band].add(position, velocity, dry, pore, capacity);
        exactly_dry += pore == 0.0;
    }
    auto band_json = nlohmann::json::array();
    for (std::size_t band = 0; band < bands.size(); ++band) {
        auto value = bands[band].json();
        value["band"] = band;
        band_json.push_back(std::move(value));
    }
    return {{"measured", true}, {"granular", granular.json()},
        {"appearance_quantization", kMatterWetAppearanceQuantization},
        {"appearance_full_saturation", appearance_full_saturation},
        {"exactly_dry_particles", exactly_dry}, {"saturation_bands", band_json},
        {"shape_semantics", "material_point_bounds_and_dry_mass_moments"}};
}
} // namespace RayTrophiSim::Fluid
