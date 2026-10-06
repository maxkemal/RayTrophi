#include "Fluid/MatterGrain.h"
#include "Fluid/APICFluidSolver.h"
#include "Fluid/FluidParticles.h"

#include <cassert>
#include <cmath>

int main() {
    using namespace RayTrophiSim::Fluid;
    MatterGrainParams p;
    const auto initial = matterGrainParamsToJson(p);
    std::string error;
    for (const auto& patch : {nlohmann::json{{"enabled", 1}},
                             nlohmann::json{{"radius_m", 0}},
                             nlohmann::json{{"max_substeps", 1.5}},
                             nlohmann::json{{"max_substeps", 4294967296ull}},
                             nlohmann::json{{"unknown", true}},
                             nlohmann::json{{"friction", -1}},
                             nlohmann::json{{"twisting_friction", -1}},
                             nlohmann::json{{"twisting_friction", 1.1}},
                             nlohmann::json{{"twisting_friction", true}},
                             nlohmann::json{{"tangential_stiffness_ratio", -.1}},
                             nlohmann::json{{"tangential_stiffness_ratio", 1.5}},
                             nlohmann::json{{"contact_resolution", 7}},
                             nlohmann::json{{"contact_resolution", 24.5}},
                             nlohmann::json{{"packing_fraction", .2}},
                             nlohmann::json{{"packing_fraction", .8}}}) {
        assert(!patchMatterGrainParams(patch, p, error));
        assert(matterGrainParamsToJson(p) == initial);
    }
    assert(patchMatterGrainParams({{"enabled", true}, {"radius_m", .05}}, p, error));
    assert(patchMatterGrainParams({{"twisting_friction", .1}}, p, error));
    assert(std::abs(p.tangential_stiffness_ratio - 2.0f / 7.0f) < 1e-7f);
    assert(patchMatterGrainParams({{"tangential_stiffness_ratio", 0.0},
        {"contact_resolution", 48}, {"packing_fraction", .64}}, p, error));
    assert(p.contact_resolution == 48 && p.tangential_stiffness_ratio == 0.0f);
    // Grain mass: bulk density / packing * sphere volume. Untagged Water
    // preset liquid density is irrelevant; granular uses dry density, so
    // only check the scaling: doubling radius multiplies mass by eight.
    {
        MatterGrainParams a, b;
        a.radius_m = .02f;
        b.radius_m = .04f;
        const float ma = matterGrainRestMassKg(a, 0u, FluidChemistryPreset::Water,
            MatterConstitutiveModel::Granular, true);
        const float mb = matterGrainRestMassKg(b, 0u, FluidChemistryPreset::Water,
            MatterConstitutiveModel::Granular, true);
        assert(ma > 0.0f && std::abs(mb / ma - 8.0f) < 1e-3f);
        b = a;
        b.packing_fraction = .3f;
        assert(std::abs(matterGrainRestMassKg(b, 0u, FluidChemistryPreset::Water,
            MatterConstitutiveModel::Granular, true) / ma - 2.0f) < 1e-3f);
    }
    assert(matterGrainParamsToJson(matterGrainParamsFromJson(matterGrainParamsToJson(p))) ==
        matterGrainParamsToJson(p));
    FluidParticles particles;
    particles.resizeAll(1);
    particles.position[0] = Vec3(0.0f);
    particles.velocity[0] = Vec3(0.0f);
    particles.rest_mass_kg[0] = 2.0f;
    particles.mass_fraction[0] = 1.0f;
    particles.affine[0].col0 = Vec3(0, 3, 0);
    particles.affine[0].col1 = Vec3(-3, 0, 0);
    particles.affine[0].col2 = Vec3(0);
    // Pile profile on a synthetic 30 degree cone of stacked grains.
    for (double degrees : {20.0, 30.0, 38.0}) {
        const float r = .02f;
        const double height = .4, slope = std::tan(degrees / 57.29577951308232);
        std::vector<Vec3> centres;
        for (float x = -1.f; x <= 1.f; x += 2 * r) {
            for (float z = -1.f; z <= 1.f; z += 2 * r) {
                const double surface = height - std::hypot(x, z) * slope;
                for (float y = r; y + r <= surface; y += 2 * r) {
                    centres.push_back(Vec3(x, y, z));
                }
            }
        }
        const auto pile = matterGrainPileProfile(centres, r);
        assert(pile.at("measured").get<bool>());
        assert(std::abs(pile.at("repose_angle_deg").get<double>() - degrees) < 3.0);
    }
    assert(!matterGrainPileProfile({Vec3(0.0f)}, .02f).at("measured").get<bool>());
    const auto diagnostics = matterGrainDiagnostics(particles, p);
    assert(std::abs(diagnostics.at("max_spin_rad_s").get<double>() - 3.0) < 1e-9);
    assert(std::abs(diagnostics.at("spin_energy_j").get<double>() - .009) < 1e-8);
}
