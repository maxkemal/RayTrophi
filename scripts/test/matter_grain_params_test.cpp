#include "Fluid/MatterGrain.h"
#include "Fluid/APICFluidSolver.h"
#include "Fluid/FluidParticles.h"
#include "Fluid/SubstanceTag.h"
#include "MaterialStateField.h"

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
    // No enabled switch: grains follow the substances (matterGrainOwnership),
    // so the key is rejected, and an old save that carries it still loads.
    assert(!patchMatterGrainParams({{"enabled", true}}, p, error));
    assert(!matterGrainParamsToJson(p).contains("enabled"));
    assert(matterGrainParamsToJson(matterGrainParamsFromJson({{"enabled", true}})) == initial);
    assert(patchMatterGrainParams({{"radius_m", .05}}, p, error));
    // The grain material moved to the substance: the domain key names its new home.
    assert(!patchMatterGrainParams({{"twisting_friction", .1}}, p, error));
    assert(error.find("grain_twisting_friction") != std::string::npos);
    assert(std::abs(p.tangential_stiffness_ratio - 2.0f / 7.0f) < 1e-7f);
    // Stiffness follows the radius: k = scale x 8e5 N/m per m x r.
    assert(patchMatterGrainParams({{"contact_resolution", 48}, {"stiffness_scale", 2.0}}, p, error));
    assert(p.contact_resolution == 48);
    assert(std::abs(p.stiffness_n_m - 2.0f * kMatterGrainStiffnessPerRadius * .05f) < 1e-2f);
    {
        // An old save keeps its stiffness as a scale; its material is dropped and named.
        std::string dropped;
        const auto old = matterGrainParamsFromJson(
            {{"radius_m", .025}, {"stiffness_n_m", 1e5}, {"friction", .3}}, &dropped);
        assert(std::abs(old.stiffness_n_m - 1e5f) < 1.0f && dropped == "friction");
        // The substance's grain material reaches the solver params.
        RayTrophiSim::SubstanceProfile sand = *RayTrophiSim::tryFindSubstance("Sand");
        sand.grain_friction = .7f;
        sand.grain_packing_fraction = .55f;
        auto applied = old;
        applyMatterGrainSubstance(applied, sand, nullptr, false);
        assert(applied.friction == .7f && applied.packing_fraction == .55f && !applied.wet_grains);
        assert(std::abs(applied.stiffness_n_m - 1e5f) < 1.0f);
    }
    // Grain mass: bulk density / packing * sphere volume. Untagged Water
    // preset liquid density is irrelevant; granular uses dry density, so
    // only check the scaling: doubling radius multiplies mass by eight.
    {
        MatterGrainParams a, b;
        a.radius_m = .02f;
        b.radius_m = .04f;
        const float ma = matterGrainRestMassKg(a, 0u, RayTrophiSim::tryFindSubstance("Sand"),
            MatterConstitutiveModel::Granular, true);
        const float mb = matterGrainRestMassKg(b, 0u, RayTrophiSim::tryFindSubstance("Sand"),
            MatterConstitutiveModel::Granular, true);
        assert(ma > 0.0f && std::abs(mb / ma - 8.0f) < 1e-3f);
        b = a;
        b.packing_fraction = .3f;
        assert(std::abs(matterGrainRestMassKg(b, 0u, RayTrophiSim::tryFindSubstance("Sand"),
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
    particles.constitutive_model[0] = static_cast<uint8_t>(MatterConstitutiveModel::Granular);
    particles.substance_tag[0] = substanceTag("Sand");
    // A colocated MPM skeleton is not part of the grain spin/pile diagnostic.
    particles.emit(Vec3(0.0f), Vec3(0.0f), 293.15f, 0.0f, substanceTag("Soil"),
        nullptr, nullptr, 2.0f, MatterConstitutiveModel::Granular);
    particles.affine[1].col0 = Vec3(0, 100, 0);
    particles.affine[1].col1 = Vec3(-100, 0, 0);
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
        // Only the thin toe (under 1.5 layers) counts as outside the pile.
        assert(pile.at("scattered_fraction").get<double>() < .1);
        const std::size_t toe = pile.at("scattered_grains").get<std::size_t>();
        // Bounced grains: a sparse floor layer out to 1.6 m and a clump
        // against one wall must neither move the axis nor flatten the slope.
        const std::size_t cone = centres.size();
        for (int n = 0; n < 400; ++n) {
            const double a = n * 2.399963, d = 1.2 + .4 * (n % 97) / 97.0;
            centres.push_back(Vec3(float(d * std::cos(a)), r, float(d * std::sin(a))));
        }
        for (float y = r; y < .15f; y += 2 * r) {
            for (float z = -.3f; z <= .3f; z += 2 * r) {
                centres.push_back(Vec3(1.7f, y, z));
            }
        }
        const auto noisy = matterGrainPileProfile(centres, r);
        assert(noisy.at("measured").get<bool>());
        assert(std::abs(noisy.at("repose_angle_deg").get<double>() - degrees) < 3.0);
        assert(noisy.at("scattered_grains").get<std::size_t>() >= centres.size() - cone + toe - 10);
    }
    {
        // A flat spread (one to two layers everywhere) is not a pile.
        std::vector<Vec3> flat;
        for (float x = -1.f; x <= 1.f; x += .04f) {
            for (float z = -1.f; z <= 1.f; z += .04f) {
                flat.push_back(Vec3(x, .02f, z));
            }
        }
        const auto pile = matterGrainPileProfile(flat, .02f);
        assert(!pile.at("measured").get<bool>());
        assert(pile.at("reason").get<std::string>().rfind("no_pile", 0) == 0);
    }
    assert(!matterGrainPileProfile({Vec3(0.0f)}, .02f).at("measured").get<bool>());
    const auto diagnostics = matterGrainDiagnostics(particles, p);
    assert(std::abs(diagnostics.at("max_spin_rad_s").get<double>() - 3.0) < 1e-9);
    assert(std::abs(diagnostics.at("spin_energy_j").get<double>() - .009) < 1e-8);
}
