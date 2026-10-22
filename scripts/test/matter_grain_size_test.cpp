#include "Fluid/MatterGrain.h"
#include "Fluid/MatterGrainBirth.h"
#include "Fluid/MatterGrainCoupling.h"
#include "MaterialStateField.h"

#include <cassert>
#include <cmath>
#include <numeric>

using namespace RayTrophiSim;
using namespace RayTrophiSim::Fluid;

int main() {
    // Bound actual unequal sphere pairs and wall contacts, including wet inertial
    // mass variation. A global minimum-mass / maximum-radius pairing is not needed.
    const double radii[] = {0.011, 0.016, 0.026, 0.036};
    constexpr double alpha = 800000.0;
    double maximum_rate = 0.0;
    double minimum_mass = 1e30;
    for (double radius : radii) {
        const double mass = 10000.0 * radius * radius * radius;
        minimum_mass = std::min(minimum_mass, mass);
        const double bound = matterGrainContactRateBound(radius, 0.036, mass, alpha);
        assert(alpha * radius / mass <= bound * (1.0 + 1e-12));
        maximum_rate = std::max(maximum_rate, bound);
    }
    for (double a : radii) {
        for (double b : radii) {
            const double ma = 10000.0 * a * a * a;
            const double mb = 10000.0 * b * b * b;
            const double pair_k = alpha * 2.0 * a * b / (a + b);
            assert(pair_k * (1.0 / ma + 1.0 / mb) <=
                2.0 * maximum_rate * (1.0 + 1e-12));
        }
    }
    const double old_rate = alpha * 0.036 / minimum_mass;
    assert(maximum_rate < old_rate * 0.5);
    const double equal_mass = 0.12;
    assert(std::abs(matterGrainContactRateBound(0.025, 0.025, equal_mass, alpha) -
        alpha * 0.025 / equal_mass) < 1e-8);
    // Explicit reset must not carry a dense pile's CFL/sleep/history into a
    // new sparse population. Device allocation sizes remain reusable.
    MatterGrainGpuRuntime stale;
    stale.cfl_contact_hint = 63;
    stale.history_fresh = false;
    stale.coupling_zero = true;
    stale.sleep_context_valid = true;
    stale.published_ids = {1, 2};
    stale.history_block = {0, 1};
    stale.published_positions = {Vec3(0), Vec3(1)};
    stale.published_masses = {1, 2};
    stale.capacity = 1024;
    stale.requested_contact_capacity = 128;
    invalidateMatterGrainAfterReset(stale);
    assert(stale.cfl_contact_hint == kMatterGrainContactBudget);
    assert(stale.history_fresh && !stale.coupling_zero && !stale.sleep_context_valid);
    assert(stale.published_ids.empty() && stale.history_block.empty());
    assert(stale.published_positions.empty() && stale.published_masses.empty());
    assert(stale.capacity == 1024 && stale.requested_contact_capacity == 128);
    MatterGrainParams settings;
    std::string error;
    const auto legacy = matterGrainParamsFromJson({{"radius_m", 0.01f}});
    assert(matterGrainRadius(legacy, 1) == matterGrainRadius(legacy, 987654));
    assert(patchMatterGrainParams({{"radius_m", 0.01f}, {"radius_max_m", 0.02f},
        {"size_distribution", "bimodal"}, {"large_grain_fraction", 0.25f}}, settings, error));
    int large = 0;
    for (uint64_t id = 1; id <= 10000; ++id) {
        const float radius = matterGrainRadius(settings, id);
        assert(radius == settings.radius_m || radius == settings.radius_max_m);
        large += radius == settings.radius_max_m;
        const auto loaded = matterGrainParamsFromJson(matterGrainParamsToJson(settings));
        assert(radius == matterGrainRadius(loaded, id));
    }
    assert(large > 2300 && large < 2700);
    for (const char* mode : {"uniform", "log_uniform"}) {
        settings.size_distribution = mode;
        double mean = 0;
        for (uint64_t id = 1; id <= 10000; ++id) {
            const float radius = matterGrainRadius(settings, id);
            assert(radius >= 0.01f && radius <= 0.02f);
            mean += radius / 10000.0;
        }
        const double expected = settings.size_distribution == "uniform"
            ? 0.015 : 0.01 / std::log(2.0);
        assert(std::abs(mean - expected) < 0.0002);
    }
    const auto before = matterGrainParamsToJson(settings);
    for (const auto& invalid : {nlohmann::json{{"radius_max_m", 0.005}},
        nlohmann::json{{"radius_max_m", 0.041}}, nlohmann::json{{"size_seed", true}},
        nlohmann::json{{"size_seed", -1}}, nlohmann::json{{"size_seed", 0.5}},
        nlohmann::json{{"size_distribution", "bad"}},
        nlohmann::json{{"large_grain_fraction", 1.01}}}) {
        assert(!patchMatterGrainParams(invalid, settings, error));
        assert(matterGrainParamsToJson(settings) == before);
    }
    // Doubling radius must multiply mass and wet capacity by eight.
    SubstanceProfile substance;
    substance.density = 1600.0f;
    auto small = settings;
    small.radius_m = 0.01f;
    auto big = small;
    big.radius_m = 0.02f;
    const float m1 = matterGrainRestMassKg(small, 0, &substance,
        MatterConstitutiveModel::Granular, false);
    const float m2 = matterGrainRestMassKg(big, 0, &substance,
        MatterConstitutiveModel::Granular, false);
    assert(m1 > 0 && std::abs(m2 / m1 - 8.0f) < 0.001f);
    assert(std::abs(matterGrainWaterCapacityKg(big) /
        matterGrainWaterCapacityKg(small) - 8.0f) < 0.001f);

    FluidSim::FluidGrid grid(20, 20, 20, 0.1f, Vec3(-1));
    FluidParticles empty;
    MatterGrainBirthFilter births(empty, grid, 0.02f);
    assert(births.accept(Vec3(0), 0.02f));
    assert(!births.accept(Vec3(0.029f, 0, 0), 0.01f));
    assert(births.accept(Vec3(0.031f, 0, 0), 0.01f));
    assert(!births.accept(Vec3(0.985f, 0, 0), 0.02f));

    // Liquid displacement must sum actual individual sphere volumes, independent
    // of trilinear cell support. Carrier order must not resample the sizes.
    FluidParticles grains;
    for (int i = 0; i < 20; ++i) {
        grains.emit(Vec3(0.1f, 0.2f, 0.3f), Vec3(0), 293.15f, 0, 0,
            nullptr, nullptr, m1, MatterConstitutiveModel::Granular);
    }
    double volume = 0.0;
    for (std::size_t i = 0; i < grains.size(); ++i) {
        const double radius = matterGrainRadius(settings, grains, i);
        volume += 4.18879020478639 * radius * radius * radius;
    }
    MatterGrainLiquidField field;
    assert(buildMatterGrainLiquidField(empty, grains, settings.radius_m, nullptr,
        Vec3(-1), 20, 20, 20, 0.1f, field, error, &settings));
    const double displaced = std::accumulate(field.solid.begin(), field.solid.end(), 0.0);
    assert(std::abs(displaced - volume) < volume * 1e-5);
    const float radius_before = matterGrainRadius(settings, grains, 0);
    std::swap(grains.particle_id[0], grains.particle_id[19]);
    assert(matterGrainRadius(settings, grains, 19) == radius_before);
}
