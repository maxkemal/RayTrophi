// User test target: link DemBondMechanics/Network/Cache/Seed, MatterGrainParams,
// and the existing particle/substance/Vec3 core. No Studio launch is required.
#include "Fluid/DemBondCache.h"
#include "Fluid/DemBondSeed.h"
#include "Fluid/FluidParticles.h"
#include "Fluid/MatterGrain.h"

#include <cassert>
#include <sstream>

namespace F = RayTrophiSim::Fluid;
namespace B = F::DemBond;

int main() {
    F::FluidParticles p;
    for (float x : {0.0f, 0.04f, 0.08f}) {
        p.emit(Vec3(x, 0.0f, 0.0f), Vec3(0.0f), 263.0f, 0.0f, 0,
            nullptr, nullptr, 1.0f, F::MatterConstitutiveModel::Granular);
    }
    F::MatterGrainParams params;
    params.radius_m = 0.025f;
    B::Material material{1.0e5, 0.25, 1.0, 1000.0, 1000.0};
    std::string error;
    uint64_t body = 99;
    const auto empty = B::encode(p.dem_bonds);
    assert(!B::bindBody(p, params, p.particle_id, material, 1.25, 0.5, 1, body, error));
    assert(B::encode(p.dem_bonds) == empty && body == 99);
    auto duplicate = p.particle_id;
    duplicate.push_back(duplicate.front());
    assert(!B::bindBody(p, params, duplicate, material, 1.25, 0.5, 0, body, error));
    assert(B::bindBody(p, params, p.particle_id, material, 1.25, 0.5, 0, body, error));
    assert(body == 1 && p.dem_bonds.edges.size() == 2 && p.dem_bonds.nodes.size() == 3);
    const auto initial = B::encode(p.dem_bonds);
    assert(!B::bindBody(p, params, p.particle_id, material, 1.25, 0.5, 0, body, error));
    assert(B::encode(p.dem_bonds) == initial);

    // Sorting/remapping IDs does not recreate or erase topology.
    auto reordered = p.particle_id;
    std::reverse(reordered.begin(), reordered.end());
    B::retain(p.dem_bonds, reordered);
    assert(B::encode(p.dem_bonds) == initial);

    // Fracture + energy survive native binary cache, not merely JSON display.
    auto& edge = p.dem_bonds.edges.front();
    B::Endpoint a{edge.state.first_id, {0.0, 0.0, 0.0}};
    B::Endpoint b{edge.state.second_id, {edge.state.rest_length_m * 1.02, 0.0, 0.0}};
    B::Response response;
    assert(B::evaluate(a, b, edge.material, edge.state, response, error));
    assert(edge.state.failure == B::Failure::Tension && response.broken_now);
    std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
    assert(B::writeCache(stream, p.dem_bonds));
    B::Network restored;
    assert(B::readCache(stream, p.particle_id, restored));
    assert(B::encode(restored) == B::encode(p.dem_bonds));
    const auto saved = B::encode(restored);
    std::stringstream truncated(stream.str().substr(0, stream.str().size() / 2));
    assert(!B::readCache(truncated, p.particle_id, restored));
    assert(B::encode(restored) == saved);
    auto corrupt = saved;
    corrupt["edges"][0][0] = uint64_t{999999};
    assert(!B::decode(corrupt, restored, error));
    assert(B::encode(restored) == saved);
    corrupt = saved;
    corrupt["nodes"][0][0] = 1.5;
    assert(!B::decode(corrupt, restored, error));
    assert(B::encode(restored) == saved);

    // Removal accounts stored/released energy once and never heals survivors.
    const double energy = edge.state.fracture_dissipation_j;
    assert(energy > 0.0);
    B::retain(restored, {p.particle_id[1], p.particle_id[2]});
    assert(restored.edges.size() == 1 && restored.retired_energy_j == energy);
    B::retain(restored, {p.particle_id[1], p.particle_id[2]});
    assert(restored.retired_energy_j == energy);
    assert(!B::removeBody(restored, 0, error));
    assert(B::removeBody(restored, 1, error));
    assert(restored.nodes.empty() && restored.edges.empty());

    // Seed settings are transactional and each layer gets its own body ID.
    B::SeedSettings seed;
    const auto defaults = B::seedJson(seed);
    assert(!B::parseSeed({{"reach_ratio", 0.1}}, seed, error));
    assert(!B::parseSeed({{"enabled", "yes"}}, seed, error));
    assert(!B::parseSeed({{"young_pa", true}}, seed, error));
    assert(!B::parseSeed({{"young_pa", 1.0e100}}, seed, error));
    assert(B::seedJson(seed) == defaults);
    assert(B::parseSeed({{"enabled", true}}, seed, error));
    F::FluidParticles seeds;
    for (float x : {0.0f, 0.04f}) {
        seeds.emit(Vec3(x, 0.0f, 0.0f), Vec3(0.0f), 263.0f, 0.0f, 0,
            nullptr, nullptr, 1.0f, F::MatterConstitutiveModel::Granular);
    }
    assert(B::bindSeed(seeds, 0, params, seed, 0, error));
    for (float x : {0.08f, 0.12f}) {
        seeds.emit(Vec3(x, 0.0f, 0.0f), Vec3(0.0f), 263.0f, 0.0f, 0,
            nullptr, nullptr, 1.0f, F::MatterConstitutiveModel::Granular);
    }
    assert(B::bindSeed(seeds, 2, params, seed, 0, error));
    assert(seeds.dem_bonds.edges.size() == 2); // no cross-layer bond, despite proximity
    assert(B::summary(seeds.dem_bonds).at("bodies") == 2);
    seeds.clear();
    assert(seeds.dem_bonds.nodes.empty() && seeds.dem_bonds.next_body == 1);
}
