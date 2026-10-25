#include "Fluid/SourceSplatRenderPlan.h"
#include "Fluid/SourceSplatGeometryJson.h"
#include "SimCache.h"
#include "Fluid/SubstanceTag.h"

#include <cassert>
#include <chrono>
#include <filesystem>
#include <limits>

using namespace RayTrophiSim;
using namespace RayTrophiSim::Fluid;

namespace {
SimulationFlowSourceDesc source(uint64_t uid, const char* mode) {
    SimulationFlowSourceDesc result;
    result.timeline_uid = uid;
    result.domain_index = 0;
    result.fluid_substance = "SharedSand";
    result.splat_geometry.mode = mode;
    result.splat_geometry.objects = {"Exemplar"};
    result.splat_geometry.mesh_budget = 1;
    return result;
}

void emit(FluidParticles& particles, uint64_t uid, Vec3 position = Vec3(0.0f)) {
    particles.emit(position, Vec3(0.0f));
    particles.birth_source_uid.back() = uid;
}

bool selected(SourceSplatRenderPlan& plan, const FluidParticles& particles,
    std::size_t index, uint32_t expected, bool expected_over = false) {
    uint32_t exemplar = 999;
    bool over = false;
    const bool result = plan.select(particles, index, Vec3(0.0f), exemplar, over);
    assert(over == expected_over);
    if (result) assert(exemplar == expected);
    return result;
}
} // namespace

int main() {
    SimulationGridDomainDesc domain;
    domain.fluid_grain_mesh_near_enabled = true;
    domain.fluid_grain_mesh_near_shape = "rock";
    auto sphere = source(11, "sphere");
    auto sand = source(22, "mesh_grains");
    auto mpm = source(33, "mesh_grains");
    auto object = source(44, "scene_object");
    auto legacy = source(55, "domain");
    std::vector<SimulationFlowSourceDesc> sources{sphere, sand, mpm, object, legacy};
    SourceSplatRenderPlan plan(domain, sources, 0);
    domain.fluid_particle_visual_children = 8;
    assert(plan.visualChildren(domain) == 8);
    SourceSplatRenderPlan legacy_only(domain, {legacy}, 0);
    assert(legacy_only.visualChildren(domain) == 1);
    std::vector<GrainExemplar> banks(4);
    banks[0].source_uid = 0;
    banks[1].source_uid = 22;
    banks[2].source_uid = 33;
    banks[3].source_uid = 44;
    plan.index(banks);

    FluidParticles particles;
    emit(particles, 11);
    emit(particles, 22);
    emit(particles, 22);
    emit(particles, 33);
    emit(particles, 44, Vec3(1000.0f));
    emit(particles, 55);
    emit(particles, 0);
    emit(particles, 999); // deleted or unresolvable source

    // Same substance, same exemplar names, independent source geometry and budgets.
    assert(!selected(plan, particles, 0, 0));
    assert(selected(plan, particles, 1, 1));
    assert(!selected(plan, particles, 2, 0, true));
    assert(selected(plan, particles, 3, 2));
    assert(selected(plan, particles, 4, 3)); // object is not camera LOD
    assert(selected(plan, particles, 5, 0));
    assert(selected(plan, particles, 6, 0));
    assert(!selected(plan, particles, 7, 0));

    // Reindexing starts a new frame budget; outside the threshold stays sphere.
    plan.index(banks);
    particles.position[1] = Vec3(2.0f);
    assert(!selected(plan, particles, 1, 0));
    assert(selected(plan, particles, 2, 1));

    // Stable source identity survives renaming; deleting the source releases its routing.
    const auto signature = plan.signature({});
    sources[1].name = "RenamedEmitter";
    SourceSplatRenderPlan renamed(domain, sources, 0);
    assert(renamed.signature({}) == signature);
    sources.erase(sources.begin() + 1);
    SourceSplatRenderPlan deleted(domain, sources, 0);
    deleted.index(banks);
    assert(!selected(deleted, particles, 2, 0));

    // Provenance follows storage operations and model partitions, independently of IDs/tags.
    FluidParticles gathered;
    gathered.gatherFrom(particles, {3, 0, 1});
    assert((gathered.birth_source_uid == std::vector<uint64_t>{33, 11, 22}));
    gathered.removeSwap(0);
    assert((gathered.birth_source_uid == std::vector<uint64_t>{22, 11}));
    gathered.compact({1, 0});
    assert((gathered.birth_source_uid == std::vector<uint64_t>{11}));
    FluidParticles destination;
    emit(destination, 0);
    destination.copyRangeFrom(0, gathered);
    assert(destination.birth_source_uid[0] == 11);
    assert(destination.particle_id[0] == gathered.particle_id[0]);

    // Disk playback retains the same source attribution as the live SoA.
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto directory = std::filesystem::temp_directory_path() /
        ("rt_source_geometry_" + std::to_string(nonce));
    std::vector<SimulationGridDomainState> states(1);
    states[0].type = SimulationDomainType::Matter;
    states[0].particles = particles;
    std::vector<MaterialStateFieldSnapshot> msf;
    assert(SimCache::writeSystemFrame(directory.string(), 1, 0, states, msf));
    std::vector<SimulationGridDomainState> replay;
    assert(SimCache::readSystemFrame(directory.string(), 1, 0, replay, msf));
    assert(replay.size() == 1);
    assert(replay[0].particles.birth_source_uid == particles.birth_source_uid);
    assert(replay[0].particles.particle_id == particles.particle_id);
    std::filesystem::remove(SimCache::frameFilePath(directory.string(), 1, 0));
    std::filesystem::remove(directory);

    // Canonical Output is substance-owned: two emitter identities share one library,
    // and even legacy cache particles without source provenance use that same binding.
    SimulationGridDomainDesc output_domain;
    SimulationGridDomainDesc::SubstanceMaterial binding;
    binding.substance = "Sand";
    binding.representation = SubstanceRepresentation::Splat;
    output_domain.fluid_substance_materials.push_back(binding);
    SourceSplatGeometry library;
    library.mode = "scene_object";
    library.objects = {"GeometryA", "GeometryB"};
    output_domain.fluid_substance_splat_geometry["Sand"] = library;
    SourceSplatRenderPlan output_plan(output_domain, sources, 0);
    const uint32_t sand_tag = substanceTag("Sand");
    const uint64_t sand_owner = (uint64_t{1} << 63) | sand_tag;
    std::vector<GrainExemplar> output_banks(2);
    for (auto& bank : output_banks) {
        bank.source_uid = sand_owner;
    }
    output_plan.index(output_banks);
    FluidParticles shared;
    for (const uint64_t uid : {11ull, 33ull, 0ull}) {
        emit(shared, uid, Vec3(1000.0f));
        shared.substance_tag.back() = sand_tag;
    }
    for (std::size_t i = 0; i < shared.size(); ++i) {
        uint32_t exemplar = 99;
        bool over = true;
        assert(output_plan.select(shared, i, Vec3(0.0f), exemplar, over));
        assert(exemplar < 2 && !over);
    }
    // Changing Sand back to SDF suspends that mesh binding for the SDF body.
    output_domain.fluid_substance_materials[0].representation = SubstanceRepresentation::SurfaceSDF;
    SourceSplatRenderPlan sdf_plan(output_domain, sources, 0);
    sdf_plan.index(output_banks);
    uint32_t ignored = 0;
    bool over = false;
    // Cached mesh banks cannot override the canonical SDF selection.
    // The bridge normally never builds this stale bank; selection also checks mode.
    assert(!sdf_plan.select(shared, 0, Vec3(0.0f), ignored, over));
    SourceSplatGeometry geometry;
    from_json(nlohmann::json{{"mode", "mesh_grains"}, {"objects", {"A", "B"}}}, geometry);
    from_json(nlohmann::json{{"mesh_budget", 0}}, geometry);
    assert(geometry.objects.size() == 2 && geometry.mesh_budget == 0);
    const auto roundtrip = nlohmann::json(geometry).get<SourceSplatGeometry>();
    assert(nlohmann::json(roundtrip) == nlohmann::json(geometry));
    for (const auto& invalid : {nlohmann::json{{"mesh_budget", -1}},
            nlohmann::json{{"mesh_budget", true}},
            nlohmann::json{{"mesh_budget", 4294967296ull}},
            nlohmann::json{{"unknown_field", 1}}}) {
        bool rejected = false;
        try {
            from_json(invalid, geometry);
        } catch (const std::exception&) {
            rejected = true;
        }
        assert(rejected);
    }
}
