#include "Fluid/SourceSplatGeometryJson.h"
#include "Fluid/SourceSplatRenderPlan.h"
#include "Fluid/SubstanceTag.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <functional>
#include <iterator>
#include <utility>

namespace RayTrophiSim::Fluid {
namespace {
bool meshMode(const SourceSplatGeometry& value) {
    return value.mode == "scene_object" || value.mode == "mesh_grains";
}
uint64_t mix(uint64_t hash, uint64_t value) {
    return (hash ^ value) * 1099511628211ull;
}
} // namespace

uint64_t sourceSplatCameraKey(const Vec3& camera) {
    const std::hash<float> hash;
    return mix(mix(hash(camera.x), hash(camera.y)), hash(camera.z));
}

uint32_t sourceSplatVisualChildren(const ParticleSimulationSystem& system,
    const SimulationGridDomainDesc& domain) {
    const auto& domains = system.gridDomains();
    const auto found = std::find_if(domains.begin(), domains.end(),
        [&](const auto& candidate) { return candidate.name == domain.name; });
    const int index = found == domains.end() ? -1 :
        static_cast<int>(std::distance(domains.begin(), found));
    return SourceSplatRenderPlan(domain, system.flowSources(), index).visualChildren(domain);
}

bool hasSourceSplatMeshes(const ParticleSimulationSystem& system, std::size_t domain) {
    if (domain < system.gridDomains().size()) {
        for (const auto& entry : system.gridDomains()[domain].fluid_substance_splat_geometry) {
            if (meshMode(entry.second)) {
                return true;
            }
        }
    }
    for (const auto& source : system.flowSources()) {
        if (source.domain_index == static_cast<int>(domain) && meshMode(source.splat_geometry)) {
            return true;
        }
    }
    return false;
}

SourceSplatRenderPlan::SourceSplatRenderPlan(const SimulationGridDomainDesc& domain,
    const std::vector<SimulationFlowSourceDesc>& sources, int domain_index) {
    Bank fallback;
    fallback.geometry.mode = "sphere";
    if (domain.fluid_particle_geometry_mode == 1) {
        fallback.geometry.mode = "scene_object";
        fallback.geometry.objects = {domain.fluid_particle_geometry_source};
        fallback.geometry.source_materials = true;
        fallback.geometry.mesh_budget = std::numeric_limits<uint32_t>::max();
    } else if (domain.fluid_grain_mesh_near_enabled) {
        fallback.geometry.mode = "mesh_grains";
        fallback.geometry.objects = domain.fluid_grain_mesh_near_sources;
        fallback.geometry.source_materials = domain.fluid_grain_mesh_near_source_materials;
        fallback.geometry.near_distance = domain.fluid_grain_mesh_near_distance;
        fallback.geometry.mesh_budget = domain.fluid_grain_mesh_near_budget;
        fallback.shape = domain.fluid_grain_mesh_near_shape;
    }
    banks_.emplace(0, std::move(fallback));
    for (const auto& source : sources) {
        if (source.domain_index != domain_index || source.timeline_uid == 0) {
            continue;
        }
        if (source.splat_geometry.mode == "domain") {
            owners_[source.timeline_uid] = 0;
        } else {
            source_overrides_ = true;
            owners_[source.timeline_uid] = source.timeline_uid;
            banks_[source.timeline_uid].geometry = source.splat_geometry;
        }
    }
    std::map<std::string, SourceSplatGeometry> output = domain.fluid_substance_splat_geometry;
    for (const auto& binding : domain.fluid_substance_materials) {
        output.try_emplace(binding.substance, SourceSplatGeometry{});
    }
    const auto views = resolveFluidViews(domain, {});
    for (const auto& entry : output) {
        const uint32_t tag = substanceTag(entry.first);
        const uint64_t owner = (uint64_t{1} << 63) | tag;
        substance_owners_[tag] = owner;
        if (entry.first == domain.fluid_params.default_substance) {
            substance_owners_[kSubstanceUntagged] = owner;
        }
        banks_[owner].geometry = views.mayDrawSplat(tag)
            ? entry.second : SourceSplatGeometry{};
        source_overrides_ = true;
    }
    for (const auto& entry : banks_) {
        has_meshes_ |= meshMode(entry.second.geometry);
    }
}

SimulationGridDomainDesc SourceSplatRenderPlan::renderDomain(
    const SimulationGridDomainDesc& domain) const {
    if (!source_overrides_ && domain.fluid_particle_geometry_mode == 2) {
        return domain;
    }
    auto result = domain;
    // Geometry belongs to the routing table. The primary pool supplies spheres;
    // the mesh pool supplies only the explicitly selected carriers.
    result.fluid_particle_geometry_mode = 0;
    result.fluid_particle_geometry_source.clear();
    result.fluid_grain_mesh_near_enabled = has_meshes_;
    result.fluid_grain_mesh_near_shape = "scene";
    result.fluid_grain_mesh_near_sources.clear();
    double budget = 0;
    result.fluid_grain_mesh_near_distance = 0.0f;
    for (const auto& entry : banks_) {
        const auto& geometry = entry.second.geometry;
        if (!meshMode(geometry)) {
            continue;
        }
        result.fluid_grain_mesh_near_sources.insert(
            result.fluid_grain_mesh_near_sources.end(),
            geometry.objects.begin(), geometry.objects.end());
        // Legacy procedural banks have no object names but still form a library.
        if (entry.second.shape != "scene") {
            result.fluid_grain_mesh_near_sources.push_back("[Legacy Grain Library]");
        }
        budget += geometry.mesh_budget;
        if (geometry.mode == "mesh_grains") {
            result.fluid_grain_mesh_near_distance = std::max(
                result.fluid_grain_mesh_near_distance, geometry.near_distance);
        }
    }
    result.fluid_grain_mesh_near_budget = static_cast<uint32_t>(std::min(
        budget, static_cast<double>(std::numeric_limits<uint32_t>::max())));
    // Source-specific distance is tested in select(). The bridge's camera key
    // tracks exact camera changes through signature integration.
    return result;
}

uint32_t SourceSplatRenderPlan::visualChildren(const SimulationGridDomainDesc& domain) const {
    auto sphere_domain = domain;
    if (source_overrides_) {
        sphere_domain.fluid_grain_mesh_near_enabled = false;
    }
    return primarySplatVisualChildren(sphere_domain);
}

uint64_t SourceSplatRenderPlan::signature(
    const std::vector<std::shared_ptr<Hittable>>& objects) const {
    std::vector<uint64_t> ids;
    for (const auto& entry : banks_) {
        ids.push_back(entry.first);
    }
    std::sort(ids.begin(), ids.end());
    uint64_t hash = 1469598103934665603ull;
    std::vector<uint64_t> owners;
    for (const auto& entry : owners_) {
        owners.push_back(entry.first);
    }
    std::sort(owners.begin(), owners.end());
    for (const auto uid : owners) {
        hash = mix(mix(hash, uid), owners_.at(uid));
    }
    for (const auto id : ids) {
        const auto& bank = banks_.at(id);
        hash = mix(hash, id);
        for (const unsigned char c : nlohmann::json(bank.geometry).dump()) {
            hash = mix(hash, c);
        }
        for (const unsigned char c : bank.shape) {
            hash = mix(hash, c);
        }
        if (meshMode(bank.geometry) && bank.shape == "scene") {
            hash = mix(hash, grainMeshSceneSignature(objects, bank.geometry.objects));
        }
    }
    return hash;
}

bool SourceSplatRenderPlan::build(const std::vector<std::shared_ptr<Hittable>>& objects,
    std::vector<GrainExemplar>& out, std::string& error) const {
    out.clear();
    std::vector<uint64_t> ids;
    for (const auto& entry : banks_) {
        ids.push_back(entry.first);
    }
    std::sort(ids.begin(), ids.end());
    for (const auto id : ids) {
        const auto& bank = banks_.at(id);
        if (!meshMode(bank.geometry)) {
            continue;
        }
        std::vector<GrainExemplar> library;
        std::string bank_error;
        if (!buildGrainExemplars(objects, bank.geometry.objects, library,
                bank_error, bank.shape)) {
            // A deleted source must not suppress another source's valid bank.
            error += "source " + std::to_string(id) + ": " + bank_error + "; ";
            continue;
        }
        for (auto& exemplar : library) {
            exemplar.source_uid = id;
            exemplar.random_orientation = bank.geometry.mode == "mesh_grains";
            if (!exemplar.random_orientation) {
                exemplar.volume_scale = 1.0f;
            }
            exemplar.source_materials = bank.geometry.source_materials && bank.shape == "scene";
            exemplar.node += " [Source " + std::to_string(id) + "]";
            out.push_back(std::move(exemplar));
        }
    }
    return !out.empty();
}

void SourceSplatRenderPlan::index(const std::vector<GrainExemplar>& exemplars) {
    for (auto& entry : banks_) {
        entry.second.indices.clear();
        entry.second.used = 0;
    }
    for (uint32_t i = 0; i < exemplars.size(); ++i) {
        const auto found = banks_.find(exemplars[i].source_uid);
        if (found != banks_.end()) {
            found->second.indices.push_back(i);
        }
    }
}

bool SourceSplatRenderPlan::select(const FluidParticles& particles, std::size_t particle,
    const Vec3& camera, uint32_t& exemplar, bool& over_budget) {
    over_budget = false;
    const uint64_t uid = particle < particles.birth_source_uid.size()
        ? particles.birth_source_uid[particle] : 0;
    const auto owner = owners_.find(uid);
    // Deleted sources fall back to spheres, never to another emitter's library.
    const uint32_t tag = particle < particles.substance_tag.size()
        ? particles.substance_tag[particle] : 0;
    const auto substance = substance_owners_.find(tag);
    if (substance == substance_owners_.end() && uid != 0 && owner == owners_.end()) {
        return false;
    }
    const uint64_t bank_id = substance != substance_owners_.end() ? substance->second :
        owner == owners_.end() ? 0 : owner->second;
    auto& bank = banks_.at(bank_id);
    if (!meshMode(bank.geometry) || bank.indices.empty()) {
        return false;
    }
    const Vec3& position = particles.position[particle];
    if (!std::isfinite(position.x) || !std::isfinite(position.y) ||
        !std::isfinite(position.z)) {
        return false;
    }
    if (bank.geometry.mode == "mesh_grains") {
        const Vec3 delta = position - camera;
        const double distance2 = static_cast<double>(delta.x) * delta.x +
            static_cast<double>(delta.y) * delta.y + static_cast<double>(delta.z) * delta.z;
        const double radius = bank.geometry.near_distance;
        if (distance2 >= radius * radius) {
            return false;
        }
    }
    if (bank.geometry.mode == "mesh_grains" && bank.used >= bank.geometry.mesh_budget) {
        over_budget = true;
        return false;
    }
    ++bank.used;
    const uint64_t identity = particle < particles.particle_id.size()
        ? particles.particle_id[particle] : particle;
    exemplar = bank.indices[grainExemplarFor(identity, static_cast<uint32_t>(bank.indices.size()))];
    return true;
}

} // namespace RayTrophiSim::Fluid
