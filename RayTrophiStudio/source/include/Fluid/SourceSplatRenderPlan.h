#pragma once

#include "ParticleSimulation.h"
#include "Fluid/GrainMeshNear.h"
#include <unordered_map>

namespace RayTrophiSim::Fluid {

uint32_t sourceSplatVisualChildren(const ParticleSimulationSystem& system,
    const SimulationGridDomainDesc& domain);
uint64_t sourceSplatCameraKey(const Vec3& camera);
bool hasSourceSplatMeshes(const ParticleSimulationSystem& system, std::size_t domain);

// A frame-local routing table. Cached mesh banks carry their owner UID so two
// sources using the same substance (or object) remain independent.
class SourceSplatRenderPlan {
public:
    SourceSplatRenderPlan(const SimulationGridDomainDesc& domain,
        const std::vector<SimulationFlowSourceDesc>& sources, int domain_index);

    SimulationGridDomainDesc renderDomain(const SimulationGridDomainDesc& domain) const;
    uint32_t visualChildren(const SimulationGridDomainDesc& domain) const;
    uint64_t signature(const std::vector<std::shared_ptr<Hittable>>& objects) const;
    bool build(const std::vector<std::shared_ptr<Hittable>>& objects,
        std::vector<GrainExemplar>& out, std::string& error) const;
    void index(const std::vector<GrainExemplar>& exemplars);
    bool select(const FluidParticles& particles, std::size_t particle,
        const Vec3& camera, uint32_t& exemplar, bool& over_budget);

private:
    struct Bank {
        SourceSplatGeometry geometry;
        std::string shape = "scene";
        std::vector<uint32_t> indices;
        uint32_t used = 0;
    };
    std::unordered_map<uint64_t, Bank> banks_;
    std::unordered_map<uint64_t, uint64_t> owners_;
    std::unordered_map<uint32_t, uint64_t> substance_owners_;
    bool has_meshes_ = false;
    bool source_overrides_ = false;
};

} // namespace RayTrophiSim::Fluid
