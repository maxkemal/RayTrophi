#include "RtApiInternal.h"
#include "ParticleSimulation.h"

namespace rtapi {
Result queueMoltenMassTransfer(const std::string& object_key,
                               const std::string& preferred_domain,
                               float mass_kg, float particles_per_kg,
                               Vec3 velocity, uint64_t& out_sequence) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    RayTrophiSim::MoltenMassTransferRequest request;
    request.object_key = object_key;
    request.preferred_domain = preferred_domain;
    request.requested_mass = mass_kg;
    request.particles_per_kg = particles_per_kg;
    request.velocity = velocity;
    out_sequence = scriptSimulationRuntime().queueMoltenMassTransfer(request);
    if (out_sequence == 0) return Result::fail(
        "object_key, mass_kg and particles_per_kg must be valid and positive");
    // This is a RUNTIME event, not an authoring mutation. A timeline render
    // resync resets grid-domain state on the first UI tick after the script
    // returns: the manual test sees particles and passes, then the workspace
    // hands control back and both the particles and Fluid Step telemetry become
    // zero. Cached future frames are stale, but the live state must survive.
    g_ctx->scene.clearSimFrameCache();
    g_ctx->scene.preserveScriptSimulationPreview();
    return Result::success();
}

Result getMoltenMassTransferInfo(MoltenMassTransferInfo& out) {
    if (!g_ctx) return notBound();
    out.live_tagged_particles = 0;
    out.mean_remaining_mass_fraction = 0.0f;
    const auto& s = scriptSimulationRuntime().moltenMassTransferStats();
    out.queued = s.queued; out.completed = s.completed;
    out.deferred_no_domain = s.deferred_no_domain;
    out.deferred_no_capacity = s.deferred_no_capacity;
    out.dropped = s.dropped;
    out.discarded_on_reset = s.discarded_on_reset;
    out.requested_mass = s.requested_mass;
    out.transferred_mass = s.transferred_mass;
    out.spawned_particles = s.spawned_particles;
    out.last_object = s.last_object; out.last_domain = s.last_domain;
    out.last_substance = s.last_substance;
    out.last_temperature_kelvin = s.last_temperature_kelvin;
    out.last_combustible_fraction = s.last_combustible_fraction;
    uint32_t tag = 2166136261u;
    for (unsigned char c : s.last_substance) tag = (tag ^ c) * 16777619u;
    double remaining = 0.0;
    const auto& domains = scriptSimulationRuntime().gridDomains();
    const auto& states = scriptSimulationRuntime().gridDomainStates();
    for (std::size_t di = 0; di < domains.size() && di < states.size(); ++di) {
        if (domains[di].name != s.last_domain) continue;
        const auto& particles = states[di].particles;
        for (std::size_t p = 0; p < particles.size(); ++p) {
            if (p >= particles.substance_tag.size() || particles.substance_tag[p] != tag) continue;
            ++out.live_tagged_particles;
            remaining += p < particles.mass_fraction.size()
                ? particles.mass_fraction[p] : 1.0f;
        }
        break;
    }
    if (out.live_tagged_particles > 0) {
        out.mean_remaining_mass_fraction = static_cast<float>(
            remaining / static_cast<double>(out.live_tagged_particles));
    }
    return Result::success();
}
} // namespace rtapi
