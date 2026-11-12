#include "Fluid/DomainVisualService.h"
#include "Fluid/DomainVisualField.h"
#include "Fluid/MatterPhaseGrid.h"
#include "Fluid/MatterSubstanceState.h"
#include "ParticleSimulation.h"
#include "InstanceGroup.h"

#include <chrono>
#include <set>

namespace RayTrophiSim::Fluid {
using namespace DomainFields;
namespace {

template<class T>
void appendSide(std::vector<T>& out, const std::vector<T>& input, std::size_t index,
                const T& fallback = T{}) {
    out.push_back(index < input.size() ? input[index] : fallback);
}

void appendRenderParticle(FluidParticles& out, const FluidParticles& input, std::size_t index) {
    appendSide(out.position, input.position, index);
    appendSide(out.velocity, input.velocity, index);
    appendSide(out.particle_id, input.particle_id, index);
    appendSide(out.birth_source_uid, input.birth_source_uid, index);
    appendSide(out.substance_tag, input.substance_tag, index);
    appendSide(out.flags, input.flags, index);
    appendSide(out.constitutive_model, input.constitutive_model, index);
    appendSide(out.rest_mass_kg, input.rest_mass_kg, index);
    appendSide(out.mass_fraction, input.mass_fraction, index, 1.0f);
    appendSide(out.pore_water_mass_kg, input.pore_water_mass_kg, index);
    appendSide(out.pore_capacity_kg, input.pore_capacity_kg, index);
}

bool eligible(const SimulationGridDomainDesc& domain, const FluidViewPlan& views,
              const FluidParticles& particles, std::size_t index) {
    if (views.viewForParticle(particles, index) != FluidView::Splat) {
        return false;
    }
    const uint32_t flags = index < particles.flags.size() ? particles.flags[index] : 0;
    const auto label = particleLabel(flags);
    if ((flags & kParticleFlagFrozen) != 0 ||
        (label != ParticleLabel::Body && label != ParticleLabel::Unknown)) {
        return false;
    }
    const uint32_t tag = index < particles.substance_tag.size() ? particles.substance_tag[index] : 0;
    const auto model = index < particles.constitutive_model.size()
        ? static_cast<MatterConstitutiveModel>(particles.constitutive_model[index])
        : MatterConstitutiveModel::Fluid;
    const auto owner = substanceTransportOwner(tag, model, domain.fluid_params);
    return owner == MatterTransportOwner::Fluid || owner == MatterTransportOwner::Mpm;
}

uint64_t streamKey(const FluidParticles& particles, std::size_t index) {
    const auto source = index < particles.birth_source_uid.size()
        ? particles.birth_source_uid[index] : 0;
    const auto tag = index < particles.substance_tag.size() ? particles.substance_tag[index] : 0;
    const auto model = index < particles.constitutive_model.size()
        ? particles.constitutive_model[index] : static_cast<uint8_t>(MatterConstitutiveModel::Fluid);
    const auto flags = index < particles.flags.size() ? particles.flags[index] : 0;
    auto key = visualHash(source);
    key = visualHash(key ^ tag);
    key = visualHash(key ^ model);
    return visualHash(key ^ (flags & kParticleLabelMask));
}

void buildRenderView(const SimulationGridDomainDesc& domain, const SimulationGridDomainState& state,
                     DomainVisualState& publication, bool include_samples = true) {
    auto& out = publication.render_particles;
    out = {};
    publication.diameters.clear();
    publication.rotations.clear();
    publication.represented_volume = 0.0;
    const auto views = resolveFluidViews(domain, distinctViewKeys(state.particles));
    std::set<uint64_t> identities;
    for (std::size_t i = 0; i < state.particles.size(); ++i) {
        if (!eligible(domain, views, state.particles, i)) {
            appendRenderParticle(out, state.particles, i);
            if (!out.particle_id.empty() && out.particle_id.back() != 0) {
                identities.insert(out.particle_id.back());
            }
            publication.diameters.push_back(-1.0f);
            publication.rotations.emplace_back(0.0f);
        }
    }
    if (!include_samples) {
        publication.render_ready = true;
        return;
    }
    for (const auto& entry : publication.streams) {
        const auto& stream = entry.second;
        for (const auto& sample : stream.runtime.state().samples) {
            publication.represented_volume += sample.represented_volume;
            const auto index = out.position.size();
            out.position.emplace_back(static_cast<float>(sample.position[0]),
                static_cast<float>(sample.position[1]), static_cast<float>(sample.position[2]));
            out.velocity.emplace_back(0.0f);
            const auto identity = visualHash(entry.first ^ visualHash(sample.uid)) | (1ull << 63);
            if (!identities.insert(identity).second) {
                throw std::runtime_error("visual sample render identity collision");
            }
            out.particle_id.push_back(identity);
            out.birth_source_uid.push_back(stream.birth_source);
            out.substance_tag.push_back(stream.tag);
            out.flags.push_back(stream.flags);
            out.constitutive_model.push_back(stream.model);
            out.rest_mass_kg.push_back(0.0f);
            out.mass_fraction.push_back(1.0f);
            out.pore_water_mass_kg.push_back(0.0f);
            out.pore_capacity_kg.push_back(0.0f);
            const double diameter = std::cbrt(6.0 * sample.represented_volume /
                3.14159265358979323846);
            if (!std::isfinite(diameter) || diameter <= 0.0 ||
                diameter > std::numeric_limits<float>::max()) {
                throw std::runtime_error("visual sample diameter is not representable");
            }
            publication.diameters.push_back(static_cast<float>(diameter));
            const auto& q = sample.orientation;
            const double x = q[0], y = q[1], z = q[2], w = q[3];
            auto rotation = Matrix4x4::identity();
            rotation.m[0][0] = static_cast<float>(1 - 2 * (y * y + z * z));
            rotation.m[0][1] = static_cast<float>(2 * (x * y - z * w));
            rotation.m[0][2] = static_cast<float>(2 * (x * z + y * w));
            rotation.m[1][0] = static_cast<float>(2 * (x * y + z * w));
            rotation.m[1][1] = static_cast<float>(1 - 2 * (x * x + z * z));
            rotation.m[1][2] = static_cast<float>(2 * (y * z - x * w));
            rotation.m[2][0] = static_cast<float>(2 * (x * z - y * w));
            rotation.m[2][1] = static_cast<float>(2 * (y * z + x * w));
            rotation.m[2][2] = static_cast<float>(1 - 2 * (x * x + y * y));
            publication.rotations.push_back(InstanceTransform::fromMatrix(rotation).rotation);
            if (index + 1 != publication.diameters.size()) {
                throw std::logic_error("visual render adapter lost index alignment");
            }
        }
    }
    publication.render_ready = true;
}

Validity traceSegment(const SimulationGridDomainDesc& domain, const FluidSim::FluidGrid& grid,
                      const std::array<double, 3>& from, std::array<double, 3>& to) {
    const double h = grid.voxel_size;
    const int dimensions[] = {grid.nx, grid.ny, grid.nz};
    for (int axis = 0; axis < 3; ++axis) {
        const double low = grid.origin[axis];
        const double high = low + dimensions[axis] * h;
        if (!std::isfinite(to[axis])) {
            return Validity::Unavailable;
        }
        if (to[axis] < low || to[axis] >= high) {
            if (domain.boundary_mode == SimulationGridDomainBoundaryMode::Open) {
                return Validity::OutsideDomain;
            }
            if (domain.boundary_mode == SimulationGridDomainBoundaryMode::Periodic) {
                return Validity::Unavailable; // Authored validation rejects this route.
            }
            to[axis] = std::clamp(to[axis], low + h * 1e-6, high - h * 1e-6);
        }
    }
    double distance = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
        distance = std::max(distance, std::abs(to[axis] - from[axis]));
    }
    const int steps = std::max(1, static_cast<int>(std::ceil(distance / (h * 0.25))));
    for (int step = 1; step <= steps; ++step) {
        std::array<int, 3> cell{};
        for (int axis = 0; axis < 3; ++axis) {
            const double p = from[axis] + (to[axis] - from[axis]) * step / steps;
            cell[axis] = std::clamp(static_cast<int>(std::floor((p - grid.origin[axis]) / h)),
                                    0, dimensions[axis] - 1);
        }
        if (grid.solid[grid.cellIndex(cell[0], cell[1], cell[2])] != 0) {
            to = from;
            return Validity::ValidData;
        }
    }
    return Validity::ValidData;
}

void publish(const SimulationGridDomainDesc& domain, const SimulationGridDomainState& state) {
    auto key = visualHash(domainVisualSettingsKey(domain.fluid_visual_samples) ^
        phaseOutputSignature(domain.fluid_phase_output) ^
        static_cast<uint64_t>(domain.fluid_render_mode));
    for (const auto route : domain.fluid_label_routes) {
        key = visualHash(key ^ static_cast<uint64_t>(route));
    }
    for (const auto& binding : domain.fluid_substance_materials) {
        for (const unsigned char c : binding.substance) {
            key = visualHash(key ^ c);
        }
        key = visualHash(key ^ static_cast<uint64_t>(binding.representation));
    }
    if (state.visual_samples && state.visual_samples->settings_key == key &&
        state.visual_samples->state_version == state.version) {
        if (state.visual_samples->current && !state.visual_samples->render_ready) {
            auto restored = std::make_shared<DomainVisualState>(*state.visual_samples);
            buildRenderView(domain, state, *restored);
            state.visual_samples = std::move(restored);
        }
        return;
    }
    const auto begin = std::chrono::steady_clock::now();
    auto next = std::make_shared<DomainVisualState>();
    next->settings_key = key;
    next->state_version = state.version;
    next->physical_time = state.render_physical_time;
    try {
        const auto& settings = domain.fluid_visual_samples;
        if (state.visual_cache_unavailable) {
            throw std::runtime_error("cache has no visual checkpoints; reset and rebake");
        }
        validateVisualSettings(settings.sampling);
        if (domain.boundary_mode == SimulationGridDomainBoundaryMode::Periodic) {
            throw std::runtime_error("CPU visual reconstruction does not support periodic boundaries");
        }
        const auto& grid = liquidGrid(state);
        if (grid.solid.size() != grid.getCellCount()) {
            throw std::runtime_error("visual collision mask is unavailable");
        }
        const uint64_t bytes = settings.working_budget_mb * 1024ull * 1024ull;
        const uint64_t staging = settings.sampling.max_samples * 768ull +
            static_cast<uint64_t>(state.particles.size()) * 160ull;
        if (staging >= bytes) {
            throw std::length_error("visual sample/checkpoint staging exceeds working budget");
        }
        if (domain.fluid_params.pore_exchange.wet_appearance_enabled) {
            throw std::runtime_error("field samples do not support spatial wet appearance");
        }
        const auto views = resolveFluidViews(domain, distinctViewKeys(state.particles));
        std::map<uint64_t, std::vector<std::size_t>> indices;
        for (std::size_t i = 0; i < state.particles.size(); ++i) {
            if (eligible(domain, views, state.particles, i)) {
                indices[streamKey(state.particles, i)].push_back(i);
            }
        }
        if (indices.size() > 128 || indices.size() > settings.sampling.max_samples) {
            throw std::length_error("visual stream count exceeds supported budget");
        }
        uint64_t remaining = settings.sampling.max_samples;
        std::size_t remaining_carriers = 0;
        for (const auto& item : indices) {
            remaining_carriers += item.second.size();
        }
        std::size_t streams_left = indices.size();
        for (const auto& item : indices) {
            auto stream_settings = settings.sampling;
            const auto share = static_cast<uint64_t>(static_cast<long double>(remaining) *
                item.second.size() / remaining_carriers);
            stream_settings.max_samples = std::clamp<uint64_t>(share, 1,
                remaining - static_cast<uint64_t>(streams_left - 1));
            remaining -= stream_settings.max_samples;
            remaining_carriers -= item.second.size();
            --streams_left;
            auto& stream = next->streams[item.first];
            const auto first = item.second.front();
            const auto& particles = state.particles;
            stream.birth_source = first < particles.birth_source_uid.size()
                ? particles.birth_source_uid[first] : 0;
            stream.tag = first < particles.substance_tag.size() ? particles.substance_tag[first] : 0;
            stream.flags = first < particles.flags.size() ? particles.flags[first] : 0;
            stream.model = first < particles.constitutive_model.size()
                ? particles.constitutive_model[first] : uint8_t{1};
            DomainVisualField field(domain, state, item.second, item.first, bytes - staging);
            stream.runtime.reset(stream_settings);
            if (state.visual_samples) {
                const auto old = state.visual_samples->streams.find(item.first);
                if (old != state.visual_samples->streams.end()) {
                    auto checkpoint = old->second.runtime.state();
                    if (checkpoint.settings.seed == stream_settings.seed) {
                        // A known-empty region no longer represents this stream.
                        // Retire it only when the current publication also has no
                        // velocity support there; never invent a zero velocity.
                        checkpoint.samples.erase(std::remove_if(checkpoint.samples.begin(),
                            checkpoint.samples.end(), [&](const auto& sample) {
                                const auto material = field.support(sample.position);
                                return usable(material.validity) && !material.occupied &&
                                    !usable(sampleMacVelocity(field, sample.position).validity);
                            }), checkpoint.samples.end());
                        stream.runtime.restore(checkpoint);
                        stream.runtime.configure(stream_settings);
                    }
                }
            }
            const auto support = [&](uint64_t, const std::array<double, 3>& p) {
                auto result = field.support(p);
                auto point = p;
                if (result.occupied && traceSegment(domain, grid, p, point) != Validity::ValidData) {
                    result.occupied = false;
                }
                if (result.occupied) {
                    const int x = static_cast<int>((p[0] - grid.origin.x) / grid.voxel_size);
                    const int y = static_cast<int>((p[1] - grid.origin.y) / grid.voxel_size);
                    const int z = static_cast<int>((p[2] - grid.origin.z) / grid.voxel_size);
                    result.occupied = grid.solid[grid.cellIndex(x, y, z)] == 0;
                }
                return result;
            };
            const auto trace = [&](uint64_t, const std::array<double, 3>& from,
                                   std::array<double, 3>& to, double, double) {
                return traceSegment(domain, grid, from, to);
            };
            const auto stats = stream.runtime.update(field, field.regions(item.first), trace, support,
                state.visual_samples && state.visual_samples->settings_key != key);
            next->unrepresented_volume += stats.reseed.unrepresented_volume;
            next->sample_count += stream.runtime.state().samples.size();
        }
        buildRenderView(domain, state, *next);
        next->current = true;
        next->status = "active";
    } catch (const std::exception& error) {
        next->streams.clear();
        if (state.visual_samples && state.visual_samples->settings_key == key) {
            next->streams = state.visual_samples->streams;
        }
        next->render_particles = {};
        next->diameters.clear();
        next->rotations.clear();
        next->current = false;
        next->sample_count = 0;
        next->unrepresented_volume = 0.0;
        next->status = error.what();
        buildRenderView(domain, state, *next, false);
    }
    next->update_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin).count();
    state.visual_samples = std::move(next);
}

} // namespace

void advanceDomainVisualSamples(const SimulationGridDomainDesc& domain,
    SimulationGridDomainState& state, double dt) {
    if (std::isfinite(dt) && dt > 0.0) {
        state.render_physical_time += dt;
    }
    if (domain.fluid_visual_samples.enabled && state.valid) {
        publish(domain, state);
    } else {
        state.visual_samples.reset();
    }
}

const FluidParticles& domainSplatParticles(const SimulationGridDomainDesc& domain,
    const SimulationGridDomainState& state) {
    if (!domain.fluid_visual_samples.enabled) {
        return state.particles;
    }
    publish(domain, state);
    return state.visual_samples->render_particles;
}

float domainSplatDiameter(const SimulationGridDomainDesc& domain,
    const SimulationGridDomainState& state, std::size_t index, float fallback) {
    if (domain.fluid_visual_samples.enabled && state.visual_samples &&
        index < state.visual_samples->diameters.size() && state.visual_samples->diameters[index] > 0) {
        return state.visual_samples->diameters[index] * domain.fluid_particle_size_multiplier;
    }
    return fallback;
}

nlohmann::json domainVisualStats(const SimulationGridDomainDesc& domain,
    const SimulationGridDomainState* state) {
    const auto* p = domain.fluid_visual_samples.enabled && state && state->visual_samples
        ? state->visual_samples.get() : nullptr;
    return {{"enabled", domain.fluid_visual_samples.enabled},
        {"route", "cpu_soa_reconstruction"}, {"current", p && p->current},
        {"status", p ? p->status : "not published"},
        {"physical_carriers", state ? state->particles.size() : 0},
        {"samples", p ? p->sample_count : 0},
        {"render_particles", p ? p->render_particles.size() : 0},
        {"unrepresented_volume", p ? p->unrepresented_volume : 0.0},
        {"represented_volume", p ? p->represented_volume : 0.0},
        {"update_ms", p ? p->update_ms : 0.0},
        {"physical_time", state ? state->render_physical_time : 0.0},
        {"state_version", state ? state->version : 0},
        {"affects_physics", false}, {"orientation", "fixed_uid"}};
}

Vec3 domainSplatRotation(const SimulationGridDomainDesc& domain,
    const SimulationGridDomainState& state, std::size_t index, const Vec3& fallback) {
    if (domain.fluid_visual_samples.enabled && state.visual_samples &&
        index < state.visual_samples->rotations.size() &&
        index < state.visual_samples->diameters.size() && state.visual_samples->diameters[index] > 0) {
        return state.visual_samples->rotations[index];
    }
    return fallback;
}

std::size_t domainVisualAllocationBytes(const SimulationGridDomainState& state) {
    if (!state.visual_samples) {
        return 0;
    }
    const auto& value = *state.visual_samples;
    std::size_t bytes = sizeof(value) + value.diameters.capacity() * sizeof(float);
    bytes += value.rotations.capacity() * sizeof(Vec3);
    for (const auto& entry : value.streams) {
        bytes += sizeof(DomainVisualStream) + 64 +
            entry.second.runtime.state().samples.capacity() * sizeof(VisualSample);
    }
    const auto& p = value.render_particles;
    bytes += (p.position.capacity() + p.velocity.capacity()) * sizeof(Vec3);
    bytes += (p.particle_id.capacity() + p.birth_source_uid.capacity()) * sizeof(uint64_t);
    bytes += (p.flags.capacity() + p.substance_tag.capacity()) * sizeof(uint32_t);
    bytes += p.constitutive_model.capacity();
    bytes += (p.rest_mass_kg.capacity() + p.mass_fraction.capacity() +
        p.pore_water_mass_kg.capacity() + p.pore_capacity_kg.capacity()) * sizeof(float);
    return bytes;
}

} // namespace RayTrophiSim::Fluid
