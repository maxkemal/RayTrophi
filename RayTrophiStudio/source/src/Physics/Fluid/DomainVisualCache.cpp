#include "Fluid/DomainVisualService.h"
#include "ParticleSimulation.h"

#include <istream>
#include <ostream>
#include <type_traits>

namespace RayTrophiSim::Fluid {
namespace {
template<class T>
bool pod(std::ostream& out, const T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    return static_cast<bool>(out.write(reinterpret_cast<const char*>(&value), sizeof(T)));
}
template<class T>
bool pod(std::istream& in, T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    return static_cast<bool>(in.read(reinterpret_cast<char*>(&value), sizeof(T)));
}
bool pod(std::istream& in, bool& value) {
    uint8_t byte = 0;
    if (!pod(in, byte) || byte > 1) {
        return false;
    }
    value = byte != 0;
    return true;
}
bool pod(std::ostream& out, const bool& value) {
    return pod(out, static_cast<uint8_t>(value ? 1 : 0));
}
bool text(std::ostream& out, const std::string& value) {
    if (value.size() > 4096) {
        return false;
    }
    return pod(out, static_cast<uint32_t>(value.size())) &&
        static_cast<bool>(out.write(value.data(), static_cast<std::streamsize>(value.size())));
}
bool text(std::istream& in, std::string& value) {
    uint32_t length = 0;
    if (!pod(in, length) || length > 4096) {
        return false;
    }
    value.resize(length);
    return static_cast<bool>(in.read(value.data(), length));
}
template<class IO, class CP>
bool checkpointHeader(IO& io, CP& cp) {
    auto& id = cp.identity;
    auto& settings = cp.settings;
    return text(io, id.domain_uid) && pod(io, id.lane) && pod(io, id.state_generation) &&
        pod(io, id.topology_generation) && pod(io, id.transform_revision) && pod(io, id.step) &&
        pod(io, id.substep) && pod(io, id.physical_time) && pod(io, cp.geometry.origin) &&
        pod(io, cp.geometry.cells) && pod(io, cp.geometry.voxel_size) &&
        pod(io, settings.target_volume) && pod(io, settings.max_samples) &&
        pod(io, settings.seed) && pod(io, settings.cfl) && pod(io, settings.reseed_hysteresis) &&
        pod(io, settings.max_substeps) && pod(io, cp.next_uid) && pod(io, cp.initialized);
}
template<class IO, class Sample>
bool sampleFields(IO& io, Sample& sample) {
    return pod(io, sample.uid) && pod(io, sample.stream_uid) && pod(io, sample.position) &&
        pod(io, sample.previous_position) && pod(io, sample.represented_volume) &&
        pod(io, sample.birth_time) && pod(io, sample.orientation);
}
} // namespace

bool writeDomainVisualCache(std::ostream& output, const SimulationGridDomainState& state) {
    const uint8_t present = state.visual_samples ? 1 : 0;
    if (!pod(output, state.render_physical_time) || !pod(output, present)) {
        return false;
    }
    if (!present) {
        return true;
    }
    const auto& value = *state.visual_samples;
    if (!pod(output, value.settings_key) || !pod(output, value.state_version) ||
        !pod(output, value.physical_time) || !pod(output, value.current) ||
        !text(output, value.status) || !pod(output, value.unrepresented_volume) ||
        !pod(output, static_cast<uint32_t>(value.streams.size()))) {
        return false;
    }
    for (const auto& entry : value.streams) {
        const auto& stream = entry.second;
        const auto& cp = stream.runtime.state();
        if (!pod(output, entry.first) || !pod(output, stream.birth_source) ||
            !pod(output, stream.tag) || !pod(output, stream.flags) || !pod(output, stream.model) ||
            !checkpointHeader(output, cp) ||
            !pod(output, static_cast<uint64_t>(cp.samples.size()))) {
            return false;
        }
        for (const auto& sample : cp.samples) {
            if (!sampleFields(output, sample)) {
                return false;
            }
        }
    }
    return true;
}

bool readDomainVisualCache(std::istream& input, SimulationGridDomainState& state) {
    try {
        double physical_time = 0.0;
        uint8_t present = 0;
        if (!pod(input, physical_time) || !std::isfinite(physical_time) || physical_time < 0.0 ||
            !pod(input, present) || present > 1) {
            return false;
        }
        std::shared_ptr<DomainVisualState> value;
        if (present) {
            value = std::make_shared<DomainVisualState>();
            uint32_t stream_count = 0;
            if (!pod(input, value->settings_key) || !pod(input, value->state_version) ||
                !pod(input, value->physical_time) || !pod(input, value->current) ||
                !text(input, value->status) || !pod(input, value->unrepresented_volume) ||
                !pod(input, stream_count) || stream_count > 128 ||
                !std::isfinite(value->physical_time) || value->physical_time != physical_time ||
                !std::isfinite(value->unrepresented_volume) || value->unrepresented_volume < 0.0) {
                return false;
            }
            uint64_t total = 0;
            for (uint32_t i = 0; i < stream_count; ++i) {
                uint64_t key = 0;
                DomainVisualStream stream;
                DomainFields::VisualCheckpoint cp;
                uint64_t count = 0;
                if (!pod(input, key) || !pod(input, stream.birth_source) ||
                    !pod(input, stream.tag) || !pod(input, stream.flags) ||
                    !pod(input, stream.model) || !checkpointHeader(input, cp) ||
                    !pod(input, count) || count > 4000000 || total + count > 4000000 ||
                    cp.identity.physical_time > physical_time ||
                    cp.identity.lane > DomainFields::Lane::Gas) {
                    return false;
                }
                DomainFields::validateVisualSettings(cp.settings);
                if (count > cp.settings.max_samples) {
                    return false;
                }
                cp.samples.resize(static_cast<std::size_t>(count));
                for (auto& sample : cp.samples) {
                    if (!sampleFields(input, sample) || sample.stream_uid != key) {
                        return false;
                    }
                }
                stream.runtime.restore(cp);
                if (!value->streams.emplace(key, std::move(stream)).second) {
                    return false;
                }
                total += count;
            }
            value->sample_count = total;
        }
        state.render_physical_time = physical_time;
        state.visual_samples = std::move(value);
        state.visual_cache_unavailable = false;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

} // namespace RayTrophiSim::Fluid
