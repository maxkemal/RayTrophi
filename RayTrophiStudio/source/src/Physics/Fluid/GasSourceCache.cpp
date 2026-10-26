#include "Fluid/GasSourceCache.h"
#include "ParticleSimulation.h"
#include <cmath>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <utility>

namespace RayTrophiSim::Fluid {
namespace {
template <typename T> bool pod(std::ostream& stream, const T& value) {
    stream.write(reinterpret_cast<const char*>(&value), sizeof(value));
    return bool(stream);
}
template <typename T> bool pod(std::istream& stream, T& value) {
    stream.read(reinterpret_cast<char*>(&value), sizeof(value));
    return bool(stream);
}
}
bool writeGasSourceCache(std::ostream& stream, const SimulationGridDomainState& state) {
    SimulationGridDomainState snapshot;
    snapshot.gas_source_images = state.gas_source_images;
    snapshot.gas_source_device = state.gas_source_device;
    snapshot.gas_source_device_current = state.gas_source_device_current;
    if (!snapshotGasSources(snapshot)) {
        return false;
    }
    const auto& images = snapshot.gas_source_images;
    if (!pod(stream, uint32_t(images.size()))) {
        return false;
    }
    if (images.empty()) {
        return true;
    }
    const std::size_t cells = state.grid.getCellCount();
    try {
        validateGasSourceBudget(cells, images.size());
    } catch (const std::exception&) {
        return false;
    }
    for (const auto& [uid, image] : images) {
        if (image.name.size() > 4096 || image.density.size() != cells ||
            image.fuel.size() != cells || !image.pending_density.empty() ||
            !image.pending_fuel.empty() || !pod(stream, uid) ||
            !pod(stream, uint32_t(image.name.size()))) {
            return false;
        }
        stream.write(image.name.data(), image.name.size());
        for (const auto* values : {&image.density, &image.fuel}) {
            stream.write(reinterpret_cast<const char*>(values->data()), cells * sizeof(float));
        }
    }
    return bool(stream);
}

bool readGasSourceCache(std::istream& stream, SimulationGridDomainState& state) {
    uint32_t count = 0;
    if (!pod(stream, count) || count > 9) {
        return false;
    }
    GasSourceImages images;
    const std::size_t cells = state.grid.getCellCount();
    if (count > 0) {
        try {
            validateGasSourceBudget(cells, count);
        } catch (const std::exception&) {
            return false;
        }
    }
    for (uint32_t index = 0; index < count; ++index) {
        uint64_t uid = 0;
        uint32_t length = 0;
        if (!pod(stream, uid) || !pod(stream, length) || length > 4096 || images.count(uid)) {
            return false;
        }
        auto& image = images[uid];
        image.name.resize(length);
        stream.read(image.name.data(), length);
        for (auto* values : {&image.density, &image.fuel}) {
            values->resize(cells);
            stream.read(reinterpret_cast<char*>(values->data()), cells * sizeof(float));
            if (!stream) {
                return false;
            }
            for (float value : *values) {
                if (!std::isfinite(value) || value < 0.0f) {
                    return false;
                }
            }
        }
    }
    if (count && !images.count(0)) {
        return false;
    }
    state.gas_source_images = std::move(images);
    state.gas_source_device.reset();
    state.gas_source_device_current = false;
    return bool(stream);
}
} // namespace RayTrophiSim::Fluid
