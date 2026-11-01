#pragma once

#include "SimCache.h"
#include "SimulationDomainLanes.h"

#include <string>
#include <utility>
#include <vector>

namespace RayTrophiSim::MatterCacheRestore {

// Validate every RAM system before restoring any grid, particle, MSF or body.
// Cached metadata is borrowed directly; this does not decompress/copy grids.
template <class Systems, class FrameCache>
bool validateRamFrame(const Systems& systems, const FrameCache& frames,
                      int frame, std::string& error) {
    const auto found = frames.find(frame);
    if (found == frames.end()) {
        error.clear();
        return true; // The caller may try its ordinary disk fallback.
    }
    if (found->second.size() != systems.size()) {
        error = "cached system count differs from the scene; rebake simulation";
        return false;
    }
    for (std::size_t i = 0; i < systems.size(); ++i) {
        if (!systems[i].runtime) {
            continue;
        }
        const auto& domains = systems[i].runtime->gridDomains();
        const auto& cached = found->second[i];
        if (cached.size() != domains.size()) {
            error = "cached domain count differs from the scene; rebake simulation";
            return false;
        }
        for (std::size_t j = 0; j < domains.size(); ++j) {
            if (!validateMatterCacheLane(cached[j].meta, domains[j], error)) {
                return false;
            }
        }
    }
    error.clear();
    return true;
}

struct DiskSystemFrame {
    std::vector<SimulationGridDomainState> domains;
    std::vector<MaterialStateFieldSnapshot> material_fields;
};

// Stage and validate the complete frame. No runtime is mutated on failure, and
// the output retains its previous contents. Invalid frames cannot look restored.
template <class Systems, class ValidSystemIds, class ReadFrame>
bool stageDiskFrame(const Systems& systems, const ValidSystemIds& valid_ids,
                    const std::string& directory, int frame,
                    std::vector<DiskSystemFrame>& output, std::string& error,
                    ReadFrame read_frame) {
    std::vector<DiskSystemFrame> candidate(systems.size());
    for (std::size_t i = 0; i < systems.size(); ++i) {
        const auto& system = systems[i];
        if (!system.runtime || valid_ids.count(system.id) == 0) {
            continue;
        }
        auto& loaded = candidate[i];
        if (!read_frame(directory, system.id, frame, loaded.domains, loaded.material_fields)) {
            error = "system " + std::to_string(system.id) + " frame " + std::to_string(frame) +
                " cache is missing, incompatible or corrupt; rebake simulation";
            return false;
        }
        if (!validateMatterCacheLanes(loaded.domains, system.runtime->gridDomains(), error)) {
            return false;
        }
    }
    output = std::move(candidate);
    error.clear();
    return true;
}

template <class Systems, class ValidSystemIds>
bool stageDiskFrame(const Systems& systems, const ValidSystemIds& valid_ids,
                    const std::string& directory, int frame,
                    std::vector<DiskSystemFrame>& output, std::string& error) {
    return stageDiskFrame(systems, valid_ids, directory, frame, output, error,
                          SimCache::readSystemFrame);
}

} // namespace RayTrophiSim::MatterCacheRestore
