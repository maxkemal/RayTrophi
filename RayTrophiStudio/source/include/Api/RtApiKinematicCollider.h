#pragma once

#include "KinematicColliderSource.h"

#include <cstdint>
#include <vector>

namespace rtapi {

struct Result;

Result listKinematicProxySets(
    std::vector<RayTrophiSim::KinematicProxySet>& out_sets);
Result getKinematicProxySet(
    uint64_t set_id,
    RayTrophiSim::KinematicProxySet& out_set);
Result createKinematicProxySet(
    const RayTrophiSim::KinematicProxySet& requested,
    RayTrophiSim::KinematicProxySet& created);
Result updateKinematicProxySet(
    uint64_t set_id,
    const RayTrophiSim::KinematicProxySet& requested);
Result removeKinematicProxySet(uint64_t set_id);

Result setKinematicProxy(
    uint64_t set_id,
    const RayTrophiSim::KinematicProxyDesc& requested,
    RayTrophiSim::KinematicProxyDesc& stored);
Result removeKinematicProxy(uint64_t set_id, uint64_t proxy_id);
Result autoFitKinematicProxySet(
    uint64_t set_id,
    const RayTrophiSim::KinematicAutoFitOptions& options,
    uint32_t& created_count);

// Read-only: does not advance authoritative solver motion history.
Result sampleKinematicProxySet(
    uint64_t set_id,
    float dt,
    std::vector<RayTrophiSim::KinematicProxySample>& out_samples);

} // namespace rtapi
