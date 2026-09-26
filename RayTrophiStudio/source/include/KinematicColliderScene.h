#pragma once

#include "KinematicColliderSource.h"

#include <cstdint>
#include <string>
#include <vector>

struct SceneData;

namespace RayTrophiSim {

// Scene adapter for the solver-neutral proxy registry. This is the only layer
// allowed to know about rig authoring/runtime data; consumers receive sampled
// primitive proxies and never inspect skeletons themselves.
bool collectKinematicJointPoses(const SceneData& scene,
                                const std::string& character,
                                std::vector<KinematicJointPose>& joints,
                                std::string& error);

bool autoFitKinematicProxySet(SceneData& scene,
                              uint64_t set_id,
                              const KinematicAutoFitOptions& options,
                              uint32_t& created_count,
                              std::string& error);

bool sampleKinematicProxySet(SceneData& scene,
                             uint64_t set_id,
                             float dt,
                             bool discontinuity,
                             std::vector<KinematicProxySample>& samples,
                             std::string& error);

// Read-only instrument for UI/Python/IPC. Sampling a copy prevents a status
// query from advancing the solver's authoritative velocity history.
bool inspectKinematicProxySet(const SceneData& scene,
                              uint64_t set_id,
                              float dt,
                              std::vector<KinematicProxySample>& samples,
                              std::string& error);

bool sampleAllKinematicProxySets(SceneData& scene,
                                 float dt,
                                 bool discontinuity,
                                 std::vector<KinematicProxySample>& samples,
                                 std::string& error);

} // namespace RayTrophiSim
