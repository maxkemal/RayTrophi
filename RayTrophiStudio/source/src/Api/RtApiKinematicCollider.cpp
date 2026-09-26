#include "Api/RtApiKinematicCollider.h"

#include "KinematicColliderScene.h"
#include "RtApiInternal.h"

#include <cmath>

namespace rtapi {
namespace {

Result mutationAllowed() {
    if (!g_ctx) {
        return notBound();
    }
    if (renderJobActive()) {
        return Result::fail("scene is locked by the final render job");
    }
    return Result::success();
}

Result registryResult(bool ok, const std::string& error) {
    return ok ? Result::success() : Result::fail(error);
}

void invalidateKinematicColliderAuthoring() {
    g_ctx->scene.kinematic_colliders.resetMotionHistory();
    invalidateScriptSimulation();
    g_viewport_raster_rebuild_pending = true;
}

} // namespace

Result listKinematicProxySets(
    std::vector<RayTrophiSim::KinematicProxySet>& out_sets) {
    if (!g_ctx) {
        return notBound();
    }
    out_sets = g_ctx->scene.kinematic_colliders.sets();
    return Result::success();
}

Result getKinematicProxySet(
    uint64_t set_id,
    RayTrophiSim::KinematicProxySet& out_set) {
    if (!g_ctx) {
        return notBound();
    }
    const RayTrophiSim::KinematicProxySet* set =
        g_ctx->scene.kinematic_colliders.findSet(set_id);
    if (!set) {
        return Result::fail("unknown_proxy_set");
    }
    out_set = *set;
    return Result::success();
}

Result createKinematicProxySet(
    const RayTrophiSim::KinematicProxySet& requested,
    RayTrophiSim::KinematicProxySet& created) {
    Result allowed = mutationAllowed();
    if (!allowed.ok) {
        return allowed;
    }
    std::string error;
    if (!g_ctx->scene.kinematic_colliders.createSet(
            requested, created, error)) {
        return Result::fail(error);
    }
    invalidateKinematicColliderAuthoring();
    return Result::success();
}

Result updateKinematicProxySet(
    uint64_t set_id,
    const RayTrophiSim::KinematicProxySet& requested) {
    Result allowed = mutationAllowed();
    if (!allowed.ok) {
        return allowed;
    }
    std::string error;
    if (!g_ctx->scene.kinematic_colliders.updateSet(
            set_id, requested, error)) {
        return Result::fail(error);
    }
    invalidateKinematicColliderAuthoring();
    return Result::success();
}

Result removeKinematicProxySet(uint64_t set_id) {
    Result allowed = mutationAllowed();
    if (!allowed.ok) {
        return allowed;
    }
    std::string error;
    if (!g_ctx->scene.kinematic_colliders.removeSet(set_id, error)) {
        return Result::fail(error);
    }
    invalidateKinematicColliderAuthoring();
    return Result::success();
}

Result setKinematicProxy(
    uint64_t set_id,
    const RayTrophiSim::KinematicProxyDesc& requested,
    RayTrophiSim::KinematicProxyDesc& stored) {
    Result allowed = mutationAllowed();
    if (!allowed.ok) {
        return allowed;
    }
    std::string error;
    if (!g_ctx->scene.kinematic_colliders.setProxy(
            set_id, requested, stored, error)) {
        return Result::fail(error);
    }
    invalidateKinematicColliderAuthoring();
    return Result::success();
}

Result removeKinematicProxy(uint64_t set_id, uint64_t proxy_id) {
    Result allowed = mutationAllowed();
    if (!allowed.ok) {
        return allowed;
    }
    std::string error;
    if (!g_ctx->scene.kinematic_colliders.removeProxy(
            set_id, proxy_id, error)) {
        return Result::fail(error);
    }
    invalidateKinematicColliderAuthoring();
    return Result::success();
}

Result autoFitKinematicProxySet(
    uint64_t set_id,
    const RayTrophiSim::KinematicAutoFitOptions& options,
    uint32_t& created_count) {
    Result allowed = mutationAllowed();
    if (!allowed.ok) {
        return allowed;
    }
    std::string error;
    if (!RayTrophiSim::autoFitKinematicProxySet(
            g_ctx->scene, set_id, options, created_count, error)) {
        return Result::fail(error);
    }
    invalidateKinematicColliderAuthoring();
    return Result::success();
}

Result sampleKinematicProxySet(
    uint64_t set_id,
    float dt,
    std::vector<RayTrophiSim::KinematicProxySample>& out_samples) {
    if (!g_ctx) {
        return notBound();
    }
    if (!(dt > 0.0f) || !std::isfinite(dt)) {
        return Result::fail("sample_dt_must_be_positive_and_finite");
    }
    std::string error;
    if (!RayTrophiSim::inspectKinematicProxySet(
            g_ctx->scene, set_id, dt, out_samples, error)) {
        return Result::fail(error);
    }
    return Result::success();
}

} // namespace rtapi
