#include "RtApiInternal.h"
#include "Renderer.h"
#include "Api/DomainWallsApi.h"
#include "Fluid/DomainWallSettings.h"

namespace rtapi {
namespace {

RayTrophiSim::SimulationGridDomainDesc* findWallDomain(const std::string& domain,
    Result& status) {
    if (!g_ctx) {
        status = notBound();
        return nullptr;
    }
    FluidDomainInfo info;
    status = getFluidDomain(domain, info);
    if (!status.ok) {
        return nullptr;
    }
    auto& domains = g_ctx->scene.ensureParticleSimulationSystem().gridDomains();
    for (auto& candidate : domains) {
        if (candidate.name == info.name && RayTrophiSim::simulationDomainHasLiquid(candidate)) {
            return &candidate;
        }
    }
    status = Result::fail("particle matter domain not found: " + domain);
    return nullptr;
}

} // namespace

Result getDomainWallContact(const std::string& domain, nlohmann::json& output) {
    Result status;
    auto* descriptor = findWallDomain(domain, status);
    if (!descriptor) {
        return status;
    }
    output = RayTrophiSim::Fluid::domainWallSettings(descriptor->fluid_params,
        descriptor->boundary_mode == RayTrophiSim::SimulationGridDomainBoundaryMode::Closed);
    return Result::success();
}

Result setDomainWallContact(const std::string& domain, const nlohmann::json& patch,
    nlohmann::json& output) {
    if (!g_ctx) {
        return notBound();
    }
    if (renderJobActive()) {
        return Result::fail("scene is locked by the final render job");
    }
    Result status;
    auto* descriptor = findWallDomain(domain, status);
    if (!descriptor) {
        return status;
    }
    std::string error;
    if (!RayTrophiSim::Fluid::patchDomainWallSettings(descriptor->fluid_params, patch, error)) {
        return Result::fail(error);
    }
    g_ctx->scene.requestSimulationTimelineRenderResync();
    g_ctx->renderer.resetCPUAccumulation();
    if (g_ctx->backend_ptr) {
        g_ctx->backend_ptr->resetAccumulation();
    }
    g_ctx->start_render = true;
    output = RayTrophiSim::Fluid::domainWallSettings(descriptor->fluid_params,
        descriptor->boundary_mode == RayTrophiSim::SimulationGridDomainBoundaryMode::Closed);
    return Result::success();
}

} // namespace rtapi
