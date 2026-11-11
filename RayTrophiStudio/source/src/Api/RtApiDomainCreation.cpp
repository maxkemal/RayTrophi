#include "Api/DomainCreationApi.h"
#include "RtApiInternal.h"
#include "Transform.h"
#include "TriangleMesh.h"
#include <algorithm>
#include <cmath>

namespace rtapi {
namespace {
bool hasFlatNode(const std::string &name) {
    if (!g_ctx || name.empty() || g_ctx->scene.isEditorPendingDeleteObjectName(name)) {
        return false;
    }
    for (const auto &object : g_ctx->scene.world.objects) {
        const auto mesh = std::dynamic_pointer_cast<TriangleMesh>(object);
        if (mesh && mesh->nodeName == name && mesh->geometry) {
            return true;
        }
    }
    return false;
}

bool finite(const Vec3 &value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}
} // namespace

std::string selectedFlatDomainSource() {
    if (!g_ctx || g_ctx->selection.selected.type != SelectableType::Object) {
        return {};
    }
    const auto &selection = g_ctx->selection.selected;
    const auto name = selection.mesh_object ? selection.mesh_object->nodeName : selection.name;
    return hasFlatNode(name) ? name : std::string();
}

Result createPlacedMatterDomain(const std::string &name, const std::string &source,
                                nlohmann::json &output) {
    if (!g_ctx) {
        return notBound();
    }
    if (renderJobActive()) {
        return Result::fail("scene is locked by the final render job");
    }
    RayTrophiSim::SimulationGridDomainDesc descriptor;
    descriptor.type = RayTrophiSim::SimulationDomainType::Matter;
    descriptor.backend = RayTrophiSim::defaultSimulationDomainBackend();
    if (!source.empty()) {
        if (!hasFlatNode(source)) {
            return Result::fail("flat mesh object not found: " + source);
        }
        bool seen = false;
        Vec3 low(0.0f), high(0.0f);
        // Aggregate all flat mesh parts sharing the canonical node identity.
        for (const auto &object : g_ctx->scene.world.objects) {
            const auto mesh = std::dynamic_pointer_cast<TriangleMesh>(object);
            if (!mesh || mesh->nodeName != source || !mesh->geometry) {
                continue;
            }
            const auto *local = mesh->geometry->get_attribute_data<Vec3>("P_orig");
            const auto *world = mesh->geometry->get_attribute_data<Vec3>("P");
            const bool transformLocal = local && mesh->transform;
            if (!transformLocal && !world) {
                return Result::fail("flat mesh has no canonical position data: " + source);
            }
            const auto matrix =
                transformLocal ? mesh->transform->getFinal() : Matrix4x4::identity();
            for (std::size_t i = 0; i < mesh->geometry->get_vertex_count(); ++i) {
                const Vec3 point = transformLocal ? matrix.transform_point(local[i]) : world[i];
                if (!finite(point)) {
                    return Result::fail("flat mesh contains nonfinite world positions: " + source);
                }
                low = seen ? Vec3::min(low, point) : point;
                high = seen ? Vec3::max(high, point) : point;
                seen = true;
            }
        }
        const Vec3 extent = high - low;
        if (!seen || !finite(extent) || extent.x <= 0 || extent.y <= 0 || extent.z <= 0) {
            return Result::fail("source mesh must have finite nonempty three-dimensional bounds");
        }
        descriptor.bounds_min = low;
        descriptor.bounds_max = high;
        descriptor.voxel_size =
            std::max({extent.x, extent.y, extent.z}) /
            static_cast<float>(std::max(
                {descriptor.resolution_x, descriptor.resolution_y, descriptor.resolution_z, 1}));
    }
    auto &runtime = g_ctx->scene.ensureParticleSimulationSystem();
    const auto exists = [&](const std::string &candidate) {
        return std::any_of(runtime.gridDomains().begin(), runtime.gridDomains().end(),
                           [&](const auto &domain) { return domain.name == candidate; });
    };
    if (!name.empty() && exists(name)) {
        return Result::fail("domain already exists: " + name);
    }
    descriptor.name = name;
    if (descriptor.name.empty()) {
        std::size_t suffix = 1;
        do {
            descriptor.name = "Physics Domain " + std::to_string(suffix++);
        } while (exists(descriptor.name));
    }
    descriptor.source_mode = RayTrophiSim::SimulationGridDomainSourceMode::ManualBox;
    descriptor.source_name.clear();
    const auto &created = g_ctx->scene.addSimulationGridDomain(descriptor);
    const int index = static_cast<int>(runtime.gridDomains().size()) - 1;
    g_ctx->selection.selectSimulationDomain(g_ctx->scene.active_particle_system_index, index,
                                            created.name);
    g_ctx->selection.selected.position = (created.bounds_min + created.bounds_max) * 0.5f;
    g_ctx->selection.selected.scale = created.bounds_max - created.bounds_min;
    invalidateScriptSimulation();
    g_ctx->renderer.resetCPUAccumulation();
    if (g_ctx->backend_ptr) {
        g_ctx->backend_ptr->resetAccumulation();
    }
    g_ctx->start_render = true;
    output = {{"domain", created.name},
              {"domain_index", index},
              {"placement", source.empty() ? "ground_default" : "flat_object_bounds"},
              {"source_object", source},
              {"bounds_min", {created.bounds_min.x, created.bounds_min.y, created.bounds_min.z}},
              {"bounds_max", {created.bounds_max.x, created.bounds_max.y, created.bounds_max.z}}};
    return Result::success();
}
} // namespace rtapi
