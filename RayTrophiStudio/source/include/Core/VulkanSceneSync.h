#pragma once

#include <cstddef>
#include <string>
#include <vector>

class World;
class Vec3;
class Renderer;
struct WorldData;
struct SceneData;
namespace Backend {
class VulkanBackendAdapter;
class IViewportBackend;
class IBackend;
}

namespace AppRuntime {

void syncVulkanWorld(Backend::VulkanBackendAdapter* backend, World& world,
                     WorldData& data, bool allow_lut_regeneration = true);
void buildRasterScene(Backend::IViewportBackend& backend, const SceneData& scene,
                      const Renderer& renderer);

// Raster viewport half of the simulation write-back, shared by Studio's frame
// tail and the Player loop. Refits the named deformed bodies (rigid / soft /
// fracture shard) in place: facade-backed nodes from their triangles, flat nodes
// from their SoA. Returns false on the first node that cannot be refit; the
// caller then schedules a full raster rebuild.
bool refitRasterDeformedNodes(Backend::IViewportBackend& backend, SceneData& scene,
                              const std::vector<std::string>& nodes);

// Raster particle pass, shared by SceneUI and the Player loop. Builds the
// billboard/sphere/pulled draw lists of every visible particle system and fluid
// domain and hands them to the viewport. `clear` uploads empty lists instead
// (Studio's Debug display mode draws an overlay). Returns the number of draws
// handed over (quads + spheres + pulled systems), for runtime telemetry.
// Cloud / local weather publication to the Vulkan adapters (RayFusion raster and
// Vulkan RT), shared by Studio's frame loop and the Player. Studio derives the
// cloud time from its timeline frame, the Player from its game clock. Publishes
// when the time, the cloud revision, the local weather/fog revision or the set
// of target backends changed; invalidate() forces the next call to publish
// (a new project or a recreated device).
struct CloudPushState {
    float time = -1.0f;
    unsigned long long cloud_revision = ~0ull;
    unsigned long long local_revision = ~0ull;
    const void* render_backend = nullptr;
    const void* viewport_backend = nullptr;
    void invalidate() { *this = CloudPushState{}; }
};
// Returns true when the cloud state was pushed this call (the adapters reset
// their accumulation, so RayFusion redraws).
bool pushCloudState(World& world, SceneData& scene, float cloud_time_s, const Vec3& camera,
                    CloudPushState& state, Backend::IBackend* render_backend,
                    Backend::IBackend* viewport_backend);

// Volume table publication (VDB + live gas), shared by SceneUI and the Player.
// VolumetricRenderer::syncVolumetricData is stateless per target, and the live
// dense-gas device addresses inside the packet are resolved per device, so each
// distinct backend gets its own call. Studio passes its render backend and the
// separate raster viewport; the Player passes only its single viewport device.
void publishVolumes(SceneData& scene, Renderer& renderer, Backend::IBackend* render_backend,
                    Backend::IBackend* viewport_backend);

std::size_t uploadParticleBillboards(Backend::VulkanBackendAdapter& viewport,
                                     const SceneData& scene, bool clear);

} // namespace AppRuntime
