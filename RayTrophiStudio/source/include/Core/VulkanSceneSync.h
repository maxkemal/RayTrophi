#pragma once

#include <cstddef>
#include <string>
#include <vector>

class World;
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
