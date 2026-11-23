#include "Core/VulkanSceneSync.h"

#include "Backend/VulkanBackend.h"
#include "Backend/IViewportBackend.h"
#include "World.h"
#include "Texture.h"
#include "AtmosphereLUT.h"
#include "Renderer.h"
#include "scene_data.h"
#include "globals.h"
#include "UI/ParticleBillboardBuilder.h"
#include "VolumetricRenderer.h"
#include "Atmosphere/AtmosphereLocalWeather.h"
#include "Atmosphere/LightningStrikeResolver.h"

#include <chrono>
#include <unordered_map>
#include <unordered_set>

namespace AppRuntime {

void syncVulkanWorld(Backend::VulkanBackendAdapter* vulkanBackend, World& world,
                     WorldData& wd, bool allowLutRegen) {
    if (!vulkanBackend) {
        return;
    }

    Backend::IBackend* envCacheKey = static_cast<Backend::IBackend*>(vulkanBackend);
    struct VulkanEnvCache {
        const Texture* texture = nullptr;
        int64_t handle = 0;
        uint64_t textureCacheGeneration = 0;
    };
    // Adapter basina onbellek: iki adapter AYRI VkDevice, yani ayni CPU
    // Texture'i her ikisine de ayri ayri yuklenir ve handle'lari karismaz.
    static std::unordered_map<Backend::IBackend*, VulkanEnvCache> s_vulkanEnvCache;

    auto uploadVulkanEnvironmentTexture = [&](const Texture* texture) -> int64_t {
        if (!texture || !texture->is_loaded() || texture->width <= 0 || texture->height <= 0) {
            s_vulkanEnvCache.erase(envCacheKey);
            return 0;
        }

        auto& cache = s_vulkanEnvCache[envCacheKey];
        VulkanRT::ImageHandle existingEnvImage{};
        const VkFormat expectedFormat = (texture->is_hdr && !texture->float_pixels.empty())
            ? VK_FORMAT_R16G16B16A16_SFLOAT   // uploadEnvironmentTexture
            : VK_FORMAT_R8G8B8A8_UNORM;
        const bool cacheValid =
            cache.texture == texture &&
            cache.handle != 0 &&
            cache.textureCacheGeneration == vulkanBackend->textureCacheGeneration() &&
            vulkanBackend->tryGetUploadedImageHandle(cache.handle, existingEnvImage) &&
            existingEnvImage.width == static_cast<uint32_t>(texture->width) &&
            existingEnvImage.height == static_cast<uint32_t>(texture->height) &&
            existingEnvImage.format == expectedFormat;
        if (cacheValid) {
            return cache.handle;
        }

        const auto uploadStart = std::chrono::steady_clock::now();
        int64_t envHandle = 0;
        uint64_t clampedTexels = 0;
        if (texture->is_hdr && !texture->float_pixels.empty()) {
            envHandle = vulkanBackend->uploadEnvironmentTexture(
                reinterpret_cast<const float*>(texture->float_pixels.data()),
                static_cast<uint32_t>(texture->width),
                static_cast<uint32_t>(texture->height),
                &clampedTexels);
        } else if (!texture->pixels.empty()) {
            envHandle = vulkanBackend->uploadTexture2D(
                texture->pixels.data(),
                static_cast<uint32_t>(texture->width),
                static_cast<uint32_t>(texture->height),
                4,
                false,
                false);
        }
        const auto uploadMs = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - uploadStart).count();
        SCENE_LOG_INFO("[World] Vulkan environment upload_ms=" + std::to_string(uploadMs) +
                       " uploaded=" + std::to_string(envHandle > 0) +
                       " format=" + std::string(texture->is_hdr ? "rgba16f" : "rgba8") +
                       " clamped_texels=" + std::to_string(clampedTexels) +
                       " size=" + std::to_string(texture->width) + "x" +
                       std::to_string(texture->height));
        cache.texture = texture;
        cache.handle = envHandle;
        cache.textureCacheGeneration = vulkanBackend->textureCacheGeneration();
        return envHandle;
    };

    if (wd.mode == WORLD_MODE_HDRI) {
        // ★ HDRI tembel yuklenir (`World::setMode`). Proje acilisinda mod
        //   HDRI olarak deserialize edilip doku henuz yuklenmemis olabilir;
        //   burada bir kez zorlamak, "dosya var ama arka plan duz renk"
        //   halini sessiz birakmamak icin.
        // ★ Bir kez DENENIR, path basina. Dosya yoksa `setHDRI` sessizce
        //   basarisiz olur; kapisiz birakmak her senkronda bir disk
        //   erisimi demekti (kare basina, HDRI modunda).
        if (!world.hasHDRI() && !world.getHDRIPath().empty()) {
            static std::unordered_set<std::string> s_attemptedHdriPaths;
            if (s_attemptedHdriPaths.insert(world.getHDRIPath()).second) {
                world.setMode(WORLD_MODE_HDRI);
                wd = world.getGPUData();
            }
        }
        int64_t envHandle = uploadVulkanEnvironmentTexture(world.getHDRITexture());
        vulkanBackend->setEnvironmentMap(envHandle != 0 ? envHandle : 0);
        if (envHandle == 0) {
            static std::unordered_set<Backend::IBackend*> s_warnedMissingEnv;
            if (s_warnedMissingEnv.insert(envCacheKey).second) {
                SCENE_LOG_WARN("[World] HDRI mode is active but no environment texture "
                               "reached this backend; its background falls back to the "
                               "world solid colour. Path: '" +
                               world.getHDRIPath() + "'");
            }
        }
    } else if (wd.mode == WORLD_MODE_NISHITA && wd.advanced.env_overlay_enabled != 0) {
        int64_t envHandle = uploadVulkanEnvironmentTexture(
            world.getNishitaEnvOverlayTexture());
        vulkanBackend->setEnvironmentMap(envHandle != 0 ? envHandle : 0);
    } else {
        s_vulkanEnvCache.erase(envCacheKey);
        vulkanBackend->setEnvironmentMap(0);
    }

    // ★★★★ Iki olcu, TEK kapi: paylasilan `needsLUTUpdate()` (dunya degisti)
    //   VEYA bu adapter'in KENDI imzasi tutmuyor (oteki adapter uretip
    //   paylasilan bayragi yuttu). Ikisi de kisitin ICINDE, yani slider
    //   surerken maliyet artmaz.
    const bool lutStaleHere =
        wd.mode == WORLD_MODE_NISHITA &&
        (world.needsLUTUpdate() || vulkanBackend->atmosphereLutStale(wd));
    if (allowLutRegen && lutStaleHere) {
        if (vulkanBackend->generateAtmosphereLUTGPU(&wd)) {
            world.clearLUTDirty();
            wd = world.getGPUData();
            return;
        }

        if (world.flushLUT()) {
            wd = world.getGPUData();
        }
        if (auto* al = world.getLUT()) {
            if (al->is_initialized()) {
                vulkanBackend->uploadAtmosphereLUT(al);
                wd = world.getGPUData();
            }
        }
    }

    vulkanBackend->setWorldData(&wd);
    // The cloud authority is not in WorldData (CUDA-shared POD); same
    // funnel, own call. After setWorldData: the packet needs this sync's
    // planet radius / altitude.
    vulkanBackend->setCloudState(world.getClouds(),
                                 world.getCloudTime(),
                                 world.cloudWindOffset(),
                                 world.cloudWindVelocity());

}

void buildRasterScene(Backend::IViewportBackend& backend, const SceneData& scene,
                      const Renderer& renderer) {
    backend.buildRasterGeometry(scene.world.objects);
    backend.syncRasterSkinnedVertices(scene.world.objects, renderer.finalBoneMatrices);
}

bool refitRasterDeformedNodes(Backend::IViewportBackend& backend, SceneData& scene,
                              const std::vector<std::string>& nodes) {
    for (const auto& node : nodes) {
        auto tris = scene.collectNodeTriangles(node);
        bool ok = false;
        if (!tris.empty()) {
            ok = backend.updateRasterMeshFromTriangles(node, tris);
        } else if (TriangleMesh* mesh = scene.getFlatNodeMesh(node)) {
            // Flat (direct SoA) mesh has no facades: refit the raster vertices from
            // its SoA so a rigid-on-flat body stays realtime (no full rebuild).
            ok = backend.updateRasterMeshFromMeshSoA(node, mesh);
        }
        if (!ok) {
            return false;
        }
    }
    return true;
}

std::size_t uploadParticleBillboards(Backend::VulkanBackendAdapter& viewport,
                                     const SceneData& scene, bool clear) {
    static ParticleBillboardUpload upload;  // reused: keeps vector capacity
    if (clear || !scene.camera) {
        upload.additive.clear();
        upload.alpha.clear();
        upload.pulled.clear();
        upload.spheres.clear();
        viewport.uploadParticleBillboards(upload);
        return 0;
    }
    // The viewport's own device decides which systems can be pulled from the
    // simulation buffers; the rest become CPU quads.
    void* viewport_device = nullptr;
    if (auto* device = viewport.getVulkanDevice()) {
        viewport_device = static_cast<void*>(device->getDevice());
    }
    ParticleBillboardBuilder::build(scene, viewport_device, upload);
    viewport.uploadParticleBillboards(upload);
    return upload.additive.size() + upload.alpha.size() + upload.spheres.size() +
           upload.pulled.size();
}

void publishVolumes(SceneData& scene, Renderer& renderer, Backend::IBackend* render_backend,
                    Backend::IBackend* viewport_backend) {
    WorldData world_data = renderer.world.getGPUData();
    if (render_backend) {
        VolumetricRenderer::syncVolumetricData(scene, render_backend, &world_data);
    }
    if (viewport_backend && viewport_backend != render_backend) {
        VolumetricRenderer::syncVolumetricData(scene, viewport_backend, &world_data);
    }
}

bool pushCloudState(World& world, SceneData& scene, float cloud_time_s, const Vec3& camera,
                    CloudPushState& state, Backend::IBackend* render_backend,
                    Backend::IBackend* viewport_backend) {
    world.setCloudTime(cloud_time_s);
    // Local weather (ATMOSPHERE_LOCAL_WINDS_FOG.md §2-4): terrain copy, birth
    // environment and fog map. Before the push below: the cells the push packs
    // read the environment published here.
    atmosphere::syncLocalWeather(world, cloud_time_s, camera);
    // Ground strikes attach to this scene's geometry (LightningStrike.h). One
    // resolver per scene object for the life of the process.
    static const SceneData* s_lightning_scene = nullptr;
    if (s_lightning_scene != &scene) {
        atmosphere::installLightningStrikeResolver(scene);
        s_lightning_scene = &scene;
    }
    const auto local_env = atmosphere::localWeatherEnvironment();
    const auto local_fog = atmosphere::localFogMapSnapshot();
    // Environment and fog map revisions in one key: both are published here,
    // outside the cloud revision. A terrain edit or a sun move re-places the
    // cell births and a new fog map changes the fog layer without touching
    // the cloud revision; both must reach the GPU too.
    const unsigned long long local_revision =
        (local_env ? local_env->revision : 0) * 1000003ull + (local_fog ? local_fog->revision : 0);
    const unsigned long long cloud_revision = world.cloudRevision();
    if (cloud_time_s == state.time && cloud_revision == state.cloud_revision &&
        local_revision == state.local_revision && render_backend == state.render_backend &&
        viewport_backend == state.viewport_backend) {
        return false;
    }
    state.time = cloud_time_s;
    state.cloud_revision = cloud_revision;
    state.local_revision = local_revision;
    state.render_backend = render_backend;
    state.viewport_backend = viewport_backend;
    for (Backend::IBackend* backend : {render_backend, viewport_backend}) {
        if (auto* vulkan = dynamic_cast<Backend::VulkanBackendAdapter*>(backend)) {
            vulkan->setCloudState(world.getClouds(), cloud_time_s, world.cloudWindOffset(),
                                  world.cloudWindVelocity());
        }
        if (render_backend == viewport_backend) {
            break;  // one device, one push
        }
    }
    return true;
}

} // namespace AppRuntime
