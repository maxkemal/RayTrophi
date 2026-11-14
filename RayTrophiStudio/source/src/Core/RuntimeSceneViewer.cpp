#include "Core/RuntimeSceneViewer.h"

#include "Core/VulkanSceneSync.h"
#include "Core/RuntimeSimulationClock.h"
#include "Backend/VulkanViewportBackend.h"
#include "ProjectManager.h"
#include "InstanceManager.h"
#include "MaterialManager.h"
#include "PrincipledBSDF.h"
#include "Backend/SceneTextureManager.h"
#include "UIContext.h"
#include "SceneHistory.h"
#include "SceneSelection.h"
#include "Renderer.h"
#include "scene_data.h"
#include "ParticleSimulation.h"
#include "TriangleMesh.h"
#include "HittableInstance.h"
#include "Camera.h"
#include "VDBVolume.h"
#include "GasVolume.h"
#include "globals.h"

#include <SDL.h>
#include <filesystem>
#include <fstream>
#include <exception>

extern std::unique_ptr<Backend::IViewportBackend> g_viewport_backend;
extern bool g_camera_dirty;
extern bool g_lights_dirty;
extern bool g_materials_dirty;
extern bool g_texture_pool_dirty;
extern bool g_geometry_dirty;
extern bool g_world_dirty;
extern bool g_viewport_world_dirty;
extern bool g_solid_viewport_active;
extern bool g_material_preview_viewport_active;
extern bool g_vulkan_device_lost;
extern bool g_dense_mesh_as_hittable;
extern bool g_gpu_refit_pending;
extern bool g_geometry_deform_pending;
extern bool g_viewport_raster_rebuild_pending;
extern bool g_viewport_raster_scatter_update_pending;
extern bool g_gas_volumes_dirty;
extern bool g_cpu_bvh_refit_pending;

namespace AppRuntime {
namespace {

rtapi::Result validateProjectSlice(const std::string& path) {
    std::ifstream input(projectPathFromUtf8(path), std::ios::binary);
    if (!input) {
        return rtapi::Result::fail("cannot read project: " + path);
    }
    const auto metadata = nlohmann::json::parse(input, nullptr, false);
    if (metadata.is_discarded() || !metadata.is_object()) {
        return rtapi::Result::fail("invalid project metadata: " + path);
    }
    // These systems still require editor finalization. Refuse before the shared
    // loader clears the current scene, rather than silently displaying less.
    // Scatter ("instances") loads as native InstanceGroups that the shared raster
    // build reads directly (VulkanViewportBackend::buildRasterGeometry); no
    // editor materialization is involved on a Vulkan device, so it is accepted.
    // VDB / gas volumes publish through the shared volume table service
    // (AppRuntime::publishVolumes); fracture still needs editor finalization.
    for (const char* key : {"fracture_recipes"}) {
        const auto found = metadata.find(key);
        if (found != metadata.end() && !found->is_null() && !found->empty()) {
            return rtapi::Result::fail(std::string("unsupported_in_player_scene: ") + key);
        }
    }
    const auto water = metadata.find("water");
    if (water != metadata.end() && !water->is_null()) {
        if (water->is_object()) {
            // Current Studio files always carry water settings, even with no
            // surfaces. Settings alone do not require the excluded water runtime.
            const auto surfaces = water->find("water_surfaces");
            if (surfaces != water->end() && !surfaces->empty()) {
                return rtapi::Result::fail("unsupported_in_player_scene: water");
            }
        } else if (!water->empty()) {
            return rtapi::Result::fail("unsupported_in_player_scene: water");
        }
    }
    // Particle systems run on the shared simulation clock and draw through the
    // shared raster billboard pass (Core/VulkanSceneSync); no longer refused.
    const auto hair = metadata.find("hair");
    if (hair != metadata.end() && hair->is_object()) {
        const auto grooms = hair->find("grooms");
        if (grooms != hair->end() && !grooms->empty()) {
            return rtapi::Result::fail("unsupported_in_player_scene: hair");
        }
    }
    return rtapi::Result::success();
}

std::shared_ptr<TriangleMesh> flatMesh(const std::shared_ptr<Hittable>& object) {
    if (auto mesh = std::dynamic_pointer_cast<TriangleMesh>(object)) {
        return mesh;
    }
    if (auto instance = std::dynamic_pointer_cast<HittableInstance>(object)) {
        // The canonical source is the instance's flat mesh, never source_triangles.
        return std::dynamic_pointer_cast<TriangleMesh>(instance->mesh);
    }
    return {};
}

} // namespace

RuntimeSceneViewer::RuntimeSceneViewer(UIContext& context, SceneHistory& history)
    : context_(context), history_(history) {
    bindProjectHost(context_, *this);
}

RuntimeSceneViewer::~RuntimeSceneViewer() {
    unbindProjectHost(*this);
    releaseBackend();
}

void RuntimeSceneViewer::releaseBackend() noexcept {
    if (!backend_) {
        return;
    }
    if (context_.backend_ptr == backend_) {
        context_.backend_ptr = nullptr;
    }
    if (context_.renderer.m_backend == backend_) {
        context_.renderer.setBackend(nullptr);
    }
    if (g_viewport_backend.get() == backend_) {
        try {
            g_viewport_backend->waitForCompletion();
            g_viewport_backend->shutdown();
        } catch (...) {
            SCENE_LOG_WARN("[Player] Vulkan shutdown failed");
        }
        g_viewport_backend.reset();
    }
    backend_ = nullptr;
    material_buffer_ready_ = false;
}

rtapi::Result RuntimeSceneViewer::initializeBackend() {
    if (backend_) {
        return rtapi::Result::success();
    }
    if (g_viewport_backend) {
        return rtapi::Result::fail("a different runtime already owns the Vulkan viewport");
    }
    auto backend = std::make_unique<Backend::VulkanViewportBackend>();
    if (!backend->initialize()) {
        return rtapi::Result::fail("Player Vulkan viewport initialization failed");
    }
    backend->setViewportMode(Backend::ViewportMode::MaterialPreview);
    // Match Studio: capture opts into synchronous readback; ordinary display
    // uses the backend's frame ring and its initial blocking seed.
    backend->setInteractiveViewportSynchronousPresent(rtapi::viewportCaptureEnabled());
    backend_ = backend.get();
    g_viewport_backend = std::move(backend);
    context_.backend_ptr = backend_;
    context_.renderer.setBackend(backend_);
    g_hasVulkan = true;
    g_hasVulkanRT = backend_->getInfo().hasHardwareRT;
    // The Player initializes Vulkan lazily; Studio has already published its
    // device discovery before the adapter snapshots these capabilities.
    // Refresh after successful creation so texture/material services see the
    // real device instead of the startup all-false capability snapshot.
    if (!getSharedSceneTextureManager()->initialize(
            captureRuntimeRenderCapabilities(), "PlayerSceneViewer")) {
        return rtapi::Result::fail("Player scene texture service initialization failed");
    }
    g_solid_viewport_active = true;
    g_material_preview_viewport_active = true;
    return rtapi::Result::success();
}

rtapi::Result RuntimeSceneViewer::prepareScene() {
    flat_meshes_ = 0;
    triangles_ = 0;
    auto& materials = MaterialManager::getInstance();
    default_material_used_ = materials.getAllMaterials().empty();
    if (default_material_used_) {
        const auto id = materials.addMaterial("Player default",
            std::make_shared<PrincipledBSDF>(Vec3(0.8f, 0.8f, 0.8f)));
        if (id != 0) {
            return rtapi::Result::fail("failed to create Player default material at slot zero");
        }
        SCENE_LOG_WARN("[Player] Project has no material table; using a default material for ID 0");
    }
    const auto& available_materials = materials.getAllMaterials();
    if (available_materials.empty()) {
        return rtapi::Result::fail("Player has no valid material table");
    }
    for (const auto& object : context_.scene.world.objects) {
        if (!object) {
            continue;
        }
        // Volumes are Hittables in world.objects (CPU ray intersection) but draw
        // through the volume table, not the raster mesh build.
        if (std::dynamic_pointer_cast<VDBVolume>(object) ||
            std::dynamic_pointer_cast<GasVolume>(object)) {
            continue;
        }
        const auto mesh = flatMesh(object);
        if (!mesh) {
            return rtapi::Result::fail(
                "unsupported_in_player_scene: geometry must use flat TriangleMesh sources");
        }
        ++flat_meshes_;
        triangles_ += mesh->num_triangles();
        const auto* ids = mesh->geometry ? mesh->geometry->get_material_ids() : nullptr;
        if (!ids && !available_materials[0]) {
            return rtapi::Result::fail("flat mesh has no valid default material");
        }
        for (size_t vertex = 0; ids && vertex < mesh->num_vertices(); ++vertex) {
            if (ids[vertex] >= available_materials.size() || !available_materials[ids[vertex]]) {
                return rtapi::Result::fail("flat mesh references missing material ID " +
                                           std::to_string(ids[vertex]));
            }
        }
    }
    if (flat_meshes_ == 0 || triangles_ == 0) {
        return rtapi::Result::fail("project has no flat mesh geometry to display");
    }
    if (!context_.scene.camera) {
        return rtapi::Result::fail("project has no active camera");
    }
    if (context_.renderer.getHairSystem().getGroomCount() != 0) {
        return rtapi::Result::fail("unsupported_in_player_scene: hair runtime");
    }

    // The reference grid is an editor overlay that a project carries in its
    // render settings (grid_opacity, default 0.4). A game runtime never draws it;
    // opacity 0 is the backend's own "hidden" value.
    context_.render_settings.grid_opacity = 0.0f;
    // Same two-way material service Studio uses: Renderer uploads to its own
    // backend and to a separate viewport backend only when one exists. The
    // Player has a single device, so this is one upload.
    context_.renderer.updateBackendMaterials(context_.scene, backend_);
    auto* vulkan = dynamic_cast<Backend::VulkanBackendAdapter*>(backend_);
    auto* device = vulkan ? vulkan->getVulkanDevice() : nullptr;
    material_buffer_ready_ = device && device->m_materialBuffer.buffer != VK_NULL_HANDLE &&
                             device->m_materialBuffer.size > 0;
    if (!material_buffer_ready_) {
        return rtapi::Result::fail("Vulkan material buffer upload did not produce a valid buffer");
    }
    buildRasterScene(*backend_, context_.scene, context_.renderer);
    // Volume table after the raster build, same order as Studio's rebuild path
    // (geometry, then syncVDBVolumesToGPU + updateBackendGasVolumes).
    publishVolumes(context_.scene, context_.renderer, nullptr, backend_);
    context_.renderer.updateBackendGasVolumes(context_.scene);
    g_gas_volumes_dirty = false;
    backend_->setLights(context_.scene.lights);
    context_.renderer.world.initializeLUT();
    auto world_data = context_.renderer.world.getGPUData();
    syncVulkanWorld(dynamic_cast<Backend::VulkanBackendAdapter*>(backend_),
                    context_.renderer.world, world_data);
    context_.scene.camera->update_camera_vectors();
    backend_->syncCamera(*context_.scene.camera);
    backend_->resetAccumulation();
    Backend::RasterFrameTelemetry previous_frame;
    consumed_baseline_ = backend_->getInteractiveViewportFrameTelemetry(previous_frame)
        ? previous_frame.frames_consumed : 0;
    g_geometry_dirty = false;
    g_materials_dirty = false;
    g_texture_pool_dirty = false;
    g_camera_dirty = false;
    g_lights_dirty = false;
    g_world_dirty = false;
    g_viewport_world_dirty = false;
    return rtapi::Result::success();
}

rtapi::Result RuntimeSceneViewer::openProject(const std::string& path) {
    if (g_vulkan_device_lost) {
        last_open_error_ = "Vulkan device lost; restart Player before opening another project";
        return rtapi::Result::fail(last_open_error_);
    }
    const auto validation = validateProjectSlice(path);
    if (!validation.ok) {
        last_open_error_ = validation.error;
        return validation;
    }
    try {
        const auto initialization = initializeBackend();
        if (!initialization.ok) {
            last_open_error_ = initialization.error;
            releaseBackend();
            return initialization;
        }
        const bool capture_enabled = rtapi::viewportCaptureEnabled();
        rtapi::setViewportCapture(false);
        if (capture_enabled) {
            rtapi::setViewportCapture(true);
        }
        scene_ready_ = false;
        material_buffer_ready_ = false;
        frame_available_ = false;
        frame_scene_revision_ = 0;
        project_path_.clear();
        context_.active_model_path.clear();
        width_ = 0;
        height_ = 0;
        frames_rendered_ = 0;
        pixel_signature_ = 0;
        simulation_stopped_ = false;
        simulation_error_.clear();
        raster_refits_ = 0;
        raster_rebuilds_ = 0;
        particle_draws_ = 0;
        particle_uploads_ = 0;
        error_.clear();
        history_.clear();
        context_.selection.clearSelection();
        // Same canonical parser/flat data loader used by Studio and rt.project.open.
        g_dense_mesh_as_hittable = true;
        if (!ProjectManager::getInstance().openProject(path, context_.scene,
                context_.render_settings, context_.renderer, backend_)) {
            error_ = "project open failed: " + path;
        } else {
            const auto preparation = prepareScene();
            if (!preparation.ok) {
                error_ = preparation.error;
            }
        }
        if (!error_.empty()) {
            last_open_error_ = error_;
            return rtapi::Result::fail(error_);
        }
        project_path_ = path;
        context_.active_model_path = path;
        context_.start_render = false;
        ++scene_revision_;
        scene_ready_ = true;
        refresh_requested_ = true;
        last_open_error_.clear();
        rtapi::notifySceneLoaded();
        return rtapi::Result::success();
    } catch (const std::exception& error) {
        error_ = error.what();
        last_open_error_ = error_;
        scene_ready_ = false;
        return rtapi::Result::fail("Player project initialization failed: " + error_);
    }
}

bool RuntimeSceneViewer::needsFrame() const {
    return scene_ready_ && backend_ &&
           (refresh_requested_ || backend_->needsViewportRender() || g_camera_dirty ||
            (rtapi::viewportCaptureEnabled() && !rtapi::viewportStatus().frame_available));
}

void RuntimeSceneViewer::requestFrame() {
    refresh_requested_ = true;
    frame_available_ = false;
    frame_scene_revision_ = 0;
    Backend::RasterFrameTelemetry previous_frame;
    if (backend_ && backend_->getInteractiveViewportFrameTelemetry(previous_frame)) {
        consumed_baseline_ = previous_frame.frames_consumed;
    }
}

void RuntimeSceneViewer::advanceRuntime(float dt_seconds) {
    if (!scene_ready_ || !backend_ || g_vulkan_device_lost || simulation_stopped_) {
        return;
    }
    auto& scene = context_.scene;
    // The raster viewport is the only render path here and it supports the
    // per-node refit, so physics write-back takes the cheap path (Studio sets the
    // same flag from its frame tail).
    scene.setDeformRefitActive(true);

    // A game runs the simulation as a free-running clock: no timeline cache, no
    // scrubbing. The raster viewport never samples simulation device buffers, so
    // there is no RT backend to fence (Studio passes null for it too).
    SimulationClockInput input;
    input.dt_seconds = dt_seconds;
    // `playing` means TIMELINE playback. updateSimulationTimeline free-runs only
    // for live_mode && !playing; playing=true takes the timeline-bake path, and
    // with a playhead parked at 0 that path never steps (first live test: the
    // clock ticked 184 times while a dropped box stayed at y=3).
    input.playing = false;
    input.live_mode = true;
    const SimulationClockResult clock = simulationClock().tick(scene, input);
    if (clock.pause_requested) {
        simulation_stopped_ = true;
        simulation_error_ = clock.step_error;
        SCENE_LOG_WARN("[Player] Simulation stopped: " + simulation_error_);
    }

    bool changed = clock.render_updated;
    // Same write-back servicing as Studio's frame tail, raster branch only.
    if (g_geometry_deform_pending) {
        g_geometry_deform_pending = false;
        const auto nodes = scene.takePendingDeformNodes();
        if (!g_viewport_raster_rebuild_pending &&
            refitRasterDeformedNodes(*backend_, scene, nodes)) {
            ++raster_refits_;
        } else {
            g_viewport_raster_rebuild_pending = true;
        }
        changed = true;
    }
    if (g_viewport_raster_rebuild_pending) {
        buildRasterScene(*backend_, scene, context_.renderer);
        g_viewport_raster_rebuild_pending = false;
        g_gpu_refit_pending = false;  // a full build carries the current transforms
        ++raster_rebuilds_;
        changed = true;
    } else if (g_gpu_refit_pending) {
        backend_->syncRasterInstanceTransforms(scene.world.objects);
        g_gpu_refit_pending = false;
        changed = true;
    }
    // Particles: re-upload every iteration while any system exists. Billboards
    // face the camera, so even a frozen system needs a new list when it moves.
    // Same builder and upload Studio's SceneUI calls.
    if (!scene.particle_systems.empty()) {
        if (auto* vulkan = dynamic_cast<Backend::VulkanBackendAdapter*>(backend_)) {
            particle_draws_ = uploadParticleBillboards(*vulkan, scene, false);
            ++particle_uploads_;
        }
    }

    // Scatter edited at runtime: same raster refresh Studio's frame tail runs;
    // a refused refresh falls back to the full raster build next iteration.
    if (g_viewport_raster_scatter_update_pending) {
        g_viewport_raster_scatter_update_pending = false;
        auto* vulkan = dynamic_cast<Backend::VulkanBackendAdapter*>(backend_);
        if (vulkan && vulkan->refreshRasterScatterInstances()) {
            changed = true;
        } else {
            g_viewport_raster_rebuild_pending = true;
        }
    }

    // Volume edits (VDB transform/shader, gas settings): Studio services the same
    // flag with syncVDBVolumesToGPU + updateBackendGasVolumes.
    if (g_gas_volumes_dirty) {
        publishVolumes(scene, context_.renderer, nullptr, backend_);
        context_.renderer.updateBackendGasVolumes(scene);
        g_gas_volumes_dirty = false;
        changed = true;
    }

    // Bulk material / texture-pool changes (per-slot edits already upload through
    // Renderer::updateBackendMaterial). Same call as Studio's dirty servicing.
    if (g_materials_dirty || g_texture_pool_dirty) {
        context_.renderer.updateBackendMaterials(scene, backend_);
        g_materials_dirty = false;
        g_texture_pool_dirty = false;
        changed = true;
    }
    if (g_lights_dirty) {
        backend_->setLights(scene.lights);
        g_lights_dirty = false;
        changed = true;
    }
    // The Player has no CPU picking/render BVH to keep current.
    g_cpu_bvh_refit_pending = false;
    if (changed) {
        backend_->resetAccumulation();
        refresh_requested_ = true;
    }
}

bool RuntimeSceneViewer::render(SDL_Surface* surface, SDL_Texture* texture) {
    if (!scene_ready_ || !backend_ || !surface || !texture) {
        return false;
    }
    try {
        if (g_vulkan_device_lost) {
            error_ = "Vulkan device lost; restart Player to recreate the scene viewer";
            scene_ready_ = false;
            frame_available_ = false;
            return false;
        }
        if (width_ != surface->w || height_ != surface->h) {
            width_ = surface->w;
            height_ = surface->h;
            Backend::RenderParams params{};
            params.imageWidth = width_;
            params.imageHeight = height_;
            params.samplesPerPixel = 1;
            params.minSamples = 1;
            params.maxBounces = 4;
            params.adaptiveThreshold = 0.01f;
            backend_->setRenderParams(params);
            if (context_.scene.camera) {
                const float aspect = static_cast<float>(width_) / height_;
                context_.scene.camera->aspect = aspect;
                context_.scene.camera->aspect_ratio = aspect;
                context_.scene.camera->update_camera_vectors();
                g_camera_dirty = true;
            }
            backend_->resetAccumulation();
            framebuffer_.assign(static_cast<size_t>(width_) * height_, 0);
        }
        if (g_camera_dirty && context_.scene.camera) {
            backend_->syncCamera(*context_.scene.camera);
            backend_->resetAccumulation();
            g_camera_dirty = false;
        }
        backend_->renderProgressive(surface, nullptr, nullptr, width_, height_,
                                     &framebuffer_, nullptr);
        if (g_vulkan_device_lost || SDL_UpdateTexture(texture, nullptr,
                surface->pixels, surface->pitch) != 0) {
            error_ = g_vulkan_device_lost ? "Vulkan device lost" : SDL_GetError();
            scene_ready_ = false;
            frame_available_ = false;
            return false;
        }
        ++frames_rendered_;
        refresh_requested_ = false;
        Backend::RasterFrameTelemetry telemetry;
        const bool measured = backend_->getInteractiveViewportFrameTelemetry(telemetry) &&
                              telemetry.available && telemetry.width == width_ &&
                              telemetry.height == height_;
        // A previous scene's telemetry is not evidence for this surface. A new
        // consumed slot or a completed dirty render establishes the new epoch.
        const bool settled = !backend_->needsViewportRender();
        frame_available_ = measured &&
            ((telemetry.frames_consumed > 0 &&
              (telemetry.frames_consumed != consumed_baseline_ || settled)) ||
             (!telemetry.async_present && telemetry.draw_calls > 0 && settled));
        if (frame_available_) {
            frame_scene_revision_ = scene_revision_;
        }
        // Cheap signature for camera/reload acceptance, not a second image render.
        const auto* bytes = static_cast<const unsigned char*>(surface->pixels);
        pixel_signature_ = 1469598103934665603ull;
        for (int y = 0; y < height_; y += (height_ / 16 + 1)) {
            for (int x = 0; x < width_; x += (width_ / 16 + 1)) {
                const auto* pixel = bytes + y * surface->pitch + x * 4;
                for (int channel = 0; channel < 3; ++channel) {
                    pixel_signature_ = (pixel_signature_ ^ pixel[channel]) * 1099511628211ull;
                }
            }
        }
        const bool capture_enabled = rtapi::viewportCaptureEnabled();
        if (capture_enabled && frame_available_) {
            rtapi::publishViewportFrame(surface->pixels, width_, height_, surface->pitch);
        }
        return true;
    } catch (const std::exception& error) {
        error_ = error.what();
        scene_ready_ = false;
        frame_available_ = false;
        return false;
    }
}

std::uint64_t RuntimeSceneViewer::scatterInstanceCount() {
    std::uint64_t count = 0;
    for (const auto& group : InstanceManager::getInstance().getGroups()) {
        count += group.instances.size();
    }
    return count;
}

std::uint64_t RuntimeSceneViewer::aliveParticles() const {
    std::uint64_t alive = 0;
    for (const auto& system : context_.scene.particle_systems) {
        if (system.runtime) {
            alive += system.runtime->aliveCount();
        }
    }
    return alive;
}

nlohmann::json RuntimeSceneViewer::simulationStatus() const {
    const SimulationClockStats& clock = simulationClock().stats();
    return {
        {"runtime_present", scene_ready_ && context_.scene.anySimulationRuntimeEnabled()},
        {"rigid_bodies", context_.scene.rigid_bodies.size()},
        {"particle_systems", context_.scene.particle_systems.size()},
        {"particles_alive", aliveParticles()},
        {"particle_draws", particle_draws_}, {"particle_uploads", particle_uploads_},
        {"stopped", simulation_stopped_}, {"error", simulation_error_},
        {"clock_ticks", clock.ticks}, {"driven_ticks", clock.driven_ticks},
        {"driven_seconds", clock.driven_seconds}, {"last_dt_seconds", clock.last_dt_seconds},
        {"live_mode", clock.last_live_mode}, {"render_updates", clock.render_updates},
        {"raster_refits", raster_refits_}, {"raster_rebuilds", raster_rebuilds_}
    };
}

nlohmann::json RuntimeSceneViewer::status() const {
    Backend::RasterFrameTelemetry telemetry;
    const bool measured = backend_ && backend_->getInteractiveViewportFrameTelemetry(telemetry) &&
                          telemetry.available;
    return {
        {"host_active", true}, {"host", "player"},
        {"stage", "static_scene_viewer"}, {"scene_loading_supported", true},
        {"rendering_supported", true}, {"animation_playback", false},
        {"scene_ready", scene_ready_}, {"frame_available", frame_available_},
        {"render_pending", needsFrame()},
        {"project_path", project_path_}, {"scene_revision", scene_revision_},
        {"frame_scene_revision", frame_scene_revision_},
        {"last_open_error", last_open_error_}, {"error", error_},
        {"flat_meshes", flat_meshes_}, {"triangles", triangles_},
        {"material_count", MaterialManager::getInstance().getAllMaterials().size()},
        {"default_material_used", default_material_used_},
        {"material_buffer_ready", material_buffer_ready_ && !g_vulkan_device_lost},
        {"width", width_}, {"height", height_}, {"render_calls", frames_rendered_},
        {"pixel_signature", std::to_string(pixel_signature_)}, {"telemetry_available", measured},
        {"async_present", measured && telemetry.async_present},
        {"frames_submitted", measured ? telemetry.frames_submitted : 0},
        {"frames_consumed", measured ? telemetry.frames_consumed : 0},
        {"draw_calls", measured ? telemetry.draw_calls : 0},
        {"visible_triangles", measured ? telemetry.visible_triangles : 0},
        {"device_lost", g_vulkan_device_lost},
        {"scatter_groups", InstanceManager::getInstance().getGroups().size()},
        {"vdb_volumes", context_.scene.vdb_volumes.size()},
        {"gas_volumes", context_.scene.gas_volumes.size()},
        {"scatter_instances", scatterInstanceCount()},
        {"simulation", simulationStatus()}
    };
}

} // namespace AppRuntime
