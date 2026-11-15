#pragma once

#include "Core/RuntimeProjectHost.h"
#include "Viewport/RasterFrameTelemetry.h"
#include "Core/VulkanSceneSync.h"

#include <cstdint>
#include <memory>
#include <vector>

struct SDL_Surface;
struct SDL_Texture;
class SceneHistory;
namespace Backend {
class IViewportBackend;
}

namespace AppRuntime {

// First Player scene slice: static flat meshes, materials, camera, lights and
// environment through the same project parser and Vulkan raster implementation.
// Unsupported authoring-derived systems are rejected before scene mutation.
class RuntimeSceneViewer final : public ProjectHost {
public:
    RuntimeSceneViewer(UIContext& context, SceneHistory& history);
    ~RuntimeSceneViewer() override;

    rtapi::Result openProject(const std::string& path) override;
    nlohmann::json status() const override;
    nlohmann::json simulationStatus() const;
    std::uint64_t aliveParticles() const;
    static std::uint64_t scatterInstanceCount();
    bool needsFrame() const;
    void requestFrame();
    // Measurement mode: make the backend record and submit a real frame. A plain
    // requestFrame() on a converged raster frame returns without recording
    // (measured: 0 frames drawn, cpu_record 0 ms, while "continuous" was on).
    void forceRedraw();
    // Drive the shared simulation clock in live mode and apply its write-back
    // (deformed bodies, moved instances) to the raster scene. Called once per
    // Player loop iteration; a scene without simulation runtime costs nothing.
    void advanceRuntime(float dt_seconds);
    bool render(SDL_Surface* surface, SDL_Texture* texture);
    // Milliseconds of the last render() call: backend frame (GPU + readback
    // into the CPU surface) and SDL texture upload.
    double lastRenderMs() const { return last_render_ms_; }
    double lastUploadMs() const { return last_upload_ms_; }
    // Backend telemetry of the last drawn frame (CPU record, submit, slot wait).
    const Backend::RasterFrameTelemetry& lastTelemetry() const { return last_telemetry_; }
    // Frame-loop telemetry owned by the executable, reported under "loop".
    void setLoopStatus(nlohmann::json status) { loop_status_ = std::move(status); }
    bool hasScene() const {
        return scene_ready_;
    }

    bool hasFrame() const {
        return scene_ready_ && frame_available_;
    }

private:
    rtapi::Result initializeBackend();
    rtapi::Result prepareScene();
    void releaseBackend() noexcept;

    UIContext& context_;
    SceneHistory& history_;
    Backend::IViewportBackend* backend_ = nullptr;
    bool scene_ready_ = false;
    bool frame_available_ = false;
    bool refresh_requested_ = false;
    bool default_material_used_ = false;
    bool material_buffer_ready_ = false;
    bool simulation_stopped_ = false;
    std::string simulation_error_;
    std::uint64_t raster_refits_ = 0;
    std::uint64_t raster_rebuilds_ = 0;
    std::uint64_t particle_draws_ = 0;
    double last_render_ms_ = 0.0;
    // Clouds evolve on the game clock (Studio uses its timeline frame).
    float cloud_time_s_ = 0.0f;
    // Scene audio transport runs on the game clock from project open.
    double audio_time_s_ = 0.0;
    std::uint64_t audio_ticks_ = 0;
    std::uint64_t cloud_pushes_ = 0;
    std::uint64_t world_syncs_ = 0;
    CloudPushState cloud_push_{};
    double last_upload_ms_ = 0.0;
    Backend::RasterFrameTelemetry last_telemetry_{};
    nlohmann::json loop_status_ = nlohmann::json::object();
    std::uint64_t particle_uploads_ = 0;
    std::string project_path_;
    std::string error_;
    std::string last_open_error_;
    std::uint64_t scene_revision_ = 0;
    std::uint64_t frames_rendered_ = 0;
    std::uint64_t consumed_baseline_ = 0;
    std::uint64_t frame_scene_revision_ = 0;
    std::uint64_t flat_meshes_ = 0;
    std::uint64_t triangles_ = 0;
    std::uint64_t pixel_signature_ = 0;
    int width_ = 0;
    int height_ = 0;
    std::vector<std::uint32_t> framebuffer_;
};

} // namespace AppRuntime
