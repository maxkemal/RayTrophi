#pragma once

#include "Core/RuntimeProjectHost.h"

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
    // Drive the shared simulation clock in live mode and apply its write-back
    // (deformed bodies, moved instances) to the raster scene. Called once per
    // Player loop iteration; a scene without simulation runtime costs nothing.
    void advanceRuntime(float dt_seconds);
    bool render(SDL_Surface* surface, SDL_Texture* texture);
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
