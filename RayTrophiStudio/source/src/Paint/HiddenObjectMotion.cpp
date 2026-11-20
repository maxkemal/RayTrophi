#include "HiddenGameObjectsInternal.h"
#include "Api/RtApi.h"
#include "Backend/IBackend.h"
#include "Backend/IViewportBackend.h"
#include "Paint/BalloonPaintDemo.h"
#include "TriangleMesh.h"
#include "UIContext.h"
#include "scene_data.h"
#include "SimulationWorld.h"
#include <cmath>
#include <memory>
#include <stdexcept>

extern std::unique_ptr<Backend::IViewportBackend> g_viewport_backend;
extern std::unique_ptr<Backend::IBackend> g_backend;

namespace Paint::HiddenDetail {
void waitHiddenResetWork(SceneData& scene) {
    auto* context = rtapi::boundContext();
    require(context && &context->scene == &scene, "Missing room reset context");
    Backend::IBackend* viewport = g_viewport_backend
        ? static_cast<Backend::IBackend*>(g_viewport_backend.get()) : context->backend_ptr;
    scene.getSimulationWorld().compute().synchronize();
    if (viewport) viewport->waitForCompletion();
    auto* render = g_backend.get();
    if (render && render != viewport) render->waitForCompletion();
}

bool moveHiddenObject(SceneData& scene, const std::string& name,
                      const Matrix4x4& matrix, std::string& error) {
    try {
        auto* context = rtapi::boundContext();
        auto* mesh = scene.getFlatNodeMesh(name);
        require(context && &context->scene == &scene && mesh && mesh->transform,
                "Missing hidden object transform/context: " + name);
        const auto previous = mesh->transform->base;
        bool shape_changed = false;
        for (int c = 0; c < 3; ++c) {
            const Vec3 before(previous.m[0][c], previous.m[1][c], previous.m[2][c]);
            const Vec3 after(matrix.m[0][c], matrix.m[1][c], matrix.m[2][c]);
            shape_changed = shape_changed ||
                std::abs(before.length() - after.length()) > 1e-4f;
        }
        // Same lightweight instance update used by balloon motion. P_orig remains
        // authoritative; CPU P need not be rebaked for a translation / yaw reset.
        for (const auto& object : scene.world.objects) {
            auto flat = std::dynamic_pointer_cast<TriangleMesh>(object);
            if (flat && flat->nodeName == name && flat->transform) {
                flat->transform->setBase(matrix);
            }
        }
        Backend::IBackend* viewport = g_viewport_backend
            ? static_cast<Backend::IBackend*>(g_viewport_backend.get()) : context->backend_ptr;
        if (viewport) {
            viewport->updateObjectTransform(name, matrix);
            viewport->resetAccumulation();
        }
        auto* render = g_backend.get();
        if (render && render != viewport && render->isUsingTLAS()) {
            render->updateObjectTransform(name, matrix);
            render->resetAccumulation();
        }
        scene.invalidateSurfaceMeshCache(name);
        // A rigid move preserves the cooked SDF in the collider's local OBB frame.
        // The existing source-name resolver updates l2w/w2l on the next sim step.
        // Never remove/re-add the collider for a yaw/translation: that discards the
        // valid grid and makes four duplicate asynchronous cooks per object.
        if (shape_changed) {
            const auto& entry = state().entries.at(name);
            if (entry.collider != "none") {
                require(setBalloonPaintCollider(scene, name, entry.collider,
                                                entry.sdf_resolution, error), error);
            }
        }
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    }
}
} // namespace Paint::HiddenDetail
