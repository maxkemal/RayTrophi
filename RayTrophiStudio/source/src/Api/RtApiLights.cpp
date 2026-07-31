/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          Api/RtApiLights.cpp
* Author:        Kemal Demirtas
* Date:          July 2026
* License:       MIT
* =========================================================================
*
* Light facade. Lights are addressed by index into scene.lights; the index is
* stable until a light is added or removed.
*
* Geometric edits (position, direction, radius, cone) reuse LightState +
* TransformLightCommand, which is the same path the viewport gizmo takes.
* Appearance edits (color, intensity, visibility, name) are not covered by
* LightState, so this file owns a small LightAppearanceCommand that mirrors
* TransformLightCommand's backend resync exactly.
*
* Moved out of RtApi.cpp (past its size budget) together with the pre-existing
* list/add/delete/setPosition functions.
*/

#include "RtApiInternal.h"

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "Light.h"
#include "PointLight.h"
#include "DirectionalLight.h"
#include "SpotLight.h"
#include "AreaLight.h"
#include "ProjectManager.h"

namespace rtapi {
namespace {

const char* lightTypeName(LightType t) {
    switch (t) {
        case LightType::Point:       return "point";
        case LightType::Directional: return "directional";
        case LightType::Spot:        return "spot";
        case LightType::Area:        return "area";
        default:                     return "unknown";
    }
}

// Color / intensity / visibility / name are plain Light members that LightState
// does not capture. Same backend resync as TransformLightCommand so a scripted
// appearance edit lands exactly like a gizmo edit.
struct LightAppearance {
    Vec3 color;
    float intensity = 1.0f;
    bool visible = true;
    std::string name;

    static LightAppearance capture(const Light& light) {
        LightAppearance a;
        a.color = light.color;
        a.intensity = light.intensity;
        a.visible = light.visible;
        a.name = light.nodeName;
        return a;
    }
    void apply(Light& light) const {
        light.color = color;
        light.intensity = intensity;
        light.visible = visible;
        light.nodeName = name;
    }
};

class LightAppearanceCommand final : public SceneCommand {
public:
    LightAppearanceCommand(std::shared_ptr<Light> light, LightAppearance before,
                           LightAppearance after, std::string description)
        : light_(std::move(light)), before_(std::move(before)),
          after_(std::move(after)), description_(std::move(description)) {}

    void execute(UIContext& ctx) override { apply(ctx, after_); }
    void undo(UIContext& ctx) override { apply(ctx, before_); }
    Type getType() const override { return Type::Generic; }
    std::string getDescription() const override { return description_; }

private:
    void apply(UIContext& ctx, const LightAppearance& state) {
        if (!light_) return;
        state.apply(*light_);
        if (ctx.backend_ptr) {
            ctx.backend_ptr->setLights(ctx.scene.lights);
            ctx.backend_ptr->resetAccumulation();
        }
        ctx.renderer.resetCPUAccumulation();
        ctx.start_render = true;
    }

    std::shared_ptr<Light> light_;
    LightAppearance before_;
    LightAppearance after_;
    std::string description_;
};

// Shared guard for every mutating entry point below.
Result acquireLight(int index, std::shared_ptr<Light>& out) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (!g_history) return Result::fail("rtapi has no SceneHistory bound");
    auto& lights = g_ctx->scene.lights;
    if (index < 0 || index >= static_cast<int>(lights.size()) || !lights[index])
        return Result::fail("light index out of range: " + std::to_string(index));
    out = lights[index];
    return Result::success();
}

bool finiteVec(const Vec3& v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

void fillInfo(const Light& light, int index, LightInfo& info) {
    info.index = index;
    info.name = light.nodeName;
    info.type = lightTypeName(light.type());
    info.position = light.position;
    info.color = light.color;
    info.intensity = light.intensity;
    info.visible = light.visible;
    info.direction = light.direction;

    switch (light.type()) {
        case LightType::Point: {
            const auto& l = static_cast<const PointLight&>(light);
            info.radius = l.getRadius();
            break;
        }
        case LightType::Directional: {
            const auto& l = static_cast<const DirectionalLight&>(light);
            info.radius = l.getDiskRadius();
            break;
        }
        case LightType::Spot: {
            const auto& l = static_cast<const SpotLight&>(light);
            info.radius = l.getRadius();
            info.spot_angle = l.getAngleDegrees();
            info.spot_falloff = l.getFalloff();
            break;
        }
        case LightType::Area: {
            const auto& l = static_cast<const AreaLight&>(light);
            info.width = l.getWidth();
            info.height = l.getHeight();
            break;
        }
        default: break;
    }
}

// Applies a LightState edit through the gizmo's own command.
Result commitLightState(std::shared_ptr<Light>& light, const LightState& before,
                        const LightState& after) {
    auto cmd = std::make_unique<TransformLightCommand>(light, before, after);
    cmd->execute(*g_ctx);   // applies new state + backend light sync
    g_history->record(std::move(cmd));
    ProjectManager::getInstance().markModified();
    return Result::success();
}

Result commitAppearance(std::shared_ptr<Light>& light, const LightAppearance& before,
                        const LightAppearance& after, const std::string& description) {
    auto cmd = std::make_unique<LightAppearanceCommand>(light, before, after, description);
    cmd->execute(*g_ctx);
    g_history->record(std::move(cmd));
    ProjectManager::getInstance().markModified();
    return Result::success();
}

} // namespace

std::vector<LightInfo> listLights() {
    std::vector<LightInfo> out;
    if (!g_ctx) return out;
    int i = 0;
    for (auto& l : g_ctx->scene.lights) {
        if (l) {
            LightInfo info;
            fillInfo(*l, i, info);
            out.push_back(std::move(info));
        }
        ++i;
    }
    return out;
}

Result getLight(int index, LightInfo& out) {
    if (!g_ctx) return notBound();
    auto& lights = g_ctx->scene.lights;
    if (index < 0 || index >= static_cast<int>(lights.size()) || !lights[index])
        return Result::fail("light index out of range: " + std::to_string(index));
    fillInfo(*lights[index], index, out);
    return Result::success();
}

Result addLight(const std::string& type, const Vec3& position, std::string& out_name) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (!g_history) return Result::fail("rtapi has no SceneHistory bound");

    // Same defaults as the UI "Add > Light" menu.
    std::shared_ptr<Light> light;
    std::string prefix;
    if (type == "point") {
        light = std::make_shared<PointLight>(position, Vec3(10, 10, 10), 0.1f);
        prefix = "Point_";
    } else if (type == "directional") {
        auto l = std::make_shared<DirectionalLight>(Vec3(-1, -1, -0.5), Vec3(5, 5, 5), 0.1f);
        l->position = position;
        light = l;
        prefix = "Directional_";
    } else if (type == "spot") {
        light = std::make_shared<SpotLight>(position, Vec3(0, -1, 0), Vec3(10, 10, 10), 45.0f, 60.0f);
        prefix = "Spot_";
    } else if (type == "area") {
        light = std::make_shared<AreaLight>(position, Vec3(1, 0, 0), Vec3(0, 0, 1), 2.0f, 2.0f, Vec3(10, 10, 10));
        prefix = "Area_";
    } else {
        return Result::fail("unknown light type (point|directional|spot|area): " + type);
    }
    light->nodeName = prefix + std::to_string(g_ctx->scene.lights.size() + 1);

    auto cmd = std::make_unique<AddLightCommand>(light);
    cmd->execute(*g_ctx);   // idempotent push + backend light sync
    g_history->record(std::move(cmd));
    ProjectManager::getInstance().markModified();
    out_name = light->nodeName;
    return Result::success();
}

Result deleteLight(int index) {
    std::shared_ptr<Light> light;
    if (Result r = acquireLight(index, light); !r) return r;

    auto cmd = std::make_unique<DeleteLightCommand>(light);
    cmd->execute(*g_ctx);
    g_history->record(std::move(cmd));
    ProjectManager::getInstance().markModified();
    return Result::success();
}

Result setLightPosition(int index, const Vec3& position) {
    std::shared_ptr<Light> light;
    if (Result r = acquireLight(index, light); !r) return r;
    if (!finiteVec(position)) return Result::fail("position components must be finite");

    const LightState before = LightState::capture(*light);
    LightState after = before;
    after.position = position;
    return commitLightState(light, before, after);
}

Result setLightDirection(int index, const Vec3& direction) {
    std::shared_ptr<Light> light;
    if (Result r = acquireLight(index, light); !r) return r;
    if (!finiteVec(direction)) return Result::fail("direction components must be finite");

    const LightType type = light->type();
    if (type != LightType::Directional && type != LightType::Spot)
        return Result::fail("direction only applies to directional and spot lights");
    // Vec3::normalize() ignores sub-millimetre vectors, so reject them here rather
    // than silently leaving the old direction in place.
    const float len_sq = direction.x * direction.x + direction.y * direction.y +
                         direction.z * direction.z;
    if (len_sq <= 1e-6f) return Result::fail("direction must be a non-degenerate vector");

    const LightState before = LightState::capture(*light);
    LightState after = before;
    const float inv_len = 1.0f / std::sqrt(len_sq);
    after.direction = Vec3(direction.x * inv_len, direction.y * inv_len, direction.z * inv_len);
    return commitLightState(light, before, after);
}

Result setLightParam(int index, const std::string& param, float value) {
    std::shared_ptr<Light> light;
    if (Result r = acquireLight(index, light); !r) return r;
    if (!std::isfinite(value)) return Result::fail(param + " must be finite");

    const LightType type = light->type();
    const LightState before = LightState::capture(*light);
    LightState after = before;

    if (param == "radius") {
        if (value < 0.0f) return Result::fail("radius must be non-negative");
        after.radius = value;
    } else if (param == "spot_angle") {
        if (type != LightType::Spot) return Result::fail("spot_angle only applies to spot lights");
        if (value <= 0.0f || value >= 180.0f)
            return Result::fail("spot_angle must be in the range (0, 180)");
        after.angle = value;
    } else if (param == "spot_falloff") {
        if (type != LightType::Spot) return Result::fail("spot_falloff only applies to spot lights");
        if (value < 0.0f) return Result::fail("spot_falloff must be non-negative");
        after.falloff = value;
    } else if (param == "width" || param == "height") {
        if (type != LightType::Area)
            return Result::fail(param + " only applies to area lights");
        if (value <= 0.0f) return Result::fail(param + " must be positive");
        if (param == "width") after.width = value; else after.height = value;
    } else if (param == "intensity") {
        // Appearance, not geometry — routed to the other command.
        return setLightIntensity(index, value);
    } else {
        return Result::fail("unknown light parameter: " + param +
                            " (radius|spot_angle|spot_falloff|width|height|intensity)");
    }
    return commitLightState(light, before, after);
}

Result setLightColor(int index, const Vec3& color) {
    std::shared_ptr<Light> light;
    if (Result r = acquireLight(index, light); !r) return r;
    if (!finiteVec(color) || color.x < 0.0f || color.y < 0.0f || color.z < 0.0f)
        return Result::fail("color components must be finite and non-negative");

    const LightAppearance before = LightAppearance::capture(*light);
    LightAppearance after = before;
    after.color = color;
    return commitAppearance(light, before, after, "Set light color");
}

Result setLightIntensity(int index, float intensity) {
    std::shared_ptr<Light> light;
    if (Result r = acquireLight(index, light); !r) return r;
    if (!std::isfinite(intensity) || intensity < 0.0f)
        return Result::fail("intensity must be finite and non-negative");

    const LightAppearance before = LightAppearance::capture(*light);
    LightAppearance after = before;
    after.intensity = intensity;
    return commitAppearance(light, before, after, "Set light intensity");
}

Result setLightVisible(int index, bool visible) {
    std::shared_ptr<Light> light;
    if (Result r = acquireLight(index, light); !r) return r;

    const LightAppearance before = LightAppearance::capture(*light);
    LightAppearance after = before;
    after.visible = visible;
    return commitAppearance(light, before, after, "Set light visibility");
}

Result renameLight(int index, const std::string& name) {
    std::shared_ptr<Light> light;
    if (Result r = acquireLight(index, light); !r) return r;
    if (name.empty()) return Result::fail("light name must not be empty");

    const LightAppearance before = LightAppearance::capture(*light);
    LightAppearance after = before;
    after.name = name;
    return commitAppearance(light, before, after, "Rename light");
}

} // namespace rtapi
