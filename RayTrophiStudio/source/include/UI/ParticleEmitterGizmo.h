#pragma once

#include "Matrix4x4.h"
#include "imgui.h"

#include <functional>

namespace ParticleEmitterGizmo {

using ProjectWorldPoint = std::function<bool(const Vec3&, ImVec2&)>;

struct DirectionalEmitterGeometry {
    Vec3 forward = Vec3(0.0f, 1.0f, 0.0f);
    Vec3 right = Vec3(1.0f, 0.0f, 0.0f);
    Vec3 up = Vec3(0.0f, 0.0f, 1.0f);
    Vec3 collar_center = Vec3(0.0f);
    Vec3 mouth_center = Vec3(0.0f);
    float length = 0.45f;
    float collar_radius = 0.07f;
    float mouth_radius = 0.12f;
};

DirectionalEmitterGeometry makeDirectionalEmitterGeometry(
    const Vec3& position,
    const Vec3& direction,
    float speed,
    float spread);

bool intersectDirectionalEmitter(const Vec3& ray_origin,
                                 const Vec3& ray_direction,
                                 const DirectionalEmitterGeometry& geometry,
                                 float padding,
                                 float& out_distance);

class ScopedViewportClip {
public:
    ScopedViewportClip(ImDrawList* background,
                       ImDrawList* foreground,
                       const ImVec2& minimum,
                       const ImVec2& maximum);
    ~ScopedViewportClip();

    ScopedViewportClip(const ScopedViewportClip&) = delete;
    ScopedViewportClip& operator=(const ScopedViewportClip&) = delete;

private:
    ImDrawList* background_ = nullptr;
    ImDrawList* foreground_ = nullptr;
};

void drawDirectionalEmitter(ImDrawList* draw_list,
                            const ProjectWorldPoint& project,
                            const Vec3& position,
                            const Vec3& direction,
                            float speed,
                            float spread,
                            ImU32 color,
                            bool selected);

} // namespace ParticleEmitterGizmo
