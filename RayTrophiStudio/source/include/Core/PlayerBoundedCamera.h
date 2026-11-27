#pragma once

#include "Camera.h"
#include <SDL.h>

#include <algorithm>
#include <cmath>
#include <string>

namespace AppRuntime {

// Bounded gameplay camera controller for Player (PC mouse + Android touch).
// Keeps look-around and dolly within tight safe offsets from the authored
// scene camera so players never clip behind walls or lose body-tracking
// calibration.
struct BoundedCameraController {
    // At the room entrance (dolly = 0) keep look angles very tight so the open front edges
    // never expose the outside HDRI; widen smoothly as the camera moves deeper into the room.
    static constexpr float kHomeMaxYawDeg = 6.0f;
    static constexpr float kDeepMaxYawDeg = 38.0f;
    static constexpr float kHomeMaxPitchDeg = 4.0f;
    static constexpr float kDeepMaxPitchDeg = 18.0f;
    static constexpr float kMinDolly = 0.0f;
    static constexpr float kMaxDolly = 4.0f;
    static constexpr float kMaxLateralOffset = 1.8f;

    static float maxYawForDolly(float d) {
        const float u = (std::max)(0.0f, (std::min)(d / kMaxDolly, 1.0f));
        return kHomeMaxYawDeg + (kDeepMaxYawDeg - kHomeMaxYawDeg) * u;
    }

    static float maxPitchForDolly(float d) {
        const float u = (std::max)(0.0f, (std::min)(d / kMaxDolly, 1.0f));
        return kHomeMaxPitchDeg + (kDeepMaxPitchDeg - kHomeMaxPitchDeg) * u;
    }

    bool enabled = true;
    bool home_captured = false;
    std::string home_project;
    Vec3 home_pos{0.0f, 0.0f, 0.0f};
    Vec3 home_forward{0.0f, 0.0f, -1.0f};
    Vec3 home_right{1.0f, 0.0f, 0.0f};
    Vec3 home_up{0.0f, 1.0f, 0.0f};
    float home_focus = 5.0f;

    float yaw_deg = 0.0f;
    float pitch_deg = 0.0f;
    float dolly = 0.0f;
    float target_yaw_deg = 0.0f;
    float target_pitch_deg = 0.0f;
    float target_dolly = 0.0f;
    bool rmb_dragging = false;

    static float clampValue(float value, float min_val, float max_val) {
        return (std::max)(min_val, (std::min)(value, max_val));
    }

    static Vec3 rotateAroundAxis(const Vec3& v, const Vec3& axis, float angle_rad) {
        const float c = std::cos(angle_rad);
        const float s = std::sin(angle_rad);
        return v * c + Vec3::cross(axis, v) * s + axis * (Vec3::dot(axis, v) * (1.0f - c));
    }

    void setEnabled(bool v) {
        enabled = v;
        if (!enabled) {
            returnHome();
        }
    }

    void resetHome(const std::string& project_key, const Camera& cam) {
        home_project = project_key;
        home_pos = cam.lookfrom;
        const Vec3 view = cam.lookat - cam.lookfrom;
        const float focus_len = view.length();
        if (focus_len > 1e-4f) {
            home_forward = view / focus_len;
            home_focus = focus_len;
        } else {
            home_forward = Vec3(0.0f, 0.0f, -1.0f);
            home_focus = 5.0f;
        }

        const Vec3 up_ref = cam.vup.length_squared() > 1e-8f
            ? cam.vup.normalize()
            : Vec3(0.0f, 1.0f, 0.0f);
        Vec3 right = Vec3::cross(home_forward, up_ref);
        if (right.length_squared() < 1e-8f) {
            const Vec3 fallback_up = (home_forward.y < 0.0f)
                ? Vec3(0.0f, 0.0f, -1.0f)
                : Vec3(0.0f, 0.0f, 1.0f);
            right = Vec3::cross(home_forward, fallback_up);
        }
        home_right = right.normalize();
        home_up = Vec3::cross(home_right, home_forward).normalize();

        yaw_deg = 0.0f;
        pitch_deg = 0.0f;
        dolly = 0.0f;
        returnHome();
        home_captured = true;
    }

    void clearHome() {
        home_captured = false;
        home_project.clear();
        yaw_deg = 0.0f;
        pitch_deg = 0.0f;
        dolly = 0.0f;
        returnHome();
    }

    void returnHome() {
        target_yaw_deg = 0.0f;
        target_pitch_deg = 0.0f;
        target_dolly = 0.0f;
        rmb_dragging = false;
    }

    bool handleEvent(const SDL_Event& event, bool balloon_held) {
        if (event.type == SDL_WINDOWEVENT &&
            event.window.event == SDL_WINDOWEVENT_FOCUS_LOST) {
            rmb_dragging = false;
            return false;
        }
        if (!enabled || !home_captured) {
            return false;
        }
        if (event.type == SDL_MOUSEBUTTONDOWN && event.button.button == SDL_BUTTON_RIGHT) {
            rmb_dragging = true;
            return true;
        }
        if (event.type == SDL_MOUSEBUTTONUP && event.button.button == SDL_BUTTON_RIGHT) {
            const bool was_dragging = rmb_dragging;
            rmb_dragging = false;
            return was_dragging;
        }
        if (event.type == SDL_MOUSEMOTION && rmb_dragging) {
            const float dx = static_cast<float>(event.motion.xrel);
            const float dy = static_cast<float>(event.motion.yrel);
            const float max_yaw = maxYawForDolly(target_dolly);
            const float max_pitch = maxPitchForDolly(target_dolly);
            target_yaw_deg = clampValue(target_yaw_deg + dx * 0.15f, -max_yaw, max_yaw);
            target_pitch_deg = clampValue(target_pitch_deg - dy * 0.15f, -max_pitch, max_pitch);
            return true;
        }
        if (event.type == SDL_MOUSEWHEEL && !balloon_held) {
            const int direction = (event.wheel.direction == SDL_MOUSEWHEEL_FLIPPED) ? -1 : 1;
            const float step = static_cast<float>(event.wheel.y * direction);
            if (step != 0.0f) {
                target_dolly = clampValue(target_dolly + step * 0.35f, kMinDolly, kMaxDolly);
                const float max_yaw = maxYawForDolly(target_dolly);
                const float max_pitch = maxPitchForDolly(target_dolly);
                target_yaw_deg = clampValue(target_yaw_deg, -max_yaw, max_yaw);
                target_pitch_deg = clampValue(target_pitch_deg, -max_pitch, max_pitch);
                return true;
            }
            return false;
        }
        if (event.type == SDL_MULTIGESTURE && !balloon_held &&
            event.mgesture.numFingers == 2) {
            if (std::isfinite(event.mgesture.dDist) && event.mgesture.dDist != 0.0f) {
                target_dolly = clampValue(
                    target_dolly + event.mgesture.dDist * 4.0f, kMinDolly, kMaxDolly);
                const float max_yaw = maxYawForDolly(target_dolly);
                const float max_pitch = maxPitchForDolly(target_dolly);
                target_yaw_deg = clampValue(target_yaw_deg, -max_yaw, max_yaw);
                target_pitch_deg = clampValue(target_pitch_deg, -max_pitch, max_pitch);
                return true;
            }
            return false;
        }
        if (event.type == SDL_KEYDOWN && !event.key.repeat &&
            event.key.keysym.sym == SDLK_r) {
            returnHome();
            return true;
        }
        return false;
    }

    bool tick(float dt, Camera& cam, bool force_home) {
        if (!home_captured) {
            return false;
        }
        if (force_home || !enabled) {
            returnHome();
        }
        if (!std::isfinite(dt) || dt <= 0.0f) {
            return false;
        }

        const float clamped_dt = (std::min)(dt, 0.1f);
        const float alpha = 1.0f - std::exp(-clamped_dt * 12.0f);

        const float dy = target_yaw_deg - yaw_deg;
        const float dp = target_pitch_deg - pitch_deg;
        const float dd = target_dolly - dolly;

        constexpr float kEps = 1e-4f;
        const bool within_eps =
            std::fabs(dy) < kEps && std::fabs(dp) < kEps && std::fabs(dd) < kEps;
        if (within_eps) {
            if (yaw_deg == target_yaw_deg &&
                pitch_deg == target_pitch_deg &&
                dolly == target_dolly) {
                return false;
            }
            yaw_deg = target_yaw_deg;
            pitch_deg = target_pitch_deg;
            dolly = target_dolly;
        } else {
            yaw_deg += dy * alpha;
            pitch_deg += dp * alpha;
            dolly += dd * alpha;
        }

        constexpr float kDegToRad = 3.14159265358979323846f / 180.0f;
        const float yaw_rad = -yaw_deg * kDegToRad;
        const float pitch_rad = pitch_deg * kDegToRad;
        const Vec3 yaw_forward = rotateAroundAxis(home_forward, home_up, yaw_rad).normalize();
        const Vec3 yaw_right = rotateAroundAxis(home_right, home_up, yaw_rad).normalize();
        const Vec3 dir = rotateAroundAxis(yaw_forward, yaw_right, pitch_rad).normalize();

        // Move strictly inward toward the area the player is looking at, never backing outside
        // the room behind home_pos, and clamp lateral/vertical travel so side walls/floor stay safe.
        const Vec3 move_dir = Vec3(dir.x, dir.y * 0.35f, dir.z).normalize();
        const Vec3 delta = move_dir * dolly;
        const float fwd_proj = clampValue(Vec3::dot(delta, home_forward), kMinDolly, kMaxDolly);
        const float lat_proj = clampValue(
            Vec3::dot(delta, home_right), -kMaxLateralOffset, kMaxLateralOffset);
        Vec3 next_pos = home_pos + home_forward * fwd_proj + home_right * lat_proj;
        next_pos.y = clampValue(home_pos.y + delta.y, home_pos.y - 0.35f, home_pos.y + 0.45f);
        cam.lookfrom = next_pos;
        cam.lookat = cam.lookfrom + dir * home_focus;
        cam.update_camera_vectors();
        cam.markDirty();
        return true;
    }
};

inline BoundedCameraController& boundedCamera() {
    static BoundedCameraController inst;
    return inst;
}

} // namespace AppRuntime
