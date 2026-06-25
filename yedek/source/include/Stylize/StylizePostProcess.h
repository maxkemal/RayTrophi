#pragma once

#include "Stylize/StylizeModeState.h"
#include "Stylize/StylizeCore.h"
#include "Vec3.h"

#include <cstdint>

namespace Stylize {

struct StylizeAOVSample {
    Vec3 albedo = Vec3(0.0f);
    Vec3 normal = Vec3(0.0f);
    Vec3 world_position = Vec3(0.0f);
    Vec3 view_dir = Vec3(0.0f, 0.0f, -1.0f);
    Vec3 sun_dir = Vec3(0.32f, 0.82f, 0.46f);
    float screen_u = 0.0f;
    float screen_v = 0.0f;
    float sun_size_degrees = 0.545f;
    float sun_elevation_degrees = 35.0f;
    bool nishita_clouds_enabled = false;
    float nishita_cloud_coverage = 0.45f;
    float nishita_cloud_density = 0.65f;
    float nishita_cloud_scale = 1.0f;
    float nishita_cloud_offset_x = 0.0f;
    float nishita_cloud_offset_z = 0.0f;
    int nishita_cloud_seed = 0;
    float depth = 0.0f;
    float edge = 0.0f;
    float pixel_scale = 0.0f;   // world units per pixel at the hit point (0 = unknown)
    uint32_t material_id = 0xFFFFFFFFu;
    bool valid = false;
    bool hit = false;
};

Vec3 applyPostProcess(const Vec3& input_color,
                      int x,
                      int y,
                      int frame_index,
                      const StylizeModeState& state);

Vec3 applyPostProcess(const Vec3& input_color,
                      const StylizeAOVSample& aov,
                      int x,
                      int y,
                      int frame_index,
                      const StylizeModeState& state);

// Convert the project StyleProfile into the device-copyable POD the shared core
// (and the GPU kernel) consume. Used by the GPU stylize path to upload the
// active profile.
StylizeCore::StyleProfileCore makeCoreProfile(const StyleProfile& profile);

} // namespace Stylize
