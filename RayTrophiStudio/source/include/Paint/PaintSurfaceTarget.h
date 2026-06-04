#pragma once

#include <string>
#include <cstdint>

namespace Paint {

enum class SurfaceType : uint8_t {
    None = 0,
    Terrain,
    Mesh
};

struct PaintSurfaceTarget {
    SurfaceType type = SurfaceType::None;
    int object_id = -1;
    int material_slot = -1;
    uint16_t material_id = 0xFFFF;
    int uv_set = 0;
    int resolution = 1024;
    std::string display_name;
};

} // namespace Paint
