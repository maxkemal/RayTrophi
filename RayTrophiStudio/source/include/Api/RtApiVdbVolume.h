#pragma once

#include "Vec3.h"

#include <string>

class VDBVolume;

namespace rtapi {
struct Result;

// VDB file import (docs/dev/PLAYER_GAME_PAINT_ROOM.md, volume acceptance). The
// body used to live in SceneUI::importVDBVolume behind a file dialog, so no test
// could put a VDB into a scene (CLAUDE.md rule 1). The panel now calls this.
struct VdbImportInfo {
    std::string name;
    Vec3 local_bounds_min;
    Vec3 local_bounds_max;
    Vec3 position;
    Vec3 scale;
    bool auto_scaled = false;     // the file was larger than 50 units and was scaled to 5
    bool gpu_uploaded = false;    // false = CPU-only (GPU upload failed), not an error
    bool fire_shader = false;     // temperature grid present -> fire preset, else smoke
    int scene_vdb_count = 0;
};

// orientation_preset: 0 = auto (EmberGen files get the axis conversion),
// 1 = keep the file's axes, 2 = Z-up file to Y-up (-90 deg about X).
// The volume is scaled down when it is larger than 50 units, centred on X/Z
// and sat on Y = 0, exactly as the panel import always did.
Result importVdbVolume(const std::string& path, int orientation_preset, VdbImportInfo& out);

// Shared by the import above and SceneUI's asset-library import.
bool shouldApplySpecialVdbOrientation(const std::string& source_hint);
void applyVdbImportOrientation(VDBVolume& vdb, int orientation_preset,
                               const std::string& source_hint);

} // namespace rtapi
