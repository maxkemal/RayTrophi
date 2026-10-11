#pragma once

namespace Viewport {
// Defer Solid viewport GPU work while its window cannot display it.
// Keep the caller's pending render request for focus/visibility restoration.
inline bool allowGpuWork(bool solid_viewport, bool focused, bool visible,
                         bool sequence_output, bool explicit_capture) {
    return !solid_viewport || sequence_output || explicit_capture ||
           (focused && visible);
}

inline int backgroundWaitMilliseconds(bool simulation_playing, bool scene_work) {
    if (simulation_playing) {
        return 1;
    }
    return scene_work ? 16 : 500;
}
}
