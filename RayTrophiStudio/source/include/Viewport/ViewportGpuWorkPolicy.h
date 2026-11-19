#pragma once

namespace Viewport {
// Defer Solid viewport GPU work while its window cannot display it.
// Keep the caller's pending render request for focus/visibility restoration.
inline bool allowGpuWork(bool solid_viewport, bool focused, bool visible,
                         bool sequence_output, bool explicit_capture) {
    return !solid_viewport || sequence_output || explicit_capture ||
           (focused && visible);
}

// live_work (2026-11-18): a live simulation or a Studio play test is running.
// Before, only timeline playback counted, so a live sim or the game in the
// background stepped once per 500 ms. ~30 Hz keeps it moving at a fraction of
// the foreground cost (nothing is presented in the background).
inline int backgroundWaitMilliseconds(bool simulation_playing, bool scene_work,
                                      bool live_work = false) {
    if (simulation_playing) {
        return 1;
    }
    if (scene_work) {
        return 16;
    }
    return live_work ? 33 : 500;
}
}
