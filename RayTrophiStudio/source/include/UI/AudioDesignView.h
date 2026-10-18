#pragma once

namespace AudioUI {
// Scoped by the bottom canvas; side-panel controls retain their normal size.
inline float designViewScale = 1.0f;
inline float designPanelScale = 1.0f;
inline float designViewPixels(float pixels) {
    return pixels * designViewScale;
}
} // namespace AudioUI
