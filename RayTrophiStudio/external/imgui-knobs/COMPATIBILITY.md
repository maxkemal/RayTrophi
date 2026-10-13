# RayTrophi integration

Upstream: https://github.com/altschuler/imgui-knobs
Pinned commit: 6b4b5923129c1e8fa4cd30f8c59439643d44f22f
SHA256 cpp: 240b0f24b87e36df94dc70e33efd35dcf9d27abd242b00e1d6b20a00fabd06bb
SHA256 h: 0a0a7fa8fe77c7463a39d6a4143e894c155cf844c11aaab4d7691ca3d3cc2500
License: MIT, retained in LICENSE.
Host: Dear ImGui 1.92.8 WIP, IMGUI_VERSION_NUM=19275.

Unmodified upstream cpp/header. Static compatibility inspection:
- For host >=19197 uses style.FontScaleMain, present in our imgui.h.
- For host <19276 uses PathStroke(color, flags, thickness), matching our header.
- DragBehavior signature matches imgui_internal.h.
- Uses the existing ImGui context/theme; no second ImGui copy is bundled.

Both vcxproj and CMake compile the one vendor cpp. All audio operations remain
in the existing shared service. Application compilation and mouse/keyboard/DPI
acceptance are performed by the user after this source delivery.
