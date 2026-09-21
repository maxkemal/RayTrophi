#pragma once
#include <string>
#include <cstdint>
#include "json.hpp"
struct UIContext;
namespace RigUI {
void drawRigSplineControls(UIContext&, const std::string& character, const std::string& control,
                           uint64_t revision, const nlohmann::json& row);
void drawRigSplineGuide(const nlohmann::json& row, const float* view, const float* projection,
                        const std::string& handle);
void drawRigIKTimeline(UIContext&, const std::string& character, const std::string& control,
                       uint64_t revision, bool contactsSupported = true);
void drawRigIKControls(UIContext&, const std::string& character);
bool drawRigIKGizmo(UIContext&, int shadingMode, bool available, bool& hit);
}
