#pragma once

#include <cstdint>
#include <string>

struct UIContext;

namespace RigUI {

void drawRigFingerFKControls(UIContext& context, const std::string& character,
                             uint64_t revision);
bool drawRigFingerFKOverlay(UIContext& context, int shadingMode, bool available,
                            bool& hit);
void selectRigFingerFKSide(const std::string& side);

} // namespace RigUI
