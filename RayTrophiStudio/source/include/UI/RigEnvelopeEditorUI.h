#pragma once

#include <string>

struct UIContext;

namespace RigUI {
void openRigEnvelopeEditor(const std::string &character);
void drawRigEnvelopeEditor(UIContext &ctx);
}
