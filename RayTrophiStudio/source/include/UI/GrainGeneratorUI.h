#pragma once

#include <string>

struct UIContext;

namespace GrainGeneratorUI {
void requestOpen(const std::string& domain = {});
bool consumeOpenRequest();
void drawPanel(UIContext& context);
}
