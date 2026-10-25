#pragma once

#include <string>

struct UIContext;

namespace GrainGeneratorUI {
void requestOpen();
bool consumeOpenRequest();
void drawPanel(UIContext& context);
}
