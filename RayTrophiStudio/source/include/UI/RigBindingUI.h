#pragma once
#include <string>
struct UIContext;
namespace RigUI {
void drawRigBinding(UIContext&, const std::string& character, const std::string& mesh,
                    bool defaultOpen = true);
}
