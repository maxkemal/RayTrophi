#pragma once
#include <string>
struct UIContext;
namespace RigUI {
void drawRigFitting(UIContext&,const std::string& character,const std::string& mesh);
void drawRigAlignmentWindow(UIContext&);
}
