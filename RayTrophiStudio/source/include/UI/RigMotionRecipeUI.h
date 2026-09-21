#pragma once
#include <cstdint>
#include <string>

struct UIContext;
namespace RigUI {
void drawRigMotionRecipes(UIContext &, const std::string &character, uint64_t revision);
}
