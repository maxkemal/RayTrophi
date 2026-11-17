#pragma once

#include <string>

namespace GameDesignUI {

// Studio "Game Design" section (Paint panel): which objects are wet paint
// surfaces and their paint model, how particle contacts become paint, and the
// balloon demo. It holds no state of its own: every value is read from and
// written through the same rtapi calls that paint.wet_* / paint.wet_demo use,
// and the project saves that live state under "game_design" (Paint/GameDesign.h).
// selected_object: the mesh the Paint panel targets, empty when none.
void drawSection(const std::string& selected_object);

} // namespace GameDesignUI
