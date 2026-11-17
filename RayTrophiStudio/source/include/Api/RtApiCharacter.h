#pragma once

#include "Api/RtApi.h"

namespace rtapi {
// Removes the complete character, including its flat mesh parts. Undoable.
Result deleteCharacter(const std::string &character);
}
