// Keep the vendored implementation in one translation unit. ufbx supports C++
// compilation; this also lets the existing CXX-only CMake source glob find it.
#include "../../../external/ufbx/ufbx.c"
