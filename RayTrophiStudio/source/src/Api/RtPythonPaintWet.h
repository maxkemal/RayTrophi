#pragma once

#include <pybind11/pybind11.h>

namespace rtpy {
// Registers rt.paint.wet_* (GPU wet paint) on the rt.paint submodule.
void registerPaintWetBindings(pybind11::module_& paint);
}
