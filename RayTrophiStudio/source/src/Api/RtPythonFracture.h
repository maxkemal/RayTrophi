#pragma once

#include <pybind11/pybind11.h>

namespace rtpy {
void registerFractureBindings(pybind11::module_& physics);
}
