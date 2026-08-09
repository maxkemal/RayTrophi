#pragma once

#include <pybind11/pybind11.h>

namespace rtpy {
void registerPressureBindings(pybind11::module_& gas);
}
