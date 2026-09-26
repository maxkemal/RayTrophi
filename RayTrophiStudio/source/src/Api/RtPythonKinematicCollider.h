#pragma once

#include <pybind11/pybind11.h>

namespace rtpy {

void registerKinematicColliderBindings(pybind11::module_& collider_module);

} // namespace rtpy
