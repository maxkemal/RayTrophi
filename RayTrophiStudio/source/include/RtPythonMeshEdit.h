#pragma once

#include <pybind11/pybind11.h>

namespace rtpy {

// Registers mesh operation queries on the already-created rt.mesh module.
void registerMeshEditBindings(pybind11::module_& mesh);

} // namespace rtpy
