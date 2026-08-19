#pragma once

#include <pybind11/pybind11.h>

namespace rtpy {
// rt.agent — the discovery/measurement surface, in-process.
void registerAgentBindings(pybind11::module_& root);
}
