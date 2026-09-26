#pragma once

#include <pybind11/pybind11.h>

namespace rtpy {
// Registers the rt.particle submodule on the top-level `rt` module.
void registerParticleBindings(pybind11::module_& module);
}
