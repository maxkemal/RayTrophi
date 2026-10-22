#pragma once

#include <pybind11/pybind11.h>

namespace rtpy {
void registerGrainGeneratorBindings(pybind11::module_& scene);
}
