#pragma once

#include <pybind11/pybind11.h>

namespace rtpy {
void registerDomainWallBindings(pybind11::module_& fluid);
}
