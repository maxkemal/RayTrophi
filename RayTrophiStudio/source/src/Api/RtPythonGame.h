#pragma once
#include <pybind11/pybind11.h>
namespace rtpy {
void registerGameBindings(pybind11::module_ &module);
}
