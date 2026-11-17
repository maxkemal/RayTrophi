#pragma once
#include <pybind11/pybind11.h>
namespace rtpy {
void registerExternalBodyBindings(pybind11::module_ &module);
}
