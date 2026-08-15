#pragma once

#include <pybind11/pybind11.h>

namespace rtpy {
void registerTemplateBindings(pybind11::module_& root);
}
