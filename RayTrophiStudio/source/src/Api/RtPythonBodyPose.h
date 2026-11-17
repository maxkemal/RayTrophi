#pragma once
#include <pybind11/pybind11.h>
namespace rtpy {
void registerBodyPoseBindings(pybind11::module_ &module);
}
