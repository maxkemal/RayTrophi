/*
* =========================================================================
* Project:       RayTrophi Studio
* File:          Api/RtUiBindings.h  (internal)
* License:       MIT
* =========================================================================
*
* Internal seam between the embedded `rt` module (RtPython.cpp) and the rt.ui
* implementation (RtUi.cpp). Kept out of include/Api/RtUi.h so the public
* header stays pybind-free.
*/
#pragma once

#include <pybind11/pybind11.h>

namespace rtui {

// Defines the `rt.ui` submodule on `module`. Called once, from inside
// PYBIND11_EMBEDDED_MODULE(rt, ...).
void registerBindings(pybind11::module_& module);

} // namespace rtui
