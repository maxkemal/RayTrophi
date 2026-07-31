/*
* =========================================================================
* Project:       RayTrophi Studio
* File:          Api/RtPyCommon.h  (internal)
* License:       MIT
* =========================================================================
*
* Helpers shared by the `rt` module's binding translation units. These were
* private to RtPython.cpp until the bindings had to be split across files;
* they live at global scope (not in a namespace) so the existing unqualified
* call sites in RtPython.cpp keep resolving unchanged.
*/
#pragma once

#include <stdexcept>
#include <string>

#include <pybind11/pybind11.h>

#include "Api/RtApi.h"
#include "Vec3.h"

// No exceptions cross the rtapi boundary: a failed Result becomes a Python
// RuntimeError here, at the edge.
inline void requireResult(const rtapi::Result& result) {
    if (!result.ok) throw std::runtime_error(result.error);
}

inline pybind11::tuple vec3ToPython(const Vec3& value) {
    return pybind11::make_tuple(value.x, value.y, value.z);
}

inline Vec3 vec3FromPython(const pybind11::handle& value) {
    pybind11::sequence values = pybind11::reinterpret_borrow<pybind11::sequence>(value);
    if (pybind11::len(values) != 3) throw pybind11::value_error("expected three components");
    return Vec3(pybind11::cast<float>(values[0]),
                pybind11::cast<float>(values[1]),
                pybind11::cast<float>(values[2]));
}

namespace rtpy {

// Defines rt.select, rt.material and rt.lights. Called once, from inside
// PYBIND11_EMBEDDED_MODULE(rt, ...).
void registerSceneBindings(pybind11::module_& module);

} // namespace rtpy
