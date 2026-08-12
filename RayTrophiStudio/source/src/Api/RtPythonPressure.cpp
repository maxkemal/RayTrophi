#include "RtPythonPressure.h"

#include "Api/RtApi.h"

#include <pybind11/stl.h>
#include <stdexcept>

namespace py = pybind11;

namespace {
void require(const rtapi::Result& result) {
    if (!result.ok) throw std::runtime_error(result.error);
}

py::dict statsDict(const rtapi::StructuralImpulseInfo& info) {
    py::dict out;
    out["queued"] = info.queued;
    out["consumed"] = info.consumed;
    out["affected_groups"] = info.affected_groups;
    out["fractured_groups"] = info.fractured_groups;
    out["last_peak_pressure_kpa"] = info.last_peak_pressure_kpa;
    out["last_max_impulse"] = info.last_max_impulse;
    out["last_projected_area_m2"] = info.last_projected_area_m2;
    return out;
}
} // namespace

namespace rtpy {
void registerPressureBindings(py::module_& gas) {
    gas.def("pressure_pulse", [](const std::string& domain,
                                  const py::tuple& center, float radius,
                                  float peak_pressure_kpa,
                                  float duration_seconds, float coupling) {
        if (center.size() != 3)
            throw std::runtime_error("center must contain three values");
        uint64_t sequence = 0;
        require(rtapi::emitGasPressurePulse(
            domain,
            Vec3(py::cast<float>(center[0]), py::cast<float>(center[1]),
                 py::cast<float>(center[2])),
            radius, peak_pressure_kpa, duration_seconds, coupling, sequence));
        return sequence;
    }, py::arg("domain"), py::arg("center"), py::arg("radius"),
       py::arg("peak_pressure_kpa"), py::arg("duration_seconds") = 0.02f,
       py::arg("coupling") = 1.0f);

    gas.def("structural_impulse_stats", []() {
        rtapi::StructuralImpulseInfo info;
        require(rtapi::getStructuralImpulseInfo(info));
        return statsDict(info);
    });
}
} // namespace rtpy
