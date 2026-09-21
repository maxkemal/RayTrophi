#include "RtPythonReflection.h"
#include "RtReflectionBindings.h"
#include <stdexcept>
namespace rtpy {
void registerReflectionBindings(pybind11::module_& module) {
    namespace py = pybind11;
    module.def("reflections", [](){ return reflectionDictionary<py::dict>(); });
    module.def("set_reflections",
        [](bool enabled, py::object samples, py::object roughnessGate,
           py::object weightGate, py::object distance, py::object followPreset) {
        // Python'da bool bir int'tir; sayisal kontroller icin acikca reddedilir.
        if (PyBool_Check(samples.ptr()) || !PyLong_Check(samples.ptr()))
            throw py::value_error("samples must be an integer");
        for (const auto* value : {&roughnessGate, &weightGate, &distance})
            if (PyBool_Check(value->ptr()) ||
                (!PyFloat_Check(value->ptr()) && !PyLong_Check(value->ptr())))
                throw py::value_error("roughness_gate, weight_gate and max_distance must be numeric");
        RayFusion::ReflectionSettings s;
        s.enabled = enabled;
        // ★★ `follow_preset` verilmediyse (None) ve sayisal kontroller
        //   varsayilanindan FARKLI ise manuel kontrole gecilir -- IPC yolundaki
        //   "sayiyi adlandirmak manuel kontrol demektir" kuraliyla ayni.
        if (followPreset.is_none()) {
            const RayFusion::ReflectionSettings defaults;
            s.followQualityPreset =
                samples.cast<int64_t>() == int64_t(defaults.samples) &&
                roughnessGate.cast<float>() == defaults.roughnessGate &&
                weightGate.cast<float>() == defaults.weightGate;
        } else {
            s.followQualityPreset = followPreset.cast<bool>();
        }
        s.samples = uint32_t(samples.cast<int64_t>());
        s.roughnessGate = roughnessGate.cast<float>();
        s.weightGate = weightGate.cast<float>();
        s.maxDistance = distance.cast<float>();
        std::string error;
        if (!RayFusion::validateReflection(s, error)) throw py::value_error(error);
        if (!rtapi::setReflection(s, error)) throw std::runtime_error(error);
        return reflectionDictionary<py::dict>();
    }, py::arg("enabled").noconvert(), py::arg("samples") = 1,
       py::arg("roughness_gate") = 0.15, py::arg("weight_gate") = 0.01,
       py::arg("max_distance") = 200.0, py::arg("follow_quality_preset") = py::none(),
       "Per-pixel specular reflections. By DEFAULT the budget follows the raster "
       "quality preset (like shadow tiles and PCF taps do); naming samples, "
       "roughness_gate or weight_gate takes manual control. Replaces the "
       "environment radiance lookup "
       "rather than adding a metal-only term, so dielectrics inherit it through "
       "the same Fresnel weight. One bounce: no mirror in a mirror. Counters lag "
       "one frame -- produce a viewport frame between writing and measuring.");
}
}
