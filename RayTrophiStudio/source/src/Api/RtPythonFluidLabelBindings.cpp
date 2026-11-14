// Studio-only Python transport. Shared services and IPC live in the original module.
#include "RtFluidLabelBindings.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

using nlohmann::json;
namespace py = pybind11;

py::dict fluidLabelsToPython(const RayTrophiSim::Fluid::ParticleLabelReport& report) {
    // One wire vocabulary for both transports, not two independently maintained
    // dictionaries. This is a read-only report; importing json cannot run IPC.
    return py::module_::import("json")
        .attr("loads")(fluidLabelsToJson(report).dump())
        .cast<py::dict>();
}
