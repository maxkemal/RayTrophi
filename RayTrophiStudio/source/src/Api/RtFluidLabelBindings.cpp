#include "RtFluidLabelBindings.h"
#include <pybind11/pybind11.h>

using nlohmann::json;
namespace py = pybind11;

json fluidLabelsToJson(const RayTrophiSim::Fluid::ParticleLabelReport& report) {
    using namespace RayTrophiSim::Fluid;
    json primary = json::object();
    json secondary = json::object();
    for (std::size_t i = 0; i < kParticleLabelCount; ++i) {
        const char* name = particleLabelName(static_cast<ParticleLabel>(i));
        primary[name] = report.primary[i];
        secondary[name] = report.secondary[i];
    }
    return {
        {"available", report.available},
        {"primary_particles", report.primary_particles},
        {"secondary_particles", report.secondary_particles},
        {"primary", primary},
        {"secondary", secondary},
        {"primary_complete", report.available && report.primary_particles > 0 &&
            report.primary[static_cast<std::size_t>(ParticleLabel::Unknown)] == 0},
        {"secondary_affects_mass", false},
        {"mist_generation", true},
        {"mist_mass_fraction_max", kParticleMistMassFraction},
        {"render_routing", "substance+label"},
        {"classifier", "mass+neighborhood_v2"},
        {"radius_voxels", kParticleLabelRadiusVoxels},
        {"detach_max_neighbors", kParticleLabelDetachNeighbors},
        {"rejoin_min_neighbors", kParticleLabelRejoinNeighbors},
        {"last_step", {
            {"on_gpu", report.last_step.on_gpu},
            {"milliseconds", report.last_step.milliseconds},
            {"bin_milliseconds", report.last_step.bin_milliseconds},
            {"classify_milliseconds", report.last_step.classify_milliseconds},
            {"occupied_bins", report.last_step.occupied_bins},
            {"center_resolved", report.last_step.center_resolved},
            {"particles", report.last_step.particles},
            {"changed", report.last_step.changed}
        }}
    };
}

py::dict fluidLabelsToPython(const RayTrophiSim::Fluid::ParticleLabelReport& report) {
    // One wire vocabulary for both transports, not two independently maintained
    // dictionaries. This is a read-only report; importing json cannot run IPC.
    return py::module_::import("json").attr("loads")(fluidLabelsToJson(report).dump())
        .cast<py::dict>();
}
