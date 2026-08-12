#include "RtPythonDebris.h"
#include "Api/RtApi.h"

#include <pybind11/stl.h>
#include <stdexcept>

namespace py = pybind11;
namespace {
void require(const rtapi::Result& r) { if (!r.ok) throw std::runtime_error(r.error); }
py::dict infoDict(const rtapi::AshDebrisInfo& i) {
    py::dict d;
    d["enabled"] = i.enabled;
    d["max_particles"] = i.max_particles;
    d["particles_per_kg"] = i.particles_per_kg;
    d["near_distance"] = i.near_distance;
    d["far_lod_scale"] = i.far_lod_scale;
    d["lifetime_seconds"] = i.lifetime_seconds;
    d["alive_particles"] = i.alive_particles;
    d["events"] = i.events;
    d["requested_particles"] = i.requested_particles;
    d["spawned_particles"] = i.spawned_particles;
    d["lod_reduced_particles"] = i.lod_reduced_particles;
    d["budget_rejected_particles"] = i.budget_rejected_particles;
    d["accepted_mass_kg"] = i.accepted_mass_kg;
    d["reservoir_mass_kg"] = i.reservoir_mass_kg;
    return d;
}
}

namespace rtpy {
void registerDebrisBindings(py::module_& debris) {
    debris.def("configure", [](bool enabled, uint64_t max_particles,
                                float particles_per_kg, float near_distance,
                                float far_lod_scale, float lifetime_seconds) {
        require(rtapi::configureAshDebris(enabled, max_particles,
            particles_per_kg, near_distance, far_lod_scale, lifetime_seconds));
    }, py::arg("enabled") = true, py::arg("max_particles") = 4096,
       py::arg("particles_per_kg") = 120.0f, py::arg("near_distance") = 12.0f,
       py::arg("far_lod_scale") = 0.25f, py::arg("lifetime_seconds") = 5.0f);
    debris.def("emit_ash", [](const py::tuple& center, float mass_kg,
                               const py::tuple& velocity,
                               float camera_distance, uint32_t seed) {
        if (center.size() != 3 || velocity.size() != 3)
            throw std::runtime_error("center and velocity must contain three values");
        uint64_t spawned = 0;
        require(rtapi::emitAshDebris(
            Vec3(py::cast<float>(center[0]), py::cast<float>(center[1]), py::cast<float>(center[2])),
            Vec3(py::cast<float>(velocity[0]), py::cast<float>(velocity[1]), py::cast<float>(velocity[2])),
            mass_kg, camera_distance, seed, spawned));
        return spawned;
    }, py::arg("center"), py::arg("mass_kg"),
       py::arg("velocity") = py::make_tuple(0, 0, 0),
       py::arg("camera_distance") = 0.0f,
       py::arg("seed") = 1u);
    debris.def("stats", []() {
        rtapi::AshDebrisInfo info;
        require(rtapi::getAshDebrisInfo(info));
        return infoDict(info);
    });
}
} // namespace rtpy
