#include "RtPythonMassTransfer.h"
#include "Api/RtApi.h"

#include <stdexcept>
namespace py = pybind11;
namespace {
void require(const rtapi::Result& r) { if (!r.ok) throw std::runtime_error(r.error); }
}
namespace rtpy {
void registerMassTransferBindings(py::module_& module) {
    py::module_ transfer = module.def_submodule(
        "mass_transfer", "Transactional molten surface mass to APIC transfer");
    transfer.def("queue", [](const std::string& object_key, float mass_kg,
                              const std::string& domain, float particles_per_kg,
                              const py::tuple& velocity) {
        if (velocity.size() != 3) throw std::runtime_error("velocity must contain three values");
        uint64_t sequence = 0;
        require(rtapi::queueMoltenMassTransfer(object_key, domain, mass_kg,
            particles_per_kg, Vec3(py::cast<float>(velocity[0]),
                                   py::cast<float>(velocity[1]),
                                   py::cast<float>(velocity[2])), sequence));
        return sequence;
    }, py::arg("object_key"), py::arg("mass_kg"), py::arg("domain") = "",
       py::arg("particles_per_kg") = 32.0f,
       py::arg("velocity") = py::make_tuple(0, 0, 0));
    transfer.def("stats", [] {
        rtapi::MoltenMassTransferInfo i; require(rtapi::getMoltenMassTransferInfo(i));
        py::dict d;
        d["queued"] = i.queued; d["completed"] = i.completed;
        d["deferred_no_domain"] = i.deferred_no_domain;
        d["deferred_no_capacity"] = i.deferred_no_capacity;
        d["dropped"] = i.dropped;
        d["discarded_on_reset"] = i.discarded_on_reset;
        d["requested_mass"] = i.requested_mass;
        d["transferred_mass"] = i.transferred_mass;
        d["spawned_particles"] = i.spawned_particles;
        d["last_object"] = i.last_object; d["last_domain"] = i.last_domain;
        d["last_substance"] = i.last_substance;
        d["last_temperature_kelvin"] = i.last_temperature_kelvin;
        d["last_combustible_fraction"] = i.last_combustible_fraction;
        d["live_tagged_particles"] = i.live_tagged_particles;
        d["mean_remaining_mass_fraction"] = i.mean_remaining_mass_fraction;
        return d;
    });
}
} // namespace rtpy
