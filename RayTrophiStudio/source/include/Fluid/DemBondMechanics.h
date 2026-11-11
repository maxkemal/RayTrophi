#pragma once

#include <array>
#include <cstdint>
#include <string>

namespace RayTrophiSim::Fluid::DemBond {

// Solver-independent, double-precision reference law. Orientations map the
// particle's material frame to world space. Runtime integration must provide
// actual accumulated orientations; instantaneous DEM spin is not an orientation.
using Vector = std::array<double, 3>;
using Quaternion = std::array<double, 4>; // w, x, y, z

struct Endpoint {
    uint64_t id = 0;
    Vector position{};
    Quaternion orientation{1.0, 0.0, 0.0, 0.0};
};

struct Material {
    double young_pa = 0.0;
    double poisson = 0.25;
    double radius_m = 0.0; // bond section, not the grain radius
    double tensile_strength_pa = 0.0;
    double shear_strength_pa = 0.0;
};

enum class Failure : uint32_t {
    None,
    Tension,
    Shear
};

struct State {
    uint64_t first_id = 0;
    uint64_t second_id = 0;
    Vector rest_separation{};
    Quaternion first_rest_orientation{1.0, 0.0, 0.0, 0.0};
    Quaternion second_rest_orientation{1.0, 0.0, 0.0, 0.0};
    double rest_length_m = 0.0;
    Failure failure = Failure::None;
    double fracture_dissipation_j = 0.0;
};

struct Response {
    Vector first_force_n{};
    Vector second_force_n{};
    Vector first_torque_nm{};
    Vector second_torque_nm{};
    double elastic_energy_j = 0.0;
    double released_energy_j = 0.0;
    double tensile_stress_pa = 0.0;
    double shear_stress_pa = 0.0;
    bool broken_now = false;
};

// Explicit authoring event; proximity alone must never call this on every step.
// No mutation on error. Input ordering is significant and identity is stable.
bool create(const Endpoint& first, const Endpoint& second, State& state,
            std::string& error);

// Small-strain corotational circular beam reference, with finite rigid rotation.
// Brittle irreversible failure releases the stored energy to a dissipation
// ledger, never as an artificial velocity kick. Compression alone does not break
// a bond. Contact, capillary bridges, thermal weakening and plastic compaction
// are separate laws; this function neither creates nor heals bonds.
bool evaluate(const Endpoint& first, const Endpoint& second, const Material& material,
              State& state, Response& response, std::string& error);

// Conservative stiffness-row bound for an explicit integrator, seconds.
// Return zero with error on invalid inputs. Callers must combine incident bonds
// per particle and include their contact law; this is not a whole-graph CFL.
double stableStep(const Material& material, double length_m, double first_mass_kg,
                  double second_mass_kg, double first_inertia_kg_m2,
                  double second_inertia_kg_m2, uint32_t maximum_degree,
                  std::string& error);

} // namespace RayTrophiSim::Fluid::DemBond
