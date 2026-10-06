#pragma once

#include "GranularContact.h"

#include <vector>

namespace RayTrophiSim::Fluid::Granular {

struct ReferenceConfig {
    std::vector<ContactBody> bodies;
    ContactParams contact;
    Vec3 gravity = Vec3(0.0f, -9.81f, 0.0f);
    bool plane_enabled = true;
    Vec3 plane_normal = Vec3(0.0f, 1.0f, 0.0f);
    float plane_offset_m = 0.0f;
    double duration_s = 1.0;
    double maximum_dt_s = 0.001;
    double sample_interval_s = 0.02;
};

struct ReferenceFrame {
    double seconds = 0.0;
    std::vector<ContactBody> bodies;
    double mass_kg = 0.0;
    double kinetic_energy_j = 0.0;
    Vec3 momentum = Vec3(0.0f);
    Vec3 angular_momentum = Vec3(0.0f);
};

struct ReferenceReport {
    std::vector<ReferenceFrame> frames;
    uint64_t micro_steps = 0;
    uint64_t candidate_pairs = 0;
    uint64_t contact_evaluations = 0;
    double maximum_micro_dt_s = 0.0;
    float maximum_overlap_ratio = 0.0f;
};

// Bounded, transient CPU reference experiment. Does not mutate the scene,
// use MPM/render parcels or consume the main simulation cache. Transactional.
bool runGrainReference(const ReferenceConfig& config, ReferenceReport& report,
                       std::string& error);

} // namespace RayTrophiSim::Fluid::Granular
