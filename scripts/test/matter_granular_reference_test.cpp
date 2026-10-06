#include "Fluid/GranularReference.h"

#include <cassert>
#include <cmath>

int main() {
    using namespace RayTrophiSim::Fluid::Granular;
    ReferenceConfig config;
    ContactBody a;
    a.id = 1;
    a.position = Vec3(0.0f, 1.0f, 0.0f);
    a.radius_m = 0.05f;
    a.mass_kg = 0.2f;
    config.bodies = {a};
    config.plane_enabled = false;
    config.duration_s = 0.2;
    config.maximum_dt_s = 0.0001;
    ReferenceReport report;
    std::string error;
    assert(runGrainReference(config, report, error));
    const auto& final = report.frames.back();
    assert(std::abs(final.seconds - 0.2) < 1e-6);
    assert(std::abs(final.bodies[0].velocity.y + 1.962f) < 0.001f);
    assert(std::abs(final.bodies[0].position.y - 0.8038f) < 0.001f);
    assert(report.contact_evaluations == 0);
    assert(final.mass_kg == report.frames.front().mass_kg);

    config.gravity = Vec3(0.0f);
    config.bodies[0].position = Vec3(-0.08f, 0.0f, 0.0f);
    config.bodies[0].velocity = Vec3(1.0f, 0.2f, 0.0f);
    ContactBody b = config.bodies[0];
    b.id = 2;
    b.position = -config.bodies[0].position;
    b.velocity = -config.bodies[0].velocity;
    b.saturation = 1.0f;
    config.bodies.push_back(b);
    assert(runGrainReference(config, report, error));
    assert(report.contact_evaluations > 0);
    assert(report.frames.back().momentum.length() < 1e-5f);
    assert((report.frames.back().angular_momentum -
        report.frames.front().angular_momentum).length() < 1e-4f);
    assert(report.frames.back().bodies[0].saturation == 0.0f);
    assert(report.frames.back().bodies[1].saturation == 1.0f);

    const auto old_steps = report.micro_steps;
    config.bodies[1].id = 1;
    assert(!runGrainReference(config, report, error));
    assert(report.micro_steps == old_steps);
    config.bodies[1].id = 2;
    config.contact.normal_stiffness_n_m = 1e30f;
    assert(!runGrainReference(config, report, error));
    assert(report.micro_steps == old_steps);
}
