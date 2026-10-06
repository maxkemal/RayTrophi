// H1-R parity with the GPU static-hold arm (rt_h1_grain_runtime_ipc.py
// --static-only): one sphere resting on a 20 degree plane. With the EPSD2
// rolling spring (mu_r = 1 > tan 20) and a Coulomb-capped tangential spring
// (mu = .5 > tan 20) it must hold; without rolling resistance it must roll.
#include "Fluid/GranularContact.h"
#include "Fluid/GranularReference.h"

#include <cassert>
#include <cmath>
#include <cstdio>

using namespace RayTrophiSim::Fluid::Granular;

static double creep(float rolling) {
    const float angle = 20.0f * 3.14159265f / 180.0f;
    const Vec3 normal(-std::sin(angle), std::cos(angle), 0.0f);
    ReferenceConfig config;
    ContactBody body;
    body.id = 1;
    body.radius_m = .025f;
    body.mass_kg = .174533f;
    // Born resting on the slope (overlap ~ its weight's static compression).
    body.position = normal * (body.radius_m - 8e-5f);
    config.bodies = {body};
    config.contact.normal_stiffness_n_m = 20000.0f;
    config.contact.normal_damping_n_s_m = 4.0f;
    config.contact.tangential_stiffness_n_m = 20000.0f * 2.0f / 7.0f;
    config.contact.tangential_damping_n_s_m = 4.0f;
    config.contact.dry_friction = .5f;
    config.contact.rolling_friction = rolling;
    config.plane_normal = normal;
    config.plane_offset_m = 0.0f;
    config.duration_s = 2.0;
    config.maximum_dt_s = 1e-4;
    config.sample_interval_s = .1;
    ReferenceReport report;
    std::string error;
    const bool ok = runGrainReference(config, report, error);
    if (!ok) std::printf("error: %s\n", error.c_str());
    assert(ok);
    const Vec3 along(std::cos(angle), std::sin(angle), 0.0f);
    const auto at = [&](std::size_t frame) {
        const Vec3 p = report.frames[frame].bodies[0].position;
        return double(p.x * along.x + p.y * along.y);
    };
    const std::size_t last = report.frames.size() - 1;
    return std::abs(at(last) - at(last - 10));  // last 1 s
}

int main() {
    const double held = creep(1.0f);
    const double rolled = creep(0.0f);
    std::printf("slope creep last 1 s: mu_r=1 %.3e m, mu_r=0 %.3e m\n", held, rolled);
    assert(held <= 5e-4);
    assert(rolled >= 1e-2);
    std::printf("PASS grain reference slope (EPSD2 parity with GPU static arm)\n");
    return 0;
}
