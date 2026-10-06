#include "Fluid/GranularReference.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <utility>

namespace RayTrophiSim::Fluid::Granular {
namespace {

using Cell = std::array<int, 3>;
using Pair = std::pair<uint64_t, uint64_t>;

bool finite(const Vec3& value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

float inertia(const ContactBody& body) {
    return 0.4f * body.mass_kg * body.radius_m * body.radius_m;
}

ReferenceFrame snapshot(const std::vector<ContactBody>& bodies, double seconds) {
    ReferenceFrame frame;
    frame.seconds = seconds;
    frame.bodies = bodies;
    for (const auto& body : bodies) {
        frame.mass_kg += body.mass_kg;
        frame.kinetic_energy_j += 0.5 * body.mass_kg * body.velocity.length_squared() +
            0.5 * inertia(body) * body.angular_velocity.length_squared();
        const Vec3 momentum = body.velocity * body.mass_kg;
        frame.momentum += momentum;
        frame.angular_momentum += body.position.cross(momentum) +
            body.angular_velocity * inertia(body);
    }
    return frame;
}

} // namespace

bool runGrainReference(const ReferenceConfig& config, ReferenceReport& report,
                       std::string& error) {
    if (config.bodies.empty() || config.bodies.size() > 64 || !finite(config.gravity) ||
        !std::isfinite(config.duration_s) || config.duration_s <= 0.0 ||
        config.duration_s > 2.0 || !std::isfinite(config.maximum_dt_s) ||
        config.maximum_dt_s < 1e-5 || config.maximum_dt_s > 0.01 ||
        !std::isfinite(config.sample_interval_s) || config.sample_interval_s < 0.02 ||
        config.sample_interval_s > config.duration_s) {
        error = "grain reference requires 1..64 bodies, duration (0,2], "
            "dt [1e-5,.01], samples >=.02";
        return false;
    }
    std::vector<ContactBody> bodies = config.bodies;
    std::sort(bodies.begin(), bodies.end(), [](const auto& a, const auto& b) {
        return a.id < b.id;
    });
    float min_radius = std::numeric_limits<float>::max();
    float max_radius = 0.0f;
    float min_mass = std::numeric_limits<float>::max();
    uint64_t previous_id = 0;
    for (const auto& body : bodies) {
        if (body.id <= previous_id || !finite(body.position) || !finite(body.velocity) ||
            !finite(body.angular_velocity) || !std::isfinite(body.mass_kg) ||
            body.mass_kg < 1e-5f || body.mass_kg > 100.0f ||
            !std::isfinite(body.radius_m) || body.radius_m < 0.001f || body.radius_m > 1.0f ||
            !std::isfinite(body.saturation) || body.saturation < 0.0f || body.saturation > 1.0f ||
            body.position.abs().max_component() > 1000.0f ||
            body.velocity.length() > 100.0f || body.angular_velocity.length() > 10000.0f) {
            error = "grain reference body identity, units, local saturation or range is invalid";
            return false;
        }
        previous_id = body.id;
        min_radius = std::min(min_radius, body.radius_m);
        max_radius = std::max(max_radius, body.radius_m);
        min_mass = std::min(min_mass, body.mass_kg);
    }
    // Validate the shared constitutive parameters even for a no-contact run.
    ContactBody check_a = bodies.front();
    ContactBody check_b = check_a;
    check_a.id = 1;
    check_b.id = 2;
    check_b.position.x += 3.0f * check_a.radius_m;
    ContactHistory check_history;
    ContactResult check_result;
    if (!evaluateSphereContact(check_a, check_b, config.contact, 0.001f,
            check_history, check_result, error)) {
        return false;
    }
    if (config.plane_enabled) {
        check_history = ContactHistory{};
        if (!evaluatePlaneContact(check_a, config.plane_normal, config.plane_offset_m,
                config.contact, 0.001f, check_history, check_result, error)) {
            return false;
        }
    }
    // Conservative contact wave/damping bound, including tangential rotation.
    const double stiffness = config.contact.normal_stiffness_n_m +
        5.0 * config.contact.tangential_stiffness_n_m;
    const double damping = config.contact.normal_damping_n_s_m +
        5.0 * config.contact.tangential_damping_n_s_m;
    const double elastic_dt = 0.1 * std::sqrt(min_mass / (2.0 * stiffness));
    const double damping_dt = damping > 0.0 ? 0.1 * min_mass / (2.0 * damping) : 0.01;
    const double base_dt = std::min({config.maximum_dt_s, elastic_dt, damping_dt});
    if (base_dt < 1e-7 || config.duration_s / base_dt * bodies.size() > 1000000.0) {
        error = "grain reference contact resolution exceeds the bounded CPU work budget";
        return false;
    }
    ReferenceReport candidate;
    candidate.frames.push_back(snapshot(bodies, 0.0));
    std::map<Pair, ContactHistory> histories;
    std::map<uint64_t, ContactHistory> plane_histories;
    const float cell_size = 2.0f * max_radius;
    const auto cellOf = [cell_size](const Vec3& position) {
        return Cell{static_cast<int>(std::floor(position.x / cell_size)),
            static_cast<int>(std::floor(position.y / cell_size)),
            static_cast<int>(std::floor(position.z / cell_size))};
    };
    double seconds = 0.0;
    double next_sample = std::min(config.sample_interval_s, config.duration_s);
    while (seconds < config.duration_s - 1e-10) {
        if ((candidate.micro_steps + 1) * bodies.size() > 1000000) {
            error = "grain reference adaptive step budget exhausted";
            return false;
        }
        float max_speed = 0.0f;
        for (const auto& body : bodies) {
            max_speed = std::max(max_speed, body.velocity.length());
        }
        const double travel_dt = 0.1 * min_radius /
            std::max(0.01f, max_speed + config.gravity.length() * static_cast<float>(base_dt));
        const double remaining = std::min(next_sample, config.duration_s) - seconds;
        const float dt = static_cast<float>(std::min({base_dt, travel_dt, remaining}));
        if (!std::isfinite(dt) || dt <= 0.0f) {
            error = "grain reference time partition failed";
            return false;
        }
        std::map<Cell, std::vector<std::size_t>> cells;
        for (std::size_t i = 0; i < bodies.size(); ++i) {
            cells[cellOf(bodies[i].position)].push_back(i);
        }
        std::vector<Vec3> forces(bodies.size(), Vec3(0.0f));
        std::vector<Vec3> torques(bodies.size(), Vec3(0.0f));
        for (std::size_t i = 0; i < bodies.size(); ++i) {
            forces[i] = config.gravity * bodies[i].mass_kg;
        }
        std::map<Pair, ContactHistory> next_histories;
        for (std::size_t i = 0; i < bodies.size(); ++i) {
            const auto& a = bodies[i];
            const Cell cell = cellOf(a.position);
            for (int x = -1; x <= 1; ++x) {
                for (int y = -1; y <= 1; ++y) {
                    for (int z = -1; z <= 1; ++z) {
                        const auto found = cells.find(Cell{cell[0] + x, cell[1] + y, cell[2] + z});
                        if (found == cells.end()) {
                            continue;
                        }
                        for (const std::size_t j : found->second) {
                            if (j <= i) {
                                continue;
                            }
                            ++candidate.candidate_pairs;
                            const auto& b = bodies[j];
                            if ((b.position - a.position).length_squared() >=
                                (a.radius_m + b.radius_m) * (a.radius_m + b.radius_m)) {
                                continue;
                            }
                            const Pair key{a.id, b.id};
                            const auto old = histories.find(key);
                            ContactHistory history = old == histories.end()
                                ? ContactHistory{} : old->second;
                            ContactResult contact;
                            if (!evaluateSphereContact(a, b, config.contact, dt,
                                    history, contact, error)) {
                                return false;
                            }
                            forces[i] += contact.force_on_a;
                            forces[j] += contact.force_on_b;
                            torques[i] += contact.torque_on_a;
                            torques[j] += contact.torque_on_b;
                            next_histories.emplace(key, history);
                            ++candidate.contact_evaluations;
                            candidate.maximum_overlap_ratio = std::max(
                                candidate.maximum_overlap_ratio,
                                contact.overlap_m / std::min(a.radius_m, b.radius_m));
                        }
                    }
                }
            }
            if (config.plane_enabled) {
                ContactResult contact;
                auto& history = plane_histories[a.id];
                if (!evaluatePlaneContact(a, config.plane_normal, config.plane_offset_m,
                        config.contact, dt, history, contact, error)) {
                    return false;
                }
                forces[i] += contact.force_on_a;
                torques[i] += contact.torque_on_a;
                if (contact.touching) {
                    ++candidate.contact_evaluations;
                    candidate.maximum_overlap_ratio = std::max(candidate.maximum_overlap_ratio,
                        contact.overlap_m / a.radius_m);
                }
            }
        }
        histories.swap(next_histories);
        for (std::size_t i = 0; i < bodies.size(); ++i) {
            auto& body = bodies[i];
            body.velocity += forces[i] * (dt / body.mass_kg);
            body.angular_velocity += torques[i] * (dt / inertia(body));
            body.position += body.velocity * dt;
            if (!finite(body.position) || !finite(body.velocity) ||
                !finite(body.angular_velocity) || body.position.abs().max_component() > 1000.0f) {
                error = "grain reference escaped finite state bounds";
                return false;
            }
        }
        ++candidate.micro_steps;
        candidate.maximum_micro_dt_s = std::max(candidate.maximum_micro_dt_s,
            static_cast<double>(dt));
        seconds += dt;
        if (seconds >= next_sample - 1e-9) {
            candidate.frames.push_back(snapshot(bodies, seconds));
            next_sample = std::min(config.duration_s, next_sample + config.sample_interval_s);
        }
    }
    report = std::move(candidate);
    error.clear();
    return true;
}

} // namespace RayTrophiSim::Fluid::Granular
