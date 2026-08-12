#include "scene_data.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace {

using RayTrophiSim::FractureGroupBounds;

// Exact projected area of an axis-aligned box onto the plane normal to `n`
// (|n| == 1). This is the area the blast front actually pushes on, which is why
// it is direction dependent: a wall struck face-on presents far more of itself
// than the same wall struck edge-on.
float projectedArea(const Vec3& extent, const Vec3& n) {
    return std::abs(n.x) * extent.y * extent.z +
           std::abs(n.y) * extent.x * extent.z +
           std::abs(n.z) * extent.x * extent.y;
}

// Mean projected area of a box over ALL directions — Cauchy's formula, S/4:
//
//     A_mean = (dx*dy + dy*dz + dz*dx) / 2
//
// ★ Used when the blast centre sits ON the group, where `projectedArea` has no
// defensible answer. A front arriving from a direction is the whole model behind
// the directional form, and at zero separation there IS no direction: `radial`
// is float noise, so the normalised vector — and therefore the area and the
// impulse — come out of rounding error. That is not a small error. It landed on
// the cluster the blast hits HARDEST, i.e. the one number the whole gate turns
// on, and it is not reproducible between runs.
//
// Physically this is also the better model there: a body engulfed by the front
// is pushed on from every side, so the area it presents is the direction average
// rather than any one face. Note it is LARGER than any single face (0.136 vs
// 0.091 m² for a 0.3 m cube), so an engulfed group gets a bigger impulse than
// the old noise value gave it — deliberately, not incidentally.
float meanProjectedArea(const Vec3& extent) {
    return (extent.x * extent.y + extent.y * extent.z + extent.z * extent.x) * 0.5f;
}

} // namespace

void SceneData::emitCombustionStructuralImpulses(float dt) {
    if (dt <= 0.0f) return;
    for (auto& system : particle_systems) {
        if (!system.runtime) continue;
        const auto& domains = system.runtime->gridDomains();
        const auto& states = system.runtime->gridDomainStates();
        for (std::size_t i = 0; i < domains.size() && i < states.size(); ++i) {
            const auto& desc = domains[i];
            const auto& state = states[i];
            if (desc.type != RayTrophiSim::SimulationDomainType::Gas) continue;
            if (!desc.fire_enabled || !desc.structural_coupling_enabled) continue;
            if (!state.valid || state.grid.interaction.empty()) continue;

            float& clock = combustion_event_clock_[desc.name];
            clock += dt;
            const float interval = std::max(desc.structural_event_interval, 1.0f / 120.0f);
            if (clock < interval) continue;
            const float elapsed = clock;
            clock = 0.0f;

            // ── Where it is burning, how hard, and over what extent ───────────
            // ★ Measured HERE rather than inside the solver on purpose. Gas
            // combustion runs on the GPU with a host download, so a reduction
            // living in the CPU kernel would simply not exist on the path that
            // actually runs. Reading the host mirror works for both backends,
            // and the interval gate means this full scan happens roughly four
            // times a second, not sixty.
            const auto& grid = state.grid;
            const std::size_t cells = grid.interaction.size();
            double sum = 0.0, sx = 0.0, sy = 0.0, sz = 0.0, sr2 = 0.0;
            std::size_t burning = 0;
            const float h = std::max(state.voxel_size, 1e-5f);
            const int plane = grid.nx * grid.ny;
            for (std::size_t c = 0; c < cells; ++c) {
                const float w = grid.interaction[c];
                if (!(w > 1e-4f)) continue;
                const int k = static_cast<int>(c / plane);
                const int rem = static_cast<int>(c - static_cast<std::size_t>(k) * plane);
                const int j = rem / grid.nx;
                const int ii = rem - j * grid.nx;
                const double wx = state.bounds_min.x + (ii + 0.5) * h;
                const double wy = state.bounds_min.y + (j + 0.5) * h;
                const double wz = state.bounds_min.z + (k + 0.5) * h;
                sum += w;
                sx += w * wx; sy += w * wy; sz += w * wz;
                sr2 += w * (wx * wx + wy * wy + wz * wz);
                ++burning;
            }
            if (burning == 0 || sum <= 1e-9) continue;

            // ★ MEAN burn rate, never the sum. A sum grows with the number of
            // cells alight, so the identical fire on a finer grid would report a
            // stronger blast — the same resolution dependence that broke the
            // fixed-point work. Extent is reported separately; intensity is
            // intensity.
            const float intensity = static_cast<float>(sum / static_cast<double>(burning));
            if (intensity < desc.structural_min_intensity) continue;

            const double inv = 1.0 / sum;
            const double cx = sx * inv, cy = sy * inv, cz = sz * inv;
            // Var = E[x²] - E[x]², summed over the axes: the rms spread of the
            // burning mass about its own centre. A bounding box would be
            // dominated by one stray ember and would grow every time one
            // drifted — exactly the shape of number that makes a blast look as
            // if it came from nowhere.
            const double spread = (sr2 * inv) - (cx * cx + cy * cy + cz * cz);
            const float radius = std::max(static_cast<float>(std::sqrt(std::max(spread, 0.0))), h);

            RayTrophiSim::StructuralImpulseEvent event;
            event.domain = desc.name;
            event.center = Vec3(static_cast<float>(cx), static_cast<float>(cy),
                                static_cast<float>(cz));
            event.radius = radius;
            event.peak_pressure_kpa = intensity * desc.structural_pressure_scale;
            // The interval this event stands for, so a sustained fire delivers a
            // sequence of honest blows instead of one blow counted every frame.
            event.duration_seconds = elapsed;
            event.coupling = 1.0f;
            queueStructuralImpulse(event);
        }
    }
}

void SceneData::queueStructuralImpulse(
    RayTrophiSim::StructuralImpulseEvent event) {
    event.sequence = ++structural_impulse_sequence_;
    event.radius = std::max(event.radius, 0.001f);
    event.peak_pressure_kpa = std::max(event.peak_pressure_kpa, 0.0f);
    event.duration_seconds = std::max(event.duration_seconds, 0.0f);
    event.coupling = std::max(event.coupling, 0.0f);
    structural_impulse_events_.push_back(event);
    ++structural_impulse_stats_.queued;
}

void SceneData::processStructuralImpulseEvents() {
    if (structural_impulse_events_.empty()) return;
    std::vector<RayTrophiSim::StructuralImpulseEvent> events;
    events.swap(structural_impulse_events_);

    // ── Gather intact groups once, then serve every event from that ───────────
    std::unordered_map<std::string, std::string> node_to_group;
    for (const auto& body : rigid_bodies) {
        if (!body.getBreakable() || body.broken) continue;
        const std::string group = body.getFractureGroup();
        if (group.empty() || body.source_name.empty()) continue;
        node_to_group.emplace(body.source_name, group);
    }
    if (node_to_group.empty()) {
        for (const auto& event : events) {
            ++structural_impulse_stats_.consumed;
            structural_impulse_stats_.last_peak_pressure_kpa = event.peak_pressure_kpa;
        }
        return;
    }

    // One pass over the objects for every group at once — the same accumulator
    // the reporting path uses, so `world_extent` and the area this impulse is
    // computed from can no longer disagree.
    std::unordered_map<std::string, FractureGroupBounds> group_bounds;
    accumulateFractureGroupBounds(node_to_group, group_bounds);

    for (const auto& event : events) {
        ++structural_impulse_stats_.consumed;
        structural_impulse_stats_.last_peak_pressure_kpa = event.peak_pressure_kpa;
        structural_impulse_stats_.last_max_impulse = 0.0f;
        structural_impulse_stats_.last_projected_area_m2 = 0.0f;

        for (const auto& entry : group_bounds) {
            const FractureGroupBounds& bounds = entry.second;
            if (!bounds.any) continue;
            Vec3 radial = bounds.center() - event.center;
            const float distance = radial.length();
            if (distance > event.radius) continue;
            const Vec3 extent = bounds.extent();
            // ★ The "is there a direction at all?" test is RELATIVE TO THE BOX,
            // not an absolute 1e-5. A separation of a millimetre against a
            // 30 cm cluster names a direction only in the arithmetic sense: the
            // vector is dominated by where the AABB's corners happened to fall,
            // and it swings wildly for a blast the user placed at the centre.
            // The old absolute epsilon let exactly that through, because it is
            // essentially never hit — the numbers it was meant to protect
            // against are the ones just above it.
            const float engulf_radius =
                std::max(extent.x, std::max(extent.y, extent.z)) * 0.05f + 1e-4f;
            const bool engulfed = distance <= engulf_radius;
            const Vec3 direction = engulfed
                ? Vec3(0.0f, 1.0f, 0.0f) : radial * (1.0f / distance);
            const float falloff = 1.0f - distance / event.radius;

            // ── Impulse, in newton-seconds ────────────────────────────────────
            // ★ THE AREA TERM IS THE POINT OF THIS REWRITE.
            //
            // The old form was `kPa * seconds * coupling * falloff`, which is not
            // an impulse in any unit system — and, worse, contained no measure of
            // the object at all. A 10 cm box and a 10 m wall in the same blast
            // received the identical push, so every fracture threshold in a scene
            // had to be hand-retuned the moment anything was rescaled, and the
            // `coupling` knob was quietly absorbing the missing area.
            //
            // Pressure is force per unit area, so the force on a body is the
            // overpressure times the area it presents to the front, and the
            // impulse is that force integrated over the pulse duration:
            //
            //     J [N s] = dp [Pa] * A_projected [m^2] * dt [s] * coupling
            //
            // coupling stays as the one honest fudge factor: the fraction of the
            // front that couples into the structure rather than flowing around
            // it. It is now dimensionless and 0..1 actually means something.
            //
            // ★ MAGNITUDES HAVE CHANGED BY ORDERS OF MAGNITUDE. Thresholds set
            // against the old expression are meaningless; break impulses are now
            // real N s and must be authored as such.
            const float area = engulfed ? meanProjectedArea(extent)
                                        : projectedArea(extent, direction);
            const float pressure_pa = event.peak_pressure_kpa * 1000.0f;
            const float impulse = pressure_pa * area *
                event.duration_seconds * event.coupling * falloff;
            if (!(impulse > 0.0f)) continue;

            ++structural_impulse_stats_.affected_groups;
            if (impulse > structural_impulse_stats_.last_max_impulse) {
                structural_impulse_stats_.last_max_impulse = impulse;
                structural_impulse_stats_.last_projected_area_m2 = area;
            }
            if (applyFractureImpulse(entry.first, event.center, direction, impulse))
                ++structural_impulse_stats_.fractured_groups;
        }
    }
}
