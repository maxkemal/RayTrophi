#include "ParticleSimulation.h"

#include <algorithm>
#include <cmath>

namespace RayTrophiSim {

bool ParticleSimulationSystem::injectGasPressurePulse(
    const std::string& domain, const Vec3& center,
    float radius, float peak_pressure_kpa) {
    if (radius <= 0.0f || peak_pressure_kpa <= 0.0f) return false;
    for (std::size_t i = 0; i < grid_domains_.size() &&
                            i < grid_domain_states_.size(); ++i) {
        if (grid_domains_[i].name != domain ||
            grid_domains_[i].type != SimulationDomainType::Gas) continue;
        auto& state = grid_domain_states_[i];
        if (!state.valid) return false;
        auto& grid = state.grid;
        const float h = std::max(state.voxel_size, 1e-4f);
        const float velocity_peak = std::min(peak_pressure_kpa * 0.02f, 80.0f);
        bool touched = false;

        // ── Only the cells the pulse can actually reach ───────────────────────
        // This used to walk the entire grid and reject by distance. On a 256³ gas
        // domain that is ~16.7M iterations to touch a sphere that may span twenty
        // cells, paid every time a pulse fires.
        auto axisRange = [&](float min_bound, int n, float lo, float hi,
                             int& out_begin, int& out_end) {
            out_begin = std::max(0, static_cast<int>(std::floor((lo - min_bound) / h)) - 1);
            out_end = std::min(n - 1, static_cast<int>(std::ceil((hi - min_bound) / h)) + 1);
        };
        int x0, x1, y0, y1, z0, z1;
        axisRange(state.bounds_min.x, grid.nx, center.x - radius, center.x + radius, x0, x1);
        axisRange(state.bounds_min.y, grid.ny, center.y - radius, center.y + radius, y0, y1);
        axisRange(state.bounds_min.z, grid.nz, center.z - radius, center.z + radius, z0, z1);
        if (x0 > x1 || y0 > y1 || z0 > z1) return false;

        // ── One write per MAC face, evaluated AT the face ─────────────────────
        // ★ The previous version looped over CELLS and added the cell's dv to both
        // of its faces on each axis. Every interior face is shared by two cells,
        // so it received the kick twice while faces on the sphere's rim received
        // it once — the blast came out roughly 2x too strong in the middle and
        // asymmetric at the edges, which is precisely the shape of artefact that
        // reads as "the explosion is off-centre".
        //
        // A staggered face is its own sample point, so sample the radial field
        // there: an x-face sits at integer x, cell-centre y and z.
        auto kick = [&](const Vec3& face_position, int axis) -> float {
            Vec3 radial = face_position - center;
            const float distance = radial.length();
            if (distance >= radius || distance < 1e-5f) return 0.0f;
            const float falloff = 1.0f - distance / radius;
            return (radial[axis] / distance) * velocity_peak * falloff;
        };
        const Vec3& origin = state.bounds_min;
        for (int z = z0; z <= z1; ++z)
        for (int y = y0; y <= y1; ++y)
        for (int x = x0; x <= x1 + 1; ++x) {
            const float dv = kick(origin + Vec3(x * h, (y + 0.5f) * h, (z + 0.5f) * h), 0);
            if (dv != 0.0f) { grid.velXAt(x, y, z) += dv; touched = true; }
        }
        for (int z = z0; z <= z1; ++z)
        for (int y = y0; y <= y1 + 1; ++y)
        for (int x = x0; x <= x1; ++x) {
            const float dv = kick(origin + Vec3((x + 0.5f) * h, y * h, (z + 0.5f) * h), 1);
            if (dv != 0.0f) { grid.velYAt(x, y, z) += dv; touched = true; }
        }
        for (int z = z0; z <= z1 + 1; ++z)
        for (int y = y0; y <= y1; ++y)
        for (int x = x0; x <= x1; ++x) {
            const float dv = kick(origin + Vec3((x + 0.5f) * h, (y + 0.5f) * h, z * h), 2);
            if (dv != 0.0f) { grid.velZAt(x, y, z) += dv; touched = true; }
        }

        // Cell-centred pressure is written for TELEMETRY AND DEBUG VIEWS ONLY.
        // The next projection solves for its own pressure and overwrites this
        // array wholesale; the physical effect of the pulse is entirely in the
        // face velocities above. Stated because the write looks load-bearing.
        if (grid.pressure.size() == static_cast<std::size_t>(grid.getCellCount())) {
            for (int z = z0; z <= z1; ++z)
            for (int y = y0; y <= y1; ++y)
            for (int x = x0; x <= x1; ++x) {
                const Vec3 p = origin +
                    Vec3((x + 0.5f) * h, (y + 0.5f) * h, (z + 0.5f) * h);
                const float distance = (p - center).length();
                if (distance >= radius) continue;
                grid.pressure[grid.cellIndex(x, y, z)] +=
                    peak_pressure_kpa * (1.0f - distance / radius);
            }
        }
        if (touched) ++state.version;
        return touched;
    }
    return false;
}

} // namespace RayTrophiSim
