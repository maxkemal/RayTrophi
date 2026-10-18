#include "Fluid/MatterGrainGasCoupling.h"

#include "Fluid/FluidParticles.h"
#include "Fluid/MatterGrainCoupling.h"
#include "FluidGrid.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace RayTrophiSim::Fluid {
namespace {

constexpr double kGasConstant = 8.314462;
constexpr double kAtmosphere = 101325.0;
constexpr double kAirMolar = 0.028965;

double grainMass(const FluidParticles& p, std::size_t g) {
    const double water = g < p.pore_water_mass_kg.size() ? p.pore_water_mass_kg[g] : 0.0;
    return double(p.rest_mass_kg[g]) * p.mass_fraction[g] + water;
}

bool finite(const Vec3& v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

// Air temperature of a cell from the Tv field (the density needs only Tv).
double cellVirtualKelvin(const FluidSim::FluidGrid& gas, std::size_t cell,
                         const MatterGrainGasAir& air) {
    if (!(air.kelvin_per_unit > 0.0f) || cell >= gas.temperature.size()) {
        return std::max(1.0, double(air.ambient_kelvin));
    }
    const double t = air.ambient_kelvin +
        (double(gas.temperature[cell]) - air.gas_ambient) * air.kelvin_per_unit;
    return std::isfinite(t) ? std::max(t, 1.0) : double(air.ambient_kelvin);
}

// Sutherland's law for air, Pa s.
double airViscosity(double kelvin) {
    constexpr double mu0 = 1.716e-5, t0 = 273.15, s = 110.4;
    return mu0 * std::pow(kelvin / t0, 1.5) * (t0 + s) / (kelvin + s);
}

// The 8 MAC faces of one velocity component around `local` (grid units,
// already offset to that component's face lattice), with trilinear weights.
// dims = face lattice size of the component.
int faceStencil(const FluidSim::FluidGrid& gas, int axis, double lx, double ly, double lz,
                std::size_t index[8], double weight[8]) {
    const int dx = gas.nx + (axis == 0 ? 1 : 0);
    const int dy = gas.ny + (axis == 1 ? 1 : 0);
    const int dz = gas.nz + (axis == 2 ? 1 : 0);
    const int i0 = static_cast<int>(std::floor(lx));
    const int j0 = static_cast<int>(std::floor(ly));
    const int k0 = static_cast<int>(std::floor(lz));
    const double fx = lx - i0, fy = ly - j0, fz = lz - k0;
    int n = 0;
    double total = 0.0;
    for (int c = 0; c < 8; ++c) {
        const int i = i0 + (c & 1), j = j0 + ((c >> 1) & 1), k = k0 + ((c >> 2) & 1);
        if (i < 0 || j < 0 || k < 0 || i >= dx || j >= dy || k >= dz) continue;
        const double w = ((c & 1) ? fx : 1.0 - fx) * (((c >> 1) & 1) ? fy : 1.0 - fy) *
            (((c >> 2) & 1) ? fz : 1.0 - fz);
        if (!(w > 0.0)) continue;
        index[n] = axis == 0 ? gas.velXIndex(i, j, k)
                 : axis == 1 ? gas.velYIndex(i, j, k) : gas.velZIndex(i, j, k);
        weight[n] = w;
        total += w;
        ++n;
    }
    // Renormalized at the domain edge: the whole reaction lands on the faces
    // that exist, so the exchanged momentum stays exactly opposite.
    if (total > 0.0) {
        for (int c = 0; c < n; ++c) weight[c] /= total;
    }
    return n;
}

struct CellLump {
    Vec3 impulse = Vec3(0.0f);     // sum of grain impulses, N s
    double grain_mass = 0.0;
    Vec3 grain_momentum = Vec3(0.0f);
    float scale = 1.0f;
};

} // namespace

void exchangeMatterGrainGasDrag(const FluidParticles& grains, float grain_radius,
                                FluidSim::FluidGrid& gas, const MatterGrainGasAir& air,
                                float dt, std::vector<Vec3>& acceleration,
                                MatterGrainStepReport& report) {
    report.gas_coupled_grains = 0;
    report.gas_drag_impulse = Vec3(0.0f);
    report.gas_reaction = Vec3(0.0f);
    report.gas_limited_cells = 0;
    report.gas_max_relative_speed = 0.0f;
    report.gas_momentum_residual = 0.0;
    const std::size_t count = grains.size();
    const std::size_t cells = static_cast<std::size_t>(std::max(0, gas.nx)) *
        std::max(0, gas.ny) * std::max(0, gas.nz);
    if (count == 0 || cells == 0 || !(dt > 0.0f) || !std::isfinite(dt) ||
        !(grain_radius > 0.0f) || !(gas.voxel_size > 0.0f) ||
        gas.vel_x.empty() || gas.vel_y.empty() || gas.vel_z.empty()) {
        return;
    }
    const double h = gas.voxel_size;
    const double cell_volume = h * h * h;
    const double diameter = 2.0 * double(grain_radius);

    // Pass 1: each grain's exact-over-the-frame impulse against its own drag.
    std::vector<Vec3> impulse(count, Vec3(0.0f));
    std::vector<int64_t> grain_cell(count, -1);
    std::unordered_map<std::size_t, CellLump> lumps;
    for (std::size_t g = 0; g < count; ++g) {
        const Vec3& x = grains.position[g];
        if (!finite(x)) continue;
        const Vec3 local = (x - gas.origin) / gas.voxel_size;
        const int i = static_cast<int>(std::floor(local.x));
        const int j = static_cast<int>(std::floor(local.y));
        const int k = static_cast<int>(std::floor(local.z));
        if (i < 0 || j < 0 || k < 0 || i >= gas.nx || j >= gas.ny || k >= gas.nz) continue;
        const double mass = grainMass(grains, g);
        if (!(mass > 0.0) || !std::isfinite(mass)) continue;
        const std::size_t cell = gas.cellIndex(i, j, k);
        const double tv = cellVirtualKelvin(gas, cell, air);
        const double rho = kAtmosphere * kAirMolar / (kGasConstant * tv);
        const double mu = airViscosity(tv);
        const Vec3 u = gas.sampleVelocity(x);
        const Vec3 v = grains.velocity[g];
        if (!finite(u) || !finite(v)) continue;
        const Vec3 relative = u - v;
        const double speed = relative.length();
        const double k_drag = matterGrainDragCoefficient(diameter, speed, rho, mu, 1.0);
        if (!(k_drag > 0.0)) continue;
        const double share = -std::expm1(-k_drag * dt / mass);
        const Vec3 j_grain = relative * static_cast<float>(mass * share);
        impulse[g] = j_grain;
        grain_cell[g] = static_cast<int64_t>(cell);
        CellLump& lump = lumps[cell];
        lump.impulse = lump.impulse + j_grain;
        lump.grain_mass += mass;
        lump.grain_momentum = lump.grain_momentum + v * static_cast<float>(mass);
        report.gas_max_relative_speed =
            std::max(report.gas_max_relative_speed, static_cast<float>(speed));
    }

    // Pass 2: no cell's gas may be pushed past the mixture's equilibrium.
    for (auto& [cell, lump] : lumps) {
        const int ci = static_cast<int>(cell % gas.nx);
        const int cj = static_cast<int>((cell / gas.nx) % gas.ny);
        const int ck = static_cast<int>(cell / (static_cast<std::size_t>(gas.nx) * gas.ny));
        const Vec3 centre = gas.origin +
            Vec3(ci + 0.5f, cj + 0.5f, ck + 0.5f) * gas.voxel_size;
        const Vec3 u = gas.sampleVelocity(centre);
        const double tv = cellVirtualKelvin(gas, cell, air);
        const double gas_mass = kAtmosphere * kAirMolar / (kGasConstant * tv) * cell_volume;
        const double total = gas_mass + lump.grain_mass;
        const Vec3 equilibrium =
            (u * static_cast<float>(gas_mass) + lump.grain_momentum) *
            static_cast<float>(1.0 / total);
        // What the gas would give up to reach the equilibrium.
        const double available = ((u - equilibrium) * static_cast<float>(gas_mass)).length();
        const double requested = lump.impulse.length();
        if (requested > available && requested > 0.0) {
            lump.scale = static_cast<float>(available / requested);
            ++report.gas_limited_cells;
        }
    }

    // Pass 3: grain accelerations and the opposite reaction on the gas faces.
    if (acceleration.size() != count) acceleration.resize(count, Vec3(0.0f));
    Vec3 to_grains(0.0f), to_gas(0.0f);
    for (std::size_t g = 0; g < count; ++g) {
        if (grain_cell[g] < 0) continue;
        const std::size_t cell = static_cast<std::size_t>(grain_cell[g]);
        const CellLump& lump = lumps[cell];
        const Vec3 j_grain = impulse[g] * lump.scale;
        if (!(j_grain.length() > 0.0f)) continue;
        const double mass = grainMass(grains, g);
        acceleration[g] = acceleration[g] + j_grain * static_cast<float>(1.0 / (mass * dt));
        to_grains = to_grains + j_grain;
        ++report.gas_coupled_grains;

        const double tv = cellVirtualKelvin(gas, cell, air);
        const double gas_cell_mass = kAtmosphere * kAirMolar / (kGasConstant * tv) * cell_volume;
        const Vec3 local = (grains.position[g] - gas.origin) / gas.voxel_size;
        const double offsets[3][3] = {{0.0, -0.5, -0.5}, {-0.5, 0.0, -0.5}, {-0.5, -0.5, 0.0}};
        const float component[3] = {-j_grain.x, -j_grain.y, -j_grain.z};
        std::vector<float>* faces[3] = {&gas.vel_x, &gas.vel_y, &gas.vel_z};
        for (int axis = 0; axis < 3; ++axis) {
            std::size_t index[8];
            double weight[8];
            const int n = faceStencil(gas, axis, local.x + offsets[axis][0],
                                      local.y + offsets[axis][1], local.z + offsets[axis][2],
                                      index, weight);
            std::vector<float>& field = *faces[axis];
            for (int c = 0; c < n; ++c) {
                if (index[c] >= field.size()) continue;
                field[index[c]] += static_cast<float>(component[axis] * weight[c] / gas_cell_mass);
            }
        }
        to_gas = to_gas - j_grain;
    }
    report.gas_drag_impulse = to_grains;
    report.gas_reaction = to_gas;
    report.gas_momentum_residual = (to_grains + to_gas).length();
}

} // namespace RayTrophiSim::Fluid
