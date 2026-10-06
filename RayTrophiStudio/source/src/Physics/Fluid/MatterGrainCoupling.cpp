#include "Fluid/MatterGrainCoupling.h"
#include "Fluid/FluidParticles.h"
#include "Fluid/FluidPhysicalMass.h"
#include "Fluid/FluidThermalLiquid.h"
#include "Fluid/SubstanceTag.h"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <unordered_map>
#include <unordered_set>

namespace RayTrophiSim::Fluid {
namespace {

constexpr double kPi = 3.14159265358979323846;
// A grain is never surrounded by less than this pore fraction; random close
// packing of equal spheres is ~0.36. Guards the voidage power law.
constexpr double kMinimumVoidage = 0.35;

bool finite(const Vec3& v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

std::size_t cellIndex(const MatterGrainLiquidField& f, int i, int j, int k) {
    return (static_cast<std::size_t>(k) * f.ny + j) * f.nx + i;
}

// Eight cell-centre neighbours of `p` with trilinear weights; cells outside
// the grid get index -1 and weight 0.
void neighbours(const MatterGrainLiquidField& f, const Vec3& p, int cells[8], double weights[8]) {
    const double lx = (p.x - f.origin.x) / f.h - .5;
    const double ly = (p.y - f.origin.y) / f.h - .5;
    const double lz = (p.z - f.origin.z) / f.h - .5;
    const int i0 = static_cast<int>(std::floor(lx));
    const int j0 = static_cast<int>(std::floor(ly));
    const int k0 = static_cast<int>(std::floor(lz));
    const double fx = lx - i0, fy = ly - j0, fz = lz - k0;
    for (int c = 0; c < 8; ++c) {
        const int i = i0 + (c & 1), j = j0 + ((c >> 1) & 1), k = k0 + ((c >> 2) & 1);
        const double w = ((c & 1) ? fx : 1.0 - fx) * (((c >> 1) & 1) ? fy : 1.0 - fy) *
            (((c >> 2) & 1) ? fz : 1.0 - fz);
        const bool inside = i >= 0 && j >= 0 && k >= 0 && i < f.nx && j < f.ny && k < f.nz;
        cells[c] = inside ? static_cast<int>(cellIndex(f, i, j, k)) : -1;
        weights[c] = inside ? w : 0.0;
    }
}

// Transport mass of a grain: dry carrier plus the water it holds (0 when dry).
double grainMass(const FluidParticles& p, std::size_t g) {
    const double water = g < p.pore_water_mass_kg.size() ? p.pore_water_mass_kg[g] : 0.0;
    return double(p.rest_mass_kg[g]) * p.mass_fraction[g] + water;
}

bool isGranular(const FluidParticles& p, std::size_t i, bool legacy_granular) {
    const auto model = i < p.constitutive_model.size()
        ? static_cast<MatterConstitutiveModel>(p.constitutive_model[i])
        : MatterConstitutiveModel::Auto;
    return model == MatterConstitutiveModel::Granular ||
        (model == MatterConstitutiveModel::Auto && legacy_granular);
}

} // namespace

double matterGrainDragCoefficient(double diameter, double relative_speed, double density,
                                  double viscosity, double voidage) {
    if (!(diameter > 0.0) || !(density > 0.0) || !(viscosity > 0.0)) {
        return 0.0;
    }
    voidage = std::clamp(voidage, kMinimumVoidage, 1.0);
    // Di Felice (1994): F = 0.5 Cd rho A eps^2 |u_r| u_r eps^-chi,
    // Cd = (0.63 + 4.8 / sqrt(Re))^2, Re = rho eps d |u_r| / mu,
    // chi = 3.7 - 0.65 exp(-(1.5 - log10 Re)^2 / 2).
    // At Re -> 0, Cd |u_r| stays finite (Stokes-like 9 mu d), so the speed is
    // floored through Re instead of dividing by a vanishing |u_r|.
    const double reynolds = std::max(density * voidage * diameter *
        std::max(relative_speed, 0.0) / viscosity, 1e-6);
    const double speed = reynolds * viscosity / (density * voidage * diameter);
    const double drag = .63 + 4.8 / std::sqrt(reynolds);
    const double log_re = std::log10(reynolds);
    const double chi = 3.7 - .65 * std::exp(-.5 * (1.5 - log_re) * (1.5 - log_re));
    const double area = .25 * kPi * diameter * diameter;
    return .5 * drag * drag * density * area * std::pow(voidage, 2.0 - chi) * speed;
}

bool buildMatterGrainLiquidField(const FluidParticles& liquid, const FluidParticles& grains,
    float grain_radius, FluidChemistryPreset chemistry_preset, const Vec3& origin,
    int nx, int ny, int nz, float h, MatterGrainLiquidField& f, std::string& error) {
    if (nx <= 0 || ny <= 0 || nz <= 0 || !(h > 0.0f) || !finite(origin)) {
        error = "grain liquid coupling needs a valid domain grid";
        return false;
    }
    f.nx = nx;
    f.ny = ny;
    f.nz = nz;
    f.origin = origin;
    f.h = h;
    const std::size_t cells = static_cast<std::size_t>(nx) * ny * nz;
    f.mass.assign(cells, 0.0);
    f.volume.assign(cells, 0.0);
    for (auto& axis : f.momentum) {
        axis.assign(cells, 0.0);
    }
    f.solid.assign(cells, 0.0);
    f.parcel_cell.assign(liquid.size(), -1);
    std::unordered_map<uint32_t, double> density_by_tag;
    for (std::size_t p = 0; p < liquid.size(); ++p) {
        const Vec3& x = liquid.position[p];
        const int i = static_cast<int>(std::floor((x.x - origin.x) / h));
        const int j = static_cast<int>(std::floor((x.y - origin.y) / h));
        const int k = static_cast<int>(std::floor((x.z - origin.z) / h));
        if (!finite(x) || i < 0 || j < 0 || k < 0 || i >= nx || j >= ny || k >= nz) {
            continue;
        }
        const double mass = double(liquid.rest_mass_kg[p]) * liquid.mass_fraction[p];
        if (!std::isfinite(mass) || mass <= 0.0) {
            continue;
        }
        const uint32_t tag = p < liquid.substance_tag.size()
            ? liquid.substance_tag[p] : kSubstanceUntagged;
        auto found = density_by_tag.find(tag);
        if (found == density_by_tag.end()) {
            // fluidParticleRestMassKg(h = 1 m, ppc = 1) is the liquid density.
            const double density = fluidParticleRestMassKg(tag, chemistry_preset, 1.0f, 1,
                MatterConstitutiveModel::Fluid, false);
            found = density_by_tag.emplace(tag, density > 0.0 ? density : 1000.0).first;
        }
        const auto c = cellIndex(f, i, j, k);
        f.parcel_cell[p] = static_cast<int>(c);
        f.mass[c] += mass;
        f.volume[c] += mass / found->second;
        f.momentum[0][c] += mass * liquid.velocity[p].x;
        f.momentum[1][c] += mass * liquid.velocity[p].y;
        f.momentum[2][c] += mass * liquid.velocity[p].z;
    }
    const double sphere = 4.0 / 3.0 * kPi * double(grain_radius) * grain_radius * grain_radius;
    for (const auto& x : grains.position) {
        int c[8];
        double w[8];
        neighbours(f, x, c, w);
        for (int n = 0; n < 8; ++n) {
            if (c[n] >= 0) {
                f.solid[c[n]] += w[n] * sphere;
            }
        }
    }
    error.clear();
    return true;
}

void prepareMatterGrainCoupling(const FluidParticles& grains, const MatterGrainParams& params,
    const Vec3& gravity, float dt, MatterGrainCouplingFrame& frame,
    MatterGrainStepReport& report) {
    const auto& f = frame.field;
    const std::size_t count = grains.size();
    frame.inputs.assign(count, {});
    frame.cells.assign(count * 8, -1);
    frame.shares.assign(count * 8, 0.0f);
    frame.buoyancy_impulse.assign(count, Vec3(0.0f, 0.0f, 0.0f));
    frame.density.assign(count, 0.0f);
    frame.submerged.assign(count, 0.0f);
    report.coupled_grains = 0;
    report.max_drag_coefficient = 0.0f;
    report.max_submerged_fraction = 0.0f;
    report.buoyancy_impulse = Vec3(0.0f, 0.0f, 0.0f);
    if (!params.fluid_coupling) {
        return;
    }
    const double r = params.radius_m;
    const double sphere = 4.0 / 3.0 * kPi * r * r * r;
    const double cell_volume = double(f.h) * f.h * f.h;
    for (std::size_t g = 0; g < count; ++g) {
        int c[8];
        double w[8];
        neighbours(f, grains.position[g], c, w);
        double weight = 0.0, liquid_mass = 0.0, liquid_volume = 0.0, solid = 0.0;
        double lump = 0.0, lump_momentum[3] = {};
        double share[8] = {};
        for (int n = 0; n < 8; ++n) {
            if (c[n] < 0) {
                continue;
            }
            weight += w[n];
            liquid_mass += w[n] * f.mass[c[n]];
            liquid_volume += w[n] * f.volume[c[n]];
            solid += w[n] * f.solid[c[n]];
            // This grain's share of the cell's liquid: its trilinear weight,
            // scaled down once the grains overlapping the cell hold more than
            // one grain volume. sum_i w_i V / max(Vs, V) = Vs / max(Vs, V) <= 1,
            // so the shares of a cell never exceed its liquid mass, and a lone
            // grain sees exactly the trilinearly sampled liquid.
            const double grains_in_cell = std::max(f.solid[c[n]], sphere);
            if (f.mass[c[n]] > 0.0 && w[n] > 0.0) {
                share[n] = w[n] * sphere / grains_in_cell * f.mass[c[n]];
                lump += share[n];
                for (int axis = 0; axis < 3; ++axis) {
                    lump_momentum[axis] += share[n] / f.mass[c[n]] * f.momentum[axis][c[n]];
                }
            }
        }
        if (weight <= 0.0 || lump <= 0.0 || liquid_volume <= 0.0) {
            continue;
        }
        const double alpha = liquid_volume / (weight * cell_volume);
        const double voidage = std::clamp(1.0 - solid / (weight * cell_volume),
            kMinimumVoidage, 1.0);
        const double submerged = std::clamp(alpha / voidage, 0.0, 1.0);
        const double density = liquid_mass / liquid_volume;
        const double grain_mass = grainMass(grains, g);
        if (!(grain_mass > 0.0) || submerged <= 0.0) {
            continue;
        }
        const Vec3 lump_velocity(static_cast<float>(lump_momentum[0] / lump),
            static_cast<float>(lump_momentum[1] / lump), static_cast<float>(lump_momentum[2] / lump));
        const double relative = (lump_velocity - grains.velocity[g]).length();
        const double beta = submerged * matterGrainDragCoefficient(2.0 * r, relative, density,
            params.drag_viscosity_pa_s, voidage);
        // Archimedes, hydrostatic: the displaced liquid's weight, upward.
        const double buoyancy = submerged * density * sphere / grain_mass;
        auto& in = frame.inputs[g];
        frame.density[g] = static_cast<float>(density);
        frame.submerged[g] = static_cast<float>(submerged);
        in.lump_velocity = lump_velocity;
        in.drag_coefficient = static_cast<float>(beta);
        in.lump_mass_kg = static_cast<float>(lump);
        in.buoyancy_acceleration = gravity * static_cast<float>(-buoyancy);
        frame.buoyancy_impulse[g] = in.buoyancy_acceleration * static_cast<float>(grain_mass * dt);
        report.buoyancy_impulse = report.buoyancy_impulse + frame.buoyancy_impulse[g];
        for (int n = 0; n < 8; ++n) {
            frame.cells[g * 8 + n] = share[n] > 0.0 ? c[n] : -1;
            frame.shares[g * 8 + n] = static_cast<float>(share[n] / lump);
        }
        ++report.coupled_grains;
        report.max_drag_coefficient = std::max(report.max_drag_coefficient, in.drag_coefficient);
        report.max_submerged_fraction = std::max(report.max_submerged_fraction,
            static_cast<float>(submerged));
    }
}

void applyMatterGrainLiquidReaction(FluidParticles& liquid, const MatterGrainCouplingFrame& frame,
    const std::vector<MatterGrainCouplingOutput>& drag, MatterGrainStepReport& report) {
    const auto& f = frame.field;
    const std::size_t count = frame.inputs.size();
    std::vector<double> impulse[3];
    for (auto& axis : impulse) {
        axis.assign(f.mass.size(), 0.0);
    }
    double grain_gain[3] = {}, unmatched = 0.0;
    report.drag_impulse = Vec3(0.0f, 0.0f, 0.0f);
    for (std::size_t g = 0; g < count && g < drag.size(); ++g) {
        const Vec3 gain = drag[g].drag_impulse + frame.buoyancy_impulse[g];
        report.drag_impulse = report.drag_impulse + drag[g].drag_impulse;
        grain_gain[0] += gain.x;
        grain_gain[1] += gain.y;
        grain_gain[2] += gain.z;
        double placed = 0.0;
        for (int n = 0; n < 8; ++n) {
            const int c = frame.cells[g * 8 + n];
            const double s = frame.shares[g * 8 + n];
            if (c < 0 || s <= 0.0) {
                continue;
            }
            impulse[0][c] -= s * gain.x;
            impulse[1][c] -= s * gain.y;
            impulse[2][c] -= s * gain.z;
            placed += s;
        }
        if (placed <= 0.0) {
            unmatched += gain.length();
        }
    }
    double liquid_gain[3] = {};
    for (std::size_t p = 0; p < liquid.size() && p < f.parcel_cell.size(); ++p) {
        const int c = f.parcel_cell[p];
        if (c < 0 || f.mass[c] <= 0.0) {
            continue;
        }
        const double mass = double(liquid.rest_mass_kg[p]) * liquid.mass_fraction[p];
        Vec3 dv(static_cast<float>(impulse[0][c] / f.mass[c]),
            static_cast<float>(impulse[1][c] / f.mass[c]),
            static_cast<float>(impulse[2][c] / f.mass[c]));
        liquid.velocity[p] = liquid.velocity[p] + dv;
        liquid_gain[0] += mass * dv.x;
        liquid_gain[1] += mass * dv.y;
        liquid_gain[2] += mass * dv.z;
    }
    report.liquid_reaction = Vec3(static_cast<float>(liquid_gain[0]),
        static_cast<float>(liquid_gain[1]), static_cast<float>(liquid_gain[2]));
    report.momentum_residual = std::sqrt(
        (grain_gain[0] + liquid_gain[0]) * (grain_gain[0] + liquid_gain[0]) +
        (grain_gain[1] + liquid_gain[1]) * (grain_gain[1] + liquid_gain[1]) +
        (grain_gain[2] + liquid_gain[2]) * (grain_gain[2] + liquid_gain[2]));
    report.unmatched_impulse = unmatched;
}

void applyMatterGrainPorosity(FluidSim::FluidGrid& grid, const FluidParticles& grains,
    float grain_radius, bool keep_collider_weights, float minimum_voidage,
    MatterGrainPorosityBackup& backup, MatterGrainStepReport& report) {
    const int nx = grid.nx, ny = grid.ny, nz = grid.nz;
    const float h = grid.voxel_size;
    report.porous_cells = 0;
    report.max_solid_fraction = 0.0f;
    if (nx <= 0 || ny <= 0 || nz <= 0 || !(h > 0.0f)) {
        return;
    }
    const std::size_t cells = static_cast<std::size_t>(nx) * ny * nz;
    const std::size_t faces[3] = {static_cast<std::size_t>(nx + 1) * ny * nz,
        static_cast<std::size_t>(nx) * (ny + 1) * nz, static_cast<std::size_t>(nx) * ny * (nz + 1)};
    backup.active = true;
    backup.u_weight = grid.u_weight;
    backup.v_weight = grid.v_weight;
    backup.w_weight = grid.w_weight;
    backup.solid_vel = grid.solid_vel;
    backup.weights_init = grid.collider_weights_init;
    backup.weights_sig = grid.collider_weights_sig;

    MatterGrainLiquidField f;
    f.nx = nx;
    f.ny = ny;
    f.nz = nz;
    f.origin = grid.origin;
    f.h = h;
    std::vector<double> solid(cells, 0.0), momentum[3];
    for (auto& axis : momentum) {
        axis.assign(cells, 0.0);
    }
    const double sphere = 4.0 / 3.0 * kPi * double(grain_radius) * grain_radius * grain_radius;
    for (std::size_t g = 0; g < grains.size(); ++g) {
        int c[8];
        double w[8];
        neighbours(f, grains.position[g], c, w);
        for (int n = 0; n < 8; ++n) {
            if (c[n] < 0) {
                continue;
            }
            const double v = w[n] * sphere;
            solid[c[n]] += v;
            momentum[0][c[n]] += v * grains.velocity[g].x;
            momentum[1][c[n]] += v * grains.velocity[g].y;
            momentum[2][c[n]] += v * grains.velocity[g].z;
        }
    }
    const double cell_volume = double(h) * h * h;
    const double cap = 1.0 - std::clamp(double(minimum_voidage), 0.05, 1.0);
    std::vector<float> fraction(cells, 0.0f);
    if (grid.solid_vel.size() != cells) {
        grid.solid_vel.assign(cells, Vec3(0.0f, 0.0f, 0.0f));
    }
    for (std::size_t c = 0; c < cells; ++c) {
        if (solid[c] <= 0.0) {
            continue;
        }
        fraction[c] = static_cast<float>(std::min(solid[c] / cell_volume, cap));
        report.max_solid_fraction = std::max(report.max_solid_fraction, fraction[c]);
        ++report.porous_cells;
        // Collider cells keep their wall velocity; the grains' mean velocity
        // is only read by the porous divergence at non-solid cells.
        if (c < grid.solid.size() && grid.solid[c]) {
            continue;
        }
        grid.solid_vel[c] = Vec3(static_cast<float>(momentum[0][c] / solid[c]),
            static_cast<float>(momentum[1][c] / solid[c]),
            static_cast<float>(momentum[2][c] / solid[c]));
    }
    std::vector<uint8_t>* weights[3] = {&grid.u_weight, &grid.v_weight, &grid.w_weight};
    for (int axis = 0; axis < 3; ++axis) {
        if (!keep_collider_weights || weights[axis]->size() != faces[axis]) {
            weights[axis]->assign(faces[axis], 255u);
        }
    }
    const auto cellFraction = [&](int i, int j, int k) -> float {
        if (i < 0 || j < 0 || k < 0 || i >= nx || j >= ny || k >= nz) {
            return -1.0f;  // outside: the face takes the in-grid side only
        }
        return fraction[(static_cast<std::size_t>(k) * ny + j) * nx + i];
    };
    const auto closeFace = [&](uint8_t& weight, float lo, float hi) {
        const float phi = lo < 0.0f ? hi : hi < 0.0f ? lo : .5f * (lo + hi);
        if (phi > 0.0f) {
            weight = static_cast<uint8_t>(std::lround(weight * (1.0f - phi)));
        }
    };
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i <= nx; ++i) {
                closeFace(grid.u_weight[grid.velXIndex(i, j, k)],
                    cellFraction(i - 1, j, k), cellFraction(i, j, k));
            }
        }
    }
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                closeFace(grid.v_weight[grid.velYIndex(i, j, k)],
                    cellFraction(i, j - 1, k), cellFraction(i, j, k));
            }
        }
    }
    for (int k = 0; k <= nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                closeFace(grid.w_weight[grid.velZIndex(i, j, k)],
                    cellFraction(i, j, k - 1), cellFraction(i, j, k));
            }
        }
    }
}

void restoreMatterGrainPorosity(FluidSim::FluidGrid& grid, MatterGrainPorosityBackup& backup) {
    if (!backup.active) {
        return;
    }
    grid.u_weight = std::move(backup.u_weight);
    grid.v_weight = std::move(backup.v_weight);
    grid.w_weight = std::move(backup.w_weight);
    grid.solid_vel = std::move(backup.solid_vel);
    grid.collider_weights_init = backup.weights_init;
    grid.collider_weights_sig = backup.weights_sig;
    backup.active = false;
}

bool applyMatterGrainLiquidAccelerationForce(const FluidParticles& liquid,
    const std::vector<Vec3>& start_velocity, const std::vector<uint64_t>& start_id,
    const FluidParticles& grains, float grain_radius, const Vec3& gravity, float dt,
    MatterGrainCouplingFrame& frame, MatterGrainStepReport& report, std::string& error) {
    const auto& f = frame.field;
    const std::size_t cells = f.mass.size();
    report.pressure_impulse = Vec3(0.0f, 0.0f, 0.0f);
    report.pressure_force = false;
    if (start_velocity.size() != liquid.size() || start_id.size() != liquid.size() ||
        liquid.particle_id.size() != liquid.size() || f.parcel_cell.size() != liquid.size() ||
        frame.inputs.size() != grains.size() || !(dt > 0.0f)) {
        error = "grain pressure force: liquid subset changed size during its step";
        return false;
    }
    if (!std::equal(start_id.begin(), start_id.end(), liquid.particle_id.begin())) {
        error = "grain pressure force: liquid parcel identities changed during its step";
        return false;
    }
    // Per cell: liquid mass and mass-weighted velocity change of its parcels.
    std::vector<double> mass(cells, 0.0), change[3];
    for (auto& axis : change) {
        axis.assign(cells, 0.0);
    }
    for (std::size_t p = 0; p < liquid.size(); ++p) {
        const int c = f.parcel_cell[p];
        if (c < 0) {
            continue;
        }
        const double m = double(liquid.rest_mass_kg[p]) * liquid.mass_fraction[p];
        const Vec3 dv = liquid.velocity[p] - start_velocity[p];
        if (!(m > 0.0) || !finite(dv)) {
            continue;
        }
        mass[c] += m;
        change[0][c] += m * dv.x;
        change[1][c] += m * dv.y;
        change[2][c] += m * dv.z;
    }
    report.pressure_force = true;
    report.buoyancy_impulse = Vec3(0.0f, 0.0f, 0.0f);
    const double sphere = 4.0 / 3.0 * kPi * double(grain_radius) * grain_radius * grain_radius;
    for (std::size_t g = 0; g < grains.size(); ++g) {
        auto& in = frame.inputs[g];
        frame.buoyancy_impulse[g] = Vec3(0.0f, 0.0f, 0.0f);
        in.buoyancy_acceleration = Vec3(0.0f, 0.0f, 0.0f);
        if (in.lump_mass_kg <= 0.0f || frame.density[g] <= 0.0f) {
            continue;
        }
        int c[8];
        double w[8];
        neighbours(f, grains.position[g], c, w);
        double weight = 0.0, a[3] = {};
        for (int n = 0; n < 8; ++n) {
            if (c[n] < 0 || mass[c[n]] <= 0.0) {
                continue;
            }
            for (int axis = 0; axis < 3; ++axis) {
                a[axis] += w[n] * change[axis][c[n]] / (mass[c[n]] * dt);
            }
            weight += w[n];
        }
        if (weight <= 0.0) {
            continue;
        }
        const Vec3 acceleration(static_cast<float>(a[0] / weight),
            static_cast<float>(a[1] / weight), static_cast<float>(a[2] / weight));
        const double displaced = double(frame.submerged[g]) * frame.density[g] * sphere;
        const Vec3 force = (acceleration - gravity) * static_cast<float>(displaced);
        in.buoyancy_acceleration = force * static_cast<float>(1.0 / grainMass(grains, g));
        frame.buoyancy_impulse[g] = force * dt;
        report.pressure_impulse = report.pressure_impulse + frame.buoyancy_impulse[g];
    }
    report.buoyancy_impulse = report.pressure_impulse;
    error.clear();
    return true;
}

void exchangeMatterGrainWater(FluidParticles& liquid, FluidParticles& grains,
    const MatterGrainCouplingFrame& frame, const MatterGrainParams& params, float dt,
    MatterGrainStepReport& report) {
    report.wet_grains = params.wet_grains;
    report.absorbed_kg = 0.0;
    report.evaporated_kg = 0.0;
    if (!params.wet_grains) {
        return;
    }
    const auto& f = frame.field;
    const std::size_t cells = f.mass.size();
    const std::size_t count = grains.size();
    constexpr double kWaterHeat = 4186.0;
    const double capacity = matterGrainWaterCapacityKg(params);
    // Current liquid per cell (the drag reaction already changed velocities).
    std::vector<double> mass(cells, 0.0), momentum[3];
    for (auto& axis : momentum) {
        axis.assign(cells, 0.0);
    }
    double liquid_before = 0.0, grain_before = 0.0;
    for (std::size_t p = 0; p < liquid.size(); ++p) {
        const double m = double(liquid.rest_mass_kg[p]) * liquid.mass_fraction[p];
        liquid_before += m;
        const int c = p < f.parcel_cell.size() ? f.parcel_cell[p] : -1;
        if (c < 0 || m <= 0.0) {
            continue;
        }
        mass[c] += m;
        momentum[0][c] += m * liquid.velocity[p].x;
        momentum[1][c] += m * liquid.velocity[p].y;
        momentum[2][c] += m * liquid.velocity[p].z;
    }
    for (std::size_t g = 0; g < count; ++g) {
        grain_before += grains.pore_water_mass_kg[g];
    }
    // Requests along each grain's lump shares, then one scale per cell so no
    // cell gives more than half its liquid in a frame.
    const bool absorbing = params.absorption_rate_per_s > 0.0f && frame.cells.size() == count * 8;
    const double uptake = 1.0 - std::exp(-double(params.absorption_rate_per_s) * dt);
    std::vector<double> want(count, 0.0), demand(cells, 0.0);
    if (absorbing) {
        for (std::size_t g = 0; g < count; ++g) {
            const double free = capacity - grains.pore_water_mass_kg[g];
            const double s = g < frame.submerged.size() ? frame.submerged[g] : 0.0;
            if (free <= 0.0 || s <= 0.0) {
                continue;
            }
            want[g] = free * uptake * s;
            for (int n = 0; n < 8; ++n) {
                const int c = frame.cells[g * 8 + n];
                if (c >= 0) {
                    demand[c] += want[g] * frame.shares[g * 8 + n];
                }
            }
        }
    }
    std::vector<double> scale(cells, 0.0), taken(cells, 0.0);
    for (std::size_t c = 0; c < cells; ++c) {
        if (demand[c] > 0.0 && mass[c] > 0.0) {
            scale[c] = std::min(1.0, .5 * mass[c] / demand[c]);
        }
    }
    report.wet_grain_count = 0;
    report.max_grain_saturation = 0.0f;
    report.grain_water_kg = 0.0;
    for (std::size_t g = 0; g < count; ++g) {
        double grant = 0.0, gained[3] = {};
        if (want[g] > 0.0) {
            for (int n = 0; n < 8; ++n) {
                const int c = frame.cells[g * 8 + n];
                if (c < 0 || scale[c] <= 0.0) {
                    continue;
                }
                const double piece = want[g] * frame.shares[g * 8 + n] * scale[c];
                taken[c] += piece;
                grant += piece;
                for (int axis = 0; axis < 3; ++axis) {
                    gained[axis] += piece * momentum[axis][c] / mass[c];
                }
            }
        }
        double water = grains.pore_water_mass_kg[g];
        if (grant > 0.0) {
            const double before = grainMass(grains, g);
            const Vec3& v = grains.velocity[g];
            grains.velocity[g] = Vec3(static_cast<float>((before * v.x + gained[0]) / (before + grant)),
                static_cast<float>((before * v.y + gained[1]) / (before + grant)),
                static_cast<float>((before * v.z + gained[2]) / (before + grant)));
            water += grant;
            report.absorbed_kg += grant;
        }
        if (params.drying_rate_per_s > 0.0f && water > 0.0) {
            const double lost = water * (1.0 - std::exp(-double(params.drying_rate_per_s) * dt));
            water -= lost;
            report.evaporated_kg += lost;
        }
        grains.pore_water_mass_kg[g] = static_cast<float>(std::max(water, 0.0));
        grains.pore_capacity_kg[g] = static_cast<float>(capacity);
        // Held water stores its sensible heat at the grain's temperature so
        // energy and mass vanish together (the ledger's ownership rule).
        const double kelvin = g < grains.temperature.size()
            ? std::max(1.0, double(grains.temperature[g])) : 293.15;
        grains.pore_water_energy_j[g] = static_cast<float>(grains.pore_water_mass_kg[g] *
            kWaterHeat * kelvin);
        if (grains.pore_water_mass_kg[g] > 0.0f) {
            ++report.wet_grain_count;
        }
        report.grain_water_kg += grains.pore_water_mass_kg[g];
        report.max_grain_saturation = std::max(report.max_grain_saturation,
            static_cast<float>(capacity > 0.0 ? grains.pore_water_mass_kg[g] / capacity : 0.0));
    }
    // Every parcel of a cell gives the same fraction, so the cell's momentum
    // leaves with its mean velocity -- what the grains received.
    double liquid_after = 0.0;
    for (std::size_t p = 0; p < liquid.size(); ++p) {
        const int c = p < f.parcel_cell.size() ? f.parcel_cell[p] : -1;
        if (c >= 0 && taken[c] > 0.0 && mass[c] > 0.0) {
            liquid.mass_fraction[p] = static_cast<float>(
                double(liquid.mass_fraction[p]) * (1.0 - taken[c] / mass[c]));
        }
        liquid_after += double(liquid.rest_mass_kg[p]) * liquid.mass_fraction[p];
    }
    report.water_balance_error_kg = (liquid_after + report.grain_water_kg + report.evaporated_kg) -
        (liquid_before + grain_before);
}

bool partitionMatterGrainOwners(const FluidParticles& p, bool legacy_granular,
    bool wet_grains, std::vector<std::size_t>& liquid, std::vector<std::size_t>& grains,
    std::string& error) {
    liquid.clear();
    grains.clear();
    const std::size_t count = p.size();
    if (p.particle_id.size() != count || p.constitutive_model.size() != count ||
        p.pore_water_mass_kg.size() != count) {
        error = "grain owner partition: particle sidecars are not sized";
        return false;
    }
    for (std::size_t i = 0; i < count; ++i) {
        const auto model = static_cast<MatterConstitutiveModel>(p.constitutive_model[i]);
        if (isFrozenParticle(p, i) || model == MatterConstitutiveModel::Elastic) {
            error = "grain domain carries a frozen/elastic carrier; no transport owner accepts it";
            return false;
        }
        if (isGranular(p, i, legacy_granular)) {
            if (model == MatterConstitutiveModel::Auto) {
                error = "grain domain: Auto carriers in a legacy-granular domain have no "
                    "explicit owner; emit Granular or Fluid";
                return false;
            }
            if (p.pore_water_mass_kg[i] != 0.0f && !wet_grains) {
                error = "grain domain: a grain holds water but wet_grains is off";
                return false;
            }
            grains.push_back(i);
        } else {
            liquid.push_back(i);
        }
    }
    // Identity order is only the deterministic default; the coordinator
    // re-orders by cell (orderMatterGrainsByCell) and the runtime carries the
    // contact history across any order by identity.
    std::stable_sort(grains.begin(), grains.end(), [&](std::size_t a, std::size_t b) {
        return p.particle_id[a] < p.particle_id[b];
    });
    error.clear();
    return true;
}

void orderMatterGrainsByCell(const FluidParticles& particles, const Vec3& origin, float cell,
    std::vector<std::size_t>& grains) {
    if (!(cell > 0.0f) || grains.size() < 2) {
        return;
    }
    // 21 bits per axis: every cell of a 100000-grain domain fits.
    const auto spread = [](uint64_t v) {
        v &= 0x1fffffull;
        v = (v | v << 32) & 0x1f00000000ffffull;
        v = (v | v << 16) & 0x1f0000ff0000ffull;
        v = (v | v << 8) & 0x100f00f00f00f00full;
        v = (v | v << 4) & 0x10c30c30c30c30c3ull;
        v = (v | v << 2) & 0x1249249249249249ull;
        return v;
    };
    std::vector<std::pair<uint64_t, std::size_t>> keyed(grains.size());
    for (std::size_t k = 0; k < grains.size(); ++k) {
        const Vec3& p = particles.position[grains[k]];
        const auto axis = [&](float value, float low) {
            return static_cast<uint64_t>(std::clamp(std::floor((value - low) / cell), 0.0f, 2097151.0f));
        };
        keyed[k] = {spread(axis(p.x, origin.x)) | spread(axis(p.y, origin.y)) << 1 |
            spread(axis(p.z, origin.z)) << 2, grains[k]};
    }
    std::sort(keyed.begin(), keyed.end(), [&](const auto& a, const auto& b) {
        return a.first != b.first ? a.first < b.first
            : particles.particle_id[a.second] < particles.particle_id[b.second];
    });
    for (std::size_t k = 0; k < grains.size(); ++k) {
        grains[k] = keyed[k].second;
    }
}

FluidParticles selectMatterParticles(const FluidParticles& particles,
                                     const std::vector<std::size_t>& order) {
    FluidParticles out = particles;
    std::vector<uint8_t> drop(particles.size(), 0);
    for (std::size_t i = order.size(); i < drop.size(); ++i) {
        drop[i] = 1;
    }
    out.compact(drop);
    for (std::size_t k = 0; k < order.size(); ++k) {
        out.copyParticleFrom(k, particles, order[k]);
    }
    return out;
}

bool mergeMatterGrainOwners(FluidParticles& particles, const FluidParticles& liquid,
                            const FluidParticles& grains, std::string& error) {
    const std::size_t nl = liquid.size(), ng = grains.size();
    if (nl + ng != particles.size()) {
        error = "grain/liquid merge: an owner changed the carrier count";
        return false;
    }
    std::unordered_set<uint64_t> identities(particles.particle_id.begin(),
        particles.particle_id.end());
    for (const auto* subset : {&liquid, &grains}) {
        for (const auto id : subset->particle_id) {
            if (identities.erase(id) != 1) {
                error = "grain/liquid merge: identity set changed";
                return false;
            }
        }
    }
    for (std::size_t k = 0; k < nl; ++k) {
        particles.copyParticleFrom(k, liquid, k);
    }
    for (std::size_t k = 0; k < ng; ++k) {
        particles.copyParticleFrom(nl + k, grains, k);
    }
    error.clear();
    return true;
}

} // namespace RayTrophiSim::Fluid
