// Grain <-> liquid coupling contract (MatterGrainCoupling.cpp), no GPU.
// Mirrors the shader's implicit pair update to check the frame budget.
#include "Fluid/MatterGrainCoupling.h"
#include "Fluid/FluidParticles.h"

#include <cassert>
#include <cmath>
#include <cstdio>

using namespace RayTrophiSim::Fluid;

namespace {

void emit(FluidParticles& p, const Vec3& x, const Vec3& v, float mass, MatterConstitutiveModel m) {
    p.emit(x, v, 0.0f, 0.0f, 0u, nullptr, nullptr, mass, m);
}

// The step shader's drag block for one grain and one substep.
void pair(Vec3& v, MatterGrainCouplingInput& in, Vec3& impulse, float m, float dt) {
    const float M = in.lump_mass_kg, beta = in.drag_coefficient;
    const Vec3 relative = (in.lump_velocity - v) * (1.0f / (1.0f + dt * beta * (1.0f / m + 1.0f / M)));
    const Vec3 settled = (v * m + (in.lump_velocity - relative) * M) * (1.0f / (m + M));
    in.lump_velocity = settled + relative;
    impulse = impulse + (settled - v) * m;
    v = settled;
}

} // namespace

int main() {
    // Di Felice limits: Stokes-like at Re -> 0 (9 mu d vs 3 pi mu d = 9.42),
    // and Cd -> 0.4 (0.63^2) for a sphere at high Re in free liquid.
    const double stokes = matterGrainDragCoefficient(.05, 0.0, 1000.0, 1e-3, 1.0);
    assert(stokes > 8.0 * 1e-3 * .05 && stokes < 10.0 * 1e-3 * .05);
    const double fast = matterGrainDragCoefficient(.05, 2.0, 1000.0, 1e-3, 1.0);
    const double area = .25 * 3.14159265 * .05 * .05;
    const double cd = fast / (.5 * 1000.0 * area * 2.0);
    assert(cd > .35 && cd < .5);
    // Denser surroundings (voidage .4) drag harder.
    assert(matterGrainDragCoefficient(.05, .1, 1000.0, 1e-3, .4) >
           matterGrainDragCoefficient(.05, .1, 1000.0, 1e-3, 1.0));

    // A still 0.4 m water block on a 0.05 m grid, one grain at mid depth and
    // one above the surface.
    FluidParticles liquid, grains;
    const float h = .05f;
    const int n = 8;
    for (int k = 0; k < n; ++k)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                for (int s = 0; s < 8; ++s) {
                    const Vec3 x((i + .25f + .5f * (s & 1)) * h, (j + .25f + .5f * ((s >> 1) & 1)) * h,
                                 (k + .25f + .5f * ((s >> 2) & 1)) * h);
                    emit(liquid, x, Vec3(0.0f, 0.0f, 0.0f), 1000.0f * h * h * h / 8.0f,
                         MatterConstitutiveModel::Fluid);
                }
    const float r = .025f;
    const float grain_mass = 1600.0f / .6f * 4.18879f * r * r * r;
    emit(grains, Vec3(.2f, .2f, .2f), Vec3(0.0f, -1.0f, 0.0f), grain_mass, MatterConstitutiveModel::Granular);
    emit(grains, Vec3(.2f, .6f, .2f), Vec3(0.0f, -1.0f, 0.0f), grain_mass, MatterConstitutiveModel::Granular);

    MatterGrainCouplingFrame frame;
    std::string error;
    // Grid 8 x 16 x 8: the upper half is dry.
    bool ok = buildMatterGrainLiquidField(liquid, grains, r, static_cast<FluidChemistryPreset>(0),
        Vec3(0.0f, 0.0f, 0.0f), n, 2 * n, n, h, frame.field, error);
    assert(ok);
    MatterGrainParams params;
    params.radius_m = r;
    MatterGrainStepReport report;
    const float dt = 1.0f / 60.0f;
    prepareMatterGrainCoupling(grains, params, Vec3(0.0f, -9.81f, 0.0f), dt, frame, report);
    assert(report.coupled_grains == 1);
    const auto& wet = frame.inputs[0];
    const auto& dry = frame.inputs[1];
    assert(dry.drag_coefficient == 0.0f && dry.lump_mass_kg == 0.0f);
    // Fully submerged: buoyancy acceleration = rho_w V g / m upward.
    const float expected = 1000.0f * 4.18879f * r * r * r * 9.81f / grain_mass;
    std::printf("buoyancy %.4f expected %.4f beta %.5f lump %.4f kg\n",
        wet.buoyancy_acceleration.y, expected, wet.drag_coefficient, wet.lump_mass_kg);
    assert(std::abs(wet.buoyancy_acceleration.y - expected) < .02f * expected);
    // A lone grain's lump is the trilinearly sampled liquid of one cell.
    assert(std::abs(wet.lump_mass_kg - 1000.0f * h * h * h) < 1e-3f * 1000.0f * h * h * h);

    // Run the shader's pair update for a frame of substeps; return the drag.
    auto input = frame.inputs;
    std::vector<MatterGrainCouplingOutput> drag(2);
    Vec3 v = grains.velocity[0];
    const int substeps = 64;
    for (int s = 0; s < substeps; ++s) {
        pair(v, input[0], drag[0].drag_impulse, grain_mass, dt / substeps);
    }
    assert(v.y > -1.0f);  // drag slowed the sinking grain
    // Momentum: grain gain + lump loss == 0.
    const Vec3 lump_loss = (input[0].lump_velocity - frame.inputs[0].lump_velocity) * wet.lump_mass_kg;
    assert((drag[0].drag_impulse + lump_loss).length() < 1e-6f);

    double before[3] = {};
    for (std::size_t p = 0; p < liquid.size(); ++p) {
        before[1] += double(liquid.rest_mass_kg[p]) * liquid.velocity[p].y;
    }
    applyMatterGrainLiquidReaction(liquid, frame, drag, report);
    double after = 0.0;
    for (std::size_t p = 0; p < liquid.size(); ++p) {
        after += double(liquid.rest_mass_kg[p]) * liquid.velocity[p].y;
    }
    const double grain_gain = drag[0].drag_impulse.y + frame.buoyancy_impulse[0].y;
    std::printf("liquid gain %.6e grain gain %.6e residual %.3e\n",
        after - before[1], grain_gain, report.momentum_residual);
    assert(std::abs((after - before[1]) + grain_gain) < 1e-6 * std::abs(grain_gain) + 1e-9);
    assert(report.momentum_residual < 1e-6);
    assert(report.unmatched_impulse == 0.0);

    // Owner partition: identity order for grains, liquid order kept.
    FluidParticles mixed;
    emit(mixed, Vec3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 0.0f, 0.0f), 1.0f, MatterConstitutiveModel::Granular);
    emit(mixed, Vec3(1.0f, 0.0f, 0.0f), Vec3(0.0f, 0.0f, 0.0f), 1.0f, MatterConstitutiveModel::Fluid);
    emit(mixed, Vec3(2.0f, 0.0f, 0.0f), Vec3(0.0f, 0.0f, 0.0f), 1.0f, MatterConstitutiveModel::Granular);
    mixed.removeSwap(0);  // grain id 3 now precedes nothing; order: [grain3, fluid2]
    emit(mixed, Vec3(3.0f, 0.0f, 0.0f), Vec3(0.0f, 0.0f, 0.0f), 1.0f, MatterConstitutiveModel::Fluid);
    std::vector<std::size_t> lo, go;
    ok = partitionMatterGrainOwners(mixed, true, false, lo, go, error);
    assert(ok && lo.size() == 2 && go.size() == 1);
    auto l = selectMatterParticles(mixed, lo);
    auto g = selectMatterParticles(mixed, go);
    assert(l.size() == 2 && g.size() == 1 && g.position[0].x == 2.0f);
    l.velocity[0] = Vec3(0.0f, 5.0f, 0.0f);
    ok = mergeMatterGrainOwners(mixed, l, g, error);
    assert(ok && mixed.size() == 3 && mixed.velocity[0].y == 5.0f && mixed.position[2].x == 2.0f);
    // A changed population is refused, never silently merged.
    l.removeSwap(1);
    assert(!mergeMatterGrainOwners(mixed, l, g, error));
    // B5 porosity: a cell packed with grains closes its faces by its solid
    // fraction (capped at 1 - minimum voidage); restore gives back the grid.
    {
        FluidSim::FluidGrid grid(4, 4, 4, .1f, Vec3(0.0f, 0.0f, 0.0f));
        FluidParticles packed;
        // 8 grains r=.025 centred in cell (1,1,1): weights 1, volume 8*6.5e-5.
        for (int s = 0; s < 8; ++s) {
            emit(packed, Vec3(.15f, .15f, .15f), Vec3(0.0f, -.5f, 0.0f), .1f,
                 MatterConstitutiveModel::Granular);
        }
        MatterGrainPorosityBackup backup;
        MatterGrainStepReport porous_report;
        applyMatterGrainPorosity(grid, packed, .025f, false, .3f, backup, porous_report);
        const float phi = 8.0f * 4.18879f * .025f * .025f * .025f / (.1f * .1f * .1f);
        std::printf("porous cells %zu max phi %.4f expected %.4f\n", porous_report.porous_cells,
            porous_report.max_solid_fraction, phi);
        assert(porous_report.porous_cells == 1);
        assert(std::abs(porous_report.max_solid_fraction - phi) < 1e-4f);
        // The +x face of cell (1,1,1) borders an empty cell: phi/2 closed.
        const int expected = static_cast<int>(std::lround(255.0f * (1.0f - .5f * phi)));
        assert(grid.u_weight[grid.velXIndex(2, 1, 1)] == expected);
        assert(grid.u_weight[grid.velXIndex(3, 1, 1)] == 255);
        assert(std::abs(grid.solid_vel[grid.cellIndex(1, 1, 1)].y + .5f) < 1e-6f);
        restoreMatterGrainPorosity(grid, backup);
        assert(grid.u_weight.empty() || grid.u_weight[grid.velXIndex(2, 1, 1)] == 255);
    }
    // B5 pressure force from the liquid's own acceleration: liquid at rest
    // (no velocity change over the step) gives exactly Archimedes, -rho V g;
    // liquid accelerating upward at g pushes twice as hard; a changed
    // population is rejected instead of guessed.
    {
        MatterGrainCouplingFrame tank;
        assert(buildMatterGrainLiquidField(liquid, grains, r, static_cast<FluidChemistryPreset>(0),
            Vec3(0.0f, 0.0f, 0.0f), n, 2 * n, n, h, tank.field, error));
        MatterGrainStepReport tank_report;
        const Vec3 gravity(0.0f, -9.81f, 0.0f);
        prepareMatterGrainCoupling(grains, params, gravity, dt, tank, tank_report);
        const float archimedes = 1000.0f * 4.18879f * r * r * r * 9.81f / grain_mass;
        assert(applyMatterGrainLiquidAccelerationForce(liquid, liquid.velocity, liquid.particle_id,
            grains, r, gravity, dt, tank, tank_report, error));
        std::printf("pressure accel %.4f archimedes %.4f\n",
            tank.inputs[0].buoyancy_acceleration.y, archimedes);
        assert(tank_report.pressure_force);
        assert(std::abs(tank.inputs[0].buoyancy_acceleration.y - archimedes) < .02f * archimedes);
        assert(std::abs(tank.inputs[0].buoyancy_acceleration.x) < 1e-6f);
        assert(tank.inputs[1].buoyancy_acceleration.length() == 0.0f);  // dry grain
        assert(std::abs(tank_report.pressure_impulse.y - archimedes * grain_mass * dt) <
               .02f * archimedes * grain_mass * dt);
        auto start = liquid.velocity;
        for (auto& v : start) v = v - Vec3(0.0f, 9.81f * dt, 0.0f);
        assert(applyMatterGrainLiquidAccelerationForce(liquid, start, liquid.particle_id,
            grains, r, gravity, dt, tank, tank_report, error));
        std::printf("accelerating liquid %.4f expected %.4f\n",
            tank.inputs[0].buoyancy_acceleration.y, 2.0f * archimedes);
        assert(std::abs(tank.inputs[0].buoyancy_acceleration.y - 2.0f * archimedes) <
               .02f * archimedes);
        auto ids = liquid.particle_id;
        std::swap(ids[0], ids[1]);
        assert(!applyMatterGrainLiquidAccelerationForce(liquid, liquid.velocity, ids,
            grains, r, gravity, dt, tank, tank_report, error));
        assert(!error.empty());
    }
    // B6 absorption: a submerged grain takes water from its cells; total
    // water and momentum are unchanged, the grain picks up the liquid velocity.
    {
        FluidParticles pool = liquid, wet = grains;
        for (auto& v : pool.velocity) v = Vec3(1.0f, 0.0f, 0.0f);
        for (auto& v : wet.velocity) v = Vec3(0.0f, 0.0f, 0.0f);
        MatterGrainCouplingFrame frame2;
        assert(buildMatterGrainLiquidField(pool, wet, r, static_cast<FluidChemistryPreset>(0),
            Vec3(0.0f, 0.0f, 0.0f), n, 2 * n, n, h, frame2.field, error));
        MatterGrainParams wet_params = params;
        wet_params.wet_grains = true;
        MatterGrainStepReport wet_report;
        prepareMatterGrainCoupling(wet, wet_params, Vec3(0.0f, -9.81f, 0.0f), dt, frame2, wet_report);
        auto momentum = [](const FluidParticles& p) {
            double m = 0.0;
            for (std::size_t i = 0; i < p.size(); ++i) {
                m += (double(p.rest_mass_kg[i]) * p.mass_fraction[i] + p.pore_water_mass_kg[i]) *
                    p.velocity[i].x;
            }
            return m;
        };
        const double before = momentum(pool) + momentum(wet);
        exchangeMatterGrainWater(pool, wet, frame2, wet_params, dt, wet_report);
        const double after = momentum(pool) + momentum(wet);
        std::printf("absorbed %.6f kg, water balance error %.3e, momentum %.6e -> %.6e, grain vx %.4f\n",
            wet_report.absorbed_kg, wet_report.water_balance_error_kg, before, after,
            wet.velocity[0].x);
        assert(wet_report.absorbed_kg > 0.0 && wet.pore_water_mass_kg[0] > 0.0f);
        assert(wet.pore_water_mass_kg[1] == 0.0f);  // the dry grain above the pool
        assert(std::abs(wet_report.water_balance_error_kg) < 1e-6);
        assert(std::abs(after - before) < 1e-6);
        assert(wet.velocity[0].x > 0.0f && wet.velocity[0].x < 1.0f);
        assert(wet.pore_water_energy_j[0] > 0.0f && wet.pore_capacity_kg[0] > 0.0f);
        // Drying alone (no liquid frame): mass leaves and is reported.
        wet_params.drying_rate_per_s = 1.0f;
        MatterGrainCouplingFrame none;
        FluidParticles empty;
        const float held = wet.pore_water_mass_kg[0];
        exchangeMatterGrainWater(empty, wet, none, wet_params, 1.0f, wet_report);
        assert(wet.pore_water_mass_kg[0] < held && wet_report.evaporated_kg > 0.0);
        assert(std::abs(wet_report.water_balance_error_kg) < 1e-6);
    }
    std::printf("PASS grain liquid coupling contract\n");
    return 0;
}
