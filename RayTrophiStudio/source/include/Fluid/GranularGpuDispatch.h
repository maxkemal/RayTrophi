#pragma once

#include "FluidParticles.h"
#include "GranularGpuState.h"
#include "GranularConstitutive.h"
#include "../SimulationCompute.h"
#include <algorithm>
#include <cstdint>

namespace RayTrophiSim::Fluid::Granular {


struct alignas(16) StressUpdateConstants {
    uint32_t count=0; float dt=0; float young_modulus=2e5f; float poisson_ratio=0.25f;
    float friction_tangent=0.7002075f; float cohesion=0; float dilatancy_tangent=0.0874887f;
    float tensile_cutoff=0;
    float fracture_strain=0.04f;float damage_rate=6.0f;float healing_rate=0.0f;
    uint32_t rebonding=0;
    float hardening_coefficient=0.0f;
    float max_stored_strain=kGranularMaxStoredStrain;
    uint32_t pad1=0,pad2=0;
};
static_assert(sizeof(StressUpdateConstants)==64);

struct alignas(16) StressP2GConstants {
    int nx=0,ny=0,nz=0,count=0,component=0;
    float ox=0,oy=0,oz=0,h=1,dt=0,inv_density=1.0f/1600.0f;
    uint32_t pad=0;
};
static_assert(sizeof(StressP2GConstants)==48);

struct alignas(16) SettleConstants {
    uint32_t count=0;float dt=0;float contact_damping=8.0f;float sleep_speed=0.03f;
    float pressure_threshold=0.1f;uint32_t pad0=0,pad1=0,pad2=0;
};
static_assert(sizeof(SettleConstants)==32);




struct ElasticStepInfo {
    float requested_young_modulus = 0.0f;
    float effective_young_modulus = 0.0f;
    int required_substeps = 1;
    bool capped = false;
    // Which of the two limits below asked for the substeps, so a subcycle that
    // is too coarse can be attributed instead of guessed at.
    int wave_substeps = 1;
    int strain_substeps = 1;
    float strain_rate = 0.0f;
    // Physical validity, which is NOT a stability question: a material can be
    // perfectly stable and still be too soft to hold its own weight.
    float overburden_pressure = 0.0f;
    float young_modulus_for_load = 0.0f;
    float softening_min = 1.0f;
    float effective_young_for_load = 0.0f;
    bool below_load = false;
};

// ★★★ TWO INDEPENDENT STABILITY LIMITS, AND ONLY ONE OF THEM CONTAINS E.
//
//  1. Elastic wave CFL:  dt < 0.35 h / sqrt(E/rho).  Demands MORE substeps as
//     E goes UP.
//  2. Deformation-gradient CFL: the stress kernel integrates F as
//     (I + dt*C)*F, which needs dt*||C|| << 1.  E does not appear in it at all.
//
// Sizing the subcycle from (1) alone leaves a hole exactly where the material
// is SOFT. Any E at or below rho*(0.35h/dt)^2 -- about 1.1 kPa for h = 0.1 m at
// 24 fps -- returns a single substep at the full frame dt, so F is integrated
// with dt*||C|| of order one. The error is multiplicative, so it does not show
// up as an immediate blow-up: strain accumulates quietly frame over frame and
// then discharges through the kernel's bad_kinematics reset as a velocity
// burst. That cliff is why "under about 1000" behaves differently in kind, not
// in degree, from the values above it.
inline ElasticStepInfo elasticStepInfo(float requested_young_modulus,
                                       float voxel_size, float dt,
                                       float strain_rate = 0.0f,
                                       float overburden_pressure = 0.0f,
                                       float softening_min = 1.0f,
                                       float density = kGranularDensity) {
    ElasticStepInfo out;
    out.requested_young_modulus = std::max(requested_young_modulus, 1.0f);
    const float safe_dt = std::max(dt, 1.0e-6f);
    const float wave_speed_limit = 0.35f * std::max(voxel_size, 1.0e-6f) / safe_dt;
    const float stable_young = std::max(density, 1.0f) * wave_speed_limit * wave_speed_limit;
    out.effective_young_modulus = std::min(out.requested_young_modulus,
                                           std::max(stable_young, 1.0f));
    out.wave_substeps = std::max(1, static_cast<int>(std::ceil(
        std::sqrt(out.requested_young_modulus /
                  std::max(out.effective_young_modulus, 1.0f)))));
    out.strain_rate = std::isfinite(strain_rate) ? std::max(strain_rate, 0.0f) : 0.0f;
    out.strain_substeps = std::max(1, static_cast<int>(std::ceil(
        safe_dt * out.strain_rate / kGranularMaxStrainIncrement)));
    out.required_substeps = std::max(out.wave_substeps, out.strain_substeps);
    out.capped = out.effective_young_modulus + 1.0e-3f < out.requested_young_modulus;
    out.overburden_pressure = std::max(overburden_pressure, 0.0f);
    out.young_modulus_for_load = out.overburden_pressure * kGranularStiffnessLoadRatio;
    // ★ The load gate must judge the stiffness the material ACTUALLY has, not
    // the authored one. A melting body softens past its own weight, and that
    // crossing IS the melt -- reading the authored value here would keep
    // reporting a healthy pile while it slumps. Substep sizing above keeps
    // using the authored value on purpose: it is the maximum over particles,
    // so it errs toward more substeps.
    out.softening_min = std::isfinite(softening_min)
        ? std::clamp(softening_min, 0.0f, 1.0f) : 1.0f;
    out.effective_young_for_load = out.requested_young_modulus * out.softening_min;
    out.below_load = out.overburden_pressure > 0.0f &&
                     out.effective_young_for_load < out.young_modulus_for_load;
    return out;
}

struct LoadMeasurement {
    float strain_rate = 0.0f;       // max ||C||, 1/s
    float column_height = 0.0f;     // vertical extent of the material, m
    float overburden_pressure = 0.0f; // rho*g*h at the bottom layer, Pa
    float softening_min = 1.0f;       // weakest particle's thermal softening
    std::size_t softened_particles = 0;
};

// One pass over the particles for the two numbers that decide whether this
// material can be simulated at all.
//
// strain_rate is the worst-case ||C|| the deformation-gradient integrator will
// meet next step, taken from the affine field the previous G2P downloaded.
// Frobenius bounds the spectral norm from above, so sizing the subcycle with it
// errs toward more substeps rather than fewer.
//
// ★ The reading is ONE FRAME STALE by construction -- C only exists on the host
// after a G2P -- which is exactly why the stress shader carries its own clamp
// as the second line of defence. Neither mechanism alone is enough: the host
// one cannot see a spike forming inside the current frame, and the shader one
// keeps the step finite without making it accurate.
//
// column_height is measured from the MATERIAL, not from the domain box. A tall
// domain holding a thin layer carries a thin layer's load, and warning about
// the box would be warning about geometry the simulation never sees.
inline LoadMeasurement measureLoad(const FluidParticles& particles,
                                   float gravity = 9.81f,
                                   float density = kGranularDensity) {
    LoadMeasurement out;
    const std::size_t n = particles.size();
    float worst_squared = 0.0f;
    float lowest = 0.0f, highest = 0.0f;
    bool have_extent = false;
    const std::size_t affine_n = std::min(n, particles.affine.size());
    for (std::size_t i = 0; i < affine_n; ++i) {
        const AffineC& c = particles.affine[i];
        const float f = c.col0.x * c.col0.x + c.col0.y * c.col0.y + c.col0.z * c.col0.z +
                        c.col1.x * c.col1.x + c.col1.y * c.col1.y + c.col1.z * c.col1.z +
                        c.col2.x * c.col2.x + c.col2.y * c.col2.y + c.col2.z * c.col2.z;
        if (std::isfinite(f) && f > worst_squared) worst_squared = f;
    }
    const std::size_t position_n = std::min(n, particles.position.size());
    for (std::size_t i = 0; i < position_n; ++i) {
        const float y = particles.position[i].y;
        if (!std::isfinite(y)) continue;
        if (!have_extent) { lowest = highest = y; have_extent = true; continue; }
        lowest = std::min(lowest, y);
        highest = std::max(highest, y);
    }
    const std::size_t softening_n = std::min(n, particles.granular_softening.size());
    for (std::size_t i = 0; i < softening_n; ++i) {
        const float v = particles.granular_softening[i];
        if (!std::isfinite(v)) continue;
        out.softening_min = std::min(out.softening_min, std::clamp(v, 0.0f, 1.0f));
        if (v < 0.999f) ++out.softened_particles;
    }
    out.strain_rate = std::sqrt(worst_squared);
    out.column_height = have_extent ? std::max(highest - lowest, 0.0f) : 0.0f;
    // Hydrostatic load at the bottom of that column. The gate it feeds is about
    // the CONSTITUTIVE MODEL, not the timestep: a corotational linear-elastic
    // predictor only means anything while the elastic strain stays small, and
    // holding pressure p needs a volumetric strain of p/K. At E = 1 kPa
    // (K = 667 Pa for nu = 0.25) a 1 m pile would need a volumetric strain near
    // 20 before it stops sinking -- so it never stops. It compacts until det(F)
    // hits the kernel's floor, resets, and falls again.
    out.overburden_pressure =
        out.column_height * std::abs(gravity) * std::max(density, 1.0f);
    return out;
}

inline bool ensureBuffers(SimulationComputeContext& compute, GpuBuffers& b,
                          std::size_t count) {
    if (count == 0) return true;
    if (b.valid() && b.capacity >= count) return true;
    destroy(&compute, b);
    const std::size_t bytes_vec = count * sizeof(Vec3);
    const std::size_t bytes_f32 = count * sizeof(float);
    const std::size_t bytes_u32 = count * sizeof(uint32_t);
    auto make = [&](const char* name, std::size_t bytes,
                    ComputeBufferUsage usage) {
        ComputeBufferDesc d;
        d.debug_name = name;
        d.size_bytes = bytes;
        d.usage = usage;
        return compute.createBuffer(d);
    };
    const auto rw = ComputeBufferUsage::Storage | ComputeBufferUsage::Upload |
                    ComputeBufferUsage::Download | ComputeBufferUsage::ReadWrite;
    b.deformation_col0 = make("granular_deformation_col0", bytes_vec, rw);
    b.deformation_col1 = make("granular_deformation_col1", bytes_vec, rw);
    b.deformation_col2 = make("granular_deformation_col2", bytes_vec, rw);
    b.plastic_volume = make("granular_plastic_volume", bytes_f32, rw);
    b.softening = make("granular_softening", bytes_f32, rw);
    b.bond_scale = make("granular_bond_scale", bytes_f32, rw);
    b.hardening = make("granular_hardening", bytes_f32, rw);
    b.material_flags = make("granular_material_flags", bytes_u32, rw);
    b.stress_diag = make("granular_stress_diag", bytes_vec, rw);
    b.stress_shear = make("granular_stress_shear", bytes_vec, rw);
    b.yield_value = make("granular_yield_value", bytes_f32, rw);
    b.plastic_increment = make("granular_plastic_increment", bytes_f32, rw);
    b.state_flags = make("granular_state_flags", bytes_u32, rw);
    b.damage = make("granular_damage", bytes_f32, rw);
    b.fracture_history = make("granular_fracture_history", bytes_f32, rw);
    if (!b.valid()) { destroy(&compute, b); return false; }
    b.capacity = count;
    return true;
}

inline bool uploadState(SimulationComputeContext& compute, const FluidParticles& p,
                        GpuBuffers& b) {
    const std::size_t n = p.size();
    if (!ensureBuffers(compute, b, n) || p.granular_deformation_col0.size() != n ||
        p.granular_deformation_col1.size() != n ||
        p.granular_deformation_col2.size() != n ||
        p.granular_plastic_volume.size() != n || p.granular_hardening.size() != n ||
        p.granular_softening.size() != n || p.granular_bond_scale.size() != n ||
        p.granular_stress_diag.size() != n || p.granular_stress_shear.size() != n ||
        p.granular_material_flags.size() != n || p.granular_damage.size() != n ||
        p.granular_fracture_history.size() != n)
        return false;
    compute.beginTransferBatch();
    bool ok = compute.uploadBuffer(b.deformation_col0, p.granular_deformation_col0.data(), n * sizeof(Vec3));
    ok = ok && compute.uploadBuffer(b.deformation_col1, p.granular_deformation_col1.data(), n * sizeof(Vec3));
    ok = ok && compute.uploadBuffer(b.deformation_col2, p.granular_deformation_col2.data(), n * sizeof(Vec3));
    ok = ok && compute.uploadBuffer(b.plastic_volume, p.granular_plastic_volume.data(), n * sizeof(float));
    ok = ok && compute.uploadBuffer(b.softening, p.granular_softening.data(), n * sizeof(float));
    ok = ok && compute.uploadBuffer(b.bond_scale, p.granular_bond_scale.data(), n * sizeof(float));
    ok = ok && compute.uploadBuffer(b.hardening, p.granular_hardening.data(), n * sizeof(float));
    ok = ok && compute.uploadBuffer(b.material_flags, p.granular_material_flags.data(), n * sizeof(uint32_t));
    ok = ok && compute.uploadBuffer(b.stress_diag, p.granular_stress_diag.data(), n * sizeof(Vec3));
    ok = ok && compute.uploadBuffer(b.stress_shear, p.granular_stress_shear.data(), n * sizeof(Vec3));
    ok = ok && compute.uploadBuffer(b.damage, p.granular_damage.data(), n * sizeof(float));
    ok = ok && compute.uploadBuffer(b.fracture_history, p.granular_fracture_history.data(), n * sizeof(float));
    return compute.endTransferBatch() && ok;
}

inline bool dispatchStressUpdate(SimulationComputeContext& compute,
                                 const GpuBuffers& b, ComputeBufferHandle affine,
                                 std::size_t count, float dt,
                                 const Parameters& params,
                                 float young_modulus, float poisson_ratio,
                                 float fracture_strain, float damage_rate,
                                 float healing_rate, bool rebonding) {
    if(count==0)return true;if(!b.valid()||!affine.valid())return false;
    StressUpdateConstants c;
    c.count=static_cast<uint32_t>(std::min<std::size_t>(count,0xffffffffu));c.dt=dt;
    c.young_modulus=std::max(young_modulus,1.0f);c.poisson_ratio=poisson_ratio;
    c.friction_tangent=std::tan(std::clamp(params.friction_angle_radians,0.0f,1.3962634f));c.cohesion=std::max(params.cohesion,0.0f);
    c.dilatancy_tangent=std::tan(std::clamp(params.dilatancy,0.0f,0.7853982f));c.tensile_cutoff=std::max(params.tensile_cutoff,0.0f);
    c.fracture_strain=std::max(fracture_strain,1.0e-5f);c.damage_rate=std::max(damage_rate,0.0f);
    c.healing_rate=std::max(healing_rate,0.0f);c.rebonding=rebonding?1u:0u;
    c.hardening_coefficient=std::max(params.hardening,0.0f);
    ComputeBufferHandle bufs[]={affine,b.stress_diag,b.stress_shear,b.plastic_volume,b.state_flags,
                                b.yield_value,b.plastic_increment,b.damage,b.hardening,
                                b.fracture_history,b.deformation_col0,
                                b.deformation_col1,b.deformation_col2,b.softening,b.bond_scale};
    ComputeDispatch cmd;cmd.kernel="sim_fluid_granular_stress_update";
    cmd.groups={(c.count+255u)/256u,1u,1u};cmd.buffers=bufs;cmd.buffer_count=15;
    cmd.constants=&c;cmd.constants_size=sizeof(c);return compute.dispatch(cmd);
}

inline bool dispatchStressP2G(SimulationComputeContext& compute,const GpuBuffers& b,
                              ComputeBufferHandle positions,ComputeBufferHandle momentum,
                              int nx,int ny,int nz,int component,const Vec3& origin,float h,
                              float dt,float density,std::size_t count){
    if(count==0)return true;if(!b.valid()||!positions.valid()||!momentum.valid())return false;
    StressP2GConstants c;c.nx=nx;c.ny=ny;c.nz=nz;c.count=static_cast<int>(std::min<std::size_t>(count,0x7fffffffu));
    c.component=component;c.ox=origin.x;c.oy=origin.y;c.oz=origin.z;c.h=h;c.dt=dt;
    c.inv_density=1.0f/std::max(density,1.0f);
    ComputeBufferHandle bufs[]={positions,b.stress_diag,b.stress_shear,momentum};
    ComputeDispatch cmd;cmd.kernel="sim_fluid_granular_stress_p2g";cmd.groups={(uint32_t(c.count)+255u)/256u,1u,1u};
    cmd.buffers=bufs;cmd.buffer_count=4;cmd.constants=&c;cmd.constants_size=sizeof(c);return compute.dispatch(cmd);
}

inline bool dispatchSettle(SimulationComputeContext& compute,const GpuBuffers& b,
                           ComputeBufferHandle velocities,std::size_t count,float dt){
    if(count==0)return true;if(!b.valid()||!velocities.valid())return false;
    SettleConstants c;c.count=static_cast<uint32_t>(std::min<std::size_t>(count,0xffffffffu));c.dt=dt;
    ComputeBufferHandle bufs[]={velocities,b.stress_diag,b.state_flags};
    ComputeDispatch cmd;cmd.kernel="sim_fluid_granular_settle";cmd.groups={(c.count+255u)/256u,1u,1u};
    cmd.buffers=bufs;cmd.buffer_count=3;cmd.constants=&c;cmd.constants_size=sizeof(c);
    return compute.dispatch(cmd);
}

} // namespace RayTrophiSim::Fluid::Granular
