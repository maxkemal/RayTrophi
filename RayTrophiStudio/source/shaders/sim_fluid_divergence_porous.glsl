// sim_fluid_divergence_porous.comp
// sim_fluid_divergence_var plus a POROUS solid: grains of a Matter domain that
// share cells with liquid (H1 B5, unresolved CFD-DEM). Face weights carry the
// pore fraction eps, so the projection enforces the mixture continuity
//   div( eps u_liquid + (1 - eps) u_grain ) = 0.
// The grain velocity sits in sv* at NON-solid cells (volume-weighted mean of
// the grains overlapping the cell). Only this kernel reads sv* there; every
// other path keeps sv* meaningful in solid cells only.
//
// --- original var notes ---
// Separate from sim_fluid_divergence.comp on purpose. The Vulkan kernel registry
// binds a FIXED buffer count per kernel name, while CUDA picks its pointers from
// the dispatch's buffer_count — so the variational form (11 buffers) cannot share
// a registration with the plain one (5). Splitting also leaves the working plain
// path untouched, which is what makes this port revertible.
//
// Physics: a MAC face is fractionally open. The open part carries the fluid
// velocity, the closed part carries the SOLID's velocity — which is why a moving
// collider pushes fluid instead of merely blocking it. For a static wall the
// solid term is 0 and this reduces exactly to the plain branch's hard zeroing.
layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    int   nx; int ny; int nz;
    int   boundary;
    float voxel_size;
    float dt;
    float sor_omega;
    int   iterations;
    int   parity;
    float density_relaxation_s;
    int   particles_per_cell;
    int   variational;
    int   gfm_active;
#ifdef RT_FLUID_WINDOW
    int begin_x; int begin_y; int begin_z;
    int extent_x; int extent_y; int extent_z;
#endif
} pc;

#include "fluid_pressure_window.glsl"

layout(set = 0, binding = 0) readonly buffer VelX { float vel_x[]; };
layout(set = 0, binding = 1) readonly buffer VelY { float vel_y[]; };
layout(set = 0, binding = 2) readonly buffer VelZ { float vel_z[]; };
layout(set = 0, binding = 3) readonly buffer Mask { float fluid_mask[]; };
layout(set = 0, binding = 4) buffer Divergence    { float divergence[]; };
layout(set = 0, binding = 5) readonly buffer UW   { float uw[]; };
layout(set = 0, binding = 6) readonly buffer VW   { float vw[]; };
layout(set = 0, binding = 7) readonly buffer WW   { float ww[]; };
layout(set = 0, binding = 8) readonly buffer SVX  { float svx[]; };
layout(set = 0, binding = 9) readonly buffer SVY  { float svy[]; };
layout(set = 0, binding = 10) readonly buffer SVZ { float svz[]; };
#ifdef RT_SPARSE_MAC
// Compact velocity pages are bound at 0..2 (docs/dev/MATTER_SPARSE_S1_SIVI_GPU.md).
layout(set = 0, binding = 11) readonly buffer MacTileMap { uint mac_tile_map[]; };
layout(set = 0, binding = 12) readonly buffer MacTileList { uint mac_tile_list[]; };
#endif
// MPM mode (2026-11-10, MATTER_SIVI_MPM_BASINC_BAGLASIMI.md section 9): the
// per-cell solid fraction phi of the MPM lane, and pc.sor_omega carries the
// mobility ratio r = rho_liquid / rho_constituent (> 0 selects the mode; the
// DEM grain path leaves it 0 and keeps the original formula below exactly).
// The face weight then holds the PRESSURE MOBILITY
//   beta = w_collider * min(1, eps + (1 - eps) r)
// -- the solid fraction answers the pressure too (Xie et al. 2026, Eq. 21:
// Dt((1-n)/rho_s + n/rho_w) lap dp) instead of being a wall of prescribed
// velocity -- and this kernel recovers w_collider and eps for the mixture flux
//   w_collider (eps u_liquid + (1 - eps) u_solid) + (1 - w_collider) u_wall.
#ifdef RT_SPARSE_MAC
layout(set = 0, binding = 13) readonly buffer SolidFraction { float solid_fraction[]; };
#else
layout(set = 0, binding = 11) readonly buffer SolidFraction { float solid_fraction[]; };
#endif
#include "sim_mac_lane.glsl"
#include "sim_mac_solid_weight.glsl"
// Velocity face read through the shared dense/compact storage contract.
float macVelocity(int component, int i, int j, int k) {
    uint a = macAddress(component, i, j, k);
    if (a == MAC_ABSENT) {
        return 0.0;
    }
    return component == 0 ? vel_x[a] : (component == 1 ? vel_y[a] : vel_z[a]);
}

int  cell_id(int i, int j, int k) { return i + j*pc.nx + k*pc.nx*pc.ny; }
int  vx_idx(int i, int j, int k)  { return i + j*(pc.nx+1) + k*(pc.nx+1)*pc.ny; }
int  vy_idx(int i, int j, int k)  { return i + j*pc.nx + k*pc.nx*(pc.ny+1); }
int  vz_idx(int i, int j, int k)  { return i + j*pc.nx + k*pc.nx*pc.ny; }

float mask_at(int i, int j, int k) {
    if (i < 0 || i >= pc.nx || j < 0 || j >= pc.ny || k < 0 || k >= pc.nz)
        return (pc.boundary == 0) ? 0.0 : -1.0;
    return fluid_mask[cell_id(i, j, k)];
}
bool is_solid(int i, int j, int k) { return mask_at(i, j, k) < -0.5; }
float rtDensityMask(ivec3 c) { return mask_at(c.x, c.y, c.z); }
#include "sim_fluid_density_relaxation.glsl"

// Fractional open weight of a MAC face. Domain-boundary faces follow the wall
// mode (open = 1 Dirichlet, closed/periodic = 0 wall) rather than the array.
float fw_x(int i, int j, int k) {
    if (i <= 0 || i >= pc.nx) return (pc.boundary == 0) ? 1.0 : 0.0;
    return macSolidWeight(0, i, j, k);
}
float fw_y(int i, int j, int k) {
    if (j <= 0 || j >= pc.ny) return (pc.boundary == 0) ? 1.0 : 0.0;
    return macSolidWeight(1, i, j, k);
}
float fw_z(int i, int j, int k) {
    if (k <= 0 || k >= pc.nz) return (pc.boundary == 0) ? 1.0 : 0.0;
    return macSolidWeight(2, i, j, k);
}

// Solid normal velocity at a face: the SOLID-side cell's component, 0 when
// neither adjacent cell is solid.
// Porous fallback: neither neighbour is a collider solid, so the closed part
// of the face is grain; its velocity is the mean of the two cells' grain
// velocities (a domain-boundary side contributes its in-grid cell only).
float porous(float lo, bool has_lo, float hi, bool has_hi) {
    if (has_lo && has_hi) return 0.5 * (lo + hi);
    return has_lo ? lo : (has_hi ? hi : 0.0);
}
float sv_x(int i, int j, int k) {
    if (i - 1 >= 0 && is_solid(i - 1, j, k)) return svx[cell_id(i - 1, j, k)];
    if (i < pc.nx && is_solid(i, j, k))      return svx[cell_id(i, j, k)];
    return porous(i - 1 >= 0 ? svx[cell_id(i - 1, j, k)] : 0.0, i - 1 >= 0,
                  i < pc.nx ? svx[cell_id(i, j, k)] : 0.0, i < pc.nx);
}
float sv_y(int i, int j, int k) {
    if (j - 1 >= 0 && is_solid(i, j - 1, k)) return svy[cell_id(i, j - 1, k)];
    if (j < pc.ny && is_solid(i, j, k))      return svy[cell_id(i, j, k)];
    return porous(j - 1 >= 0 ? svy[cell_id(i, j - 1, k)] : 0.0, j - 1 >= 0,
                  j < pc.ny ? svy[cell_id(i, j, k)] : 0.0, j < pc.ny);
}
float sv_z(int i, int j, int k) {
    if (k - 1 >= 0 && is_solid(i, j, k - 1)) return svz[cell_id(i, j, k - 1)];
    if (k < pc.nz && is_solid(i, j, k))      return svz[cell_id(i, j, k)];
    return porous(k - 1 >= 0 ? svz[cell_id(i, j, k - 1)] : 0.0, k - 1 >= 0,
                  k < pc.nz ? svz[cell_id(i, j, k)] : 0.0, k < pc.nz);
}

// MPM solid fraction of a face: mean of its in-grid cells (MatterMpmPorosity.cpp
// writes the face weights from the same rule, so w / beta is exact).
float cellFraction(ivec3 c, out bool inside) {
    inside = all(greaterThanEqual(c, ivec3(0))) && c.x < pc.nx && c.y < pc.ny && c.z < pc.nz;
    return inside ? clamp(solid_fraction[cell_id(c.x, c.y, c.z)], 0.0, 1.0) : 0.0;
}
float faceFraction(ivec3 lo, ivec3 hi) {
    bool il, ih;
    float a = cellFraction(lo, il), b = cellFraction(hi, ih);
    return il && ih ? 0.5 * (a + b) : (il ? a : b);
}
// Mixture flux through a face whose weight w is the pressure mobility.
float mpmFlux(float w, float phi, float r, float u_liquid, float u_solid) {
    if (!(phi > 1e-4)) return w * u_liquid + (1.0 - w) * u_solid;
    float eps = 1.0 - phi;
    float beta = min(1.0, eps + phi * r);
    float w_collider = beta > 1e-6 ? clamp(w / beta, 0.0, 1.0) : 0.0;
    return w_collider * (eps * u_liquid + phi * u_solid) + (1.0 - w_collider) * u_solid;
}

void main() {
    int id = fluidPressureCellIndex();
    if (id >= pc.nx * pc.ny * pc.nz) return;

    if (fluid_mask[id] < 0.5) { divergence[id] = 0.0; return; }

    int i = id % pc.nx;
    int j = (id / pc.nx) % pc.ny;
    int k = id / (pc.nx * pc.ny);
    float inv_h = pc.voxel_size > 1e-6 ? 1.0 / pc.voxel_size : 1.0;

    float vx_lo = macVelocity(0, i,     j, k);
    float vx_hi = macVelocity(0, i + 1, j, k);
    float vy_lo = macVelocity(1, i, j,     k);
    float vy_hi = macVelocity(1, i, j + 1, k);
    float vz_lo = macVelocity(2, i, j, k    );
    float vz_hi = macVelocity(2, i, j, k + 1);

    float wxl = fw_x(i,     j, k), wxh = fw_x(i + 1, j, k);
    float wyl = fw_y(i, j,     k), wyh = fw_y(i, j + 1, k);
    float wzl = fw_z(i, j, k    ), wzh = fw_z(i, j, k + 1);

    if (pc.sor_omega > 0.0) {
        float r = pc.sor_omega;
        vx_lo = mpmFlux(wxl, faceFraction(ivec3(i - 1, j, k), ivec3(i, j, k)), r, vx_lo, sv_x(i, j, k));
        vx_hi = mpmFlux(wxh, faceFraction(ivec3(i, j, k), ivec3(i + 1, j, k)), r, vx_hi, sv_x(i + 1, j, k));
        vy_lo = mpmFlux(wyl, faceFraction(ivec3(i, j - 1, k), ivec3(i, j, k)), r, vy_lo, sv_y(i, j, k));
        vy_hi = mpmFlux(wyh, faceFraction(ivec3(i, j, k), ivec3(i, j + 1, k)), r, vy_hi, sv_y(i, j + 1, k));
        vz_lo = mpmFlux(wzl, faceFraction(ivec3(i, j, k - 1), ivec3(i, j, k)), r, vz_lo, sv_z(i, j, k));
        vz_hi = mpmFlux(wzh, faceFraction(ivec3(i, j, k), ivec3(i, j, k + 1)), r, vz_hi, sv_z(i, j, k + 1));
    } else {
        vx_lo = wxl * vx_lo + (1.0 - wxl) * sv_x(i,     j, k);
        vx_hi = wxh * vx_hi + (1.0 - wxh) * sv_x(i + 1, j, k);
        vy_lo = wyl * vy_lo + (1.0 - wyl) * sv_y(i, j,     k);
        vy_hi = wyh * vy_hi + (1.0 - wyh) * sv_y(i, j + 1, k);
        vz_lo = wzl * vz_lo + (1.0 - wzl) * sv_z(i, j, k    );
        vz_hi = wzh * vz_hi + (1.0 - wzh) * sv_z(i, j, k + 1);
    }

    float div = ((vx_hi - vx_lo) + (vy_hi - vy_lo) + (vz_hi - vz_lo)) * inv_h;

    // Density correction with the pore volume as the target. residual_init
    // pulls a cell's mean count to particles_per_cell; a cell that is
    // (1 - eps) grain holds only eps of that liquid at rest. Without this the
    // correction refills the pores and undoes the exclusion the projection
    // just enforced. eps is the mean open weight of the cell's interior faces
    // that border no collider (porosity is the only thing that closes those).
    // The extra term is the difference between the shared two-sided term at
    // target eps*ppc and at ppc (sim_fluid_density_relaxation.glsl), so the
    // sum of both is that term at the pore target exactly. It enters as
    // rhs = -div h^2/dt, removed over density_relaxation_s (1/s).
    if (pc.density_relaxation_s > 0.0 && pc.particles_per_cell > 0) {
        float open = 0.0, faces = 0.0;
        if (i > 0 && !is_solid(i - 1, j, k))          { open += wxl; faces += 1.0; }
        if (i + 1 < pc.nx && !is_solid(i + 1, j, k))  { open += wxh; faces += 1.0; }
        if (j > 0 && !is_solid(i, j - 1, k))          { open += wyl; faces += 1.0; }
        if (j + 1 < pc.ny && !is_solid(i, j + 1, k))  { open += wyh; faces += 1.0; }
        if (k > 0 && !is_solid(i, j, k - 1))          { open += wzl; faces += 1.0; }
        if (k + 1 < pc.nz && !is_solid(i, j, k + 1))  { open += wzh; faces += 1.0; }
        float eps = faces > 0.0 ? clamp(open / faces, 0.0, 1.0) : 1.0;
        // MPM mode: the face weights are pressure mobilities (~1 for snow),
        // the pore volume is the cell's own solid fraction.
        if (pc.sor_omega > 0.0) eps = 1.0 - clamp(solid_fraction[id], 0.0, 1.0);
        if (eps < 0.999) {
            float ppc = float(pc.particles_per_cell);
            RtDensitySample sample_here = rtDensitySample(ivec3(i, j, k));
            float extra = rtDensityExcess(sample_here, eps * ppc, ppc) -
                          rtDensityExcess(sample_here, ppc, ppc);
            div -= extra / max(pc.density_relaxation_s, pc.dt);
        }
    }
    divergence[id] = div;
}
