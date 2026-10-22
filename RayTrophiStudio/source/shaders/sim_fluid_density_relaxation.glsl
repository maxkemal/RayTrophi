// sim_fluid_density_relaxation.glsl - the liquid density relaxation term,
// shared by sim_fluid_cg_residual_init, sim_sparse_pressure and
// sim_fluid_divergence_porous. Same formula as APICFluidSolver.cpp and the
// CUDA residual kernel (fluid_cg_residual_init_kernel).
//
// The includer defines float rtDensityMask(ivec3 c): the cell's parcel count
// (>= 0.5 fluid, 0 air, < -0.5 solid), with out-of-domain cells read as the
// domain's walls (open boundary = air, closed = solid).
//
// Two-sided (2026-10-20). The 27-cell mean count is pulled to the target from
// both sides; the under-filled side only in a cell whose six faces all open
// onto liquid, so neither the free surface nor a wall layer is sucked in.
// There is no dead band any more: the one-sided term with a one-standard-error
// band let a resting column sink to the band's edge at 8 parcels per cell
// (-5 % after 600 frames, still falling) and rectified count noise into a 10 %
// swell at 27 (D0 probe). The 27-cell mean is what keeps the noise small.
struct RtDensitySample {
    float mean;
    bool interior;
};

RtDensitySample rtDensitySample(ivec3 c) {
    float sum = 0.0;
    float n = 0.0;
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                float m = rtDensityMask(c + ivec3(dx, dy, dz));
                if (m >= 0.5) {
                    sum += m;
                    n += 1.0;
                }
            }
        }
    }
    bool interior = true;
    for (int axis = 0; axis < 3; ++axis) {
        for (int direction = -1; direction <= 1; direction += 2) {
            ivec3 q = c;
            q[axis] += direction;
            if (rtDensityMask(q) < 0.5) {
                interior = false;
            }
        }
    }
    RtDensitySample s;
    s.mean = n > 0.0 ? sum / n : 0.0;
    s.interior = interior;
    return s;
}

// Excess over `target` parcels, as a share of particles_per_cell. The caller
// divides by max(density_relaxation_s, dt) to get the 1/s source.
float rtDensityExcess(RtDensitySample s, float target, float ppc) {
    float e = (s.mean - target) / ppc;
    if (!s.interior) {
        e = max(e, 0.0);
    }
    return clamp(e, -0.5, 4.0);
}
