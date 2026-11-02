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
// both sides; the under-filled side only in a cell with no AIR face, so the
// free surface is never sucked in.
// ★ 2026-11-02: walls and solids no longer end "interior". Excluding them made
// the floor and every wall layer one-sided -- over-full pushed, under-full
// never pulled -- and that rectified count noise into a swell: a resting
// 12x6x12-cell tank rose 1.4 mm in 30 frames with relaxation on, 0.05 mm off.
// There is no dead band any more: the one-sided term with a one-standard-error
// band let a resting column sink to the band's edge at 8 parcels per cell
// (-5 % after 600 frames, still falling) and rectified count noise into a 10 %
// swell at 27 (D0 probe). The 27-cell mean is what keeps the noise small.
// ★★ 2026-11-02: the mean is TENT-weighted (1-2-1 per axis), not a box. The
// box's transform goes negative at high frequency (1/3 + 2/3 cos k = -1/3 at
// k = pi per axis), so the correction pumped every grid-scale count mode
// instead of damping it: a resting tank grew exponentially (0.015 -> 0.27 m/s
// in 1.5 s), faster the shorter the relaxation time, on CPU and Vulkan alike,
// and was still there at FLIP 0. The tent's transform ((1 + cos k)/2 per axis)
// is never negative. A linear model of the loop grows 6600x in 2 s with the
// box and decays with the tent at every relaxation time and frame rate.
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
                    float w = float((2 - abs(dx)) * (2 - abs(dy)) * (2 - abs(dz)));
                    sum += w * m;
                    n += w;
                }
            }
        }
    }
    bool interior = true;
    for (int axis = 0; axis < 3; ++axis) {
        for (int direction = -1; direction <= 1; direction += 2) {
            ivec3 q = c;
            q[axis] += direction;
            float m = rtDensityMask(q);
            if (m > -0.5 && m < 0.5) {
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
