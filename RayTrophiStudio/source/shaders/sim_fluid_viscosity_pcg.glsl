// sim_fluid_viscosity_pcg.glsl - implicit viscosity solved to a TOLERANCE.
// (docs/dev/VISKOZ_SIVI_URETIM_KALITESI.md)
//
// Replaces the fixed red-black Gauss-Seidel sweeps of sim_fluid_viscosity.glsl
// on the GPU, and since 2026-11-09 the per-component Laplacian too: the
// VARIATIONAL viscosity of Batty & Bridson (2008). Backward Euler minimises
//   sum_faces V_f (u - u*)^2 / 2
//   + dt sum_cells nu V_c (u_x^2 + v_y^2 + w_z^2)
//   + dt/2 sum_edges nu V_e ((u_y + v_x)^2 + (u_z + w_x)^2 + (v_z + w_y)^2)
// i.e. the full stress div(2 mu D), components coupled through the edge
// terms, with LIQUID VOLUME FRACTIONS as control volumes. The free surface is
// then stress-free by construction and sub-cell: a half-full top cell weighs
// half. The cell-mask Laplacian counted it full, so a 2-3 cell honey film
// flowed as if up to h/2 thicker (flux ~ H^3): Huppert radius 1.3x at h 2 mm
// with a converged solve (2026-11-09).
// Fractions: parcel count / particles_per_cell in liquid cells that touch air
// (interior liquid, solid cells and closed walls count 1, air 0); a face is
// the mean of its two cells, an edge of its four. Unknowns are liquid faces
// AND the air faces that share an edge with liquid (their kinetic weight is a
// small kAirMass): the solve extrapolates them stress-free. Holding them at
// their stale value coupled the free surface to garbage through every surface
// edge -- the core stiffened but a thin skirt ran away (Huppert 1.37x,
// exponent 0.23, 2026-11-09). Solid faces hold the solid's velocity. wall_slip scales every edge term that touches a solid face
// (1 = no tangential stress at the wall). The system is symmetric positive
// definite, so Jacobi-preconditioned conjugate gradients converge.
//
// Why: honey's smooth modes (the whole puddle sagging) are the slowest for GS;
// each e-fold takes ~12 a sweeps, and a reaches ~70 a frame at h 2 mm. Sixteen
// sweeps applied 1/9..1/25 of the viscosity -- a 48 ml honey puddle spread
// 1.3-1.5x Huppert's radius with exponent 0.19 instead of 0.125 (2026-11-09).
// "Too few sweeps under-apply, never explode" was the silent failure.
//
// Vectors live in LANE space: lane = component * M + local, M = the per-
// component lane count (dense: largest face array; sparse: compact pages).
// x is the velocity itself, read/written through its MAC storage.
// Modes (pc.mode):
//   0 init     r = b - A x, d = diag, z = r/d, p = z     partial: r.z
//   1 apply    q = A p                                   partial: p.q
//   2 update   x += alpha p, r -= alpha q, z = r/d       partial: r.z
//   3 search   p = z + beta p
//   4 finish   one group sums the partials -> scalars (pc.stage 0/1/2)
// scalars: [0] rz  [1] pq  [2] alpha  [3] beta  [4] rz0  [5] frozen  [6] its
layout(local_size_x = 256) in;
layout(push_constant) uniform PC {
    int   nx; int ny; int nz;
    int   boundary;      // 0 open, 1 closed, 2 periodic
    float voxel_size;
    float dt;
    float alpha;         // uniform a = nu dt / h^2
    float solid_weight;  // 1 - wall_slip
    int   mode;
    int   has_nu;
    int   stage;
    uint  partial_count;
    uint  lanes_per_component;
    uint  tiles_x; uint tiles_y; uint tiles_z; uint compact_faces;
    float particles_per_cell;  // target parcel count of a full liquid cell
} pc;
layout(set = 0, binding = 0) buffer VelX { float vel_x[]; };
layout(set = 0, binding = 1) buffer VelY { float vel_y[]; };
layout(set = 0, binding = 2) buffer VelZ { float vel_z[]; };
layout(set = 0, binding = 3) readonly buffer Mask { float fluid_mask[]; };
layout(set = 0, binding = 4) readonly buffer RhsX { float rhs_x[]; };
layout(set = 0, binding = 5) readonly buffer RhsY { float rhs_y[]; };
layout(set = 0, binding = 6) readonly buffer RhsZ { float rhs_z[]; };
layout(set = 0, binding = 7) readonly buffer SVX { float svx[]; };
layout(set = 0, binding = 8) readonly buffer SVY { float svy[]; };
layout(set = 0, binding = 9) readonly buffer SVZ { float svz[]; };
layout(set = 0, binding = 10) readonly buffer Nu { float nu_cell[]; };
layout(set = 0, binding = 11) buffer R { float vec_r[]; };
layout(set = 0, binding = 12) buffer P { float vec_p[]; };
layout(set = 0, binding = 13) buffer Q { float vec_q[]; };
layout(set = 0, binding = 14) buffer D { float vec_d[]; };
layout(set = 0, binding = 15) buffer Scalars { double scalars[]; };
layout(set = 0, binding = 16) buffer Partials { double partials[]; };
#ifdef RT_SPARSE_VISCOSITY
layout(set = 0, binding = 17) readonly buffer TileMap { uint tile_map[]; };
layout(set = 0, binding = 18) readonly buffer TileList { uint tile_list[]; };
#endif
#ifdef RT_SPARSE_MAC
layout(set = 0, binding = 19) readonly buffer MacTileMap { uint mac_tile_map[]; };
layout(set = 0, binding = 20) readonly buffer MacTileList { uint mac_tile_list[]; };
#endif
#include "sim_mac_lane.glsl"

shared double sdata[256];

int cell_id(int i, int j, int k) { return i + j * pc.nx + k * pc.nx * pc.ny; }
bool cell_in_range(int i, int j, int k) {
    return i >= 0 && i < pc.nx && j >= 0 && j < pc.ny && k >= 0 && k < pc.nz;
}
bool cell_solid(int i, int j, int k) {
    return cell_in_range(i, j, k) && fluid_mask[cell_id(i, j, k)] < -0.5;
}
bool cell_fluid(int i, int j, int k) {
    return cell_in_range(i, j, k) && fluid_mask[cell_id(i, j, k)] > 0.5;
}
float solid_vel(int i, int j, int k, int comp) {
    if (!cell_in_range(i, j, k)) return 0.0;
    int c = cell_id(i, j, k);
    return comp == 0 ? svx[c] : (comp == 1 ? svy[c] : svz[c]);
}
float cell_nu(int i, int j, int k) {
    i = clamp(i, 0, pc.nx - 1);
    j = clamp(j, 0, pc.ny - 1);
    k = clamp(k, 0, pc.nz - 1);
    return nu_cell[cell_id(i, j, k)];
}
float face_nu(int comp, ivec3 f) {
    ivec3 lo = f;
    lo[comp] -= 1;
    return 0.5 * (cell_nu(lo.x, lo.y, lo.z) + cell_nu(f.x, f.y, f.z));
}
float readVel(int comp, ivec3 f) {
    uint a = macAddress(comp, f.x, f.y, f.z);
    if (a == MAC_ABSENT) return 0.0;
    return comp == 0 ? vel_x[a] : (comp == 1 ? vel_y[a] : vel_z[a]);
}
void addVel(int comp, ivec3 f, float dv) {
    uint a = macAddress(comp, f.x, f.y, f.z);
    if (a == MAC_ABSENT) return;
    if (comp == 0) vel_x[a] += dv;
    else if (comp == 1) vel_y[a] += dv;
    else vel_z[a] += dv;
}
float rhsAt(int comp, uint local) {
    return comp == 0 ? rhs_x[local] : (comp == 1 ? rhs_y[local] : rhs_z[local]);
}

// kind: 0 fluid, 1 air, 2 solid (mirrors sim_fluid_viscosity.glsl classify).
int classify(int comp, ivec3 f, out float solid_value) {
    solid_value = 0.0;
    ivec3 fdim = ivec3(pc.nx, pc.ny, pc.nz);
    fdim[comp] += 1;
    if (any(lessThan(f, ivec3(0))) || any(greaterThanEqual(f, fdim))) {
        return pc.boundary == 1 ? 2 : 1;
    }
    ivec3 lo = f;
    lo[comp] -= 1;
    bool lo_solid = cell_solid(lo.x, lo.y, lo.z);
    bool hi_solid = cell_solid(f.x, f.y, f.z);
    if (lo_solid || hi_solid) {
        solid_value = lo_solid ? solid_vel(lo.x, lo.y, lo.z, comp)
                               : solid_vel(f.x, f.y, f.z, comp);
        return 2;
    }
    if (cell_fluid(lo.x, lo.y, lo.z) || cell_fluid(f.x, f.y, f.z)) {
        return 0;
    }
    return 1;
}

// ── Lane <-> face ───────────────────────────────────────────────────────────
#ifdef RT_SPARSE_VISCOSITY
bool laneFace(uint local, int comp, out ivec3 face) {
    face = ivec3(0);
    if (local >= pc.compact_faces) return false;
    uint tile = tile_list[local / 576u + 1u];
    uvec3 coordinate = uvec3(tile % pc.tiles_x, (tile / pc.tiles_x) % pc.tiles_y,
                             tile / (pc.tiles_x * pc.tiles_y));
    uvec3 extent = uvec3(8u);
    extent[comp] = 9u;
    uint l = local % 576u;
    face = ivec3(coordinate * 8u + uvec3(l % extent.x, (l / extent.x) % extent.y,
                                         l / (extent.x * extent.y)));
    ivec3 dims = ivec3(pc.nx, pc.ny, pc.nz);
    ivec3 fdim = dims;
    ++fdim[comp];
    if (any(greaterThanEqual(face, fdim))) return false;
    ivec3 owner = face;
    owner[comp] = min(owner[comp], dims[comp] - 1);
    return all(equal(uvec3(owner) / 8u, coordinate));
}
int localOf(int comp, ivec3 f) {
    ivec3 dims = ivec3(pc.nx, pc.ny, pc.nz);
    ivec3 fdim = dims;
    ++fdim[comp];
    if (any(lessThan(f, ivec3(0))) || any(greaterThanEqual(f, fdim))) return -1;
    ivec3 owner = f;
    owner[comp] = min(owner[comp], dims[comp] - 1);
    uvec3 tile = uvec3(owner) / 8u;
    uint slot = tile_map[tile.x + pc.tiles_x * (tile.y + pc.tiles_y * tile.z)];
    if (slot >= 0xfffffffeu) return -1;
    uvec3 extent = uvec3(8u);
    extent[comp] = 9u;
    uvec3 l = uvec3(f) - tile * 8u;
    return int(slot * 576u + l.x + extent.x * (l.y + extent.y * l.z));
}
#else
bool laneFace(uint local, int comp, out ivec3 face) {
    ivec3 fdim = ivec3(pc.nx, pc.ny, pc.nz);
    fdim[comp] += 1;
    face = ivec3(0);
    if (local >= uint(fdim.x * fdim.y * fdim.z)) return false;
    int id = int(local);
    face = ivec3(id % fdim.x, (id / fdim.x) % fdim.y, id / (fdim.x * fdim.y));
    return true;
}
int localOf(int comp, ivec3 f) {
    ivec3 fdim = ivec3(pc.nx, pc.ny, pc.nz);
    fdim[comp] += 1;
    if (any(lessThan(f, ivec3(0))) || any(greaterThanEqual(f, fdim))) return -1;
    return f.x + fdim.x * (f.y + fdim.y * f.z);
}
#endif

// Liquid volume fraction of a cell (see the header).
float cellW(ivec3 c) {
    if (!cell_in_range(c.x, c.y, c.z)) {
        return pc.boundary == 1 ? 1.0 : 0.0;
    }
    float m = fluid_mask[cell_id(c.x, c.y, c.z)];
    if (m < -0.5) return 1.0;
    if (m <= 0.5) return 0.0;
    for (int axis = 0; axis < 3; ++axis) {
        for (int side = -1; side <= 1; side += 2) {
            ivec3 n = c;
            n[axis] += side;
            bool air = cell_in_range(n.x, n.y, n.z)
                ? (fluid_mask[cell_id(n.x, n.y, n.z)] > -0.5 &&
                   fluid_mask[cell_id(n.x, n.y, n.z)] <= 0.5)
                : pc.boundary != 1;
            if (air) {
                return clamp(m / max(pc.particles_per_cell, 1.0), 0.0, 1.0);
            }
        }
    }
    return 1.0;
}
// nu * dt / h^2 of a cell (uniform: pc.alpha).
float cellK(ivec3 c) {
    return pc.has_nu != 0
        ? cell_nu(c.x, c.y, c.z) * pc.dt / (max(pc.voxel_size, 1.0e-6) * max(pc.voxel_size, 1.0e-6))
        : pc.alpha;
}

const float kAirMass = 0.01;

// An air face in a liquid edge's stencil: one of the cells of its four edges
// (the two cells it separates and their neighbours across the other axes) is
// liquid.
bool nearLiquid(int comp, ivec3 f) {
    ivec3 ec = ivec3(0);
    ec[comp] = 1;
    for (int side = 0; side < 2; ++side) {
        ivec3 c0 = side == 0 ? f : f - ec;
        if (cell_fluid(c0.x, c0.y, c0.z)) return true;
        for (int step = 1; step <= 2; ++step) {
            int a = (comp + step) % 3;
            ivec3 ea = ivec3(0);
            ea[a] = 1;
            ivec3 up = c0 + ea;
            ivec3 dn = c0 - ea;
            if (cell_fluid(up.x, up.y, up.z) || cell_fluid(dn.x, dn.y, dn.z)) return true;
        }
    }
    return false;
}
// 0 liquid unknown, 3 air unknown, 1 air known, 2 solid.
int faceKind(int comp, ivec3 g, out float sv) {
    int kind = classify(comp, g, sv);
    if (kind == 1) {
        ivec3 fdim = ivec3(pc.nx, pc.ny, pc.nz);
        fdim[comp] += 1;
        bool inside = all(greaterThanEqual(g, ivec3(0))) && all(lessThan(g, fdim));
        if (inside && nearLiquid(comp, g)) kind = 3;
    }
    return kind;
}

// One face's contribution to a row: coeff * (its p if an unknown and use_p,
// its velocity / solid value otherwise; a known face adds 0 under use_p).
float faceTerm(int comp, ivec3 g, float coeff, bool use_p, int kind, float sv) {
    if (kind == 0 || kind == 3) {
        if (use_p) {
            int ln = localOf(comp, g);
            return ln >= 0 ? coeff * vec_p[uint(comp) * pc.lanes_per_component + uint(ln)] : 0.0;
        }
        return coeff * readVel(comp, g);
    }
    if (use_p) return 0.0;
    return coeff * (kind == 2 ? sv : readVel(comp, g));
}

// Row of unknown face f (component comp): row = diag * x_f + off, with off the
// signed neighbour terms. vf is the face's own control volume.
void variational(int comp, ivec3 f, bool use_p, out float diag, out float vf, out float off) {
    ivec3 ec = ivec3(0);
    ec[comp] = 1;
    float w_hi = cellW(f);
    float w_lo = cellW(f - ec);
    vf = max(0.5 * (w_hi + w_lo), kAirMass);
    diag = vf;
    off = 0.0;
    float sv;
    // Normal stress: the cells on either side along the face's own axis.
    float k_hi = 2.0 * cellK(f) * w_hi;
    float k_lo = 2.0 * cellK(f - ec) * w_lo;
    int kind = faceKind(comp, f + ec, sv);
    diag += k_hi;
    off += faceTerm(comp, f + ec, -k_hi, use_p, kind, sv);
    kind = faceKind(comp, f - ec, sv);
    diag += k_lo;
    off += faceTerm(comp, f - ec, -k_lo, use_p, kind, sv);
    // Shear: two edges per other axis.
    for (int step = 1; step <= 2; ++step) {
        int a = (comp + step) % 3;
        ivec3 ea = ivec3(0);
        ea[a] = 1;
        for (int side = 0; side < 2; ++side) {
            // upper edge: cells f, f-ec, f+ea, f-ec+ea; neighbour f+ea;
            //   cross faces (component a) hi = f+ea, lo = f-ec+ea, sign -1.
            // lower edge: cells f, f-ec, f-ea, f-ec-ea; neighbour f-ea;
            //   cross faces hi = f, lo = f-ec, sign +1.
            ivec3 shift = side == 0 ? ea : -ea;
            ivec3 cross_base = side == 0 ? f + ea : f;
            float sign = side == 0 ? -1.0 : 1.0;
            float v_e = 0.25 * (w_hi + w_lo + cellW(f + shift) + cellW(f - ec + shift));
            float k_e = 0.25 * (cellK(f) + cellK(f - ec) + cellK(f + shift) +
                                cellK(f - ec + shift)) * v_e;
            if (!(k_e > 0.0)) continue;
            float sv_n, sv_hi, sv_lo;
            int kind_n = faceKind(comp, f + shift, sv_n);
            int kind_hi = faceKind(a, cross_base, sv_hi);
            int kind_lo = faceKind(a, cross_base - ec, sv_lo);
            if (kind_n == 2 || kind_hi == 2 || kind_lo == 2) {
                k_e *= pc.solid_weight;
            }
            diag += k_e;
            off += faceTerm(comp, f + shift, -k_e, use_p, kind_n, sv_n);
            off += faceTerm(a, cross_base, sign * k_e, use_p, kind_hi, sv_hi);
            off += faceTerm(a, cross_base - ec, -sign * k_e, use_p, kind_lo, sv_lo);
        }
    }
}

uint flatGroup() {
    return gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x;
}
void emitPartial(double value) {
    uint tid = gl_LocalInvocationID.x;
    sdata[tid] = value;
    barrier();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        barrier();
    }
    if (tid == 0u && flatGroup() < pc.partial_count) {
        partials[flatGroup()] = sdata[0];
    }
}

void finish() {
    uint tid = gl_LocalInvocationID.x;
    double sum = 0.0LF;
    for (uint i = tid; i < pc.partial_count; i += 256u) sum += partials[i];
    sdata[tid] = sum;
    barrier();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        barrier();
    }
    if (tid != 0u) return;
    double total = sdata[0];
    if (pc.stage == 0) {
        scalars[0] = total;
        scalars[4] = total;
        scalars[5] = total > 0.0LF ? 0.0LF : 1.0LF;  // nothing to solve
        scalars[6] = 0.0LF;
    } else if (pc.stage == 1) {
        if (scalars[5] != 0.0LF) return;
        scalars[1] = total;
        if (!(total > 1.0e-300LF)) {
            scalars[2] = 0.0LF;
            scalars[5] = 1.0LF;
        } else {
            scalars[2] = scalars[0] / total;
        }
    } else {
        if (scalars[5] != 0.0LF) return;
        scalars[3] = scalars[0] > 0.0LF ? total / scalars[0] : 0.0LF;
        scalars[0] = total;
        scalars[6] += 1.0LF;
    }
}

void main() {
    if (pc.mode == 4) {
        finish();
        return;
    }
    uint lane = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * gl_NumWorkGroups.x * 256u;
    uint M = pc.lanes_per_component;
    double contribution = 0.0LF;
    bool frozen = pc.mode != 0 && scalars[5] != 0.0LF;
    if (lane < 3u * M && !frozen) {
        int comp = int(lane / M);
        uint local = lane % M;
        ivec3 f;
        float dummy;
        bool face_active = false;
        if (laneFace(local, comp, f)) {
            int own = faceKind(comp, f, dummy);
            face_active = own == 0 || own == 3;
        }
        if (pc.mode == 0) {
            float r = 0.0, d = 0.0;
            if (face_active) {
                float diag, vf, off;
                variational(comp, f, false, diag, vf, off);
                float x = readVel(comp, f);
                r = vf * rhsAt(comp, local) - (diag * x + off);
                d = diag;
            }
            float z = d > 0.0 ? r / d : 0.0;
            vec_r[lane] = r;
            vec_d[lane] = d;
            vec_p[lane] = z;
            vec_q[lane] = 0.0;
            contribution = double(r) * double(z);
        } else if (pc.mode == 1) {
            float q = 0.0;
            if (face_active) {
                float diag, vf, off;
                variational(comp, f, true, diag, vf, off);
                q = diag * vec_p[lane] + off;
            }
            vec_q[lane] = q;
            contribution = double(vec_p[lane]) * double(q);
        } else if (pc.mode == 2) {
            float a = float(scalars[2]);
            if (face_active) {
                addVel(comp, f, a * vec_p[lane]);
            }
            float r = vec_r[lane] - a * vec_q[lane];
            vec_r[lane] = r;
            float d = vec_d[lane];
            contribution = d > 0.0 ? double(r) * double(r / d) : 0.0LF;
        } else if (pc.mode == 3) {
            float d = vec_d[lane];
            float z = d > 0.0 ? vec_r[lane] / d : 0.0;
            vec_p[lane] = z + float(scalars[3]) * vec_p[lane];
        }
    }
    if (pc.mode <= 2) {
        emitPartial(contribution);
    }
}
