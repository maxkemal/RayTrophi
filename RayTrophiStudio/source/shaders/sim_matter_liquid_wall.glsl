// sim_matter_liquid_wall.glsl - the matter LIQUID lane's collider wall
// (docs/dev/COLLIDER_TEMAS_YUZEYI.md, "Sivi seridi").
//
// Replaced sim_fluid_zero_solid_faces on the liquid lane. That kernel zeroed
// every face next to a collider cell after P2G AND again right before G2P, so
// G2P read a moving collider as a still wall: a rising sphere lifted ~0 of a
// chocolate pile (2026-11-08, h 0.1 m, 1.25 m/s; the liquid's top stayed below
// the sphere every frame). And the projection, which fixes a closed face at the
// solid's velocity, could pull without limit: removing only the zeroing would
// have glued every liquid, water included, to anything that moves away.
//
// The separating wall (Bridson, "Fluid Simulation for CG", inequality
// boundary): a liquid touching a solid may leave it, never enter it. Per
// collider face, from the liquid's own face velocity u (P2G + viscosity) and
// the solid's vs, s = +1 when the solid is the low cell:
//   r = s * (u - vs)                  > 0: the liquid moves away from the solid
//   hold = adhesion * dt / (rho * h)  the velocity the surface's wetting holds
//   r <= hold: held     -> face closed (weight 0); the projection sets it to vs
//   r >  hold: released -> face OPEN (weight 1) with u = vs + s * (r - hold);
//              the variational matrix counts an open face to a non-fluid cell
//              as a p = 0 ghost, i.e. the gap behind a leaving solid is air.
// Collider faces always carry weight 0 from the host's binary pass
// (ParticleSimulation.cpp: faces of a solid cell), so writing 0 or 1 here each
// substep is exact and needs no restore.
//
// mode 0 (after P2G and before G2P): domain walls 0, faces inside a collider
//   (both cells solid) the solid's velocity; collider faces keep their value --
//   after P2G the liquid's own (the FLIP baseline then holds it, so the wall
//   impulse reaches the parcels as post - pre), before G2P the projection's.
// mode 1 (inside the pressure solve, after the solid weights are uploaded and
//   before the divergence): the decision above.
// mode 3 (before mode 1, glued colliders only): the ADHESION BAND. A binary
//   voxel collider moving 5 cm a frame at h 0.1 m opens an air cell between
//   the liquid and its surface, and an air face cannot hold: 1 MPa lifted no
//   chocolate (2026-11-08 live, the same failure the MPM lane had on 11-02).
//   An air cell whose centre is within h of a glued surface (solid_phi < h)
//   and that touches liquid is marked solid (-2) for this solve, carrying the
//   velocity and adhesion of the solid cell across the surface (half a cell
//   past phi = 0 along -grad phi). Mode 1 then holds or releases its faces
//   like any collider face. Unglued colliders get no band: water lets go.
// mode 2 (after the solve, every exit): band cells back to air (mask 0).
// Dense or compact (RT_SPARSE_MAC) storage; the same address serves velocity
// and weight pages. Vulkan only.
layout(local_size_x = 256) in;
layout(push_constant) uniform PC {
    int nx; int ny; int nz;
    int mode;
    float dt;
    float density;   // liquid density, kg/m^3
    float h;
    int boundary;    // 0 open, 1 closed, 2 periodic (walls of 1 are solid)
} pc;
layout(set = 0, binding = 0) buffer VelX { float vel_x[]; };
layout(set = 0, binding = 1) buffer VelY { float vel_y[]; };
layout(set = 0, binding = 2) buffer VelZ { float vel_z[]; };
layout(set = 0, binding = 3) buffer Mask { float fluid_mask[]; };
layout(set = 0, binding = 4) buffer UW { float uw[]; };
layout(set = 0, binding = 5) buffer VW { float vw[]; };
layout(set = 0, binding = 6) buffer WW { float ww[]; };
layout(set = 0, binding = 7) buffer SVX { float svx[]; };
layout(set = 0, binding = 8) buffer SVY { float svy[]; };
layout(set = 0, binding = 9) buffer SVZ { float svz[]; };
layout(set = 0, binding = 10) buffer Glue { float contact_glue[]; };
layout(set = 0, binding = 11) readonly buffer Phi { float solid_phi[]; };
#ifdef RT_SPARSE_MAC
layout(set = 0, binding = 12) readonly buffer MacTileMap { uint mac_tile_map[]; };
layout(set = 0, binding = 13) readonly buffer MacTileList { uint mac_tile_list[]; };
#endif
const float kFar = 1.0e29;
const float kBandMark = -2.0;
#include "sim_mac_lane.glsl"

bool outsideCell(ivec3 c) {
    return any(lessThan(c, ivec3(0))) || any(greaterThanEqual(c, ivec3(pc.nx, pc.ny, pc.nz)));
}
int cellId(ivec3 c) {
    return (c.z * pc.ny + c.y) * pc.nx + c.x;
}
bool solidCell(ivec3 c) {
    return fluid_mask[cellId(c)] < -0.5;
}
bool fluidCell(ivec3 c) {
    return fluid_mask[cellId(c)] > 0.5;
}
float solidVel(ivec3 c, int component) {
    int q = cellId(c);
    return component == 0 ? svx[q] : (component == 1 ? svy[q] : svz[q]);
}
float readVel(int component, uint a) {
    return component == 0 ? vel_x[a] : (component == 1 ? vel_y[a] : vel_z[a]);
}
void writeVel(int component, uint a, float v) {
    if (component == 0) vel_x[a] = v;
    else if (component == 1) vel_y[a] = v;
    else vel_z[a] = v;
}
void writeWeight(int component, uint a, float w) {
    if (component == 0) uw[a] = w;
    else if (component == 1) vw[a] = w;
    else ww[a] = w;
}

void face(int component, ivec3 f, uint a) {
    ivec3 e = ivec3(0);
    e[component] = 1;
    ivec3 low = f - e;
    bool low_out = outsideCell(low);
    bool high_out = outsideCell(f);
    if (low_out || high_out) {
        // Domain wall: the old rule. Periodic seams are not walls.
        if (pc.mode == 0 && pc.boundary == 1) {
            writeVel(component, a, 0.0);
        }
        return;
    }
    bool solid_low = solidCell(low);
    bool solid_high = solidCell(f);
    if (solid_low && solid_high) {
        if (pc.mode == 0) {
            writeVel(component, a, 0.5 * (solidVel(low, component) + solidVel(f, component)));
        }
        return;
    }
    if (!solid_low && !solid_high) {
        return;
    }
    if (pc.mode == 0) {
        return;
    }
    ivec3 solid = solid_low ? low : f;
    ivec3 other = solid_low ? f : low;
    if (!fluidCell(other)) {
        // Air beside the collider: nothing to hold, keep the face closed.
        writeWeight(component, a, 0.0);
        return;
    }
    float s = solid_low ? 1.0 : -1.0;
    float vs = solidVel(solid, component);
    float r = s * (readVel(component, a) - vs);
    float glue = max(contact_glue[cellId(solid)], 0.0);
    float hold = glue * pc.dt / (max(pc.density, 1.0) * max(pc.h, 1e-6));
    if (fluid_mask[cellId(solid)] < -1.5 && r <= 0.0) {
        // A band cell is air: liquid closing the gap meets no wall yet. Only
        // a liquid leaving a glued surface is held across it.
        writeWeight(component, a, 1.0);
        return;
    }
    if (r <= hold) {
        writeWeight(component, a, 0.0);
        writeVel(component, a, vs);
    } else {
        writeWeight(component, a, 1.0);
        writeVel(component, a, vs + s * (r - hold));
    }
}

float phiAt(ivec3 c) {
    c = clamp(c, ivec3(0), ivec3(pc.nx - 1, pc.ny - 1, pc.nz - 1));
    float v = solid_phi[cellId(c)];
    return v >= kFar ? 2.0 * pc.h : v;
}

void bandCell(int id) {
    ivec3 c = ivec3(id % pc.nx, (id / pc.nx) % pc.ny, id / (pc.nx * pc.ny));
    float m = fluid_mask[id];
    if (pc.mode == 2) {
        if (m > kBandMark - 0.5 && m < kBandMark + 0.5) {
            fluid_mask[id] = 0.0;
        }
        return;
    }
    // Air only: not solid, not liquid.
    if (m < -0.5 || m > 0.5) {
        return;
    }
    float phi = solid_phi[id];
    if (!(phi < pc.h)) {
        return;
    }
    bool touches_liquid = false;
    for (int axis = 0; axis < 3; ++axis) {
        for (int side = -1; side <= 1; side += 2) {
            ivec3 n = c;
            n[axis] += side;
            if (!outsideCell(n) && fluidCell(n)) {
                touches_liquid = true;
            }
        }
    }
    if (!touches_liquid) {
        return;
    }
    vec3 grad = vec3(phiAt(c + ivec3(1, 0, 0)) - phiAt(c - ivec3(1, 0, 0)),
                     phiAt(c + ivec3(0, 1, 0)) - phiAt(c - ivec3(0, 1, 0)),
                     phiAt(c + ivec3(0, 0, 1)) - phiAt(c - ivec3(0, 0, 1)));
    float gl = length(grad);
    if (gl < 1e-6) {
        return;
    }
    vec3 n = grad / gl;
    ivec3 wall = ivec3(floor(vec3(c) + 0.5 - n * (max(phi, 0.0) / pc.h + 0.5)));
    if (outsideCell(wall) || fluid_mask[cellId(wall)] > -0.5 ||
        fluid_mask[cellId(wall)] < -1.5) {
        return;  // not a real collider cell (or itself a band cell)
    }
    int q = cellId(wall);
    float glue = contact_glue[q];
    if (!(glue > 0.0)) {
        return;
    }
    fluid_mask[id] = kBandMark;
    svx[id] = svx[q];
    svy[id] = svy[q];
    svz[id] = svz[q];
    contact_glue[id] = glue;
}

void main() {
    // groups256 may fold a large count into y.
    uint lane = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * gl_NumWorkGroups.x * 256u;
    if (pc.mode >= 2) {
        if (lane < uint(pc.nx * pc.ny * pc.nz)) {
            bandCell(int(lane));
        }
        return;
    }
    for (int component = 0; component < 3; ++component) {
        ivec3 f;
        uint a;
        if (macLaneFace(lane, component, f, a)) {
            face(component, f, a);
        }
    }
}
