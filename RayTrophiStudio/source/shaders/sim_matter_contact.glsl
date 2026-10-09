#include "sim_dispatch.glsl"
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer V0X { float v0x[]; };
layout(set = 0, binding = 1) buffer V0Y { float v0y[]; };
layout(set = 0, binding = 2) buffer V0Z { float v0z[]; };
layout(set = 0, binding = 3) buffer V1X { float v1x[]; };
layout(set = 0, binding = 4) buffer V1Y { float v1y[]; };
layout(set = 0, binding = 5) buffer V1Z { float v1z[]; };
layout(set = 0, binding = 6) readonly buffer M0X { float m0x[]; };
layout(set = 0, binding = 7) readonly buffer M0Y { float m0y[]; };
layout(set = 0, binding = 8) readonly buffer M0Z { float m0z[]; };
layout(set = 0, binding = 9) readonly buffer M1X { float m1x[]; };
layout(set = 0, binding = 10) readonly buffer M1Y { float m1y[]; };
layout(set = 0, binding = 11) readonly buffer M1Z { float m1z[]; };
layout(set = 0, binding = 12) readonly buffer G0X { float g0x[]; };
layout(set = 0, binding = 13) readonly buffer G0Y { float g0y[]; };
layout(set = 0, binding = 14) readonly buffer G0Z { float g0z[]; };
layout(set = 0, binding = 15) readonly buffer G1X { float g1x[]; };
layout(set = 0, binding = 16) readonly buffer G1Y { float g1y[]; };
layout(set = 0, binding = 17) readonly buffer G1Z { float g1z[]; };
layout(set = 0, binding = 18) buffer Stats { uint pair_count[]; };
layout(push_constant) uniform Params {
    int nx; int ny; int nz; float friction;
} pc;
#ifdef RT_SPARSE_MAC
// Liquid lane (0) velocity and mass on compact pages at 0..2 and 6..8; the
// granular lane stays dense (docs/dev/MATTER_SPARSE_S1_SIVI_GPU.md).
layout(set = 0, binding = 19) readonly buffer MacTileMap { uint mac_tile_map[]; };
layout(set = 0, binding = 20) readonly buffer MacTileList { uint mac_tile_list[]; };
#endif
#include "sim_mac_lane.glsl"

// Each lattice tuple owns exactly one X, Y and Z face. No invocation writes
// another tuple's face. The inverse-mass diagonal handles staggered masses;
// each impulse is equal/opposite and dissipative in that face-mass metric.
void main() {
    uint count = uint((pc.nx + 1) * (pc.ny + 1) * (pc.nz + 1));
    uint id = simLane256(count);
    if (id >= count) return;
    int i = int(id) % (pc.nx + 1);
    int j = (int(id) / (pc.nx + 1)) % (pc.ny + 1);
    int k = int(id) / ((pc.nx + 1) * (pc.ny + 1));
    ivec3 face = ivec3(i + j * (pc.nx + 1) + k * (pc.nx + 1) * pc.ny,
        i + j * pc.nx + k * pc.nx * (pc.ny + 1),
        i + j * pc.nx + k * pc.nx * pc.ny);
    bvec3 valid = bvec3(j < pc.ny && k < pc.nz,
        i < pc.nx && k < pc.nz, i < pc.nx && j < pc.ny);
    // Lane-0 storage index of each face (= the dense index when dense). A
    // liquid face without a resident page has no liquid mass: no contact.
    uvec3 liquid = uvec3(MAC_ABSENT);
    for (int axis = 0; axis < 3; ++axis) {
        if (valid[axis]) {
            liquid[axis] = macAddress(axis, i, j, k);
            valid[axis] = liquid[axis] != MAC_ABSENT;
        }
    }
    vec3 ml = vec3(0.0), mg = vec3(0.0), vl = vec3(0.0), vg = vec3(0.0);
    vec3 normal = vec3(0.0);
    if (valid[0]) {
        ml[0] = m0x[liquid[0]];
        mg[0] = m1x[face[0]];
        if (ml[0] > 1e-8 && mg[0] > 1e-8) {
            vl[0] = v0x[liquid[0]];
            vg[0] = v1x[face[0]];
            normal[0] = g1x[face[0] * 3 + 0] / mg[0] -
                g0x[face[0] * 3 + 0] / ml[0];
        } else {
            valid[0] = false;
        }
    }
    if (valid[1]) {
        ml[1] = m0y[liquid[1]];
        mg[1] = m1y[face[1]];
        if (ml[1] > 1e-8 && mg[1] > 1e-8) {
            vl[1] = v0y[liquid[1]];
            vg[1] = v1y[face[1]];
            normal[1] = g1y[face[1] * 3 + 1] / mg[1] -
                g0y[face[1] * 3 + 1] / ml[1];
        } else {
            valid[1] = false;
        }
    }
    if (valid[2]) {
        ml[2] = m0z[liquid[2]];
        mg[2] = m1z[face[2]];
        if (ml[2] > 1e-8 && mg[2] > 1e-8) {
            vl[2] = v0z[liquid[2]];
            vg[2] = v1z[face[2]];
            normal[2] = g1z[face[2] * 3 + 2] / mg[2] -
                g0z[face[2] * 3 + 2] / ml[2];
        } else {
            valid[2] = false;
        }
    }
    float normal_length = length(normal);
    if (normal_length < 1e-8 || any(isnan(normal)) || any(isinf(normal))) return;
    normal /= normal_length;
    vec3 inverse_mass = vec3(0.0);
    for (int axis = 0; axis < 3; ++axis) {
        if (valid[axis]) inverse_mass[axis] = 1.0 / ml[axis] + 1.0 / mg[axis];
    }
    vec3 relative = vl - vg;
    float closing = dot(relative, normal);
    if (closing <= 0.0) return;
    float denominator = dot(normal * normal, inverse_mass);
    if (denominator <= 0.0) return;
    float normal_impulse = closing / denominator;
    vec3 impulse = normal * normal_impulse;
    vec3 after_normal = relative - inverse_mass * impulse;
    vec3 tangent = after_normal - normal * dot(after_normal, normal);
    float tangent_speed = length(tangent);
    if (tangent_speed > 1e-8) {
        vec3 direction = tangent / tangent_speed;
        float tangent_denominator = dot(direction * direction, inverse_mass);
        if (tangent_denominator > 0.0) {
            float tangent_impulse = min(tangent_speed / tangent_denominator,
                max(pc.friction, 0.0) * normal_impulse);
            impulse += direction * tangent_impulse;
        }
    }
    if (any(isnan(impulse)) || any(isinf(impulse))) return;
    if (valid[0]) {
        v0x[liquid[0]] = vl[0] - impulse[0] / ml[0];
        v1x[face[0]] = vg[0] + impulse[0] / mg[0];
    }
    if (valid[1]) {
        v0y[liquid[1]] = vl[1] - impulse[1] / ml[1];
        v1y[face[1]] = vg[1] + impulse[1] / mg[1];
    }
    if (valid[2]) {
        v0z[liquid[2]] = vl[2] - impulse[2] / ml[2];
        v1z[face[2]] = vg[2] + impulse[2] / mg[2];
    }
    atomicAdd(pair_count[0], 1u);
}
