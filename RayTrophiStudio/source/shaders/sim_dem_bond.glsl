// Independent speculative DEM frame buffers. Edge evaluation has one writer;
// endpoint gather has one writer per particle. No float atomics or CPU substeps.
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer Positions { float positions[]; };
layout(std430, binding = 1) buffer Velocities { float velocities[]; };
layout(std430, binding = 2) buffer Affines { float affines[]; };
layout(std430, binding = 3) buffer Scratch { float scratch[]; };
layout(std430, binding = 4) readonly buffer Mass { float masses[]; };
layout(std430, binding = 5) buffer Poses { vec4 poses[]; };
struct Edge {
    uvec4 meta;
    vec4 rest;
    vec4 qa;
    vec4 qb;
    vec4 material;
    vec4 history;
};
layout(std430, binding = 6) buffer Edges { Edge edges[]; };
layout(std430, binding = 7) readonly buffer Adjacency { uint adjacency[]; };
layout(std430, binding = 8) buffer Loads { vec4 loads[]; };
layout(std430, binding = 9) buffer Status { uint status; };
layout(push_constant) uniform Constants {
    uvec4 meta; // particles, edges, substep, reserved
    vec4 step; // dt
} pc;
#include "sim_dispatch.glsl"

bool finite3(vec3 v) { return !any(isnan(v)) && !any(isinf(v)); }
bool finite4(vec4 v) { return !any(isnan(v)) && !any(isinf(v)); }
vec4 conjq(vec4 q) { return vec4(q.x, -q.yzw); }
vec4 mulq(vec4 a, vec4 b) {
    return vec4(a.x * b.x - dot(a.yzw, b.yzw),
        a.x * b.yzw + b.x * a.yzw + cross(a.yzw, b.yzw));
}
vec3 rotateq(vec4 q, vec3 v) {
    vec3 t = 2.0 * cross(q.yzw, v);
    return v + q.x * t + cross(q.yzw, t);
}
vec3 position(uint i, uint bank) {
    uint b = 3u * i;
    return bank == 0u ? vec3(positions[b], positions[b+1u], positions[b+2u])
        : vec3(scratch[b], scratch[b+1u], scratch[b+2u]);
}
vec3 velocity(uint i, uint bank) {
    uint b = 3u * i;
    if (bank == 0u) {
        return vec3(velocities[b], velocities[b+1u], velocities[b+2u]);
    }
    b += 3u * pc.meta.x;
    return vec3(scratch[b], scratch[b+1u], scratch[b+2u]);
}
vec3 omega(uint i, uint bank) {
    if (bank == 0u) {
        uint b = 9u * i;
        return 0.5 * vec3(affines[b+5u]-affines[b+7u],
            affines[b+6u]-affines[b+2u], affines[b+1u]-affines[b+3u]);
    }
    uint b = 6u * pc.meta.x + 3u * i;
    return vec3(scratch[b], scratch[b+1u], scratch[b+2u]);
}
void storeMotion(uint i, uint bank, vec3 v, vec3 w) {
    if (bank == 0u) {
        uint b = 3u * i;
        velocities[b] = v.x; velocities[b+1u] = v.y; velocities[b+2u] = v.z;
        b = 9u * i;
        affines[b] = 0.0; affines[b+1u] = w.z; affines[b+2u] = -w.y;
        affines[b+3u] = -w.z; affines[b+4u] = 0.0; affines[b+5u] = w.x;
        affines[b+6u] = w.y; affines[b+7u] = -w.x; affines[b+8u] = 0.0;
    } else {
        uint b = 3u * pc.meta.x + 3u * i;
        scratch[b] = v.x; scratch[b+1u] = v.y; scratch[b+2u] = v.z;
        b += 3u * pc.meta.x;
        scratch[b] = w.x; scratch[b+1u] = w.y; scratch[b+2u] = w.z;
    }
}

void main() {
#ifdef DEM_BOND_EVALUATE
    uint e = simLane256(pc.meta.y);
    if (e >= pc.meta.y) { return; }
    loads[2u * e] = vec4(0.0);
    loads[2u * e + 1u] = vec4(0.0);
    Edge edge = edges[e];
    if (edge.meta.z != 0u) { return; }
    vec4 da = mulq(poses[edge.meta.x], conjq(edge.qa));
    vec4 db = mulq(poses[edge.meta.y], conjq(edge.qb));
    if (dot(da, db) < 0.0) { db = -db; }
    vec4 frame = normalize(da + db);
    vec3 d = position(edge.meta.y, pc.meta.z & 1u) -
             position(edge.meta.x, pc.meta.z & 1u);
    vec3 displacement = rotateq(conjq(frame), d) - edge.rest.xyz;
    vec3 axis = edge.rest.xyz / edge.rest.w;
    float extension = dot(displacement, axis);
    vec3 shear = displacement - extension * axis;
    vec4 relative = mulq(conjq(da), db);
    float sine = length(relative.yzw);
    float angle = 2.0 * atan(sine, max(0.0, relative.x));
    vec3 rotation = relative.yzw * (sine > 1e-6 ? angle / sine : 2.0);
    float twist = dot(rotation, axis);
    vec3 bend = rotation - twist * axis;
    float radius = edge.material.z;
    float area = 3.141592653589793 * radius * radius;
    float moment = area * radius * radius / 4.0;
    float polar = 2.0 * moment;
    float young = edge.material.x;
    float g = young / (2.0 * (1.0 + edge.material.y));
    float kn = young * area / edge.rest.w;
    float ks = g * area / edge.rest.w;
    float kb = young * moment / edge.rest.w;
    float kt = g * polar / edge.rest.w;
    vec3 force = rotateq(frame, kn * extension * axis + ks * shear);
    vec3 couple = rotateq(frame, kt * twist * axis + kb * bend);
    float tensile = max(0.0, kn * extension / area + kb * length(bend) * radius / moment);
    float tau = ks * length(shear) / area + kt * abs(twist) * radius / polar;
    float energy = 0.5 * (kn * extension * extension + ks * dot(shear, shear) +
        kb * dot(bend, bend) + kt * twist * twist);
    if (!finite3(force) || !finite3(couple) ||
        !finite4(vec4(tensile, tau, energy, area))) {
        atomicOr(status, 1u);
        return;
    }
    if (tensile > edge.material.w || tau > edge.history.x) {
        edges[e].meta.z = tensile > edge.material.w ? 1u : 2u;
        edges[e].history.z += energy;
        edges[e].history.y = 0.0;
        if (!finite4(edges[e].history)) { atomicOr(status, 1u); }
        return;
    }
    edges[e].history.y = energy;
    loads[2u * e] = vec4(force, 0.0);
    loads[2u * e + 1u] = vec4(couple, 0.0);
#else
    uint i = simLane256(pc.meta.x);
    if (i >= pc.meta.x) { return; }
#ifdef DEM_BOND_KICK
    if (adjacency[i] == adjacency[i+1u]) { return; }
    uint bank = pc.meta.z & 1u;
    vec3 force = vec3(0.0), torque = vec3(0.0);
    for (uint slot = adjacency[i]; slot < adjacency[i+1u]; ++slot) {
        uint e = adjacency[slot];
        Edge edge = edges[e];
        vec3 f = loads[2u * e].xyz;
        vec3 moment = loads[2u * e + 1u].xyz;
        vec3 d = position(edge.meta.y, bank) - position(edge.meta.x, bank);
        bool first = edge.meta.x == i;
        force += first ? f : -f;
        torque += 0.5 * cross(d, f) + (first ? moment : -moment);
    }
    float mass = masses[i], radius = masses[pc.meta.x + i];
    vec3 v = velocity(i, bank) + pc.step.x * force / mass;
    vec3 w = omega(i, bank) + pc.step.x * torque / (0.4 * mass * radius * radius);
    if (!finite3(v) || !finite3(w)) { atomicOr(status, 1u); return; }
    storeMotion(i, bank, v, w);
#else
    // Canonical members retain their spin history after their last bond breaks.
    // Unbound particles have a zero transient quaternion and need no update.
    if (dot(poses[i], poses[i]) < 0.5) { return; }
    vec3 w = omega(i, (pc.meta.z & 1u) ^ 1u);
    float speed = length(w);
    float half_angle = 0.5 * pc.step.x * speed;
    vec4 increment = vec4(cos(half_angle),
        w * (speed > 1e-8 ? sin(half_angle) / speed : 0.5 * pc.step.x));
    vec4 q = normalize(mulq(increment, poses[i]));
    if (!finite4(q)) { atomicOr(status, 1u); return; }
    poses[i] = q;
#endif
#endif
}
