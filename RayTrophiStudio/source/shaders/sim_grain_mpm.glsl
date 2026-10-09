// Bidirectional Jacobi contact. Both owners gather the same pair impulse from
// immutable velocities, then apply in a separate dispatch. No floating atomics,
// host pairs, full state transfers, or liquid drag on continuum skeletons.
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer GrainPosition { float gx[]; };
layout(std430, binding = 1) buffer GrainVelocity { float gv[]; };
layout(std430, binding = 2) buffer GrainScratch { float scratch[]; };
layout(std430, binding = 3) readonly buffer GrainMass { float gm[]; };
layout(std430, binding = 4) readonly buffer MpmPosition { float mx[]; };
layout(std430, binding = 5) buffer MpmVelocity { float mv[]; };
layout(std430, binding = 6) readonly buffer Metadata { uvec4 metadata[]; };
layout(std430, binding = 7) buffer Heads { uint heads[]; };
layout(std430, binding = 8) buffer Links { uint links[]; };
layout(std430, binding = 9) buffer Delta { vec4 delta[]; };
layout(std430, binding = 10) buffer Impulses { vec4 impulses[]; };
layout(std430, binding = 11) buffer Diagnostics { uint stats[]; };
layout(std430, binding = 12) buffer Degrees { uint degrees[]; };
layout(push_constant) uniform Constants {
    uvec4 meta; // grains, MPM parcels, buckets per owner, grain read bank
    vec4 params; // grain radius, hash cell size, friction, frame dt
} pc;
const uint EMPTY = 0xffffffffu;

uint count() { return pc.meta.x + pc.meta.y; }
uint owner(uint i) { return i < pc.meta.x ? 0u : 1u; }
uint canonical(uint i) { return metadata[i - pc.meta.x].z; }
vec3 position(uint i) {
    uint b = 3u * i;
    if (owner(i) == 0u) {
        if (pc.meta.w == 0u) return vec3(gx[b], gx[b + 1u], gx[b + 2u]);
        return vec3(scratch[b], scratch[b + 1u], scratch[b + 2u]);
    }
    b = 3u * canonical(i);
    return vec3(mx[b], mx[b + 1u], mx[b + 2u]);
}
vec3 velocity(uint i) {
    uint b = 3u * i;
    if (owner(i) == 0u) {
        if (pc.meta.w == 0u) return vec3(gv[b], gv[b + 1u], gv[b + 2u]);
        b += 3u * pc.meta.x;
        return vec3(scratch[b], scratch[b + 1u], scratch[b + 2u]);
    }
    b = 3u * canonical(i);
    return vec3(mv[b], mv[b + 1u], mv[b + 2u]);
}
float mass(uint i) {
    return owner(i) == 0u ? gm[i] : uintBitsToFloat(metadata[i - pc.meta.x].x);
}
float radius(uint i) {
    return owner(i) == 0u ? pc.params.x
        : uintBitsToFloat(metadata[i - pc.meta.x].y);
}
ivec3 cell(uint i) { return ivec3(floor(position(i) / pc.params.y)); }
uint hash(ivec3 c) {
    uvec3 u = uvec3(c);
    return ((u.x * 73856093u) ^ (u.y * 19349663u) ^ (u.z * 83492791u)) &
        (pc.meta.z - 1u);
}
// Called with grain first on BOTH sides. Unilateral, dissipative normal
// impulse plus Coulomb friction; penetration bias resolves overlap gently.
// Relaxation uses both measured graph degrees, without a neighbour ceiling.
vec3 pairImpulse(uint grain, uint mpm) {
    vec3 distance = position(grain) - position(mpm);
    float distance_length = length(distance);
    float overlap = radius(grain) + radius(mpm) - distance_length;
    if (overlap <= 0.0) return vec3(0.0);
    vec3 normal = distance_length > 1e-8 ? distance / distance_length : vec3(0.0, 1.0, 0.0);
    vec3 relative = velocity(grain) - velocity(mpm);
    float vn = dot(relative, normal);
    float effective = 1.0 / (1.0 / mass(grain) + 1.0 / mass(mpm));
    float bias = min(0.2 * overlap, 0.1 * pc.params.x) / pc.params.w;
    float relaxation = 1.0 / float(max(1u, max(degrees[grain], degrees[mpm])));
    float jn = max(bias - vn, 0.0) * effective * relaxation;
    vec3 tangent = relative - vn * normal;
    float speed = length(tangent);
    vec3 jt = speed > 1e-8 ? -tangent / speed *
        min(effective * speed * relaxation, pc.params.z * jn) : vec3(0.0);
    return jn * normal + jt;
}

void main() {
    uint i = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * gl_NumWorkGroups.x * 256u;
#ifdef CONTACT_INIT
    if (i < count()) impulses[i] = vec4(0.0);
    if (i < 8u) stats[i] = i == 1u ? 2u : 0u;
#elif defined(CONTACT_CLEAR)
    if (i < 2u * pc.meta.z) heads[i] = EMPTY;
#elif defined(CONTACT_HASH)
    if (i >= count()) return;
    links[i] = atomicExchange(heads[owner(i) * pc.meta.z + hash(cell(i))], i);
#elif defined(CONTACT_COUNT) || defined(CONTACT_GATHER)
    if (i >= count()) return;
    ivec3 here = cell(i);
    vec3 impulse = vec3(0.0);
    uint contacts = 0u;
    uint events = 0u;
    for (int z = -1; z <= 1; ++z) {
        for (int y = -1; y <= 1; ++y) {
            for (int x = -1; x <= 1; ++x) {
                ivec3 neighbour = here + ivec3(x, y, z);
                uint j = heads[(1u - owner(i)) * pc.meta.z + hash(neighbour)];
                while (j != EMPTY) {
                    if (all(equal(cell(j), neighbour)) &&
                        length(position(i) - position(j)) < radius(i) + radius(j)) {
                        ++contacts;
#ifdef CONTACT_GATHER
                        vec3 pair = owner(i) == 0u ? pairImpulse(i, j) : -pairImpulse(j, i);
                        impulse += pair;
                        if (dot(pair, pair) > 0.0) ++events;
#endif
                    }
                    j = links[j];
                }
            }
        }
    }
#ifdef CONTACT_COUNT
    degrees[i] = contacts;
#else
    atomicMax(stats[3], contacts);
    if (owner(i) == 0u && atomicAdd(stats[2], events) > 0xffffffffu - events) {
        atomicAdd(stats[4], 1u);
    }
    delta[i] = vec4(impulse, 0.0);
#endif
#elif defined(CONTACT_APPLY)
    if (i >= count()) return;
    vec3 before = velocity(i);
    vec3 value = before + delta[i].xyz / mass(i);
    uint b = 3u * i;
    if (owner(i) == 0u) {
        if (pc.meta.w == 0u) {
            for (uint k = 0u; k < 3u; ++k) gv[b + k] = value[k];
        } else {
            b += 3u * pc.meta.x;
            for (uint k = 0u; k < 3u; ++k) scratch[b + k] = value[k];
        }
    } else {
        b = 3u * canonical(i);
        for (uint k = 0u; k < 3u; ++k) mv[b + k] = value[k];
    }
    impulses[i] += vec4((value - before) * mass(i), 0.0);
#endif
}
