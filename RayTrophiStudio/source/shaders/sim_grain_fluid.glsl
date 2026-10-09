// Current-geometry CFD-DEM. The same trilinear lump partition and Di Felice
// drag as MatterGrainCoupling.cpp, with exact opposite support for each tick.
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer FluidVelocity { float velocity[]; };
layout(std430, binding = 1) buffer Coupling { vec4 coupling[]; };
layout(std430, binding = 2) readonly buffer GrainMass { float grain_mass[]; };
layout(std430, binding = 3) readonly buffer Parcels { uvec4 parcels[]; };
layout(std430, binding = 4) buffer Heads { uint heads[]; };
layout(std430, binding = 5) buffer Links { uint links[]; };
layout(std430, binding = 6) buffer GrainCells { uvec2 grain_cells[]; };
layout(std430, binding = 7) buffer Field { vec4 field[]; };
layout(std430, binding = 8) buffer Reaction { uint reaction_data[]; };
layout(std430, binding = 9) buffer Previous { vec4 previous[]; };
layout(std430, binding = 10) buffer Delta { vec4 delta[]; };
layout(std430, binding = 11) buffer Impulses { vec4 impulses[]; };
layout(std430, binding = 12) buffer Buoyancy { vec4 buoyancy[]; };
layout(std430, binding = 13) readonly buffer FluidPositions { float fp[]; };
layout(std430, binding = 14) readonly buffer GrainPositions { float gp[]; };
layout(std430, binding = 15) readonly buffer GrainScratch { float scratch[]; };
layout(std430, binding = 16) buffer AuxField { uint auxiliary_data[]; };
layout(std430, binding = 17) buffer TotalGain { vec4 total_gain[]; };
layout(std430, binding = 18) readonly buffer External { vec4 external_acceleration[]; };
layout(std430, binding = 19) buffer Metrics { vec4 metrics[]; };
layout(std430, binding = 20) readonly buffer GrainVelocity { float grain_velocity[]; };
layout(push_constant) uniform Constants {
    uvec4 counts; // fluid count, buckets per owner, grain count, tick
    vec4 step;
    vec4 origin_h;
    vec4 material; // r, sphere volume, minimum voidage
    uvec4 dimensions;
    vec4 gravity;
} pc;
const uint EMPTY = 0xffffffffu;
vec3 position(uint i) {
    if (i < pc.counts.x) {
        uint b = 3u * parcels[i].x;
        return vec3(fp[b], fp[b + 1u], fp[b + 2u]);
    }
    uint g = i - pc.counts.x;
    uint b = 3u * g;
    return (pc.counts.w & 1u) == 0u ? vec3(gp[b], gp[b + 1u], gp[b + 2u])
        : vec3(scratch[b], scratch[b + 1u], scratch[b + 2u]);
}
vec3 grainVelocity(uint g) {
    uint b = 3u * g;
    if ((pc.counts.w & 1u) == 0u) {
        return vec3(grain_velocity[b], grain_velocity[b + 1u], grain_velocity[b + 2u]);
    }
    b += 3u * pc.counts.z;
    return vec3(scratch[b], scratch[b + 1u], scratch[b + 2u]);
}
ivec3 cell(uint i) { return ivec3(floor((position(i) - pc.origin_h.xyz) / pc.origin_h.w)); }
bool inside(ivec3 c) { return all(greaterThanEqual(c, ivec3(0))) &&
    all(lessThan(c, ivec3(pc.dimensions.xyz))); }
uint hash(ivec3 c) {
    uvec3 u = uvec3(c);
    return ((u.x * 73856093u) ^ (u.y * 19349663u) ^ (u.z * 83492791u)) &
        (pc.counts.y - 1u);
}
uint firstFluid(ivec3 c) {
    uint j = heads[hash(c)];
    while (j != EMPTY && !all(equal(cell(j), c))) j = links[j];
    return j;
}
uint firstGrain(ivec3 c) {
    uint j = heads[pc.counts.y + hash(c)];
    while (j != EMPTY && !all(equal(cell(j), c))) j = links[j];
    return j;
}
uint fieldLeader(ivec3 c) {
    uint first = firstFluid(c);
    return first != EMPTY ? first : firstGrain(c);
}
vec4 auxiliary(uint i) {
    uint b = 4u * i;
    return uintBitsToFloat(uvec4(auxiliary_data[b], auxiliary_data[b + 1u],
        auxiliary_data[b + 2u], auxiliary_data[b + 3u]));
}
// Core float CAS is portable to the existing Vulkan path. No float-atomic
// extension, retry ceiling or neighbour truncation changes the physical sum.
void addSolid(uint address, float value) {
    uint before = atomicAdd(auxiliary_data[address], 0u);
    for (;;) {
        uint wanted = floatBitsToUint(uintBitsToFloat(before) + value);
        uint actual = atomicCompSwap(auxiliary_data[address], before, wanted);
        if (actual == before) return;
        before = actual;
    }
}
void addReaction(uint address, float value) {
    uint before = atomicAdd(reaction_data[address], 0u);
    for (;;) {
        uint wanted = floatBitsToUint(uintBitsToFloat(before) + value);
        uint actual = atomicCompSwap(reaction_data[address], before, wanted);
        if (actual == before) return;
        before = actual;
    }
}
float weight(vec3 p, ivec3 c) {
    vec3 local = (p - pc.origin_h.xyz) / pc.origin_h.w - 0.5;
    vec3 w = max(vec3(0.0), vec3(1.0) - abs(local - vec3(c)));
    return w.x * w.y * w.z;
}
float solidVolume(ivec3 c) {
    float volume = 0.0;
    for (int z = -1; z <= 1; ++z) for (int y = -1; y <= 1; ++y)
        for (int x = -1; x <= 1; ++x) {
            ivec3 near = c + ivec3(x, y, z);
            uint j = heads[pc.counts.y + hash(near)];
            while (j != EMPTY) {
                if (all(equal(cell(j), near))) volume += weight(position(j), c) * pc.material.y;
                j = links[j];
            }
        }
    return volume;
}
float dragCoefficient(float speed, float density, float viscosity, float voidage) {
    float d = 2.0 * pc.material.x;
    float re = max(density * voidage * d * max(speed, 0.0) / viscosity, 1e-6);
    float corrected_speed = re * viscosity / (density * voidage * d);
    float cd = pow(0.63 + 4.8 / sqrt(re), 2.0);
    float chi = 3.7 - 0.65 * exp(-0.5 * pow(1.5 - log(re) / log(10.0), 2.0));
    return 0.5 * cd * density * (0.25 * 3.141592654 * d * d) *
        pow(voidage, 2.0 - chi) * corrected_speed;
}
void main() {
    uint i = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * gl_NumWorkGroups.x * 256u;
#ifdef FLUID_CLEAR
    if (pc.counts.w == 0u) {
        if (i < 2u * pc.counts.y) heads[i] = EMPTY;
    } else if (i < pc.counts.x + pc.counts.z) {
        uint previous_bucket = links[pc.counts.x + pc.counts.z + i];
        if (previous_bucket != EMPTY) atomicExchange(heads[previous_bucket], EMPTY);
    }
#elif defined(FLUID_HASH)
    if (i >= pc.counts.x + pc.counts.z) return;
    ivec3 c = cell(i);
    links[i] = EMPTY;
    uint bucket = EMPTY;
    if (i >= pc.counts.x || inside(c)) {
        bucket = (i < pc.counts.x ? 0u : pc.counts.y) + hash(c);
        links[i] = atomicExchange(heads[bucket], i);
    }
    links[pc.counts.x + pc.counts.z + i] = bucket;
#elif defined(FLUID_CELLS)
    if (i >= pc.counts.x + pc.counts.z) return;
    ivec3 c = cell(i);
    if (!inside(c) || fieldLeader(c) != i) return;
    vec3 momentum = vec3(0.0);
    float mass = 0.0, volume = 0.0, viscous = 0.0;
    uint j = heads[hash(c)];
    while (j != EMPTY) {
        if (all(equal(cell(j), c))) {
            uvec4 parcel = parcels[j];
            float m = uintBitsToFloat(parcel.y);
            uint b = 3u * parcel.x;
            momentum += m * vec3(velocity[b], velocity[b + 1u], velocity[b + 2u]);
            mass += m;
            volume += m / uintBitsToFloat(parcel.z);
            viscous += m * uintBitsToFloat(parcel.w);
        }
        j = links[j];
    }
    field[i] = vec4(mass > 0.0 ? momentum / mass : vec3(0.0), mass);
    uint b = 4u * i;
    auxiliary_data[b] = floatBitsToUint(volume);
    auxiliary_data[b + 1u] = floatBitsToUint(viscous);
    auxiliary_data[b + 2u] = 0u;
    auxiliary_data[b + 3u] = 0u;
    for (uint axis = 0u; axis < 4u; ++axis) reaction_data[b + axis] = 0u;
#elif defined(FLUID_SOLID)
    if (i >= pc.counts.z) return;
    vec3 p = position(pc.counts.x + i);
    ivec3 base = ivec3(floor((p - pc.origin_h.xyz) / pc.origin_h.w - 0.5));
    for (uint n = 0u; n < 8u; ++n) {
        ivec3 c = base + ivec3(n & 1u, (n >> 1u) & 1u, (n >> 2u) & 1u);
        if (!inside(c)) continue;
        uint leader = fieldLeader(c);
        float w = weight(p, c);
        if (leader != EMPTY && w > 0.0) addSolid(4u * leader + 2u, w * pc.material.y);
    }
#elif defined(FLUID_REFRESH)
    if (i >= pc.counts.z) return;
    vec3 p = position(pc.counts.x + i);
    ivec3 base = ivec3(floor((p - pc.origin_h.xyz) / pc.origin_h.w - 0.5));
    bool has_liquid = false;
    for (uint n = 0u; n < 8u; ++n) {
        ivec3 c = base + ivec3(n & 1u, (n >> 1u) & 1u, (n >> 2u) & 1u);
        if (inside(c) && firstFluid(c) != EMPTY && weight(p, c) > 0.0) has_liquid = true;
    }
    if (!has_liquid) {
        coupling[3u * i] = vec4(0.0);
        coupling[3u * i + 1u] = vec4(external_acceleration[i].xyz, 0.0);
        buoyancy[i] = metrics[i] = vec4(0.0);
        for (uint n = 0u; n < 8u; ++n) grain_cells[8u * i + n] = uvec2(EMPTY, 0u);
        return;
    }
    float weights = 0.0, mass = 0.0, volume = 0.0, solid = 0.0, wet = 0.0, mu = 0.0;
    float lump = 0.0;
    vec3 momentum = vec3(0.0);
    float shares[8];
    float cv = pc.origin_h.w * pc.origin_h.w * pc.origin_h.w;
    for (uint n = 0u; n < 8u; ++n) {
        ivec3 c = base + ivec3(n & 1u, (n >> 1u) & 1u, (n >> 2u) & 1u);
        grain_cells[8u * i + n] = uvec2(EMPTY, 0u);
        shares[n] = 0.0;
        if (!inside(c)) continue;
        float w = weight(p, c);
        weights += w;
        uint leader = fieldLeader(c);
        if (leader == EMPTY) {
            solid += w * solidVolume(c);
            continue;
        }
        vec4 f = field[leader], a = auxiliary(leader);
        solid += w * a.z;
        if (f.w <= 0.0) continue;
        mass += w * f.w;
        volume += w * a.x;
        mu += w * a.y;
        float pores = clamp(1.0 - a.z / cv, pc.material.z, 1.0) * cv;
        wet += w * min(1.0, a.x / (0.5 * pores));
        float share = w * pc.material.y / max(a.z, pc.material.y) * f.w;
        shares[n] = share;
        lump += share;
        momentum += share * f.xyz;
        grain_cells[8u * i + n] = uvec2(leader, 0u);
    }
    buoyancy[i] = vec4(0.0);
    metrics[i] = vec4(0.0);
    if (weights > 0.0 && lump > 0.0 && volume > 0.0) {
        vec3 sampled = momentum / lump;
        float density = mass / volume;
        float voidage = clamp(1.0 - solid / (weights * cv), pc.material.z, 1.0);
        float submerged = clamp(wet / weights, 0.0, 1.0);
        float beta = submerged * dragCoefficient(length(sampled - grainVelocity(i)),
            density, mu / mass, voidage);
        vec3 lift = -pc.gravity.xyz * (submerged * density * pc.material.y / grain_mass[i]);
        coupling[3u * i] = vec4(sampled, beta);
        coupling[3u * i + 1u] = vec4(lift + external_acceleration[i].xyz, lump);
        buoyancy[i] = vec4(lift, 0.0);
        metrics[i] = vec4(submerged, beta, lump, 0.0);
        for (uint n = 0u; n < 8u; ++n)
            grain_cells[8u * i + n].y = floatBitsToUint(shares[n] / lump);
    } else {
        coupling[3u * i] = vec4(0.0);
        coupling[3u * i + 1u] = vec4(external_acceleration[i].xyz, 0.0);
    }
#elif defined(FLUID_DELTA)
    if (i >= pc.counts.z) return;
    vec3 drag = coupling[3u * i + 2u].xyz;
    vec3 before = pc.counts.w == 0u ? vec3(0.0) : previous[i].xyz;
    vec3 value = drag - before + buoyancy[i].xyz * grain_mass[i] * pc.step.x;
    delta[i] = vec4(value, 0.0);
    previous[i] = vec4(drag, 0.0);
    total_gain[i] = vec4(value + (pc.counts.w == 0u ? vec3(0.0) : total_gain[i].xyz), 0.0);
#elif defined(FLUID_REACTION)
    if (i >= pc.counts.z) return;
    for (uint n = 0u; n < 8u; ++n) {
        uvec2 edge = grain_cells[8u * i + n];
        if (edge.x == EMPTY) continue;
        vec3 value = -uintBitsToFloat(edge.y) * delta[i].xyz;
        for (uint axis = 0u; axis < 3u; ++axis) {
            if (value[axis] != 0.0) addReaction(4u * edge.x + axis, value[axis]);
        }
    }
#elif defined(FLUID_APPLY)
    if (i >= pc.counts.x) return;
    ivec3 c = cell(i);
    uint leader = inside(c) ? firstFluid(c) : EMPTY;
    uint b = 3u * parcels[i].x;
    vec3 before = vec3(velocity[b], velocity[b + 1u], velocity[b + 2u]);
    vec3 after = before + (leader != EMPTY && field[leader].w > 0.0
        ? uintBitsToFloat(uvec3(reaction_data[4u * leader], reaction_data[4u * leader + 1u],
            reaction_data[4u * leader + 2u])) / field[leader].w : vec3(0.0));
    for (uint axis = 0u; axis < 3u; ++axis) velocity[b + axis] = after[axis];
    vec3 actual = (after - before) * uintBitsToFloat(parcels[i].y);
    impulses[i] = vec4(actual + (pc.counts.w == 0u ? vec3(0.0) : impulses[i].xyz), 0.0);
#endif
}
