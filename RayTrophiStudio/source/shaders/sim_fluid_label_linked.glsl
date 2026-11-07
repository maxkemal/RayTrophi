// Shared 80-byte PC and linked-bin storage. One next index per live particle,
// not 64 reserved indices per spatial cell. Separate entry-point names prevent
// legacy fixed-slot SPIR-V from binding the smaller next-index buffer.
layout(local_size_x = 256) in;
layout(push_constant) uniform PC {
    ivec4 bin_dim_count; ivec4 bin_min_cap;
    int bin_count; float radius; uint label_mask; uint label_shift;
    uint frozen_mask; uint body_label; uint spray_label; uint frozen_label;
    float mist_mass_fraction; uint mist_label; uvec2 padding;
} pc;
const uint LINKED_LABEL_ABI = 1u;
const uint EMPTY = 0xffffffffu;

uint labelInvocation() {
    return gl_GlobalInvocationID.x +
        gl_GlobalInvocationID.y * gl_NumWorkGroups.x * 256u;
}

#if defined(LABEL_LINK_CLEAR)
layout(set=0, binding=0) buffer Heads { uint heads[]; };
layout(set=0, binding=1) buffer Stats { uint stats[]; };
void main() {
    if (pc.padding.x != LINKED_LABEL_ABI) return;
    uint id = labelInvocation();
    if (id < uint(pc.bin_count)) heads[id] = EMPTY;
    if (id < 4u) stats[id] = 0u;
    if (id == 0u) stats[4] = 1u;
}
#else
layout(set=0, binding=0) readonly buffer Positions { float positions[]; };
#if defined(LABEL_LINK_SCATTER)
layout(set=0, binding=1) buffer Heads { uint heads[]; };
layout(set=0, binding=2) buffer Next { uint next_index[]; };
layout(set=0, binding=3) buffer Stats { uint stats[]; };
layout(set=0, binding=4) readonly buffer MassFraction { float mass_fraction[]; };
#else
layout(set=0, binding=1) buffer Flags { uint flags[]; };
layout(set=0, binding=2) readonly buffer Heads { uint heads[]; };
layout(set=0, binding=3) readonly buffer Next { uint next_index[]; };
layout(set=0, binding=4) buffer Stats { uint stats[]; };
layout(set=0, binding=5) readonly buffer MassFraction { float mass_fraction[]; };
#endif

vec3 labelPosition(uint id) {
    return vec3(positions[id * 3u], positions[id * 3u + 1u], positions[id * 3u + 2u]);
}

bool validLabelBin(ivec3 bin) {
    return all(greaterThanEqual(bin, ivec3(0))) &&
        all(lessThan(bin, pc.bin_dim_count.xyz));
}

uint labelBinIndex(ivec3 bin) {
    return uint(bin.x + bin.y * pc.bin_dim_count.x +
        bin.z * pc.bin_dim_count.x * pc.bin_dim_count.y);
}

#if defined(LABEL_LINK_SCATTER)
void main() {
    if (pc.padding.x != LINKED_LABEL_ABI) return;
    uint id = labelInvocation();
    if (id == 0u) atomicOr(stats[4], 2u);
    if (id >= uint(pc.bin_dim_count.w) || !(pc.radius > 0.0)) return;
    float mass = mass_fraction[id];
    if (mass > 0.0 && mass <= pc.mist_mass_fraction) return;
    vec3 p = labelPosition(id);
    if (any(isnan(p)) || any(isinf(p))) return;
    ivec3 bin = ivec3(floor(p / pc.radius)) - pc.bin_min_cap.xyz;
    if (!validLabelBin(bin)) return;
    uint previous = atomicExchange(heads[labelBinIndex(bin)], id);
    next_index[id] = previous;
    if (previous == EMPTY) atomicAdd(stats[1], 1u);
}
#else
uint gather(uint own, vec3 p, ivec3 bin, uint found) {
    if (!validLabelBin(bin)) return found;
    uint other = heads[labelBinIndex(bin)];
    uint visited = 0u;
    while (other != EMPTY && found < 6u) {
        // A malformed chain rejects this GPU result instead of dropping a
        // neighbour or hanging the invocation. Publication then uses CPU labels.
        if (other >= uint(pc.bin_dim_count.w) || visited++ >= uint(pc.bin_dim_count.w)) {
            atomicOr(stats[0], 1u);
            return found;
        }
        if (other != own) {
            vec3 delta = p - labelPosition(other);
            if (dot(delta, delta) <= pc.radius * pc.radius) ++found;
        }
        other = next_index[other];
    }
    return found;
}

void main() {
    if (pc.padding.x != LINKED_LABEL_ABI) return;
    uint id = labelInvocation();
    if (id == 0u) atomicOr(stats[4], 4u);
    if (id >= uint(pc.bin_dim_count.w)) return;
    uint previous = (flags[id] & pc.label_mask) >> pc.label_shift;
    uint label = 0u;
    vec3 p = labelPosition(id);
    if ((flags[id] & pc.frozen_mask) != 0u) {
        label = pc.frozen_label;
    } else if (mass_fraction[id] > 0.0 && mass_fraction[id] <= pc.mist_mass_fraction) {
        label = pc.mist_label;
    } else if (!any(isnan(p)) && !any(isinf(p)) && pc.radius > 0.0) {
        ivec3 center = ivec3(floor(p / pc.radius)) - pc.bin_min_cap.xyz;
        uint found = gather(id, p, center, 0u);
        if (found >= 6u) atomicAdd(stats[3], 1u);
        for (int z = -1; z <= 1 && found < 6u; ++z)
        for (int y = -1; y <= 1 && found < 6u; ++y)
        for (int x = -1; x <= 1 && found < 6u; ++x) {
            if (x == 0 && y == 0 && z == 0) continue;
            found = gather(id, p, center + ivec3(x, y, z), found);
        }
        label = found <= 2u ? pc.spray_label : found >= 6u ? pc.body_label :
            previous == pc.spray_label ? pc.spray_label : pc.body_label;
    }
    if (label != previous) atomicAdd(stats[2], 1u);
    flags[id] = (flags[id] & ~pc.label_mask) | ((label << pc.label_shift) & pc.label_mask);
}
#endif
#endif
