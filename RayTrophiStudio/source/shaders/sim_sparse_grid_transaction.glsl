#include "sim_dispatch.glsl"
#include "sim_sparse_mac_address.glsl"
layout(local_size_x = 256) in;
layout(push_constant) uniform PC {
    int nx; int ny; int nz; int face_axis;
    uint tiles_x; uint tiles_y; uint tiles_z; uint old_active;
    uint new_active; uint page_values; float background; uint padding;
} pc;
layout(set = 0, binding = 0) readonly buffer OldMap { uint old_map[]; };
layout(set = 0, binding = 1) buffer NewMap { uint new_map[]; };
layout(set = 0, binding = 2) readonly buffer OldField { float old_field[]; };
layout(set = 0, binding = 3) buffer NewField { float new_field[]; };
layout(set = 0, binding = 4) readonly buffer OldKeys { uint old_keys[]; };
layout(set = 0, binding = 5) readonly buffer NewKeys { uint new_keys[]; };
layout(set = 0, binding = 6) buffer Validation { uint invalid[]; };

void main() {
#ifdef RT_GRID_MAP_CLEAR
    uint count = pc.tiles_x * pc.tiles_y * pc.tiles_z;
    uint lane = simLane256(count);
    if (lane >= count) return;
    new_map[lane] = 0u;
#elif defined(RT_GRID_MAP_SEED)
    uint lane = simLane256(pc.new_active);
    if (lane >= pc.new_active) return;
    new_map[new_keys[lane]] = lane + 1u;
#elif defined(RT_GRID_RETIRE)
    uint count = pc.old_active * pc.page_values;
    uint lane = simLane256(count);
    if (lane >= count) return;
    if (isnan(old_field[lane]) || isinf(old_field[lane])) {
        atomicOr(invalid[0], 2u);
        return;
    }
    uint key = old_keys[lane / pc.page_values];
    if (new_map[key] == 0u && old_field[lane] != pc.background) {
        atomicOr(invalid[0], 1u);
    }
#elif defined(RT_GRID_REMAP)
    uint count = pc.new_active * pc.page_values;
    uint lane = simLane256(count);
    if (lane >= count) return;
    uint key = new_keys[lane / pc.page_values];
    uint old_slot = old_map[key];
    // Page strides/location are unchanged by a topology transaction. Slot
    // identities may change; physical key identity may not.
    new_field[lane] = old_slot == 0u ? pc.background
        : old_field[(old_slot - 1u) * pc.page_values + lane % pc.page_values];
#endif
}
