layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    int nx; int ny; int nz; int boundary;
    float voxel_size; float dt; float sor_omega;
    int iterations; int parity;
    float density_correction; int particles_per_cell;
    int variational; int gfm_active;
    uint tiles_x; uint tiles_y; uint tiles_z;
    uint tile_count; uint compact_count; uint partial_count; uint reserved;
} pc;

layout(set = 0, binding = 0) buffer TileMap { uint tile_map[]; };
layout(set = 0, binding = 1) buffer TileList { uint tile_list[]; };
layout(set = 0, binding = 2) buffer Pressure { float pressure[]; };
layout(set = 0, binding = 3) buffer Residual { float residual[]; };
layout(set = 0, binding = 4) buffer Z { float z[]; };
layout(set = 0, binding = 5) buffer Search { float search[]; };
layout(set = 0, binding = 6) buffer As { float product[]; };
layout(set = 0, binding = 7) buffer Diag { float diagonal[]; };
layout(set = 0, binding = 8) buffer Partials { double partials[]; };
layout(set = 0, binding = 9) buffer Scalars { double scalars[]; };
layout(set = 0, binding = 10) readonly buffer Mask { float mask[]; };
layout(set = 0, binding = 11) readonly buffer Divergence { float divergence[]; };
layout(set = 0, binding = 12) readonly buffer UW { float uw[]; };
layout(set = 0, binding = 13) readonly buffer VW { float vw[]; };
layout(set = 0, binding = 14) readonly buffer WW { float ww[]; };
layout(set = 0, binding = 15) buffer DensePressure { float dense_pressure[]; };

const uint EMPTY = 0xffffffffu;
const uint LOCKED = 0xfffffffeu;
shared double reduction[256];

uint laneIndex() {
    uint group = gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x;
#ifdef SPARSE_CLEAR
    uint count = pc.tile_count;
#elif defined(SPARSE_MARK) || defined(SPARSE_DENSE_CLEAR)
    uint count = uint(pc.nx) * uint(pc.ny) * uint(pc.nz);
#else
    uint count = pc.compact_count;
#endif
    if (group >= (count + 255u) / 256u) {
        return 0xffffffffu;
    }
    return group * 256u + gl_LocalInvocationID.x;
}

uint groupIndex() {
    return gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x;
}

uint denseIndex(ivec3 cell) {
    return uint(cell.x) + uint(pc.nx) * (uint(cell.y) + uint(pc.ny) * uint(cell.z));
}

bool inside(ivec3 cell) {
    return all(greaterThanEqual(cell, ivec3(0))) &&
        all(lessThan(cell, ivec3(pc.nx, pc.ny, pc.nz)));
}

uint tileIndex(ivec3 cell) {
    uvec3 tile = uvec3(cell) / 8u;
    return tile.x + pc.tiles_x * (tile.y + pc.tiles_y * tile.z);
}

ivec3 compactCell(uint lane) {
    uint tile = tile_list[lane / 512u + 1u];
    uvec3 coordinate = uvec3(tile % pc.tiles_x,
        (tile / pc.tiles_x) % pc.tiles_y, tile / (pc.tiles_x * pc.tiles_y));
    uint local = lane % 512u;
    return ivec3(coordinate * 8u + uvec3(local % 8u, (local / 8u) % 8u, local / 64u));
}

uint compactIndex(ivec3 cell) {
    uint slot = tile_map[tileIndex(cell)];
    if (slot == EMPTY || slot == LOCKED) {
        return EMPTY;
    }
    uvec3 local = uvec3(cell) % 8u;
    return slot * 512u + local.x + 8u * (local.y + 8u * local.z);
}

float faceWeight(ivec3 cell, int axis, int direction) {
    ivec3 face = cell;
    if (direction > 0) {
        ++face[axis];
    }
    int dimension = axis == 0 ? pc.nx : (axis == 1 ? pc.ny : pc.nz);
    if (face[axis] <= 0 || face[axis] >= dimension) {
        return pc.boundary == 0 ? 1.0 : 0.0;
    }
    if (axis == 0) {
        return uw[face.x + (pc.nx + 1) * (face.y + pc.ny * face.z)];
    }
    if (axis == 1) {
        return vw[face.x + pc.nx * (face.y + (pc.ny + 1) * face.z)];
    }
    return ww[face.x + pc.nx * (face.y + pc.ny * face.z)];
}

void emitReduction(double contribution) {
    uint local = gl_LocalInvocationID.x;
    reduction[local] = contribution;
    barrier();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (local < stride) {
            reduction[local] += reduction[local + stride];
        }
        barrier();
    }
    if (local == 0u && groupIndex() < pc.partial_count) {
        partials[groupIndex()] = reduction[0];
    }
}

void main() {
    uint lane = laneIndex();
#ifdef SPARSE_DENSE_CLEAR
    if (lane < uint(pc.nx) * uint(pc.ny) * uint(pc.nz)) {
        dense_pressure[lane] = 0.0;
    }
#endif
#ifdef SPARSE_CLEAR
    if (lane < pc.tile_count) {
        tile_map[lane] = EMPTY;
    }
    if (lane == 0u) {
        tile_list[0] = 0u;
    }
#endif
#ifdef SPARSE_MARK
    uint count = uint(pc.nx) * uint(pc.ny) * uint(pc.nz);
    if (lane >= count || mask[lane] < 0.5) {
        return;
    }
    ivec3 cell = ivec3(lane % uint(pc.nx), (lane / uint(pc.nx)) % uint(pc.ny),
                      lane / (uint(pc.nx) * uint(pc.ny)));
    uint tile = tileIndex(cell);
    if (atomicCompSwap(tile_map[tile], EMPTY, LOCKED) == EMPTY) {
        uint slot = atomicAdd(tile_list[0], 1u);
        tile_list[slot + 1u] = tile;
        tile_map[tile] = slot;
    }
#endif
#ifdef SPARSE_INIT
    if (lane >= pc.compact_count) {
        return;
    }
    pressure[lane] = 0.0;
    residual[lane] = 0.0;
    diagonal[lane] = 0.0;
    ivec3 cell = compactCell(lane);
    if (!inside(cell) || mask[denseIndex(cell)] < 0.5) {
        return;
    }
    float diag = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
        for (int direction = -1; direction <= 1; direction += 2) {
            ivec3 neighbour = cell;
            neighbour[axis] += direction;
            if (pc.variational != 0) {
                diag += faceWeight(cell, axis, direction);
            } else if (inside(neighbour)) {
                diag += mask[denseIndex(neighbour)] > -0.5 ? 1.0 : 0.0;
            } else {
                diag += pc.boundary == 0 ? 1.0 : 0.0;
            }
        }
    }
    diagonal[lane] = pc.variational != 0 && diag < 1e-6 ? 1.0 : diag;
    uint id = denseIndex(cell);
    float h = pc.voxel_size > 1e-6 ? pc.voxel_size : 1.0;
    float inv_dt = pc.dt > 1e-8 ? 1.0 / pc.dt : 0.0;
    float rhs = -divergence[id] * h * h * inv_dt;
    if (pc.density_correction > 0.0 && pc.particles_per_cell > 0) {
        float over = mask[id] - float(pc.particles_per_cell);
        if (over > 0.0) {
            rhs += pc.density_correction * h * inv_dt * over / float(pc.particles_per_cell);
        }
    }
    residual[lane] = rhs;
#endif
#ifdef SPARSE_JACOBI
    double contribution = 0.0LF;
    if (lane < pc.compact_count) {
        float value = diagonal[lane] > 0.5 ? residual[lane] / diagonal[lane] : 0.0;
        z[lane] = value;
        contribution = double(residual[lane]) * double(value);
    }
    emitReduction(contribution);
#endif
#ifdef SPARSE_COPY
    if (lane < pc.compact_count) {
        search[lane] = z[lane];
    }
#endif
#ifdef SPARSE_SPMV
    double contribution = 0.0LF;
    if (lane < pc.compact_count) {
        float value = 0.0;
        ivec3 cell = compactCell(lane);
        if (inside(cell) && mask[denseIndex(cell)] >= 0.5) {
            float sum = 0.0;
            for (int axis = 0; axis < 3; ++axis) {
                for (int direction = -1; direction <= 1; direction += 2) {
                    ivec3 neighbour = cell;
                    neighbour[axis] += direction;
                    if (!inside(neighbour) || mask[denseIndex(neighbour)] <= 0.5) {
                        continue;
                    }
                    uint index = compactIndex(neighbour);
                    // MARK covers every fluid cell; a missing slot cannot be air.
                    if (index != EMPTY) {
                        float weight = pc.variational != 0
                            ? faceWeight(cell, axis, direction) : 1.0;
                        sum += weight * search[index];
                    }
                }
            }
            value = diagonal[lane] * search[lane] - sum;
        }
        product[lane] = value;
        contribution = double(search[lane]) * double(value);
    }
    emitReduction(contribution);
#endif
#ifdef SPARSE_AXPY
    if (lane < pc.compact_count && scalars[6] == 0.0LF) {
        float alpha = float(scalars[3]);
        pressure[lane] += alpha * search[lane];
        residual[lane] -= alpha * product[lane];
    }
#endif
#ifdef SPARSE_ZPBY
    if (lane < pc.compact_count) {
        search[lane] = z[lane] + float(scalars[4]) * search[lane];
    }
#endif
#ifdef SPARSE_SCATTER
    if (lane < pc.compact_count) {
        ivec3 cell = compactCell(lane);
        if (inside(cell)) {
            dense_pressure[denseIndex(cell)] = pressure[lane];
        }
    }
#endif
}
