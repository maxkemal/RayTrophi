#pragma once

#include "Fluid/DomainWallSettings.h"
#include <cstdint>

struct FluidAdvectTailGpuConstants {
    int nx = 0, ny = 0, nz = 0, particle_count = 0;
    int boundary = 0;
    float origin_x = 0.0f, origin_y = 0.0f, origin_z = 0.0f;
    float voxel_size = 1.0f, dt = 0.0f;
    float velocity_damping = 1.0f, max_velocity = 0.0f;
    float wall_damping = 0.0f, air_drag = 0.0f;
    int air_threshold = 1, substeps = 1;
    float air_wind_x = 0.0f, air_wind_y = 0.0f, air_wind_z = 0.0f, pad = 0.0f;
    float wall_friction = 0.0f, gravity_x = 0.0f, gravity_y = 0.0f, gravity_z = 0.0f;
    float wall_adhesion_acceleration = 0.0f;
};
static_assert(sizeof(FluidAdvectTailGpuConstants) == 100);

struct MpmBoundaryConstants {
    int nx, ny, nz;
    uint32_t flags;
    float dt, density, h, friction;
    float restitution, gravity_x, gravity_y, gravity_z;
    float adhesion_pa;
};
static_assert(sizeof(MpmBoundaryConstants) == 52);
