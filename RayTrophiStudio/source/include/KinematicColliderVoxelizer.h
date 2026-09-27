#pragma once

#include "KinematicColliderSource.h"

#include <cstdint>
#include <string>
#include <vector>

namespace FluidSim {
class FluidGrid;
}

namespace RayTrophiSim {

// What one solver step stamped for one proxy in one domain. The velocity is
// the one written into solid_vel during that step, not a later re-sample, and
// a proxy that stamped nothing is still recorded with zero cells.
struct KinematicStampRecord {
    std::string domain;
    uint32_t consumer_mask = 0;
    int frame = 0;
    uint64_t set_id = 0;
    uint64_t proxy_id = 0;
    std::string proxy_name;
    uint32_t stamped_cells = 0;
    Vec3 linear_velocity = Vec3(0.0f);
    Vec3 angular_velocity = Vec3(0.0f);
    bool velocity_valid = false;
};

void prepareKinematicColliderGrid(FluidSim::FluidGrid& grid);

// stamped_cells, when given, receives one count per sample (same order).
bool voxelizeKinematicColliders(
    FluidSim::FluidGrid& grid,
    const std::vector<KinematicProxySample>& samples,
    uint32_t consumer_mask,
    std::vector<uint32_t>* stamped_cells = nullptr);

// Appends one record per sample this consumer_mask selects.
void appendKinematicStampRecords(
    std::vector<KinematicStampRecord>& log,
    const std::string& domain,
    uint32_t consumer_mask,
    int frame,
    const std::vector<KinematicProxySample>& samples,
    const std::vector<uint32_t>& stamped_cells);

} // namespace RayTrophiSim
