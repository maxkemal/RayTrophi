#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>

namespace RayTrophiSim::Fluid::DomainFields {

enum class Validity { ValidData, KnownBackground, Unavailable, OutsideDomain };
enum class Lane { Liquid, Mpm, Dem, Gas };
enum class Residence { Host, Device, Both };
enum class Location { Cell, FaceX, FaceY, FaceZ };
enum class Unit { LengthPerSecond, VolumeFraction, MassPerVolume, Kelvin, Length };
enum class Channel { VelocityX, VelocityY, VelocityZ, VolumeFraction, Density, Temperature,
                     SolidPhi };

struct Identity {
    std::string domain_uid;
    Lane lane = Lane::Liquid;
    std::uint64_t state_generation = 0;
    std::uint64_t topology_generation = 0;
    std::uint64_t transform_revision = 0;
    std::uint64_t step = 0;
    std::uint32_t substep = 0;
    double physical_time = 0.0;
};

struct Geometry {
    std::array<double, 3> origin{};
    std::array<int, 3> cells{};
    double voxel_size = 0.0;
};

struct ChannelDescription {
    Location location = Location::Cell;
    Unit unit = Unit::VolumeFraction;
    Residence residence = Residence::Host;
    double background = 0.0;
    bool available = false;
};

struct Scalar {
    double value = 0.0;
    Validity validity = Validity::Unavailable;
};

inline bool usable(Validity validity) {
    return validity == Validity::ValidData || validity == Validity::KnownBackground;
}

// Internal publication contract, not a live FluidGrid view. Implementations must
// own immutable storage/topology and retain it for the lifetime of this handle.
// A device producer publishes only after its completion fence; read() must not
// trigger an implicit readback. Missing host data returns Unavailable.
// Face coordinates are lattice indices, including explicitly supplied halos.
class Snapshot {
public:
    virtual ~Snapshot() = default;
    virtual const Identity& identity() const = 0;
    virtual const Geometry& geometry() const = 0;
    virtual ChannelDescription describe(Channel channel) const = 0;
    virtual Scalar read(Channel channel, int x, int y, int z) const = 0;
};

using SnapshotHandle = std::shared_ptr<const Snapshot>;

} // namespace RayTrophiSim::Fluid::DomainFields
