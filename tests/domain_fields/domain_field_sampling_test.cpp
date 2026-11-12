#include "Fluid/DomainFieldSampling.h"

#include <cstdlib>
#include <iostream>

using namespace RayTrophiSim::Fluid::DomainFields;

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(1);
    }
}

// Analytic field, independent of dense/sparse storage. Negative world origin
// and non-unit spacing detect incorrect local/staggered coordinate conversion.
class AnalyticSnapshot final : public Snapshot {
public:
    Identity id;
    Geometry grid{{-4.0, -3.0, -2.0}, {8, 8, 8}, 0.25};
    bool background = false;
    bool missing = false;
    bool device = false;
    bool nonfinite = false;
    bool reject_zero_weight_corner = false;

    const Identity& identity() const override { return id; }
    const Geometry& geometry() const override { return grid; }

    ChannelDescription describe(Channel channel) const override {
        const int axis = static_cast<int>(channel);
        constexpr Location locations[] = {Location::FaceX, Location::FaceY, Location::FaceZ};
        return {locations[axis], Unit::LengthPerSecond,
                device ? Residence::Device : Residence::Host, 0.0, true};
    }

    Scalar read(Channel channel, int x, int y, int z) const override {
        if (missing || (reject_zero_weight_corner && x > 2)) {
            return {};
        }
        if (background) {
            return {0.0, Validity::KnownBackground};
        }
        if (nonfinite) {
            return {std::numeric_limits<double>::quiet_NaN(), Validity::ValidData};
        }
        const int axis = static_cast<int>(channel);
        const int indices[] = {x, y, z};
        std::array<double, 3> p{};
        for (int a = 0; a < 3; ++a) {
            p[a] = grid.origin[a] + (indices[a] + (a == axis ? 0.0 : 0.5)) *
                grid.voxel_size;
        }
        // Affine shear plus solid-body rotation and translation.
        const double velocity[] = {2.0 - p[1] + 0.3 * p[2],
                                   -1.0 + p[0], 0.5 + 0.2 * p[0]};
        return {velocity[axis], Validity::ValidData};
    }
};

} // namespace

int main() {
    AnalyticSnapshot snapshot;
    for (int i = 0; i < 20; ++i) {
        const std::array<double, 3> p{-3.8 + i * 0.06, -2.8 + i * 0.05,
                                     -1.8 + i * 0.04};
        const auto result = sampleMacVelocity(snapshot, p);
        require(result.validity == Validity::ValidData, "affine field unavailable");
        const double expected[] = {2.0 - p[1] + 0.3 * p[2], -1.0 + p[0],
                                   0.5 + 0.2 * p[0]};
        for (int axis = 0; axis < 3; ++axis) {
            require(std::abs(result.value[axis] - expected[axis]) < 1e-12,
                    "staggered affine interpolation differs from analytic field");
        }
    }
    const std::array<double, 3> p{-3.5, -2.5, -1.5};
    snapshot.background = true;
    require(sampleMacVelocity(snapshot, p).validity == Validity::KnownBackground,
            "known zero background lost");
    snapshot.missing = true;
    const auto missing = sampleMacVelocity(snapshot, p);
    require(missing.validity == Validity::Unavailable && missing.value[0] == 0.0,
            "missing stencil leaked a partial velocity");
    snapshot.missing = false;
    snapshot.device = true;
    require(sampleMacVelocity(snapshot, p).validity == Validity::Unavailable,
            "device-only snapshot silently used as host data");
    snapshot.device = false;
    snapshot.background = false;
    snapshot.nonfinite = true;
    require(sampleMacVelocity(snapshot, p).validity == Validity::Unavailable,
            "non-finite face accepted");
    snapshot.nonfinite = false;
    require(sampleMacVelocity(snapshot, {-2.0, -2.5, -1.5}).validity ==
                Validity::OutsideDomain, "upper domain boundary must be exclusive");
    require(sampleMacVelocity(snapshot, {-4.01, -2.5, -1.5}).validity ==
                Validity::OutsideDomain, "outside point accepted");
    snapshot.reject_zero_weight_corner = true;
    require(sampleMacVelocity(snapshot, p).validity == Validity::ValidData,
            "zero-weight unavailable corner invalidated exact lattice sampling");
    snapshot.grid.voxel_size = 0.0;
    require(sampleMacVelocity(snapshot, p).validity == Validity::Unavailable,
            "invalid geometry accepted");
    std::cout << "domain field sampling PASS\n";
}
