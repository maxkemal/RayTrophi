#pragma once

#include "Fluid/DomainVisualReseed.h"
#include "Fluid/FluidParticles.h"

#include <unordered_map>

namespace RayTrophiSim {
struct SimulationGridDomainDesc;
struct SimulationGridDomainState;
namespace Fluid {

// Explicit CPU reconstruction route, not the solver's projected MAC buffer.
// One instance contains exactly one source/substance/mechanics stream.
class DomainVisualField final : public DomainFields::Snapshot {
public:
    struct Hash {
        std::size_t operator()(const std::array<int, 3>& cell) const;
    };
    struct Cell {
        double volume = 0.0;
        std::array<double, 3> momentum{};
    };

    DomainVisualField(const SimulationGridDomainDesc& domain,
        const SimulationGridDomainState& state, const std::vector<std::size_t>& indices,
        uint64_t stream, uint64_t byte_budget);
    const DomainFields::Identity& identity() const override {
        return identity_;
    }
    const DomainFields::Geometry& geometry() const override {
        return geometry_;
    }
    DomainFields::ChannelDescription describe(DomainFields::Channel channel) const override;
    DomainFields::Scalar read(DomainFields::Channel channel, int x, int y, int z) const override;
    std::vector<DomainFields::SampleRegion> regions(uint64_t stream) const;
    DomainFields::MatterSupport support(const std::array<double, 3>& position) const;
    std::size_t cellCount() const {
        return cells_.size();
    }

private:
    DomainFields::Identity identity_;
    DomainFields::Geometry geometry_;
    bool closed_ = true;
    std::unordered_map<std::array<int, 3>, Cell, Hash> cells_;
};

} // namespace Fluid
} // namespace RayTrophiSim
