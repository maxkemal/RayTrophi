#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace RayTrophiSim {
struct SimulationGridDomainDesc;
namespace Fluid {
class FluidParticles;
struct MatterWetPaletteEntry {
    uint32_t substance_tag = 0;
    int dry_material = -1;
    std::array<int, 8> materials{};
};
struct MatterWetPalette {
    std::vector<MatterWetPaletteEntry> entries;
    bool legacy_granular = false;
    float appearance_full_saturation = 1.0f;
    void appendMaterialKeys(std::vector<int>& keys) const;
    std::size_t sourceIndex(const FluidParticles& particles, std::size_t particle,
                           const std::vector<int>& material_keys,
                           std::size_t dry_source) const;
};
MatterWetPalette prepareMatterWetPalette(const SimulationGridDomainDesc& domain,
                                        const FluidParticles& particles);
} // namespace Fluid
} // namespace RayTrophiSim
