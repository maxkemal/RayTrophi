#pragma once

#include "Fluid/FluidViewResolver.h"
#include "Vec3.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace RayTrophiSim::Fluid {

// Internal field preparation only. No VDB registration, materials or solver writes.
enum class FogFieldKind : uint8_t {
    Carrier,
    GasAggregate
};

struct FogFieldKey {
    FogFieldKind kind = FogFieldKind::Carrier;
    uint32_t substance_tag = 0;
    MatterOutputPhase phase = MatterOutputPhase::Unknown;
    bool operator<(const FogFieldKey& other) const;
    bool operator==(const FogFieldKey& other) const;
};

struct FogFieldGrid {
    int nx = 0;
    int ny = 0;
    int nz = 0;
    float voxel = 0.0f;
    Vec3 origin = Vec3(0.0f);
};

enum class GasFogTemperatureUnits : uint8_t {
    Kelvin,
    Normalized3000
};

struct GasFogFieldView {
    FogFieldGrid grid;
    const std::vector<float>* density = nullptr;
    const std::vector<float>* temperature = nullptr;
    GasFogTemperatureUnits temperature_units = GasFogTemperatureUnits::Kelvin;
    // Caller must explicitly identify an authoritative render snapshot. This
    // helper must never read a stale GPU-resident host mirror as current gas.
    bool authoritative = false;
};

struct MatterFogFieldInput {
    const FluidParticles* particles = nullptr;
    const FluidViewPlan* plan = nullptr;
    FogFieldGrid carrier_grid;
    float parcel_density = 1.0f;
    uint32_t default_substance_tag = 0;
    const GasFogFieldView* gas = nullptr;
    std::size_t max_streams = 32;
    // Required explicit budget for retained dense fields and peak gather scratch.
    // Does not include preexisting input/output storage or stream metadata allocations.
    std::size_t max_dense_bytes = 0;
};

struct MatterFogField {
    FogFieldKey key;
    FogFieldGrid grid;
    std::size_t carrier_count = 0;
    std::vector<float> density;
    // Physical values only. Empty means unavailable, not cold or an artistic override.
    std::vector<float> temperature_kelvin;
};

struct MatterFogFields {
    std::vector<MatterFogField> fields;
    std::size_t peak_dense_bytes_bound = 0;
};

// Transactional: refusal leaves output unchanged. Phase/matter Fog fields are
// separate from aggregate gas; source-specific gas is deliberately not invented.
bool buildMatterFogFields(const MatterFogFieldInput& input,
                         MatterFogFields& output, std::string& error);

} // namespace RayTrophiSim::Fluid
