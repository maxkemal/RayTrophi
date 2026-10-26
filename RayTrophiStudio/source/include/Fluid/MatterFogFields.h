#pragma once

#include "Fluid/FluidViewResolver.h"
#include "Fluid/GasSourceProvenance.h"
#include "Vec3.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace RayTrophiSim::Fluid {

// Internal field preparation only. No VDB registration, materials or solver writes.
enum class FogFieldKind : uint8_t {
    Carrier,
    GasAggregate,
    GasSource
};

struct FogFieldKey {
    FogFieldKind kind = FogFieldKind::Carrier;
    uint32_t substance_tag = 0;
    MatterOutputPhase phase = MatterOutputPhase::Unknown;
    uint64_t producer_uid = 0;
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
    const GasSourceImages* sources = nullptr;
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
    std::string producer_name;
    FogFieldGrid grid;
    std::size_t carrier_count = 0;
    std::vector<float> density;
    // Physical values only. Empty means unavailable, not cold or an artistic override.
    std::vector<float> temperature_kelvin;
};

struct MatterFogFields {
    std::vector<MatterFogField> fields;
    std::size_t peak_dense_bytes_bound = 0;
    double gas_max_abs_density_error = 0.0;
};

// Transactional: refusal leaves output unchanged. Phase/matter Fog fields are
// separate from gas. Source fields partition its authoritative render density.
bool buildMatterFogFields(const MatterFogFieldInput& input,
                         MatterFogFields& output, std::string& error);

} // namespace RayTrophiSim::Fluid
