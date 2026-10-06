#pragma once

#include "Vec3.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace RayTrophiSim {

enum class MatterExchangeKind : uint8_t {
    Melting,
    Freezing,
    Vaporization,
    Combustion,
    Pyrolysis,
    MistToGas,
    Condensation,
    Absorption,
    Drainage
};

const char* matterExchangeKindName(MatterExchangeKind kind);

struct MatterExchangeRecord {
    uint64_t event_id = 0;
    MatterExchangeKind kind = MatterExchangeKind::Melting;
    std::string source;
    std::string target;
    std::string substance;
    double source_mass_kg = 0.0;
    double target_mass_kg = 0.0;
    double source_energy_j = 0.0;
    double target_energy_j = 0.0;
    double latent_required_j = 0.0;
    double latent_accounted_j = 0.0;
    Vec3 source_momentum_kg_m_s;
    Vec3 target_momentum_kg_m_s;
};

struct MatterExchangeSummary {
    uint64_t step = 0;
    std::size_t events = 0;
    double source_mass_kg = 0.0;
    double target_mass_kg = 0.0;
    double mass_error_kg = 0.0;
    double source_energy_j = 0.0;
    double target_energy_j = 0.0;
    double latent_required_j = 0.0;
    double latent_accounted_j = 0.0;
    double energy_error_j = 0.0;
    Vec3 source_momentum_kg_m_s;
    Vec3 target_momentum_kg_m_s;
};

class MatterExchangeLedger {
public:
    void beginStep(uint64_t step);
    void record(MatterExchangeRecord record);

    uint64_t step() const { return step_; }
    const std::vector<MatterExchangeRecord>& records() const { return records_; }
    MatterExchangeSummary summary() const;

private:
    uint64_t step_ = 0;
    std::vector<MatterExchangeRecord> records_;
};

} // namespace RayTrophiSim
