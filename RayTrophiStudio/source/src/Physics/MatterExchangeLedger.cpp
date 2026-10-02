#include "MatterExchangeLedger.h"

#include <cmath>
#include <utility>

namespace RayTrophiSim {

const char* matterExchangeKindName(MatterExchangeKind kind) {
    switch (kind) {
        case MatterExchangeKind::Melting: return "melting";
        case MatterExchangeKind::Freezing: return "freezing";
        case MatterExchangeKind::Vaporization: return "vaporization";
        case MatterExchangeKind::Combustion: return "combustion";
        case MatterExchangeKind::Pyrolysis: return "pyrolysis";
        case MatterExchangeKind::MistToGas: return "mist_to_gas";
        case MatterExchangeKind::Condensation: return "condensation";
    }
    return "unknown";
}

void MatterExchangeLedger::beginStep(uint64_t step) {
    step_ = step;
    records_.clear();
}

void MatterExchangeLedger::record(MatterExchangeRecord record) {
    const bool finite = std::isfinite(record.source_mass_kg) &&
        std::isfinite(record.target_mass_kg) &&
        std::isfinite(record.source_energy_j) &&
        std::isfinite(record.target_energy_j) &&
        std::isfinite(record.latent_required_j) &&
        std::isfinite(record.latent_accounted_j);
    if (!finite || record.source_mass_kg < 0.0 || record.target_mass_kg < 0.0) {
        return;
    }
    records_.push_back(std::move(record));
}

MatterExchangeSummary MatterExchangeLedger::summary() const {
    MatterExchangeSummary out;
    out.step = step_;
    out.events = records_.size();
    for (const MatterExchangeRecord& record : records_) {
        out.source_mass_kg += record.source_mass_kg;
        out.target_mass_kg += record.target_mass_kg;
        out.source_energy_j += record.source_energy_j;
        out.target_energy_j += record.target_energy_j;
        out.latent_required_j += record.latent_required_j;
        out.latent_accounted_j += record.latent_accounted_j;
        out.source_momentum_kg_m_s += record.source_momentum_kg_m_s;
        out.target_momentum_kg_m_s += record.target_momentum_kg_m_s;
    }
    out.mass_error_kg = out.target_mass_kg - out.source_mass_kg;
    out.energy_error_j =
        (out.target_energy_j + out.latent_required_j) -
        (out.source_energy_j + out.latent_accounted_j);
    return out;
}

} // namespace RayTrophiSim
