#include "RtApiInternal.h"

#include "MatterExchangeLedger.h"
#include "ParticleSimulation.h"

#include <algorithm>
#include <utility>

namespace rtapi {

MatterExchangeReport matterExchangeReport() {
    MatterExchangeReport out;
    if (!g_ctx) return out;

    for (const auto& system : g_ctx->scene.particle_systems) {
        if (!system.runtime) continue;
        out.traced = true;
        const auto& ledger = system.runtime->matterExchangeLedger();
        out.step = std::max(out.step, ledger.step());
        const RayTrophiSim::MatterExchangeSummary summary = ledger.summary();
        out.source_mass_kg += summary.source_mass_kg;
        out.target_mass_kg += summary.target_mass_kg;
        out.source_energy_j += summary.source_energy_j;
        out.target_energy_j += summary.target_energy_j;
        out.latent_required_j += summary.latent_required_j;
        out.latent_accounted_j += summary.latent_accounted_j;

        for (const auto& record : ledger.records()) {
            MatterExchangeEntry entry;
            entry.event_id = record.event_id;
            entry.kind = RayTrophiSim::matterExchangeKindName(record.kind);
            entry.source = record.source;
            entry.target = record.target;
            entry.substance = record.substance;
            entry.source_mass_kg = record.source_mass_kg;
            entry.target_mass_kg = record.target_mass_kg;
            entry.source_energy_j = record.source_energy_j;
            entry.target_energy_j = record.target_energy_j;
            entry.latent_required_j = record.latent_required_j;
            entry.latent_accounted_j = record.latent_accounted_j;
            entry.source_momentum_kg_m_s = record.source_momentum_kg_m_s;
            entry.target_momentum_kg_m_s = record.target_momentum_kg_m_s;
            out.exchanges.push_back(std::move(entry));
        }
    }
    out.mass_error_kg = out.target_mass_kg - out.source_mass_kg;
    out.energy_error_j =
        (out.target_energy_j + out.latent_required_j) -
        (out.source_energy_j + out.latent_accounted_j);
    return out;
}

} // namespace rtapi
