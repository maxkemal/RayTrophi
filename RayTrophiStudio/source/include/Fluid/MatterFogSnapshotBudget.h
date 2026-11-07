#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace RayTrophiSim::Fluid {

enum class MatterFogSnapshotPurpose {
    Inspection,
    Publication
};

// On-demand diagnostics stay small. Live publication is not a diagnostic and
// may use the authored domain ceiling, bounded to 1 GiB of temporary Fog work.
// This is not a reservation, a GPU allocation or a process-wide memory estimate.
inline std::size_t matterFogSnapshotBudgetBytes(MatterFogSnapshotPurpose purpose,
                                               uint32_t domain_budget_mb,
                                               bool enforce_domain_budget) {
    constexpr std::size_t mib = 1024u * 1024u;
    if (purpose == MatterFogSnapshotPurpose::Inspection) {
        return 128u * mib;
    }
    const uint32_t limit = enforce_domain_budget && domain_budget_mb != 0u
        ? std::min(domain_budget_mb, 1024u) : 1024u;
    return static_cast<std::size_t>(limit) * mib;
}

} // namespace RayTrophiSim::Fluid
