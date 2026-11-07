#include "Fluid/MatterFogSnapshotBudget.h"

#include <cassert>
#include <limits>

int main() {
    using namespace RayTrophiSim::Fluid;
    constexpr std::size_t mib = 1024u * 1024u;
    const auto inspection = MatterFogSnapshotPurpose::Inspection;
    const auto publication = MatterFogSnapshotPurpose::Publication;
    assert(matterFogSnapshotBudgetBytes(inspection, 18688, true) == 128u * mib);
    assert(matterFogSnapshotBudgetBytes(publication, 18688, true) == 1024u * mib);
    assert(matterFogSnapshotBudgetBytes(publication, 64, true) == 64u * mib);
    assert(matterFogSnapshotBudgetBytes(publication, 64, false) == 1024u * mib);
    assert(matterFogSnapshotBudgetBytes(publication, 0, true) == 1024u * mib);
    assert(matterFogSnapshotBudgetBytes(publication,
        std::numeric_limits<uint32_t>::max(), true) == 1024u * mib);
    // Observed 222^3 scene: diagnostic refusal must not imply render refusal.
    constexpr std::size_t cells = 222u * 222u * 222u;
    const auto snapshot_peak = cells * 9u * sizeof(float) + 100000u;
    const auto retained_snapshot = cells * 2u * sizeof(float);
    const auto fields_and_scratch = cells * 6u * sizeof(float) + 100000u;
    assert(snapshot_peak > matterFogSnapshotBudgetBytes(inspection, 18688, true));
    const auto budget = matterFogSnapshotBudgetBytes(publication, 18688, true);
    assert(snapshot_peak < budget);
    assert(retained_snapshot < budget);
    assert(fields_and_scratch < budget - retained_snapshot);
}
