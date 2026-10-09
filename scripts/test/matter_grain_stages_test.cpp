#include "Fluid/MatterGrainStages.h"

#include <cassert>
#include <string>
#include <vector>

// Models the Verlet-list ping-pong: each substep k runs list_clear, hash, list_build (the conditional list rebuild,
// all at substep k) and the step that reads bank k&1 and writes bank (k+1)&1.
// The frame must end in the canonical bank 0, and no substep may read the bank
// it writes.
int main() {
    using RayTrophiSim::Fluid::dispatchMatterGrainStages;
    const std::vector<std::string> group{"sim_matter_grain_list_clear", "sim_matter_grain_hash",
        "sim_matter_grain_list_build", "sim_matter_grain_step"};
    for (uint32_t count : {2u, 4u, 820u}) {
        std::vector<std::string> calls;
        uint32_t state_bank = 0, expected_substep = 0;
        std::size_t in_group = 0;
        assert(dispatchMatterGrainStages(count, [&](const char* kernel, uint32_t substep) {
            const std::string name(kernel);
            calls.push_back(name);
            // Contact history is updated in place: no frame prologue.
            assert(name != "sim_matter_grain_clear");
            assert(name == group[in_group]);
            assert(substep == expected_substep);
            if (++in_group == group.size()) {
                const uint32_t read_bank = substep & 1u;
                assert(read_bank == state_bank);
                state_bank = read_bank ^ 1u;
                in_group = 0;
                ++expected_substep;
            }
            return true;
        }));
        assert(state_bank == 0 && expected_substep == count && in_group == 0);
        assert(calls.size() == group.size() * count);
        for (std::size_t fail = 0; fail < calls.size(); ++fail) {
            std::size_t visited = 0;
            assert(!dispatchMatterGrainStages(count, [&](const char*, uint32_t) {
                return visited++ != fail;
            }));
            assert(visited == fail + 1);
        }
    }
    // Zero and odd counts would leave the frame in the scratch bank.
    for (uint32_t bad : {0u, 1u, 3u, 819u}) {
        assert(!dispatchMatterGrainStages(bad, [](const char*, uint32_t) { return true; }));
    }
}
