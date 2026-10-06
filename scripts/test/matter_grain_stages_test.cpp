#include "Fluid/MatterGrainStages.h"

#include <cassert>
#include <string>
#include <vector>

// Models the fused ping-pong: substep k reads bank k&1 and hash generation
// k+1, writes bank (k+1)&1 and inserts generation k+2. The frame must end in
// the canonical bank 0, and no substep may read the bank it writes.
int main() {
    using RayTrophiSim::Fluid::dispatchMatterGrainStages;
    for (uint32_t count : {2u, 4u, 820u}) {
        std::vector<std::string> calls;
        bool cleared = false;
        uint32_t state_bank = 0, hashed_generation = 0, expected_substep = 0;
        assert(dispatchMatterGrainStages(count, [&](const char* kernel, uint32_t substep) {
            const std::string name(kernel);
            calls.push_back(name);
            if (name == "sim_matter_grain_clear") {
                assert(calls.size() == 1 && substep == 0);
                cleared = true;
            } else if (name == "sim_matter_grain_hash") {
                assert(cleared && calls.size() == 2 && state_bank == 0);
                hashed_generation = 1;
            } else {
                assert(name == "sim_matter_grain_step");
                assert(substep == expected_substep++);
                const uint32_t read_bank = substep & 1u;
                assert(read_bank == state_bank && hashed_generation == substep + 1);
                state_bank = read_bank ^ 1u;
                hashed_generation = substep + 2;
            }
            return true;
        }));
        assert(state_bank == 0 && expected_substep == count);
        assert(calls.size() == count + 2);
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
