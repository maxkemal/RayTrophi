#include "Fluid/MatterEmissionBudget.h"

#include <algorithm>
#include <cassert>
#include <numeric>
#include <limits>

using namespace RayTrophiSim::Fluid;

int main() {
    assert(!validateMatterPoolWeight(1.0f));
    assert(validateMatterPoolWeight(0.0f));
    assert(validateMatterPoolWeight(std::numeric_limits<float>::quiet_NaN()));
    assert(validateMatterPoolWeight(std::numeric_limits<float>::infinity()));
    assert(allocateMatterEmissionBudget({}, 100, 0).empty());
    assert((allocateMatterEmissionBudget({{100, 1}, {100, 1}}, 20, 0) ==
        std::vector<std::size_t>{10, 10}));
    assert((allocateMatterEmissionBudget({{100, 1}, {100, 3}}, 20, 0) ==
        std::vector<std::size_t>{5, 15}));
    assert((allocateMatterEmissionBudget({{2, 1}, {100, 1}}, 20, 0) ==
        std::vector<std::size_t>{2, 18}));
    assert((allocateMatterEmissionBudget({{2, 1}, {3, 1}}, 100, 0) ==
        std::vector<std::size_t>{2, 3}));
    assert((allocateMatterEmissionBudget({{1, 1}, {1, 1}}, 1, 0) ==
        std::vector<std::size_t>{1, 0}));
    assert((allocateMatterEmissionBudget({{1, 1}, {1, 1}}, 1, 1) ==
        std::vector<std::size_t>{0, 1}));
    for (std::size_t capacity = 0; capacity < 200; ++capacity) {
        for (uint64_t rotation = 0; rotation < 4; ++rotation) {
            const std::vector<MatterEmissionRequest> requests = {
                {5, 0.001}, {70, 1000}, {21, 3}, {0, 1}};
            const auto granted = allocateMatterEmissionBudget(requests, capacity, rotation);
            assert(std::accumulate(granted.begin(), granted.end(), std::size_t{0}) ==
                std::min<std::size_t>(capacity, 96));
            for (std::size_t i = 0; i < requests.size(); ++i) {
                assert(granted[i] <= requests[i].particles);
            }
        }
    }
}
