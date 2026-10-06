#include "Fluid/MatterEmissionBudget.h"

#include <algorithm>
#include <cmath>

namespace RayTrophiSim::Fluid {

const char* validateMatterPoolWeight(float weight) {
    if (!std::isfinite(weight) || weight < 0.001f || weight > 1000.0f) {
        return "particle_pool_weight must be finite and in [0.001, 1000]";
    }
    return nullptr;
}

std::vector<std::size_t> allocateMatterEmissionBudget(
    const std::vector<MatterEmissionRequest>& requests, std::size_t capacity,
    uint64_t rotation) {
    std::vector<std::size_t> granted(requests.size(), 0);
    std::vector<std::size_t> active;
    for (std::size_t i = 0; i < requests.size(); ++i) {
        if (requests[i].particles && std::isfinite(requests[i].weight) &&
            requests[i].weight > 0.0) {
            active.push_back(i);
        }
    }
    while (capacity && !active.empty()) {
        double total_weight = 0.0;
        for (const auto i : active) {
            total_weight += requests[i].weight;
        }
        const std::size_t round_capacity = capacity;
        bool satisfied = false;
        for (auto it = active.begin(); it != active.end();) {
            const auto i = *it;
            const double share = static_cast<double>(round_capacity) *
                requests[i].weight / total_weight;
            const auto remaining = requests[i].particles - granted[i];
            if (static_cast<double>(remaining) <= share) {
                granted[i] += remaining;
                capacity -= remaining;
                it = active.erase(it);
                satisfied = true;
            } else {
                ++it;
            }
        }
        if (satisfied) {
            continue;
        }
        struct Remainder {
            std::size_t index;
            double fraction;
            std::size_t tie;
        };
        std::vector<Remainder> remainders;
        const auto offset = static_cast<std::size_t>(rotation % requests.size());
        for (const auto i : active) {
            const double share = static_cast<double>(round_capacity) *
                requests[i].weight / total_weight;
            const auto whole = std::min(capacity, static_cast<std::size_t>(share));
            granted[i] += whole;
            capacity -= whole;
            remainders.push_back({i, share - std::floor(share),
                (i + requests.size() - offset) % requests.size()});
        }
        std::sort(remainders.begin(), remainders.end(), [](const auto& a, const auto& b) {
            if (a.fraction != b.fraction) {
                return a.fraction > b.fraction;
            }
            return a.tie < b.tie;
        });
        for (const auto& remainder : remainders) {
            if (!capacity) {
                break;
            }
            if (granted[remainder.index] < requests[remainder.index].particles) {
                ++granted[remainder.index];
                --capacity;
            }
        }
        break;
    }
    return granted;
}

} // namespace RayTrophiSim::Fluid
