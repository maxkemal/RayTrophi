#pragma once

#include "json.hpp"

#include <chrono>
#include <cstddef>
#include <utility>

namespace Paint {

struct BalloonPopTiming {
    using Clock = std::chrono::steady_clock;
    using Milliseconds = std::chrono::duration<double, std::milli>;

    Milliseconds pattern{};
    Milliseconds burst_effects{};
    Milliseconds particle_enqueue_sample{};
    Milliseconds launch_effect_sample{};
    std::size_t accepted_drops = 0;
    std::size_t sampled_drops = 0;

    template <class Operation>
    static void measure(Milliseconds& total, Operation&& operation) {
        const auto start = Clock::now();
        std::forward<Operation>(operation)();
        total += Clock::now() - start;
    }

    template <class Enqueue, class Effect>
    void measureDrop(Enqueue&& enqueue, Effect&& effect) {
        // Limit clock reads in a burst with thousands of droplets.
        // A stride of 17 also walks through the drop-effect cadence of 8.
        if (accepted_drops++ % 17 == 0) {
            ++sampled_drops;
            measure(particle_enqueue_sample, std::forward<Enqueue>(enqueue));
            measure(launch_effect_sample, std::forward<Effect>(effect));
        } else {
            std::forward<Enqueue>(enqueue)();
            std::forward<Effect>(effect)();
        }
    }

    nlohmann::json report() const {
        return {
            {"pattern_ms", pattern.count()},
            {"burst_effects_ms", burst_effects.count()},
            {"particle_enqueue_sample_ms", particle_enqueue_sample.count()},
            {"launch_effect_sample_ms", launch_effect_sample.count()},
            {"sampled_drops", sampled_drops},
            {"accepted_drops", accepted_drops},
        };
    }
};

} // namespace Paint
