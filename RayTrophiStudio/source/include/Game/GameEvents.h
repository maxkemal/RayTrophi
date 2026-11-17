#pragma once
// Game event queue: the single seam between event PRODUCERS (body-pose gestures,
// proxy contacts, particle/paint events, IPC `game.post_event`) and the CONSUMER
// (the rule engine, which drains the queue once per simulation tick).
//
// Header-only on purpose: producers and the consumer are written by different
// agents in parallel, and neither may depend on the other's .cpp existing.
// Plain floats only (no Vec3) so this header pulls in nothing from the engine.
//
// Contract:
//  - postEvent() may be called from any thread.
//  - drainEvents() is called by the rule engine only, from the simulation tick.
//  - Events that nobody drains are dropped once the queue holds kMaxPending
//    entries; the drop is COUNTED (droppedEvents()), never silent.

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace Game {

struct Event {
    std::string type;          // "gesture.stomp", "gesture.arms_up", "contact", "zone.enter", ...
    std::string target;        // scene object / zone / balloon name; empty if none
    int         person = -1;   // persistent person id from input.set_body_poses; -1 if none
    float       position[3] = { 0.0f, 0.0f, 0.0f };  // world space, meters
    float       normal[3]   = { 0.0f, 1.0f, 0.0f };
    float       value = 0.0f;  // event-specific magnitude (speed in m/s, count, ratio)
    double      time_s = 0.0;  // game clock seconds at the producer
};

namespace detail {
struct EventQueue {
    std::mutex         mutex;
    std::vector<Event> pending;
    std::uint64_t      posted = 0;
    std::uint64_t      dropped = 0;
};
inline EventQueue& eventQueue() {
    static EventQueue queue;
    return queue;
}
} // namespace detail

inline constexpr std::size_t kMaxPending = 4096;

inline void postEvent(Event event) {
    auto& q = detail::eventQueue();
    std::lock_guard<std::mutex> lock(q.mutex);
    ++q.posted;
    if (q.pending.size() >= kMaxPending) {
        ++q.dropped;
        return;
    }
    q.pending.push_back(std::move(event));
}

// Moves every pending event into `out` (appended) and returns how many.
inline std::size_t drainEvents(std::vector<Event>& out) {
    auto& q = detail::eventQueue();
    std::lock_guard<std::mutex> lock(q.mutex);
    const std::size_t count = q.pending.size();
    for (auto& event : q.pending) out.push_back(std::move(event));
    q.pending.clear();
    return count;
}

inline std::uint64_t postedEvents() {
    auto& q = detail::eventQueue();
    std::lock_guard<std::mutex> lock(q.mutex);
    return q.posted;
}

inline std::uint64_t droppedEvents() {
    auto& q = detail::eventQueue();
    std::lock_guard<std::mutex> lock(q.mutex);
    return q.dropped;
}

} // namespace Game
