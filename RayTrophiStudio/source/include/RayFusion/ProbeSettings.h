#pragma once
// Process-wide RayFusion probe settings that must outlive a backend and travel
// with the project. The backend's own state is created lazily and per device,
// so a project could not set it before the device existed; this is the single
// authority and the backend follows it at the top of every service call.
//
// WHY it exists: the producer choice (traced rays vs sky bake) used to live
// only in the backend. Studio could switch traced off, but nothing saved it, so
// the Player re-traced the whole probe field on every open (slow start, and an
// image that differed from the Studio one the author had approved).
#include <atomic>

namespace RayFusion {
// 1 = traced rays (default for a project that stores nothing), 0 = sky bake.
inline std::atomic<int>& probeProducerTraced() {
    static std::atomic<int> value{1};
    return value;
}
} // namespace RayFusion
