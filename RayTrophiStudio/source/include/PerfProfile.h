#pragma once
// Scoped wall-clock + working-set instrumentation whose results are READABLE
// FROM SCRIPT/IPC (rt.perf / perf.list / perf.get).
//
// ★★★ Why this replaces MeshProfileTimer.h
//
// The former mesh profiler wrote its numbers to the in-app Scene Log and then
// had its macro compiled out to `((void)0)`.  Both halves of that were fatal for
// this repo's actual test model: an agent driving the app over IPC cannot read
// the Scene Log, and a disabled macro measures nothing at all.  The result was
// instrumentation that existed in the source tree and answered no question.
//
// Here the measurement is a VALUE in a registry.  Logging is optional and off by
// default; the registry is the product.  A section is written by whoever does
// the work — main thread, node-graph worker, or the async BVH task — and read
// back later by name.
//
// ★ Reads deliberately do NOT go through the UI queue.  Every other IPC query
// is enqueued onto the frame loop, which is correct when the answer depends on
// scene state.  It is wrong here: the single most useful moment to ask "what is
// this build spending its time on" is while the UI thread is busy, and an
// enqueued read would block behind exactly the work it wants to describe.  The
// registry carries its own lock instead.

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace rtperf {

struct Section {
    std::string name;
    double   last_ms = 0.0;
    double   total_ms = 0.0;
    double   max_ms = 0.0;
    uint64_t count = 0;
    // Process working-set delta across the scope.  Terrain build cost at 4k is
    // dominated by allocation, not arithmetic, so a timing without an allocation
    // figure next to it invites optimizing the wrong half.
    double   last_rss_delta_mb = 0.0;
    double   rss_after_mb = 0.0;
    // Monotonic write order, so a caller can tell which sections belong to the
    // most recent operation without clearing the registry first.
    uint64_t seq = 0;
};

void record(const std::string& name, double ms, double rss_delta_mb, double rss_after_mb);

// Newest write first.
std::vector<Section> snapshot();
bool get(const std::string& name, Section& out);
void reset();

// Mirror every completed section into the Scene Log as well.  Off by default:
// the registry is the readable surface, the log is a convenience while sitting
// in front of the app.
void setLogging(bool enabled);
bool logging();

// Process working set in bytes.  Declared here so no TU needs <windows.h>.
std::size_t workingSetBytes();

struct Scope {
    std::string tag;
    std::chrono::high_resolution_clock::time_point t0;
    std::size_t rss0;

    explicit Scope(std::string t)
        : tag(std::move(t)),
          t0(std::chrono::high_resolution_clock::now()),
          rss0(workingSetBytes()) {}

    ~Scope() {
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        const std::size_t rss1 = workingSetBytes();
        const double mb = 1024.0 * 1024.0;
        record(tag, ms,
               (static_cast<double>(rss1) - static_cast<double>(rss0)) / mb,
               static_cast<double>(rss1) / mb);
    }

    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
};

} // namespace rtperf

#define RTPERF_CONCAT2(a, b) a##b
#define RTPERF_CONCAT(a, b) RTPERF_CONCAT2(a, b)
#define RTPERF_SCOPE(tag) ::rtperf::Scope RTPERF_CONCAT(rtperf_scope_, __LINE__)(tag)
