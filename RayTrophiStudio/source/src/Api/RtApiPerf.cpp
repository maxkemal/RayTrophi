/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          Api/RtApiPerf.cpp
 * Author:        Kemal Demirtas
 * License:       MIT
 * =========================================================================
 *
 * rt.perf — scoped build/render timings as readable values.
 *
 * ★ The perfSection family does not touch the scene, so unlike every other
 * rtapi entry point it needs no UIContext and is safe to call from the IPC
 * thread without enqueueing onto the frame loop. That is the whole point: the
 * interesting moment to ask "what is this spending its time on" is while the
 * frame loop is busy doing it.
 *
 * ★★ THE GPU KERNEL TIMING PAIR AT THE BOTTOM IS THE EXCEPTION. It reads the
 * simulation's compute backend, which a scene teardown can replace, so its IPC
 * handlers ARE enqueued. Said here because a file header that claims a property
 * for everything below it is exactly how a later reader adds an unenqueued
 * scene read by accident.
 */

#include "RtApiInternal.h"
#include "PerfProfile.h"
#include "scene_data.h"
#include "SimulationCompute.h"
#include "SimulationWorld.h"

namespace rtapi {

std::vector<PerfSection> perfSections() {
    std::vector<PerfSection> out;
    for (const rtperf::Section& s : rtperf::snapshot()) {
        PerfSection entry;
        entry.name = s.name;
        entry.last_ms = s.last_ms;
        entry.total_ms = s.total_ms;
        entry.max_ms = s.max_ms;
        entry.count = s.count;
        entry.last_rss_delta_mb = s.last_rss_delta_mb;
        entry.rss_after_mb = s.rss_after_mb;
        entry.rss_measured = s.rss_measured;
        entry.seq = s.seq;
        out.push_back(std::move(entry));
    }
    return out;
}

bool perfSection(const std::string& name, PerfSection& out) {
    rtperf::Section s;
    if (!rtperf::get(name, s)) return false;
    out.name = s.name;
    out.last_ms = s.last_ms;
    out.total_ms = s.total_ms;
    out.max_ms = s.max_ms;
    out.count = s.count;
    out.last_rss_delta_mb = s.last_rss_delta_mb;
    out.rss_after_mb = s.rss_after_mb;
    out.rss_measured = s.rss_measured;
    out.seq = s.seq;
    return true;
}

Result perfReset() {
    rtperf::reset();
    return Result::success();
}

Result perfSetLogging(bool enabled) {
    rtperf::setLogging(enabled);
    return Result::success();
}

bool perfLogging() { return rtperf::logging(); }

// ── GPU kernel timing ────────────────────────────────────────────────────────
//
// ★ Unlike the rest of this file these DO need the scene: the timestamps live
// in the simulation's compute backend, which is owned by the simulation world.
// They are still cheap reads of an accumulator, not a device query.

namespace {
RayTrophiSim::SimulationComputeContext* simCompute() {
    if (!g_ctx) return nullptr;
    return &g_ctx->scene.simulation_world.compute();
}
} // namespace

GpuKernelTimingReport gpuKernelTimings(bool reset) {
    GpuKernelTimingReport out;
    auto* compute = simCompute();
    if (!compute) return out;                 // supported stays false: ABSENCE
    out.supported = compute->supportsGpuTimestamps();
    if (!out.supported) return out;
    out.enabled = compute->gpuTimestampsEnabled();

    std::vector<RayTrophiSim::GpuKernelTiming> timings;
    if (!compute->fetchGpuTimings(timings, reset)) return out;
    out.kernels.reserve(timings.size());
    for (const auto& t : timings) {
        out.kernels.push_back(GpuKernelTime{t.kernel, t.ms, t.calls});
        out.total_ms += t.ms;
    }
    return out;
}

Result setGpuKernelTiming(bool enabled) {
    auto* compute = simCompute();
    if (!compute) return Result::fail("no simulation compute context");
    if (!compute->supportsGpuTimestamps())
        return Result::fail("compute queue does not support timestamp queries");
    compute->setGpuTimestampsEnabled(enabled);
    return Result::success();
}

} // namespace rtapi
