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
 * ★ These functions do not touch the scene, so unlike every other rtapi entry
 * point they need no UIContext and are safe to call from the IPC thread without
 * enqueueing onto the frame loop. That is the whole point: the interesting
 * moment to ask "what is this spending its time on" is while the frame loop is
 * busy doing it.
 */

#include "RtApiInternal.h"
#include "PerfProfile.h"

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

} // namespace rtapi
