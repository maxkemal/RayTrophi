#pragma once
// Lightweight scoped timer for diagnosing mesh-operation cost (subdivision apply,
// shading recompute, editable-cache build, BVH / backend rebuild). Logs the elapsed
// wall time to the in-app Scene Log via SCENE_LOG_INFO on scope exit.
//
// TEMPORARY INSTRUMENTATION — remove the MESH_PROFILE_SCOPE(...) call sites (and this
// header) once the 2M-poly apply/sculpt bottleneck is located.
#include <chrono>
#include <cstdio>
#include <cstddef>
#include <string>
#include <utility>
#include "globals.h" // SCENE_LOG_INFO / g_sceneLog

namespace meshprof_detail {
// Process working set in bytes, without pulling <windows.h> into every TU that
// profiles. x64: __stdcall is ignored; K32GetProcessMemoryInfo is a kernel32
// forwarder so no psapi.lib link is needed.
extern "C" void* __stdcall GetCurrentProcess(void);
struct MemCounters {
    unsigned long cb;
    unsigned long PageFaultCount;
    std::size_t PeakWorkingSetSize, WorkingSetSize;
    std::size_t QuotaPeakPagedPoolUsage, QuotaPagedPoolUsage;
    std::size_t QuotaPeakNonPagedPoolUsage, QuotaNonPagedPoolUsage;
    std::size_t PagefileUsage, PeakPagefileUsage;
};
extern "C" int __stdcall K32GetProcessMemoryInfo(void*, MemCounters*, unsigned long);
inline std::size_t workingSetBytes() {
    MemCounters pmc;
    pmc.cb = static_cast<unsigned long>(sizeof(pmc));
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), &pmc, pmc.cb)) return pmc.WorkingSetSize;
    return 0;
}
} // namespace meshprof_detail

struct MeshProfileTimer {
    std::string tag;
    std::chrono::high_resolution_clock::time_point t0;
    std::size_t rss0;

    explicit MeshProfileTimer(std::string t)
        : tag(std::move(t)), t0(std::chrono::high_resolution_clock::now()),
          rss0(meshprof_detail::workingSetBytes()) {}

    ~MeshProfileTimer() {
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        const std::size_t rss1 = meshprof_detail::workingSetBytes();
        const double nowMB = static_cast<double>(rss1) / (1024.0 * 1024.0);
        const double deltaMB = (static_cast<double>(rss1) - static_cast<double>(rss0)) / (1024.0 * 1024.0);
        char buf[96];
        std::snprintf(buf, sizeof(buf), "%.1f ms (RSS %+.0f MB, now %.0f MB)", ms, deltaMB, nowMB);
        SCENE_LOG_INFO(std::string("[MESHPROF] ") + tag + ": " + buf);
    }
};

#define MP_CONCAT2(a, b) a##b
#define MP_CONCAT(a, b) MP_CONCAT2(a, b)
#define MESH_PROFILE_SCOPE(tag) ((void)0)
