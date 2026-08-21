#include "PerfProfile.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <mutex>
#include <unordered_map>

#include "globals.h" // SCENE_LOG_INFO

namespace rtperf {
namespace {

std::mutex g_mutex;
std::unordered_map<std::string, Section> g_sections;
uint64_t g_seq = 0;
std::atomic<bool> g_logging{false};

// x64: __stdcall is ignored; K32GetProcessMemoryInfo is a kernel32 forwarder, so
// no psapi.lib link is required and <windows.h> stays out of the header.
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

} // namespace

std::size_t workingSetBytes() {
    MemCounters pmc;
    pmc.cb = static_cast<unsigned long>(sizeof(pmc));
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), &pmc, pmc.cb)) return pmc.WorkingSetSize;
    return 0;
}

void record(const std::string& name, double ms, double rss_delta_mb, double rss_after_mb) {
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        Section& s = g_sections[name];
        s.name = name;
        s.last_ms = ms;
        s.total_ms += ms;
        s.max_ms = (std::max)(s.max_ms, ms);
        s.count += 1;
        s.last_rss_delta_mb = rss_delta_mb;
        s.rss_after_mb = rss_after_mb;
        s.seq = ++g_seq;
    }
    if (g_logging.load(std::memory_order_relaxed)) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "%.1f ms (RSS %+.0f MB, now %.0f MB)",
                      ms, rss_delta_mb, rss_after_mb);
        SCENE_LOG_INFO(std::string("[PERF] ") + name + ": " + buf);
    }
}

std::vector<Section> snapshot() {
    std::vector<Section> out;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        out.reserve(g_sections.size());
        for (const auto& entry : g_sections) out.push_back(entry.second);
    }
    std::sort(out.begin(), out.end(),
              [](const Section& a, const Section& b) { return a.seq > b.seq; });
    return out;
}

bool get(const std::string& name, Section& out) {
    std::lock_guard<std::mutex> lock(g_mutex);
    const auto it = g_sections.find(name);
    if (it == g_sections.end()) return false;
    out = it->second;
    return true;
}

void reset() {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_sections.clear();
    // ★ Deliberately NOT resetting g_seq. Ordering must stay monotonic across a
    // reset, otherwise a caller that reset mid-operation cannot tell a fresh
    // section from a stale one that happens to carry a low sequence number.
}

void setLogging(bool enabled) { g_logging.store(enabled, std::memory_order_relaxed); }
bool logging() { return g_logging.load(std::memory_order_relaxed); }

} // namespace rtperf
