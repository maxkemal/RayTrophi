#include "Atmosphere/LightningStrike.h"

#include <array>
#include <mutex>

namespace atmosphere {
namespace {

struct CacheEntry {
    bool used = false;
    uint32_t key = 0;
    uint64_t generation = 0;
    bool found = false;
    LightningStrikeTarget target;
};

// Four lanes, each alive for under a second; a few spare slots cover the
// render, viewport and thunder-audio evaluations of neighbouring times.
constexpr int kCacheSize = 16;
constexpr int kLanes = 4;

std::mutex g_mutex;
LightningStrikeResolver g_resolver;
std::array<CacheEntry, kCacheSize> g_cache;
int g_next = 0;
std::array<LightningStrikeTarget, kLanes> g_lanes;

} // namespace

void setLightningStrikeResolver(LightningStrikeResolver resolver) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_resolver = std::move(resolver);
    g_cache = {};
}

bool resolveLightningStrike(uint32_t eventKey, float x, float z, float topY,
                            float strikingDistance, LightningStrikeTarget& out) {
    LightningStrikeResolver resolver;
    uint64_t generation = 0;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (!g_resolver.query) return false;
        resolver = g_resolver;
    }
    if (resolver.generation) generation = resolver.generation();
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        for (const CacheEntry& e : g_cache) {
            if (e.used && e.key == eventKey && e.generation == generation) {
                out = e.target;
                return e.found;
            }
        }
    }
    LightningStrikeTarget target;
    target.striking_distance_m = strikingDistance;
    const bool found = resolver.query(x, z, topY, strikingDistance, target) && target.found;
    target.found = found;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        CacheEntry& e = g_cache[static_cast<size_t>(g_next)];
        g_next = (g_next + 1) % kCacheSize;
        e.used = true;
        e.key = eventKey;
        e.generation = generation;
        e.found = found;
        e.target = target;
    }
    out = target;
    return found;
}

void publishLightningStrike(int lane, const LightningStrikeTarget& target) {
    if (lane < 0 || lane >= kLanes) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    g_lanes[static_cast<size_t>(lane)] = target;
}

LightningStrikeTarget lightningStrikeForLane(int lane) {
    if (lane < 0 || lane >= kLanes) return {};
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_lanes[static_cast<size_t>(lane)];
}

} // namespace atmosphere
