#pragma once

/**
 * Where a ground strike attaches (docs/dev/LIGHTNING_CHANNEL_GROWTH.md).
 *
 * The bolt used to end at the authored ground altitude (y = 0 by default):
 * a strike on a mountain ran on into the rock and its branches, clamped to
 * that floor, piled up around one spot (2026-10-13). The leader now attaches
 * by the rolling-sphere rule: of everything within the striking distance of
 * its path (terrain, meshes, scattered trees), the point a sphere of that
 * radius descending on the leader's vertical touches first.
 *
 * The atmosphere layer owns the cache and the stats; the scene query is
 * installed by the scene (Scene/LightningStrikeResolver.cpp), because only
 * the scene knows its geometry. Main thread.
 */

#include <cstdint>
#include <functional>
#include <string>

namespace atmosphere {

struct LightningStrikeTarget {
    bool found = false;
    float x = 0.0f, y = 0.0f, z = 0.0f;   // scene coordinates of the attachment point
    std::string kind;                     // "terrain" | "mesh" | "scatter"
    std::string object;                   // mesh name or scatter group, empty for terrain
    float striking_distance_m = 0.0f;
    float contact_height_m = 0.0f;        // rolling-sphere centre height at first contact
    int candidates = 0;                   // surface samples + scatter instances examined
};

// Attachment point of a leader descending on the vertical through (x, z)
// from topY: of every surface point within strikingDistance, the one with the
// highest y + sqrt(d^2 - r^2) (r = its horizontal distance from the vertical).
using LightningStrikeQuery = std::function<bool(float x, float z, float topY,
                                                float strikingDistance,
                                                LightningStrikeTarget& out)>;

struct LightningStrikeResolver {
    LightningStrikeQuery query;
    // Changes whenever geometry a strike can attach to changes; drops the cache.
    std::function<uint64_t()> generation;
};

void setLightningStrikeResolver(LightningStrikeResolver resolver);

// Cached per event key: an event attaches once for its whole life (no
// re-query per frame, no wandering endpoint). False = no resolver or nothing
// under the strike; the caller falls back to lightning.ground_altitude_m.
bool resolveLightningStrike(uint32_t eventKey, float x, float z, float topY,
                            float strikingDistance, LightningStrikeTarget& out);

// The attachment of the event in each lane at the last evaluation
// (found = false: not a ground strike, or the fallback altitude was used).
void publishLightningStrike(int lane, const LightningStrikeTarget& target);
LightningStrikeTarget lightningStrikeForLane(int lane);

} // namespace atmosphere
