#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rtapi {
struct Result;
struct ParticleSystemRef;

// Particle contact behaviour and contact events (docs/dev/PLAYER_GAME_PAINT_ROOM.md
// step 2). A domain header so the shared RtApi.h does not change.
struct ParticleCollisionSettingsInfo {
    std::string response = "bounce";   // "bounce" | "die"
    bool record_events = false;
    uint32_t event_capacity = 65536u;
};

struct ParticleCollisionEventInfo {
    float position[3] = {0.0f, 0.0f, 0.0f};
    float normal[3] = {0.0f, 1.0f, 0.0f};
    int collider = -1;           // collider index; -1 = the system ground plane
    uint32_t profile = 0;        // appearance profile id
    float speed = 0.0f;          // m/s before the response
    float age_fraction = 0.0f;   // age / lifetime
    uint32_t particle = 0;       // slot
    bool died = false;
};

struct ParticleCollisionEventsInfo {
    uint32_t pending = 0;        // events held until consumed (<= capacity)
    uint64_t overflow = 0;       // contacts dropped because the buffer was full
    uint32_t capacity = 0;
    bool device = false;         // the device buffer was read (resident path)
    // The last step: where it ran and what the collision stage did.
    std::string gpu_status;
    bool collision_stage_on_device = false;
    uint32_t gpu_colliders = 0;
    uint32_t cpu_hits_last_step = 0;    // CPU reference only
    uint32_t cpu_kills_last_step = 0;   // CPU reference only
    std::vector<ParticleCollisionEventInfo> events;
};

Result getParticleCollisionSettings(ParticleCollisionSettingsInfo& out,
                                    const ParticleSystemRef& system);
Result setParticleCollisionSettings(const ParticleCollisionSettingsInfo& info,
                                    const ParticleSystemRef& system);
// Reads up to max_events of the recorded contacts; consume = true empties the
// buffer (the paint stage consumes every frame). On the resident path this is
// a small synchronous download: a measurement / consumer call.
Result readParticleCollisionEvents(uint32_t max_events, bool consume,
                                   ParticleCollisionEventsInfo& out,
                                   const ParticleSystemRef& system);

} // namespace rtapi
