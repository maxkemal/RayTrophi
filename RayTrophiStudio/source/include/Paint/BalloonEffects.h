#pragma once

#include "Vec3.h"
#include "json.hpp"

#include <cstdint>
#include <string>
#include <vector>

struct SceneData;

namespace Paint {

// Celebration effects of the balloon paint room: confetti, firework sparkles
// with trails and glitter, sparks where a sparkle bounces and sparks where a
// paint drop lands. BalloonPaintDemo owns the game rules and calls these; this
// module owns the effect particle systems (BalloonConfetti, BalloonSparkles,
// BalloonSparkleTrails) and their per-frame sub-emitters. Main thread only.
//
// None of the effect systems records collision events: the wet paint consumes
// the events of EVERY recording system and would paint them. Contacts are
// found here instead: velocity jumps for sparkles, and for paint drops the
// death of their invisible CPU twins (ghosts). Nothing is read from the GPU.
struct BalloonEffectsSetup {
    std::vector<Vec3> palette;                 // linear colours; gold is appended
    Vec3 gold = Vec3(1.0f, 0.62f, 0.08f);
    std::vector<std::string> droplet_systems;  // GPU paint droplet systems (colliders, gravity, drag)
    uint32_t seed = 0;
    int sparkles = 90;                         // sparks per surprise pop / milestone burst
    bool sparkle_trails = true;
    float drop_sparks = 0.25f;                 // chance a sampled drop contact throws sparks
    bool drop_trails = true;                   // paint drops leave trails in their colour
};

void prepareBalloonEffects(SceneData& scene, const BalloonEffectsSetup& setup);
void resetBalloonEffects();
// Copies every collider of the first droplet system into the confetti and
// sparkle systems (their own bounce/friction). No-op before prepare.
void mirrorBalloonEffectColliders(SceneData& scene);
// Paper cards from a point, mostly upward and outward.
void balloonConfettiBurst(SceneData& scene, const Vec3& centre, float radius, int cards,
                          float speed_min, float speed_max);
// Cards falling from just above spawn_max.y over the spawn volume's x/z.
void balloonConfettiRain(SceneData& scene, const Vec3& spawn_min, const Vec3& spawn_max,
                         int cards);
// Every paint drop the demo spawns (colour = palette index): a sampled few get
// an invisible CPU twin that flies the same path and carries the drop's trail.
void balloonDropLaunched(SceneData& scene, const Vec3& position, const Vec3& velocity,
                         int color, float size_scale, float lifetime);
// One firework shell of `sparkles` sparks.
void balloonSparkles(SceneData& scene, const Vec3& centre, float radius);
// Every frame BEFORE the simulation step (the drop tap reads the previous step).
void tickBalloonEffects(SceneData& scene, float dt);
nlohmann::json balloonEffectsStatus(SceneData* scene);

// Cost of the last tick's stages (ms), and the effect systems' last step
// (ParticleSimulationStats::total_ms summed) - for the demo's per-event maxima.
struct BalloonEffectsTimings {
    double contacts_ms = 0.0;
    double drop_trails_ms = 0.0;
    double sparkles_ms = 0.0;
    double step_ms = 0.0;
};
BalloonEffectsTimings balloonEffectsTimings(SceneData& scene);

} // namespace Paint
