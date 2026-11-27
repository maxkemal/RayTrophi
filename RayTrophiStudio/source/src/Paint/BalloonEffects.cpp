#include "Paint/BalloonEffects.h"

#include "ParticleAppearanceProfile.h"
#include "ParticleSimulation.h"
#include "scene_data.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>

namespace Paint {
namespace {

using json = nlohmann::json;
using RayTrophiSim::ParticleAppearanceProfile;
using RayTrophiSim::ParticleCollisionEvent;
using RayTrophiSim::ParticleSimulationSystem;
using RayTrophiSim::ParticleSpawnDesc;

constexpr const char* kConfettiSystem = "BalloonConfetti";
constexpr const char* kSparkleSystem = "BalloonSparkles";
constexpr const char* kTrailSystem = "BalloonSparkleTrails";
// Invisible CPU twins of sampled paint drops (trail sources, see tickDropTrails).
constexpr const char* kGhostSystem = "BalloonDropGhosts";
constexpr double kTrailInterval = 1.0 / 30.0;
constexpr size_t kConfettiReserve = 220 * 12;
constexpr int kConfettiMotesPerBurst = 16;
constexpr size_t kSparkPoolMoteLimit = 384;
constexpr int kImpactGlintsPerTick = 24;
// Sparkle physics, shared by effectSystem() and the bounce detector.
const Vec3 kSparkGravity(0.0f, -4.0f, 0.0f);
constexpr float kSparkDrag = 1.4f;
// Paint drop sparks come from ghost deaths (ghostContacts).
constexpr int kDropSparksPerContact = 3;
constexpr int kDropSparkFrameBudget = 48;   // sparks per frame, all systems
// Budgets (2026-11-18, measured in the Player, 3 balloons at once): the first
// cut peaked at 12.8k live trail particles and +12 ms simulation, the frame
// rate halved (and every live particle is also a CPU billboard rebuilt each
// frame). Trails now live 0.12..0.2 s and the trail system has a hard cap.
constexpr size_t kTrailCap = 3000;

struct SparkSeen {
    Vec3 velocity;
    float age = -1.0f;
};

struct GhostSeen {
    Vec3 position;
    Vec3 velocity;
    float age = -1.0f;      // < 0: slot was empty last tick
    float lifetime = 0.0f;
    int color = -1;
};

struct Effects {
    bool prepared = false;
    BalloonEffectsSetup setup;
    std::mt19937 random;
    std::vector<uint32_t> confetti_profiles;  // palette + gold
    std::vector<uint32_t> sparkle_profiles;   // palette + gold, kSparkleSystem
    std::vector<uint32_t> impact_profiles;    // same order, kTrailSystem
    std::vector<uint32_t> mote_profiles;     // same order, kSparkleSystem
    std::vector<uint32_t> drop_spark_profiles;// same order, kSparkleSystem (no trail)
    std::vector<uint32_t> drop_trail_profiles;// same order, kTrailSystem
    std::vector<uint32_t> ghost_profiles;     // same order, kGhostSystem (never drawn)
    BalloonEffectsTimings timings;            // last tick
    uint64_t drop_launches = 0;               // drops reported by the demo
    uint64_t ghosts_spawned = 0;
    double drop_trail_clock = 0.0;
    uint64_t drop_trail_spawned = 0;
    std::vector<uint32_t> trail_profiles;     // same order, kTrailSystem
    std::vector<uint32_t> glitter_profiles;   // same order, kTrailSystem
    std::vector<SparkSeen> spark_last;
    std::vector<GhostSeen> ghost_last;        // per ghost slot, last tick
    double trail_clock = 0.0;
    uint64_t sparkles_spawned = 0;
    uint64_t confetti_motes_spawned = 0;
    uint64_t trail_spawned = 0;
    uint64_t glitter_spawned = 0;
    uint64_t impacts = 0;
    uint64_t impact_sparks = 0;
    uint64_t drop_contacts_seen = 0;
    uint64_t drop_sparks_spawned = 0;
};
Effects fx;

ParticleSimulationSystem* system(SceneData& scene, const std::string& name) {
    for (auto& entry : scene.particle_systems) {
        if (entry.name == name) return entry.runtime.get();
    }
    return nullptr;
}

std::shared_ptr<ParticleSimulationSystem> systemShared(SceneData& scene, const std::string& name) {
    for (auto& entry : scene.particle_systems) {
        if (entry.name == name) return entry.runtime;
    }
    return nullptr;
}

bool colliderCooked(const RayTrophiSim::ParticleColliderDesc& c) {
    if (c.source_mode != RayTrophiSim::ParticleColliderSourceMode::ObjectMeshSDF) return true;
    return c.sdf_grid_data && !c.sdf_grid_data->empty() && c.sdf_nx > 0 && c.sdf_ny > 0 &&
           c.sdf_nz > 0;
}

float randomRange(float lo, float hi) {
    return std::uniform_real_distribution<float>(lo, hi)(fx.random);
}

size_t randomIndex(size_t count) {
    return std::uniform_int_distribution<size_t>(0, count - 1)(fx.random);
}

// Unit vector, uniform over the sphere; z_min > -1 biases it upward.
Vec3 randomDirection(float z_min = -1.0f) {
    const float z = randomRange(z_min, 1.0f);
    const float angle = randomRange(0.0f, 6.28318530718f);
    const float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
    return Vec3(r * std::cos(angle), z, r * std::sin(angle));
}

int profileIndex(const std::vector<uint32_t>& list, uint32_t profile) {
    const auto it = std::find(list.begin(), list.end(), profile);
    return it == list.end() ? -1 : static_cast<int>(it - list.begin());
}

std::string colorKey(const Vec3& c) {
    char text[16];
    std::snprintf(text, sizeof(text), "#%02X%02X%02X",
                  static_cast<unsigned>(std::clamp(c.x, 0.0f, 1.0f) * 255.0f + 0.5f),
                  static_cast<unsigned>(std::clamp(c.y, 0.0f, 1.0f) * 255.0f + 0.5f),
                  static_cast<unsigned>(std::clamp(c.z, 0.0f, 1.0f) * 255.0f + 0.5f));
    return text;
}

// Opaque card colour (confetti).
ParticleAppearanceProfile cardProfile(const std::string& name, const Vec3& color, float size) {
    ParticleAppearanceProfile p;
    p.name = name;
    p.blend = RayTrophiSim::ParticleAppearanceBlend::Alpha;
    p.color_ramp = {{0.0f, color}, {1.0f, color}};
    // The last 20 % of its life the card shrinks away (the RT quad reads the
    // size, not the opacity): before, a card resting on the floor just vanished.
    p.opacity_curve = {{0.0f, 1.0f}, {0.8f, 1.0f}, {1.0f, 0.0f}};
    p.size_curve = {{0.0f, size}, {0.8f, size}, {1.0f, 0.0f}};
    p.emission_curve = {{0.0f, 1.0f}, {1.0f, 1.0f}};
    return p;
}

float peakChannel(const Vec3& color) {
    return std::max({color.x, color.y, color.z});
}

Vec3 highlightTint(const Vec3& paint, float warm_share) {
    const Vec3 warm(1.0f, 0.90f, 0.68f);
    const Vec3 hue = paint / std::max(peakChannel(paint), 0.08f);
    const Vec3 tint = hue * (1.0f - warm_share) + warm * warm_share;
    return tint / peakChannel(tint);
}

ParticleAppearanceProfile foilProfile(const std::string& name, const Vec3& color) {
    auto profile = cardProfile(name, color, 0.067f);
    const Vec3 low = color * 0.8f;
    const Vec3 reflection = highlightTint(color, 0.72f);
    // Bake display-limited tints into the ramp so interpolating colour and
    // emission independently cannot overshoot the peak on the LDR overlay.
    profile.color_ramp = {{0.0f, low}, {0.13f, low}, {0.20f, reflection},
                          {0.28f, low}, {0.49f, low}, {0.56f, reflection * 0.90f},
                          {0.65f, low}, {1.0f, color * 0.55f}};
    return profile;
}

ParticleAppearanceProfile sparkProfile(const std::string& name, const Vec3& color, float size,
                                       float emission) {
    ParticleAppearanceProfile p;
    p.name = name;
    p.blend = RayTrophiSim::ParticleAppearanceBlend::Alpha;
    p.color_ramp = {{0.0f, color}, {1.0f, color}};
    p.opacity_curve = {{0.0f, 1.0f}, {0.6f, 0.85f}, {1.0f, 0.0f}};
    p.size_curve = {{0.0f, size}, {1.0f, size * 0.4f}};
    p.emission_curve = {{0.0f, emission}, {0.6f, emission * 0.6f}, {1.0f, emission * 0.3f}};
    return p;
}

ParticleAppearanceProfile hotSparkProfile(const std::string& name, const Vec3& color,
                                         float size, float peak) {
    auto profile = sparkProfile(name, color, size, 1.0f);
    // Light sprites on the LDR overlay must add light to the scene.
    // Alpha blending replaces bright background pixels with the dimmer tail.
    profile.blend = RayTrophiSim::ParticleAppearanceBlend::Additive;
    profile.opacity_curve = {{0.0f, 0.72f}, {0.34f, 0.58f}, {1.0f, 0.0f}};
    // Keep the hue in the billboard input so the shader can draw a pale
    // core and a more saturated halo from the same colour.
    const Vec3 hue = highlightTint(color, 0.12f);
    profile.color_ramp = {{0.0f, hue * peak}, {0.12f, hue * (peak * 0.84f)},
                          {0.34f, hue * (peak * 0.57f)}, {1.0f, hue * 0.16f}};
    profile.emission_curve = {{0.0f, 1.0f}, {1.0f, 1.0f}};
    return profile;
}

ParticleAppearanceProfile coloredTrailProfile(const std::string& name, const Vec3& color,
                                              float size, float brightness) {
    auto profile = sparkProfile(name, color, size, 1.0f);
    profile.blend = RayTrophiSim::ParticleAppearanceBlend::Additive;
    profile.opacity_curve = {{0.0f, 0.49f}, {0.35f, 0.26f},
                             {0.72f, 0.09f}, {1.0f, 0.0f}};
    profile.size_curve = {{0.0f, size}, {0.40f, size * 0.59f}, {1.0f, size * 0.16f}};
    const Vec3 tint = highlightTint(color, 0.08f);
    const Vec3 saturated = tint * tint;
    // Pale particles hit the LDR ceiling sooner when several paths overlap.
    const float lightness = Vec3::dot(color, Vec3(0.2126f, 0.7152f, 0.0722f));
    const float strength = brightness * (1.0f - 0.34f * lightness);
    profile.color_ramp = {{0.0f, saturated * strength},
                          {1.0f, saturated * (strength * 0.35f)}};
    profile.emission_curve = {{0.0f, 1.0f}, {1.0f, 1.0f}};
    return profile;
}

// Glitter: a tiny drifting point in the paint colour that glints twice over
// its 1.2..1.8 s life (1.1..1.7 Hz, every point on its own phase), the second
// glint dimmer (colour and opacity together).
// Flash safety: a few pixels each, below 3 Hz, never in step. ★ The glint count
// is tied to the lifetime: shortening the life with four glints made 3.3 Hz.
ParticleAppearanceProfile glitterProfile(const std::string& name, const Vec3& color) {
    ParticleAppearanceProfile p;
    p.name = name;
    p.blend = RayTrophiSim::ParticleAppearanceBlend::Additive;
    const Vec3 glint = highlightTint(color, 0.12f);
    const Vec3 base = highlightTint(color, 0.16f) * 0.28f;
    p.color_ramp = {{0.0f, base}, {0.2f, glint * 0.98f}, {0.45f, base},
                    {0.7f, glint * 0.76f}, {1.0f, base}};
    p.opacity_curve = {{0.0f, 0.2f}, {0.2f, 0.72f}, {0.45f, 0.14f}, {0.7f, 0.55f},
                       {1.0f, 0.0f}};
    p.size_curve = {{0.0f, 0.06f}, {1.0f, 0.036f}};
    p.emission_curve = {{0.0f, 1.0f}, {1.0f, 1.0f}};
    return p;
}

// The module owns these profiles: a project saved with an older look must not
// keep it (findOrAdd matches by name only), so the stored curves are replaced.
uint32_t upsertProfile(ParticleSimulationSystem& runtime, const ParticleAppearanceProfile& wanted) {
    ParticleAppearanceProfile p = wanted;
    p.id = runtime.findOrAddAppearanceProfile(wanted);
    runtime.updateAppearanceProfile(p);
    return p.id;
}

// A CPU-only effect system: no events, Bounce on contact.
ParticleSimulationSystem* effectSystem(SceneData& scene, const char* name, const Vec3& gravity,
                                       float drag, size_t reserve,
                                       SceneData::ParticleRenderShape shape, bool billboard_only,
                                       bool emissive = false) {
    if (!system(scene, name)) {
        scene.addParticleSystemObject(name);
    }
    auto* runtime = system(scene, name);
    runtime->clear();
    runtime->clearEmitters();
    runtime->setGravity(gravity);
    runtime->setLinearDrag(drag);
    auto& physics = runtime->physicsSettings();
    physics.execution_policy = RayTrophiSim::ParticleExecutionPolicy::CPU;
    physics.record_collision_events = false;
    physics.collision_response = RayTrophiSim::ParticleCollisionResponse::Bounce;
    runtime->reserve(reserve);
    for (auto& entry : scene.particle_systems) {
        if (entry.name == name) {
            entry.enabled = true;
            entry.visible = true;
            entry.render.emitter_only = false;
            entry.render.shape = shape;
            entry.render.emissive = emissive;
            entry.render.roughness = 0.5f;
            // Spark systems draw ONLY as billboards (per-profile colour). The RT
            // instance bridge gives a whole system ONE material - with no emitter
            // the default orange x emission, which saturated to white - and its
            // opaque sphere hid the coloured billboard. 2026-11-18: "weak white tail".
            entry.render.render_in_raytrace = !billboard_only;
            entry.render.inherit_color_from_emitter = false;
            SceneData::applyParticleSystemEnabledState(entry);
        }
    }
    return runtime;
}

// Every collider of the first droplet system, copied with the effect's own
// restitution/friction. The module owns these systems, so their collider list
// is replaced, never merged.
void mirrorDropletColliders(SceneData& scene, const char* name, float restitution,
                            float friction) {
    if (fx.setup.droplet_systems.empty()) return;
    auto target = systemShared(scene, name);
    auto* source = system(scene, fx.setup.droplet_systems.front());
    if (!target || !source) return;
    while (!target->colliders().empty()) {
        target->removeCollider(target->colliders().size() - 1);
    }
    target->setCollisionPlane(source->collisionPlaneY(), source->collisionPlaneEnabled(),
                              restitution);
    for (const auto& c : source->colliders()) {
        RayTrophiSim::ParticleColliderDesc desc = c;
        desc.restitution = restitution;
        desc.friction = friction;
        auto& created = target->addCollider(desc);
        // The copy SHARES the droplet collider's cooked grid (sdf_grid_data is a
        // shared pointer; configure refuses until the droplets are cooked), so
        // there is nothing to bake. Only a still-uncooked copy is sent to the
        // same (GPU) bake set_collider uses - never a second bake of done work.
        if (!colliderCooked(created)) {
            scene.rebuildSDFColliderAsync(created, target);
        }
    }
}

void spawnCard(ParticleSimulationSystem& confetti, const Vec3& position, const Vec3& velocity,
               float lifetime) {
    ParticleSpawnDesc card;
    card.position = position;
    card.velocity = velocity;
    card.lifetime_seconds = lifetime;
    card.mass = 0.0005f;
    card.appearance_profile_id = fx.confetti_profiles[randomIndex(fx.confetti_profiles.size())];
    card.size_scale = randomRange(0.7f, 1.3f);
    card.rotation = randomRange(0.0f, 6.28318530718f);
    card.angular_velocity = randomRange(-9.0f, 9.0f);
    confetti.spawn(card);
}

void spawnConfettiMote(ParticleSimulationSystem* motes, const Vec3& position,
                       const Vec3& velocity, int& budget) {
    if (budget <= 0 || !motes || fx.mote_profiles.empty() ||
        motes->aliveCount() >= kSparkPoolMoteLimit) {
        return;
    }
    ParticleSpawnDesc mote;
    mote.position = position;
    mote.velocity = velocity;
    mote.lifetime_seconds = randomRange(0.55f, 0.95f);
    mote.mass = 0.0005f;
    mote.appearance_profile_id = fx.mote_profiles[randomIndex(fx.mote_profiles.size())];
    mote.size_scale = randomRange(0.65f, 1.15f);
    motes->spawn(mote);
    ++fx.confetti_motes_spawned;
    --budget;
}

// Small fan of sparks thrown off a surface along its normal.
void sparkFan(ParticleSimulationSystem& sparks, const Vec3& position, const Vec3& normal,
              uint32_t profile, int count, float speed_scale, float life_min, float life_max) {
    for (int k = 0; k < count; ++k) {
        const Vec3 direction = (normal * 1.3f + randomDirection()).normalize();
        ParticleSpawnDesc spark;
        spark.position = position + normal * 0.02f;
        spark.velocity = direction * (randomRange(1.5f, 3.5f) * speed_scale);
        spark.lifetime_seconds = randomRange(life_min, life_max);
        spark.mass = 0.0005f;
        spark.appearance_profile_id = profile;
        spark.size_scale = randomRange(0.7f, 1.3f);
        sparks.spawn(spark);
    }
}

// Paint drop contacts -> sparks, from the GHOSTS: a ghost that disappears
// before its lifetime is up died on a collider (Die response), i.e. its drop
// just landed there. Contact = last position plus last step's travel; the
// sparks fly back off the surface (opposite the arrival direction).
// ★ Replaced a read of the droplets' DEVICE event buffer (first 64 slots per
// system, sentinel-marked). Measured 2026-11-18 in a fresh Player: 31 ms on the
// first pop and ~5 ms on every frame after - each synchronous transfer on the
// simulation context waited for the frame work in flight. Nothing is read
// from the GPU now, and nothing is written into the wet paint's event buffer.
void ghostContacts(SceneData& scene, float dt) {
    auto* ghosts = system(scene, kGhostSystem);
    auto* sparks = system(scene, kSparkleSystem);
    if (!ghosts || !sparks) return;
    const auto& b = ghosts->buffers();
    if (fx.ghost_last.size() < b.alive.size()) fx.ghost_last.resize(b.alive.size());
    int budget = kDropSparkFrameBudget;
    for (size_t i = 0; i < b.alive.size(); ++i) {
        GhostSeen& seen = fx.ghost_last[i];
        const bool alive = b.alive[i] != 0;
        const bool reused = alive && seen.age >= 0.0f && b.age_seconds[i] < seen.age;
        if (seen.age >= 0.0f && (!alive || reused) && seen.age < seen.lifetime - 0.1f) {
            ++fx.drop_contacts_seen;
            if (budget > 0 && fx.setup.drop_sparks > 0.0f &&
                randomRange(0.0f, 1.0f) < fx.setup.drop_sparks && seen.color >= 0) {
                const float speed = seen.velocity.length();
                const Vec3 normal = speed > 1e-4f ? seen.velocity * (-1.0f / speed) : Vec3(0.0f, 1.0f, 0.0f);
                sparkFan(*sparks, seen.position + seen.velocity * dt, normal,
                         fx.drop_spark_profiles[static_cast<size_t>(seen.color)], kDropSparksPerContact,
                         std::min(1.0f, speed / 6.0f), 0.25f, 0.5f);
                fx.drop_sparks_spawned += kDropSparksPerContact;
                budget -= kDropSparksPerContact;
            }
        }
        if (!alive) {
            seen.age = -1.0f;
            continue;
        }
        seen.position = Vec3(b.position_x[i], b.position_y[i], b.position_z[i]);
        seen.velocity = Vec3(b.velocity_x[i], b.velocity_y[i], b.velocity_z[i]);
        seen.age = b.age_seconds[i];
        seen.lifetime = b.lifetime_seconds[i];
        seen.color = profileIndex(fx.ghost_profiles, b.appearance_profile[i]);
    }
}

// Paint drop trails, 30 times a second, from GHOSTS: an invisible CPU twin of
// every kGhostEvery-th drop, launched with the drop's position and velocity
// under the droplet systems' gravity and drag and dying on the same (mirrored)
// colliders, so it flies the drop's path. The trail follows the ghost.
// ★ The first version read the GPU drops back instead (syncHostState every
// frame, so every resident step mirrored its kinematics): measured 2026-11-18
// in the Player, +36 ms simulation per frame with 3 balloons in the air.
// Nothing is read from the GPU now.
constexpr int kGhostEvery = 8;
constexpr size_t kGhostCap = 600;

float trailBirthWidth(float age) {
    // Right after the pop the sources still overlap near the burst origin.
    return 0.52f + 0.48f * std::clamp(age / 0.14f, 0.0f, 1.0f);
}

void tickDropTrails(SceneData& scene, float dt) {
    auto* trails = system(scene, kTrailSystem);
    auto* ghosts = system(scene, kGhostSystem);
    if (!trails || !ghosts || !fx.setup.drop_trails || ghosts->aliveCount() == 0) {
        fx.drop_trail_clock = 0.0;
        return;
    }
    fx.drop_trail_clock += dt;
    if (fx.drop_trail_clock < kTrailInterval || trails->aliveCount() >= kTrailCap) return;
    fx.drop_trail_clock = 0.0;
    const float half_tick = static_cast<float>(kTrailInterval) * 0.5f;
    const auto& b = ghosts->buffers();
    for (size_t i = 0; i < b.alive.size() && trails->aliveCount() < kTrailCap; ++i) {
        if (!b.alive[i]) continue;
        const int color = profileIndex(fx.ghost_profiles, b.appearance_profile[i]);
        if (color < 0) continue;
        const Vec3 p(b.position_x[i], b.position_y[i], b.position_z[i]);
        const Vec3 v(b.velocity_x[i], b.velocity_y[i], b.velocity_z[i]);
        // Two dots per tick (at the drop and half a tick behind) keep the
        // streak continuous at 4.5..9 m/s.
        const float width = trailBirthWidth(b.age_seconds[i]);
        for (int k = 0; k < 2 && trails->aliveCount() < kTrailCap; ++k) {
            ParticleSpawnDesc dot;
            dot.position = p - v * (half_tick * static_cast<float>(k));
            dot.velocity = v * 0.05f;
            dot.lifetime_seconds = 0.15f;
            dot.mass = 0.0005f;
            dot.appearance_profile_id = fx.drop_trail_profiles[static_cast<size_t>(color)];
            dot.size_scale = b.size_scale[i] * width * (k == 0 ? 1.0f : 0.70f);
            trails->spawn(dot);
            ++fx.drop_trail_spawned;
        }
    }
}

// A dead head whose age is still below its lifetime hit a collider. Use
// the CPU solver's resolved position for one stationary fading glint.
void tickSparkles(SceneData& scene, float dt) {
    auto* sparks = system(scene, kSparkleSystem);
    auto* trails = system(scene, kTrailSystem);
    if (!sparks || !trails) {
        return;
    }
    const bool alive = sparks->aliveCount() > 0;
    const bool collision_kills = sparks->stats().collision_kills > 0;
    if (!alive && !collision_kills) {
        fx.trail_clock = 0.0;
        fx.spark_last.clear();
        return;
    }
    const auto& b = sparks->buffers();
    const size_t slots = b.alive.size();
    fx.spark_last.resize(slots);
    int impact_budget = kImpactGlintsPerTick;
    for (size_t i = 0; i < slots; ++i) {
        auto& seen = fx.spark_last[i];
        const float age = b.age_seconds[i];
        if (!b.alive[i]) {
            if (collision_kills && seen.age >= 0.0f && age > seen.age &&
                age < b.lifetime_seconds[i] && impact_budget > 0 &&
                trails->aliveCount() < kTrailCap) {
                const int color = profileIndex(fx.sparkle_profiles, b.appearance_profile[i]);
                if (color >= 0) {
                    ParticleSpawnDesc glint;
                    glint.position = Vec3(b.position_x[i], b.position_y[i], b.position_z[i]) -
                                     seen.velocity.normalize() * 0.018f;
                    glint.lifetime_seconds = 0.13f;
                    glint.appearance_profile_id = fx.impact_profiles[static_cast<size_t>(color)];
                    glint.size_scale = b.size_scale[i];
                    trails->spawn(glint);
                    ++fx.impacts;
                    ++fx.impact_sparks;
                    --impact_budget;
                }
            }
            seen.age = -1.0f;
            continue;
        }
        seen.velocity = Vec3(b.velocity_x[i], b.velocity_y[i], b.velocity_z[i]);
        seen.age = age;
    }
    if (!alive) {
        fx.trail_clock = 0.0;
        return;
    }

    fx.trail_clock += dt;
    if (!fx.setup.sparkle_trails || fx.trail_clock < kTrailInterval) return;
    fx.trail_clock = 0.0;
    const auto& now = sparks->buffers();
    for (size_t i = 0; i < now.alive.size() && trails->aliveCount() < kTrailCap; ++i) {
        if (!now.alive[i]) continue;
        const int color = profileIndex(fx.sparkle_profiles, now.appearance_profile[i]);
        if (color < 0) continue;   // motes and drop sparks do not shed trails
        const Vec3 p(now.position_x[i], now.position_y[i], now.position_z[i]);
        const Vec3 v(now.velocity_x[i], now.velocity_y[i], now.velocity_z[i]);
        const float life = now.lifetime_seconds[i] > 0.0f
            ? now.age_seconds[i] / now.lifetime_seconds[i] : 1.0f;
        ParticleSpawnDesc dot;
        dot.position = p;
        dot.velocity = v * 0.1f;
        dot.lifetime_seconds = 0.16f;
        dot.mass = 0.0005f;
        dot.appearance_profile_id = fx.trail_profiles[static_cast<size_t>(color)];
        dot.size_scale = now.size_scale[i] * trailBirthWidth(now.age_seconds[i]) *
                         std::max(0.25f, 1.0f - life);
        trails->spawn(dot);
        ++fx.trail_spawned;
        if (trails->aliveCount() < kTrailCap && randomRange(0.0f, 1.0f) < 0.05f) {
            ParticleSpawnDesc glint;
            glint.position = p;
            glint.velocity = v * 0.05f + randomDirection() * 0.15f;
            glint.lifetime_seconds = randomRange(1.2f, 1.8f);
            glint.mass = 0.0005f;
            glint.appearance_profile_id = fx.glitter_profiles[static_cast<size_t>(color)];
            glint.size_scale = randomRange(0.7f, 1.3f);
            trails->spawn(glint);
            ++fx.glitter_spawned;
        }
    }
}

} // namespace

void prepareBalloonEffects(SceneData& scene, const BalloonEffectsSetup& setup) {
    fx = Effects{};
    fx.setup = setup;
    fx.random.seed(setup.seed ^ 0x5EEDF00Du);
    std::vector<Vec3> colors = setup.palette;
    colors.push_back(setup.gold);

    // Stop drawing confetti on its first surface contact.
    auto* confetti = effectSystem(scene, kConfettiSystem, Vec3(0.0f, -2.5f, 0.0f), 2.2f,
                                  kConfettiReserve, SceneData::ParticleRenderShape::Quad, false);
    confetti->physicsSettings().collision_response = RayTrophiSim::ParticleCollisionResponse::Die;
    // Heads, drop sparks and confetti motes die on contact. Their trails
    // and short contact glints just drift and expire without colliders.
    auto* sparkles = effectSystem(scene, kSparkleSystem, kSparkGravity, kSparkDrag, 4096,
                                  SceneData::ParticleRenderShape::Sphere, true, true);
    sparkles->physicsSettings().collision_response = RayTrophiSim::ParticleCollisionResponse::Die;
    auto* trails = effectSystem(scene, kTrailSystem, Vec3(0.0f, -0.5f, 0.0f), 6.0f, 16384,
                                SceneData::ParticleRenderShape::Sphere, true, true);
    // Trails have 0 colliders and billboard-only rendering, so Auto lets them run on GPU.
    trails->physicsSettings().execution_policy = RayTrophiSim::ParticleExecutionPolicy::Auto;
    // Drop ghosts: the droplets' own gravity and drag, Die on contact, never
    // drawn (emitter_only keeps them simulating; visible=false would stop them).
    Vec3 drop_gravity(0.0f, -9.81f, 0.0f);
    float drop_drag = 0.15f;
    if (!setup.droplet_systems.empty()) {
        if (auto* first = system(scene, setup.droplet_systems.front())) {
            drop_gravity = first->gravity();
            drop_drag = first->linearDrag();
        }
    }
    auto* ghosts = effectSystem(scene, kGhostSystem, drop_gravity, drop_drag, kGhostCap,
                                SceneData::ParticleRenderShape::Sphere, true);
    ghosts->physicsSettings().collision_response = RayTrophiSim::ParticleCollisionResponse::Die;
    for (auto& entry : scene.particle_systems) {
        if (entry.name == kGhostSystem) entry.render.emitter_only = true;
    }
    for (size_t c = 0; c < colors.size(); ++c) {
        const Vec3& color = colors[c];
        const bool gold = c + 1 == colors.size();
        const std::string key = colorKey(color);
        // Confetti keeps its original names ("Confetti Gold"): saved projects match.
        fx.confetti_profiles.push_back(upsertProfile(*confetti,
            foilProfile(gold ? std::string("Confetti Gold") : "Confetti " + key, color)));
        fx.sparkle_profiles.push_back(upsertProfile(*sparkles,
            hotSparkProfile("Sparkle " + key, color, 0.18f, 0.98f)));
        fx.impact_profiles.push_back(upsertProfile(*trails,
            hotSparkProfile("Sparkle Quench " + key, color, 0.085f, 0.88f)));
        fx.mote_profiles.push_back(upsertProfile(*sparkles,
            glitterProfile("Confetti Mote " + key, color)));
        fx.drop_spark_profiles.push_back(upsertProfile(*sparkles,
            hotSparkProfile("Drop Spark " + key, color, 0.08f, 0.92f)));
        fx.trail_profiles.push_back(upsertProfile(*trails,
            coloredTrailProfile("Sparkle Trail " + key, color, 0.075f, 0.53f)));
        fx.glitter_profiles.push_back(upsertProfile(*trails,
            glitterProfile("Sparkle Glitter " + key, color)));
        fx.drop_trail_profiles.push_back(upsertProfile(*trails,
            coloredTrailProfile("Drop Trail " + key, color, 0.05f, 0.44f)));
        fx.ghost_profiles.push_back(upsertProfile(*ghosts,
            sparkProfile("Drop Ghost " + key, color, 0.05f, 1.0f)));
    }
    fx.prepared = true;
    mirrorBalloonEffectColliders(scene);
}

void resetBalloonEffects() {
    fx = Effects{};
}

void mirrorBalloonEffectColliders(SceneData& scene) {
    if (!fx.prepared) return;
    mirrorDropletColliders(scene, kConfettiSystem, 0.0f, 0.0f);    // dies on contact
    mirrorDropletColliders(scene, kSparkleSystem, 0.0f, 0.0f);    // quenched on contact
    mirrorDropletColliders(scene, kGhostSystem, 0.0f, 0.0f);       // ghosts die there
}

void balloonDropLaunched(SceneData& scene, const Vec3& position, const Vec3& velocity,
                         int color, float size_scale, float lifetime) {
    if (!fx.prepared || (!fx.setup.drop_trails && fx.setup.drop_sparks <= 0.0f) || color < 0 ||
        static_cast<size_t>(color) >= fx.ghost_profiles.size()) {
        return;
    }
    if (fx.drop_launches++ % kGhostEvery != 0) return;
    auto* ghosts = system(scene, kGhostSystem);
    if (!ghosts || ghosts->aliveCount() >= kGhostCap) return;
    ParticleSpawnDesc ghost;
    ghost.position = position;
    ghost.velocity = velocity;
    ghost.lifetime_seconds = lifetime;
    ghost.mass = 0.001f;
    ghost.appearance_profile_id = fx.ghost_profiles[static_cast<size_t>(color)];
    ghost.size_scale = size_scale;
    ghosts->spawn(ghost);
    ++fx.ghosts_spawned;
}

void balloonConfettiBurst(SceneData& scene, const Vec3& centre, float radius, int cards,
                          float speed_min, float speed_max) {
    auto* confetti = system(scene, kConfettiSystem);
    if (!fx.prepared || !confetti || fx.confetti_profiles.empty()) return;
    auto* motes = system(scene, kSparkleSystem);
    int mote_budget = kConfettiMotesPerBurst;
    for (int i = 0; i < cards; ++i) {
        const Vec3 direction = randomDirection(-0.3f);
        const Vec3 position = centre + direction * radius;
        spawnCard(*confetti, position, direction * randomRange(speed_min, speed_max),
                  randomRange(2.5f, 4.0f));
        if (i % 8 == 0) {
            spawnConfettiMote(motes, position, direction * randomRange(0.7f, 1.5f), mote_budget);
        }
    }
}

void balloonConfettiRain(SceneData& scene, const Vec3& spawn_min, const Vec3& spawn_max, int cards) {
    auto* confetti = system(scene, kConfettiSystem);
    if (!fx.prepared || !confetti || fx.confetti_profiles.empty()) return;
    auto* motes = system(scene, kSparkleSystem);
    int mote_budget = kConfettiMotesPerBurst;
    for (int i = 0; i < cards; ++i) {
        // Start inside the room so rain cannot settle on top of the ceiling.
        const Vec3 position(randomRange(spawn_min.x, spawn_max.x),
                            spawn_max.y + randomRange(0.3f, 0.6f),
                            randomRange(spawn_min.z, spawn_max.z));
        const Vec3 velocity(randomRange(-0.3f, 0.3f), randomRange(-0.6f, 0.0f),
                            randomRange(-0.3f, 0.3f));
        spawnCard(*confetti, position, velocity, randomRange(4.0f, 6.0f));
        if (i % 16 == 0) {
            spawnConfettiMote(motes, position, velocity, mote_budget);
        }
    }
}

BalloonBurstSummary balloonBurstPattern(std::mt19937& random, int drops, float balloon_size,
                                        const BalloonBurstShape& shape,
                                        std::vector<BalloonBurstDrop>& out,
                                        const Vec3* directed_target) {
    out.clear();
    BalloonBurstSummary summary;
    if (drops <= 0) return summary;
    out.reserve(static_cast<size_t>(drops));
    constexpr float kTwoPi = 6.28318530718f;
    constexpr float kMaxSpeed = 12.0f;
    auto uni = [&](float lo, float hi) { return std::uniform_real_distribution<float>(lo, hi)(random); };
    auto uniInt = [&](int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(random); };
    auto sphere = [&]() {
        const float z = uni(-1.0f, 1.0f);
        const float a = uni(0.0f, kTwoPi);
        const float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
        return Vec3(r * std::cos(a), z, r * std::sin(a));
    };
    // Uniform direction within half_angle of a unit axis.
    auto cone = [&](const Vec3& axis, float half_angle) {
        const float cos_t = uni(std::cos(half_angle), 1.0f);
        const float sin_t = std::sqrt(std::max(0.0f, 1.0f - cos_t * cos_t));
        const float phi = uni(0.0f, kTwoPi);
        const Vec3 helper = std::fabs(axis.y) < 0.9f ? Vec3(0.0f, 1.0f, 0.0f) : Vec3(1.0f, 0.0f, 0.0f);
        const Vec3 u = Vec3::cross(axis, helper).normalize();
        const Vec3 v = Vec3::cross(axis, u);
        return axis * cos_t + (u * std::cos(phi) + v * std::sin(phi)) * sin_t;
    };
    auto capped = [&](const Vec3& velocity) {
        const float speed = velocity.length();
        return speed > kMaxSpeed ? velocity * (kMaxSpeed / speed) : velocity;
    };
    uint32_t clump_id = 0;
    auto push = [&](const Vec3& offset, const Vec3& velocity, float size) {
        BalloonBurstDrop d;
        d.offset = offset;
        d.velocity = capped(velocity);
        d.size_scale = std::clamp(size, 0.35f, 2.2f);
        d.clump = clump_id;
        out.push_back(d);
    };

    const float clump = std::clamp(shape.clumpiness, 0.0f, 1.0f);
    const float radius = 0.22f * balloon_size;
    const bool directed = directed_target != nullptr && directed_target->length_squared() > 1e-6f;
    const Vec3 tear = directed ? directed_target->normalize() : sphere();
    const float directed_half_angle = std::acos(0.78f);
    summary.tear = tear;
    // Hardness of a direction: the tear side bursts up to 1.4x, the far side 0.6x.
    auto hardness = [&](const Vec3& dir) { return 1.0f + 0.4f * Vec3::dot(dir, tear); };

    // Shares, drawn per pop. When crushed against a wall, ~75-80% of drops form
    // heavy blobs and main jets directed into the wall while ~20-25% mist naturally.
    const float spray_share = clump <= 0.0f
        ? 1.0f
        : (directed ? uni(0.20f, 0.25f)
                    : std::clamp((1.0f - clump) * uni(0.75f, 1.25f) + 0.06f, 0.0f, 1.0f));
    const int spray = std::min(drops, static_cast<int>(std::lround(drops * spray_share)));
    int rest = drops - spray;
    const int blobs = rest >= 32 ? uniInt(1, 3) : 0;
    std::vector<int> blob_sizes;
    for (int b = 0; b < blobs; ++b) {
        const int n = std::min(rest / 4, std::max(4, static_cast<int>(drops * uni(0.02f, 0.05f))));
        if (n <= 0) break;
        blob_sizes.push_back(n);
        rest -= n;
    }
    const int jets_max = std::max(3, shape.jets_max);
    int jets = rest > 0 ? std::clamp(uniInt(3, jets_max), 1, std::max(1, rest / 6)) : 0;

    // Ligament masses: gamma(0.6) is heavy-tailed, so one or two ligaments
    // carry most of the paint and some carry almost none.
    std::vector<int> jet_sizes;
    if (jets > 0) {
        std::gamma_distribution<float> gamma(0.6f, 1.0f);
        std::vector<float> w(static_cast<size_t>(jets));
        float total = 0.0f;
        for (float& x : w) { x = gamma(random) + 1e-3f; total += x; }
        int given = 0;
        size_t heaviest = 0;
        for (size_t k = 0; k < w.size(); ++k) {
            const int n = static_cast<int>(std::floor(rest * w[k] / total));
            jet_sizes.push_back(n);
            given += n;
            if (w[k] > w[heaviest]) heaviest = k;
        }
        jet_sizes[heaviest] += rest - given;
        summary.heaviest_jet_share = rest > 0 ? static_cast<float>(jet_sizes[heaviest]) / rest : 0.0f;
    }

    // Ligaments: beads along a stretched sheet finger, in clumps.
    for (int n_k : jet_sizes) {
        if (n_k <= 0) continue;
        ++summary.jets;
        Vec3 axis = directed ? cone(tear, directed_half_angle * 0.8f) : sphere();
        if (!directed && uni(0.0f, 1.0f) < 0.5f) {
            axis = (axis + tear * uni(0.3f, 1.2f)).normalize();
        }
        const float speed = uni(4.5f, 8.0f) * hardness(axis);
        const float half_angle = directed ? uni(0.05f, 0.16f) : uni(0.05f, 0.25f);
        for (int i = 0; i < n_k;) {
            // Clump size: geometric, mostly single beads, sometimes eight.
            int m = 1;
            while (m < 8 && uni(0.0f, 1.0f) < 0.55f) ++m;
            m = std::min(m, n_k - i);
            const float s = uni(0.0f, 1.0f);   // 0 = root (slow), 1 = tip (fast)
            const Vec3 dir = cone(axis, half_angle * (1.2f - 0.6f * s));
            const Vec3 v = dir * (speed * (0.4f + 0.8f * s));
            const float bead = std::exp(std::normal_distribution<float>(0.0f, 0.3f)(random)) *
                               (0.8f + 0.5f * s) * (m > 1 ? 1.15f : 1.0f);
            const Vec3 at = dir * (radius * uni(0.7f, 1.0f));
            ++clump_id;
            for (int j = 0; j < m; ++j) {
                push(at + sphere() * 0.012f, v + sphere() * uni(0.0f, 0.18f), bead * uni(0.85f, 1.1f));
            }
            if (m > 1) ++summary.clumps;
            summary.jet_drops += m;
            i += m;
        }
    }
    // Heavy blobs: the paint pooled low in the balloon, slow, tight, falling.
    for (int n : blob_sizes) {
        ++summary.blobs;
        const Vec3 dir = directed ? cone(tear, directed_half_angle)
                                  : (sphere() + Vec3(0.0f, -0.6f, 0.0f)).normalize();
        const Vec3 v = dir * (uni(2.0f, 4.5f) * hardness(dir));
        const Vec3 at = dir * (radius * 0.6f);
        const float size = uni(1.3f, 1.9f);
        ++clump_id;
        for (int j = 0; j < n; ++j) {
            push(at + sphere() * 0.02f, v + sphere() * uni(0.0f, 0.1f), size * uni(0.85f, 1.1f));
        }
        summary.blob_drops += n;
    }
    // Fine spray: the even part, small and fast.
    for (int i = 0; i < spray; ++i) {
        const Vec3 dir = sphere();
        ++clump_id;
        push(dir * (radius * uni(0.0f, 1.0f)), dir * (uni(3.5f, 10.0f) * hardness(dir)), uni(0.4f, 0.85f));
    }
    summary.spray_drops = spray;
    return summary;
}

void balloonSparkles(SceneData& scene, const Vec3& centre, float radius) {
    auto* sparks = system(scene, kSparkleSystem);
    const int count = fx.setup.sparkles;
    if (!fx.prepared || !sparks || fx.sparkle_profiles.empty() || count <= 0) return;
    for (int i = 0; i < count; ++i) {
        const Vec3 direction = randomDirection();
        ParticleSpawnDesc spark;
        spark.position = centre + direction * radius;
        spark.velocity = direction * randomRange(3.0f, 6.5f);
        spark.lifetime_seconds = randomRange(1.1f, 1.9f);
        spark.mass = 0.0005f;
        spark.appearance_profile_id = fx.sparkle_profiles[randomIndex(fx.sparkle_profiles.size())];
        spark.size_scale = randomRange(0.7f, 1.3f);
        sparks->spawn(spark);
    }
    fx.sparkles_spawned += static_cast<uint64_t>(count);
}

void tickBalloonEffects(SceneData& scene, float dt) {
    if (!fx.prepared) return;
    using Clock = std::chrono::steady_clock;
    const auto ms = [](Clock::time_point a, Clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    const auto t0 = Clock::now();
    ghostContacts(scene, dt);
    const auto t1 = Clock::now();
    tickDropTrails(scene, dt);
    const auto t2 = Clock::now();
    tickSparkles(scene, dt);
    const auto t3 = Clock::now();
    fx.timings.contacts_ms = ms(t0, t1);
    fx.timings.drop_trails_ms = ms(t1, t2);
    fx.timings.sparkles_ms = ms(t2, t3);
}

BalloonEffectsTimings balloonEffectsTimings(SceneData& scene) {
    BalloonEffectsTimings t = fx.timings;
    t.step_ms = 0.0;
    if (fx.prepared) {
        for (const char* name : {kConfettiSystem, kSparkleSystem, kTrailSystem, kGhostSystem}) {
            if (auto* runtime = system(scene, name)) t.step_ms += runtime->stats().total_ms;
        }
    }
    return t;
}

json balloonEffectsStatus(SceneData* scene) {
    json systems = json::object();
    if (fx.prepared && scene) {
        for (const char* name : {kConfettiSystem, kSparkleSystem, kTrailSystem, kGhostSystem}) {
            auto* runtime = system(*scene, name);
            if (!runtime) continue;
            const auto& b = runtime->buffers();
            json min_y = nullptr;
            for (size_t i = 0; i < b.alive.size(); ++i) {
                if (b.alive[i] && (min_y.is_null() || b.position_y[i] < min_y.get<float>())) {
                    min_y = b.position_y[i];
                }
            }
            size_t ready = 0;
            for (const auto& c : runtime->colliders()) {
                if (c.enabled && colliderCooked(c)) ++ready;
            }
            // Lowest live particle: a collider that holds shows min_y resting on
            // the floor; none = it sinks below.
            systems[name] = {{"alive", runtime->aliveCount()}, {"min_y", min_y},
                             {"colliders", runtime->colliders().size()},
                             {"colliders_ready", ready}};
        }
    }
    return {{"systems", systems}, {"sparkles_spawned", fx.sparkles_spawned},
            {"confetti_motes_spawned", fx.confetti_motes_spawned},
            {"trail_spawned", fx.trail_spawned}, {"glitter_spawned", fx.glitter_spawned},
            {"impacts", fx.impacts}, {"impact_sparks", fx.impact_sparks},
            {"drop_contacts_seen", fx.drop_contacts_seen},
            {"drop_sparks_spawned", fx.drop_sparks_spawned},
            {"drop_trail_spawned", fx.drop_trail_spawned},
            {"ghosts_spawned", fx.ghosts_spawned}};
}

} // namespace Paint
