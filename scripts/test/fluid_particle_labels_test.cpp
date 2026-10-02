// Standalone core regression. Build/run is performed by the user, not Codex.
#include "Fluid/FluidParticleLabels.h"
#include "Fluid/FluidParticles.h"
#include "Fluid/FluidFoam.h"

#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>
#include <algorithm>
#include <random>

using namespace RayTrophiSim::Fluid;

static ParticleLabel label(const FluidParticles& particles, std::size_t i) {
    return particleLabel(particles.flags[i]);
}

// Independent O(N^2) oracle: no spatial buckets or capped traversal. This checks
// the optimized lookup against the geometric contract, including warm scratch
// reuse across different domains, radius changes and identical particle counts.
static void compareWithBruteForce(FluidParticles& particles, float voxel,
                                 const std::vector<uint32_t>& solid_tags) {
    const double radius = static_cast<double>(voxel) * kParticleLabelRadiusVoxels;
    std::vector<uint32_t> expected = particles.flags;
    const auto eligible = [&](std::size_t i) {
        const auto& p = particles.position[i];
        return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z) &&
            particles.mass_fraction[i] > kParticleMistMassFraction &&
            std::find(solid_tags.begin(), solid_tags.end(), particles.substance_tag[i]) ==
                solid_tags.end();
    };
    for (std::size_t i = 0; i < particles.size(); ++i) {
        ParticleLabel result = ParticleLabel::Unknown;
        if ((particles.flags[i] & kParticleFlagFrozen) != 0u) {
            result = ParticleLabel::Frozen;
        } else if (particles.mass_fraction[i] > 0.0f &&
                   particles.mass_fraction[i] <= kParticleMistMassFraction &&
                   std::find(solid_tags.begin(), solid_tags.end(),
                             particles.substance_tag[i]) == solid_tags.end()) {
            result = ParticleLabel::Mist;
        } else if (eligible(i)) {
            unsigned neighbors = 0;
            for (std::size_t j = 0; j < particles.size(); ++j) {
                if (j == i || !eligible(j)) {
                    continue;
                }
                const auto& p = particles.position[i];
                const auto& q = particles.position[j];
                const double x = static_cast<double>(p.x) - q.x;
                const double y = static_cast<double>(p.y) - q.y;
                const double z = static_cast<double>(p.z) - q.z;
                if (x * x + y * y + z * z <= radius * radius) {
                    ++neighbors;
                }
            }
            result = neighbors <= 2 ? ParticleLabel::Spray :
                neighbors >= 6 ? ParticleLabel::Body :
                label(particles, i) == ParticleLabel::Spray ? ParticleLabel::Spray :
                ParticleLabel::Body;
        }
        setParticleLabel(expected[i], result);
    }
    const auto measured = updateParticleLabels(particles, voxel, false, &solid_tags);
    assert(particles.flags == expected);
    assert(measured.center_resolved <= particles.size());
    assert(measured.occupied_bins <= particles.size());
    assert(std::abs(measured.milliseconds - measured.bin_milliseconds -
                    measured.classify_milliseconds) < 0.0001);
}

static void checkOptimizedLookup() {
    std::mt19937 random(20260928u);
    std::uniform_real_distribution<float> position(-3.0f, 3.0f);
    for (int scene = 0; scene < 12; ++scene) {
        FluidParticles particles;
        const int count = scene % 3 == 0 ? 384 : scene % 3 == 1 ? 17 : 384;
        const float offset = scene % 2 == 0 ? -20.0f : 80.0f;
        for (int i = 0; i < count; ++i) {
            const float spread = i < count / 2 ? 0.08f : 1.0f;
            particles.emit(Vec3(offset + position(random) * spread,
                                position(random) * spread, position(random) * spread),
                           Vec3(0.0f), 293.0f, 0.0f, i % 7 == 0 ? 42u : 0u);
            setParticleLabel(particles.flags.back(), i % 2 == 0
                ? ParticleLabel::Body : ParticleLabel::Spray);
            if (i % 13 == 0) {
                particles.flags.back() |= kParticleFlagFrozen;
            }
        }
        const std::vector<uint32_t> solids = scene % 2 == 0
            ? std::vector<uint32_t>{42u} : std::vector<uint32_t>{};
        compareWithBruteForce(particles, scene % 2 == 0 ? 0.2f : 0.5f, solids);
        for (auto& p : particles.position) {
            p.x += 17.0f;
            p.y *= 0.3f;
        }
        compareWithBruteForce(particles, 0.3f, solids);
    }

    // Exactly on the radius and immediately outside it; negative cell boundary.
    FluidParticles boundary;
    boundary.emit(Vec3(-1.5f, 0.0f, 0.0f), Vec3(0.0f));
    for (int i = 0; i < 6; ++i) {
        boundary.emit(Vec3(0.0f), Vec3(0.0f));
    }
    compareWithBruteForce(boundary, 1.0f, {});
    assert(label(boundary, 0) == ParticleLabel::Body);
    boundary.position[0].x = std::nextafter(-1.5f, -2.0f);
    compareWithBruteForce(boundary, 1.0f, {});
    assert(label(boundary, 0) == ParticleLabel::Spray);
}

int main() {
    checkOptimizedLookup();
    FluidParticles particles;
    particles.emit(Vec3(-10.0f, 0.0f, 0.0f), Vec3(1.0f, 2.0f, 3.0f));
    assert(label(particles, 0) == ParticleLabel::Unknown);
    particles.mass_fraction[0] = 0.4f;
    particles.flags[0] |= 2u; // unrelated outflow bit must survive
    auto stats = updateParticleLabels(particles, 1.0f, false);
    assert(label(particles, 0) == ParticleLabel::Spray);
    assert(stats.particles == 1 && stats.changed == 1);
    assert(particles.mass_fraction[0] == 0.4f && particles.flags[0] & 2u);
    assert(particles.velocity[0].x == 1.0f && particles.position[0].x == -10.0f);

    // A genuinely low-mass remnant is mist even inside a dense neighbourhood.
    particles.mass_fraction[0] = kParticleMistMassFraction;
    updateParticleLabels(particles, 1.0f, false);
    assert(label(particles, 0) == ParticleLabel::Mist);
    particles.mass_fraction[0] = 0.4f;

    // Dense support rejoins; label does not depend on world-coordinate sign.
    for (int i = 0; i < 6; ++i) {
        particles.emit(Vec3(-10.0f + 0.05f * (i + 1), 0.0f, 0.0f), Vec3(0.0f));
    }
    updateParticleLabels(particles, 1.0f, false);
    assert(label(particles, 0) == ParticleLabel::Body);

    // Four neighbours lie in the hysteresis gap: keep the measured prior state.
    particles.compact({0, 0, 0, 0, 0, 1, 1});
    updateParticleLabels(particles, 1.0f, false);
    assert(label(particles, 0) == ParticleLabel::Body);
    setParticleLabel(particles.flags[0], ParticleLabel::Spray);
    updateParticleLabels(particles, 1.0f, false);
    assert(label(particles, 0) == ParticleLabel::Spray);

    particles.flags[0] |= kParticleFlagFrozen;
    updateParticleLabels(particles, 1.0f, false);
    assert(label(particles, 0) == ParticleLabel::Frozen);
    particles.flags[0] &= ~kParticleFlagFrozen;
    updateParticleLabels(particles, 1.0f, false);
    assert(label(particles, 0) == ParticleLabel::Body);

    // Compaction must carry labels, not just positions. New parcels stay unknown.
    particles.flags.back() |= kParticleFlagFrozen;
    updateParticleLabels(particles, 1.0f, false);
    particles.removeSwap(0);
    assert(label(particles, 0) == ParticleLabel::Frozen);
    particles.flags.back() |= kParticleFlagFrozen;
    updateParticleLabels(particles, 1.0f, false);
    particles.compact({1, 0, 0, 0});
    assert(label(particles, particles.size() - 1) == ParticleLabel::Frozen);
    particles.emit(Vec3(100.0f), Vec3(0.0f));
    assert(label(particles, particles.size() - 1) == ParticleLabel::Unknown);

    // Invalid geometry cannot fabricate a body or cause float-to-int UB.
    FluidParticles invalid;
    invalid.emit(Vec3((std::numeric_limits<float>::quiet_NaN)(), 0.0f, 0.0f), Vec3(0.0f));
    invalid.emit(Vec3((std::numeric_limits<float>::max)(), 0.0f, 0.0f), Vec3(0.0f));
    updateParticleLabels(invalid, 1.0f, false);
    assert(label(invalid, 0) == ParticleLabel::Unknown);
    assert(label(invalid, 1) == ParticleLabel::Unknown);
    invalid.position[0] = Vec3(0.0f);
    updateParticleLabels(invalid, 0.0f, false);
    assert(label(invalid, 0) == ParticleLabel::Unknown);

    FluidParticles solids;
    solids.emit(Vec3(0.0f), Vec3(0.0f), 293.0f, 0.0f, 42u);
    const std::vector<uint32_t> solid_tags{42u};
    updateParticleLabels(solids, 1.0f, false, &solid_tags);
    assert(label(solids, 0) == ParticleLabel::Unknown);
    updateParticleLabels(solids, 1.0f, true);
    assert(label(solids, 0) == ParticleLabel::Unknown);
    solids.flags[0] |= kParticleFlagFrozen;
    updateParticleLabels(solids, 1.0f, true);
    assert(label(solids, 0) == ParticleLabel::Frozen);

    const auto counts = countParticleLabels(particles);
    assert(std::accumulate(counts.begin(), counts.end(), uint64_t{0}) == particles.size());
    FoamParticles foam;
    foam.emit(Vec3(0.0f), Vec3(0.0f), 1.0f, FoamType::Spray);
    foam.emit(Vec3(0.0f), Vec3(0.0f), 1.0f, FoamType::Foam);
    foam.emit(Vec3(0.0f), Vec3(0.0f), 1.0f, FoamType::Bubble);
    const auto secondary = countSecondaryParticleLabels(foam);
    assert(secondary[static_cast<std::size_t>(ParticleLabel::Spray)] == 1);
    assert(secondary[static_cast<std::size_t>(ParticleLabel::Foam)] == 1);
    assert(secondary[static_cast<std::size_t>(ParticleLabel::Bubble)] == 1);
    assert(secondary[static_cast<std::size_t>(ParticleLabel::Mist)] == 0);
    assert(std::accumulate(secondary.begin(), secondary.end(), uint64_t{0}) == foam.size());
    particles.clear();
    assert(countParticleLabels(particles) == ParticleLabelCounts{});
    assert(updateParticleLabels(particles, 1.0f, false).particles == 0);
    return 0;
}
