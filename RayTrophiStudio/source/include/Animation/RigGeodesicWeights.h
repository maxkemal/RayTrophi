#pragma once
#include "Animation/RigEnvelopeWeights.h"
#include "json.hpp"
#include <cstdint>
#include <string>

class SceneData;

namespace RigAuthoring {
// Geodesic voxel binding (after Dionne & de Lasa 2013): the bound meshes are voxelized (surface
// rasterized, outside flood-filled, the rest is interior), each bone's distance is grown through
// the solid voxels only, and a vertex's weight falls off with that distance. Distance through
// the body, not through the air, keeps an arm from pulling the torso it hangs beside, and the
// surface shell keeps open or multi-part meshes usable. Output passes the strongest-four contract.
struct GeodesicWeightSettings {
    int resolution = 96; // Voxels along the largest bound extent, 32..256.
    float falloff = 2.f; // Weight = distance^-falloff, .5..8.
};

bool validGeodesicWeightSettings(const GeodesicWeightSettings &settings);
bool previewGeodesicWeights(const SceneData &scene, const std::string &character,
                            const GeodesicWeightSettings &settings, nlohmann::json &report,
                            std::string &error);
// Stages weights for every registered bound mesh into an EnvelopeWeightState with algorithm
// "geodesic_voxel_v1"; envelope settings and bone profiles are carried over unchanged.
bool stageGeodesicWeights(const SceneData &scene, const std::string &character,
                          const GeodesicWeightSettings &settings, uint64_t expectedRevision,
                          EnvelopeWeightState &state, nlohmann::json &report, std::string &error);
} // namespace RigAuthoring
