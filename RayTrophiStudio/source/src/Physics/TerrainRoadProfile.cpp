#include "TerrainRoadProfile.h"

#include <algorithm>
#include <cmath>

namespace TerrainNodesV2 {

const char* roadCrossingModeName(RoadCrossingMode mode) {
    switch (mode) {
        case RoadCrossingMode::Terrain: return "terrain";
        case RoadCrossingMode::Bridge:  return "bridge";
        case RoadCrossingMode::Ford:    return "ford";
        case RoadCrossingMode::Tunnel:  return "tunnel";
        case RoadCrossingMode::Auto:
        default:                        return "auto";
    }
}

bool parseRoadCrossingMode(const std::string& text, RoadCrossingMode& out) {
    if (text == "auto")    { out = RoadCrossingMode::Auto;    return true; }
    if (text == "terrain") { out = RoadCrossingMode::Terrain; return true; }
    if (text == "bridge")  { out = RoadCrossingMode::Bridge;  return true; }
    if (text == "ford")    { out = RoadCrossingMode::Ford;    return true; }
    if (text == "tunnel")  { out = RoadCrossingMode::Tunnel;  return true; }
    return false;
}

std::vector<RoadProfile> builtinRoadProfiles() {
    std::vector<RoadProfile> profiles;

    // A footpath follows the ground: it is allowed to be steep and it barely
    // moves earth. Giving it the same cut/fill budget as a road is what turns a
    // hiking trail into a highway cutting.
    RoadProfile footpath;
    footpath.id = "footpath";
    footpath.displayName = "Footpath";
    footpath.carve.roadWidthMeters = 1.2f;
    footpath.carve.shoulderWidthMeters = 0.4f;
    footpath.carve.gradingFalloffMeters = 0.8f;
    footpath.carve.foliageExclusionMarginMeters = 0.6f;
    footpath.carve.maxGradePercent = 25.0f;
    footpath.carve.maxCutMeters = 1.5f;
    footpath.carve.maxFillMeters = 1.0f;
    // A trail sheds off its own camber and has no ditch to dig. Giving it one
    // would carve a drainage channel either side of a hiking path.
    footpath.carve.crownMeters = 0.03f;
    footpath.carve.ditchWidthMeters = 0.0f;
    footpath.carve.ditchDepthMeters = 0.0f;
    profiles.push_back(footpath);

    RoadProfile dirt;
    dirt.id = "dirt_road";
    dirt.displayName = "Dirt Road";
    dirt.carve.roadWidthMeters = 4.0f;
    dirt.carve.shoulderWidthMeters = 1.5f;
    dirt.carve.gradingFalloffMeters = 3.0f;
    dirt.carve.foliageExclusionMarginMeters = 2.0f;
    dirt.carve.maxGradePercent = 15.0f;
    dirt.carve.maxCutMeters = 5.0f;
    dirt.carve.maxFillMeters = 4.0f;
    dirt.carve.crownMeters = 0.10f;
    dirt.carve.ditchWidthMeters = 0.9f;
    dirt.carve.ditchDepthMeters = 0.35f;
    profiles.push_back(dirt);

    // A main road buys its shallow grade with earthworks, so its budget is the
    // largest of the three - but still bounded. Unbounded is what produced
    // mountain-high embankments before the cut/fill envelope existed.
    RoadProfile main;
    main.id = "main_road";
    main.displayName = "Main Road";
    main.carve.roadWidthMeters = 8.0f;
    main.carve.shoulderWidthMeters = 2.5f;
    main.carve.gradingFalloffMeters = 6.0f;
    main.carve.foliageExclusionMarginMeters = 4.0f;
    main.carve.maxGradePercent = 8.0f;
    main.carve.maxCutMeters = 12.0f;
    main.carve.maxFillMeters = 9.0f;
    // Crown and ditch are the reason a main road does not read as a river bed
    // to the flow solver: the surface sheds and the ditch carries.
    main.carve.crownMeters = 0.18f;
    main.carve.ditchWidthMeters = 1.8f;
    main.carve.ditchDepthMeters = 0.7f;
    profiles.push_back(main);

    return profiles;
}

bool findRoadProfile(const std::string& id, RoadProfile& out) {
    const auto profiles = builtinRoadProfiles();
    const auto found = std::find_if(profiles.begin(), profiles.end(),
        [&id](const RoadProfile& profile) { return profile.id == id; });
    if (found == profiles.end()) return false;
    out = *found;
    return true;
}

bool validateRoadCarveSettings(const RoadCarveSettings& settings, const std::string& label,
                               std::string* error) {
    const std::string prefix = label.empty() ? std::string() : ("'" + label + "': ");
    const float values[] = {settings.roadWidthMeters, settings.shoulderWidthMeters,
                            settings.gradingFalloffMeters,
                            settings.foliageExclusionMarginMeters,
                            settings.maxGradePercent, settings.elevationOffsetMeters,
                            settings.crownMeters, settings.ditchWidthMeters,
                            settings.ditchDepthMeters};
    for (float value : values) {
        if (!std::isfinite(value)) {
            if (error) *error = prefix + "all road profile values must be finite";
            return false;
        }
    }
    if (settings.roadWidthMeters <= 0.0f || settings.shoulderWidthMeters < 0.0f ||
        settings.gradingFalloffMeters < 0.0f ||
        settings.foliageExclusionMarginMeters < 0.0f ||
        settings.maxGradePercent < 0.0f || settings.maxGradePercent > 100.0f) {
        if (error) *error = prefix + "invalid width, falloff, exclusion margin, or grade";
        return false;
    }
    if (!std::isfinite(settings.maxCutMeters) || !std::isfinite(settings.maxFillMeters) ||
        settings.maxCutMeters < 0.0f || settings.maxFillMeters < 0.0f) {
        if (error) *error = prefix + "max cut and max fill must be finite and non-negative";
        return false;
    }
    // A negative crown is a road that collects water down its centreline, and a
    // ditch deeper than the cut budget is a trench the envelope never sees. Both
    // are refused rather than clamped: a clamped write reads back as a write
    // that landed and quietly did something else.
    if (settings.crownMeters < 0.0f || settings.ditchWidthMeters < 0.0f ||
        settings.ditchDepthMeters < 0.0f) {
        if (error) *error = prefix + "crown, ditch width and ditch depth must be non-negative";
        return false;
    }
    if (settings.crownMeters > 1.0f) {
        if (error) *error = prefix + "crown above 1 m is a roof, not a road camber";
        return false;
    }
    return true;
}

nlohmann::json roadCarveSettingsToJson(const RoadCarveSettings& settings) {
    nlohmann::json value;
    value["roadWidthMeters"] = settings.roadWidthMeters;
    value["shoulderWidthMeters"] = settings.shoulderWidthMeters;
    value["gradingFalloffMeters"] = settings.gradingFalloffMeters;
    value["foliageExclusionMarginMeters"] = settings.foliageExclusionMarginMeters;
    value["maxGradePercent"] = settings.maxGradePercent;
    value["elevationOffsetMeters"] = settings.elevationOffsetMeters;
    value["maxCutMeters"] = settings.maxCutMeters;
    value["maxFillMeters"] = settings.maxFillMeters;
    value["crownMeters"] = settings.crownMeters;
    value["ditchWidthMeters"] = settings.ditchWidthMeters;
    value["ditchDepthMeters"] = settings.ditchDepthMeters;
    value["usePointWidth"] = settings.usePointWidth;
    return value;
}

RoadCarveSettings roadCarveSettingsFromJson(const nlohmann::json& value,
                                            const RoadCarveSettings& fallback) {
    RoadCarveSettings settings = fallback;
    if (!value.is_object()) return settings;
    settings.roadWidthMeters =
        (std::max)(0.01f, value.value("roadWidthMeters", settings.roadWidthMeters));
    settings.shoulderWidthMeters =
        (std::max)(0.0f, value.value("shoulderWidthMeters", settings.shoulderWidthMeters));
    settings.gradingFalloffMeters =
        (std::max)(0.0f, value.value("gradingFalloffMeters", settings.gradingFalloffMeters));
    settings.foliageExclusionMarginMeters =
        (std::max)(0.0f, value.value("foliageExclusionMarginMeters",
                                     settings.foliageExclusionMarginMeters));
    settings.maxGradePercent = (std::max)(0.0f, (std::min)(100.0f,
        value.value("maxGradePercent", settings.maxGradePercent)));
    settings.elevationOffsetMeters =
        value.value("elevationOffsetMeters", settings.elevationOffsetMeters);
    settings.maxCutMeters = (std::max)(0.0f, value.value("maxCutMeters", settings.maxCutMeters));
    settings.maxFillMeters = (std::max)(0.0f, value.value("maxFillMeters", settings.maxFillMeters));
    settings.crownMeters = (std::max)(0.0f, value.value("crownMeters", settings.crownMeters));
    settings.ditchWidthMeters =
        (std::max)(0.0f, value.value("ditchWidthMeters", settings.ditchWidthMeters));
    settings.ditchDepthMeters =
        (std::max)(0.0f, value.value("ditchDepthMeters", settings.ditchDepthMeters));
    settings.usePointWidth = value.value("usePointWidth", settings.usePointWidth);
    return settings;
}

} // namespace TerrainNodesV2
