#include "Api/RtApi.h"
#include "Api/RtApiInternal.h"
#include "MeshEdit/SplineEditService.h"
#include "MeshEdit/SplineSerialization.h"
#include "MeshEdit/SplineObject.h"
#include "MeshEdit/SplineObjectService.h"
#include "MeshEdit/SplineAnimation.h"
#include "MeshEdit/SplineSurfaceAuthoring.h"
#include "TerrainRoadNetwork.h"
#include "Ray.h"
#include "ProjectManager.h"
#include "json.hpp"
#include "TerrainManager.h"
#include "NodeSystem/Graph.h"

#include <algorithm>
#include <optional>

namespace rtapi {
namespace {

std::shared_ptr<MeshEdit::SplineObject> findSpline(const std::string& name) {
    if (!g_ctx) return {};
    for (const auto& object : g_ctx->scene.world.objects) {
        auto spline = std::dynamic_pointer_cast<MeshEdit::SplineObject>(object);
        if (spline && spline->nodeName == name) return spline;
    }
    return {};
}

class SplineSnapshotCommand final : public SceneCommand {
public:
    SplineSnapshotCommand(std::shared_ptr<MeshEdit::SplineObject> spline,
                          nlohmann::json before, nlohmann::json after,
                          std::string description,
                          std::optional<ObjectAnimationTrack> beforeTrack = std::nullopt,
                          std::optional<ObjectAnimationTrack> afterTrack = std::nullopt)
        : spline_(std::move(spline)), before_(std::move(before)),
          after_(std::move(after)), description_(std::move(description)),
          beforeTrack_(std::move(beforeTrack)), afterTrack_(std::move(afterTrack)) {}

    void execute(UIContext& ctx) override { apply(ctx, after_, afterTrack_); }
    void undo(UIContext& ctx) override { apply(ctx, before_, beforeTrack_); }
    Type getType() const override { return Type::Generic; }
    std::string getDescription() const override { return description_; }

private:
    void apply(UIContext& ctx, const nlohmann::json& state,
               const std::optional<ObjectAnimationTrack>& track) {
        if (!spline_) return;
        std::string error;
        if (MeshEdit::deserializeSpline(state, *spline_, error)) {
            if (track) ctx.scene.timeline.tracks[spline_->nodeName] = *track;
            // A spline is a source object, but its dependent profile/modifier
            // output still needs the same scene mutation notification as any
            // other authoring command.
            scheduleSceneMutationRebuilds(ctx, false);
            ctx.start_render = true;
        }
    }

    std::shared_ptr<MeshEdit::SplineObject> spline_;
    nlohmann::json before_;
    nlohmann::json after_;
    std::string description_;
    std::optional<ObjectAnimationTrack> beforeTrack_;
    std::optional<ObjectAnimationTrack> afterTrack_;
};

Result writableSpline(const std::string& name,
                      std::shared_ptr<MeshEdit::SplineObject>& out) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (!g_history) return Result::fail("rtapi has no SceneHistory bound");
    out = findSpline(name);
    if (!out) return Result::fail("spline not found: " + name);
    return Result::success();
}

void recordSplineEdit(const std::shared_ptr<MeshEdit::SplineObject>& spline,
                      const nlohmann::json& before, const std::string& description,
                      const std::optional<ObjectAnimationTrack>& beforeTrack = std::nullopt) {
    const nlohmann::json after = MeshEdit::serializeSpline(*spline);
    std::optional<ObjectAnimationTrack> afterTrack;
    if (beforeTrack) {
        const auto it = g_ctx->scene.timeline.tracks.find(spline->nodeName);
        if (it != g_ctx->scene.timeline.tracks.end()) afterTrack = it->second;
    }
    g_history->record(std::make_unique<SplineSnapshotCommand>(
        spline, before, after, description, beforeTrack, afterTrack));
    ProjectManager::getInstance().markModified();
}

std::optional<ObjectAnimationTrack> captureSplineTrack(const std::string& name) {
    const auto it = g_ctx->scene.timeline.tracks.find(name);
    return it == g_ctx->scene.timeline.tracks.end()
        ? std::nullopt : std::optional<ObjectAnimationTrack>(it->second);
}

void restoreSplineTransaction(const std::shared_ptr<MeshEdit::SplineObject>& spline,
                              const nlohmann::json& objectState,
                              const std::optional<ObjectAnimationTrack>& trackState) {
    std::string ignored;
    MeshEdit::deserializeSpline(objectState, *spline, ignored);
    if (trackState) g_ctx->scene.timeline.tracks[spline->nodeName] = *trackState;
}

} // namespace

Result listSplines(std::vector<SplineInfo>& out) {
    if (!g_ctx) return notBound();
    out.clear();
    for (const auto& object : g_ctx->scene.world.objects) {
        auto spline = std::dynamic_pointer_cast<MeshEdit::SplineObject>(object);
        if (!spline) continue;
        out.push_back({spline->nodeName,
                       MeshEdit::splineCurveTypeName(spline->spline.curveType),
                       static_cast<int>(spline->plane), spline->spline.isClosed,
                       spline->spline.points.size()});
    }
    return Result::success();
}

Result createSpline(const std::string& primitive, const std::string& name,
                    const std::string& plane, std::string& out_name) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (!g_history) return Result::fail("rtapi has no SceneHistory bound");

    MeshEdit::SplinePrimitiveType type;
    if (primitive == "circle") type = MeshEdit::SplinePrimitiveType::Circle;
    else if (primitive == "rectangle") type = MeshEdit::SplinePrimitiveType::Rectangle;
    else if (primitive == "line" || primitive == "open_line")
        type = MeshEdit::SplinePrimitiveType::OpenLine;
    else if (primitive == "arc" || primitive == "open_arc")
        type = MeshEdit::SplinePrimitiveType::OpenArc;
    else if (primitive == "empty") type = MeshEdit::SplinePrimitiveType::Empty;
    else return Result::fail("unknown spline primitive '" + primitive +
                             "' (expected circle|rectangle|open_line|open_arc|empty)");

    MeshEdit::SplinePlane targetPlane;
    if (plane == "xy") targetPlane = MeshEdit::SplinePlane::XY;
    else if (plane == "xz") targetPlane = MeshEdit::SplinePlane::XZ;
    else if (plane == "yz") targetPlane = MeshEdit::SplinePlane::YZ;
    // Free is the plane a drawn route needs: a profile is planar, a path is not.
    else if (plane == "free") targetPlane = MeshEdit::SplinePlane::Free;
    else return Result::fail("unknown spline plane '" + plane + "' (expected xy|xz|yz|free)");

    auto object = MeshEdit::addSplinePrimitiveObject(
        *g_ctx, *g_history, type, name, targetPlane);
    if (!object) return Result::fail("failed to create spline object");
    out_name = object->nodeName;
    scheduleSceneMutationRebuilds(*g_ctx, false);
    g_ctx->start_render = true;
    ProjectManager::getInstance().markModified();
    return Result::success();
}

Result getSpline(const std::string& name, std::string& out_json) {
    if (!g_ctx) return notBound();
    const auto spline = findSpline(name);
    if (!spline) return Result::fail("spline not found: " + name);
    out_json = MeshEdit::serializeSpline(*spline).dump();
    return Result::success();
}

Result setSpline(const std::string& name, const std::string& json_payload) {
    std::shared_ptr<MeshEdit::SplineObject> spline;
    const Result guard = writableSpline(name, spline);
    if (!guard.ok) return guard;
    const nlohmann::json before = MeshEdit::serializeSpline(*spline);
    try {
        std::string error;
        if (!MeshEdit::deserializeSpline(nlohmann::json::parse(json_payload), *spline, error))
            return Result::fail(error);
        const auto trackIt = g_ctx->scene.timeline.tracks.find(name);
        if (trackIt != g_ctx->scene.timeline.tracks.end()) {
            const auto edited = MeshEdit::captureSplineAnimation(*spline);
            for (const auto& key : trackIt->second.keyframes) {
                if (!key.has_spline) continue;
                if (!MeshEdit::splineAnimationTopologyCompatible(key.spline, edited, &error)) {
                    std::string restoreError;
                    MeshEdit::deserializeSpline(before, *spline, restoreError);
                    return Result::fail(
                        "animated spline topology must be changed with insert/subdivide/extrude: " + error);
                }
            }
        }
    } catch (const std::exception& e) {
        return Result::fail(std::string("invalid spline JSON: ") + e.what());
    }
    recordSplineEdit(spline, before, "Set spline " + name);
    return Result::success();
}

Result insertSplinePoint(const std::string& name, int segment, float t, int& out_index) {
    std::shared_ptr<MeshEdit::SplineObject> spline;
    const Result guard = writableSpline(name, spline);
    if (!guard.ok) return guard;
    const nlohmann::json before = MeshEdit::serializeSpline(*spline);
    const auto beforeTrack = captureSplineTrack(name);
    if (!MeshEdit::SplineEditService::insertPoint(spline->spline, segment, t, &out_index))
        return Result::fail("spline insert is invalid for the requested segment or curve type");
    std::string topologyError;
    if (!MeshEdit::propagateSplineInsertToKeys(
            g_ctx->scene.timeline, name, segment, t, &topologyError)) {
        restoreSplineTransaction(spline, before, beforeTrack);
        return Result::fail(topologyError);
    }
    recordSplineEdit(spline, before, "Insert spline point " + name, beforeTrack);
    return Result::success();
}

Result subdivideSpline(const std::string& name, const std::vector<int>& segments,
                       int cuts, int& out_last_index) {
    if (segments.empty()) return Result::fail("segments must not be empty");
    if (cuts < 1 || cuts > 128) return Result::fail("cuts must be between 1 and 128");
    std::shared_ptr<MeshEdit::SplineObject> spline;
    const Result guard = writableSpline(name, spline);
    if (!guard.ok) return guard;
    const nlohmann::json before = MeshEdit::serializeSpline(*spline);
    const auto beforeTrack = captureSplineTrack(name);
    std::vector<int> ordered = segments;
    std::sort(ordered.begin(), ordered.end(), std::greater<int>());
    bool changed = false;
    out_last_index = -1;
    for (const int segment : ordered) {
        int inserted = -1;
        if (MeshEdit::SplineEditService::subdivideSegment(
                spline->spline, segment, cuts, &inserted)) {
            std::string topologyError;
            if (!MeshEdit::propagateSplineSubdivideToKeys(
                    g_ctx->scene.timeline, name, segment, cuts, &topologyError)) {
                restoreSplineTransaction(spline, before, beforeTrack);
                return Result::fail(topologyError);
            }
            changed = true;
            out_last_index = inserted;
        }
    }
    if (!changed) return Result::fail("no requested spline segment could be subdivided");
    recordSplineEdit(spline, before, "Subdivide spline " + name, beforeTrack);
    return Result::success();
}

Result extrudeSplineEndpoint(const std::string& name, int endpoint, const Vec3& position,
                             int& out_index) {
    std::shared_ptr<MeshEdit::SplineObject> spline;
    const Result guard = writableSpline(name, spline);
    if (!guard.ok) return guard;
    const nlohmann::json before = MeshEdit::serializeSpline(*spline);
    const auto beforeTrack = captureSplineTrack(name);
    const BezierSpline beforeSpline = spline->spline;
    if (!MeshEdit::SplineEditService::extrudeEndpoint(
            spline->spline, endpoint, position, &out_index))
        return Result::fail("extrude requires an open spline endpoint");
    std::string topologyError;
    if (!MeshEdit::propagateSplineExtrudeToKeys(
            g_ctx->scene.timeline, name, beforeSpline, endpoint, position, &topologyError)) {
        restoreSplineTransaction(spline, before, beforeTrack);
        return Result::fail(topologyError);
    }
    recordSplineEdit(spline, before, "Extrude spline endpoint " + name, beforeTrack);
    return Result::success();
}

Result raycastScene(const Vec3& origin, const Vec3& direction,
                    const std::string& filter, RaycastHit& out) {
    if (!g_ctx) return notBound();
    out = RaycastHit{};

    const float length = direction.length();
    if (!(length > 1e-6f)) return Result::fail("direction must be a non-zero vector");

    MeshEdit::SurfaceFilter surfaceFilter;
    if (filter.empty() || filter == "mesh_and_terrain")
        surfaceFilter = MeshEdit::SurfaceFilter::MeshAndTerrain;
    else if (filter == "terrain_only") surfaceFilter = MeshEdit::SurfaceFilter::TerrainOnly;
    else if (filter == "ground_plane") surfaceFilter = MeshEdit::SurfaceFilter::GroundPlaneOnly;
    // Refused rather than defaulted: a typo silently falling back to the full
    // policy would report a mesh hit to a caller that asked for terrain only,
    // and nothing about the answer would look wrong.
    else return Result::fail("unknown raycast filter '" + filter +
                             "' (expected mesh_and_terrain|terrain_only|ground_plane)");

    const Ray ray(origin, direction / length);
    MeshEdit::SurfaceSnapResult snap;
    if (!MeshEdit::raycastSurface(g_ctx->scene, ray, surfaceFilter, snap)) {
        out.kind = "none";
        return Result::success();
    }
    out.hit = true;
    switch (snap.kind) {
    case MeshEdit::SurfaceHitKind::Mesh: out.kind = "mesh"; break;
    case MeshEdit::SurfaceHitKind::Terrain: out.kind = "terrain"; break;
    case MeshEdit::SurfaceHitKind::GroundPlane: out.kind = "ground_plane"; break;
    default: out.kind = "none"; break;
    }
    out.object = snap.object_name;
    out.position = snap.position;
    out.normal = snap.normal;
    out.distance = snap.distance;
    return Result::success();
}

Result appendSplinePoint(const std::string& name, const Vec3& position, int& out_index) {
    std::shared_ptr<MeshEdit::SplineObject> spline;
    const Result guard = writableSpline(name, spline);
    if (!guard.ok) return guard;
    const nlohmann::json before = MeshEdit::serializeSpline(*spline);
    const auto beforeTrack = captureSplineTrack(name);
    const BezierSpline beforeSpline = spline->spline;
    const int previousEndpoint = beforeSpline.points.empty()
        ? -1 : static_cast<int>(beforeSpline.points.size()) - 1;
    if (!MeshEdit::SplineEditService::appendPointAtEnd(spline->spline, position, &out_index))
        return Result::fail("append requires an open spline");
    // Keyframe propagation is a topology edit and only applies once the append
    // was an extrude, i.e. the curve already had two points. The first two
    // points create the curve rather than change its shape.
    if (previousEndpoint >= 0 && spline->spline.points.size() > 2) {
        std::string topologyError;
        if (!MeshEdit::propagateSplineExtrudeToKeys(
                g_ctx->scene.timeline, name, beforeSpline, previousEndpoint, position,
                &topologyError)) {
            restoreSplineTransaction(spline, before, beforeTrack);
            return Result::fail(topologyError);
        }
    }
    recordSplineEdit(spline, before, "Append spline point " + name, beforeTrack);
    return Result::success();
}

Result insertSplineKeyframe(const std::string& name, int frame,
                            bool include_object_transform, bool include_points) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    const auto spline = findSpline(name);
    if (!spline) return Result::fail("spline not found: " + name);
    std::string error;
    if (!MeshEdit::insertSplineAnimationKey(g_ctx->scene.timeline, *spline, frame,
                                            include_object_transform, include_points, &error)) {
        return Result::fail(error);
    }
    g_ctx->start_render = true;
    ProjectManager::getInstance().markModified();
    return Result::success();
}

Result removeSplineKeyframe(const std::string& name, int frame,
                            bool remove_object_transform, bool remove_points) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (!findSpline(name)) return Result::fail("spline not found: " + name);
    std::string error;
    if (!MeshEdit::removeSplineAnimationKey(g_ctx->scene.timeline, name, frame,
                                            remove_object_transform, remove_points, &error)) {
        return Result::fail(error);
    }
    g_ctx->start_render = true;
    ProjectManager::getInstance().markModified();
    return Result::success();
}

Result listSplineKeyframes(const std::string& name, std::vector<SplineKeyInfo>& out) {
    if (!g_ctx) return notBound();
    if (!findSpline(name)) return Result::fail("spline not found: " + name);
    out.clear();
    for (const auto& info : MeshEdit::listSplineAnimationKeys(g_ctx->scene.timeline, name)) {
        out.push_back({info.frame, info.has_object_transform, info.has_points, info.point_count});
    }
    return Result::success();
}

Result splineAnimationSelfTest(std::string& out_details) {
    return MeshEdit::runSplineAnimationSelfTest(&out_details)
        ? Result::success() : Result::fail(out_details);
}


// ---------------------------------------------------------------------------
// Road assignments
// ---------------------------------------------------------------------------
namespace {

bool splineObjectExists(const std::string& name) {
    return static_cast<bool>(findSpline(name));
}

RoadCarveValues toCarveValues(const TerrainNodesV2::RoadCarveSettings& settings) {
    RoadCarveValues values;
    values.road_width = settings.roadWidthMeters;
    values.shoulder_width = settings.shoulderWidthMeters;
    values.grading_falloff = settings.gradingFalloffMeters;
    values.foliage_margin = settings.foliageExclusionMarginMeters;
    values.max_grade_percent = settings.maxGradePercent;
    values.elevation_offset = settings.elevationOffsetMeters;
    values.max_cut_meters = settings.maxCutMeters;
    values.max_fill_meters = settings.maxFillMeters;
    values.crown_meters = settings.crownMeters;
    values.ditch_width = settings.ditchWidthMeters;
    values.ditch_depth = settings.ditchDepthMeters;
    values.use_point_width = settings.usePointWidth;
    return values;
}

TerrainNodesV2::RoadCarveSettings fromCarveValues(const RoadCarveValues& values) {
    TerrainNodesV2::RoadCarveSettings settings;
    settings.roadWidthMeters = values.road_width;
    settings.shoulderWidthMeters = values.shoulder_width;
    settings.gradingFalloffMeters = values.grading_falloff;
    settings.foliageExclusionMarginMeters = values.foliage_margin;
    settings.maxGradePercent = values.max_grade_percent;
    settings.elevationOffsetMeters = values.elevation_offset;
    settings.maxCutMeters = values.max_cut_meters;
    settings.maxFillMeters = values.max_fill_meters;
    settings.crownMeters = values.crown_meters;
    settings.ditchWidthMeters = values.ditch_width;
    settings.ditchDepthMeters = values.ditch_depth;
    settings.usePointWidth = values.use_point_width;
    return settings;
}

RoadAssignmentInfo toAssignmentInfo(const TerrainNodesV2::RoadAssignment& assignment) {
    RoadAssignmentInfo info;
    info.spline_object = assignment.splineObject;
    info.profile_id = assignment.profileId;
    info.crossing_mode = TerrainNodesV2::roadCrossingModeName(assignment.crossingMode);
    info.enabled = assignment.enabled;
    info.has_override = assignment.hasOverride;
    info.curve_exists = splineObjectExists(assignment.splineObject);
    // effectiveCarve() is the solver's own accessor, not a reimplementation of
    // its rules here - the readback and the solve cannot disagree.
    info.effective = toCarveValues(assignment.effectiveCarve());
    return info;
}

void invalidateTerrainGraphs() {
    for (auto& terrain : TerrainManager::getInstance().getTerrains()) {
        if (terrain.nodeGraph) {
            terrain.nodeGraph->markAllDirty();
        }
    }
}

} // namespace

Result listRoadProfiles(std::vector<RoadProfileInfo>& out) {
    out.clear();
    for (const auto& profile : TerrainNodesV2::builtinRoadProfiles()) {
        RoadProfileInfo info;
        info.id = profile.id;
        info.display_name = profile.displayName;
        info.carve = toCarveValues(profile.carve);
        out.push_back(std::move(info));
    }
    return Result::success();
}

Result assignRoadProfile(const std::string& spline_object, const std::string& profile_id) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    // Refused rather than accepted-and-flagged: an assignment pointing at a curve
    // that does not exist would look identical to a working one in every listing.
    if (!splineObjectExists(spline_object))
        return Result::fail("spline not found: " + spline_object);
    std::string error;
    if (!TerrainNodesV2::RoadNetworkRegistry::getInstance().assignProfile(
            spline_object, profile_id, &error)) {
        return Result::fail(error);
    }
    ProjectManager::getInstance().markModified();
    invalidateTerrainGraphs();
    return Result::success();
}

Result clearRoadProfile(const std::string& spline_object) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (!TerrainNodesV2::RoadNetworkRegistry::getInstance().clearProfile(spline_object))
        return Result::fail("no road assignment for '" + spline_object + "'");
    ProjectManager::getInstance().markModified();
    invalidateTerrainGraphs();
    return Result::success();
}

Result getRoadAssignment(const std::string& spline_object, RoadAssignmentInfo& out) {
    if (!g_ctx) return notBound();
    const auto* assignment =
        TerrainNodesV2::RoadNetworkRegistry::getInstance().find(spline_object);
    if (!assignment) return Result::fail("no road assignment for '" + spline_object + "'");
    out = toAssignmentInfo(*assignment);
    return Result::success();
}

Result listRoadAssignments(std::vector<RoadAssignmentInfo>& out) {
    if (!g_ctx) return notBound();
    out.clear();
    for (const auto& assignment :
         TerrainNodesV2::RoadNetworkRegistry::getInstance().assignments()) {
        out.push_back(toAssignmentInfo(assignment));
    }
    return Result::success();
}

Result setRoadCrossingMode(const std::string& spline_object, const std::string& mode) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    TerrainNodesV2::RoadCrossingMode parsed;
    if (!TerrainNodesV2::parseRoadCrossingMode(mode, parsed))
        return Result::fail("unknown crossing mode '" + mode +
                            "' (expected auto|terrain|bridge|ford|tunnel)");
    std::string error;
    if (!TerrainNodesV2::RoadNetworkRegistry::getInstance().setCrossingMode(
            spline_object, parsed, &error)) {
        return Result::fail(error);
    }
    ProjectManager::getInstance().markModified();
    invalidateTerrainGraphs();
    return Result::success();
}

Result setRoadEnabled(const std::string& spline_object, bool enabled) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    std::string error;
    if (!TerrainNodesV2::RoadNetworkRegistry::getInstance().setEnabled(
            spline_object, enabled, &error)) {
        return Result::fail(error);
    }
    ProjectManager::getInstance().markModified();
    invalidateTerrainGraphs();
    return Result::success();
}

Result setRoadCarveOverride(const std::string& spline_object, const RoadCarveValues& values) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    const TerrainNodesV2::RoadCarveSettings settings = fromCarveValues(values);
    std::string error;
    // The SAME validator the solver runs. A second copy of these bounds here
    // would eventually accept what the solve refuses, and the road would simply
    // stop carving with nothing reported anywhere.
    if (!TerrainNodesV2::validateRoadCarveSettings(settings, spline_object, &error))
        return Result::fail(error);
    if (!TerrainNodesV2::RoadNetworkRegistry::getInstance().setCarveOverride(
            spline_object, settings, &error)) {
        return Result::fail(error);
    }
    ProjectManager::getInstance().markModified();
    invalidateTerrainGraphs();
    return Result::success();
}

Result clearRoadCarveOverride(const std::string& spline_object) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    std::string error;
    if (!TerrainNodesV2::RoadNetworkRegistry::getInstance().clearCarveOverride(
            spline_object, &error)) {
        return Result::fail(error);
    }
    ProjectManager::getInstance().markModified();
    invalidateTerrainGraphs();
    return Result::success();
}

Result getRoadDiagnostics(RoadDiagnostics& out) {
    if (!g_ctx) return notBound();
    out = RoadDiagnostics{};
    const auto& registry = TerrainNodesV2::RoadNetworkRegistry::getInstance();
    for (const auto& assignment : registry.assignments()) {
        ++out.assignment_count;
        if (assignment.enabled) ++out.enabled_count;
        TerrainNodesV2::RoadProfile profile;
        if (!assignment.hasOverride &&
            !TerrainNodesV2::findRoadProfile(assignment.profileId, profile)) {
            out.unknown_profiles.push_back(assignment.splineObject);
        }
    }
    out.dangling = registry.danglingAssignments(&splineObjectExists);
    return Result::success();
}

} // namespace rtapi
