#include "Api/RtApi.h"
#include "Api/RtApiInternal.h"
#include "MeshEdit/SplineEditService.h"
#include "MeshEdit/SplineSerialization.h"
#include "MeshEdit/SplineObject.h"
#include "MeshEdit/SplineObjectService.h"
#include "MeshEdit/SplineAnimation.h"
#include "ProjectManager.h"
#include "json.hpp"

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
    else return Result::fail("unknown spline primitive '" + primitive +
                             "' (expected circle|rectangle|open_line|open_arc)");

    MeshEdit::SplinePlane targetPlane;
    if (plane == "xy") targetPlane = MeshEdit::SplinePlane::XY;
    else if (plane == "xz") targetPlane = MeshEdit::SplinePlane::XZ;
    else if (plane == "yz") targetPlane = MeshEdit::SplinePlane::YZ;
    else return Result::fail("unknown spline plane '" + plane + "' (expected xy|xz|yz)");

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

} // namespace rtapi
