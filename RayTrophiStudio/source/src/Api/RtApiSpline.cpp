#include "Api/RtApi.h"
#include "Api/RtApiInternal.h"
#include "MeshEdit/SplineEditService.h"
#include "MeshEdit/SplineSerialization.h"
#include "MeshEdit/SplineObject.h"
#include "ProjectManager.h"
#include "json.hpp"

#include <algorithm>

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
                          std::string description)
        : spline_(std::move(spline)), before_(std::move(before)),
          after_(std::move(after)), description_(std::move(description)) {}

    void execute(UIContext& ctx) override { apply(ctx, after_); }
    void undo(UIContext& ctx) override { apply(ctx, before_); }
    Type getType() const override { return Type::Generic; }
    std::string getDescription() const override { return description_; }

private:
    void apply(UIContext& ctx, const nlohmann::json& state) {
        if (!spline_) return;
        std::string error;
        if (MeshEdit::deserializeSpline(state, *spline_, error)) {
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
                      const nlohmann::json& before, const std::string& description) {
    const nlohmann::json after = MeshEdit::serializeSpline(*spline);
    g_history->record(std::make_unique<SplineSnapshotCommand>(
        spline, before, after, description));
    ProjectManager::getInstance().markModified();
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
    if (!MeshEdit::SplineEditService::insertBezierPoint(spline->spline, segment, t, &out_index))
        return Result::fail("spline insert is invalid for the requested segment or curve type");
    recordSplineEdit(spline, before, "Insert spline point " + name);
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
    std::vector<int> ordered = segments;
    std::sort(ordered.begin(), ordered.end(), std::greater<int>());
    bool changed = false;
    out_last_index = -1;
    for (const int segment : ordered) {
        int inserted = -1;
        if (MeshEdit::SplineEditService::subdivideBezierSegment(
                spline->spline, segment, cuts, &inserted)) {
            changed = true;
            out_last_index = inserted;
        }
    }
    if (!changed) return Result::fail("no requested spline segment could be subdivided");
    recordSplineEdit(spline, before, "Subdivide spline " + name);
    return Result::success();
}

Result extrudeSplineEndpoint(const std::string& name, int endpoint, const Vec3& position,
                             int& out_index) {
    std::shared_ptr<MeshEdit::SplineObject> spline;
    const Result guard = writableSpline(name, spline);
    if (!guard.ok) return guard;
    const nlohmann::json before = MeshEdit::serializeSpline(*spline);
    if (!MeshEdit::SplineEditService::extrudeEndpoint(
            spline->spline, endpoint, position, &out_index))
        return Result::fail("extrude requires an open spline endpoint");
    recordSplineEdit(spline, before, "Extrude spline endpoint " + name);
    return Result::success();
}

} // namespace rtapi
