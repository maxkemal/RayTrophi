#include "MeshEdit/SplineAnimation.h"

#include "KeyframeSystem.h"
#include "MeshEdit/SplineObject.h"
#include "MeshEdit/SplineEditService.h"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace MeshEdit {
namespace {

Vec3 lerpVec(const Vec3& a, const Vec3& b, float t) {
    return a * (1.0f - t) + b * t;
}

bool finiteVec(const Vec3& value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

bool finitePoint(const SplinePointAnimationState& point) {
    return finiteVec(point.position) && finiteVec(point.tangent_in) &&
        finiteVec(point.tangent_out) && finiteVec(point.user_color) &&
        std::isfinite(point.user_data1) && std::isfinite(point.user_data2) &&
        std::isfinite(point.user_data3);
}

BezierSpline snapshotSpline(const SplineAnimationSnapshot& snapshot) {
    BezierSpline spline;
    spline.curveType = snapshot.curve_type;
    spline.isClosed = snapshot.closed;
    spline.knots = snapshot.knots;
    spline.points.reserve(snapshot.points.size());
    for (const auto& state : snapshot.points) {
        BezierControlPoint point;
        point.position = state.position;
        point.tangentIn = state.tangent_in;
        point.tangentOut = state.tangent_out;
        point.userData1 = state.user_data1;
        point.userData2 = state.user_data2;
        point.userData3 = state.user_data3;
        point.userColor = state.user_color;
        point.handleMode = static_cast<BezierControlPoint::HandleMode>(
            std::clamp(state.handle_mode, 0, 2));
        point.autoTangent = state.auto_tangent;
        spline.points.push_back(point);
    }
    return spline;
}

SplineAnimationSnapshot splineSnapshot(const BezierSpline& spline) {
    SplineAnimationSnapshot snapshot;
    snapshot.curve_type = spline.curveType;
    snapshot.closed = spline.isClosed;
    snapshot.knots = spline.knots;
    snapshot.points.reserve(spline.points.size());
    for (const auto& point : spline.points) {
        snapshot.points.push_back({point.position, point.tangentIn, point.tangentOut,
            point.userData1, point.userData2, point.userData3, point.userColor,
            static_cast<int>(point.handleMode), point.autoTangent});
    }
    return snapshot;
}

template <typename Mutation>
bool mutateSplineKeys(TimelineManager& timeline, const std::string& objectName,
                      Mutation&& mutation, std::string* error) {
    const auto trackIt = timeline.tracks.find(objectName);
    if (trackIt == timeline.tracks.end()) return true;
    std::vector<std::pair<Keyframe*, SplineAnimationSnapshot>> updates;
    for (auto& key : trackIt->second.keyframes) {
        if (!key.has_spline) continue;
        BezierSpline spline = snapshotSpline(key.spline);
        if (!mutation(spline)) {
            if (error) *error = "could not apply spline topology edit to key at frame " +
                                std::to_string(key.frame);
            return false;
        }
        updates.emplace_back(&key, splineSnapshot(spline));
    }
    for (auto& update : updates) update.first->spline = std::move(update.second);
    return true;
}

} // namespace

SplineAnimationSnapshot captureSplineAnimation(const SplineObject& object) {
    SplineAnimationSnapshot result;
    result.curve_type = object.spline.curveType;
    result.closed = object.spline.isClosed;
    result.knots = object.spline.knots;
    result.points.reserve(object.spline.points.size());
    for (const auto& point : object.spline.points) {
        SplinePointAnimationState state;
        state.position = point.position;
        state.tangent_in = point.tangentIn;
        state.tangent_out = point.tangentOut;
        state.user_data1 = point.userData1;
        state.user_data2 = point.userData2;
        state.user_data3 = point.userData3;
        state.user_color = point.userColor;
        state.handle_mode = static_cast<int>(point.handleMode);
        state.auto_tangent = point.autoTangent;
        result.points.push_back(state);
    }
    return result;
}

bool splineAnimationTopologyCompatible(const SplineAnimationSnapshot& a,
                                       const SplineAnimationSnapshot& b,
                                       std::string* error) {
    if (a.curve_type != b.curve_type) {
        if (error) *error = "spline animation keys must use the same curve type";
        return false;
    }
    if (a.closed != b.closed) {
        if (error) *error = "spline animation keys must use the same open/closed state";
        return false;
    }
    if (a.points.size() != b.points.size()) {
        if (error) *error = "spline animation keys must have the same control-point count";
        return false;
    }
    if (a.knots.size() != b.knots.size()) {
        if (error) *error = "spline animation keys must have the same knot topology";
        return false;
    }
    for (size_t i = 0; i < a.knots.size(); ++i) {
        if (std::abs(a.knots[i] - b.knots[i]) > 1.0e-6f) {
            if (error) *error = "spline knot values cannot change between animation keys";
            return false;
        }
    }
    return true;
}

SplineAnimationSnapshot interpolateSplineAnimation(const SplineAnimationSnapshot& a,
                                                    const SplineAnimationSnapshot& b,
                                                    float t) {
    std::string ignored;
    if (!splineAnimationTopologyCompatible(a, b, &ignored)) return t < 1.0f ? a : b;
    const float u = std::clamp(t, 0.0f, 1.0f);
    SplineAnimationSnapshot result = a;
    for (size_t i = 0; i < result.points.size(); ++i) {
        auto& out = result.points[i];
        const auto& lhs = a.points[i];
        const auto& rhs = b.points[i];
        out.position = lerpVec(lhs.position, rhs.position, u);
        out.tangent_in = lerpVec(lhs.tangent_in, rhs.tangent_in, u);
        out.tangent_out = lerpVec(lhs.tangent_out, rhs.tangent_out, u);
        out.user_data1 = lhs.user_data1 * (1.0f - u) + rhs.user_data1 * u;
        out.user_data2 = lhs.user_data2 * (1.0f - u) + rhs.user_data2 * u;
        out.user_data3 = lhs.user_data3 * (1.0f - u) + rhs.user_data3 * u;
        out.user_color = lerpVec(lhs.user_color, rhs.user_color, u);
        out.handle_mode = u < 1.0f ? lhs.handle_mode : rhs.handle_mode;
        out.auto_tangent = u < 1.0f ? lhs.auto_tangent : rhs.auto_tangent;
    }
    return result;
}

bool evaluateSplineAnimationSnapshot(const ObjectAnimationTrack& track, int frame,
                                     SplineAnimationSnapshot& out) {
    const Keyframe* previous = nullptr;
    const Keyframe* next = nullptr;
    for (const auto& key : track.keyframes) {
        if (!key.has_spline) continue;
        if (key.frame <= frame) previous = &key;
        if (key.frame >= frame) { next = &key; break; }
    }
    if (!previous && !next) return false;
    if (!previous) { out = next->spline; return true; }
    if (!next || previous == next) { out = previous->spline; return true; }
    const float span = static_cast<float>(next->frame - previous->frame);
    const float t = span > 0.0f
        ? static_cast<float>(frame - previous->frame) / span : 0.0f;
    out = interpolateSplineAnimation(previous->spline, next->spline, t);
    return true;
}

bool applySplineAnimation(SplineObject& object, const SplineAnimationSnapshot& snapshot,
                          std::string* error) {
    const SplineAnimationSnapshot current = captureSplineAnimation(object);
    if (!splineAnimationTopologyCompatible(current, snapshot, error)) return false;
    for (size_t i = 0; i < snapshot.points.size(); ++i) {
        const auto& source = snapshot.points[i];
        if (!finitePoint(source)) {
            if (error) *error = "spline animation contains a non-finite point value";
            return false;
        }
        auto& target = object.spline.points[i];
        target.position = source.position;
        target.tangentIn = source.tangent_in;
        target.tangentOut = source.tangent_out;
        target.userData1 = std::max(0.0f, source.user_data1);
        target.userData2 = source.user_data2;
        target.userData3 = source.user_data3;
        target.userColor = source.user_color;
        target.handleMode = static_cast<BezierControlPoint::HandleMode>(
            std::clamp(source.handle_mode, 0, 2));
        target.autoTangent = source.auto_tangent;
    }
    if (std::any_of(object.spline.points.begin(), object.spline.points.end(),
                    [](const BezierControlPoint& point) { return point.autoTangent; })) {
        object.spline.calculateAutoTangents();
    }
    return true;
}

bool insertSplineAnimationKey(TimelineManager& timeline, const SplineObject& object,
                              int frame, bool includeObjectTransform, bool includePoints,
                              std::string* error) {
    if (frame < 0) { if (error) *error = "frame must be non-negative"; return false; }
    if (!includeObjectTransform && !includePoints) {
        if (error) *error = "at least one spline key channel must be enabled";
        return false;
    }
    Keyframe key(frame);
    if (includeObjectTransform) {
        if (!object.transform) { if (error) *error = "spline has no object transform"; return false; }
        key.has_transform = true;
        key.transform = TransformKeyframe(object.transform->position,
                                          object.transform->rotation,
                                          object.transform->scale);
    }
    if (includePoints) {
        key.spline = captureSplineAnimation(object);
        key.has_spline = true;
        const auto trackIt = timeline.tracks.find(object.nodeName);
        if (trackIt != timeline.tracks.end()) {
            for (const auto& existing : trackIt->second.keyframes) {
                if (!existing.has_spline) continue;
                if (!splineAnimationTopologyCompatible(existing.spline, key.spline, error)) return false;
            }
        }
    }
    timeline.insertKeyframe(object.nodeName, key);
    return true;
}

bool removeSplineAnimationKey(TimelineManager& timeline, const std::string& objectName,
                              int frame, bool removeObjectTransform, bool removePoints,
                              std::string* error) {
    auto trackIt = timeline.tracks.find(objectName);
    if (trackIt == timeline.tracks.end()) {
        if (error) *error = "spline animation track was not found";
        return false;
    }
    Keyframe* key = trackIt->second.getKeyframeAt(frame);
    if (!key) { if (error) *error = "spline animation key was not found at frame"; return false; }
    if (removeObjectTransform) {
        key->has_transform = false;
        key->transform.clearAllChannels();
    }
    if (removePoints) {
        key->has_spline = false;
        key->spline = SplineAnimationSnapshot{};
    }
    const bool empty = !key->has_transform && !key->has_material && !key->has_light &&
        !key->has_camera && !key->has_world && !key->has_terrain && !key->has_water &&
        !key->has_emitter && !key->has_anim_graph && !key->has_spline;
    if (empty) trackIt->second.removeKeyframe(frame);
    if (trackIt->second.keyframes.empty()) timeline.tracks.erase(trackIt);
    return true;
}

std::vector<SplineKeyframeInfo> listSplineAnimationKeys(
    const TimelineManager& timeline, const std::string& objectName) {
    std::vector<SplineKeyframeInfo> result;
    const auto trackIt = timeline.tracks.find(objectName);
    if (trackIt == timeline.tracks.end()) return result;
    for (const auto& key : trackIt->second.keyframes) {
        if (!key.has_transform && !key.has_spline) continue;
        result.push_back({key.frame, key.has_transform, key.has_spline,
                          key.has_spline ? key.spline.points.size() : 0u});
    }
    return result;
}

bool applySplineAnimationTrack(SplineObject& object, const ObjectAnimationTrack& track,
                               int frame, bool* objectTransformChanged,
                               bool* pointsChanged, std::string* error) {
    if (objectTransformChanged) *objectTransformChanged = false;
    if (pointsChanged) *pointsChanged = false;
    const Keyframe evaluated = track.evaluate(frame);
    if (evaluated.has_transform && object.transform) {
        if (evaluated.transform.has_position) object.transform->position = evaluated.transform.position;
        if (evaluated.transform.has_rotation) object.transform->rotation = evaluated.transform.rotation;
        if (evaluated.transform.has_scale) object.transform->scale = evaluated.transform.scale;
        object.transform->updateMatrix();
        if (objectTransformChanged) *objectTransformChanged = true;
    }
    if (evaluated.has_spline) {
        if (!applySplineAnimation(object, evaluated.spline, error)) return false;
        if (pointsChanged) *pointsChanged = true;
    }
    return evaluated.has_transform || evaluated.has_spline;
}

bool propagateSplineInsertToKeys(TimelineManager& timeline, const std::string& objectName,
                                 int segment, float t, std::string* error) {
    return mutateSplineKeys(timeline, objectName,
        [=](BezierSpline& spline) {
            return SplineEditService::insertPoint(spline, segment, t, nullptr);
        }, error);
}

bool propagateSplineSubdivideToKeys(TimelineManager& timeline, const std::string& objectName,
                                    int segment, int cuts, std::string* error) {
    return mutateSplineKeys(timeline, objectName,
        [=](BezierSpline& spline) {
            return SplineEditService::subdivideSegment(spline, segment, cuts, nullptr);
        }, error);
}

bool propagateSplineExtrudeToKeys(TimelineManager& timeline, const std::string& objectName,
                                  const BezierSpline& before, int endpoint,
                                  const Vec3& newPosition, std::string* error) {
    if (endpoint < 0 || endpoint >= static_cast<int>(before.points.size())) {
        if (error) *error = "invalid spline endpoint";
        return false;
    }
    const bool atEnd = endpoint == static_cast<int>(before.points.size()) - 1;
    const Vec3 delta = newPosition - before.points[static_cast<size_t>(endpoint)].position;
    return mutateSplineKeys(timeline, objectName,
        [=](BezierSpline& spline) {
            if (spline.points.empty()) return false;
            const int keyEndpoint = atEnd ? static_cast<int>(spline.points.size()) - 1 : 0;
            const Vec3 keyPosition = spline.points[static_cast<size_t>(keyEndpoint)].position + delta;
            return SplineEditService::extrudeEndpoint(spline, keyEndpoint, keyPosition, nullptr);
        }, error);
}

bool propagateSplineRemovePointToKeys(TimelineManager& timeline, const std::string& objectName,
                                      int pointIndex, std::string* error) {
    return mutateSplineKeys(timeline, objectName,
        [=](BezierSpline& spline) {
            if (pointIndex < 0 || pointIndex >= static_cast<int>(spline.points.size())) return false;
            spline.removePoint(pointIndex);
            return true;
        }, error);
}

void to_json(nlohmann::json& j, const SplinePointAnimationState& value) {
    j = {{"p", value.position}, {"ti", value.tangent_in}, {"to", value.tangent_out},
         {"u", {value.user_data1, value.user_data2, value.user_data3}},
         {"c", value.user_color}, {"hm", value.handle_mode}, {"auto", value.auto_tangent}};
}

void from_json(const nlohmann::json& j, SplinePointAnimationState& value) {
    if (j.contains("p")) j.at("p").get_to(value.position);
    if (j.contains("ti")) j.at("ti").get_to(value.tangent_in);
    if (j.contains("to")) j.at("to").get_to(value.tangent_out);
    if (j.contains("u") && j["u"].is_array() && j["u"].size() >= 3) {
        value.user_data1 = j["u"][0].get<float>();
        value.user_data2 = j["u"][1].get<float>();
        value.user_data3 = j["u"][2].get<float>();
    }
    if (j.contains("c")) j.at("c").get_to(value.user_color);
    value.handle_mode = std::clamp(j.value("hm", 2), 0, 2);
    value.auto_tangent = j.value("auto", true);
}

void to_json(nlohmann::json& j, const SplineAnimationSnapshot& value) {
    j = {{"curve", static_cast<int>(value.curve_type)}, {"closed", value.closed},
         {"knots", value.knots}, {"points", value.points}};
}

void from_json(const nlohmann::json& j, SplineAnimationSnapshot& value) {
    value.curve_type = static_cast<SplineCurveType>(std::clamp(j.value("curve", 1), 0, 2));
    value.closed = j.value("closed", false);
    value.knots = j.value("knots", std::vector<float>{});
    value.points = j.value("points", std::vector<SplinePointAnimationState>{});
}

bool runSplineAnimationSelfTest(std::string* details) {
    SplineObject object;
    object.nodeName = "SplineAnimationSelfTest";
    object.spline.curveType = SplineCurveType::Bezier;
    object.spline.addPoint(Vec3(0.0f, 0.0f, 0.0f));
    object.spline.addPoint(Vec3(2.0f, 0.0f, 0.0f));
    TimelineManager timeline;
    std::string error;
    const bool first = insertSplineAnimationKey(timeline, object, 0, true, true, &error);
    object.spline.points[0].position.y = 4.0f;
    object.spline.points[0].userData1 = 3.0f;
    object.transform->position = Vec3(10.0f, 0.0f, 0.0f);
    object.transform->updateMatrix();
    const bool second = insertSplineAnimationKey(timeline, object, 10, true, true, &error);
    int insertedIndex = -1;
    const bool objectInserted = SplineEditService::insertPoint(
        object.spline, 0, 0.5f, &insertedIndex);
    const bool keysInserted = objectInserted && propagateSplineInsertToKeys(
        timeline, object.nodeName, 0, 0.5f, &error);
    SplineObject evaluated;
    evaluated.nodeName = object.nodeName;
    evaluated.spline = object.spline;
    bool transformChanged = false, pointsChanged = false;
    const bool applied = applySplineAnimationTrack(
        evaluated, timeline.tracks[object.nodeName], 5,
        &transformChanged, &pointsChanged, &error);
    const bool topologyPreserved = evaluated.spline.points.size() == 3 &&
        timeline.tracks[object.nodeName].keyframes.size() == 2 &&
        timeline.tracks[object.nodeName].keyframes[0].spline.points.size() == 3 &&
        timeline.tracks[object.nodeName].keyframes[1].spline.points.size() == 3;
    const bool pass = first && second && objectInserted && keysInserted && applied &&
        transformChanged && pointsChanged && topologyPreserved &&
        std::abs(evaluated.spline.points[0].position.y - 2.0f) < 1.0e-4f &&
        std::abs(evaluated.spline.points[0].userData1 - 2.0f) < 1.0e-4f &&
        std::abs(evaluated.transform->position.x - 5.0f) < 1.0e-4f;
    if (details) {
        std::ostringstream out;
        out << (pass ? "PASS" : "FAIL")
            << " point_y=" << evaluated.spline.points[0].position.y
            << " radius=" << evaluated.spline.points[0].userData1
            << " object_x=" << evaluated.transform->position.x
            << " points_after_insert=" << evaluated.spline.points.size();
        if (!error.empty()) out << " error=" << error;
        *details = out.str();
    }
    return pass;
}

} // namespace MeshEdit
