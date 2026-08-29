#pragma once

#include "BezierSpline.h"
#include "Vec3.h"
#include "json.hpp"

#include <cstddef>
#include <string>
#include <vector>

struct TimelineManager;
struct ObjectAnimationTrack;

namespace MeshEdit {

class SplineObject;

struct SplinePointAnimationState {
    Vec3 position = Vec3(0.0f);
    Vec3 tangent_in = Vec3(0.0f);
    Vec3 tangent_out = Vec3(0.0f);
    float user_data1 = 1.0f;
    float user_data2 = 0.0f;
    float user_data3 = 0.0f;
    Vec3 user_color = Vec3(1.0f);
    int handle_mode = 2;
    bool auto_tangent = true;
};

// Point/control animation payload. Object TRS remains in TransformKeyframe so
// an entire spline can move while its controls deform independently.
struct SplineAnimationSnapshot {
    SplineCurveType curve_type = SplineCurveType::Bezier;
    bool closed = false;
    std::vector<float> knots;
    std::vector<SplinePointAnimationState> points;
};

struct SplineKeyframeInfo {
    int frame = 0;
    bool has_object_transform = false;
    bool has_points = false;
    std::size_t point_count = 0;
};

SplineAnimationSnapshot captureSplineAnimation(const SplineObject& object);
bool splineAnimationTopologyCompatible(const SplineAnimationSnapshot& a,
                                       const SplineAnimationSnapshot& b,
                                       std::string* error = nullptr);
SplineAnimationSnapshot interpolateSplineAnimation(const SplineAnimationSnapshot& a,
                                                    const SplineAnimationSnapshot& b,
                                                    float t);
bool evaluateSplineAnimationSnapshot(const ObjectAnimationTrack& track, int frame,
                                     SplineAnimationSnapshot& out);
bool applySplineAnimation(SplineObject& object, const SplineAnimationSnapshot& snapshot,
                          std::string* error = nullptr);

bool insertSplineAnimationKey(TimelineManager& timeline, const SplineObject& object,
                              int frame, bool includeObjectTransform, bool includePoints,
                              std::string* error = nullptr);
bool removeSplineAnimationKey(TimelineManager& timeline, const std::string& objectName,
                              int frame, bool removeObjectTransform, bool removePoints,
                              std::string* error = nullptr);
std::vector<SplineKeyframeInfo> listSplineAnimationKeys(
    const TimelineManager& timeline, const std::string& objectName);
bool applySplineAnimationTrack(SplineObject& object, const ObjectAnimationTrack& track,
                               int frame, bool* objectTransformChanged = nullptr,
                               bool* pointsChanged = nullptr, std::string* error = nullptr);

// Keep already-authored point keys structurally aligned when authoring changes
// spline topology. Each key receives the same curve-space operation, so adding
// a control does not disable or index-shift the existing deformation animation.
bool propagateSplineInsertToKeys(TimelineManager& timeline, const std::string& objectName,
                                 int segment, float t, std::string* error = nullptr);
bool propagateSplineSubdivideToKeys(TimelineManager& timeline, const std::string& objectName,
                                    int segment, int cuts, std::string* error = nullptr);
bool propagateSplineExtrudeToKeys(TimelineManager& timeline, const std::string& objectName,
                                  const BezierSpline& before, int endpoint,
                                  const Vec3& newPosition, std::string* error = nullptr);
bool propagateSplineRemovePointToKeys(TimelineManager& timeline, const std::string& objectName,
                                      int pointIndex, std::string* error = nullptr);

void to_json(nlohmann::json& j, const SplinePointAnimationState& value);
void from_json(const nlohmann::json& j, SplinePointAnimationState& value);
void to_json(nlohmann::json& j, const SplineAnimationSnapshot& value);
void from_json(const nlohmann::json& j, SplineAnimationSnapshot& value);

bool runSplineAnimationSelfTest(std::string* details = nullptr);

} // namespace MeshEdit
