#pragma once

#include "BezierSpline.h"

#include <string>

namespace MeshEdit {

struct SplineEvaluation {
    Vec3 position = Vec3(0.0f);
    Vec3 tangent = Vec3(1.0f, 0.0f, 0.0f);
    Vec3 normal = Vec3(0.0f, 1.0f, 0.0f);
    Vec3 right = Vec3(0.0f, 0.0f, 1.0f);
    int segment = -1;
    float local_t = 0.0f;
    float global_t = 0.0f;
    bool valid = false;
};

struct SplineClosestPoint {
    SplineEvaluation evaluation;
    float distance_squared = 0.0f;
};

// Curve-type-aware evaluation shared by viewport authoring and geometry
// consumers. The service deliberately owns no scene or UI state.
class SplineEvaluationService final {
public:
    static bool validate(const BezierSpline& spline, std::string* error = nullptr);
    static int segmentCount(const BezierSpline& spline);

    static SplineEvaluation evaluate(const BezierSpline& spline, float t,
                                     const Vec3& up = Vec3(0.0f, 1.0f, 0.0f));
    static SplineEvaluation evaluateSegment(const BezierSpline& spline, int segment,
                                            float localT,
                                            const Vec3& up = Vec3(0.0f, 1.0f, 0.0f));

    static float arcLength(const BezierSpline& spline, int samplesPerSegment = 16);
    static SplineClosestPoint closestPoint(const BezierSpline& spline, const Vec3& point,
                                           int samplesPerSegment = 16,
                                           int refinementSteps = 8);
};

bool runSplineEvaluationSelfTest(std::string* details = nullptr);

} // namespace MeshEdit
