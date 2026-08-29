#include "MeshEdit/SplineEvaluationService.h"
#include "MeshEdit/SplineEditService.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

namespace MeshEdit {
namespace {

Vec3 safeUnit(const Vec3& value, const Vec3& fallback) {
    return value.length_squared() > 1.0e-12f ? value.normalize() : fallback;
}

void buildFrame(SplineEvaluation& result, const Vec3& up) {
    result.tangent = safeUnit(result.tangent, Vec3(1.0f, 0.0f, 0.0f));
    Vec3 right = result.tangent.cross(up);
    if (right.length_squared() <= 1.0e-12f)
        right = result.tangent.cross(Vec3(1.0f, 0.0f, 0.0f));
    if (right.length_squared() <= 1.0e-12f)
        right = result.tangent.cross(Vec3(0.0f, 0.0f, 1.0f));
    result.right = safeUnit(right, Vec3(0.0f, 0.0f, 1.0f));
    result.normal = safeUnit(result.right.cross(result.tangent), up);
}

bool finite(const Vec3& value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

} // namespace

bool SplineEvaluationService::validate(const BezierSpline& spline, std::string* error) {
    const size_t minimum = spline.curveType == SplineCurveType::BSpline ? 4u : 2u;
    if (spline.points.size() < minimum) {
        if (error) *error = spline.curveType == SplineCurveType::BSpline
            ? "Uniform cubic B-spline evaluation requires at least four control points."
            : "Spline evaluation requires at least two control points.";
        return false;
    }
    for (const BezierControlPoint& point : spline.points) {
        if (!finite(point.position) || !finite(point.tangentIn) || !finite(point.tangentOut)) {
            if (error) *error = "Spline control points and handles must be finite.";
            return false;
        }
    }
    return true;
}

int SplineEvaluationService::segmentCount(const BezierSpline& spline) {
    if (spline.points.size() < 2) return 0;
    if (spline.curveType == SplineCurveType::BSpline)
        return static_cast<int>(spline.segmentCount());
    return spline.isClosed ? static_cast<int>(spline.points.size())
                           : static_cast<int>(spline.points.size()) - 1;
}

SplineEvaluation SplineEvaluationService::evaluate(const BezierSpline& spline, float t,
                                                    const Vec3& up) {
    const int segments = segmentCount(spline);
    if (segments <= 0 || !validate(spline)) return {};
    const float globalT = std::clamp(t, 0.0f, 1.0f);
    const float scaled = globalT * static_cast<float>(segments);
    const int segment = std::min(static_cast<int>(std::floor(scaled)), segments - 1);
    const float localT = segment == segments - 1 && globalT >= 1.0f
        ? 1.0f : scaled - static_cast<float>(segment);
    SplineEvaluation result = evaluateSegment(spline, segment, localT, up);
    result.global_t = globalT;
    return result;
}

SplineEvaluation SplineEvaluationService::evaluateSegment(const BezierSpline& spline,
                                                           int segment, float localT,
                                                           const Vec3& up) {
    SplineEvaluation result;
    const int segments = segmentCount(spline);
    if (segment < 0 || segment >= segments || !validate(spline)) return result;

    const float u = std::clamp(localT, 0.0f, 1.0f);
    const size_t i0 = static_cast<size_t>(segment);
    const size_t i1 = (i0 + 1) % spline.points.size();
    if (spline.curveType == SplineCurveType::Linear) {
        result.position = spline.points[i0].position * (1.0f - u) +
                          spline.points[i1].position * u;
        result.tangent = spline.points[i1].position - spline.points[i0].position;
    } else if (spline.curveType == SplineCurveType::Bezier) {
        const BezierControlPoint& a = spline.points[i0];
        const BezierControlPoint& b = spline.points[i1];
        result.position = BezierMath::evaluateCubic(
            a.position, a.position + a.tangentOut,
            b.position + b.tangentIn, b.position, u);
        result.tangent = BezierMath::evaluateCubicTangent(
            a.position, a.position + a.tangentOut,
            b.position + b.tangentIn, b.position, u);
    } else {
        result.position = spline.evaluateBSplineSegment(i0, u);
        result.tangent = spline.evaluateBSplineTangentSegment(i0, u);
    }

    result.segment = segment;
    result.local_t = u;
    result.global_t = (static_cast<float>(segment) + u) / static_cast<float>(segments);
    result.valid = finite(result.position) && finite(result.tangent);
    if (result.valid) buildFrame(result, up);
    return result;
}

float SplineEvaluationService::arcLength(const BezierSpline& spline, int samplesPerSegment) {
    const int segments = segmentCount(spline);
    if (segments <= 0 || !validate(spline)) return 0.0f;
    const int samples = std::max(1, samplesPerSegment);
    float total = 0.0f;
    for (int segment = 0; segment < segments; ++segment) {
        SplineEvaluation previous = evaluateSegment(spline, segment, 0.0f);
        for (int i = 1; i <= samples; ++i) {
            const SplineEvaluation current = evaluateSegment(
                spline, segment, static_cast<float>(i) / static_cast<float>(samples));
            if (!previous.valid || !current.valid) return 0.0f;
            total += (current.position - previous.position).length();
            previous = current;
        }
    }
    return total;
}

SplineClosestPoint SplineEvaluationService::closestPoint(const BezierSpline& spline,
                                                          const Vec3& point,
                                                          int samplesPerSegment,
                                                          int refinementSteps) {
    SplineClosestPoint best;
    best.distance_squared = std::numeric_limits<float>::infinity();
    const int segments = segmentCount(spline);
    if (segments <= 0 || !validate(spline)) return best;
    const int samples = std::max(2, samplesPerSegment);
    for (int segment = 0; segment < segments; ++segment) {
        float bestLocal = 0.0f;
        float bestSegmentDistance = std::numeric_limits<float>::infinity();
        for (int i = 0; i <= samples; ++i) {
            const float local = static_cast<float>(i) / static_cast<float>(samples);
            const SplineEvaluation candidate = evaluateSegment(spline, segment, local);
            const float distance = (candidate.position - point).length_squared();
            if (candidate.valid && distance < bestSegmentDistance) {
                bestSegmentDistance = distance;
                bestLocal = local;
            }
            if (candidate.valid && distance < best.distance_squared) {
                best.distance_squared = distance;
                best.evaluation = candidate;
            }
        }
        float radius = 1.0f / static_cast<float>(samples);
        float low = std::max(0.0f, bestLocal - radius);
        float high = std::min(1.0f, bestLocal + radius);
        for (int step = 0; step < std::max(0, refinementSteps); ++step) {
            const float left = low + (high - low) / 3.0f;
            const float right = high - (high - low) / 3.0f;
            const auto a = evaluateSegment(spline, segment, left);
            const auto b = evaluateSegment(spline, segment, right);
            if ((a.position - point).length_squared() <= (b.position - point).length_squared()) high = right;
            else low = left;
        }
        const auto refined = evaluateSegment(spline, segment, (low + high) * 0.5f);
        const float distance = (refined.position - point).length_squared();
        if (refined.valid && distance < best.distance_squared) {
            best.distance_squared = distance;
            best.evaluation = refined;
        }
    }
    return best;
}

bool runSplineEvaluationSelfTest(std::string* details) {
    BezierSpline linear;
    linear.curveType = SplineCurveType::Linear;
    linear.addPoint(Vec3(0.0f, 0.0f, 0.0f));
    linear.addPoint(Vec3(4.0f, 0.0f, 0.0f));
    const auto middle = SplineEvaluationService::evaluate(linear, 0.5f);
    const auto closest = SplineEvaluationService::closestPoint(linear, Vec3(1.0f, 2.0f, 0.0f));

    BezierSpline bspline;
    bspline.curveType = SplineCurveType::BSpline;
    bspline.addPoint(Vec3(0.0f, 0.0f, 0.0f));
    bspline.addPoint(Vec3(1.0f, 0.0f, 0.0f));
    bspline.addPoint(Vec3(2.0f, 0.0f, 0.0f));
    bspline.addPoint(Vec3(3.0f, 0.0f, 0.0f));
    const auto cubicMiddle = SplineEvaluationService::evaluate(bspline, 0.5f);
    const auto beforeInsert = SplineEvaluationService::evaluateSegment(bspline, 0, 0.37f);
    int insertedControl = -1;
    const bool knotInserted = SplineEditService::insertPoint(
        bspline, 0, 0.37f, &insertedControl);
    const auto afterInsertLeft = SplineEvaluationService::evaluateSegment(bspline, 0, 1.0f);
    const auto afterInsertRight = SplineEvaluationService::evaluateSegment(bspline, 1, 0.0f);

    const bool pass = middle.valid && cubicMiddle.valid && closest.evaluation.valid &&
        knotInserted && insertedControl >= 0 && afterInsertLeft.valid && afterInsertRight.valid &&
        std::abs(middle.position.x - 2.0f) < 1.0e-4f &&
        std::abs(cubicMiddle.position.x - 1.5f) < 1.0e-4f &&
        (beforeInsert.position - afterInsertLeft.position).length() < 1.0e-4f &&
        (beforeInsert.position - afterInsertRight.position).length() < 1.0e-4f &&
        std::abs(SplineEvaluationService::arcLength(linear) - 4.0f) < 1.0e-4f &&
        std::abs(closest.evaluation.position.x - 1.0f) < 2.0e-2f;
    if (details) {
        std::ostringstream out;
        out << (pass ? "PASS" : "FAIL")
            << " linear_mid=" << middle.position.x
            << " bspline_mid=" << cubicMiddle.position.x
            << " knot_insert=" << (knotInserted ? "ok" : "failed")
            << " length=" << SplineEvaluationService::arcLength(linear)
            << " closest=" << closest.evaluation.position.x;
        *details = out.str();
    }
    return pass;
}

} // namespace MeshEdit
