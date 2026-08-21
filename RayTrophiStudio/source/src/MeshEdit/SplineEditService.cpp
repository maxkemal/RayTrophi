#include "MeshEdit/SplineEditService.h"

#include <algorithm>

namespace MeshEdit {
namespace {

bool validSegment(const BezierSpline& spline, int index) {
    return spline.points.size() >= 2 && index >= 0 &&
        index < static_cast<int>(spline.segmentCount());
}

int nextIndex(const BezierSpline& spline, int index) {
    return (index + 1) % static_cast<int>(spline.points.size());
}

} // namespace

bool SplineEditService::insertBezierPoint(BezierSpline& spline, int segmentIndex,
                                          float t, int* insertedIndex) {
    if (!validSegment(spline, segmentIndex)) return false;
    t = std::clamp(t, 0.001f, 0.999f);
    if (spline.curveType == SplineCurveType::BSpline) return false;

    if (spline.curveType == SplineCurveType::Linear) {
        const int next = nextIndex(spline, segmentIndex);
        BezierControlPoint inserted(
            spline.points[static_cast<size_t>(segmentIndex)].position * (1.0f - t) +
            spline.points[static_cast<size_t>(next)].position * t);
        inserted.autoTangent = false;
        inserted.handleMode = BezierControlPoint::HandleMode::Free;
        const int insertion = segmentIndex + 1;
        spline.points.insert(spline.points.begin() + insertion, inserted);
        if (insertedIndex) *insertedIndex = insertion;
        return true;
    }

    const int next = nextIndex(spline, segmentIndex);
    const auto& a = spline.points[static_cast<size_t>(segmentIndex)];
    const auto& b = spline.points[static_cast<size_t>(next)];
    const Vec3 p0 = a.position;
    const Vec3 p1 = a.position + a.tangentOut;
    const Vec3 p2 = b.position + b.tangentIn;
    const Vec3 p3 = b.position;

    Vec3 left0, left1, left2, left3;
    Vec3 right0, right1, right2, right3;
    BezierMath::subdivideCubic(p0, p1, p2, p3, t,
                               left0, left1, left2, left3,
                               right0, right1, right2, right3);

    spline.points[static_cast<size_t>(segmentIndex)].tangentOut = left1 - left0;
    spline.points[static_cast<size_t>(next)].tangentIn = right2 - right3;

    BezierControlPoint inserted(left3);
    inserted.tangentIn = left2 - left3;
    inserted.tangentOut = right1 - right0;
    inserted.handleMode = BezierControlPoint::HandleMode::Free;
    inserted.autoTangent = false;
    inserted.userData1 = a.userData1 + (b.userData1 - a.userData1) * t;
    inserted.userData2 = a.userData2 + (b.userData2 - a.userData2) * t;
    inserted.userData3 = a.userData3 + (b.userData3 - a.userData3) * t;
    inserted.userColor = a.userColor + (b.userColor - a.userColor) * t;

    const int insertion = segmentIndex + 1;
    if (insertion == static_cast<int>(spline.points.size())) {
        spline.points.push_back(inserted);
    } else {
        spline.points.insert(spline.points.begin() + insertion, inserted);
    }
    if (insertedIndex) *insertedIndex = insertion;
    return true;
}

bool SplineEditService::subdivideBezierSegment(BezierSpline& spline, int segmentIndex,
                                               int cutCount, int* lastInsertedIndex) {
    if (!validSegment(spline, segmentIndex) || cutCount < 1) return false;
    const int originalSegment = segmentIndex;
    int last = -1;
    for (int cut = 1; cut <= cutCount; ++cut) {
        const float target = static_cast<float>(cut) / static_cast<float>(cutCount + 1);
        const float previous = static_cast<float>(cut - 1) / static_cast<float>(cutCount + 1);
        const float localT = (target - previous) / (1.0f - previous);
        if (!insertBezierPoint(spline, originalSegment + cut - 1, localT, &last)) return false;
    }
    if (lastInsertedIndex) *lastInsertedIndex = last;
    return true;
}

bool SplineEditService::extrudeEndpoint(BezierSpline& spline, int endpointIndex,
                                        const Vec3& position, int* insertedIndex) {
    if (spline.isClosed || spline.points.size() < 2) return false;
    if (endpointIndex != 0 && endpointIndex != static_cast<int>(spline.points.size()) - 1) return false;

    const bool atEnd = endpointIndex == static_cast<int>(spline.points.size()) - 1;
    const auto& source = spline.points[static_cast<size_t>(endpointIndex)];
    BezierControlPoint point = source;
    point.position = position;
    point.autoTangent = false;
    point.handleMode = BezierControlPoint::HandleMode::Mirrored;

    const Vec3 direction = atEnd
        ? (source.position - spline.points[spline.points.size() - 2].position)
        : (source.position - spline.points[1].position);
    point.tangentIn = direction * (atEnd ? -0.333f : 0.333f);
    point.tangentOut = point.tangentIn * -1.0f;

    const int insertion = atEnd ? static_cast<int>(spline.points.size()) : 0;
    if (atEnd) spline.points.push_back(point);
    else spline.points.insert(spline.points.begin(), point);
    if (insertedIndex) *insertedIndex = insertion;
    return true;
}

} // namespace MeshEdit
