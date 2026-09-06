#pragma once

#include "BezierSpline.h"

namespace MeshEdit {

// Pure spline authoring operations. UI, scripting and IPC should call this
// service instead of mutating BezierSpline::points directly.
class SplineEditService final {
public:
    // segmentIndex identifies the segment start; t is in [0, 1]. The new
    // anchor is inserted at the exact evaluated curve position.
    static bool insertPoint(BezierSpline& spline, int segmentIndex, float t,
                            int* insertedIndex = nullptr);

    // Adds cutCount anchors while preserving the Linear or Bezier segment shape.
    static bool subdivideSegment(BezierSpline& spline, int segmentIndex,
                                 int cutCount, int* lastInsertedIndex = nullptr);

    // Extrude is valid only for an open spline and an endpoint index.
    static bool extrudeEndpoint(BezierSpline& spline, int endpointIndex,
                                const Vec3& position, int* insertedIndex = nullptr);

    // Append at the tail, valid from ZERO points. extrudeEndpoint needs two
    // points to derive a tangent direction, which is why drawing a curve from
    // nothing was impossible: the only way to start one was to place a
    // parametric primitive somewhere and then extrude it. This is the same
    // operation for the 0-, 1- and n-point cases so a Draw tool, the Extrude
    // tool and the script surface all share one append path.
    static bool appendPointAtEnd(BezierSpline& spline, const Vec3& position,
                                 int* insertedIndex = nullptr);

    // Compatibility names for older River/profile call sites. New code should
    // use the curve-type-neutral operations above.
    static bool insertBezierPoint(BezierSpline& spline, int segmentIndex, float t,
                                  int* insertedIndex = nullptr) {
        return insertPoint(spline, segmentIndex, t, insertedIndex);
    }
    static bool subdivideBezierSegment(BezierSpline& spline, int segmentIndex,
                                       int cutCount, int* lastInsertedIndex = nullptr) {
        return subdivideSegment(spline, segmentIndex, cutCount, lastInsertedIndex);
    }
};

} // namespace MeshEdit
