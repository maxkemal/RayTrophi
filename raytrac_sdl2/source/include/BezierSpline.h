// ═══════════════════════════════════════════════════════════════════════════════
// BEZIER SPLINE SYSTEM - Reusable Bezier Curve Primitives
// ═══════════════════════════════════════════════════════════════════════════════
// Use cases: Rivers, Hair/Fur, Camera Paths, Road Systems, Cables, etc.
// ═══════════════════════════════════════════════════════════════════════════════
#pragma once

#include "Vec3.h"
#include <vector>
#include <cmath>
#include <algorithm>

// ═══════════════════════════════════════════════════════════════════════════════
// BEZIER CONTROL POINT
// ═══════════════════════════════════════════════════════════════════════════════
struct BezierControlPoint {
    Vec3 position;           // World position
    Vec3 tangentIn;          // Incoming tangent handle (relative to position)
    Vec3 tangentOut;         // Outgoing tangent handle (relative to position)
    
    // Handle modes
    enum class HandleMode {
        Free,       // Tangents are independent
        Aligned,    // Tangents are opposite but different lengths
        Mirrored    // Tangents are exactly mirrored
    };
    HandleMode handleMode = HandleMode::Mirrored;
    bool autoTangent = true; // Auto-calculate smooth tangents
    
    // User data (can store width, radius, etc. depending on use case)
    float userData1 = 1.0f;  // e.g., width for rivers, thickness for hair
    float userData2 = 0.0f;  // e.g., depth, taper, etc.
    float userData3 = 0.0f;  // Reserved
    Vec3 userColor = Vec3(1.0f); // e.g., color variation
    
    BezierControlPoint() = default;
    BezierControlPoint(const Vec3& pos) : position(pos) {}
    BezierControlPoint(const Vec3& pos, float data1) : position(pos), userData1(data1) {}
    
    // Set tangent out and auto-mirror tangent in (if mode requires)
    void setTangentOut(const Vec3& t) {
        tangentOut = t;
        if (handleMode == HandleMode::Mirrored) {
            tangentIn = t * -1.0f;
        } else if (handleMode == HandleMode::Aligned) {
            float inLen = tangentIn.length();
            if (t.length() > 0.001f) {
                tangentIn = t.normalize() * (-inLen);
            }
        }
    }
    
    // Set tangent in and auto-mirror tangent out (if mode requires)
    void setTangentIn(const Vec3& t) {
        tangentIn = t;
        if (handleMode == HandleMode::Mirrored) {
            tangentOut = t * -1.0f;
        } else if (handleMode == HandleMode::Aligned) {
            float outLen = tangentOut.length();
            if (t.length() > 0.001f) {
                tangentOut = t.normalize() * (-outLen);
            }
        }
    }
};

// ═══════════════════════════════════════════════════════════════════════════════
// BEZIER CURVE MATH
// ═══════════════════════════════════════════════════════════════════════════════
namespace BezierMath {

    // ─────────────────────────────────────────────────────────────────────────
    // Quadratic Bezier (3 control points)
    // ─────────────────────────────────────────────────────────────────────────
    inline Vec3 evaluateQuadratic(const Vec3& p0, const Vec3& p1, const Vec3& p2, float t) {
        float u = 1.0f - t;
        return p0 * (u * u) + p1 * (2.0f * u * t) + p2 * (t * t);
    }
    
    inline Vec3 evaluateQuadraticTangent(const Vec3& p0, const Vec3& p1, const Vec3& p2, float t) {
        float u = 1.0f - t;
        return (p1 - p0) * (2.0f * u) + (p2 - p1) * (2.0f * t);
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Cubic Bezier (4 control points)
    // ─────────────────────────────────────────────────────────────────────────
    inline Vec3 evaluateCubic(const Vec3& p0, const Vec3& p1, const Vec3& p2, const Vec3& p3, float t) {
        float u = 1.0f - t;
        float uu = u * u;
        float uuu = uu * u;
        float tt = t * t;
        float ttt = tt * t;
        
        return p0 * uuu + 
               p1 * (3.0f * uu * t) + 
               p2 * (3.0f * u * tt) + 
               p3 * ttt;
    }
    
    inline Vec3 evaluateCubicTangent(const Vec3& p0, const Vec3& p1, const Vec3& p2, const Vec3& p3, float t) {
        float u = 1.0f - t;
        float uu = u * u;
        float tt = t * t;
        
        return (p1 - p0) * (3.0f * uu) + 
               (p2 - p1) * (6.0f * u * t) + 
               (p3 - p2) * (3.0f * tt);
    }
    
    inline Vec3 evaluateCubicNormal(const Vec3& p0, const Vec3& p1, const Vec3& p2, const Vec3& p3, float t, const Vec3& up = Vec3(0, 1, 0)) {
        Vec3 tangent = evaluateCubicTangent(p0, p1, p2, p3, t).normalize();
        Vec3 right = tangent.cross(up);
        if (right.length() < 0.001f) {
            right = tangent.cross(Vec3(1, 0, 0));
        }
        return right.cross(tangent).normalize();
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Second derivative (for curvature)
    // ─────────────────────────────────────────────────────────────────────────
    inline Vec3 evaluateCubicSecondDerivative(const Vec3& p0, const Vec3& p1, const Vec3& p2, const Vec3& p3, float t) {
        float u = 1.0f - t;
        return (p2 - p1 * 2.0f + p0) * (6.0f * u) + 
               (p3 - p2 * 2.0f + p1) * (6.0f * t);
    }
    
    // Curvature at point t
    inline float curvature(const Vec3& p0, const Vec3& p1, const Vec3& p2, const Vec3& p3, float t) {
        Vec3 d1 = evaluateCubicTangent(p0, p1, p2, p3, t);
        Vec3 d2 = evaluateCubicSecondDerivative(p0, p1, p2, p3, t);
        Vec3 cross = d1.cross(d2);
        float d1Len = d1.length();
        if (d1Len < 0.0001f) return 0.0f;
        return cross.length() / (d1Len * d1Len * d1Len);
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Arc Length Approximation
    // ─────────────────────────────────────────────────────────────────────────
    inline float approximateArcLength(const Vec3& p0, const Vec3& p1, const Vec3& p2, const Vec3& p3, int samples = 16) {
        float length = 0.0f;
        Vec3 prev = p0;
        for (int i = 1; i <= samples; ++i) {
            float t = (float)i / (float)samples;
            Vec3 curr = evaluateCubic(p0, p1, p2, p3, t);
            length += (curr - prev).length();
            prev = curr;
        }
        return length;
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // De Casteljau Subdivision (splits curve at t)
    // ─────────────────────────────────────────────────────────────────────────
    inline void subdivideCubic(
        const Vec3& p0, const Vec3& p1, const Vec3& p2, const Vec3& p3, float t,
        Vec3& left0, Vec3& left1, Vec3& left2, Vec3& left3,
        Vec3& right0, Vec3& right1, Vec3& right2, Vec3& right3)
    {
        Vec3 q0 = p0 + (p1 - p0) * t;
        Vec3 q1 = p1 + (p2 - p1) * t;
        Vec3 q2 = p2 + (p3 - p2) * t;
        
        Vec3 r0 = q0 + (q1 - q0) * t;
        Vec3 r1 = q1 + (q2 - q1) * t;
        
        Vec3 s0 = r0 + (r1 - r0) * t;
        
        left0 = p0; left1 = q0; left2 = r0; left3 = s0;
        right0 = s0; right1 = r1; right2 = q2; right3 = p3;
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Closest Point on Curve (approximate)
    // ─────────────────────────────────────────────────────────────────────────
    inline float closestPointOnCurve(
        const Vec3& p0, const Vec3& p1, const Vec3& p2, const Vec3& p3,
        const Vec3& point, int samples = 32)
    {
        float bestT = 0.0f;
        float bestDist = 1e10f;
        
        for (int i = 0; i <= samples; ++i) {
            float t = (float)i / (float)samples;
            Vec3 curvePoint = evaluateCubic(p0, p1, p2, p3, t);
            float dist = (curvePoint - point).length_squared();
            if (dist < bestDist) {
                bestDist = dist;
                bestT = t;
            }
        }
        
        // Refine with binary search
        float step = 1.0f / (2.0f * samples);
        for (int i = 0; i < 4; ++i) {
            float tMinus = std::max(0.0f, bestT - step);
            float tPlus = std::min(1.0f, bestT + step);
            
            Vec3 pMinus = evaluateCubic(p0, p1, p2, p3, tMinus);
            Vec3 pPlus = evaluateCubic(p0, p1, p2, p3, tPlus);
            
            float dMinus = (pMinus - point).length_squared();
            float dPlus = (pPlus - point).length_squared();
            
            if (dMinus < bestDist) { bestDist = dMinus; bestT = tMinus; }
            if (dPlus < bestDist) { bestDist = dPlus; bestT = tPlus; }
            
            step *= 0.5f;
        }
        
        return bestT;
    }
    
} // namespace BezierMath

// ═══════════════════════════════════════════════════════════════════════════════
// BEZIER SPLINE (Multiple connected curves)
// ═══════════════════════════════════════════════════════════════════════════════
class BezierSpline {
public:
    std::vector<BezierControlPoint> points;
    bool isClosed = false;  // Connect last point back to first
    
    // ─────────────────────────────────────────────────────────────────────────
    // Control Point Management
    // ─────────────────────────────────────────────────────────────────────────
    void addPoint(const Vec3& position, float userData1 = 1.0f) {
        points.emplace_back(position, userData1);
        if (points.size() > 1) {
            calculateAutoTangents();
        }
    }
    
    void insertPoint(int index, const Vec3& position, float userData1 = 1.0f) {
        if (index < 0) index = 0;
        if (index > (int)points.size()) index = (int)points.size();
        points.insert(points.begin() + index, BezierControlPoint(position, userData1));
        calculateAutoTangents();
    }
    
    void removePoint(int index) {
        if (index >= 0 && index < (int)points.size()) {
            points.erase(points.begin() + index);
            calculateAutoTangents();
        }
    }
    
    void clear() {
        points.clear();
    }
    
    size_t pointCount() const { return points.size(); }
    size_t segmentCount() const { 
        if (points.size() < 2) return 0;
        return isClosed ? points.size() : points.size() - 1;
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Auto-calculate smooth tangents (Catmull-Rom style)
    // ─────────────────────────────────────────────────────────────────────────
    void calculateAutoTangents() {
        size_t n = points.size();
        if (n < 2) return;
        
        for (size_t i = 0; i < n; ++i) {
            if (!points[i].autoTangent) continue;
            
            size_t prevIdx = (i > 0) ? i - 1 : (isClosed ? n - 1 : i);
            size_t nextIdx = (i < n - 1) ? i + 1 : (isClosed ? 0 : i);
            
            Vec3 prev = points[prevIdx].position;
            Vec3 curr = points[i].position;
            Vec3 next = points[nextIdx].position;
            
            // Catmull-Rom tangent
            Vec3 tangent = (next - prev) * 0.5f;
            
            // Scale by distance to neighbors for smooth transitions
            float distPrev = (i > 0 || isClosed) ? (curr - prev).length() : 0.0f;
            float distNext = (i < n - 1 || isClosed) ? (next - curr).length() : 0.0f;
            float avgDist = (distPrev + distNext) * 0.5f * 0.33f;
            if (avgDist < 0.001f) avgDist = 0.1f;
            
            if (tangent.length() > 0.001f) {
                tangent = tangent.normalize() * avgDist;
            }
            
            points[i].tangentOut = tangent;
            points[i].tangentIn = tangent * -1.0f;
        }
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Evaluation along entire spline [0, 1]
    // ─────────────────────────────────────────────────────────────────────────
    Vec3 samplePosition(float t) const {
        if (points.size() < 2) return points.empty() ? Vec3(0) : points[0].position;
        
        t = std::clamp(t, 0.0f, 1.0f);
        size_t segments = segmentCount();
        float scaled = t * segments;
        size_t seg = (size_t)std::floor(scaled);
        if (seg >= segments) seg = segments - 1;
        float localT = scaled - seg;
        
        size_t i0 = seg;
        size_t i1 = (seg + 1) % points.size();
        
        const auto& p0 = points[i0];
        const auto& p1 = points[i1];
        
        return BezierMath::evaluateCubic(
            p0.position,
            p0.position + p0.tangentOut,
            p1.position + p1.tangentIn,
            p1.position,
            localT
        );
    }
    
    Vec3 sampleTangent(float t) const {
        if (points.size() < 2) return Vec3(1, 0, 0);
        
        t = std::clamp(t, 0.0f, 1.0f);
        size_t segments = segmentCount();
        float scaled = t * segments;
        size_t seg = (size_t)std::floor(scaled);
        if (seg >= segments) seg = segments - 1;
        float localT = scaled - seg;
        
        size_t i0 = seg;
        size_t i1 = (seg + 1) % points.size();
        
        const auto& p0 = points[i0];
        const auto& p1 = points[i1];
        
        return BezierMath::evaluateCubicTangent(
            p0.position,
            p0.position + p0.tangentOut,
            p1.position + p1.tangentIn,
            p1.position,
            localT
        ).normalize();
    }
    
    Vec3 sampleNormal(float t, const Vec3& up = Vec3(0, 1, 0)) const {
        Vec3 tangent = sampleTangent(t);
        Vec3 right = tangent.cross(up);
        if (right.length() < 0.001f) right = tangent.cross(Vec3(1, 0, 0));
        return right.cross(tangent).normalize();
    }
    
    Vec3 sampleRight(float t, const Vec3& up = Vec3(0, 1, 0)) const {
        Vec3 tangent = sampleTangent(t);
        Vec3 right = tangent.cross(up);
        if (right.length() < 0.001f) right = tangent.cross(Vec3(1, 0, 0));
        return right.normalize();
    }
    
    // Sample user data (linearly interpolated)
    float sampleUserData1(float t) const {
        if (points.size() < 2) return points.empty() ? 1.0f : points[0].userData1;
        
        t = std::clamp(t, 0.0f, 1.0f);
        size_t segments = segmentCount();
        float scaled = t * segments;
        size_t seg = (size_t)std::floor(scaled);
        if (seg >= segments) seg = segments - 1;
        float localT = scaled - seg;
        
        size_t i0 = seg;
        size_t i1 = (seg + 1) % points.size();
        
        return points[i0].userData1 * (1.0f - localT) + points[i1].userData1 * localT;
    }
    
    float sampleUserData2(float t) const {
        if (points.size() < 2) return points.empty() ? 0.0f : points[0].userData2;
        
        t = std::clamp(t, 0.0f, 1.0f);
        size_t segments = segmentCount();
        float scaled = t * segments;
        size_t seg = (size_t)std::floor(scaled);
        if (seg >= segments) seg = segments - 1;
        float localT = scaled - seg;
        
        size_t i0 = seg;
        size_t i1 = (seg + 1) % points.size();
        
        return points[i0].userData2 * (1.0f - localT) + points[i1].userData2 * localT;
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Arc Length
    // ─────────────────────────────────────────────────────────────────────────
    float totalArcLength(int samplesPerSegment = 16) const {
        if (points.size() < 2) return 0.0f;
        
        float total = 0.0f;
        size_t segments = segmentCount();
        
        for (size_t seg = 0; seg < segments; ++seg) {
            size_t i0 = seg;
            size_t i1 = (seg + 1) % points.size();
            
            total += BezierMath::approximateArcLength(
                points[i0].position,
                points[i0].position + points[i0].tangentOut,
                points[i1].position + points[i1].tangentIn,
                points[i1].position,
                samplesPerSegment
            );
        }
        
        return total;
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Generate uniform samples along the spline
    // ─────────────────────────────────────────────────────────────────────────
    std::vector<Vec3> generateUniformSamples(int totalSamples) const {
        std::vector<Vec3> samples;
        samples.reserve(totalSamples);
        
        for (int i = 0; i < totalSamples; ++i) {
            float t = (float)i / (float)(totalSamples - 1);
            samples.push_back(samplePosition(t));
        }
        
        return samples;
    }
    
    // Generate samples with user data
    struct SplineSample {
        Vec3 position;
        Vec3 tangent;
        Vec3 normal;
        Vec3 right;
        float t;
        float userData1;
        float userData2;
    };
    
    std::vector<SplineSample> generateDetailedSamples(int totalSamples, const Vec3& up = Vec3(0, 1, 0)) const {
        std::vector<SplineSample> samples;
        samples.reserve(totalSamples);
        
        for (int i = 0; i < totalSamples; ++i) {
            float t = (float)i / (float)(totalSamples - 1);
            SplineSample s;
            s.t = t;
            s.position = samplePosition(t);
            s.tangent = sampleTangent(t);
            s.right = sampleRight(t, up);
            s.normal = s.right.cross(s.tangent).normalize();
            s.userData1 = sampleUserData1(t);
            s.userData2 = sampleUserData2(t);
            samples.push_back(s);
        }
        
        return samples;
    }
};
