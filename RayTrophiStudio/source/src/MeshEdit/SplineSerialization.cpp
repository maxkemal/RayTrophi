#include "MeshEdit/SplineSerialization.h"

#include <algorithm>
#include <cmath>

namespace MeshEdit {
namespace {

nlohmann::json vec3Json(const Vec3& value) {
    return {value.x, value.y, value.z};
}

bool readVec3(const nlohmann::json& value, Vec3& out) {
    if (!value.is_array() || value.size() != 3) return false;
    for (size_t i = 0; i < 3; ++i) if (!value[i].is_number()) return false;
    out = Vec3(value[0].get<float>(), value[1].get<float>(), value[2].get<float>());
    return true;
}

nlohmann::json matrixJson(const Matrix4x4& matrix) {
    nlohmann::json rows = nlohmann::json::array();
    for (int r = 0; r < 4; ++r) {
        rows.push_back({matrix.m[r][0], matrix.m[r][1], matrix.m[r][2], matrix.m[r][3]});
    }
    return rows;
}

bool readMatrix(const nlohmann::json& value, Matrix4x4& out) {
    if (!value.is_array() || value.size() != 4) return false;
    for (int r = 0; r < 4; ++r) {
        if (!value[r].is_array() || value[r].size() != 4) return false;
        for (int c = 0; c < 4; ++c) {
            if (!value[r][c].is_number()) return false;
            out.m[r][c] = value[r][c].get<float>();
        }
    }
    return true;
}

} // namespace

const char* splineCurveTypeName(SplineCurveType type) {
    switch (type) {
    case SplineCurveType::Linear: return "linear";
    case SplineCurveType::Bezier: return "bezier";
    case SplineCurveType::BSpline: return "bspline";
    }
    return "bezier";
}

bool parseSplineCurveType(const std::string& value, SplineCurveType& out) {
    if (value == "linear") { out = SplineCurveType::Linear; return true; }
    if (value == "bezier") { out = SplineCurveType::Bezier; return true; }
    if (value == "bspline" || value == "b-spline") { out = SplineCurveType::BSpline; return true; }
    return false;
}

nlohmann::json serializeSpline(const SplineObject& object) {
    nlohmann::json points = nlohmann::json::array();
    for (const auto& point : object.spline.points) {
        points.push_back({
            {"position", vec3Json(point.position)},
            {"tangent_in", vec3Json(point.tangentIn)},
            {"tangent_out", vec3Json(point.tangentOut)},
            {"handle_mode", static_cast<int>(point.handleMode)},
            {"auto_tangent", point.autoTangent},
            {"user_data", {point.userData1, point.userData2, point.userData3}},
            {"user_color", vec3Json(point.userColor)}
        });
    }
    const auto& skin = object.skin_display;
    return {
        {"schema", "rt.spline.v1"},
        {"name", object.nodeName},
        {"plane", static_cast<int>(object.plane)},
        {"curve_type", splineCurveTypeName(object.spline.curveType)},
        {"closed", object.spline.isClosed},
        {"knots", object.spline.knots},
        {"transform", matrixJson(object.transform ? object.transform->base : Matrix4x4::identity())},
        {"pivot_offset", vec3Json(object.transform ? object.transform->pivot_offset : Vec3(0.0f))},
        {"points", std::move(points)},
        {"skin_display", {
            {"enabled", skin.enabled}, {"host", skin.host_name},
            {"custom_profile", skin.custom_profile}, {"radius", skin.radius},
            {"path_samples", skin.path_samples}, {"radial_segments", skin.radial_segments},
            {"cap_start", skin.cap_start}, {"cap_end", skin.cap_end},
            {"use_point_radius", skin.use_point_radius},
            {"taper_start", skin.taper_start}, {"taper_end", skin.taper_end},
            {"taper_falloff", skin.taper_falloff},
            {"twist_start_degrees", skin.twist_start_degrees},
            {"twist_end_degrees", skin.twist_end_degrees},
            {"wave_amplitude", skin.wave_amplitude}, {"wave_cycles", skin.wave_cycles},
            {"wave_phase_degrees", skin.wave_phase_degrees},
            {"wave_noise", skin.wave_noise}, {"wave_seed", skin.wave_seed},
            {"wave_axis", skin.wave_axis}
        }}
    };
}

bool deserializeSpline(const nlohmann::json& payload, SplineObject& object,
                       std::string& error) {
    if (!payload.is_object()) { error = "spline payload must be an object"; return false; }
    const std::string curveName = payload.value("curve_type", "bezier");
    SplineCurveType curveType;
    if (!parseSplineCurveType(curveName, curveType)) {
        error = "unknown spline curve_type: " + curveName;
        return false;
    }
    if (!payload.contains("points") || !payload["points"].is_array()) {
        error = "spline payload requires a points array";
        return false;
    }
    std::vector<BezierControlPoint> points;
    points.reserve(payload["points"].size());
    for (const auto& item : payload["points"]) {
        if (!item.is_object()) { error = "spline point must be an object"; return false; }
        BezierControlPoint point;
        if (!readVec3(item.value("position", nlohmann::json()), point.position)) {
            error = "spline point position must be a three-component array"; return false;
        }
        readVec3(item.value("tangent_in", nlohmann::json({0.0f, 0.0f, 0.0f})), point.tangentIn);
        readVec3(item.value("tangent_out", nlohmann::json({0.0f, 0.0f, 0.0f})), point.tangentOut);
        point.handleMode = static_cast<BezierControlPoint::HandleMode>(
            std::clamp(item.value("handle_mode", 2), 0, 2));
        point.autoTangent = item.value("auto_tangent", true);
        if (item.contains("user_data") && item["user_data"].is_array() && item["user_data"].size() == 3) {
            point.userData1 = item["user_data"][0].get<float>();
            point.userData2 = item["user_data"][1].get<float>();
            point.userData3 = item["user_data"][2].get<float>();
        }
        readVec3(item.value("user_color", nlohmann::json({1.0f, 1.0f, 1.0f})), point.userColor);
        points.push_back(point);
    }
    if (curveType == SplineCurveType::BSpline && points.size() < 4) {
        error = "B-Spline requires at least four control points";
        return false;
    }
    std::vector<float> knots;
    if (payload.contains("knots")) {
        if (!payload["knots"].is_array()) {
            error = "spline knots must be a numeric array";
            return false;
        }
        knots.reserve(payload["knots"].size());
        for (const auto& knot : payload["knots"]) {
            if (!knot.is_number()) {
                error = "spline knots must be a numeric array";
                return false;
            }
            knots.push_back(knot.get<float>());
        }
        if (!knots.empty() && knots.size() != points.size() + 4) {
            error = "cubic B-Spline knots must contain point_count + 4 values";
            return false;
        }
        for (size_t i = 1; i < knots.size(); ++i) {
            if (!std::isfinite(knots[i]) || knots[i] < knots[i - 1]) {
                error = "spline knots must be finite and nondecreasing";
                return false;
            }
        }
        if (!knots.empty() && !std::isfinite(knots.front())) {
            error = "spline knots must be finite and nondecreasing";
            return false;
        }
    }
    Matrix4x4 transform = Matrix4x4::identity();
    if (payload.contains("transform") && !readMatrix(payload["transform"], transform)) {
        error = "spline transform must be a 4x4 numeric matrix";
        return false;
    }
    Vec3 pivotOffset(0.0f);
    if (payload.contains("pivot_offset") && !readVec3(payload["pivot_offset"], pivotOffset)) {
        error = "spline pivot_offset must be a three-component array";
        return false;
    }
    object.nodeName = payload.value("name", object.nodeName);
    object.plane = static_cast<SplinePlane>(std::clamp(payload.value("plane", 1), 0, 2));
    object.spline.curveType = curveType;
    object.spline.isClosed = payload.value("closed", false);
    object.spline.points = std::move(points);
    object.spline.knots = curveType == SplineCurveType::BSpline
        ? std::move(knots) : std::vector<float>{};
    object.transform = std::make_shared<Transform>(transform);
    // Preserve the serialized render matrix while restoring the independent
    // authoring pivot used by the viewport gizmo and profile generators.
    object.transform->setPivotOffset(pivotOffset, true);
    object.skin_display = {};
    if (payload.contains("skin_display") && payload["skin_display"].is_object()) {
        const auto& skin = payload["skin_display"];
        object.skin_display.enabled = skin.value("enabled", false);
        object.skin_display.host_name = skin.value("host", std::string());
        object.skin_display.custom_profile = skin.value("custom_profile", std::string());
        object.skin_display.radius = std::max(0.0001f, skin.value("radius", 0.1f));
        object.skin_display.path_samples = std::clamp(skin.value("path_samples", 48), 2, 1024);
        object.skin_display.radial_segments = std::clamp(skin.value("radial_segments", 12), 3, 256);
        object.skin_display.cap_start = skin.value("cap_start", true);
        object.skin_display.cap_end = skin.value("cap_end", true);
        object.skin_display.use_point_radius = skin.value("use_point_radius", true);
        object.skin_display.taper_start = std::max(0.0f, skin.value("taper_start", 1.0f));
        object.skin_display.taper_end = std::max(0.0f, skin.value("taper_end", 1.0f));
        object.skin_display.taper_falloff = std::clamp(skin.value("taper_falloff", 1.0f), 0.01f, 32.0f);
        object.skin_display.twist_start_degrees = skin.value("twist_start_degrees", 0.0f);
        object.skin_display.twist_end_degrees = skin.value("twist_end_degrees", 0.0f);
        object.skin_display.wave_amplitude = skin.value("wave_amplitude", 0.0f);
        object.skin_display.wave_cycles = skin.value("wave_cycles", 1.0f);
        object.skin_display.wave_phase_degrees = skin.value("wave_phase_degrees", 0.0f);
        object.skin_display.wave_noise = std::max(0.0f, skin.value("wave_noise", 0.0f));
        object.skin_display.wave_seed = skin.value("wave_seed", 0);
        object.skin_display.wave_axis = std::clamp(skin.value("wave_axis", 1), 0, 2);
        const auto& display = object.skin_display;
        if (!std::isfinite(display.radius) ||
            !std::isfinite(display.taper_start) || !std::isfinite(display.taper_end) ||
            !std::isfinite(display.taper_falloff) ||
            !std::isfinite(display.twist_start_degrees) ||
            !std::isfinite(display.twist_end_degrees) ||
            !std::isfinite(display.wave_amplitude) || !std::isfinite(display.wave_cycles) ||
            !std::isfinite(display.wave_phase_degrees) || !std::isfinite(display.wave_noise)) {
            error = "spline skin_display contains a non-finite numeric value";
            return false;
        }
    }
    object.selected_point = -1;
    object.selected_points.clear();
    return true;
}

} // namespace MeshEdit
