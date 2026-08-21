#include "MeshEdit/SplineSerialization.h"

#include <algorithm>

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
    return {
        {"schema", "rt.spline.v1"},
        {"name", object.nodeName},
        {"plane", static_cast<int>(object.plane)},
        {"curve_type", splineCurveTypeName(object.spline.curveType)},
        {"closed", object.spline.isClosed},
        {"transform", matrixJson(object.transform ? object.transform->base : Matrix4x4::identity())},
        {"points", std::move(points)}
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
    Matrix4x4 transform = Matrix4x4::identity();
    if (payload.contains("transform") && !readMatrix(payload["transform"], transform)) {
        error = "spline transform must be a 4x4 numeric matrix";
        return false;
    }
    object.nodeName = payload.value("name", object.nodeName);
    object.plane = static_cast<SplinePlane>(std::clamp(payload.value("plane", 1), 0, 2));
    object.spline.curveType = curveType;
    object.spline.isClosed = payload.value("closed", false);
    object.spline.points = std::move(points);
    object.transform = std::make_shared<Transform>(transform);
    object.selected_point = -1;
    object.selected_points.clear();
    return true;
}

} // namespace MeshEdit
