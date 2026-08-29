#include "GeometryNodesV2.h"
#include "MeshEdit/GeometryCurveNodes.h"

#include "MeshEdit/CurveNodeData.h"
#include "MeshEdit/ProfileSweep.h"
#include "MeshEdit/SplineEvaluationService.h"
#include "MeshEdit/SplineSerialization.h"
#include "NodeSystem/NodeRegistry.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>

namespace GeometryNodesV2 {
namespace {

NodeSystem::CurveValue curveInput(GeometryNodeBase& node, int inputIndex,
                                  NodeSystem::EvaluationContext& ctx) {
    NodeSystem::CurveValue curve;
    NodeSystem::tryGetCurve(node.getInputValue(inputIndex, ctx), curve);
    return curve;
}

float controlRadius(const BezierSpline& spline, const MeshEdit::SplineEvaluation& sample) {
    if (spline.points.empty() || sample.segment < 0) return 1.0f;
    const size_t count = spline.points.size();
    size_t a = static_cast<size_t>(sample.segment) % count;
    size_t b = (a + 1) % count;
    if (spline.curveType == SplineCurveType::BSpline && count >= 4) {
        a = (a + 1) % count;
        b = (a + 1) % count;
    }
    const float r0 = spline.points[a].userData1;
    const float r1 = spline.points[b].userData1;
    return std::max(0.0f, r0 * (1.0f - sample.local_t) + r1 * sample.local_t);
}

float controlTwist(const BezierSpline& spline, const MeshEdit::SplineEvaluation& sample) {
    if (spline.points.empty() || sample.segment < 0) return 0.0f;
    const size_t count = spline.points.size();
    size_t a = static_cast<size_t>(sample.segment) % count;
    size_t b = (a + 1) % count;
    if (spline.curveType == SplineCurveType::BSpline && count >= 4) {
        a = (a + 1) % count;
        b = (a + 1) % count;
    }
    return spline.points[a].userData2 * (1.0f - sample.local_t) +
           spline.points[b].userData2 * sample.local_t;
}

float signedNoise(uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return static_cast<float>(value & 0x00ffffffu) / 8388607.5f - 1.0f;
}

BezierSpline worldCurve(const MeshEdit::CurveNodeData& data) {
    BezierSpline result = data.spline;
    for (auto& point : result.points) {
        point.position = data.local_to_world.transform_point(point.position);
        point.tangentIn = data.local_to_world.transform_vector(point.tangentIn);
        point.tangentOut = data.local_to_world.transform_vector(point.tangentOut);
    }
    return result;
}

BezierSpline canonicalProfile(const MeshEdit::CurveNodeData& data) {
    BezierSpline result = data.spline;
    auto remap = [&](const Vec3& value) {
        if (data.plane == MeshEdit::SplinePlane::XZ) return Vec3(value.x, value.z, 0.0f);
        if (data.plane == MeshEdit::SplinePlane::YZ) return Vec3(value.z, value.y, 0.0f);
        return Vec3(value.x, value.y, 0.0f);
    };
    Vec3 scale(1.0f);
    if (data.plane == MeshEdit::SplinePlane::XY) scale = Vec3(data.object_scale.x, data.object_scale.y, 1.0f);
    else if (data.plane == MeshEdit::SplinePlane::XZ) scale = Vec3(data.object_scale.x, data.object_scale.z, 1.0f);
    else scale = Vec3(data.object_scale.z, data.object_scale.y, 1.0f);
    for (auto& point : result.points) {
        point.position = remap(point.position - data.pivot_offset) * scale;
        point.tangentIn = remap(point.tangentIn) * scale;
        point.tangentOut = remap(point.tangentOut) * scale;
    }
    return result;
}

BezierSpline circleProfile(int segments) {
    BezierSpline profile;
    profile.curveType = SplineCurveType::Linear;
    profile.isClosed = true;
    const int count = std::clamp(segments, 3, 256);
    constexpr float twoPi = 6.28318530717958647692f;
    for (int i = 0; i < count; ++i) {
        const float angle = twoPi * static_cast<float>(i) / static_cast<float>(count);
        profile.addPoint(Vec3(std::cos(angle), std::sin(angle), 0.0f));
    }
    return profile;
}

std::vector<float> uniformParameters(const BezierSpline& spline, int count) {
    std::vector<float> result(static_cast<size_t>(count), 0.0f);
    if (count <= 1) return result;
    const int denseCount = std::max(64, count * 8);
    std::vector<float> cumulative(static_cast<size_t>(denseCount + 1), 0.0f);
    Vec3 previous = MeshEdit::SplineEvaluationService::evaluate(spline, 0.0f).position;
    for (int i = 1; i <= denseCount; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(denseCount);
        const auto sample = MeshEdit::SplineEvaluationService::evaluate(spline, t);
        cumulative[static_cast<size_t>(i)] = cumulative[static_cast<size_t>(i - 1)] +
            (sample.position - previous).length();
        previous = sample.position;
    }
    const float total = cumulative.back();
    if (total <= 1.0e-8f) {
        for (int i = 0; i < count; ++i) result[static_cast<size_t>(i)] =
            static_cast<float>(i) / static_cast<float>(count - 1);
        return result;
    }
    for (int i = 0; i < count; ++i) {
        const float target = total * static_cast<float>(i) / static_cast<float>(count - 1);
        const auto upper = std::lower_bound(cumulative.begin(), cumulative.end(), target);
        const size_t hi = static_cast<size_t>(std::distance(cumulative.begin(), upper));
        if (hi == 0) { result[static_cast<size_t>(i)] = 0.0f; continue; }
        if (hi >= cumulative.size()) { result[static_cast<size_t>(i)] = 1.0f; continue; }
        const float loDistance = cumulative[hi - 1];
        const float span = std::max(1.0e-8f, cumulative[hi] - loDistance);
        const float fraction = (target - loDistance) / span;
        result[static_cast<size_t>(i)] =
            (static_cast<float>(hi - 1) + fraction) / static_cast<float>(denseCount);
    }
    return result;
}

} // namespace

SplineObjectNode::SplineObjectNode() {
    name = "Spline Object";
    geometryNodeType = NodeType::SplineObject;
    outputs.push_back(NodeSystem::Pin::createOutput("Curve", NodeSystem::DataType::Curve));
    metadata.displayName = "Spline Object";
    metadata.category = "Input";
    metadata.description = "Live editable spline snapshot. Re-evaluates from the scene source.";
    metadata.headerColor = IM_COL32(64, 205, 190, 255);
    headerColor = ImVec4(0.25f, 0.80f, 0.74f, 1.0f);
}

std::string SplineObjectNode::getTypeId() const { return "GeoV2.SplineObject"; }
void SplineObjectNode::serializeParams(nlohmann::json& j) const { j["object"] = std::string(objectName); }
void SplineObjectNode::deserializeParams(const nlohmann::json& j) {
    const std::string value = j.value("object", std::string());
    std::snprintf(objectName, sizeof(objectName), "%s", value.c_str());
}
void SplineObjectNode::drawContent() {
    ImGui::SetNextItemWidth(170.0f);
    const char* preview = objectName[0] ? objectName : "<select spline>";
    if (g_splineObjectListProvider && ImGui::BeginCombo("Spline", preview)) {
        const auto names = g_splineObjectListProvider();
        for (const auto& value : names) {
            const bool selected = value == objectName;
            if (ImGui::Selectable(value.c_str(), selected))
                std::snprintf(objectName, sizeof(objectName), "%s", value.c_str());
            if (selected) ImGui::SetItemDefaultFocus();
        }
        if (names.empty()) ImGui::TextDisabled("(no spline objects)");
        ImGui::EndCombo();
    } else if (!g_splineObjectListProvider) {
        ImGui::InputText("Spline", objectName, sizeof(objectName));
    }
}
NodeSystem::PinValue SplineObjectNode::compute(int, NodeSystem::EvaluationContext& ctx) {
    auto* gctx = getGeometryContext(ctx);
    if (!gctx || !gctx->resolveSpline) {
        ctx.addError(id, "Spline Object: no spline resolver in GeometryContext");
        return {};
    }
    if (!objectName[0]) {
        ctx.addError(id, "Spline Object: no spline selected");
        return {};
    }
    auto curve = gctx->resolveSpline(objectName);
    if (!curve) {
        ctx.addError(id, std::string("Spline Object: '") + objectName + "' was not found");
        return {};
    }
    std::string error;
    if (!MeshEdit::SplineEvaluationService::validate(curve->spline, &error)) {
        ctx.addError(id, "Spline Object: " + error);
        return {};
    }
    return NodeSystem::CurveValue(curve);
}

ResampleCurveNode::ResampleCurveNode() {
    name = "Resample Curve";
    geometryNodeType = NodeType::ResampleCurve;
    inputs.push_back(NodeSystem::Pin::createInput("Curve", NodeSystem::DataType::Curve));
    outputs.push_back(NodeSystem::Pin::createOutput("Curve", NodeSystem::DataType::Curve));
    metadata.displayName = "Resample Curve";
    metadata.category = "Curve";
    metadata.description = "Samples a curve to evenly spaced linear controls while preserving point radius.";
    metadata.headerColor = IM_COL32(64, 205, 190, 255);
    headerColor = ImVec4(0.25f, 0.80f, 0.74f, 1.0f);
}
std::string ResampleCurveNode::getTypeId() const { return "GeoV2.ResampleCurve"; }
void ResampleCurveNode::serializeParams(nlohmann::json& j) const {
    j["samples"] = samples; j["uniform_distance"] = uniformDistance;
}
void ResampleCurveNode::deserializeParams(const nlohmann::json& j) {
    samples = std::clamp(j.value("samples", 32), 2, 1024);
    uniformDistance = j.value("uniform_distance", true);
}
void ResampleCurveNode::drawContent() {
    ImGui::SliderInt("Samples", &samples, 2, 256);
    ImGui::Checkbox("Uniform Distance", &uniformDistance);
}
NodeSystem::PinValue ResampleCurveNode::compute(int, NodeSystem::EvaluationContext& ctx) {
    auto input = curveInput(*this, 0, ctx);
    if (!input) { ctx.addError(id, "Resample Curve: no curve input"); return {}; }
    const int count = std::clamp(samples, 2, 1024);
    const int evaluationCount = input->spline.isClosed ? count + 1 : count;
    const auto parameters = uniformDistance
        ? uniformParameters(input->spline, evaluationCount) : std::vector<float>{};
    auto output = std::make_shared<MeshEdit::CurveNodeData>(*input);
    output->spline.points.clear();
    output->spline.knots.clear();
    output->spline.curveType = SplineCurveType::Linear;
    for (int i = 0; i < evaluationCount; ++i) {
        const float t = uniformDistance ? parameters[static_cast<size_t>(i)]
            : static_cast<float>(i) / static_cast<float>(evaluationCount - 1);
        const auto sample = MeshEdit::SplineEvaluationService::evaluate(input->spline, t);
        if (!sample.valid) { ctx.addError(id, "Resample Curve: spline evaluation failed"); return {}; }
        BezierControlPoint point(sample.position);
        point.userData1 = controlRadius(input->spline, sample);
        point.userData2 = controlTwist(input->spline, sample);
        output->spline.points.push_back(point);
    }
    if (input->spline.isClosed && output->spline.points.size() > 2)
        output->spline.points.pop_back();
    output->spline.isClosed = input->spline.isClosed;
    return NodeSystem::CurveValue(output);
}

CurveTaperNode::CurveTaperNode() {
    name = "Curve Taper";
    geometryNodeType = NodeType::CurveTaper;
    inputs.push_back(NodeSystem::Pin::createInput("Curve", NodeSystem::DataType::Curve));
    outputs.push_back(NodeSystem::Pin::createOutput("Curve", NodeSystem::DataType::Curve));
    metadata.displayName = "Curve Taper";
    metadata.category = "Curve";
    metadata.description = "Scales the curve radius from start to end without changing topology.";
    metadata.headerColor = IM_COL32(64, 205, 190, 255);
    headerColor = ImVec4(0.25f, 0.80f, 0.74f, 1.0f);
}
std::string CurveTaperNode::getTypeId() const { return "GeoV2.CurveTaper"; }
void CurveTaperNode::serializeParams(nlohmann::json& j) const {
    j["start_scale"] = startScale; j["end_scale"] = endScale; j["exponent"] = exponent;
}
void CurveTaperNode::deserializeParams(const nlohmann::json& j) {
    startScale = std::max(0.0f, j.value("start_scale", 1.0f));
    endScale = std::max(0.0f, j.value("end_scale", 0.0f));
    exponent = std::clamp(j.value("exponent", 1.0f), 0.01f, 32.0f);
}
void CurveTaperNode::drawContent() {
    ImGui::DragFloat("Start Scale", &startScale, 0.01f, 0.0f, 1000.0f, "%.3f");
    ImGui::DragFloat("End Scale", &endScale, 0.01f, 0.0f, 1000.0f, "%.3f");
    ImGui::DragFloat("Falloff", &exponent, 0.01f, 0.01f, 32.0f, "%.3f");
}
NodeSystem::PinValue CurveTaperNode::compute(int, NodeSystem::EvaluationContext& ctx) {
    auto input = curveInput(*this, 0, ctx);
    if (!input) { ctx.addError(id, "Curve Taper: no curve input"); return {}; }
    auto output = std::make_shared<MeshEdit::CurveNodeData>(*input);
    const size_t count = output->spline.points.size();
    for (size_t i = 0; i < count; ++i) {
        const float u = count > 1 ? static_cast<float>(i) / static_cast<float>(count - 1) : 0.0f;
        const float shaped = std::pow(std::clamp(u, 0.0f, 1.0f), std::clamp(exponent, 0.01f, 32.0f));
        const float scale = std::max(0.0f, startScale * (1.0f - shaped) + endScale * shaped);
        output->spline.points[i].userData1 *= scale;
    }
    return NodeSystem::CurveValue(output);
}

CurveTwistNode::CurveTwistNode() {
    name = "Curve Twist";
    geometryNodeType = NodeType::CurveTwist;
    inputs.push_back(NodeSystem::Pin::createInput("Curve", NodeSystem::DataType::Curve));
    outputs.push_back(NodeSystem::Pin::createOutput("Curve", NodeSystem::DataType::Curve));
    metadata.displayName = "Curve Twist";
    metadata.category = "Curve";
    metadata.description = "Rotates the swept profile along the curve while preserving topology.";
    metadata.headerColor = IM_COL32(64, 205, 190, 255);
    headerColor = ImVec4(0.25f, 0.80f, 0.74f, 1.0f);
}
std::string CurveTwistNode::getTypeId() const { return "GeoV2.CurveTwist"; }
void CurveTwistNode::serializeParams(nlohmann::json& j) const {
    j["start_degrees"] = startDegrees; j["end_degrees"] = endDegrees;
}
void CurveTwistNode::deserializeParams(const nlohmann::json& j) {
    startDegrees = j.value("start_degrees", 0.0f);
    endDegrees = j.value("end_degrees", 360.0f);
}
void CurveTwistNode::drawContent() {
    ImGui::DragFloat("Start", &startDegrees, 1.0f, -100000.0f, 100000.0f, "%.1f deg");
    ImGui::DragFloat("End", &endDegrees, 1.0f, -100000.0f, 100000.0f, "%.1f deg");
}
NodeSystem::PinValue CurveTwistNode::compute(int, NodeSystem::EvaluationContext& ctx) {
    auto input = curveInput(*this, 0, ctx);
    if (!input) { ctx.addError(id, "Curve Twist: no curve input"); return {}; }
    auto output = std::make_shared<MeshEdit::CurveNodeData>(*input);
    constexpr float radians = 0.01745329251994329577f;
    const size_t count = output->spline.points.size();
    for (size_t i = 0; i < count; ++i) {
        const float u = count > 1 ? static_cast<float>(i) / static_cast<float>(count - 1) : 0.0f;
        output->spline.points[i].userData2 +=
            (startDegrees * (1.0f - u) + endDegrees * u) * radians;
    }
    return NodeSystem::CurveValue(output);
}

CurveWaveNode::CurveWaveNode() {
    name = "Curve Wave + Noise";
    geometryNodeType = NodeType::CurveWave;
    inputs.push_back(NodeSystem::Pin::createInput("Curve", NodeSystem::DataType::Curve));
    outputs.push_back(NodeSystem::Pin::createOutput("Curve", NodeSystem::DataType::Curve));
    metadata.displayName = "Curve Wave + Noise";
    metadata.category = "Curve";
    metadata.description = "Offsets curve controls with a deterministic wave and optional noise.";
    metadata.headerColor = IM_COL32(64, 205, 190, 255);
    headerColor = ImVec4(0.25f, 0.80f, 0.74f, 1.0f);
}
std::string CurveWaveNode::getTypeId() const { return "GeoV2.CurveWave"; }
void CurveWaveNode::serializeParams(nlohmann::json& j) const {
    j["amplitude"] = amplitude; j["frequency"] = frequency;
    j["phase_degrees"] = phaseDegrees; j["noise"] = noise;
    j["seed"] = seed; j["axis"] = axis;
}
void CurveWaveNode::deserializeParams(const nlohmann::json& j) {
    amplitude = j.value("amplitude", 0.1f);
    frequency = j.value("frequency", 1.0f);
    phaseDegrees = j.value("phase_degrees", 0.0f);
    noise = std::max(0.0f, j.value("noise", 0.0f));
    seed = j.value("seed", 0); axis = std::clamp(j.value("axis", 1), 0, 2);
}
void CurveWaveNode::drawContent() {
    static const char* axes[] = {"X", "Y", "Z"};
    ImGui::Combo("Offset Axis", &axis, axes, 3);
    ImGui::DragFloat("Amplitude", &amplitude, 0.01f, -10000.0f, 10000.0f, "%.3f");
    ImGui::DragFloat("Cycles", &frequency, 0.05f, -1000.0f, 1000.0f, "%.3f");
    ImGui::DragFloat("Phase", &phaseDegrees, 1.0f, -100000.0f, 100000.0f, "%.1f deg");
    ImGui::DragFloat("Noise", &noise, 0.01f, 0.0f, 10000.0f, "%.3f");
    ImGui::DragInt("Seed", &seed, 1.0f);
}
NodeSystem::PinValue CurveWaveNode::compute(int, NodeSystem::EvaluationContext& ctx) {
    auto input = curveInput(*this, 0, ctx);
    if (!input) { ctx.addError(id, "Curve Wave: no curve input"); return {}; }
    auto output = std::make_shared<MeshEdit::CurveNodeData>(*input);
    constexpr float twoPi = 6.28318530717958647692f;
    constexpr float radians = 0.01745329251994329577f;
    const size_t count = output->spline.points.size();
    for (size_t i = 0; i < count; ++i) {
        const float u = count > 1 ? static_cast<float>(i) / static_cast<float>(count - 1) : 0.0f;
        const float wave = std::sin(twoPi * frequency * u + phaseDegrees * radians) * amplitude;
        const float random = signedNoise(static_cast<uint32_t>(i) ^
            (static_cast<uint32_t>(seed) * 0x9e3779b9u)) * std::max(0.0f, noise);
        Vec3 offset(0.0f); offset[std::clamp(axis, 0, 2)] = wave + random;
        output->spline.points[i].position += offset;
    }
    return NodeSystem::CurveValue(output);
}

CurveToMeshNode::CurveToMeshNode() {
    name = "Curve to Mesh";
    geometryNodeType = NodeType::CurveToMesh;
    inputs.push_back(NodeSystem::Pin::createInput("Path", NodeSystem::DataType::Curve));
    auto profile = NodeSystem::Pin::createInput("Profile", NodeSystem::DataType::Curve);
    profile.optional = true;
    inputs.push_back(profile);
    outputs.push_back(NodeSystem::Pin::createOutput("Geometry", NodeSystem::DataType::Geometry));
    metadata.displayName = "Curve to Mesh";
    metadata.category = "Curve";
    metadata.description = "Builds a solid tube/cable, or sweeps an optional closed spline profile.";
    metadata.headerColor = IM_COL32(64, 205, 190, 255);
    headerColor = ImVec4(0.25f, 0.80f, 0.74f, 1.0f);
}
std::string CurveToMeshNode::getTypeId() const { return "GeoV2.CurveToMesh"; }
void CurveToMeshNode::serializeParams(nlohmann::json& j) const {
    j["path_samples"] = pathSamples; j["profile_samples"] = profileSamples;
    j["radius"] = radius; j["use_point_radius"] = usePointRadius;
    j["cap_start"] = capStart; j["cap_end"] = capEnd;
    j["material_id"] = materialId;
}
void CurveToMeshNode::deserializeParams(const nlohmann::json& j) {
    pathSamples = std::clamp(j.value("path_samples", 48), 2, 1024);
    profileSamples = std::clamp(j.value("profile_samples", 12), 3, 256);
    radius = std::max(0.0001f, j.value("radius", 0.1f));
    usePointRadius = j.value("use_point_radius", true);
    capStart = j.value("cap_start", true); capEnd = j.value("cap_end", true);
    materialId = std::clamp(j.value("material_id", -1), -1, 65535);
}
void CurveToMeshNode::drawContent() {
    ImGui::SliderInt("Path Samples", &pathSamples, 2, 256);
    ImGui::SliderInt("Profile Samples", &profileSamples, 3, 64);
    ImGui::DragFloat("Radius", &radius, 0.01f, 0.0001f, 10000.0f, "%.4f");
    ImGui::Checkbox("Use Point Radius", &usePointRadius);
    ImGui::Checkbox("Cap Start", &capStart); ImGui::SameLine(); ImGui::Checkbox("Cap End", &capEnd);
    ImGui::DragInt("Material ID", &materialId, 1.0f, -1, 65535);
    ImGui::TextDisabled("Material -1 inherits the Geometry Graph host material.");
    ImGui::TextDisabled("Profile is optional; empty uses a circular cable profile.");
}
NodeSystem::PinValue CurveToMeshNode::compute(int, NodeSystem::EvaluationContext& ctx) {
    auto pathData = curveInput(*this, 0, ctx);
    if (!pathData) { ctx.addError(id, "Curve to Mesh: no path curve"); return {}; }
    BezierSpline path = worldCurve(*pathData);
    auto* gctx = getGeometryContext(ctx);
    if (gctx && gctx->baseMesh && gctx->baseMesh->transform) {
        const Matrix4x4 worldToHost = gctx->baseMesh->transform->getFinal().inverse();
        for (auto& point : path.points) {
            point.position = worldToHost.transform_point(point.position);
            point.tangentIn = worldToHost.transform_vector(point.tangentIn);
            point.tangentOut = worldToHost.transform_vector(point.tangentOut);
        }
    }
    if (path.isClosed) { ctx.addError(id, "Curve to Mesh: path must be open"); return {}; }
    auto profileData = curveInput(*this, 1, ctx);
    BezierSpline profile = profileData ? canonicalProfile(*profileData) : circleProfile(profileSamples);
    if (!profile.isClosed) { ctx.addError(id, "Curve to Mesh: profile curve must be closed"); return {}; }
    MeshEdit::ProfileSweepSettings settings;
    settings.path_samples = std::clamp(pathSamples, 2, 1024);
    settings.profile_samples = std::clamp(profileSamples, 3, 256);
    settings.profile_scale = std::max(0.0001f, radius);
    settings.use_path_radius = usePointRadius;
    settings.cap_start = capStart; settings.cap_end = capEnd;
    auto result = MeshEdit::buildProfileSweep(profile, path, settings);
    if (!result.report.ok || !result.geometry) {
        const std::string message = result.report.diagnostics.empty()
            ? "geometry generation failed" : result.report.diagnostics.front().message;
        ctx.addError(id, "Curve to Mesh: " + message);
        return {};
    }
    uint16_t outputMaterial = 0;
    if (materialId >= 0) {
        outputMaterial = static_cast<uint16_t>(materialId);
    } else if (gctx && gctx->baseMesh && gctx->baseMesh->geometry) {
        const uint16_t* sourceMaterials =
            gctx->baseMesh->geometry->get_attribute_data<uint16_t>("materialID");
        if (sourceMaterials && gctx->baseMesh->geometry->get_vertex_count() > 0)
            outputMaterial = sourceMaterials[0];
    }
    if (uint16_t* generatedMaterials =
            result.geometry->get_attribute_data_mut<uint16_t>("materialID")) {
        std::fill(generatedMaterials,
                  generatedMaterials + result.geometry->get_vertex_count(), outputMaterial);
    }
    auto mesh = std::make_shared<TriangleMesh>();
    mesh->geometry = result.geometry;
    mesh->transform = std::make_shared<Transform>();
    if (gctx && gctx->baseMesh && gctx->baseMesh->transform)
        *mesh->transform = *gctx->baseMesh->transform;
    return NodeSystem::GeometryValue(mesh);
}

std::size_t curveGraphSourceSignature(
    const GeometryNodeGraphV2& graph,
    const std::vector<std::shared_ptr<Hittable>>& sceneObjects,
    bool* usesCurves) {
    bool foundCurveNode = false;
    std::size_t seed = 1469598103934665603ull;
    auto combine = [&](std::size_t value) {
        seed ^= value + static_cast<std::size_t>(0x9e3779b9u) + (seed << 6) + (seed >> 2);
    };
    for (const auto& node : graph.nodes) {
        const std::string type = node->getTypeId();
        if (type == "GeoV2.SplineObject" || type == "GeoV2.ResampleCurve" ||
            type == "GeoV2.CurveTaper" || type == "GeoV2.CurveTwist" ||
            type == "GeoV2.CurveWave" || type == "GeoV2.CurveToMesh") foundCurveNode = true;
        combine(std::hash<std::string>{}(type));
        if (const auto* geometryNode = dynamic_cast<const GeometryNodeBase*>(node.get())) {
            nlohmann::json params = nlohmann::json::object();
            geometryNode->serializeParams(params);
            combine(std::hash<std::string>{}(params.dump()));
        }
    }
    for (const auto& link : graph.links) {
        combine(link.startPinId); combine(link.endPinId);
    }
    if (foundCurveNode) {
        for (const auto& object : sceneObjects) {
            auto spline = std::dynamic_pointer_cast<MeshEdit::SplineObject>(object);
            if (!spline) continue;
            combine(std::hash<std::string>{}(MeshEdit::serializeSpline(*spline).dump()));
        }
    }
    if (usesCurves) *usesCurves = foundCurveNode;
    return seed;
}

namespace {
NodeSystem::AutoRegisterNode<SplineObjectNode> regSplineObject("GeoV2.SplineObject");
NodeSystem::AutoRegisterNode<ResampleCurveNode> regResampleCurve("GeoV2.ResampleCurve");
NodeSystem::AutoRegisterNode<CurveTaperNode> regCurveTaper("GeoV2.CurveTaper");
NodeSystem::AutoRegisterNode<CurveTwistNode> regCurveTwist("GeoV2.CurveTwist");
NodeSystem::AutoRegisterNode<CurveWaveNode> regCurveWave("GeoV2.CurveWave");
NodeSystem::AutoRegisterNode<CurveToMeshNode> regCurveToMesh("GeoV2.CurveToMesh");
}

} // namespace GeometryNodesV2
