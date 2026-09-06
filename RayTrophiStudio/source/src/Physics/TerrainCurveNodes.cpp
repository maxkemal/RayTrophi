#include "TerrainCurveNodes.h"

#include "MeshEdit/SplineEvaluationService.h"
#include "NodeSystem/NodeRegistry.h"

#include <algorithm>
#include <cstdio>

namespace TerrainNodesV2 {

TerrainCurveInputNode::TerrainCurveInputNode() {
    name = "Curve Input";
    terrainNodeType = NodeType::CurveInput;
    outputs.push_back(NodeSystem::Pin::createOutput("Curve", NodeSystem::DataType::Curve));
    metadata.displayName = "Curve Input";
    metadata.category = "Input";
    metadata.description = "Reads an immutable scene spline snapshot for terrain evaluation";
    metadata.headerColor = IM_COL32(64, 205, 190, 255);
    headerColor = ImVec4(0.25f, 0.80f, 0.74f, 1.0f);
}

NodeSystem::PinValue TerrainCurveInputNode::compute(
    int, NodeSystem::EvaluationContext& ctx) {
    TerrainContext* terrainContext = getTerrainContext(ctx);
    if (!terrainContext) {
        ctx.addError(id, "Curve Input: no TerrainContext");
        return {};
    }
    if (!splineObject[0]) {
        ctx.addError(id, "Curve Input: no spline selected");
        return {};
    }
    const auto found = terrainContext->curveSnapshots.find(splineObject);
    if (found == terrainContext->curveSnapshots.end() || !found->second) {
        ctx.addError(id, std::string("Curve Input: '") + splineObject + "' was not found");
        return {};
    }
    std::string validationError;
    if (!MeshEdit::SplineEvaluationService::validate(found->second->spline, &validationError)) {
        ctx.addError(id, "Curve Input: " + validationError);
        return {};
    }
    return found->second;
}

void TerrainCurveInputNode::drawContent() {
    ImGui::SetNextItemWidth(160.0f);
    const char* preview = splineObject[0] ? splineObject : "<select spline>";
    if (g_terrainSplineListProvider && ImGui::BeginCombo("Spline", preview)) {
        const auto names = g_terrainSplineListProvider();
        for (const auto& value : names) {
            const bool selected = value == splineObject;
            if (ImGui::Selectable(value.c_str(), selected)) {
                std::snprintf(splineObject, sizeof(splineObject), "%s", value.c_str());
                dirty = true;
            }
            if (selected) ImGui::SetItemDefaultFocus();
        }
        if (names.empty()) ImGui::TextDisabled("(no spline objects)");
        ImGui::EndCombo();
    } else if (!g_terrainSplineListProvider &&
               ImGui::InputText("Spline", splineObject, sizeof(splineObject))) {
        dirty = true;
    }
}

void TerrainCurveInputNode::serializeToJson(nlohmann::json& j) const {
    TerrainNodeBase::serializeToJson(j);
    j["splineObject"] = std::string(splineObject);
}

void TerrainCurveInputNode::deserializeFromJson(const nlohmann::json& j) {
    TerrainNodeBase::deserializeFromJson(j);
    const std::string value = j.value("splineObject", std::string());
    std::snprintf(splineObject, sizeof(splineObject), "%s", value.c_str());
}

TerrainCurveToMaskNode::TerrainCurveToMaskNode() {
    name = "Curve to Mask";
    terrainNodeType = NodeType::CurveToMask;
    inputs.push_back(NodeSystem::Pin::createInput("Curve", NodeSystem::DataType::Curve));
    outputs.push_back(NodeSystem::Pin::createOutput(
        "Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
    outputs.back().imageUnit = NodeSystem::ImageUnit::Unitless;
    metadata.displayName = "Curve to Mask";
    metadata.category = "Mask";
    metadata.description = "Rasterizes an authored curve as a metric stroke or closed footprint";
    metadata.headerColor = IM_COL32(154, 89, 190, 255);
    headerColor = ImVec4(0.60f, 0.35f, 0.74f, 1.0f);
}

NodeSystem::PinValue TerrainCurveToMaskNode::compute(
    int, NodeSystem::EvaluationContext& ctx) {
    NodeSystem::CurveValue curve;
    if (!NodeSystem::tryGetCurve(getInputValue(0, ctx), curve) || !curve) {
        ctx.addError(id, "Curve to Mask: Curve input is required");
        return {};
    }
    TerrainContext* terrainContext = getTerrainContext(ctx);
    if (!terrainContext) {
        ctx.addError(id, "Curve to Mask: no TerrainContext");
        return {};
    }
    NodeSystem::Image2DData result;
    std::string rasterError;
    if (!rasterizeCurveMask(*curve, terrainContext->terrainWorldToLocal,
                            terrainContext->width, terrainContext->height,
                            terrainContext->scale_xz, settings, result,
                            &rasterError)) {
        ctx.addError(id, "Curve to Mask: " + rasterError);
        return {};
    }
    return result;
}

void TerrainCurveToMaskNode::drawContent() {
    const char* modes[] = {"Stroke", "Closed Fill"};
    int mode = static_cast<int>(settings.mode);
    if (ImGui::Combo("Mode", &mode, modes, 2)) {
        settings.mode = static_cast<CurveMaskMode>(mode);
        dirty = true;
    }
    if (settings.mode == CurveMaskMode::Stroke) {
        if (ImGui::DragFloat("Width", &settings.widthMeters, 0.1f, 0.0f, 100000.0f, "%.2f m")) dirty = true;
        if (ImGui::Checkbox("Point Width", &settings.usePointWidth)) dirty = true;
    }
    if (ImGui::DragFloat("Falloff", &settings.falloffMeters, 0.1f, 0.0f, 100000.0f, "%.2f m")) dirty = true;
    if (ImGui::Checkbox("Invert", &settings.invert)) dirty = true;
}

void TerrainCurveToMaskNode::serializeToJson(nlohmann::json& j) const {
    TerrainNodeBase::serializeToJson(j);
    j["mode"] = static_cast<int>(settings.mode);
    j["widthMeters"] = settings.widthMeters;
    j["falloffMeters"] = settings.falloffMeters;
    j["usePointWidth"] = settings.usePointWidth;
    j["invert"] = settings.invert;
}

void TerrainCurveToMaskNode::deserializeFromJson(const nlohmann::json& j) {
    TerrainNodeBase::deserializeFromJson(j);
    const int mode = j.value("mode", static_cast<int>(settings.mode));
    settings.mode = mode == static_cast<int>(CurveMaskMode::ClosedFill)
        ? CurveMaskMode::ClosedFill : CurveMaskMode::Stroke;
    settings.widthMeters = (std::max)(0.0f, j.value("widthMeters", settings.widthMeters));
    settings.falloffMeters = (std::max)(0.0f, j.value("falloffMeters", settings.falloffMeters));
    settings.usePointWidth = j.value("usePointWidth", settings.usePointWidth);
    settings.invert = j.value("invert", settings.invert);
}

namespace {
NodeSystem::AutoRegisterNode<TerrainCurveInputNode>
    regCurveInput("TerrainV2.CurveInput");
NodeSystem::AutoRegisterNode<TerrainCurveToMaskNode>
    regCurveToMask("TerrainV2.CurveToMask");
} // namespace

} // namespace TerrainNodesV2
