#include "TerrainRoadCarveNode.h"

#include "NodeSystem/NodeRegistry.h"

#include <algorithm>
#include <cmath>
#include <functional>

namespace TerrainNodesV2 {
namespace {

void combineHash(uint64_t& seed, uint64_t value) {
    seed ^= value + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2);
}

uint64_t solveKey(const NodeSystem::Image2DData& height,
                  const MeshEdit::CurveNodeData& curve,
                  const RoadCarveSettings& settings,
                  const TerrainContext& terrain) {
    uint64_t key = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(height.data.get()));
    combineHash(key, static_cast<uint64_t>(height.width));
    combineHash(key, static_cast<uint64_t>(height.height));
    combineHash(key, static_cast<uint64_t>(curve.source_signature));
    const auto hashFloat = [&key](float value) {
        combineHash(key, static_cast<uint64_t>(std::hash<float>{}(value)));
    };
    hashFloat(terrain.scale_xz);
    hashFloat(terrain.scale_y);
    hashFloat(settings.roadWidthMeters);
    hashFloat(settings.shoulderWidthMeters);
    hashFloat(settings.gradingFalloffMeters);
    hashFloat(settings.foliageExclusionMarginMeters);
    hashFloat(settings.maxGradePercent);
    hashFloat(settings.elevationOffsetMeters);
    hashFloat(settings.maxCutMeters);
    hashFloat(settings.maxFillMeters);
    // Every dial that changes the solve belongs in the key. A dial that is NOT
    // hashed here does nothing when it is turned, and the symptom is not an
    // error - it is a fix that appears not to work.
    hashFloat(settings.crownMeters);
    hashFloat(settings.ditchWidthMeters);
    hashFloat(settings.ditchDepthMeters);
    combineHash(key, settings.usePointWidth ? 1ull : 0ull);
    for (int row = 0; row < 4; ++row)
        for (int column = 0; column < 4; ++column)
            hashFloat(terrain.terrainWorldToLocal.m[row][column]);
    return key;
}

NodeSystem::Pin makeImageOutput(const char* name,
                                NodeSystem::ImageSemantic semantic,
                                NodeSystem::ImageUnit unit) {
    NodeSystem::Pin pin = NodeSystem::Pin::createOutput(
        name, NodeSystem::DataType::Image2D, semantic);
    pin.imageUnit = unit;
    return pin;
}

} // namespace

TerrainRoadCarveNode::TerrainRoadCarveNode() {
    name = "Road Carve";
    terrainNodeType = NodeType::RoadCarve;
    inputs.push_back(NodeSystem::Pin::createInput(
        "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
    inputs.push_back(NodeSystem::Pin::createInput("Curve", NodeSystem::DataType::Curve));
    outputs.push_back(makeImageOutput("Height", NodeSystem::ImageSemantic::Height,
                                      NodeSystem::ImageUnit::Unknown));
    outputs.push_back(makeImageOutput("Road Core", NodeSystem::ImageSemantic::Mask,
                                      NodeSystem::ImageUnit::Unitless));
    outputs.push_back(makeImageOutput("Shoulder", NodeSystem::ImageSemantic::Mask,
                                      NodeSystem::ImageUnit::Unitless));
    outputs.push_back(makeImageOutput("Cut", NodeSystem::ImageSemantic::PhysicalScalar,
                                      NodeSystem::ImageUnit::Meters));
    outputs.push_back(makeImageOutput("Fill", NodeSystem::ImageSemantic::PhysicalScalar,
                                      NodeSystem::ImageUnit::Meters));
    outputs.push_back(makeImageOutput("Foliage Exclusion", NodeSystem::ImageSemantic::Mask,
                                      NodeSystem::ImageUnit::Unitless));
    outputs.push_back(NodeSystem::Pin::createOutput(
        "Snapshot Revision", NodeSystem::DataType::Int));
    // Appended after Snapshot Revision: serialized graphs map pins by index, so
    // a new output goes at the END or every saved wire shifts by one.
    outputs.push_back(makeImageOutput("Ditch", NodeSystem::ImageSemantic::Mask,
                                      NodeSystem::ImageUnit::Unitless));
    metadata.displayName = "Road Carve";
    metadata.category = "Landform";
    metadata.description = "Grade-limited road terrain and one-revision infrastructure fields";
    metadata.headerColor = IM_COL32(153, 112, 72, 255);
    headerColor = ImVec4(0.60f, 0.44f, 0.28f, 1.0f);
}

NodeSystem::PinValue TerrainRoadCarveNode::compute(
    int outputIndex, NodeSystem::EvaluationContext& ctx) {
    NodeSystem::Image2DData height;
    if (!NodeSystem::tryGetImage(getInputValue(0, ctx), height) ||
        height.semantic != NodeSystem::ImageSemantic::Height) {
        ctx.addError(id, "Road Carve: Height input is required");
        return {};
    }
    NodeSystem::CurveValue curve;
    if (!NodeSystem::tryGetCurve(getInputValue(1, ctx), curve) || !curve) {
        ctx.addError(id, "Road Carve: Curve input is required");
        return {};
    }
    TerrainContext* terrain = getTerrainContext(ctx);
    if (!terrain) {
        ctx.addError(id, "Road Carve: no TerrainContext");
        return {};
    }

    const uint64_t key = solveKey(height, *curve, settings, *terrain);
    if (!cachedResult_ || cachedKey_ != key) {
        auto solved = std::make_shared<RoadCarveResult>();
        std::string solveError;
        if (!solveRoadCarve(height, terrain->scale_xz, terrain->scale_y,
                            *curve, terrain->terrainWorldToLocal, settings,
                            nextRevision_++, *solved, &solveError)) {
            ctx.addError(id, "Road Carve: " + solveError);
            return {};
        }
        cachedKey_ = key;
        cachedResult_ = std::move(solved);
    }

    switch (outputIndex) {
        case 0: return cachedResult_->height;
        case 1: return cachedResult_->roadCore;
        case 2: return cachedResult_->shoulder;
        case 3: return cachedResult_->cut;
        case 4: return cachedResult_->fill;
        case 5: return cachedResult_->foliageExclusion;
        case 6: return static_cast<int>(cachedResult_->revision);
        case 7: return cachedResult_->ditch;
        default:
            ctx.addError(id, "Road Carve: invalid output index");
            return {};
    }
}

void TerrainRoadCarveNode::drawContent() {
    if (ImGui::DragFloat("Road Width", &settings.roadWidthMeters, 0.1f, 0.1f, 1000.0f, "%.2f m")) dirty = true;
    if (ImGui::DragFloat("Shoulder", &settings.shoulderWidthMeters, 0.1f, 0.0f, 1000.0f, "%.2f m")) dirty = true;
    if (ImGui::DragFloat("Grade Falloff", &settings.gradingFalloffMeters, 0.1f, 0.0f, 1000.0f, "%.2f m")) dirty = true;
    if (ImGui::DragFloat("Foliage Margin", &settings.foliageExclusionMarginMeters, 0.1f, 0.0f, 1000.0f, "%.2f m")) dirty = true;
    if (ImGui::DragFloat("Max Grade", &settings.maxGradePercent, 0.1f, 0.0f, 100.0f, "%.1f %%")) dirty = true;
    if (ImGui::DragFloat("Elevation Offset", &settings.elevationOffsetMeters, 0.05f, -1000.0f, 1000.0f, "%.2f m")) dirty = true;
    if (ImGui::DragFloat("Max Cut", &settings.maxCutMeters, 0.1f, 0.0f, 1000.0f, "%.2f m")) dirty = true;
    if (ImGui::DragFloat("Max Fill", &settings.maxFillMeters, 0.1f, 0.0f, 1000.0f, "%.2f m")) dirty = true;
    if (ImGui::DragFloat("Crown", &settings.crownMeters, 0.01f, 0.0f, 1.0f, "%.3f m")) dirty = true;
    if (ImGui::DragFloat("Ditch Width", &settings.ditchWidthMeters, 0.05f, 0.0f, 20.0f, "%.2f m")) dirty = true;
    if (ImGui::DragFloat("Ditch Depth", &settings.ditchDepthMeters, 0.05f, 0.0f, 10.0f, "%.2f m")) dirty = true;
    if (ImGui::Checkbox("Point Width", &settings.usePointWidth)) dirty = true;
    if (cachedResult_) {
        ImGui::TextDisabled("Snapshot revision: %llu",
            static_cast<unsigned long long>(cachedResult_->revision));
        // What the solve actually did to the terrain. A road that quietly built a
        // mountain-high embankment is the failure this node shipped with; the
        // numbers make the same event impossible to miss.
        ImGui::TextDisabled("Peak cut %.2f m / fill %.2f m",
                            cachedResult_->peakCutMeters, cachedResult_->peakFillMeters);
        if (cachedResult_->routeSampleCount > 0 &&
            cachedResult_->envelopeClampedSamples > 0) {
            const float ratio = 100.0f *
                static_cast<float>(cachedResult_->envelopeClampedSamples) /
                static_cast<float>(cachedResult_->routeSampleCount);
            ImGui::TextColored(ImVec4(1.0f, 0.78f, 0.30f, 1.0f),
                               "Cut/fill limit binding on %.0f%% of the route", ratio);
        }
        if (cachedResult_->gradeExceededSamples > 0) {
            // Not a warning about the dial - a statement about the terrain: the
            // route crosses ground too steep to hold this grade inside the
            // cut/fill budget, so the road follows the ground instead.
            ImGui::TextColored(ImVec4(1.0f, 0.78f, 0.30f, 1.0f),
                               "%d sample(s) steeper than Max Grade (terrain wins)",
                               cachedResult_->gradeExceededSamples);
        }
    }
}

void TerrainRoadCarveNode::serializeToJson(nlohmann::json& j) const {
    TerrainNodeBase::serializeToJson(j);
    j["roadWidthMeters"] = settings.roadWidthMeters;
    j["shoulderWidthMeters"] = settings.shoulderWidthMeters;
    j["gradingFalloffMeters"] = settings.gradingFalloffMeters;
    j["foliageExclusionMarginMeters"] = settings.foliageExclusionMarginMeters;
    j["maxGradePercent"] = settings.maxGradePercent;
    j["maxCutMeters"] = settings.maxCutMeters;
    j["maxFillMeters"] = settings.maxFillMeters;
    j["elevationOffsetMeters"] = settings.elevationOffsetMeters;
    j["crownMeters"] = settings.crownMeters;
    j["ditchWidthMeters"] = settings.ditchWidthMeters;
    j["ditchDepthMeters"] = settings.ditchDepthMeters;
    j["usePointWidth"] = settings.usePointWidth;
}

void TerrainRoadCarveNode::deserializeFromJson(const nlohmann::json& j) {
    TerrainNodeBase::deserializeFromJson(j);
    settings.roadWidthMeters = (std::max)(0.1f, j.value("roadWidthMeters", settings.roadWidthMeters));
    settings.shoulderWidthMeters = (std::max)(0.0f, j.value("shoulderWidthMeters", settings.shoulderWidthMeters));
    settings.gradingFalloffMeters = (std::max)(0.0f, j.value("gradingFalloffMeters", settings.gradingFalloffMeters));
    settings.foliageExclusionMarginMeters = (std::max)(0.0f, j.value("foliageExclusionMarginMeters", settings.foliageExclusionMarginMeters));
    settings.maxGradePercent = (std::max)(0.0f, (std::min)(100.0f,
        j.value("maxGradePercent", settings.maxGradePercent)));
    settings.maxCutMeters = (std::max)(0.0f, j.value("maxCutMeters", settings.maxCutMeters));
    settings.maxFillMeters = (std::max)(0.0f, j.value("maxFillMeters", settings.maxFillMeters));
    settings.elevationOffsetMeters = j.value("elevationOffsetMeters", settings.elevationOffsetMeters);
    settings.crownMeters = (std::max)(0.0f, j.value("crownMeters", settings.crownMeters));
    settings.ditchWidthMeters = (std::max)(0.0f, j.value("ditchWidthMeters", settings.ditchWidthMeters));
    settings.ditchDepthMeters = (std::max)(0.0f, j.value("ditchDepthMeters", settings.ditchDepthMeters));
    settings.usePointWidth = j.value("usePointWidth", settings.usePointWidth);
    cachedKey_ = 0;
    cachedResult_.reset();
}

namespace {
NodeSystem::AutoRegisterNode<TerrainRoadCarveNode>
    regRoadCarve("TerrainV2.RoadCarve");
} // namespace

} // namespace TerrainNodesV2
