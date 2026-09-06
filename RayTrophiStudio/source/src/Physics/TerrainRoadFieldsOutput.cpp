#include "TerrainRoadFieldsOutput.h"

#include "NodeSystem/NodeRegistry.h"

#include <array>
#include <cmath>

namespace TerrainNodesV2 {
namespace {

struct RoadFieldContract {
    const char* name;
    NodeSystem::ImageSemantic semantic;
    NodeSystem::ImageUnit unit;
    // Which input pin carries it. Ditch is pin 6, AFTER Snapshot Revision,
    // because serialized graphs map pins by index: a new pin inserted in the
    // middle would rewire every saved graph by one.
    int inputIndex;
};

constexpr std::array<RoadFieldContract, 6> kRoadFields = {{
    {"infrastructure.road_core", NodeSystem::ImageSemantic::Mask,
     NodeSystem::ImageUnit::Unitless, 0},
    {"infrastructure.shoulder", NodeSystem::ImageSemantic::Mask,
     NodeSystem::ImageUnit::Unitless, 1},
    {"infrastructure.cut", NodeSystem::ImageSemantic::PhysicalScalar,
     NodeSystem::ImageUnit::Meters, 2},
    {"infrastructure.fill", NodeSystem::ImageSemantic::PhysicalScalar,
     NodeSystem::ImageUnit::Meters, 3},
    {"infrastructure.foliage_exclusion", NodeSystem::ImageSemantic::Mask,
     NodeSystem::ImageUnit::Unitless, 4},
    // The road pushes water, the ditch carries it. Publishing the ditch is what
    // keeps the drainage network intact once road_core is excluded from channel
    // classification - excluding the road WITHOUT this is the plausible-looking
    // failure: no river on the road, and the water that should run beside it
    // goes nowhere.
    {"infrastructure.ditch", NodeSystem::ImageSemantic::Mask,
     NodeSystem::ImageUnit::Unitless, 6}
}};

NodeSystem::Pin makeFieldInput(const char* label,
                               NodeSystem::ImageSemantic semantic,
                               NodeSystem::ImageUnit unit) {
    auto pin = NodeSystem::Pin::createInput(
        label, NodeSystem::DataType::Image2D, semantic);
    pin.imageUnit = unit;
    pin.updateVisualCache();
    return pin;
}

bool validateField(const NodeSystem::Image2DData& field,
                   const RoadFieldContract& contract,
                   int width, int height,
                   std::string& error) {
    if (!field.isValid() || !field.data || field.channels != 1) {
        error = std::string(contract.name) + " is missing or is not single-channel";
        return false;
    }
    if (field.width != width || field.height != height ||
        field.data->size() != static_cast<size_t>(width) * height) {
        error = std::string(contract.name) + " does not match the terrain field grid";
        return false;
    }
    if (field.semantic != contract.semantic || field.unit != contract.unit) {
        error = std::string(contract.name) + " has an incompatible semantic or unit";
        return false;
    }
    for (float value : *field.data) {
        if (!std::isfinite(value)) {
            error = std::string(contract.name) + " contains a non-finite sample";
            return false;
        }
        if (contract.semantic == NodeSystem::ImageSemantic::Mask &&
            (value < 0.0f || value > 1.0f)) {
            error = std::string(contract.name) + " contains a mask sample outside 0..1";
            return false;
        }
    }
    return true;
}

} // namespace

TerrainRoadFieldsOutputNode::TerrainRoadFieldsOutputNode() {
    name = "Road Fields Output";
    terrainNodeType = NodeType::RoadFieldsOutput;
    inputs.push_back(makeFieldInput("Road Core", kRoadFields[0].semantic, kRoadFields[0].unit));
    inputs.push_back(makeFieldInput("Shoulder", kRoadFields[1].semantic, kRoadFields[1].unit));
    inputs.push_back(makeFieldInput("Cut", kRoadFields[2].semantic, kRoadFields[2].unit));
    inputs.push_back(makeFieldInput("Fill", kRoadFields[3].semantic, kRoadFields[3].unit));
    inputs.push_back(makeFieldInput("Foliage Exclusion", kRoadFields[4].semantic, kRoadFields[4].unit));
    inputs.push_back(NodeSystem::Pin::createInput(
        "Snapshot Revision", NodeSystem::DataType::Int));
    inputs.push_back(makeFieldInput("Ditch", kRoadFields[5].semantic, kRoadFields[5].unit));
    metadata.displayName = "Road Fields Output";
    metadata.category = "Output";
    metadata.description = "Atomically publishes one Road Carve snapshot for materials, hydrology and foliage";
    metadata.headerColor = IM_COL32(153, 112, 72, 255);
    metadata.iconType = (int)UIWidgets::IconType::Console;
    headerColor = ImVec4(0.60f, 0.44f, 0.28f, 1.0f);
}

NodeSystem::PinValue TerrainRoadFieldsOutputNode::compute(
    int outputIndex, NodeSystem::EvaluationContext& ctx) {
    (void)outputIndex;
    lastPublishedRevision = 0;
    lastPublishSucceeded = false;
    lastPublishError.clear();

    auto fail = [&](const std::string& message) -> NodeSystem::PinValue {
        lastPublishError = message;
        ctx.addError(id, "Road Fields Output: " + message);
        return {};
    };

    auto* terrainContext = getTerrainContext(ctx);
    if (!terrainContext || !terrainContext->terrain)
        return fail("terrain context is unavailable");

    // Source ownership is part of the contract. Matching dimensions alone are
    // insufficient: fields from two road solves can describe different
    // curve/height snapshots while looking structurally compatible.
    //
    // The rule is "all six pins come from ONE road solver", and it used to be
    // written as a hard-coded "TerrainV2.RoadCarve". That spelled the intent as
    // a type name, so the moment a second road solver existed
    // (TerrainV2.RoadNetwork, which carves every assigned curve in one pass) a
    // perfectly valid graph was refused with no hint that the node type was the
    // objection. Live testing found it immediately: the terrain WAS carved and
    // not one field was published.
    auto* graph = ctx.getGraph();
    if (!graph) return fail("graph context is unavailable");
    NodeSystem::NodeBase* source = nullptr;
    for (const auto& input : inputs) {
        NodeSystem::Pin* sourcePin = graph->getInputSource(input.id);
        NodeSystem::NodeBase* candidate = sourcePin
            ? graph->getPinOwner(sourcePin->id) : nullptr;
        if (!candidate)
            return fail("all six fields and Snapshot Revision must be connected");
        const std::string candidateType = candidate->getTypeId();
        if (candidateType != "TerrainV2.RoadCarve" &&
            candidateType != "TerrainV2.RoadNetwork")
            return fail("every input must come directly from one Road Carve or "
                        "Road Network node (got '" + candidateType + "')");
        if (!sourcePin || sourcePin->name != input.name)
            return fail("road solver outputs must match the corresponding field inputs");
        if (!source) source = candidate;
        else if (candidate != source)
            return fail("inputs come from different road solver snapshots");
    }

    std::array<NodeSystem::Image2DData, 6> fields;
    const int width = terrainContext->terrain->heightmap.width;
    const int height = terrainContext->terrain->heightmap.height;
    for (size_t index = 0; index < fields.size(); ++index) {
        if (!NodeSystem::tryGetImage(getInputValue(kRoadFields[index].inputIndex, ctx),
                                     fields[index]))
            return fail(std::string(kRoadFields[index].name) + " is unavailable");
        std::string error;
        if (!validateField(fields[index], kRoadFields[index], width, height, error))
            return fail(error);
    }

    int revision = 0;
    if (!NodeSystem::tryGetInt(getInputValue(5, ctx), revision) || revision <= 0)
        return fail("Snapshot Revision is invalid");

    // Commit only after the entire bundle passes validation. shared_ptr storage
    // preserves the immutable solve products without copying five field grids.
    for (size_t index = 0; index < fields.size(); ++index) {
        terrainContext->terrain->analysisFields[kRoadFields[index].name] = fields[index].data;
    }
    lastPublishedRevision = static_cast<uint64_t>(revision);
    lastPublishSucceeded = true;
    return {};
}

void TerrainRoadFieldsOutputNode::drawContent() {
    if (lastPublishSucceeded) {
        ImGui::TextColored(ImVec4(0.42f, 0.82f, 0.60f, 1.0f),
                           "Published revision %llu",
                           static_cast<unsigned long long>(lastPublishedRevision));
        ImGui::TextDisabled("6 canonical infrastructure fields");
    } else if (!lastPublishError.empty()) {
        ImGui::TextColored(ImVec4(0.95f, 0.45f, 0.35f, 1.0f),
                           "Not published: %s", lastPublishError.c_str());
    } else {
        ImGui::TextDisabled("Connect one Road Carve snapshot");
    }
}

namespace {
NodeSystem::AutoRegisterNode<TerrainRoadFieldsOutputNode>
    regRoadFieldsOutput("TerrainV2.RoadFieldsOutput");
} // namespace

} // namespace TerrainNodesV2
