#include "TerrainNodesV2.h"
#include "TerrainSatMapNodes.h"

#include <string>

namespace TerrainNodesV2 {

bool TerrainNodeGraphV2::addSatMapSetup(const std::string& preset, float x, float y) {
    HeightOutputNode* heightOutput = nullptr;
    SurfaceComposerNode* composer = nullptr;
    TerrainAnalysisNode* analysis = nullptr;
    HydraulicErosionNode* hydraulic = nullptr;
    FlowMaskNode* flowMask = nullptr;
    SoilDepthNode* soilDepth = nullptr;
    BiomeComposerNode* biome = nullptr;
    SnowClimateNode* snow = nullptr;
    TerrainSatMapColorRampNode* colorRamp = nullptr;
    TerrainSatMapOutputNode* colorOutput = nullptr;

    for (const auto& node : nodes) {
        if (!heightOutput) heightOutput = dynamic_cast<HeightOutputNode*>(node.get());
        if (!composer) composer = dynamic_cast<SurfaceComposerNode*>(node.get());
        if (!analysis) analysis = dynamic_cast<TerrainAnalysisNode*>(node.get());
        if (!hydraulic) hydraulic = dynamic_cast<HydraulicErosionNode*>(node.get());
        if (!flowMask) flowMask = dynamic_cast<FlowMaskNode*>(node.get());
        if (!soilDepth) soilDepth = dynamic_cast<SoilDepthNode*>(node.get());
        if (!biome) biome = dynamic_cast<BiomeComposerNode*>(node.get());
        if (!snow) snow = dynamic_cast<SnowClimateNode*>(node.get());
        if (!colorRamp) colorRamp = dynamic_cast<TerrainSatMapColorRampNode*>(node.get());
        if (!colorOutput) colorOutput = dynamic_cast<TerrainSatMapOutputNode*>(node.get());
    }
    if (!heightOutput || heightOutput->inputs.empty()) return false;

    const auto sourceForInput = [this](uint32_t inputPin) -> uint32_t {
        for (const auto& link : links)
            if (link.endPinId == inputPin) return link.startPinId;
        return 0;
    };
    const auto ensureLink = [this, &sourceForInput](uint32_t outputPin, uint32_t inputPin) {
        if (outputPin != 0 && inputPin != 0 && sourceForInput(inputPin) == 0)
            addLink(outputPin, inputPin);
    };

    uint32_t primaryHeight = sourceForInput(heightOutput->inputs[0].id);
    // Surface Composer deliberately receives pre-snow ground. Prefer that
    // source so snow geometry does not skew SatMap height normalization.
    if (composer && !composer->inputs.empty()) {
        const uint32_t groundHeight = sourceForInput(composer->inputs[0].id);
        if (groundHeight != 0) primaryHeight = groundHeight;
    }
    if (primaryHeight == 0) return false;

    if (!colorRamp) colorRamp = dynamic_cast<TerrainSatMapColorRampNode*>(
        addTerrainNode(NodeType::SatMapColorRamp, x, y));
    if (!colorOutput) colorOutput = dynamic_cast<TerrainSatMapOutputNode*>(
        addTerrainNode(NodeType::SatMapOutput, x + 270.0f, y));
    if (!colorRamp || !colorOutput || colorRamp->inputs.size() < 9 ||
        colorRamp->outputs.empty() || colorOutput->inputs.empty()) return false;

    colorRamp->applyPreset(preset.empty() ? "Temperate" : preset);
    ensureLink(primaryHeight, colorRamp->inputs[0].id);
    if (analysis && !analysis->outputs.empty())
        ensureLink(analysis->outputs[0].id, colorRamp->inputs[1].id);

    uint32_t flowSource = 0;
    if (composer && composer->inputs.size() >= 3) flowSource = sourceForInput(composer->inputs[2].id);
    if (flowSource == 0 && hydraulic && hydraulic->outputs.size() >= 4) flowSource = hydraulic->outputs[3].id;
    if (flowSource == 0 && flowMask && !flowMask->outputs.empty()) flowSource = flowMask->outputs[0].id;
    ensureLink(flowSource, colorRamp->inputs[2].id);

    uint32_t soilSource = composer && composer->inputs.size() >= 2
        ? sourceForInput(composer->inputs[1].id) : 0;
    if (soilSource == 0 && soilDepth && !soilDepth->outputs.empty())
        soilSource = soilDepth->outputs[0].id;
    ensureLink(soilSource, colorRamp->inputs[3].id);

    if (composer && composer->inputs.size() >= 8) {
        ensureLink(sourceForInput(composer->inputs[5].id), colorRamp->inputs[4].id);
        ensureLink(sourceForInput(composer->inputs[6].id), colorRamp->inputs[5].id);
        ensureLink(sourceForInput(composer->inputs[7].id), colorRamp->inputs[6].id);
    }
    if (snow && snow->outputs.size() >= 5) {
        ensureLink(snow->outputs[1].id, colorRamp->inputs[4].id);
        ensureLink(snow->outputs[2].id, colorRamp->inputs[5].id);
        ensureLink(snow->outputs[3].id, colorRamp->inputs[6].id);
        ensureLink(snow->outputs[4].id, colorRamp->inputs[7].id);
    }

    uint32_t grassSource = composer && composer->inputs.size() >= 9
        ? sourceForInput(composer->inputs[8].id) : 0;
    if (grassSource == 0 && biome && biome->outputs.size() >= 2)
        grassSource = biome->outputs[1].id;
    ensureLink(grassSource, colorRamp->inputs[8].id);
    ensureLink(colorRamp->outputs[0].id, colorOutput->inputs[0].id);

    markAllDirty();
    return true;
}

} // namespace TerrainNodesV2
