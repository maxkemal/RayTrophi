#include "TerrainRoadNetworkNode.h"

#include "NodeSystem/NodeRegistry.h"

#include <algorithm>
#include <cmath>
#include <functional>

namespace TerrainNodesV2 {
namespace {

void combineHash(uint64_t& seed, uint64_t value) {
    seed ^= value + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2);
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

TerrainRoadNetworkNode::TerrainRoadNetworkNode() {
    name = "Road Network";
    terrainNodeType = NodeType::RoadNetwork;
    inputs.push_back(NodeSystem::Pin::createInput(
        "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
    // Optional, and appended so serialized graphs keep mapping pins by index.
    // Water is the ONLY thing that tells the solver where a crossing is: with
    // this pin unconnected, Auto behaves exactly as Terrain and a Ford reports
    // that it could not be resolved instead of quietly becoming a normal road.
    inputs.push_back(NodeSystem::Pin::createInput(
        "Water", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
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
    // Appended after Snapshot Revision on purpose: serialized graphs map pins by
    // index, so a new output goes at the END or every saved wire shifts by one.
    outputs.push_back(makeImageOutput("Ditch", NodeSystem::ImageSemantic::Mask,
                                      NodeSystem::ImageUnit::Unitless));
    metadata.displayName = "Road Network";
    metadata.category = "Landform";
    metadata.description =
        "Carves every curve with a road assignment in one grade-limited solve";
    metadata.headerColor = IM_COL32(153, 112, 72, 255);
    headerColor = ImVec4(0.60f, 0.44f, 0.28f, 1.0f);
}

NodeSystem::PinValue TerrainRoadNetworkNode::compute(
    int outputIndex, NodeSystem::EvaluationContext& ctx) {
    NodeSystem::Image2DData height;
    if (!NodeSystem::tryGetImage(getInputValue(0, ctx), height) ||
        height.semantic != NodeSystem::ImageSemantic::Height) {
        ctx.addError(id, "Road Network: Height input is required");
        return {};
    }
    TerrainContext* terrain = getTerrainContext(ctx);
    if (!terrain) {
        ctx.addError(id, "Road Network: no TerrainContext");
        return {};
    }
    NodeSystem::Image2DData water;
    const bool haveWater = NodeSystem::tryGetImage(getInputValue(1, ctx), water) &&
                           water.isValid();
    if (haveWater && (water.width != height.width || water.height != height.height)) {
        ctx.addError(id, "Road Network: Water must match the Height grid");
        return {};
    }

    const RoadNetworkRegistry& registry = RoadNetworkRegistry::getInstance();
    const auto& assignments = registry.assignments();

    std::vector<RoadCarveInput> roads;
    std::vector<std::string> missing;
    roads.reserve(assignments.size());
    for (const RoadAssignment& assignment : assignments) {
        if (!assignment.enabled) continue;
        // The profile is re-checked here rather than trusted from the registry.
        // A saved project can name a profile this build does not have, and
        // substituting a default would carve a plausible road of the wrong class.
        RoadProfile profile;
        if (!assignment.hasOverride && !findRoadProfile(assignment.profileId, profile)) {
            ctx.addError(id, "Road Network: '" + assignment.splineObject +
                             "' uses unknown profile '" + assignment.profileId + "'");
            continue;
        }
        const auto found = terrain->curveSnapshots.find(assignment.splineObject);
        if (found == terrain->curveSnapshots.end() || !found->second) {
            missing.push_back(assignment.splineObject);
            continue;
        }
        RoadCarveInput road;
        road.curve = found->second.get();
        road.settings = assignment.effectiveCarve();
        road.crossing = assignment.crossingMode;
        road.label = assignment.splineObject;
        roads.push_back(road);
    }

    lastMissingCurves_.clear();
    for (size_t i = 0; i < missing.size(); ++i) {
        if (i) lastMissingCurves_ += ", ";
        lastMissingCurves_ += missing[i];
    }
    if (!missing.empty()) {
        // Reported, not skipped. A road assignment whose curve was deleted stops
        // grading, and without this the terrain simply comes back different with
        // nothing to explain why.
        ctx.addError(id, "Road Network: no curve found for " + lastMissingCurves_);
    }

    if (roads.empty()) {
        lastStatus_ = assignments.empty()
            ? "No road assignments in the scene"
            : "No assigned curve could be resolved";
        // Passing the terrain through unchanged is the honest answer: there are
        // no roads, so there is nothing to carve. The status line says so.
        if (outputIndex == 0) return height;
        ctx.addError(id, "Road Network: " + lastStatus_);
        return {};
    }

    uint64_t key = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(height.data.get()));
    combineHash(key, static_cast<uint64_t>(height.width));
    combineHash(key, static_cast<uint64_t>(height.height));
    // The registry generation covers profile, crossing mode, enabled state and
    // overrides in one number, so a settings change re-solves without diffing.
    combineHash(key, registry.generation());
    combineHash(key, haveWater
        ? static_cast<uint64_t>(reinterpret_cast<uintptr_t>(water.data.get())) : 0ull);
    const auto hashFloat = [&key](float value) {
        combineHash(key, static_cast<uint64_t>(std::hash<float>{}(value)));
    };
    hashFloat(terrain->scale_xz);
    hashFloat(terrain->scale_y);
    for (const RoadCarveInput& road : roads) {
        combineHash(key, static_cast<uint64_t>(road.curve->source_signature));
    }
    for (int row = 0; row < 4; ++row)
        for (int column = 0; column < 4; ++column)
            hashFloat(terrain->terrainWorldToLocal.m[row][column]);

    if (!cachedResult_ || cachedKey_ != key) {
        auto solved = std::make_shared<RoadCarveResult>();
        std::string solveError;
        if (!solveRoadNetworkCarve(height, terrain->scale_xz, terrain->scale_y, roads,
                                   terrain->terrainWorldToLocal,
                                   haveWater ? &water : nullptr, nextRevision_++,
                                   *solved, &solveError)) {
            ctx.addError(id, "Road Network: " + solveError);
            return {};
        }
        cachedKey_ = key;
        cachedResult_ = std::move(solved);
    }

    lastStatus_.clear();
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
            ctx.addError(id, "Road Network: invalid output index");
            return {};
    }
}

void TerrainRoadNetworkNode::drawContent() {
    const auto& assignments = RoadNetworkRegistry::getInstance().assignments();
    int enabled = 0;
    for (const auto& assignment : assignments) if (assignment.enabled) ++enabled;
    ImGui::TextDisabled("%d assignment(s), %d enabled",
                        static_cast<int>(assignments.size()), enabled);
    if (assignments.empty()) {
        ImGui::TextDisabled("Assign a profile to a spline");
        ImGui::TextDisabled("(terrain.road.assign_profile)");
    }
    if (!lastMissingCurves_.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.45f, 1.0f), "Missing curve: %s",
                           lastMissingCurves_.c_str());
    }
    if (!lastStatus_.empty()) ImGui::TextDisabled("%s", lastStatus_.c_str());
    if (cachedResult_) {
        ImGui::TextDisabled("Solved %d road(s), revision %llu",
                            cachedResult_->roadCount,
                            static_cast<unsigned long long>(cachedResult_->revision));
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
            ImGui::TextColored(ImVec4(1.0f, 0.78f, 0.30f, 1.0f),
                               "%d sample(s) steeper than Max Grade (terrain wins)",
                               cachedResult_->gradeExceededSamples);
        }
        if (cachedResult_->bridgeSamples > 0 || cachedResult_->tunnelSamples > 0 ||
            cachedResult_->fordSamples > 0) {
            ImGui::TextDisabled("Crossings: %d bridged, %d bored, %d forded",
                                cachedResult_->bridgeSamples,
                                cachedResult_->tunnelSamples,
                                cachedResult_->fordSamples);
        }
        if (!cachedResult_->crossingDiagnostic.empty()) {
            // A declared crossing that could not be resolved is reported, never
            // downgraded in silence: a ford that quietly became a normal road
            // looks entirely plausible in the viewport.
            ImGui::TextColored(ImVec4(1.0f, 0.78f, 0.30f, 1.0f), "%s",
                               cachedResult_->crossingDiagnostic.c_str());
        }
    }
}

NodeSystem::AutoRegisterNode<TerrainRoadNetworkNode>
    reg_TerrainRoadNetwork("TerrainV2.RoadNetwork");

} // namespace TerrainNodesV2
