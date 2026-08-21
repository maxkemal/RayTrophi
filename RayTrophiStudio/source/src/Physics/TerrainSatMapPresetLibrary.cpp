#include "TerrainSatMapPresetLibrary.h"

#include "TerrainNodesV2.h"
#include "TerrainSatMapNodes.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <set>
#include <unordered_map>

namespace TerrainNodesV2 {
namespace {

using json = nlohmann::json;
namespace fs = std::filesystem;

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

bool readStop(const json& value, SatMapPresetStop& stop) {
    if (!value.is_array() || value.size() < 4) return false;
    stop.position = std::clamp(value[0].get<float>(), 0.0f, 1.0f);
    stop.r = std::clamp(value[1].get<float>(), 0.0f, 1.0f);
    stop.g = std::clamp(value[2].get<float>(), 0.0f, 1.0f);
    stop.b = std::clamp(value[3].get<float>(), 0.0f, 1.0f);
    stop.a = value.size() > 4 ? std::clamp(value[4].get<float>(), 0.0f, 1.0f) : 1.0f;
    return true;
}

bool readRecipe(const json& root, const fs::path& path,
                SatMapPresetRecipe& recipe, std::string& error) {
    if (!root.is_object()) { error = "root must be an object"; return false; }
    recipe.id = root.value("id", "");
    recipe.label = root.value("label", recipe.id);
    recipe.category = root.value("category", "General");
    recipe.description = root.value("description", "");
    recipe.version = std::max(root.value("version", 1), 1);
    recipe.basePreset = root.value("basePreset", "Temperate");
    recipe.sourcePath = path.string();
    if (recipe.id.empty()) { error = "missing id"; return false; }
    if (!root.contains("layers") || !root["layers"].is_array()) {
        error = "layers must be an array"; return false;
    }
    std::set<std::string> layerIds;
    for (const auto& item : root["layers"]) {
        if (!item.is_object()) { error = "layer must be an object"; return false; }
        SatMapPresetLayer layer;
        layer.id = item.value("id", "");
        layer.label = item.value("label", layer.id);
        layer.primary = lower(item.value("primary", ""));
        layer.opacity = std::clamp(item.value("opacity", 1.0f), 0.0f, 1.0f);
        layer.maskPower = std::clamp(item.value("maskPower", 1.0f), 0.05f, 8.0f);
        layer.detailStrength = std::clamp(item.value("detailStrength", 0.075f), 0.0f, 0.35f);
        layer.detailScale = std::clamp(item.value("detailScale", 180.0f), 4.0f, 2048.0f);
        layer.autoNormalize = item.value("autoNormalize", false);
        if (layer.id.empty() || layer.primary.empty() || !layerIds.insert(layer.id).second) {
            error = "layer ids must be non-empty and unique"; return false;
        }
        if (!item.contains("stops") || !item["stops"].is_array()) {
            error = "layer '" + layer.id + "' has no stops"; return false;
        }
        for (const auto& stopJson : item["stops"]) {
            SatMapPresetStop stop;
            if (!readStop(stopJson, stop)) {
                error = "layer '" + layer.id + "' has an invalid stop"; return false;
            }
            layer.stops.push_back(stop);
        }
        if (layer.stops.size() < 2 || layer.stops.size() > 32) {
            error = "layer '" + layer.id + "' requires 2..32 stops"; return false;
        }
        std::sort(layer.stops.begin(), layer.stops.end(),
            [](const auto& a, const auto& b) { return a.position < b.position; });
        if (item.contains("conditions")) {
            if (!item["conditions"].is_array()) {
                error = "conditions must be an array"; return false;
            }
            for (const auto& conditionJson : item["conditions"]) {
                SatMapPresetCondition condition;
                condition.field = lower(conditionJson.value("field", ""));
                condition.minimum = conditionJson.value("min", 0.0f);
                condition.maximum = conditionJson.value("max", 1.0f);
                condition.gamma = std::clamp(conditionJson.value("gamma", 1.0f), 0.01f, 8.0f);
                condition.invert = conditionJson.value("invert", false);
                if (condition.field.empty() || !(condition.maximum > condition.minimum)) {
                    error = "layer '" + layer.id + "' has an invalid condition"; return false;
                }
                layer.conditions.push_back(condition);
            }
        }
        recipe.layers.push_back(std::move(layer));
    }
    return true;
}

std::vector<fs::path> presetDirectories() {
    return {
        fs::path("assets/terrain/satmap_presets"),
        fs::path("RayTrophiStudio/assets/terrain/satmap_presets"),
        fs::path("../../assets/terrain/satmap_presets"),
        fs::path("../assets/terrain/satmap_presets")
    };
}

constexpr const char* managedPrefix = "[SatMap Recipe] ";

bool recipeUses(const SatMapPresetRecipe& recipe, const std::string& field) {
    for (const auto& layer : recipe.layers) {
        if (layer.primary == field) return true;
        for (const auto& condition : layer.conditions)
            if (condition.field == field) return true;
    }
    return false;
}

} // namespace

SatMapPresetLibrary& SatMapPresetLibrary::instance() {
    static SatMapPresetLibrary library;
    return library;
}

bool SatMapPresetLibrary::reload(std::string* error) {
    presets_.clear();
    diagnostics_.clear();
    fs::path directory;
    std::error_code ec;
    for (const auto& candidate : presetDirectories()) {
        if (fs::is_directory(candidate, ec)) { directory = candidate; break; }
        ec.clear();
    }
    if (directory.empty()) {
        if (error) *error = "SatMap preset directory not found: assets/terrain/satmap_presets";
        return false;
    }
    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(directory, ec)) {
        if (entry.is_regular_file() && lower(entry.path().extension().string()) == ".json")
            files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end());
    std::set<std::string> ids;
    for (const auto& path : files) {
        try {
            std::ifstream stream(path);
            json root;
            stream >> root;
            SatMapPresetRecipe recipe;
            std::string parseError;
            if (!readRecipe(root, path, recipe, parseError)) {
                diagnostics_.push_back(path.string() + ": " + parseError);
                continue;
            }
            if (!ids.insert(recipe.id).second) {
                diagnostics_.push_back(path.string() + ": duplicate id '" + recipe.id + "'");
                continue;
            }
            presets_.push_back(std::move(recipe));
        } catch (const std::exception& e) {
            diagnostics_.push_back(path.string() + ": " + e.what());
        }
    }
    std::sort(presets_.begin(), presets_.end(), [](const auto& a, const auto& b) {
        return a.category == b.category ? a.label < b.label : a.category < b.category;
    });
    if (presets_.empty()) {
        if (error) *error = diagnostics_.empty() ? "no SatMap presets found" : diagnostics_.front();
        return false;
    }
    return true;
}

const SatMapPresetRecipe* SatMapPresetLibrary::find(const std::string& id) const {
    const std::string wanted = lower(id);
    for (const auto& preset : presets_)
        if (lower(preset.id) == wanted) return &preset;
    return nullptr;
}

bool TerrainNodeGraphV2::applySatMapPresetRecipe(
    const std::string& presetId, std::string* error,
    std::vector<std::string>* warnings, float x, float y) {
    auto fail = [error](const std::string& message) {
        if (error) *error = message;
        return false;
    };
    auto& library = SatMapPresetLibrary::instance();
    std::string loadError;
    if (!library.reload(&loadError)) return fail(loadError);
    const SatMapPresetRecipe* recipe = library.find(presetId);
    if (!recipe) return fail("unknown SatMap preset recipe: " + presetId);

    std::vector<uint32_t> managedIds;
    for (const auto& node : nodes)
        if (node->name.rfind(managedPrefix, 0) == 0) managedIds.push_back(node->id);
    for (uint32_t id : managedIds) removeNode(id);

    HeightOutputNode* heightOutput = nullptr;
    SurfaceComposerNode* composer = nullptr;
    TerrainAnalysisNode* analysis = nullptr;
    HydraulicErosionNode* hydraulic = nullptr;
    FlowMaskNode* flowMask = nullptr;
    SoilDepthNode* soilDepth = nullptr;
    BiomeComposerNode* biome = nullptr;
    TerrainGrassMaskNode* grassMask = nullptr;
    TerrainSurfaceMasksNode* surfaceMasks = nullptr;
    ExposureMaskNode* exposure = nullptr;
    SnowClimateNode* snow = nullptr;
    TerrainSatMapOutputNode* output = nullptr;
    for (const auto& node : nodes) {
        if (!heightOutput) heightOutput = dynamic_cast<HeightOutputNode*>(node.get());
        if (!composer) composer = dynamic_cast<SurfaceComposerNode*>(node.get());
        if (!analysis) analysis = dynamic_cast<TerrainAnalysisNode*>(node.get());
        if (!hydraulic) hydraulic = dynamic_cast<HydraulicErosionNode*>(node.get());
        if (!flowMask) flowMask = dynamic_cast<FlowMaskNode*>(node.get());
        if (!soilDepth) soilDepth = dynamic_cast<SoilDepthNode*>(node.get());
        if (!biome) biome = dynamic_cast<BiomeComposerNode*>(node.get());
        if (!grassMask) grassMask = dynamic_cast<TerrainGrassMaskNode*>(node.get());
        if (!surfaceMasks) surfaceMasks = dynamic_cast<TerrainSurfaceMasksNode*>(node.get());
        if (!exposure) exposure = dynamic_cast<ExposureMaskNode*>(node.get());
        if (!snow) snow = dynamic_cast<SnowClimateNode*>(node.get());
        if (!output) output = dynamic_cast<TerrainSatMapOutputNode*>(node.get());
    }
    if (!heightOutput || heightOutput->inputs.empty())
        return fail("SatMap recipe requires a connected Height Output");

    const auto sourceForInput = [this](uint32_t inputPin) -> uint32_t {
        for (const auto& link : links) if (link.endPinId == inputPin) return link.startPinId;
        return 0;
    };
    uint32_t heightSource = sourceForInput(heightOutput->inputs[0].id);
    if (composer && !composer->inputs.empty()) {
        const uint32_t ground = sourceForInput(composer->inputs[0].id);
        if (ground) heightSource = ground;
    }
    if (!heightSource) return fail("Height Output has no upstream height source");

    const auto connect = [this](uint32_t source, uint32_t target) {
        return source != 0 && target != 0 && addLink(source, target) != 0;
    };
    const bool hydraulicConnected = hydraulic && !hydraulic->inputs.empty() &&
        sourceForInput(hydraulic->inputs[0].id) != 0;
    uint32_t flowSource = hydraulicConnected && hydraulic->outputs.size() > 3
        ? hydraulic->outputs[3].id : (flowMask && !flowMask->outputs.empty() ? flowMask->outputs[0].id : 0);
    if (!flowSource && recipeUses(*recipe, "flow")) {
        flowMask = dynamic_cast<FlowMaskNode*>(addTerrainNode(NodeType::FlowMask, x - 620.0f, y + 240.0f));
        if (flowMask) {
            flowMask->name = std::string(managedPrefix) + "Flow";
            connect(heightSource, flowMask->inputs[0].id);
            flowSource = flowMask->outputs[0].id;
        }
    }
    const bool needsAnalysis = recipeUses(*recipe, "slope") || recipeUses(*recipe, "concavity") ||
        recipeUses(*recipe, "convexity") || recipeUses(*recipe, "valley") || recipeUses(*recipe, "wetness");
    if (!analysis && needsAnalysis) {
        analysis = dynamic_cast<TerrainAnalysisNode*>(addTerrainNode(NodeType::TerrainAnalysis, x - 620.0f, y));
        if (analysis) {
            analysis->name = std::string(managedPrefix) + "Analysis";
            connect(heightSource, analysis->inputs[0].id);
            if (flowSource) connect(flowSource, analysis->inputs[1].id);
        }
    }
    uint32_t soilSource = soilDepth && !soilDepth->outputs.empty() ? soilDepth->outputs[0].id : 0;
    if (!soilSource && composer && composer->inputs.size() > 1)
        soilSource = sourceForInput(composer->inputs[1].id);
    if (!exposure && (recipeUses(*recipe, "exposure") || recipeUses(*recipe, "moss"))) {
        exposure = dynamic_cast<ExposureMaskNode*>(addTerrainNode(NodeType::ExposureMask, x - 620.0f, y + 420.0f));
        if (exposure) {
            exposure->name = std::string(managedPrefix) + "Exposure";
            connect(heightSource, exposure->inputs[0].id);
        }
    }
    const bool needsSurface = recipeUses(*recipe, "cavity") || recipeUses(*recipe, "mud") || recipeUses(*recipe, "moss");
    if (!surfaceMasks && needsSurface) {
        surfaceMasks = dynamic_cast<TerrainSurfaceMasksNode*>(addTerrainNode(NodeType::SurfaceMasks, x - 360.0f, y + 300.0f));
        if (surfaceMasks) {
            surfaceMasks->name = std::string(managedPrefix) + "Surface Masks";
            connect(heightSource, surfaceMasks->inputs[0].id);
            if (analysis && analysis->outputs.size() > 4) connect(analysis->outputs[4].id, surfaceMasks->inputs[1].id);
            if (flowSource) connect(flowSource, surfaceMasks->inputs[2].id);
            if (soilSource) connect(soilSource, surfaceMasks->inputs[3].id);
            if (exposure && !exposure->outputs.empty()) connect(exposure->outputs[0].id, surfaceMasks->inputs[4].id);
        }
    }
    uint32_t grassSource = grassMask && !grassMask->outputs.empty() ? grassMask->outputs[0].id : 0;
    if (!grassSource && biome && biome->outputs.size() > 1) grassSource = biome->outputs[1].id;
    if (!grassSource && recipeUses(*recipe, "grass")) {
        grassMask = dynamic_cast<TerrainGrassMaskNode*>(addTerrainNode(NodeType::GrassMask, x - 350.0f, y + 520.0f));
        if (grassMask) {
            grassMask->name = std::string(managedPrefix) + "Grass";
            connect(heightSource, grassMask->inputs[0].id);
            if (soilSource) connect(soilSource, grassMask->inputs[1].id);
            if (flowSource) connect(flowSource, grassMask->inputs[2].id);
            if (analysis && analysis->outputs.size() > 4) {
                connect(analysis->outputs[0].id, grassMask->inputs[3].id);
                connect(analysis->outputs[4].id, grassMask->inputs[4].id);
            }
            grassSource = grassMask->outputs[0].id;
        }
    }

    std::unordered_map<std::string, uint32_t> fields;
    fields["height"] = heightSource;
    fields["flow"] = flowSource;
    fields["channel_width"] = hydraulicConnected && hydraulic->outputs.size() > 6 ? hydraulic->outputs[6].id : 0;
    fields["erosion"] = hydraulicConnected && hydraulic->outputs.size() > 1 ? hydraulic->outputs[1].id : 0;
    fields["deposition"] = hydraulicConnected && hydraulic->outputs.size() > 2 ? hydraulic->outputs[2].id : 0;
    fields["soil"] = soilSource;
    fields["grass"] = grassSource;
    fields["exposure"] = exposure && !exposure->outputs.empty() ? exposure->outputs[0].id : 0;
    if (analysis && analysis->outputs.size() >= 5) {
        fields["slope"] = analysis->outputs[0].id; fields["concavity"] = analysis->outputs[1].id;
        fields["convexity"] = analysis->outputs[2].id; fields["valley"] = analysis->outputs[3].id;
        fields["wetness"] = analysis->outputs[4].id;
    }
    if (surfaceMasks && surfaceMasks->outputs.size() >= 3) {
        fields["cavity"] = surfaceMasks->outputs[0].id;
        fields["mud"] = surfaceMasks->outputs[1].id;
        fields["moss"] = surfaceMasks->outputs[2].id;
    }
    const uint32_t snowSource = snow && snow->outputs.size() > 1 ? snow->outputs[1].id : 0;

    auto* base = dynamic_cast<TerrainSatMapColorRampNode*>(addTerrainNode(NodeType::SatMapColorRamp, x, y));
    if (!base) return fail("failed to create base SatMap ColorRamp");
    base->name = std::string(managedPrefix) + recipe->label + " Base";
    base->applyPreset(recipe->basePreset);
    connect(heightSource, base->inputs[0].id);
    if (fields["slope"]) connect(fields["slope"], base->inputs[1].id);
    if (flowSource) connect(flowSource, base->inputs[2].id);
    if (soilSource) connect(soilSource, base->inputs[3].id);
    if (snowSource) connect(snowSource, base->inputs[4].id);
    if (snow && snow->outputs.size() > 4) {
        connect(snow->outputs[2].id, base->inputs[5].id);
        connect(snow->outputs[3].id, base->inputs[6].id);
        connect(snow->outputs[4].id, base->inputs[7].id);
    }
    if (grassSource) connect(grassSource, base->inputs[8].id);
    uint32_t currentColor = base->outputs[0].id;
    float layerX = x + 300.0f;

    for (const auto& layer : recipe->layers) {
        const uint32_t primary = fields[layer.primary];
        std::vector<std::string> missingFields;
        if (!primary) missingFields.push_back(layer.primary);
        for (const auto& condition : layer.conditions)
            if (!fields[condition.field] &&
                std::find(missingFields.begin(), missingFields.end(), condition.field) == missingFields.end())
                missingFields.push_back(condition.field);
        if (!missingFields.empty()) {
            std::string missingText;
            for (const auto& field : missingFields) {
                if (!missingText.empty()) missingText += ", ";
                missingText += field;
            }
            if (warnings) warnings->push_back(
                "Skipped '" + layer.label + "': missing " + missingText);
            continue;
        }
        auto* ramp = dynamic_cast<TerrainSatMapColorRampNode*>(addTerrainNode(NodeType::SatMapColorRamp, layerX, y + 210.0f));
        if (!ramp) return fail("failed to create layer ColorRamp");
        ramp->name = std::string(managedPrefix) + layer.label + " Color";
        ramp->preset = "Custom"; ramp->autoDeriveMasks = false;
        ramp->autoNormalize = layer.autoNormalize;
        ramp->slopeBlend = ramp->flowBlend = ramp->soilBlend = ramp->grassBlend = 0.0f;
        ramp->detailStrength = layer.detailStrength; ramp->detailScale = layer.detailScale;
        ramp->stops.clear();
        for (const auto& stop : layer.stops)
            ramp->stops.push_back({stop.position, stop.r, stop.g, stop.b, stop.a});
        connect(primary, ramp->inputs[0].id);

        std::vector<SatMapPresetCondition> conditions = layer.conditions;
        if (conditions.empty()) conditions.push_back({layer.primary, 0.0f, 1.0f, 1.0f, false});
        uint32_t coverage = 0;
        for (const auto& condition : conditions) {
            auto* remap = dynamic_cast<RemapNode*>(addTerrainNode(NodeType::Remap, layerX, y + 430.0f));
            if (!remap) return fail("failed to create condition remap");
            remap->name = std::string(managedPrefix) + layer.label + " / " + condition.field;
            remap->maskMode = true; remap->syncSemantic();
            remap->inputs[0].acceptImageSemantic(NodeSystem::ImageSemantic::Height);
            remap->inputs[0].acceptImageSemantic(NodeSystem::ImageSemantic::PhysicalScalar);
            remap->inputMin = condition.minimum; remap->inputMax = condition.maximum;
            remap->outputMin = condition.invert ? 1.0f : 0.0f;
            remap->outputMax = condition.invert ? 0.0f : 1.0f;
            remap->gamma = condition.gamma; remap->clampOutput = true;
            connect(fields[condition.field], remap->inputs[0].id);
            uint32_t conditionMask = remap->outputs[0].id;
            if (!coverage) coverage = conditionMask;
            else {
                auto* combine = dynamic_cast<TerrainPaintMaskCombineNode*>(
                    addTerrainNode(NodeType::PaintMaskCombine, layerX + 190.0f, y + 430.0f));
                if (!combine) return fail("failed to create mask combine");
                combine->name = std::string(managedPrefix) + layer.label + " Conditions";
                connect(coverage, combine->inputs[0].id); connect(conditionMask, combine->inputs[1].id);
                coverage = combine->outputs[0].id;
            }
        }
        if (snowSource) {
            auto* snowExclude = dynamic_cast<RemapNode*>(addTerrainNode(NodeType::Remap, layerX + 20.0f, y + 550.0f));
            auto* combine = dynamic_cast<TerrainPaintMaskCombineNode*>(
                addTerrainNode(NodeType::PaintMaskCombine, layerX + 210.0f, y + 550.0f));
            if (!snowExclude || !combine) return fail("failed to create protected snow exclusion");
            snowExclude->name = std::string(managedPrefix) + layer.label + " Snow Exclusion";
            snowExclude->maskMode = true; snowExclude->syncSemantic();
            snowExclude->inputMin = 0.05f; snowExclude->inputMax = 0.95f;
            snowExclude->outputMin = 1.0f; snowExclude->outputMax = 0.0f;
            combine->name = std::string(managedPrefix) + layer.label + " Protected Coverage";
            connect(snowSource, snowExclude->inputs[0].id);
            connect(coverage, combine->inputs[0].id); connect(snowExclude->outputs[0].id, combine->inputs[1].id);
            coverage = combine->outputs[0].id;
        }
        auto* blend = dynamic_cast<TerrainSatMapBlendNode*>(addTerrainNode(NodeType::SatMapBlend, layerX + 280.0f, y));
        if (!blend) return fail("failed to create SatMap Blend");
        blend->name = std::string(managedPrefix) + layer.label + " Blend";
        blend->opacity = layer.opacity; blend->maskPower = layer.maskPower;
        connect(currentColor, blend->inputs[0].id); connect(ramp->outputs[0].id, blend->inputs[1].id);
        connect(coverage, blend->inputs[2].id);
        currentColor = blend->outputs[0].id;
        layerX += 340.0f;
    }

    if (!output) output = dynamic_cast<TerrainSatMapOutputNode*>(addTerrainNode(NodeType::SatMapOutput, layerX + 80.0f, y));
    if (!output) return fail("failed to create SatMap Output");
    connect(currentColor, output->inputs[0].id);
    markAllDirty();
    return true;
}

} // namespace TerrainNodesV2
