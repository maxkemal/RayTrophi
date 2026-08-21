#pragma once

#include <string>
#include <vector>

namespace TerrainNodesV2 {

struct SatMapPresetStop {
    float position = 0.0f;
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
    float a = 1.0f;
};

struct SatMapPresetCondition {
    std::string field;
    float minimum = 0.0f;
    float maximum = 1.0f;
    float gamma = 1.0f;
    bool invert = false;
};

struct SatMapPresetLayer {
    std::string id;
    std::string label;
    std::string primary;
    std::vector<SatMapPresetCondition> conditions;
    std::vector<SatMapPresetStop> stops;
    float opacity = 1.0f;
    float maskPower = 1.0f;
    float detailStrength = 0.075f;
    float detailScale = 180.0f;
    bool autoNormalize = false;
};

struct SatMapPresetRecipe {
    std::string id;
    std::string label;
    std::string category;
    std::string description;
    int version = 1;
    std::string basePreset = "Temperate";
    std::vector<SatMapPresetLayer> layers;
    std::string sourcePath;
};

class SatMapPresetLibrary {
public:
    static SatMapPresetLibrary& instance();

    bool reload(std::string* error = nullptr);
    const std::vector<SatMapPresetRecipe>& presets() const { return presets_; }
    const SatMapPresetRecipe* find(const std::string& id) const;
    const std::vector<std::string>& diagnostics() const { return diagnostics_; }

private:
    std::vector<SatMapPresetRecipe> presets_;
    std::vector<std::string> diagnostics_;
};

} // namespace TerrainNodesV2
