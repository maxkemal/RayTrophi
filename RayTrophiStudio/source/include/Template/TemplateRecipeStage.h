#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

class Camera;
class Light;
class Material;
class TriangleMesh;

namespace raytrophi::templates {

struct TemplateRecipeStage {
    std::string preset;
    std::shared_ptr<TriangleMesh> mesh;
    std::shared_ptr<Material> material;
    std::shared_ptr<Light> light;
    std::shared_ptr<Camera> camera;
    bool ready = false;
    std::string code = "invalid_recipe";
    std::vector<std::string> errors;
};

class TemplateRecipeStager {
public:
    static TemplateRecipeStage stage(const std::filesystem::path& recipe_path);

    // Canonical list of presets this stager can actually commit.
    //
    // Preflight must ask here rather than keep its own copy. A second list
    // would let `prepare` report `ready` for a preset that `open` then refuses,
    // silently breaking the loader's only promise: ready means committable.
    // stage() dispatches from the same table, so the two cannot drift.
    static const std::vector<std::string>& supportedPresets();
    static bool supportsPreset(const std::string& preset);
};

} // namespace raytrophi::templates
