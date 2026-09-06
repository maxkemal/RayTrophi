#include "Import/ModelImport.h"
#include "globals.h"
#include <filesystem>
#include <stdexcept>
namespace rtimport {
ImportedModel loadSceneModel(const std::string& path, const std::string& prefix, bool singleFacade) {
    ImportOptions options;
    const std::filesystem::path file(path);
    options.importPrefix = prefix.empty() ? file.parent_path().filename().string() + "_" + file.stem().string() : prefix;
    options.emitSingleFacadePerMesh = singleFacade;
    ImportedModel out;
    std::string error;
    if (!loadModel(path, options, out, error)) {
        SCENE_LOG_ERROR("[Import] " + error);
        throw std::runtime_error(error);
    }
    const auto& s = out.stats;
    const std::string label = s.reader == "cgltf" ? "glTF" : s.reader;
    SCENE_LOG_INFO("[" + label + "] direct reader: " + std::to_string(s.mesh_count) + " mesh(es), " +
        std::to_string(s.triangle_count) + " tri, " + std::to_string(s.animation_count) + " clip(s), " +
        std::to_string(s.bone_count) + " bone(s), " + std::to_string(s.instance_count) + " instance(s) in " +
        std::to_string(s.seconds_total) + " s");
    {
        SCENE_LOG_INFO("[" + label + "] " + std::to_string(s.material_count) + " material(s), " +
            std::to_string(s.image_count) + " image(s)");
        SCENE_LOG_INFO("[" + label + "]   parse " + std::to_string(s.seconds_parse) +
            " s | materials+textures " + std::to_string(s.seconds_materials) +
            " s | geometry " + std::to_string(s.seconds_geometry) +
            " s | animation " + std::to_string(s.seconds_animation) + " s");
    }
    return out;
}
}
