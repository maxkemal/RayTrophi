#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <atomic>
#include "Texture.h"
// Assimp headers
#include <assimp/Exporter.hpp>
// Forward Declarations
struct SceneData;
class Material;
struct GpuMaterial;
class Hittable;

struct ExportSettings {
    bool export_geometry = true;
    bool export_materials = true;
    bool export_lights = true;
    bool export_cameras = true;
    bool export_animations = true;
    bool export_skinning = true;  // Bones/Weights
    bool bake_transforms = false; // If true, bakes static transforms (good for static meshes)
    bool binary_mode = true;      // .glb vs .gltf
    bool export_selected_only = false; 
    bool embed_textures = true;
};

class SceneExporter {
public:
    static SceneExporter& getInstance() {
        static SceneExporter instance;
        return instance;
    }
    
    // UI Logic
    bool show_export_popup = false;
    std::atomic<bool> is_exporting{false}; // Thread-safe status flag
    std::string current_export_status = ""; // For HUD feedback
    
    ExportSettings settings;
    bool drawExportPopup(SceneData& scene); // Returns true if user clicked Export

    /**
     * @brief Exports the entire scene to a GLTF/GLB file.
     * 
     * @param filepath Output path (e.g. "C:/Exports/scene.glb")
     * @param scene Reference to the active SceneData
     * @param settings Export configuration
     * @param selected_objects Optional list of selected objects (if export_selected_only is true)
     * @return true if successful
     */
    bool exportScene(const std::string& filepath, SceneData& scene, const ExportSettings& settings, 
                     const std::vector<std::shared_ptr<Hittable>>& selected_objects = {});

private:
    SceneExporter() = default;
    ~SceneExporter() = default;
    
    // Non-copyable
    SceneExporter(const SceneExporter&) = delete;
    SceneExporter& operator=(const SceneExporter&) = delete;

    // Helpers
    aiMaterial* createAssimpMaterial(const std::shared_ptr<Material>& mat, const std::string& name);
    void processNode(aiNode* node, const struct ExportNode& exportNode, std::vector<aiMesh*>& meshes, std::vector<aiMaterial*>& materials);

    // Temp storage for export
    std::vector<aiTexture*> accumulated_textures;
    std::map<std::string, std::string> texture_dedup_map; // Key: Ptr/Path, Value: EmbeddedName ("*0")
};
