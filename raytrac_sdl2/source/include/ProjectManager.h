#pragma once

#include "ProjectData.h"
#include "scene_data.h"
#include "Camera.h"
#include "Light.h"
#include "json.hpp"
#include <string>
#include <vector>
#include <functional>
#include <fstream>
#include <unordered_map>

// TextureType enum forward declaration (defined in Texture.h)
enum class TextureType;

// Forward declarations
class Renderer;
class OptixWrapper;

// ============================================================================
// RayTrophi Project Manager
// Handles .rtp package creation, loading, and asset management
// ============================================================================

class ProjectManager {
public:
    // Singleton access
    static ProjectManager& getInstance() {
        static ProjectManager instance;
        return instance;
    }
    
    // ========================================================================
    // Project Lifecycle
    // ========================================================================
    
    // Create a new empty project
    void newProject(SceneData& scene, Renderer& renderer);
    
    // Save current project to .rtp file (NEW: includes geometry, lights, cameras)
    // Returns true on success
    bool saveProject(const std::string& filepath, SceneData& scene, RenderSettings& settings, Renderer& renderer,
                     std::function<void(int, const std::string&)> progress_callback = nullptr);
    
    // Save without dialog if path already known
    bool saveProject(SceneData& scene, RenderSettings& settings, Renderer& renderer,
                     std::function<void(int, const std::string&)> progress_callback = nullptr);
    
    // Synchronize ProjectData with live SceneData (Captures moves, deletes, etc.)
    void syncProjectToScene(SceneData& scene);
    
    // Load project from .rtp file
    // Clears current scene and loads everything from package
    bool openProject(const std::string& filepath, SceneData& scene, 
                     RenderSettings& settings, Renderer& renderer, OptixWrapper* optix_gpu,
                     std::function<void(int, const std::string&)> progress_callback = nullptr);
    
    // ========================================================================
    // Asset Import (adds to current project, doesn't clear scene)
    // ========================================================================
    
    // Import a 3D model file into the project
    // Copies file to package, loads geometry, adds to scene
    bool importModel(const std::string& filepath, SceneData& scene,
                     Renderer& renderer, OptixWrapper* optix_gpu,
                     std::function<void(int, const std::string&)> progress_callback = nullptr,
                     bool rebuild = true);
    
    // Import a texture file into the project
    bool importTexture(const std::string& filepath);
    
    // ========================================================================
    // Procedural Object Management
    // ========================================================================
    
    // Add a procedural object (Cube, Plane, etc.)
    uint32_t addProceduralObject(ProceduralMeshType type, const std::string& name,
                                 const Matrix4x4& transform, SceneData& scene,
                                 Renderer& renderer, OptixWrapper* optix_gpu);
    
    // Remove procedural object by ID
    bool removeProceduralObject(uint32_t id, SceneData& scene);
    
    // ========================================================================
    // Accessors
    // ========================================================================
    
    ProjectData& getProjectData() { return g_project; }
    const ProjectData& getProjectData() const { return g_project; }
    
    bool hasUnsavedChanges() const { return g_project.is_modified; }
    std::string getProjectName() const { return g_project.project_name; }
    std::string getCurrentFilePath() const { return g_project.current_file_path; }
    
    // Mark project as modified (triggers "unsaved changes" prompt)
    void markModified() { g_project.is_modified = true; }
    
    // ========================================================================
    // Package Utilities
    // ========================================================================
    
    // Extract a file from current .rtp package to memory
    std::vector<uint8_t> extractFileFromPackage(const std::string& internal_path);
    
    // Check if a file exists in current package
    bool fileExistsInPackage(const std::string& internal_path);
    
private:
    ProjectManager() = default;
    ~ProjectManager() = default;
    ProjectManager(const ProjectManager&) = delete;
    ProjectManager& operator=(const ProjectManager&) = delete;
    
    // Internal helpers
    std::string generatePackagePath(const std::string& original_path, const std::string& folder, uint32_t id);
    bool writeZipPackage(const std::string& filepath);
    bool readZipPackage(const std::string& filepath);
    
    // ========================================================================
    // Geometry Serialization (Self-Contained Project)
    // ========================================================================
    
    // Write all scene geometry to binary file
    bool writeGeometryBinary(std::ofstream& out, const SceneData& scene);
    
    // Read geometry from binary file and recreate scene objects
    bool readGeometryBinary(std::ifstream& in, SceneData& scene);
    
    // ========================================================================
    // Component Serialization
    // ========================================================================
    
    // Serialize lights to JSON
    nlohmann::json serializeLights(const std::vector<std::shared_ptr<Light>>& lights);
    void deserializeLights(const nlohmann::json& j, std::vector<std::shared_ptr<Light>>& lights);
    
    // Serialize cameras to JSON
    nlohmann::json serializeCameras(const std::vector<std::shared_ptr<Camera>>& cameras, size_t active_index);
    void deserializeCameras(const nlohmann::json& j, SceneData& scene);
    
    // Serialize render settings
    nlohmann::json serializeRenderSettings(const RenderSettings& settings);
    void deserializeRenderSettings(const nlohmann::json& j, RenderSettings& settings);
    
    // Serialize VDB Volumes
    nlohmann::json serializeVDBVolumes(const std::vector<std::shared_ptr<VDBVolume>>& vdb_volumes);
    void deserializeVDBVolumes(const nlohmann::json& j, SceneData& scene);

    // Serialize textures (with embed option)
    nlohmann::json serializeTextures(std::ofstream& bin_out, bool embed_textures);
    void deserializeTextures(const nlohmann::json& j, std::ifstream& bin_in, const std::string& project_dir);
    
    // Temporary storage for package contents during editing
    struct PackageFile {
        std::string path;
        std::vector<uint8_t> data;
    };
    std::vector<PackageFile> m_package_files;
    
    // ========================================================================
    // Embedded Texture Cache - stores texture binary data in memory
    // Avoids writing temp files to disk when loading embedded textures
    // ========================================================================
public:
    struct EmbeddedTextureData {
        std::vector<char> data;
        TextureType type;
    };
    std::unordered_map<std::string, EmbeddedTextureData> m_embedded_texture_cache;
    
    // Get embedded texture data by name (returns nullptr if not found)
    const EmbeddedTextureData* getEmbeddedTexture(const std::string& name) const {
        auto it = m_embedded_texture_cache.find(name);
        return (it != m_embedded_texture_cache.end()) ? &it->second : nullptr;
    }
    
    // Clear embedded texture cache (call on new/open project)
    void clearEmbeddedTextureCache() { m_embedded_texture_cache.clear(); }
    
private:
    
public:
    // ========================================================================
    // Project Save Settings
    // ========================================================================
    struct SaveSettings {
        bool embed_textures = true;        // Pack textures into binary (true = self-contained project)
        bool embed_missing_only = false;   // Embed all textures, not just missing ones
        bool save_geometry = true;         // Save scene geometry (for self-contained)
    };
    SaveSettings save_settings;
};

// Convenience macro
#define g_ProjectManager ProjectManager::getInstance()
