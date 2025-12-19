#pragma once

#include "ProjectData.h"
#include "scene_data.h"
#include <string>
#include <vector>
#include <functional>

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
    void newProject();
    
    // Save current project to .rtp file
    // Returns true on success
    bool saveProject(const std::string& filepath);
    
    // Save without dialog if path already known
    bool saveProject();
    
    // Load project from .rtp file
    // Clears current scene and loads everything from package
    bool openProject(const std::string& filepath, SceneData& scene, 
                     RenderSettings& settings, Renderer& renderer, OptixWrapper* optix_gpu);
    
    // ========================================================================
    // Asset Import (adds to current project, doesn't clear scene)
    // ========================================================================
    
    // Import a 3D model file into the project
    // Copies file to package, loads geometry, adds to scene
    bool importModel(const std::string& filepath, SceneData& scene,
                     Renderer& renderer, OptixWrapper* optix_gpu,
                     std::function<void(int, const std::string&)> progress_callback = nullptr);
    
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
    
    // Temporary storage for package contents during editing
    struct PackageFile {
        std::string path;
        std::vector<uint8_t> data;
    };
    std::vector<PackageFile> m_package_files;
};

// Convenience macro
#define g_ProjectManager ProjectManager::getInstance()
