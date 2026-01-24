/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          ProjectData.h
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once

#include <string>
#include <vector>
#include <memory>
#include "matrix4x4.h"
#include "Vec3.h"

// ============================================================================
// RayTrophi Project System - Core Data Structures
// ============================================================================

// Procedural mesh types that can be recreated from code
enum class ProceduralMeshType {
    None = 0,
    Cube,
    Plane,
    Sphere,
    Cylinder,
    // Add more as needed
};

// Represents an imported 3D model file
struct ImportedModelData {
    uint32_t id = 0;                        // Unique ID within project
    std::string original_path;               // Original file path (for reference)
    std::string package_path;                // Path inside .rtp package (e.g., "models/0001_car.glb")
    std::string display_name;                // User-friendly name
    
    // Objects that were DELETED by user after import
    // When loading: Load model, then remove these nodes
    std::vector<std::string> deleted_objects;
    
    // Per-object transforms (each mesh node in the model)
    struct ObjectInstance {
        std::string node_name;               // Node name from Assimp
        Matrix4x4 transform;                 // World transform
        uint16_t material_id = 0;            // Material assignment
        bool visible = true;
    };
    std::vector<ObjectInstance> objects;
};

// Represents a procedural object (created via Add > Cube, etc.)
struct ProceduralObjectData {
    uint32_t id = 0;                        // Unique ID
    ProceduralMeshType mesh_type = ProceduralMeshType::None;
    std::string display_name;                // User-assigned name
    Matrix4x4 transform;                     // World transform
    uint16_t material_id = 0;                // Material assignment
    bool visible = true;
};

// Represents a texture file in the project
struct TextureAssetData {
    uint32_t id = 0;
    std::string original_path;               // Original file path
    std::string package_path;                // Path inside package (e.g., "textures/0001_diffuse.png")
    std::string usage;                       // "diffuse", "normal", "roughness", etc.
};

// Main project container
struct ProjectData {
    // Metadata
    std::string project_name = "Untitled";
    std::string format_version = "2.0";
    std::string author;
    std::string description;
    
    // Asset tracking
    std::vector<ImportedModelData> imported_models;
    std::vector<ProceduralObjectData> procedural_objects;
    std::vector<TextureAssetData> texture_assets;
    
    // ID counters for unique assignment
    uint32_t next_model_id = 1;
    uint32_t next_object_id = 1;
    uint32_t next_texture_id = 1;
    
    // Project state
    bool is_modified = false;                // Unsaved changes flag
    std::string current_file_path;           // Currently open .rtp file path
    
    // Helper methods
    void clear() {
        project_name = "Untitled";
        imported_models.clear();
        procedural_objects.clear();
        texture_assets.clear();
        next_model_id = 1;
        next_object_id = 1;
        next_texture_id = 1;
        is_modified = false;
        current_file_path.clear();
    }
    
    uint32_t generateModelId() { return next_model_id++; }
    uint32_t generateObjectId() { return next_object_id++; }
    uint32_t generateTextureId() { return next_texture_id++; }
    
    // Find model by ID
    ImportedModelData* findModelById(uint32_t id) {
        for (auto& m : imported_models) {
            if (m.id == id) return &m;
        }
        return nullptr;
    }
    
    // Find procedural object by ID
    ProceduralObjectData* findProceduralById(uint32_t id) {
        for (auto& p : procedural_objects) {
            if (p.id == id) return &p;
        }
        return nullptr;
    }
};

// Global project instance (similar to Unity's current project)
// Defined in ProjectManager.cpp
extern ProjectData g_project;

