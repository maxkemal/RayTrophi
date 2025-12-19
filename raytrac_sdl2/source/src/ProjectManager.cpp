#include "ProjectManager.h"
#include "globals.h"
#include "Renderer.h"
#include "OptixWrapper.h"
#include "AssimpLoader.h"
#include "Triangle.h"
#include "MaterialManager.h"
#include "PrincipledBSDF.h"
#include "json.hpp"
#include <fstream>
#include <filesystem>
#include <sstream>
#include <iomanip>

using json = nlohmann::json;
namespace fs = std::filesystem;

// Global project data instance
ProjectData g_project;

// ============================================================================
// Helper Functions
// ============================================================================

static json vec3ToJson(const Vec3& v) {
    return { v.x, v.y, v.z };
}

static Vec3 jsonToVec3(const json& j) {
    if (j.is_array() && j.size() >= 3)
        return Vec3(j[0], j[1], j[2]);
    return Vec3(0, 0, 0);
}

static json mat4ToJson(const Matrix4x4& m) {
    json j = json::array();
    for(int i = 0; i < 4; ++i)
        for(int k = 0; k < 4; ++k)
            j.push_back(m.m[i][k]);
    return j;
}

static Matrix4x4 jsonToMat4(const json& j) {
    Matrix4x4 m;
    if (j.is_array() && j.size() == 16) {
        int idx = 0;
        for(int i = 0; i < 4; ++i)
            for(int k = 0; k < 4; ++k)
                m.m[i][k] = j[idx++];
    }
    return m;
}

// Generate internal package path like "models/0001_filename.glb"
std::string ProjectManager::generatePackagePath(const std::string& original_path, 
                                                 const std::string& folder, 
                                                 uint32_t id) {
    fs::path orig(original_path);
    std::ostringstream oss;
    oss << folder << "/" << std::setfill('0') << std::setw(4) << id << "_" << orig.filename().string();
    return oss.str();
}

// ============================================================================
// Project Lifecycle
// ============================================================================

void ProjectManager::newProject() {
    g_project.clear();
    m_package_files.clear();
    SCENE_LOG_INFO("New project created.");
}

bool ProjectManager::saveProject() {
    if (g_project.current_file_path.empty()) {
        SCENE_LOG_ERROR("No file path set. Use saveProject(filepath) instead.");
        return false;
    }
    return saveProject(g_project.current_file_path);
}

bool ProjectManager::saveProject(const std::string& filepath) {
    // Build the JSON structure
    json root;
    
    // Manifest / Metadata
    root["format_version"] = g_project.format_version;
    root["project_name"] = g_project.project_name;
    root["author"] = g_project.author;
    root["description"] = g_project.description;
    
    // ID counters (for reload continuity)
    root["next_model_id"] = g_project.next_model_id;
    root["next_object_id"] = g_project.next_object_id;
    root["next_texture_id"] = g_project.next_texture_id;
    
    // Imported Models
    json modelsJson = json::array();
    for (const auto& model : g_project.imported_models) {
        json m;
        m["id"] = model.id;
        m["original_path"] = model.original_path;
        m["package_path"] = model.package_path;
        m["display_name"] = model.display_name;
        
        // Save deleted objects list
        m["deleted_objects"] = model.deleted_objects;
        
        json objectsJson = json::array();
        for (const auto& obj : model.objects) {
            json o;
            o["node_name"] = obj.node_name;
            o["transform"] = mat4ToJson(obj.transform);
            o["material_id"] = obj.material_id;
            o["visible"] = obj.visible;
            objectsJson.push_back(o);
        }
        m["objects"] = objectsJson;
        modelsJson.push_back(m);
    }
    root["imported_models"] = modelsJson;
    
    // Procedural Objects
    json proceduralJson = json::array();
    for (const auto& proc : g_project.procedural_objects) {
        json p;
        p["id"] = proc.id;
        p["mesh_type"] = (int)proc.mesh_type;
        p["display_name"] = proc.display_name;
        p["transform"] = mat4ToJson(proc.transform);
        p["material_id"] = proc.material_id;
        p["visible"] = proc.visible;
        proceduralJson.push_back(p);
    }
    root["procedural_objects"] = proceduralJson;
    
    // Texture Assets
    json texturesJson = json::array();
    for (const auto& tex : g_project.texture_assets) {
        json t;
        t["id"] = tex.id;
        t["original_path"] = tex.original_path;
        t["package_path"] = tex.package_path;
        t["usage"] = tex.usage;
        texturesJson.push_back(t);
    }
    root["texture_assets"] = texturesJson;
    
    // For now, save as plain JSON (ZIP packaging will be added later)
    // TODO: Implement writeZipPackage() for full .rtp support
    
    std::ofstream out(filepath);
    if (!out.is_open()) {
        SCENE_LOG_ERROR("Failed to save project: " + filepath);
        return false;
    }
    
    out << root.dump(4);
    out.close();
    
    g_project.current_file_path = filepath;
    g_project.is_modified = false;
    
    SCENE_LOG_INFO("Project saved: " + filepath);
    return true;
}

bool ProjectManager::openProject(const std::string& filepath, SceneData& scene,
                                  RenderSettings& settings, Renderer& renderer, 
                                  OptixWrapper* optix_gpu) {
    std::ifstream in(filepath);
    if (!in.is_open()) {
        SCENE_LOG_ERROR("Failed to open project: " + filepath);
        return false;
    }
    
    json root;
    try {
        in >> root;
    } catch (const std::exception& e) {
        SCENE_LOG_ERROR("Failed to parse project file: " + std::string(e.what()));
        return false;
    }
    in.close();
    
    // Clear current project and scene
    newProject();
    scene.clear();
    renderer.resetCPUAccumulation();
    if (optix_gpu) optix_gpu->resetAccumulation();
    
    // Load metadata
    g_project.format_version = root.value("format_version", "2.0");
    g_project.project_name = root.value("project_name", "Untitled");
    g_project.author = root.value("author", "");
    g_project.description = root.value("description", "");
    g_project.next_model_id = root.value("next_model_id", 1);
    g_project.next_object_id = root.value("next_object_id", 1);
    g_project.next_texture_id = root.value("next_texture_id", 1);
    
    // Get project folder (for resolving relative paths)
    fs::path project_folder = fs::path(filepath).parent_path();
    
    // Load Imported Models
    if (root.contains("imported_models")) {
        for (const auto& m : root["imported_models"]) {
            ImportedModelData model;
            model.id = m.value("id", 0);
            model.original_path = m.value("original_path", "");
            model.package_path = m.value("package_path", "");
            model.display_name = m.value("display_name", "Model");
            
            // Load deleted objects list
            if (m.contains("deleted_objects") && m["deleted_objects"].is_array()) {
                for (const auto& d : m["deleted_objects"]) {
                    model.deleted_objects.push_back(d.get<std::string>());
                }
            }
            
            // Resolve actual file path
            std::string actual_path;
            fs::path pkg_path = project_folder / model.package_path;
            if (fs::exists(pkg_path)) {
                actual_path = pkg_path.string();
            } else if (fs::exists(model.original_path)) {
                actual_path = model.original_path;
                SCENE_LOG_WARN("Using original path for: " + model.display_name);
            } else {
                SCENE_LOG_ERROR("Model file not found: " + model.package_path);
                continue;
            }
            
            // Remember object count before loading
            size_t objects_before = scene.world.objects.size();
            
            // Load with Assimp (add to scene without clearing)
            renderer.create_scene(scene, optix_gpu, actual_path, nullptr);
            
            // Remove deleted objects from scene
            if (!model.deleted_objects.empty()) {
                auto& objs = scene.world.objects;
                objs.erase(
                    std::remove_if(objs.begin() + objects_before, objs.end(),
                        [&model](const std::shared_ptr<Hittable>& obj) {
                            auto tri = std::dynamic_pointer_cast<Triangle>(obj);
                            if (!tri) return false;
                            return std::find(model.deleted_objects.begin(), 
                                           model.deleted_objects.end(), 
                                           tri->nodeName) != model.deleted_objects.end();
                        }),
                    objs.end());
                SCENE_LOG_INFO("Removed " + std::to_string(model.deleted_objects.size()) + " deleted objects from " + model.display_name);
            }
            
            // Load object transforms
            if (m.contains("objects")) {
                for (const auto& o : m["objects"]) {
                    ImportedModelData::ObjectInstance inst;
                    inst.node_name = o.value("node_name", "");
                    if (o.contains("transform")) {
                        inst.transform = jsonToMat4(o["transform"]);
                    }
                    inst.material_id = o.value("material_id", 0);
                    inst.visible = o.value("visible", true);
                    model.objects.push_back(inst);
                }
            }
            
            g_project.imported_models.push_back(model);
        }
    }
    
    // Apply transforms to loaded objects
    // Match by node name
    for (const auto& model : g_project.imported_models) {
        for (const auto& inst : model.objects) {
            for (auto& obj : scene.world.objects) {
                auto tri = std::dynamic_pointer_cast<Triangle>(obj);
                if (tri && tri->nodeName == inst.node_name) {
                    auto th = tri->getTransformHandle();
                    if (!th) {
                        th = std::make_shared<Transform>();
                        tri->setTransformHandle(th);
                    }
                    th->setBase(inst.transform);
                    tri->setMaterialID(inst.material_id);
                    tri->updateTransformedVertices();
                }
            }
        }
    }
    
    // Load Procedural Objects
    if (root.contains("procedural_objects")) {
        for (const auto& p : root["procedural_objects"]) {
            ProceduralObjectData proc;
            proc.id = p.value("id", 0);
            proc.mesh_type = (ProceduralMeshType)p.value("mesh_type", 0);
            proc.display_name = p.value("display_name", "Object");
            if (p.contains("transform")) {
                proc.transform = jsonToMat4(p["transform"]);
            }
            proc.material_id = p.value("material_id", 0);
            proc.visible = p.value("visible", true);
            
            // Recreate the procedural mesh
            // TODO: Implement createProceduralMesh helper
            
            g_project.procedural_objects.push_back(proc);
        }
    }
    
    // Load Texture Assets (for reference tracking)
    if (root.contains("texture_assets")) {
        for (const auto& t : root["texture_assets"]) {
            TextureAssetData tex;
            tex.id = t.value("id", 0);
            tex.original_path = t.value("original_path", "");
            tex.package_path = t.value("package_path", "");
            tex.usage = t.value("usage", "");
            g_project.texture_assets.push_back(tex);
        }
    }
    
    g_project.current_file_path = filepath;
    g_project.is_modified = false;
    
    // Rebuild acceleration structures
    renderer.rebuildBVH(scene, settings.UI_use_embree);
    if (optix_gpu) {
        renderer.rebuildOptiXGeometry(scene, optix_gpu);
        optix_gpu->setLightParams(scene.lights);
        if (scene.camera) optix_gpu->setCameraParams(*scene.camera);
    }
    
    SCENE_LOG_INFO("Project loaded: " + filepath);
    return true;
}

// ============================================================================
// Asset Import
// ============================================================================

bool ProjectManager::importModel(const std::string& filepath, SceneData& scene,
                                  Renderer& renderer, OptixWrapper* optix_gpu,
                                  std::function<void(int, const std::string&)> progress_callback) {
    if (!fs::exists(filepath)) {
        SCENE_LOG_ERROR("File not found: " + filepath);
        return false;
    }
    
    // Generate package path and unique ID
    uint32_t id = g_project.generateModelId();
    std::string package_path = generatePackagePath(filepath, "models", id);
    
    // Unique prefix for this import (prevents name collision on re-import)
    std::string import_prefix = std::to_string(id) + "_";
    
    // Create model data entry
    ImportedModelData model;
    model.id = id;
    model.original_path = filepath;
    model.package_path = package_path;
    model.display_name = fs::path(filepath).stem().string();
    
    // Copy file to package storage (if project folder exists)
    if (!g_project.current_file_path.empty()) {
        fs::path project_folder = fs::path(g_project.current_file_path).parent_path();
        fs::path dest_path = project_folder / package_path;
        
        try {
            fs::create_directories(dest_path.parent_path());
            fs::copy_file(filepath, dest_path, fs::copy_options::overwrite_existing);
        } catch (const std::exception& e) {
            SCENE_LOG_WARN("Could not copy model file: " + std::string(e.what()));
        }
    }
    
    // Remember current object count to track new objects
    size_t objects_before = scene.world.objects.size();
    
    // Load with Assimp - append mode (don't clear existing scene)
    renderer.create_scene(scene, optix_gpu, filepath, progress_callback, true);  // append = true
    
    // Track new objects and add unique prefix to prevent name collision
    for (size_t i = objects_before; i < scene.world.objects.size(); ++i) {
        auto tri = std::dynamic_pointer_cast<Triangle>(scene.world.objects[i]);
        if (tri) {
            // Add unique prefix to node name to prevent collision on re-import
            std::string original_name = tri->nodeName;
            std::string unique_name = import_prefix + original_name;
            tri->setNodeName(unique_name);
            
            ImportedModelData::ObjectInstance inst;
            inst.node_name = unique_name;  // Store the unique name
            auto th = tri->getTransformHandle();
            if (th) {
                inst.transform = th->base;
            }
            inst.material_id = tri->getMaterialID();
            inst.visible = true;
            model.objects.push_back(inst);
        }
    }
    
    g_project.imported_models.push_back(model);
    g_project.is_modified = true;
    
    // CRITICAL: Rebuild acceleration structures after import
    extern RenderSettings render_settings;  // From globals or Main.cpp
    renderer.rebuildBVH(scene, render_settings.UI_use_embree);
    renderer.resetCPUAccumulation();
    
    if (optix_gpu) {
        renderer.rebuildOptiXGeometry(scene, optix_gpu);
        
        // IMPORTANT: Update lights on GPU after geometry rebuild
        optix_gpu->setLightParams(scene.lights);
        
        // Update camera on GPU as well
        if (scene.camera) {
            optix_gpu->setCameraParams(*scene.camera);
        }
        
        optix_gpu->resetAccumulation();
    }
    
    SCENE_LOG_INFO("Model imported: " + model.display_name + " (ID: " + std::to_string(id) + ", " + std::to_string(model.objects.size()) + " objects)");
    return true;
}

bool ProjectManager::importTexture(const std::string& filepath) {
    if (!fs::exists(filepath)) {
        SCENE_LOG_ERROR("Texture file not found: " + filepath);
        return false;
    }
    
    uint32_t id = g_project.generateTextureId();
    std::string package_path = generatePackagePath(filepath, "textures", id);
    
    TextureAssetData tex;
    tex.id = id;
    tex.original_path = filepath;
    tex.package_path = package_path;
    tex.usage = "unknown";
    
    // Copy to project folder if exists
    if (!g_project.current_file_path.empty()) {
        fs::path project_folder = fs::path(g_project.current_file_path).parent_path();
        fs::path dest_path = project_folder / package_path;
        
        try {
            fs::create_directories(dest_path.parent_path());
            fs::copy_file(filepath, dest_path, fs::copy_options::overwrite_existing);
        } catch (const std::exception& e) {
            SCENE_LOG_WARN("Could not copy texture file: " + std::string(e.what()));
        }
    }
    
    g_project.texture_assets.push_back(tex);
    g_project.is_modified = true;
    
    SCENE_LOG_INFO("Texture imported: " + fs::path(filepath).filename().string());
    return true;
}

// ============================================================================
// Procedural Object Management
// ============================================================================

uint32_t ProjectManager::addProceduralObject(ProceduralMeshType type, const std::string& name,
                                              const Matrix4x4& transform, SceneData& scene,
                                              Renderer& renderer, OptixWrapper* optix_gpu) {
    uint32_t id = g_project.generateObjectId();
    
    ProceduralObjectData proc;
    proc.id = id;
    proc.mesh_type = type;
    proc.display_name = name;
    proc.transform = transform;
    proc.material_id = 0;
    proc.visible = true;
    
    // Create actual geometry based on type
    // (This is similar to scene_ui_menu.hpp Add > Mesh logic)
    // TODO: Consolidate mesh creation into helper functions
    
    g_project.procedural_objects.push_back(proc);
    g_project.is_modified = true;
    
    return id;
}

bool ProjectManager::removeProceduralObject(uint32_t id, SceneData& scene) {
    auto it = std::find_if(g_project.procedural_objects.begin(), 
                           g_project.procedural_objects.end(),
                           [id](const ProceduralObjectData& p) { return p.id == id; });
    
    if (it == g_project.procedural_objects.end()) {
        return false;
    }
    
    // TODO: Remove corresponding triangles from scene.world.objects
    
    g_project.procedural_objects.erase(it);
    g_project.is_modified = true;
    
    return true;
}

// ============================================================================
// Package Utilities (Placeholder for ZIP implementation)
// ============================================================================

std::vector<uint8_t> ProjectManager::extractFileFromPackage(const std::string& internal_path) {
    // TODO: Implement ZIP extraction
    // For now, try to read from project folder
    if (!g_project.current_file_path.empty()) {
        fs::path project_folder = fs::path(g_project.current_file_path).parent_path();
        fs::path file_path = project_folder / internal_path;
        
        if (fs::exists(file_path)) {
            std::ifstream in(file_path, std::ios::binary);
            if (in) {
                return std::vector<uint8_t>(std::istreambuf_iterator<char>(in), {});
            }
        }
    }
    return {};
}

bool ProjectManager::fileExistsInPackage(const std::string& internal_path) {
    // TODO: Implement ZIP check
    if (!g_project.current_file_path.empty()) {
        fs::path project_folder = fs::path(g_project.current_file_path).parent_path();
        return fs::exists(project_folder / internal_path);
    }
    return false;
}

bool ProjectManager::writeZipPackage(const std::string& filepath) {
    // TODO: Implement with miniz
    // For now, use folder-based approach
    SCENE_LOG_WARN("ZIP packaging not yet implemented. Using folder structure.");
    return saveProject(filepath);
}

bool ProjectManager::readZipPackage(const std::string& filepath) {
    // TODO: Implement with miniz
    SCENE_LOG_WARN("ZIP packaging not yet implemented.");
    return false;
}
