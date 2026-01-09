#include "ProjectManager.h"
#include "globals.h"
#include "Renderer.h"
#include "OptixWrapper.h"
#include "AssimpLoader.h"
#include "Triangle.h"
#include "MaterialManager.h"
#include "PrincipledBSDF.h"
#include "PointLight.h"
#include "DirectionalLight.h"
#include "SpotLight.h"
#include "AreaLight.h"
#include "WaterSystem.h"
#include "json.hpp"
#include <fstream>
#include <filesystem>
#include <sstream>
#include <iomanip>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <thread>
#include <chrono>

#include <chrono>
#include "TerrainManager.h"
#include "RiverSpline.h"

#include "stb_image_write.h"

// Helper for stbi_write_to_func
namespace {
    void write_to_vector_func(void* context, void* data, int size) {
        auto* vec = static_cast<std::vector<char>*>(context);
        const char* d = static_cast<const char*>(data);
        vec->insert(vec->end(), d, d + size);
    }
}

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
    for(int i=0; i<4; ++i) for(int k=0; k<4; ++k) m.m[i][k] = (i==k ? 1.0f : 0.0f);

    if (j.is_array() && j.size() == 16) {
        int idx = 0;
        for(int i = 0; i < 4; ++i)
            for(int k = 0; k < 4; ++k)
                m.m[i][k] = j[idx++];
    }
    return m;
}

// Helper to check for identity matrix with epsilon
static bool isIdentity(const Matrix4x4& m) {
    const float epsilon = 0.0001f;
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            float expected = (r == c) ? 1.0f : 0.0f;
            if (std::abs(m.m[r][c] - expected) > epsilon) return false;
        }
    }
    return true;
}

// Binary IO Helpers
static void writeStringBinary(std::ofstream& out, const std::string& str) {
    uint16_t len = static_cast<uint16_t>(str.length());
    out.write(reinterpret_cast<const char*>(&len), sizeof(len));
    if (len > 0) out.write(str.data(), len);
}

static std::string readStringBinary(std::ifstream& in) {
    uint16_t len = 0;
    in.read(reinterpret_cast<char*>(&len), sizeof(len));
    if (len == 0) return "";
    std::string str(len, '\0');
    in.read(&str[0], len);
    return str;
}

// Generate internal package path like "models/0001/filename.glb"
std::string ProjectManager::generatePackagePath(const std::string& original_path, 
                                                 const std::string& folder, 
                                                 uint32_t id) {
    fs::path orig(original_path);
    std::ostringstream oss;
    // Use subfolders to prevent name collisions and preserve relative paths (e.g. for .bin files)
    oss << folder << "/" << std::setfill('0') << std::setw(4) << id << "/" << orig.filename().string();
    return oss.str();
}

// ============================================================================
// Serialization / Sync Logic
// ============================================================================

void ProjectManager::syncProjectToScene(SceneData& scene) {
    // 1. Build a Quick Lookup Map for Scene Objects
    std::unordered_map<std::string, std::shared_ptr<Triangle>> scene_obj_map;
    // Utilize cache if available, otherwise build it
    for (const auto& obj : scene.world.objects) {
        if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
            if (!tri->nodeName.empty()) {
                scene_obj_map[tri->nodeName] = tri;
            }
        }
    }

    // 2. Iterate Imported Models and Sync
    for (auto& model : g_project.imported_models) {
        // Clear old deleted list - we will rebuild it
        // model.deleted_objects.clear(); // Keep manual ones? No, state should track scene.
        // Actually, if we just check existing model.objects, we might miss objects that were ALREADY deleted and removed from model.objects?
        // Wait, imported_models.objects is the LIST of ALL imported nodes.
        // We assume model.objects contains everything from the source file.
        // If it was partial, we are in trouble.
        // Assimp load populates it fully in importModel phase.
        
        // Re-evaluate deleted objects
        std::vector<std::string> current_deleted;
        
        for (auto& inst : model.objects) {
            auto it = scene_obj_map.find(inst.node_name);
            if (it != scene_obj_map.end()) {
                // Object Exists in Scene - Update Status
                auto tri = it->second;
                
                // Update Transform
                if (auto th = tri->getTransformHandle()) {
                    inst.transform = th->base;
                } else {
                    // Fallback if no handle (shouldn't happen for valid objects)
                }
                
                // Update Material
                inst.material_id = tri->getMaterialID();
                
                // Update Visibility (approximation, as Triangle doesn't strictly have 'visible' flag exposed easily usually, 
                // but let's assume if it's in the list, it's visible. 
                // If you implement a specific hide flag in Triangle, use it here.)
                inst.visible = true; 
            } else {
                // Object MISSING from Scene -> It is DELETED
                current_deleted.push_back(inst.node_name);
                inst.visible = false; // Mark implicit
            }
        }
        
        model.deleted_objects = current_deleted;
    }
    
    // 3. Sync Procedurals
    // Procedurals are usually directly managed in `procedural_objects` list by Add/Remove calls.
    // But we should sync transforms just in case.
    for (auto& proc : g_project.procedural_objects) {
        // Find procedural object in scene by Name (or ID if we stored it in Triangle?)
        // Currently relying on Name matching for procedurals is risky if duplicates allowed.
        // But let's try.
        // Better: Procedural creation assigns 'nodeName' = 'display_name'.
        auto it = scene_obj_map.find(proc.display_name);
        if (it != scene_obj_map.end()) {
             if (auto th = it->second->getTransformHandle()) {
                 proc.transform = th->base;
             }
             proc.material_id = it->second->getMaterialID();
        }
    }
}

// ============================================================================
// Project Lifecycle
// ============================================================================

void ProjectManager::newProject() {
    g_project.clear();
    m_package_files.clear();
    RiverManager::getInstance().clear();
    SCENE_LOG_INFO("New project created.");
}

// ============================================================================
// NEW SAVE PROJECT (Self-Contained with Geometry)
// ============================================================================

bool ProjectManager::saveProject(SceneData& scene, RenderSettings& settings,
                                  std::function<void(int, const std::string&)> progress_callback) {
    if (g_project.current_file_path.empty()) {
        SCENE_LOG_ERROR("No file path set. Use saveProject(filepath, scene, settings) instead.");
        return false;
    }
    return saveProject(g_project.current_file_path, scene, settings, progress_callback);
}

bool ProjectManager::saveProject(const std::string& filepath, SceneData& scene, RenderSettings& settings,
                                  std::function<void(int, const std::string&)> progress_callback) {
    if (progress_callback) progress_callback(0, "Preparing to save...");

    // Sync project data with current scene state
    syncProjectToScene(scene);

    // 1. Prepare Safe Save paths
    fs::path final_json_path(filepath);
    fs::path final_bin_path = final_json_path;
    final_bin_path += ".bin";
    
    fs::path temp_json_path = final_json_path;
    temp_json_path += ".tmp";
    fs::path temp_bin_path = final_bin_path;
    temp_bin_path += ".tmp";

    // 2. Open Streams
    std::ofstream out_json(temp_json_path);
    std::ofstream out_bin(temp_bin_path, std::ios::binary);

    if (!out_json.is_open() || !out_bin.is_open()) {
        SCENE_LOG_ERROR("Failed to create temporary save files.");
        return false;
    }

    try {
        if (progress_callback) progress_callback(5, "Writing geometry...");
        
        // Write geometry to binary file FIRST
        if (save_settings.save_geometry) {
            writeGeometryBinary(out_bin, scene);
        }
        
        if (progress_callback) progress_callback(30, "Writing metadata...");

        // 3. Write JSON 
        json root;
        root["format_version"] = "3.0";
        root["project_name"] = g_project.project_name;
        root["author"] = g_project.author;
        root["description"] = g_project.description;
        root["next_model_id"] = g_project.next_model_id;
        root["next_object_id"] = g_project.next_object_id;
        root["next_texture_id"] = g_project.next_texture_id;
        root["has_geometry"] = save_settings.save_geometry;
        
        // Procedural objects (still saved for non-geometry mode)
        json procedurals_arr = json::array();
        for (const auto& proc : g_project.procedural_objects) {
            json p;
            p["id"] = proc.id;
            p["mesh_type"] = static_cast<int>(proc.mesh_type);
            p["display_name"] = proc.display_name;
            p["transform"] = mat4ToJson(proc.transform);
            p["material_id"] = proc.material_id;
            p["visible"] = proc.visible;
            procedurals_arr.push_back(p);
        }
        root["procedural_objects"] = procedurals_arr;
        
        if (progress_callback) progress_callback(50, "Saving materials...");
        
        // Materials
        auto& mat_mgr = MaterialManager::getInstance();
        root["materials"] = mat_mgr.serialize(fs::path(filepath).parent_path().string());
        
        if (progress_callback) progress_callback(60, "Saving lights...");
        
        // Lights
        root["lights"] = serializeLights(scene.lights);
        
        if (progress_callback) progress_callback(70, "Saving cameras...");
        
        // Cameras
        root["cameras"] = serializeCameras(scene.cameras, scene.active_camera_index);
        
        if (progress_callback) progress_callback(80, "Saving render settings...");
        
        // Render Settings
        root["render_settings"] = serializeRenderSettings(settings);
        
        // Timeline/Animation (keyframes)
        if (progress_callback) progress_callback(82, "Saving animation keyframes...");
        json j_timeline;
        scene.timeline.serialize(j_timeline);
        root["timeline"] = j_timeline;
        
        // BoneData (skeleton/skinning information)
        if (progress_callback) progress_callback(83, "Saving bone data...");
        {
            json j_bones;
            
            // Save bone name to index mapping
            json j_bone_map = json::object();
            for (const auto& [name, idx] : scene.boneData.boneNameToIndex) {
                j_bone_map[name] = idx;
            }
            j_bones["boneNameToIndex"] = j_bone_map;
            
            // Save bone offset matrices
            json j_offsets = json::object();
            for (const auto& [name, matrix] : scene.boneData.boneOffsetMatrices) {
                j_offsets[name] = mat4ToJson(matrix);
            }
            j_bones["boneOffsetMatrices"] = j_offsets;
            
            // Save global inverse transform
            j_bones["globalInverseTransform"] = mat4ToJson(scene.boneData.globalInverseTransform);
            
            root["boneData"] = j_bones;
        }
        
        // AnimationData (bone animation keyframes)
        if (progress_callback) progress_callback(83, "Saving animation data...");
        {
            json j_animations = json::array();
            
            for (const auto& anim : scene.animationDataList) {
                json j_anim;
                j_anim["name"] = anim.name;
                j_anim["duration"] = anim.duration;
                j_anim["ticksPerSecond"] = anim.ticksPerSecond;
                j_anim["startFrame"] = anim.startFrame;
                j_anim["endFrame"] = anim.endFrame;
                
                // Position keys
                json j_pos_keys = json::object();
                for (const auto& [nodeName, keys] : anim.positionKeys) {
                    json j_keys = json::array();
                    for (const auto& key : keys) {
                        j_keys.push_back({
                            {"time", key.mTime},
                            {"x", key.mValue.x},
                            {"y", key.mValue.y},
                            {"z", key.mValue.z}
                        });
                    }
                    j_pos_keys[nodeName] = j_keys;
                }
                j_anim["positionKeys"] = j_pos_keys;
                
                // Rotation keys
                json j_rot_keys = json::object();
                for (const auto& [nodeName, keys] : anim.rotationKeys) {
                    json j_keys = json::array();
                    for (const auto& key : keys) {
                        j_keys.push_back({
                            {"time", key.mTime},
                            {"w", key.mValue.w},
                            {"x", key.mValue.x},
                            {"y", key.mValue.y},
                            {"z", key.mValue.z}
                        });
                    }
                    j_rot_keys[nodeName] = j_keys;
                }
                j_anim["rotationKeys"] = j_rot_keys;
                
                // Scaling keys
                json j_scale_keys = json::object();
                for (const auto& [nodeName, keys] : anim.scalingKeys) {
                    json j_keys = json::array();
                    for (const auto& key : keys) {
                        j_keys.push_back({
                            {"time", key.mTime},
                            {"x", key.mValue.x},
                            {"y", key.mValue.y},
                            {"z", key.mValue.z}
                        });
                    }
                    j_scale_keys[nodeName] = j_keys;
                }
                j_anim["scalingKeys"] = j_scale_keys;
                
                j_animations.push_back(j_anim);
            }
            
            root["animationDataList"] = j_animations;
            SCENE_LOG_INFO("[ProjectManager] Saved " + std::to_string(scene.animationDataList.size()) + " animation clips.");
        }
        
        // Water System
        if (progress_callback) progress_callback(83, "Saving water surfaces...");
        root["water"] = WaterManager::getInstance().serialize();

        // Terrain System
        if (progress_callback) progress_callback(84, "Saving terrain system...");
        auto abs_path = std::filesystem::absolute(filepath);
        std::string terrainDir = abs_path.parent_path().string();
        SCENE_LOG_INFO("[ProjectManager] Saving terrain system to: " + terrainDir);
        root["terrain_system"] = TerrainManager::getInstance().serialize(terrainDir);
        
        // River System
        if (progress_callback) progress_callback(84, "Saving river system...");
        root["rivers"] = RiverManager::getInstance().serialize();
        SCENE_LOG_INFO("[ProjectManager] Saved " + std::to_string(RiverManager::getInstance().getRivers().size()) + " rivers");
        
        // UI Settings (Pro Camera, Viewport, etc.)
        if (!scene.ui_settings_json_str.empty()) {
            try {
                root["ui_settings"] = json::parse(scene.ui_settings_json_str);
            } catch (...) {
                SCENE_LOG_WARN("Failed to parse UI settings JSON for saving.");
            }
        }

        // Textures (with embed option)
        if (progress_callback) progress_callback(85, "Processing textures...");
        root["textures"] = serializeTextures(out_bin, save_settings.embed_textures);
        
        // Write JSON - Use error handler to replace invalid UTF-8 characters
        out_json << root.dump(2, ' ', false, nlohmann::json::error_handler_t::replace);
        
        out_json.flush();
        out_bin.flush();
        out_json.close();
        out_bin.close();
        
        if (progress_callback) progress_callback(95, "Finalizing...");

        // Atomic Commit
        try {
            if (fs::exists(final_json_path)) fs::remove(final_json_path);
            if (fs::exists(final_bin_path)) fs::remove(final_bin_path);
            fs::rename(temp_json_path, final_json_path);
            fs::rename(temp_bin_path, final_bin_path);
        } catch (const fs::filesystem_error& e) {
            SCENE_LOG_ERROR("FileSystem Error during rename: " + std::string(e.what()));
            return false;
        }
        
    } catch (const std::exception& e) {
        SCENE_LOG_ERROR("Save failed: " + std::string(e.what()));
        if (out_json.is_open()) out_json.close();
        if (out_bin.is_open()) out_bin.close();
        if (fs::exists(temp_json_path)) fs::remove(temp_json_path);
        if (fs::exists(temp_bin_path)) fs::remove(temp_bin_path);
        return false;
    }
    
    g_project.current_file_path = filepath;
    g_project.is_modified = false;
    
    if (progress_callback) progress_callback(100, "Done.");
    SCENE_LOG_INFO("Project saved successfully: " + filepath);
    return true;
}

bool ProjectManager::openProject(const std::string& filepath, SceneData& scene,
                                  RenderSettings& settings, Renderer& renderer, 
                                  OptixWrapper* optix_gpu,
                                  std::function<void(int, const std::string&)> progress_callback) {
    
    if (progress_callback) progress_callback(0, "Opening project file...");
    
    // 1. Read JSON Metadata
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

    // 2. Prepare Binary Stream
    std::string bin_path = filepath + ".bin";
    std::ifstream in_bin(bin_path, std::ios::binary);
    bool has_binary = in_bin.is_open();
    
    // Clear current project and scene
    if (progress_callback) progress_callback(5, "Clearing scene...");
    
    // =========================================================================
    // SYNCHRONIZATION: Ensure all GPU operations complete before clearing
    // =========================================================================
    if (optix_gpu) {
        cudaError_t err = cudaDeviceSynchronize();  // Wait for all pending CUDA work
        if (err != cudaSuccess) {
            SCENE_LOG_ERROR("cudaDeviceSynchronize failed during project close: " + std::string(cudaGetErrorString(err)));
            // Attempt to reset device if context is corrupted?
            // cudaDeviceReset(); // Risky as it destroys everything including textures
        }
        
        // Clear any sticky errors
        cudaError_t lastErr = cudaGetLastError();
        if (lastErr != cudaSuccess) {
            SCENE_LOG_ERROR("Clearing sticky CUDA error: " + std::string(cudaGetErrorString(lastErr)));
        }
    }
    
    // Brief wait to allow render threads to fully stop
    // (rendering_stopped_* flags are set by caller before this function)
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    newProject();
    TerrainManager::getInstance().removeAllTerrains(scene);
    scene.clear();
    
    // CRITICAL: Create a temporary placeholder camera immediately after clear()
    // This prevents null pointer crashes if render threads access scene.camera
    // during the brief period before cameras are loaded from project file
    auto temp_camera = std::make_shared<Camera>(
        Vec3(0, 2, 5), Vec3(0, 0, 0), Vec3(0, 1, 0),
        60.0f, 16.0f / 9.0f, 0.0f, 10.0f, 6);
    temp_camera->nodeName = "Loading...";
    scene.addCamera(temp_camera);
    
    renderer.resetCPUAccumulation();
    if (optix_gpu) optix_gpu->resetAccumulation();
    
    try {
        // Load metadata
        // Load format version safely (handle potential number/string mismatch from prev versions)
        if (root.contains("format_version")) {
             if (root["format_version"].is_number()) {
                 g_project.format_version = std::to_string(root["format_version"].get<float>()).substr(0, 3); // "2.0"
             } else {
                 g_project.format_version = root.value("format_version", "2.0");
             }
        } else {
             g_project.format_version = "2.0";
        }
        g_project.project_name = root.value("project_name", "Untitled");
        g_project.author = root.value("author", "");
        g_project.description = root.value("description", "");
        g_project.next_model_id = root.value("next_model_id", 1);
        g_project.next_object_id = root.value("next_object_id", 1);
        g_project.next_texture_id = root.value("next_texture_id", 1);
        
        fs::path project_folder = fs::path(filepath).parent_path();
        
        // Check if this is v3.0 format (self-contained geometry)
        bool is_v3 = (g_project.format_version == "3.0" || g_project.format_version == "3.0");
        bool has_geometry = root.value("has_geometry", false);
        
        if (is_v3 && has_geometry && has_binary) {
            // ============================================================
            // V3.0 FORMAT: Load geometry from binary
            // ============================================================
            if (progress_callback) progress_callback(10, "Loading geometry...");

            if (!readGeometryBinary(in_bin, scene)) {
                SCENE_LOG_ERROR("Failed to read geometry from binary file.");
                return false;
            }

            // Load Textures FIRST (so materials can find them)
            if (root.contains("textures")) {
                if (progress_callback) progress_callback(35, "Restoring textures...");
                deserializeTextures(root["textures"], in_bin, project_folder.string());
            }

            if (progress_callback) progress_callback(40, "Loading materials...");

            // Load Materials
            if (root.contains("materials")) {
                MaterialManager::getInstance().deserialize(root["materials"], project_folder.string());
                // NOTE: GPU material update moved to AFTER rebuildOptiXGeometry (line ~505)
                // to ensure SBT records are valid before material buffer update.
            }

            if (progress_callback) progress_callback(50, "Loading lights...");

            // Load Lights
            if (root.contains("lights")) {
                deserializeLights(root["lights"], scene.lights);
            }

            if (progress_callback) progress_callback(60, "Loading cameras...");

            // Load Cameras
            if (root.contains("cameras")) {
                deserializeCameras(root["cameras"], scene);
            }
            else {
                // Create default camera if none exists
                auto default_cam = std::make_shared<Camera>(
                    Vec3(0, 2, 5), Vec3(0, 0, 0), Vec3(0, 1, 0),
                    60.0f, 16.0f / 9.0f, 0.0f, 10.0f, 10);
                default_cam->nodeName = "Default Camera";
                scene.addCamera(default_cam);
            }

            if (progress_callback) progress_callback(70, "Loading render settings...");

            // Load Render Settings
            if (root.contains("render_settings")) {
                deserializeRenderSettings(root["render_settings"], settings);
            }

            // Load Water Surfaces
            if (root.contains("water")) {
                WaterManager::getInstance().deserialize(root["water"], scene);
            }

            // Load Terrain System (Must be before Timeline so tracks can find terrains)
            if (root.contains("terrain_system")) {
                if (progress_callback) progress_callback(75, "Loading terrain system...");
                auto abs_path = std::filesystem::absolute(filepath);
                std::string terrainDir = abs_path.parent_path().string();
                SCENE_LOG_INFO("[ProjectManager] Loading terrain system from: " + terrainDir);
                TerrainManager::getInstance().deserialize(root["terrain_system"], terrainDir, scene);
            }
            
            // Load River System (After terrain so rivers can follow terrain)
            if (root.contains("rivers")) {
                if (progress_callback) progress_callback(76, "Loading river system...");
                RiverManager::getInstance().clear();
                RiverManager::getInstance().deserialize(root["rivers"], scene);
                SCENE_LOG_INFO("[ProjectManager] Loaded " + std::to_string(RiverManager::getInstance().getRivers().size()) + " rivers");
            }
            
            // Load Timeline/Animation (keyframes) - AFTER Objects are created
            if (root.contains("timeline")) {
                scene.timeline.deserialize(root["timeline"]);
                SCENE_LOG_INFO("[ProjectManager] Loaded timeline with " + std::to_string(scene.timeline.tracks.size()) + " tracks.");
            }
            
            // Load BoneData (skeleton/skinning information)
            if (root.contains("boneData")) {
                if (progress_callback) progress_callback(78, "Loading bone data...");
                auto& j_bones = root["boneData"];
                
                // Clear existing bone data
                scene.boneData.clear();
                
                // Load bone name to index mapping
                if (j_bones.contains("boneNameToIndex")) {
                    for (auto& [name, idx] : j_bones["boneNameToIndex"].items()) {
                        scene.boneData.boneNameToIndex[name] = idx.get<int>();
                    }
                }
                
                // Load bone offset matrices
                if (j_bones.contains("boneOffsetMatrices")) {
                    for (auto& [name, mat] : j_bones["boneOffsetMatrices"].items()) {
                        scene.boneData.boneOffsetMatrices[name] = jsonToMat4(mat);
                    }
                }
                
                // Load global inverse transform
                if (j_bones.contains("globalInverseTransform")) {
                    scene.boneData.globalInverseTransform = jsonToMat4(j_bones["globalInverseTransform"]);
                }
                
                // Rebuild reverse lookup for O(1) access
                scene.boneData.rebuildReverseLookup();
                
                SCENE_LOG_INFO("[ProjectManager] Loaded bone data: " + std::to_string(scene.boneData.boneNameToIndex.size()) + " bones.");
            }
            
            // Load UI Settings
            if (root.contains("ui_settings")) {
                scene.ui_settings_json_str = root["ui_settings"].dump();
                scene.load_counter++; // Signal SceneUI to reload settings
            }
            
            // Load AnimationData (bone animation keyframes)
            if (root.contains("animationDataList")) {
                if (progress_callback) progress_callback(79, "Loading animation data...");
                scene.animationDataList.clear();
                
                for (const auto& j_anim : root["animationDataList"]) {
                    AnimationData anim;
                    anim.name = j_anim.value("name", "");
                    anim.duration = j_anim.value("duration", 0.0);
                    anim.ticksPerSecond = j_anim.value("ticksPerSecond", 24.0);
                    anim.startFrame = j_anim.value("startFrame", 0);
                    anim.endFrame = j_anim.value("endFrame", 0);
                    
                    // Position keys
                    if (j_anim.contains("positionKeys")) {
                        for (auto& [nodeName, j_keys] : j_anim["positionKeys"].items()) {
                            std::vector<aiVectorKey> keys;
                            for (const auto& k : j_keys) {
                                aiVectorKey key;
                                key.mTime = k.value("time", 0.0);
                                key.mValue.x = k.value("x", 0.0f);
                                key.mValue.y = k.value("y", 0.0f);
                                key.mValue.z = k.value("z", 0.0f);
                                keys.push_back(key);
                            }
                            anim.positionKeys[nodeName] = keys;
                        }
                    }
                    
                    // Rotation keys
                    if (j_anim.contains("rotationKeys")) {
                        for (auto& [nodeName, j_keys] : j_anim["rotationKeys"].items()) {
                            std::vector<aiQuatKey> keys;
                            for (const auto& k : j_keys) {
                                aiQuatKey key;
                                key.mTime = k.value("time", 0.0);
                                key.mValue.w = k.value("w", 1.0f);
                                key.mValue.x = k.value("x", 0.0f);
                                key.mValue.y = k.value("y", 0.0f);
                                key.mValue.z = k.value("z", 0.0f);
                                keys.push_back(key);
                            }
                            anim.rotationKeys[nodeName] = keys;
                        }
                    }
                    
                    // Scaling keys
                    if (j_anim.contains("scalingKeys")) {
                        for (auto& [nodeName, j_keys] : j_anim["scalingKeys"].items()) {
                            std::vector<aiVectorKey> keys;
                            for (const auto& k : j_keys) {
                                aiVectorKey key;
                                key.mTime = k.value("time", 0.0);
                                key.mValue.x = k.value("x", 1.0f);
                                key.mValue.y = k.value("y", 1.0f);
                                key.mValue.z = k.value("z", 1.0f);
                                keys.push_back(key);
                            }
                            anim.scalingKeys[nodeName] = keys;
                        }
                    }
                    
                    scene.animationDataList.push_back(anim);
                }
                
                SCENE_LOG_INFO("[ProjectManager] Loaded " + std::to_string(scene.animationDataList.size()) + " animation clips.");
            }

            // Load Procedural Objects (still supported)
            if (root.contains("procedural_objects")) {
                for (const auto& p : root["procedural_objects"]) {
                    ProceduralObjectData proc;
                    proc.id = p.value("id", 0);
                    proc.mesh_type = static_cast<ProceduralMeshType>(p.value("mesh_type", 0));
                    proc.display_name = p.value("display_name", "Procedural");
                    if (p.contains("transform")) proc.transform = jsonToMat4(p["transform"]);
                    proc.material_id = p.value("material_id", 0);
                    proc.visible = p.value("visible", true);
                    g_project.procedural_objects.push_back(proc);

                    // Create procedural geometry...
                    // (Same logic as legacy, but can be skipped if already in geometry binary)
                }

            }
            else {
                // No geometry in binary - cannot load
                SCENE_LOG_ERROR("Project file is in legacy format. Please re-import your models.");
                return false;
            }
        }
    } catch (const std::exception& e) {
        SCENE_LOG_ERROR("Error during project loading: " + std::string(e.what()));
        return false;
    }
    
    if (in_bin.is_open()) in_bin.close();
    
    g_project.current_file_path = filepath;
    g_project.is_modified = false;
    
    if (progress_callback) progress_callback(90, "Preparing for GPU sync...");

    // Mark scene as ID-Ready before rebuilding
    scene.initialized = true;
    
    // =========================================================================
    // THREAD SAFETY: DO NOT call OptiX rebuild from thread!
    // Set flags and let Main loop handle GPU operations on main thread.
    // =========================================================================
    g_needs_geometry_rebuild = true;
    g_needs_optix_sync = (optix_gpu != nullptr);
    
    // Mark dirty flags for complete refresh
    g_camera_dirty = true;
    g_lights_dirty = true;
    g_world_dirty = true;

    // Force initial camera update to GPU to prevent stale camera glitch
    // This ensures the renderer uses the newly loaded camera immediately, preventing the 'default camera' flash
    if (scene.camera) {
        scene.camera->update_camera_vectors();
        if (optix_gpu) optix_gpu->setCameraParams(*scene.camera);
    }
    
    if (progress_callback) progress_callback(100, "Done.");
    SCENE_LOG_INFO("Project loaded: " + filepath + " (GPU rebuild pending on main thread)");
    return true;
}

// ============================================================================
// Asset Import
// ============================================================================

bool ProjectManager::importModel(const std::string& filepath, SceneData& scene,
                                  Renderer& renderer, OptixWrapper* optix_gpu,
                                  std::function<void(int, const std::string&)> progress_callback,
                                  bool rebuild) {
    if (!fs::exists(filepath)) {
        SCENE_LOG_ERROR("File not found: " + filepath);
        return false;
    }
    
    if (progress_callback) progress_callback(0, "Preparing import...");

    // Generate package path and unique ID
    uint32_t id = g_project.generateModelId();
    std::string package_path = generatePackagePath(filepath, "models", id);
    std::string import_prefix = std::to_string(id) + "_";
    
    ImportedModelData model;
    model.id = id;
    model.original_path = filepath;
    model.package_path = package_path;
    model.display_name = fs::path(filepath).stem().string();
    
    // NOTE: In v3.0+ format, geometry is embedded in the binary file (.bin)
    // so we NO LONGER copy the original model file to the project folder.
    // This saves disk space and avoids confusion.
    // The original_path is kept for reference only (e.g., to show source).

    
    size_t objects_before = scene.world.objects.size();
    
    if (progress_callback) progress_callback(20, "Parsing model geometry...");

    // Load with Assimp - append mode (don't clear existing scene)
    // AssimpLoader usually just takes time, ideally we'd pass progress callback into it too
    renderer.create_scene(scene, optix_gpu, filepath, progress_callback, true);  // append = true
    
    if (progress_callback) progress_callback(80, "Processing objects...");

    // Track new objects and add unique prefix to prevent name collision
    for (size_t i = objects_before; i < scene.world.objects.size(); ++i) {
        auto tri = std::dynamic_pointer_cast<Triangle>(scene.world.objects[i]);
        if (tri) {
            std::string original_name = tri->nodeName;
            std::string unique_name = import_prefix + original_name;
            tri->setNodeName(unique_name);
            
            ImportedModelData::ObjectInstance inst;
            inst.node_name = unique_name;
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
    
    // AUTO-ACTIVATE CAMERA: If imported model has cameras, set first new one as active
    if (!scene.cameras.empty()) {
        // Find if we have new cameras (cameras added during this import)
        // Simple approach: if active camera is default and we have imported cameras, activate first imported
        // Better: check if camera count increased
        size_t camera_count = scene.cameras.size();
        if (camera_count > 1 && scene.active_camera_index == 0) {
            // Switch to the newly imported camera (likely the last one)
            scene.setActiveCamera(camera_count - 1);
            SCENE_LOG_INFO("Auto-activated imported camera: Camera #" + std::to_string(camera_count - 1));
        } else if (camera_count == 1 && scene.camera) {
            // Only one camera exists, ensure it's active
            scene.setActiveCamera(0);
        }
    }
    
    if (rebuild) {
        if (progress_callback) progress_callback(90, "Rebuilding BVH...");
        extern RenderSettings render_settings;  // From globals or Main.cpp
        renderer.rebuildBVH(scene, render_settings.UI_use_embree);
        renderer.resetCPUAccumulation();
        
        if (optix_gpu) {
            if (progress_callback) progress_callback(95, "Uploading to GPU...");
            renderer.rebuildOptiXGeometry(scene, optix_gpu);
            optix_gpu->setLightParams(scene.lights);
            if (scene.camera) {
                optix_gpu->setCameraParams(*scene.camera);
            }
            optix_gpu->resetAccumulation();
        }
    }
    
    if (progress_callback) progress_callback(100, "Done.");
    SCENE_LOG_INFO("Model imported: " + model.display_name + " (ID: " + std::to_string(id) + ", " + std::to_string(model.objects.size()) + " objects)");
    return true;
}

// ... Rest of the file (procedural, textures, etc.) ...
// Just keeping the rest as is, but ensuring the file is complete.
// Since I'm using write_to_file, I must include EVERYTHING.

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
    
    g_project.procedural_objects.erase(it);
    g_project.is_modified = true;
    
    return true;
}

std::vector<uint8_t> ProjectManager::extractFileFromPackage(const std::string& internal_path) {
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
    if (!g_project.current_file_path.empty()) {
        fs::path project_folder = fs::path(g_project.current_file_path).parent_path();
        return fs::exists(project_folder / internal_path);
    }
    return false;
}

bool ProjectManager::writeZipPackage(const std::string& filepath) {
    SCENE_LOG_WARN("ZIP packaging not yet implemented. Save using regular project format.");
    return false; // Not implemented
}

bool ProjectManager::readZipPackage(const std::string& filepath) {
    SCENE_LOG_WARN("ZIP packaging not yet implemented.");
    return false;
}

// ============================================================================
// GEOMETRY SERIALIZATION (Self-Contained Project)
// ============================================================================

// Binary format magic number and version (v4 adds skinning data)
static constexpr char RTP_MAGIC[4] = {'R', 'T', 'P', '4'};
static constexpr uint32_t RTP_VERSION = 4;

bool ProjectManager::writeGeometryBinary(std::ofstream& out, const SceneData& scene) {
    // Write header
    out.write(RTP_MAGIC, 4);
    uint32_t version = RTP_VERSION;
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    
    // Collect all triangles
    std::vector<std::shared_ptr<Triangle>> triangles;
    std::unordered_map<std::shared_ptr<Transform>, uint32_t> transform_map;
    std::vector<std::shared_ptr<Transform>> transforms;
    
    // FILTERING: Identify terrain triangles to exclude from binary geometry
    std::unordered_set<std::shared_ptr<Triangle>> terrain_triangles;
    auto& terrains = TerrainManager::getInstance().getTerrains();
    for (auto& t : terrains) {
        for (auto& tri : t.mesh_triangles) {
            terrain_triangles.insert(tri);
        }
    }

    for (const auto& obj : scene.world.objects) {
        if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
            // Skip terrain chunks
            if (terrain_triangles.count(tri)) continue;

            triangles.push_back(tri);
            
            // Track unique transforms
            auto th = tri->getTransformHandle();
            if (th && transform_map.find(th) == transform_map.end()) {
                transform_map[th] = static_cast<uint32_t>(transforms.size());
                transforms.push_back(th);
            }
        }
    }
    
    // Write transform count and data
    uint32_t transform_count = static_cast<uint32_t>(transforms.size());
    out.write(reinterpret_cast<const char*>(&transform_count), sizeof(transform_count));
    
    for (const auto& tr : transforms) {
        out.write(reinterpret_cast<const char*>(&tr->base), sizeof(Matrix4x4));
    }
    
    // Write triangle count
    uint32_t tri_count = static_cast<uint32_t>(triangles.size());
    out.write(reinterpret_cast<const char*>(&tri_count), sizeof(tri_count));
    
    // Write each triangle
    for (const auto& tri : triangles) {
        // Vertices (original, not transformed)
        Vec3 v0 = tri->getOriginalVertexPosition(0);
        Vec3 v1 = tri->getOriginalVertexPosition(1);
        Vec3 v2 = tri->getOriginalVertexPosition(2);
        out.write(reinterpret_cast<const char*>(&v0), sizeof(Vec3));
        out.write(reinterpret_cast<const char*>(&v1), sizeof(Vec3));
        out.write(reinterpret_cast<const char*>(&v2), sizeof(Vec3));
        
        // Normals
        Vec3 n0 = tri->getOriginalVertexNormal(0);
        Vec3 n1 = tri->getOriginalVertexNormal(1);
        Vec3 n2 = tri->getOriginalVertexNormal(2);
        out.write(reinterpret_cast<const char*>(&n0), sizeof(Vec3));
        out.write(reinterpret_cast<const char*>(&n1), sizeof(Vec3));
        out.write(reinterpret_cast<const char*>(&n2), sizeof(Vec3));
        
        // UVs
        Vec2 uv0 = tri->t0;
        Vec2 uv1 = tri->t1;
        Vec2 uv2 = tri->t2;
        out.write(reinterpret_cast<const char*>(&uv0), sizeof(Vec2));
        out.write(reinterpret_cast<const char*>(&uv1), sizeof(Vec2));
        out.write(reinterpret_cast<const char*>(&uv2), sizeof(Vec2));
        
        // Material ID
        uint16_t mat_id = tri->getMaterialID();
        out.write(reinterpret_cast<const char*>(&mat_id), sizeof(mat_id));
        
        // Transform index
        uint32_t tr_idx = 0xFFFFFFFF; // No transform
        auto th = tri->getTransformHandle();
        if (th) {
            auto it = transform_map.find(th);
            if (it != transform_map.end()) {
                tr_idx = it->second;
            }
        }
        out.write(reinterpret_cast<const char*>(&tr_idx), sizeof(tr_idx));
        
        // Node name (length-prefixed string)
        writeStringBinary(out, tri->nodeName);
        
        // Skinning data (v4+)
        uint8_t has_skin = tri->hasSkinData() ? 1 : 0;
        out.write(reinterpret_cast<const char*>(&has_skin), sizeof(has_skin));
        
        if (has_skin) {
            // Write bone weights for each vertex (3 vertices)
            for (int vi = 0; vi < 3; ++vi) {
                const auto& weights = tri->getSkinBoneWeights(vi);
                uint8_t weight_count = static_cast<uint8_t>(std::min(weights.size(), (size_t)255));
                out.write(reinterpret_cast<const char*>(&weight_count), sizeof(weight_count));
                
                for (size_t w = 0; w < weight_count; ++w) {
                    int32_t bone_idx = weights[w].first;
                    float weight = weights[w].second;
                    out.write(reinterpret_cast<const char*>(&bone_idx), sizeof(bone_idx));
                    out.write(reinterpret_cast<const char*>(&weight), sizeof(weight));
                }
            }
        }
    }
    
    SCENE_LOG_INFO("[ProjectManager] Wrote " + std::to_string(tri_count) + " triangles, " + 
                   std::to_string(transform_count) + " transforms to binary.");
    return true;
}

bool ProjectManager::readGeometryBinary(std::ifstream& in, SceneData& scene) {
    // Read and validate header (accept both RTP3 and RTP4 formats)
    char magic[4];
    in.read(magic, 4);
    
    // Check for valid magic (RTP3 or RTP4)
    bool is_v3 = (magic[0] == 'R' && magic[1] == 'T' && magic[2] == 'P' && magic[3] == '3');
    bool is_v4 = (magic[0] == 'R' && magic[1] == 'T' && magic[2] == 'P' && magic[3] == '4');
    
    if (!is_v3 && !is_v4) {
        SCENE_LOG_ERROR("[ProjectManager] Invalid geometry file format.");
        return false;
    }
    
    uint32_t version;
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version > RTP_VERSION) {
        SCENE_LOG_WARN("[ProjectManager] Newer geometry format (v" + std::to_string(version) + "), some features may not load.");
    }
    
    // Read transforms
    uint32_t transform_count;
    in.read(reinterpret_cast<char*>(&transform_count), sizeof(transform_count));
    
    std::vector<std::shared_ptr<Transform>> transforms;
    transforms.reserve(transform_count);
    
    for (uint32_t i = 0; i < transform_count; ++i) {
        auto tr = std::make_shared<Transform>();
        in.read(reinterpret_cast<char*>(&tr->base), sizeof(Matrix4x4));
        transforms.push_back(tr);
    }
    
    // Read triangles
    uint32_t tri_count;
    in.read(reinterpret_cast<char*>(&tri_count), sizeof(tri_count));
    
    scene.world.objects.reserve(scene.world.objects.size() + tri_count);
    
    for (uint32_t i = 0; i < tri_count; ++i) {
        // Vertices
        Vec3 v0, v1, v2;
        in.read(reinterpret_cast<char*>(&v0), sizeof(Vec3));
        in.read(reinterpret_cast<char*>(&v1), sizeof(Vec3));
        in.read(reinterpret_cast<char*>(&v2), sizeof(Vec3));
        
        // Normals
        Vec3 n0, n1, n2;
        in.read(reinterpret_cast<char*>(&n0), sizeof(Vec3));
        in.read(reinterpret_cast<char*>(&n1), sizeof(Vec3));
        in.read(reinterpret_cast<char*>(&n2), sizeof(Vec3));
        
        // UVs
        Vec2 uv0, uv1, uv2;
        in.read(reinterpret_cast<char*>(&uv0), sizeof(Vec2));
        in.read(reinterpret_cast<char*>(&uv1), sizeof(Vec2));
        in.read(reinterpret_cast<char*>(&uv2), sizeof(Vec2));
        
        // Material ID
        uint16_t mat_id;
        in.read(reinterpret_cast<char*>(&mat_id), sizeof(mat_id));
        
        // Transform index
        uint32_t tr_idx;
        in.read(reinterpret_cast<char*>(&tr_idx), sizeof(tr_idx));
        
        // Node name
        std::string node_name = readStringBinary(in);
        
        // Create triangle
        auto tri = std::make_shared<Triangle>(v0, v1, v2, n0, n1, n2, uv0, uv1, uv2, mat_id);
        tri->setNodeName(node_name);
        
        // Read skinning data (v4+)
        bool has_skin_data = false;
        if (version >= 4) {
            uint8_t has_skin;
            in.read(reinterpret_cast<char*>(&has_skin), sizeof(has_skin));
            has_skin_data = (has_skin != 0);
            
            if (has_skin_data) {
                tri->initializeSkinData();
                
                // Read bone weights for each vertex
                for (int vi = 0; vi < 3; ++vi) {
                    uint8_t weight_count;
                    in.read(reinterpret_cast<char*>(&weight_count), sizeof(weight_count));
                    
                    std::vector<std::pair<int, float>> weights;
                    weights.reserve(weight_count);
                    
                    for (uint8_t w = 0; w < weight_count; ++w) {
                        int32_t bone_idx;
                        float weight;
                        in.read(reinterpret_cast<char*>(&bone_idx), sizeof(bone_idx));
                        in.read(reinterpret_cast<char*>(&weight), sizeof(weight));
                        weights.emplace_back(bone_idx, weight);
                    }
                    
                    tri->setSkinBoneWeights(vi, weights);
                }
            }
        }
        
        // Assign transform
        if (tr_idx < transforms.size()) {
            tri->setTransformHandle(transforms[tr_idx]);
            
            // CRITICAL FIX: DO NOT call updateTransformedVertices for skinned meshes!
            // Skinned mesh vertices are computed by the animation system using bone matrices.
            // Applying the object transform here would cause double transformation or incorrect positioning.
            // Non-skinned meshes still need their transforms applied.
            if (!has_skin_data) {
                tri->updateTransformedVertices();
            }
            // Note: Skinned meshes will be updated when animation system runs
        }
        
        tri->update_bounding_box();
        scene.world.objects.push_back(tri);
    }
    
    SCENE_LOG_INFO("[ProjectManager] Loaded " + std::to_string(tri_count) + " triangles, " + 
                   std::to_string(transform_count) + " transforms from binary.");
    return true;
}

// ============================================================================
// LIGHT SERIALIZATION
// ============================================================================

json ProjectManager::serializeLights(const std::vector<std::shared_ptr<Light>>& lights) {
    json arr = json::array();
    
    for (size_t i = 0; i < lights.size(); ++i) {
        const auto& light = lights[i];
        json l;
        
        l["id"] = i;
        l["type"] = static_cast<int>(light->type());
        l["position"] = vec3ToJson(light->position);
        l["direction"] = vec3ToJson(light->direction);
        l["color"] = vec3ToJson(light->color);
        l["intensity"] = light->intensity;
        l["radius"] = light->radius;
        
        // Spot light specific
        if (light->type() == LightType::Spot) {
            auto spot = std::dynamic_pointer_cast<SpotLight>(light);
            if (spot) {
                l["angle"] = spot->getAngleDegrees();
                l["falloff"] = spot->getFalloff();
            }
        }
        
        // Area light specific
        if (light->type() == LightType::Area) {
            l["width"] = light->width;
            l["height"] = light->height;
        }
        
        arr.push_back(l);
    }
    
    return arr;
}

void ProjectManager::deserializeLights(const json& j, std::vector<std::shared_ptr<Light>>& lights) {
    lights.clear();
    
    for (const auto& l : j) {
        LightType type = static_cast<LightType>(l.value("type", 0));
        
        Vec3 position = jsonToVec3(l.value("position", json::array({0, 5, 0})));
        Vec3 direction = jsonToVec3(l.value("direction", json::array({0, -1, 0})));
        Vec3 color = jsonToVec3(l.value("color", json::array({1, 1, 1})));
        float intensity = l.value("intensity", 100.0f);
        float radius = l.value("radius", 0.01f);
        
        std::shared_ptr<Light> light;
        
        switch (type) {
            case LightType::Point: {
                auto pl = std::make_shared<PointLight>(position, color * intensity, radius);
                light = pl;
                break;
            }
            case LightType::Directional: {
                auto dl = std::make_shared<DirectionalLight>(direction, color * intensity, radius);
                dl->position = position; // Very distant, but use position for reference
                light = dl;
                break;
            }
            case LightType::Spot: {
                float angle = l.value("angle", 45.0f);
                auto sl = std::make_shared<SpotLight>(position, direction, color * intensity, angle, radius);
                sl->setFalloff(l.value("falloff", 0.1f));
                light = sl;
                break;
            }
            case LightType::Area: {
                float width = l.value("width", 1.0f);
                float height = l.value("height", 1.0f);
                Vec3 u_vec(1, 0, 0);
                Vec3 v_vec(0, 0, 1);
                auto al = std::make_shared<AreaLight>(position, u_vec, v_vec, width, height, color * intensity);
                light = al;
                break;
            }
            default:
                SCENE_LOG_WARN("[ProjectManager] Unknown light type: " + std::to_string(static_cast<int>(type)));
                continue;
        }
        
        if (light) {
            light->nodeName = l.value("name", "Light_" + std::to_string(lights.size()));
            lights.push_back(light);
        }
    }
    
    SCENE_LOG_INFO("[ProjectManager] Loaded " + std::to_string(lights.size()) + " lights.");
}

// ============================================================================
// CAMERA SERIALIZATION
// ============================================================================

json ProjectManager::serializeCameras(const std::vector<std::shared_ptr<Camera>>& cameras, size_t active_index) {
    json arr = json::array();
    
    for (size_t i = 0; i < cameras.size(); ++i) {
        const auto& cam = cameras[i];
        json c;
        
        c["id"] = i;
        c["name"] = cam->nodeName.empty() ? ("Camera_" + std::to_string(i)) : cam->nodeName;
        c["position"] = vec3ToJson(cam->lookfrom);
        c["target"] = vec3ToJson(cam->lookat);
        c["up"] = vec3ToJson(cam->vup);
        c["fov"] = cam->vfov;
        c["aperture"] = cam->aperture;
        c["focus_dist"] = cam->focus_dist;
        c["is_active"] = (i == active_index);
        
        // Extended Camera Parameters
        c["iso"] = cam->iso_preset_index;
        c["shutter"] = cam->shutter_preset_index;
        c["fstop_idx"] = cam->fstop_preset_index;
        c["auto_exposure"] = cam->auto_exposure;
        c["ev_comp"] = cam->ev_compensation;
        c["output_aspect_idx"] = cam->output_aspect_index;
        c["use_physical_lens"] = cam->use_physical_lens;
        c["focal_length_mm"] = cam->focal_length_mm;
        c["sensor_width_mm"] = cam->sensor_width_mm;
        c["enable_motion_blur"] = cam->enable_motion_blur;
        c["rig_mode"] = static_cast<int>(cam->rig_mode);
        c["dolly_pos"] = cam->dolly_position;
        c["lens_radius"] = cam->lens_radius;
        c["blade_count"] = cam->blade_count;
        c["distortion"] = cam->distortion;
        c["body_preset_index"] = cam->body_preset_index;
        c["iso_val"] = cam->iso;
        c["sensor_height_mm"] = cam->sensor_height_mm;
        
        arr.push_back(c);
    }
    
    return arr;
}

void ProjectManager::deserializeCameras(const json& j, SceneData& scene) {
    scene.cameras.clear();
    size_t active_idx = 0;
    
    for (const auto& c : j) {
        Vec3 position = jsonToVec3(c.value("position", json::array({0, 2, 5})));
        Vec3 target = jsonToVec3(c.value("target", json::array({0, 0, 0})));
        Vec3 up = jsonToVec3(c.value("up", json::array({0, 1, 0})));
        float fov = c.value("fov", 60.0f);
        float aperture = c.value("aperture", 0.0f);
        float focus_dist = c.value("focus_dist", 10.0f);
        int blade_count = c.value("blade_count", 6);
        
        // Get aspect ratio from render settings (or default 16:9)
        float aspect = 16.0f / 9.0f;
        
        auto cam = std::make_shared<Camera>(position, target, up, fov, aspect, aperture, focus_dist, blade_count);
        cam->nodeName = c.value("name", "Camera");
        
        // Restore Extended Parameters
        cam->iso_preset_index = c.value("iso", 1);
        cam->shutter_preset_index = c.value("shutter", 1);
        cam->fstop_preset_index = c.value("fstop_idx", 4);
        cam->auto_exposure = c.value("auto_exposure", true);
        cam->ev_compensation = c.value("ev_comp", 0.0f);
        cam->output_aspect_index = c.value("output_aspect_idx", 2);
        cam->use_physical_lens = c.value("use_physical_lens", false);
        cam->focal_length_mm = c.value("focal_length_mm", 50.0f);
        cam->sensor_width_mm = c.value("sensor_width_mm", 36.0f);
        cam->enable_motion_blur = c.value("enable_motion_blur", false);
        cam->rig_mode = static_cast<Camera::RigMode>(c.value("rig_mode", 0));
        cam->dolly_position = c.value("dolly_pos", 0.0f);
        cam->lens_radius = c.value("lens_radius", aperture * 0.5f);
        cam->lens_radius = c.value("lens_radius", aperture * 0.5f);
        cam->distortion = c.value("distortion", 0.0f);
        cam->body_preset_index = c.value("body_preset_index", 1); // Default to Full Frame (index 1)
        cam->iso = c.value("iso_val", 100);
        cam->sensor_height_mm = c.value("sensor_height_mm", 24.0f);
        
        scene.cameras.push_back(cam);
        
        if (c.value("is_active", false)) {
            active_idx = scene.cameras.size() - 1;
        }
    }
    
    if (!scene.cameras.empty()) {
        scene.setActiveCamera(active_idx);
    }
    
    SCENE_LOG_INFO("[ProjectManager] Loaded " + std::to_string(scene.cameras.size()) + " cameras.");
}

// ============================================================================
// RENDER SETTINGS SERIALIZATION
// ============================================================================

json ProjectManager::serializeRenderSettings(const RenderSettings& settings) {
    json j;
    
    j["samples_per_pixel"] = settings.samples_per_pixel;
    j["max_bounces"] = settings.max_bounces;
    j["final_render_width"] = settings.final_render_width;
    j["final_render_height"] = settings.final_render_height;
    j["use_gpu"] = settings.use_optix;
    j["use_embree"] = settings.UI_use_embree;
    j["use_denoiser"] = settings.use_denoiser;
    j["render_use_denoiser"] = settings.render_use_denoiser;
    j["max_samples"] = settings.max_samples;
    j["min_samples"] = settings.min_samples;
    j["use_adaptive_sampling"] = settings.use_adaptive_sampling;
    j["variance_threshold"] = settings.variance_threshold;
    
    return j;
}

void ProjectManager::deserializeRenderSettings(const json& j, RenderSettings& settings) {
    settings.samples_per_pixel = j.value("samples_per_pixel", 1);
    settings.max_bounces = j.value("max_bounces", 10);
    settings.final_render_width = j.value("final_render_width", 1280);
    settings.final_render_height = j.value("final_render_height", 720);
    settings.use_optix = j.value("use_gpu", false);
    settings.UI_use_embree = j.value("use_embree", true);
    settings.use_denoiser = j.value("use_denoiser", false);
    settings.render_use_denoiser = j.value("render_use_denoiser", true);
    settings.max_samples = j.value("max_samples", 32);
    settings.min_samples = j.value("min_samples", 1);
    settings.use_adaptive_sampling = j.value("use_adaptive_sampling", true);
    settings.variance_threshold = j.value("variance_threshold", 0.1f);
    
    SCENE_LOG_INFO("[ProjectManager] Loaded render settings.");
}

// ============================================================================
// TEXTURE SERIALIZATION (Path or Embed)
// ============================================================================

json ProjectManager::serializeTextures(std::ofstream& bin_out, bool embed_textures) {
    json arr = json::array();
    
    // Get all textures from MaterialManager
    auto& mgr = MaterialManager::getInstance();
    std::unordered_map<std::string, uint32_t> texture_map; // path -> id
    uint32_t texture_id = 0;
    
    SCENE_LOG_INFO("Starting texture serialization for " + std::to_string(mgr.getMaterialCount()) + " materials.");

    for (size_t i = 0; i < mgr.getMaterialCount(); ++i) {
        auto mat = mgr.getMaterial(static_cast<uint16_t>(i));
        if (!mat) continue;

        // CRITICAL FIX: Must cast to PrincipledBSDF to access texture properties correctly!
        // Material base class may have different/shadowed albedoProperty member
        if (mat->type() != MaterialType::PrincipledBSDF) continue;
        
        PrincipledBSDF* pbsdf = dynamic_cast<PrincipledBSDF*>(mat);
        if (!pbsdf) continue;

        if (pbsdf->albedoProperty.texture) 
            SCENE_LOG_INFO("DEBUG: Mat " + mat->materialName + " HAS Albedo Tex: " + pbsdf->albedoProperty.texture->name);
        else 
            SCENE_LOG_INFO("DEBUG: Mat " + mat->materialName + " has NO Albedo Tex");
        
        // Check all texture properties - use PrincipledBSDF pointer for correct member access
        auto checkTexture = [&](MaterialProperty& prop, const std::string& usage) {
            if (prop.texture && prop.texture->is_loaded()) {
                std::string path = prop.texture->name;
                
                // Fallback for unnamed textures (embedded or generated)
                if (path.empty()) {
                    path = "texture_" + std::to_string(texture_id) + ".png"; // Default name
                    SCENE_LOG_WARN("Texture has no name, using fallback: " + path);
                }

                // Skip if already processed
                if (texture_map.find(path) != texture_map.end()) return;
                
                json tex_entry;
                tex_entry["id"] = texture_id;
                tex_entry["usage"] = usage;
                tex_entry["width"] = prop.texture->width;
                tex_entry["height"] = prop.texture->height;
                
                bool should_embed = embed_textures;
                if (save_settings.embed_missing_only && !fs::exists(path)) {
                    should_embed = true;
                }
                
                // Name fallback override for embedding
                if (prop.texture->name.empty() || prop.texture->name.find("embedded_") == 0) {
                     should_embed = true; // Must embed if no file exists or it's an internal embedded texture
                }

                tex_entry["original_name"] = path; // Save Original Name for restoration

                if (should_embed) {
                    // Embed texture data
                    tex_entry["mode"] = "embed";
                    tex_entry["offset"] = static_cast<long long>(bin_out.tellp());
                    
                    // DEBUG: Log state for embedded texture serialization
                    SCENE_LOG_INFO("Embedding texture: " + path + 
                        " | pixels.empty()=" + std::to_string(prop.texture->pixels.empty()) +
                        " | float_pixels.empty()=" + std::to_string(prop.texture->float_pixels.empty()) +
                        " | width=" + std::to_string(prop.texture->width) +
                        " | height=" + std::to_string(prop.texture->height) +
                        " | is_loaded=" + std::to_string(prop.texture->is_loaded()));
                    
                    // IF texture has pixels in memory, write them (handle generated/unnamed textures)
                    if (!prop.texture->pixels.empty()) {
                         // Convert CompactVec4 to raw RGBA bytes for stbi_write_png
                         std::vector<uint8_t> raw_pixels(prop.texture->pixels.size() * 4);
                         for (size_t pi = 0; pi < prop.texture->pixels.size(); ++pi) {
                             const auto& px = prop.texture->pixels[pi];
                             raw_pixels[pi * 4 + 0] = px.r;
                             raw_pixels[pi * 4 + 1] = px.g;
                             raw_pixels[pi * 4 + 2] = px.b;
                             raw_pixels[pi * 4 + 3] = px.a;
                         }
                         
                         // Encode to PNG in memory
                         std::vector<char> png_data;
                         stbi_write_png_to_func(write_to_vector_func, &png_data, 
                             prop.texture->width, prop.texture->height, 4, 
                             raw_pixels.data(), prop.texture->width * 4);
                         
                         if (!png_data.empty()) {
                            bin_out.write(png_data.data(), png_data.size());
                            tex_entry["size"] = png_data.size();
                            tex_entry["format"] = ".png";
                            SCENE_LOG_INFO("Embedded memory texture: " + path + " Size: " + std::to_string(png_data.size()));
                         } else {
                             SCENE_LOG_ERROR("Failed to encode memory texture to PNG: " + path);
                         }

                    } else if (!prop.texture->float_pixels.empty()) {
                        // Handle Float textures (EXR/HDR) - TODO: Need stbi_write_hdr or similar
                        // For now, skip or implement later.
                        SCENE_LOG_WARN("Float texture embedding not fully implemented (skipping encode): " + path);

                    } else if (fs::exists(path)) {
                        // Read from file
                        std::ifstream tex_file(path, std::ios::binary);
                        if (tex_file) {
                            tex_file.seekg(0, std::ios::end);
                            size_t size = tex_file.tellg();
                            tex_file.seekg(0, std::ios::beg);
                            
                            std::vector<char> data(size);
                            tex_file.read(data.data(), size);
                            bin_out.write(data.data(), size);
                            
                            tex_entry["size"] = size;
                            tex_entry["format"] = fs::path(path).extension().string();
                        }
                    } 
                } else {
                    // Path reference
                    tex_entry["mode"] = "path";
                    tex_entry["path"] = path;
                }
                
                texture_map[path] = texture_id++;
                arr.push_back(tex_entry);
            }
        };
        
        // Use PrincipledBSDF pointer for correct texture property access
        checkTexture(pbsdf->albedoProperty, "albedo");
        checkTexture(pbsdf->normalProperty, "normal");
        checkTexture(pbsdf->roughnessProperty, "roughness");
        checkTexture(pbsdf->metallicProperty, "metallic");
        checkTexture(pbsdf->emissionProperty, "emission");
        checkTexture(pbsdf->opacityProperty, "opacity");
        checkTexture(pbsdf->transmissionProperty, "transmission");
    }
    
    SCENE_LOG_INFO("[ProjectManager] Serialized " + std::to_string(arr.size()) + " textures.");
    return arr;
}

void ProjectManager::deserializeTextures(const json& j, std::ifstream& bin_in, const std::string& project_dir) {
    // Clear previous embedded texture cache
    m_embedded_texture_cache.clear();
    
    for (const auto& tex : j) {
        std::string mode = tex.value("mode", "path");

        if (mode == "embed") {
            // Store embedded texture in memory cache (NO DISK WRITE!)
            // MaterialManager::deserializeProperty will load from this cache
            long long offset = tex.value("offset", 0LL);
            size_t size = tex.value("size", 0);
            std::string original_name = tex.value("original_name", "");
            std::string usage = tex.value("usage", "albedo");

            if (size > 0 && bin_in.is_open() && !original_name.empty()) {
                bin_in.seekg(offset);
                std::vector<char> data(size);
                bin_in.read(data.data(), size);
                
                // Determine TextureType from usage string
                TextureType texType = TextureType::Albedo;
                if (usage == "normal") texType = TextureType::Normal;
                else if (usage == "roughness") texType = TextureType::Roughness;
                else if (usage == "metallic") texType = TextureType::Metallic;
                else if (usage == "emission") texType = TextureType::Emission;
                else if (usage == "opacity") texType = TextureType::Opacity;
                else if (usage == "transmission") texType = TextureType::Transmission;
                
                // Store in memory cache - keyed by original_name
                m_embedded_texture_cache[original_name] = {std::move(data), texType};
                
                SCENE_LOG_INFO("[EMBED CACHE] Stored texture in memory: " + original_name + 
                               " (" + std::to_string(size) + " bytes)");
            }
        }
        // Path mode textures are loaded directly by material deserializer from disk
    }
    SCENE_LOG_INFO("[ProjectManager] Cached " + std::to_string(m_embedded_texture_cache.size()) + 
                   " embedded textures in memory (no temp files).");
}
