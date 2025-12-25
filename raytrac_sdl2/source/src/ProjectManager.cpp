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
#include <unordered_set>
#include <unordered_map>
#include <algorithm>

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
    SCENE_LOG_INFO("New project created.");
}


bool ProjectManager::saveProject(std::function<void(int, const std::string&)> progress_callback) {
    if (g_project.current_file_path.empty()) {
        SCENE_LOG_ERROR("No file path set. Use saveProject(filepath) instead.");
        return false;
    }
    return saveProject(g_project.current_file_path, progress_callback);
}

// Main Save Implementation
bool ProjectManager::saveProject(const std::string& filepath, std::function<void(int, const std::string&)> progress_callback) {
    if (progress_callback) progress_callback(0, "Preparing to save...");

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
        if (progress_callback) progress_callback(5, "Writing metadata...");

        // 3. Write JSON Header (Streamed)
        out_json << "{\n";
        out_json << "  \"format_version\": " << json(g_project.format_version) << ",\n";
        out_json << "  \"project_name\": " << json(g_project.project_name) << ",\n";
        out_json << "  \"author\": " << json(g_project.author) << ",\n";
        out_json << "  \"description\": " << json(g_project.description) << ",\n";
        out_json << "  \"next_model_id\": " << g_project.next_model_id << ",\n";
        out_json << "  \"next_object_id\": " << g_project.next_object_id << ",\n";
        out_json << "  \"next_texture_id\": " << g_project.next_texture_id << ",\n";
        
        // 4. IMPORTED MODELS - SPARSE SAVE OPTIMIZATION
        out_json << "  \"imported_models\": [\n";
        
        size_t total_models = g_project.imported_models.size();
        for (size_t i = 0; i < total_models; ++i) {
            auto& model = g_project.imported_models[i];
            
            // Update Progress
            if (progress_callback) {
                int p = 10 + (int)((float)i / total_models * 80.0f); // 10% to 90%
                std::string msg = "Saving model data: " + model.display_name;
                progress_callback(p, msg);
            }

            // Record current offset in binary file
            long long bin_start = out_bin.tellp();
            size_t written_count = 0;
            
            // Write ONLY modified objects to binary
            for (const auto& obj : model.objects) {
                bool is_default = isIdentity(obj.transform) && obj.visible; 
                
                if (!is_default) {
                    writeStringBinary(out_bin, obj.node_name);
                    out_bin.write(reinterpret_cast<const char*>(&obj.transform), sizeof(Matrix4x4));
                    out_bin.write(reinterpret_cast<const char*>(&obj.material_id), sizeof(uint16_t));
                    uint8_t visible_byte = obj.visible ? 1 : 0;
                    out_bin.write(reinterpret_cast<const char*>(&visible_byte), sizeof(uint8_t));
                    written_count++;
                }
            }
            
            // Write metadata to JSON
            out_json << "    {\n";
            out_json << "      \"id\": " << model.id << ",\n";
            out_json << "      \"original_path\": " << json(model.original_path) << ",\n";
            out_json << "      \"package_path\": " << json(model.package_path) << ",\n";
            out_json << "      \"display_name\": " << json(model.display_name) << ",\n";
            
            // Deleted Objects
            out_json << "      \"deleted_objects\": [";
            std::sort(model.deleted_objects.begin(), model.deleted_objects.end());
            model.deleted_objects.erase(std::unique(model.deleted_objects.begin(), model.deleted_objects.end()), model.deleted_objects.end());
            for (size_t k = 0; k < model.deleted_objects.size(); ++k) {
                out_json << json(model.deleted_objects[k]);
                if (k < model.deleted_objects.size() - 1) out_json << ",";
            }
            out_json << "],\n";
            
            // Binary Reference
            out_json << "      \"objects_bin_offset\": " << bin_start << ",\n";
            out_json << "      \"objects_count\": " << written_count << "\n";
            
            out_json << "    }";
            if (i < total_models - 1) out_json << ",";
            out_json << "\n";
        }
        out_json << "  ],\n"; // End imported_models
        
        if (progress_callback) progress_callback(90, "Saving procedural objects...");

        // 5. PROCEDURAL OBJECTS
        out_json << "  \"procedural_objects\": [\n";
        for (size_t i = 0; i < g_project.procedural_objects.size(); ++i) {
            const auto& proc = g_project.procedural_objects[i];
             out_json << "    {\n";
            out_json << "      \"id\": " << proc.id << ",\n";
            out_json << "      \"mesh_type\": " << (int)proc.mesh_type << ",\n";
            out_json << "      \"display_name\": " << json(proc.display_name) << ",\n"; 
            
            out_json << "      \"transform\": [";
            for(int r=0; r<4; ++r) for(int c=0; c<4; ++c) {
                 out_json << proc.transform.m[r][c] << ( (r==3 && c==3) ? "" : ",");
            }
            out_json << "],\n";
            
            out_json << "      \"material_id\": " << proc.material_id << ",\n";
            out_json << "      \"visible\": " << (proc.visible ? "true" : "false") << "\n";
            out_json << "    }";
            if (i < g_project.procedural_objects.size() - 1) out_json << ",";
            out_json << "\n";
        }
        out_json << "  ],\n";
        
        // 6. TEXTURE ASSETS
        out_json << "  \"texture_assets\": [\n";
        for (size_t i = 0; i < g_project.texture_assets.size(); ++i) {
            const auto& tex = g_project.texture_assets[i];
             out_json << "    {\n";
            out_json << "      \"id\": " << tex.id << ",\n";
            out_json << "      \"original_path\": " << json(tex.original_path) << ",\n";
            out_json << "      \"package_path\": " << json(tex.package_path) << ",\n";
            out_json << "      \"usage\": " << json(tex.usage) << "\n";
            out_json << "    }";
            if (i < g_project.texture_assets.size() - 1) out_json << ",";
            out_json << "\n";
        }
        out_json << "  ]\n";
        
        out_json << "}"; // End Root
        
        out_json.flush();
        out_bin.flush(); // Ensure everything is written
        
        out_json.close();
        out_bin.close();
        
        // ---------------------------------------------------------
        // COPY ASSETS TO PACKAGE FOLDER (Portability Logic)
        // ---------------------------------------------------------
        if (progress_callback) progress_callback(92, "Ensuring asset portability...");
        
        fs::path project_folder = final_json_path.parent_path();
        
        // Copy Models
        for (const auto& model : g_project.imported_models) {
            if (model.package_path.empty()) continue;
            
            fs::path dest = project_folder / model.package_path;
            if (!fs::exists(dest) && fs::exists(model.original_path)) {
                try {
                    fs::create_directories(dest.parent_path());
                    fs::copy_file(model.original_path, dest, fs::copy_options::overwrite_existing);
                    
                    // Also copy associated .bin file if it exists
                    fs::path src_bin = fs::path(model.original_path).replace_extension(".bin");
                    if (fs::exists(src_bin)) {
                        // Ensure .bin is copied to the same folder as the .gltf with the SAME NAME
                        // Since packagePath now uses original filename, replacing extension works perfectly.
                        fs::path dest_bin = dest;
                        dest_bin.replace_extension(".bin");
                        fs::copy_file(src_bin, dest_bin, fs::copy_options::overwrite_existing);
                        // No log needed per file to avoid spam
                    }
                } catch (const std::exception& e) {
                    SCENE_LOG_WARN("Failed to copy model to project: " + model.display_name);
                }
            }
        }
        
        // Copy Textures
        for (const auto& tex : g_project.texture_assets) {
             if (tex.package_path.empty()) continue;
             
             fs::path dest = project_folder / tex.package_path;
             if (!fs::exists(dest) && fs::exists(tex.original_path)) {
                 try {
                     fs::create_directories(dest.parent_path());
                     fs::copy_file(tex.original_path, dest, fs::copy_options::overwrite_existing);
                 } catch (const std::exception& e) {
                     SCENE_LOG_WARN("Failed to copy texture to project: " + fs::path(tex.original_path).filename().string());
                 }
             }
        }
        
        if (progress_callback) progress_callback(95, "Finalizing...");

        // 7. Atomic Commit
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
    newProject();
    scene.clear();
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
        // ... Load other metadata
        
        fs::path project_folder = fs::path(filepath).parent_path();
        
        // Load Imported Models
        if (root.contains("imported_models")) {
            auto& models = root["imported_models"];
            size_t total_count = models.size();
            
            for (size_t i = 0; i < total_count; ++i) {
                const auto& m = models[i];
                
                // Update Progress
                if (progress_callback) {
                    int p = 10 + (int)((float)i / total_count * 70.0f); // 10% to 80%
                    std::string dname = m.value("display_name", "Model");
                    progress_callback(p, "Loading model: " + dname);
                }

                ImportedModelData model;
                model.id = m.value("id", 0);
                model.original_path = m.value("original_path", "");
                model.package_path = m.value("package_path", "");
                model.display_name = m.value("display_name", "Model");
                
                // Deleted Objects
                if (m.contains("deleted_objects") && m["deleted_objects"].is_array()) {
                    for (const auto& d : m["deleted_objects"]) {
                        model.deleted_objects.push_back(d.get<std::string>());
                    }
                }
                
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
                
                size_t objects_before = scene.world.objects.size();
                renderer.create_scene(scene, optix_gpu, actual_path, nullptr);
                
                // RESTORE OBJECT PREFIXES (Critical for Sync)
                std::string import_prefix = std::to_string(model.id) + "_";
                for (size_t k = objects_before; k < scene.world.objects.size(); ++k) {
                    auto tri = std::dynamic_pointer_cast<Triangle>(scene.world.objects[k]);
                    if (tri) {
                        tri->setNodeName(import_prefix + tri->nodeName);
                    }
                }
                
                // Apply Deletions
                if (!model.deleted_objects.empty()) {
                    std::unordered_set<std::string> deleted_set(model.deleted_objects.begin(), model.deleted_objects.end());
                    auto& objs = scene.world.objects;
                    objs.erase(
                        std::remove_if(objs.begin() + objects_before, objs.end(),
                            [&](const std::shared_ptr<Hittable>& obj) {
                                auto tri = std::dynamic_pointer_cast<Triangle>(obj);
                                if (!tri) return false;
                                return deleted_set.count(tri->nodeName) > 0;
                            }),
                        objs.end());
                }
                
                // LOAD OBJECT OVERRIDES (Sparse Load)
                if (has_binary && m.contains("objects_bin_offset")) {
                    long long offset = m["objects_bin_offset"];
                    size_t count = m.value("objects_count", 0);
                    
                    if (count > 0) {
                        in_bin.seekg(offset);
                        model.objects.reserve(count);
                        
                        for(size_t k=0; k<count; ++k) {
                            ImportedModelData::ObjectInstance inst;
                            inst.node_name = readStringBinary(in_bin);
                            in_bin.read(reinterpret_cast<char*>(&inst.transform), sizeof(Matrix4x4));
                            in_bin.read(reinterpret_cast<char*>(&inst.material_id), sizeof(uint16_t));
                            uint8_t visible_byte = 1;
                            in_bin.read(reinterpret_cast<char*>(&visible_byte), sizeof(uint8_t));
                            inst.visible = (visible_byte != 0);
                            model.objects.push_back(inst);
                        }
                    }
                } else if (m.contains("objects")) {
                    // JSON fallback
                     for (const auto& o : m["objects"]) {
                        ImportedModelData::ObjectInstance inst;
                        inst.node_name = o.value("node_name", "");
                        if (o.contains("transform")) {
                             inst.transform = jsonToMat4(o["transform"]);
                        } else {
                             for(int r=0; r<4; ++r) for(int c=0; c<4; ++c) inst.transform.m[r][c] = (r==c ? 1.0f : 0.0f);
                        }
                        inst.material_id = o.value("material_id", 0);
                        inst.visible = o.value("visible", true);
                        model.objects.push_back(inst);
                    }
                }
                
                g_project.imported_models.push_back(model);
            }
        }
        
        if (progress_callback) progress_callback(80, "Applying transforms...");

        // Apply Transforms (O(N) Map-based approach)
        std::unordered_map<std::string_view, std::shared_ptr<Triangle>> scene_map;
        scene_map.reserve(scene.world.objects.size());
        for (auto& obj : scene.world.objects) {
             if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
                 scene_map[tri->nodeName] = tri;
             }
        }

        for (const auto& model : g_project.imported_models) {
            for (const auto& inst : model.objects) {
                auto it = scene_map.find(inst.node_name);
                if (it != scene_map.end()) {
                    auto tri = it->second;
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
        
        // Load Procedural Objects
        if (root.contains("procedural_objects")) {
             for (const auto& p : root["procedural_objects"]) {
                 ProceduralObjectData proc;
                 proc.id = p.value("id", 0);
                 proc.mesh_type = (ProceduralMeshType)p.value("mesh_type", 0);
                 proc.display_name = p.value("display_name", "Procedural");
                 if (p.contains("transform")) proc.transform = jsonToMat4(p["transform"]);
                 proc.material_id = p.value("material_id", 0);
                 proc.visible = p.value("visible", true);
                 g_project.procedural_objects.push_back(proc);
                 
                 // Create Scene Object
                 std::shared_ptr<Transform> t = std::make_shared<Transform>();
                 t->setBase(proc.transform);
                 
                 if (proc.mesh_type == ProceduralMeshType::Plane) {
                     Vec3 v0(-1, 0, 1), v1(1, 0, 1), v2(1, 0, -1), v3(-1, 0, -1);
                     Vec3 n(0, 1, 0);
                     Vec2 t0(0, 0), t1(1, 0), t2(1, 1), t3(0, 1);
                     
                     auto tri1 = std::make_shared<Triangle>(v0, v1, v2, n, n, n, t0, t1, t2, proc.material_id);
                     tri1->setTransformHandle(t); tri1->setNodeName(proc.display_name); tri1->update_bounding_box();
                     
                     auto tri2 = std::make_shared<Triangle>(v0, v2, v3, n, n, n, t0, t2, t3, proc.material_id);
                     tri2->setTransformHandle(t); tri2->setNodeName(proc.display_name); tri2->update_bounding_box();
                     
                     if(proc.visible) {
                        scene.world.objects.push_back(tri1);
                        scene.world.objects.push_back(tri2);
                     }
                 }
                 else if (proc.mesh_type == ProceduralMeshType::Cube) {
                    Vec3 pts[8] = {
                        Vec3(-1,-1, 1), Vec3( 1,-1, 1), Vec3( 1, 1, 1), Vec3(-1, 1, 1),
                        Vec3(-1,-1,-1), Vec3( 1,-1,-1), Vec3( 1, 1,-1), Vec3(-1, 1,-1)
                    };
                    int indices[36] = {
                        0,1,2, 2,3,0, 1,5,6, 6,2,1, 7,6,5, 5,4,7, 4,0,3, 3,7,4, 4,5,1, 1,0,4, 3,2,6, 6,7,3
                    };
                    for(int k=0; k<36; k+=3) {
                        Vec3 v0 = pts[indices[k]];
                        Vec3 v1 = pts[indices[k+1]];
                        Vec3 v2 = pts[indices[k+2]];
                        Vec3 n = (v1-v0).cross(v2-v0).normalize();
                        auto tri = std::make_shared<Triangle>(v0, v1, v2, n, n, n, Vec2(0,0), Vec2(1,0), Vec2(0,1), proc.material_id);
                        tri->setTransformHandle(t);
                        tri->setNodeName(proc.display_name);
                        tri->update_bounding_box();
                        if(proc.visible) scene.world.objects.push_back(tri);
                    }
                 }
             }
        }

        // Load Textures
        if (root.contains("texture_assets")) {
            for (const auto& t : root["texture_assets"]) {
                TextureAssetData tex;
                tex.id = t.value("id", 0);
                tex.original_path = t.value("original_path", "");
                tex.package_path = t.value("package_path", "");
                tex.usage = t.value("usage", "unknown");
                g_project.texture_assets.push_back(tex);
            }
        }
        
    } catch (const std::exception& e) {
        SCENE_LOG_ERROR("Error during project loading: " + std::string(e.what()));
        return false;
    }
    
    if (in_bin.is_open()) in_bin.close();
    
    g_project.current_file_path = filepath;
    g_project.is_modified = false;
    
    if (progress_callback) progress_callback(90, "Rebuilding acceleration structures...");

    // Rebuild Acceleration Structures
    renderer.rebuildBVH(scene, settings.UI_use_embree);
    
    if (optix_gpu) {
        if (progress_callback) progress_callback(95, "Uploading to GPU...");
        renderer.rebuildOptiXGeometry(scene, optix_gpu);
        optix_gpu->setLightParams(scene.lights);
        if (scene.camera) optix_gpu->setCameraParams(*scene.camera);
    }
    
    if (progress_callback) progress_callback(100, "Done.");
    SCENE_LOG_INFO("Project loaded: " + filepath);
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
    
    // Copy file to package storage (if project folder exists)
    if (!g_project.current_file_path.empty()) {
        if (progress_callback) progress_callback(10, "Copying model file...");
        fs::path project_folder = fs::path(g_project.current_file_path).parent_path();
        fs::path dest_path = project_folder / package_path;
        
        try {
            fs::create_directories(dest_path.parent_path());
            fs::copy_file(filepath, dest_path, fs::copy_options::overwrite_existing);
            
            // Check for associated .bin file (GLTF standard)
            fs::path bin_path = fs::path(filepath).replace_extension(".bin");
            if (fs::exists(bin_path)) {
                 fs::path dest_bin = dest_path;
                 dest_bin.replace_extension(".bin");
                 fs::copy_file(bin_path, dest_bin, fs::copy_options::overwrite_existing);
                 SCENE_LOG_INFO("Copied associated bin file: " + dest_bin.filename().string());
            }
        } catch (const std::exception& e) {
            SCENE_LOG_WARN("Could not copy model files: " + std::string(e.what()));
        }
    }
    
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
    SCENE_LOG_WARN("ZIP packaging not yet implemented. Using folder structure.");
    return saveProject(filepath);
}

bool ProjectManager::readZipPackage(const std::string& filepath) {
    SCENE_LOG_WARN("ZIP packaging not yet implemented.");
    return false;
}
