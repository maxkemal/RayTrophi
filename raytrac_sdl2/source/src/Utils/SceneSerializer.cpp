#include "SceneSerializer.h"
#include "globals.h"
#include "Renderer.h"
#include "OptixWrapper.h"
#include "default_scene_creator.hpp"
#include "World.h"
#include <fstream>
#include <iostream>
#include "json.hpp"
#include <filesystem>
#include "TerrainManager.h"
#include <unordered_set>

using json = nlohmann::json;

// Helper to convert Vec3 to JSON array
json vec3ToJson(const Vec3& v) {
    return { v.x, v.y, v.z };
}

// Helper to convert JSON array to Vec3
Vec3 jsonToVec3(const json& j) {
    if (j.is_array() && j.size() >= 3)
        return Vec3(j[0], j[1], j[2]);
    return Vec3(0, 0, 0);
}

// Helper for Matrix4x4 serialization
json mat4ToJson(const Matrix4x4& m) {
    json j = json::array();
    for(int i=0; i<4; ++i)
        for(int k=0; k<4; ++k)
            j.push_back(m.m[i][k]);
    return j;
}

Matrix4x4 jsonToMat4(const json& j) {
    Matrix4x4 m;
    if (j.is_array() && j.size() == 16) {
        int idx = 0;
        for(int i=0; i<4; ++i)
            for(int k=0; k<4; ++k)
                m.m[i][k] = j[idx++];
    }
    return m;
}

void SceneSerializer::Serialize(const SceneData& scene, const RenderSettings& settings, const std::string& filepath) {
    // 1. Prepare Safe Save paths
    std::string temp_path = filepath + ".tmp";
    
    // 2. Open Stream (Streaming Write)
    std::ofstream out(temp_path);
    if (!out.is_open()) {
        SCENE_LOG_ERROR("Failed to create temporary save file: " + temp_path);
        return;
    }

    // Direct streaming to file - Eliminates massive single-core CPU spike for JSON construction
    out << "{\n";

    // 1. Scene Info
    extern std::string active_model_path;
    out << "  \"model_path\": " << json(active_model_path) << ",\n";

    // 2. Camera
    if (scene.camera) {
        out << "  \"camera\": {\n";
        out << "    \"lookfrom\": " << vec3ToJson(scene.camera->lookfrom) << ",\n";
        out << "    \"lookat\": " << vec3ToJson(scene.camera->lookat) << ",\n";
        out << "    \"vup\": " << vec3ToJson(scene.camera->vup) << ",\n";
        out << "    \"vfov\": " << scene.camera->vfov << ",\n";
        out << "    \"aperture\": " << scene.camera->aperture << ",\n";
        out << "    \"focus_dist\": " << scene.camera->focus_dist << "\n";
        out << "  },\n";
    }

    // 3. Lights
    out << "  \"lights\": [\n";
    for (size_t i = 0; i < scene.lights.size(); ++i) {
        const auto& light = scene.lights[i];
        out << "    {\n";
        out << "      \"type\": " << (int)light->type() << ",\n";
        out << "      \"position\": " << vec3ToJson(light->position) << ",\n";
        out << "      \"color\": " << vec3ToJson(light->color) << ",\n";
        out << "      \"intensity\": " << light->intensity << ",\n";
        out << "      \"name\": " << json(light->nodeName);
        
        if (auto dl = std::dynamic_pointer_cast<DirectionalLight>(light)) {
            out << ",\n      \"direction\": " << vec3ToJson(dl->direction) << ",\n";
            out << "      \"radius\": " << dl->radius;
        }
        else if (auto pl = std::dynamic_pointer_cast<PointLight>(light)) {
            out << ",\n      \"radius\": " << pl->radius;
        }
        else if (auto sl = std::dynamic_pointer_cast<SpotLight>(light)) {
            out << ",\n      \"direction\": " << vec3ToJson(sl->direction) << ",\n";
            out << "      \"radius\": " << sl->radius << ",\n";
            out << "      \"angle\": " << sl->getAngleDegrees() << ",\n";
            out << "      \"falloff\": " << sl->getFalloff();
        }
        else if (auto al = std::dynamic_pointer_cast<AreaLight>(light)) {
            out << ",\n      \"u\": " << vec3ToJson(al->u) << ",\n";
            out << "      \"v\": " << vec3ToJson(al->v) << ",\n";
            out << "      \"width\": " << al->width << ",\n";
            out << "      \"height\": " << al->height;
        }
        out << "\n    }";
        if (i < scene.lights.size() - 1) out << ",";
        out << "\n";
    }
    out << "  ],\n";

    // 4. Object Transforms (Streamed!)
    out << "  \"objects\": [\n";

    // FILTERING: Identify terrain triangles to exclude from generic serialization
    std::unordered_set<std::shared_ptr<Hittable>> terrain_triangles;
    auto& terrains = TerrainManager::getInstance().getTerrains();
    for (auto& t : terrains) {
        for (auto& tri : t.mesh_triangles) {
            terrain_triangles.insert(tri);
        }
    }

    bool first_obj = true;
    for (const auto& obj : scene.world.objects) {
        // Skip terrain chunks (they are saved in terrain_system)
        if (std::dynamic_pointer_cast<Hittable>(obj) && terrain_triangles.count(std::dynamic_pointer_cast<Hittable>(obj))) {
            continue;
        }

        auto tri = std::dynamic_pointer_cast<Triangle>(obj);
        if (tri) { // Serialize all triangles to maintain index alignment
            if (!first_obj) out << ",\n";
            
            out << "    {";
            out << "\"name\":" << json(tri->nodeName) << ",";
            
            std::string meshType = "Model";
            if (tri->nodeName == "Cube") meshType = "Cube";
            else if (tri->nodeName == "Plane") meshType = "Plane";
            out << "\"mesh_type\":\"" << meshType << "\",";

            auto th = tri->getTransformHandle();
            if (th) {
                out << "\"transform\":[";
                const auto& m = th->base;
                for(int r=0; r<4; ++r) for(int c=0; c<4; ++c) {
                    out << m.m[r][c] << ((r==3 && c==3) ? "" : ",");
                }
                out << "],";
            }
            out << "\"material_id\":" << tri->getMaterialID();
            out << "}";
            
            first_obj = false;
        }
    }
    out << "\n  ],\n";

    // 5. Render Settings
    out << "  \"settings\": {\n";
    out << "    \"quality_preset\": " << (int)settings.quality_preset << ",\n";
    out << "    \"samples_per_pixel\": " << settings.samples_per_pixel << ",\n";
    out << "    \"max_bounces\": " << settings.max_bounces << ",\n";
    out << "    \"use_adaptive\": " << (settings.use_adaptive_sampling ? "true" : "false") << ",\n";
    out << "    \"use_denoiser\": " << (settings.use_denoiser ? "true" : "false") << ",\n";
    out << "    \"use_optix\": " << (settings.use_optix ? "true" : "false") << ",\n";
    out << "    \"persistent_tonemap\": " << (settings.persistent_tonemap ? "true" : "false") << "\n"; 
    out << "  },\n";

    // Terrain System
    auto abs_path = std::filesystem::absolute(filepath);
    std::string terrainDir = abs_path.parent_path().string();
    SCENE_LOG_INFO("[SceneSerializer] Saving terrain system to: " + terrainDir);
    json terrainJson = TerrainManager::getInstance().serialize(terrainDir);
    out << "  \"terrain_system\": " << terrainJson << ",\n";

    // 6. PostFX
    const auto& pp = scene.color_processor.params;
    out << "  \"postfx\": {\n";
    out << "    \"exposure\": " << pp.global_exposure << ",\n";
    out << "    \"gamma\": " << pp.global_gamma << ",\n";
    out << "    \"saturation\": " << pp.saturation << ",\n";
    out << "    \"color_temperature\": " << pp.color_temperature << ",\n";
    out << "    \"tone_mapping\": " << (int)pp.tone_mapping_type << ",\n";
    out << "    \"vignette_enabled\": " << (pp.enable_vignette ? "true" : "false") << ",\n";
    out << "    \"vignette_strength\": " << pp.vignette_strength << "\n";
    out << "  },\n";

    // 7. World
    out << "  \"world\": {\n";
    out << "    \"background_color\": " << vec3ToJson(scene.background_color) << "\n";
    out << "  },\n";

    // 8. Timeline
    json j_timeline;
    scene.timeline.serialize(j_timeline);
    // Indent the timeline json for better readability in file
    std::string timeline_str = j_timeline.dump(2); // Use indent 2
    // We need to verify if dump is consistent or if we should just stream it.
    // Streaming simple json object is fine.
    out << "  \"timeline\": " << timeline_str << "\n";

    out << "}";
    out.flush();
    out.close();

    // 3. Atomic Commit
    try {
        if (std::filesystem::exists(filepath)) std::filesystem::remove(filepath);
        std::filesystem::rename(temp_path, filepath);
        SCENE_LOG_INFO("Scene saved to: " + filepath);
    } catch (const std::exception& e) {
        SCENE_LOG_ERROR("Failed to rename save file: " + std::string(e.what()));
    }
}

bool SceneSerializer::Deserialize(SceneData& scene, RenderSettings& settings, Renderer& renderer, OptixWrapper* optix_gpu, const std::string& filepath) {
    std::ifstream in(filepath);
    if (!in.is_open()) {
        SCENE_LOG_ERROR("Failed to open scene file: " + filepath);
        return false;
    }

    json root;
    try {
        in >> root;
    } catch (...) {
        SCENE_LOG_ERROR("Failed to parse JSON scene file.");
        return false;
    }

    // 1. Load Base Scene (Model or Default)
    extern std::string active_model_path;
    std::string model_path = root.value("model_path", "Untitled");
    active_model_path = model_path;

    // Clear current scene
    scene.clear();
    renderer.resetCPUAccumulation();
    if (optix_gpu) optix_gpu->resetAccumulation();

    // NOTE: Do NOT call createDefaultScene here - geometry is loaded from binary in ProjectManager
    // Only load external models if specified
    if (!model_path.empty() && model_path != "Untitled") {
        renderer.create_scene(scene, optix_gpu, model_path, nullptr); 
    }
    // If model_path is "Untitled", scene geometry will be loaded from .bin file by ProjectManager

    // 2. Apply Camera
    if (root.contains("camera")) {
        json c = root["camera"];
        if (!scene.camera) {
             scene.camera = std::make_shared<Camera>(Vec3(0,0,5), Vec3(0,0,0), Vec3(0,1,0), 60.0, 1.0, 0.0, 1.0, 5);
        }
        scene.camera->lookfrom = jsonToVec3(c["lookfrom"]);
        scene.camera->lookat = jsonToVec3(c["lookat"]);
        scene.camera->vup = jsonToVec3(c["vup"]);
        scene.camera->vfov = c.value("vfov", 60.0);
        scene.camera->aperture = c.value("aperture", 0.0f);
        scene.camera->focus_dist = c.value("focus_dist", 10.0f);
        scene.camera->update_camera_vectors();
    }

    // 3. Apply Lights (Recreate them)
    if (root.contains("lights") && root["lights"].is_array() && !root["lights"].empty()) {
        scene.lights.clear();
        for (const auto& l : root["lights"]) {
            std::shared_ptr<Light> lightPtr = nullptr;
            int type = l.value("type", 0);
            Vec3 pos = jsonToVec3(l["position"]);
            Vec3 col = jsonToVec3(l["color"]);
            float inten = l.value("intensity", 1.0f);
            std::string name = l.value("name", "Light");
            
            if (type == (int)LightType::Point) {
                float r = l.value("radius", 0.1f);
                auto pl = std::make_shared<PointLight>(pos, col * inten, r);
                lightPtr = pl;
            } else if (type == (int)LightType::Directional) {
                Vec3 dir = jsonToVec3(l["direction"]);
                float r = l.value("radius", 0.0f);
                auto dl = std::make_shared<DirectionalLight>(dir, col * inten, r);
                dl->position = pos; 
                lightPtr = dl;
            } else if (type == (int)LightType::Spot) {
                Vec3 dir = jsonToVec3(l["direction"]);
                float range = l.value("radius", 20.0f);
                float angle = l.value("angle", 45.0f);
                float falloff = l.value("falloff", 0.5f);
                auto sl = std::make_shared<SpotLight>(pos, dir, col * inten, angle, range);
                sl->setFalloff(falloff);
                lightPtr = sl;
            } else if (type == (int)LightType::Area) {
                Vec3 u = jsonToVec3(l["u"]);
                Vec3 v = jsonToVec3(l["v"]);
                float w = l.value("width", 1.0f);
                float h = l.value("height", 1.0f);
                auto al = std::make_shared<AreaLight>(pos, u, v, w, h, col * inten);
                lightPtr = al;
            }
            
            if (lightPtr) {
                lightPtr->nodeName = name;
                scene.lights.push_back(lightPtr);
            }
        }
    }

    // 4. Transform Object Updates (match by index for now)
    if (root.contains("objects")) {
        const auto& objsJson = root["objects"];
        size_t limit = std::min(scene.world.objects.size(), objsJson.size());
        
        for (size_t i = 0; i < limit; ++i) {
            auto tri = std::dynamic_pointer_cast<Triangle>(scene.world.objects[i]);
            const auto& o = objsJson[i];
            
            if (tri) {
                auto th = tri->getTransformHandle(); 
                if (!th) {
                     th = std::make_shared<Transform>();
                     tri->setTransformHandle(th);
                }
                
                if (o.contains("transform")) {
                    th->setBase(jsonToMat4(o["transform"]));
                }
                
                if (o.contains("name")) {
                    tri->nodeName = o["name"];
                }
                
                if (o.contains("material_id")) {
                    tri->setMaterialID(o["material_id"]);
                }
                tri->updateTransformedVertices();
            }
        }
    }

    // 5. Render Settings
    if (root.contains("settings")) {
        json s = root["settings"];
        settings.quality_preset = (QualityPreset)s.value("quality_preset", 0);
        settings.samples_per_pixel = s.value("samples_per_pixel", 1);
        settings.max_bounces = s.value("max_bounces", 10);
        settings.use_adaptive_sampling = s.value("use_adaptive", true);
        settings.use_denoiser = s.value("use_denoiser", false);
        settings.use_optix = s.value("use_optix", true);
        settings.persistent_tonemap = s.value("persistent_tonemap", false);
    }

    // 6. PostFX (Color Processing)
    if (root.contains("postfx")) {
        json pf = root["postfx"];
        auto& pp = scene.color_processor.params;
        pp.global_exposure = pf.value("exposure", 1.0f);
        pp.global_gamma = pf.value("gamma", 2.2f);
        pp.saturation = pf.value("saturation", 1.0f);
        pp.color_temperature = pf.value("color_temperature", 6500.0f);
        pp.tone_mapping_type = (ToneMappingType)pf.value("tone_mapping", 4); // 4 = None
        pp.enable_vignette = pf.value("vignette_enabled", false);
        pp.vignette_strength = pf.value("vignette_strength", 0.0f);
    }

    // 7. World/Environment
    if (root.contains("world")) {
        json w = root["world"];
        scene.background_color = jsonToVec3(w["background_color"]);
        // Extended World settings (HDRI, Nishita) can be loaded here
        // renderer.world.setColor(scene.background_color); // If needed
    }

    // 8. Timeline
    if (root.contains("timeline")) {
        scene.timeline.deserialize(root["timeline"]);
    }
    
    // 9. Terrains (Must be after materials/timeline but before BVH build)
    // 9. Terrains (Must be after materials/timeline but before BVH build)
    if (root.contains("terrain_system")) {
        // CLEANUP: Purge any zombie terrain mesh chunks loaded from binary geometry
        // to prevent duplication (static mesh blocking the editable terrain).
        auto& objs = scene.world.objects;
        size_t purged_count = 0;
        for (auto it = objs.begin(); it != objs.end(); ) {
            auto tri = std::dynamic_pointer_cast<Triangle>(*it);
            if (tri && tri->nodeName.find("Terrain_") == 0 && tri->nodeName.find("_Chunk") != std::string::npos) {
                it = objs.erase(it);
                purged_count++;
            } else {
                ++it;
            }
        }
        if (purged_count > 0) SCENE_LOG_INFO("[SceneSerializer] Purged " + std::to_string(purged_count) + " zombie terrain chunks.");

        // NEW: Clear TerrainManager before deserializing to avoid duplication after scene reload
        TerrainManager::getInstance().removeAllTerrains(scene);

        auto abs_path = std::filesystem::absolute(filepath);
        std::string terrainDir = abs_path.parent_path().string();
        SCENE_LOG_INFO("[SceneSerializer] Loading terrain system from: " + terrainDir);
        TerrainManager::getInstance().deserialize(root["terrain_system"], terrainDir, scene);
    }
    
    // 10. Rebuild All
    renderer.rebuildBVH(scene, settings.UI_use_embree);
    
    if (optix_gpu) {
        renderer.rebuildOptiXGeometry(scene, optix_gpu);
        optix_gpu->setLightParams(scene.lights);
        if (scene.camera) optix_gpu->setCameraParams(*scene.camera);
    }

    SCENE_LOG_INFO("Scene loaded from: " + filepath);
    
    // DEBUG: Verify Animation Tracks
    for(const auto& [name, track] : scene.timeline.tracks) {
        std::string msg = "[Anim] Track: " + name + " Keys: " + std::to_string(track.keyframes.size());
        SCENE_LOG_INFO(msg);
        
        // Debug first keyframe flags
        if (!track.keyframes.empty()) {
            const auto& k = track.keyframes[0];
            if (k.has_light) {
                std::string l_msg = "  - Light Keyframe 0: Pos=" + std::to_string(k.light.has_position) + 
                                    " Int=" + std::to_string(k.light.has_intensity);
                SCENE_LOG_INFO(l_msg);
            }
        }
    }

    return true;
}
