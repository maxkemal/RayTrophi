#include "SceneSerializer.h"
#include "globals.h"
#include "Renderer.h"
#include "OptixWrapper.h"
#include "Triangle.h"
#include "PointLight.h"
#include "DirectionalLight.h"
#include "SpotLight.h"
#include "AreaLight.h"
#include "TerrainManager.h"
#include "MaterialManager.h"
#include "InstanceManager.h"
#include "WaterSystem.h"
#include "Light.h"
#include "Transform.h"
#include "Material.h"
#include "MeshModifiers.h"
#include "json.hpp"
#include "simdjson.h"
#include <fstream>
#include <filesystem>
#include <iostream>
#include <algorithm>
#include "Backend/IBackend.h"
#include "Backend/OptixBackend.h"
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

// simdjson helpers
static json sjsonToNlohmann(simdjson::dom::element el) {
    return json::parse(std::string(simdjson::minify(el)));
}
static json sjsonToNlohmann(simdjson::simdjson_result<simdjson::dom::element> res) {
    simdjson::dom::element el;
    if (res.get(el)) return json();
    return sjsonToNlohmann(el);
}

static Vec3 sjsonToVec3(simdjson::dom::element el) {
    simdjson::dom::array arr;
    if (el.get_array().get(arr)) return Vec3(0, 0, 0);
    float x = 0, y = 0, z = 0;
    size_t i = 0;
    for (simdjson::dom::element val : arr) {
        double d = 0;
        val.get(d);
        if (i == 0) x = (float)d;
        else if (i == 1) y = (float)d;
        else if (i == 2) z = (float)d;
        i++;
    }
    return Vec3(x, y, z);
}
static Vec3 sjsonToVec3(simdjson::simdjson_result<simdjson::dom::element> res) {
    simdjson::dom::element el;
    if (res.get(el)) return Vec3(0, 0, 0);
    return sjsonToVec3(el);
}

static Matrix4x4 sjsonToMat4(simdjson::dom::element el) {
    Matrix4x4 m;
    simdjson::dom::array arr;
    if (el.get_array().get(arr) || arr.size() != 16) return m;
    int idx = 0;
    for (simdjson::dom::element val : arr) {
        int r = idx / 4;
        int c = idx % 4;
        double d = 0;
        val.get(d);
        m.m[r][c] = (float)d;
        idx++;
    }
    return m;
}
static Matrix4x4 sjsonToMat4(simdjson::simdjson_result<simdjson::dom::element> res) {
    simdjson::dom::element el;
    if (res.get(el)) return Matrix4x4();
    return sjsonToMat4(el);
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
    json root;

    // 1. Metadata
    extern std::string active_model_path;
    root["model_path"] = active_model_path;

    // 2. Camera
    if (scene.camera) {
        root["camera"]["lookfrom"] = vec3ToJson(scene.camera->lookfrom);
        root["camera"]["lookat"] = vec3ToJson(scene.camera->lookat);
        root["camera"]["vup"] = vec3ToJson(scene.camera->vup);
        root["camera"]["vfov"] = scene.camera->vfov;
        root["camera"]["aperture"] = scene.camera->aperture;
        root["camera"]["focus_dist"] = scene.camera->focus_dist;
    }

    // 3. Lights
    root["lights"] = json::array();
    for (const auto& l : scene.lights) {
        json lj;
        lj["type"] = (int)l->type();
        lj["position"] = vec3ToJson(l->position);
        lj["color"] = vec3ToJson(l->color);
        lj["intensity"] = 1.0f; // Simplified for now as intensity is often baked in color
        lj["name"] = l->nodeName;

        if (l->type() == LightType::Point) {
            lj["radius"] = std::static_pointer_cast<PointLight>(l)->radius;
        } else if (l->type() == LightType::Directional) {
            lj["direction"] = vec3ToJson(std::static_pointer_cast<DirectionalLight>(l)->direction);
            lj["radius"] = std::static_pointer_cast<DirectionalLight>(l)->radius;
        } else if (l->type() == LightType::Spot) {
            auto sl = std::static_pointer_cast<SpotLight>(l);
            lj["direction"] = vec3ToJson(sl->direction);
            lj["radius"] = sl->radius;
            lj["angle"] = sl->getAngleDegrees();
            lj["falloff"] = sl->getFalloff();
        } else if (l->type() == LightType::Area) {
            auto al = std::static_pointer_cast<AreaLight>(l);
            lj["u"] = vec3ToJson(al->u);
            lj["v"] = vec3ToJson(al->v);
            lj["width"] = al->width;
            lj["height"] = al->height;
        }
        root["lights"].push_back(lj);
    }

    // 4. Objects (Transforms only for scene serializer)
    root["objects"] = json::array();
    for (const auto& obj : scene.world.objects) {
        auto tri = std::dynamic_pointer_cast<Triangle>(obj);
        if (tri) {
            json oj;
            oj["name"] = tri->nodeName;
            oj["material_id"] = tri->getMaterialID();
            auto th = tri->getTransformHandle();
            if (th) oj["transform"] = mat4ToJson(th->base);
            root["objects"].push_back(oj);
        }
    }

    // 5. Settings
    root["settings"]["quality_preset"] = (int)settings.quality_preset;
    root["settings"]["samples_per_pixel"] = settings.samples_per_pixel;
    root["settings"]["max_bounces"] = settings.max_bounces;
    root["settings"]["use_adaptive"] = settings.use_adaptive_sampling;
    root["settings"]["use_denoiser"] = settings.use_denoiser;
    root["settings"]["use_optix"] = settings.use_optix;
    root["settings"]["persistent_tonemap"] = settings.persistent_tonemap;

    // 6. PostFX
    const auto& pp = scene.color_processor.params;
    root["postfx"]["exposure"] = pp.global_exposure;
    root["postfx"]["gamma"] = pp.global_gamma;
    root["postfx"]["saturation"] = pp.saturation;
    root["postfx"]["color_temperature"] = pp.color_temperature;
    root["postfx"]["tone_mapping"] = (int)pp.tone_mapping_type;
    root["postfx"]["vignette_enabled"] = pp.enable_vignette;
    root["postfx"]["vignette_strength"] = pp.vignette_strength;

    // 7. World
    root["world"]["background_color"] = vec3ToJson(scene.background_color);

    // 8. Timeline
    json timelineJson;
    scene.timeline.serialize(timelineJson);
    root["timeline"] = timelineJson;

    // 8.5 Modifiers
    root["mesh_modifiers"] = json::object();
    for (const auto& [nodeName, stack] : scene.mesh_modifiers) {
        if (!stack.modifiers.empty()) {
            json stackJson;
            stack.serialize(stackJson);
            root["mesh_modifiers"][nodeName] = stackJson;
        }
    }

    // 9. Terrains
    auto abs_path = std::filesystem::absolute(filepath);
    std::string terrainDir = abs_path.parent_path().string();
    root["terrain_system"] = TerrainManager::getInstance().serialize(terrainDir);

    // Write to file with optimized buffer
    std::ofstream out(filepath);
    if (out.is_open()) {
        // Optimization: Use dump(0) for fastest serialization and smallest file size
        // Indentation is only for readability, which isn't needed for auto-saves or performance
        out << root.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
        out.close();
    }
}

bool SceneSerializer::Deserialize(SceneData& scene, RenderSettings& settings, Renderer& renderer, Backend::IBackend* backend, const std::string& filepath) {
    simdjson::dom::parser parser;
    simdjson::dom::element root;
    
    auto error = parser.load(filepath).get(root);
    if (error) {
        SCENE_LOG_ERROR("Failed to parse JSON scene file with simdjson: " + std::string(simdjson::error_message(error)));
        return false;
    }

    // 1. Load Base Scene (Model or Default)
    extern std::string active_model_path;
    std::string model_path = "Untitled";
    {
        std::string_view path_sv;
        if (!root["model_path"].get(path_sv)) model_path = std::string(path_sv);
    }
    active_model_path = model_path;

    // Clear current scene
    scene.clear();
    renderer.resetCPUAccumulation();
    if (backend) backend->resetAccumulation();

    // 2. Apply Camera
    simdjson::dom::element cam;
    if (!root["camera"].get(cam)) {
        if (!scene.camera) {
             scene.camera = std::make_shared<Camera>(Vec3(0,0,5), Vec3(0,0,0), Vec3(0,1,0), 60.0, 1.0, 0.0, 1.0, 5);
        }
        scene.camera->lookfrom = sjsonToVec3(cam["lookfrom"]);
        scene.camera->lookat = sjsonToVec3(cam["lookat"]);
        scene.camera->vup = sjsonToVec3(cam["vup"]);
        
        double vfov = 60.0, aperture = 0.0, focus_dist = 10.0;
        cam["vfov"].get(vfov);
        cam["aperture"].get(aperture);
        cam["focus_dist"].get(focus_dist);
        
        scene.camera->vfov = vfov;
        scene.camera->aperture = (float)aperture;
        scene.camera->focus_dist = (float)focus_dist;
        scene.camera->update_camera_vectors();
    }

    // 3. Apply Lights
    simdjson::dom::array lights;
    if (!root["lights"].get(lights)) {
        scene.lights.clear();
        for (auto l : lights) {
            std::shared_ptr<Light> lightPtr = nullptr;
            int64_t type = 0;
            l["type"].get(type);
            
            Vec3 pos = sjsonToVec3(l["position"]);
            Vec3 col = sjsonToVec3(l["color"]);
            double intensity = 1.0;
            l["intensity"].get(intensity);
            
            std::string_view name_sv = "Light";
            l["name"].get(name_sv);
            std::string name(name_sv);
            
            if (type == (int)LightType::Point) {
                double r = 0.1;
                l["radius"].get(r);
                lightPtr = std::make_shared<PointLight>(pos, col * (float)intensity, (float)r);
            } else if (type == (int)LightType::Directional) {
                Vec3 dir = sjsonToVec3(l["direction"]);
                double r = 0.0;
                l["radius"].get(r);
                auto dl = std::make_shared<DirectionalLight>(dir, col * (float)intensity, (float)r);
                dl->position = pos; 
                lightPtr = dl;
            } else if (type == (int)LightType::Spot) {
                Vec3 dir = sjsonToVec3(l["direction"]);
                double range = 20.0, angle = 45.0, falloff = 0.5;
                l["radius"].get(range);
                l["angle"].get(angle);
                l["falloff"].get(falloff);
                auto sl = std::make_shared<SpotLight>(pos, dir, col * (float)intensity, (float)angle, (float)range);
                sl->setFalloff((float)falloff);
                lightPtr = sl;
            } else if (type == (int)LightType::Area) {
                Vec3 u = sjsonToVec3(l["u"]);
                Vec3 v = sjsonToVec3(l["v"]);
                double w = 1.0, h = 1.0;
                l["width"].get(w);
                l["height"].get(h);
                lightPtr = std::make_shared<AreaLight>(pos, u, v, (float)w, (float)h, col * (float)intensity);
            }
            
            if (lightPtr) {
                lightPtr->nodeName = name;
                scene.lights.push_back(lightPtr);
            }
        }
    }

    // 4. Transform Object Updates
    simdjson::dom::array objs;
    if (!root["objects"].get(objs)) {
        size_t limit = std::min(scene.world.objects.size(), objs.size());
        size_t idx = 0;
        for (auto o : objs) {
            if (idx >= scene.world.objects.size()) break;
            auto tri = std::dynamic_pointer_cast<Triangle>(scene.world.objects[idx]);
            if (tri) {
                auto th = tri->getTransformHandle(); 
                if (!th) {
                     th = std::make_shared<Transform>();
                     tri->setTransformHandle(th);
                }
                
                simdjson::dom::element trans;
                if (!o["transform"].get(trans)) {
                    th->setBase(sjsonToMat4(trans));
                }
                
                std::string_view obj_name;
                if (!o["name"].get(obj_name)) tri->nodeName = std::string(obj_name);
                
                int64_t mat_id = 0;
                if (!o["material_id"].get(mat_id)) tri->setMaterialID((int)mat_id);

                tri->updateTransformedVertices();
            }
            idx++;
        }
    }

    // 5. Render Settings
    simdjson::dom::element s;
    if (!root["settings"].get(s)) {
        int64_t q = 0, spp = 1, bounces = 10;
        bool adaptive = true, denoiser = false, optix = true, tonemap = false;
        
        s["quality_preset"].get(q);
        s["samples_per_pixel"].get(spp);
        s["max_bounces"].get(bounces);
        s["use_adaptive"].get(adaptive);
        s["use_denoiser"].get(denoiser);
        s["use_optix"].get(optix);
        s["persistent_tonemap"].get(tonemap);

        settings.quality_preset = (QualityPreset)q;
        settings.samples_per_pixel = (int)spp;
        settings.max_bounces = (int)bounces;
        settings.use_adaptive_sampling = adaptive;
        settings.use_denoiser = denoiser;
        settings.use_optix = optix;
        settings.persistent_tonemap = tonemap;
    }

    // 6. PostFX
    simdjson::dom::element pf;
    if (!root["postfx"].get(pf)) {
        auto& pp = scene.color_processor.params;
        double exp = 1.0, gam = 2.2, sat = 1.0, temp = 6500.0, vig_s = 0.0;
        int64_t tone = 4;
        bool vig_e = false;

        pf["exposure"].get(exp);
        pf["gamma"].get(gam);
        pf["saturation"].get(sat);
        pf["color_temperature"].get(temp);
        pf["tone_mapping"].get(tone);
        pf["vignette_enabled"].get(vig_e);
        pf["vignette_strength"].get(vig_s);

        pp.global_exposure = (float)exp;
        pp.global_gamma = (float)gam;
        pp.saturation = (float)sat;
        pp.color_temperature = (float)temp;
        pp.tone_mapping_type = (ToneMappingType)tone;
        pp.enable_vignette = vig_e;
        pp.vignette_strength = (float)vig_s;
    }

    // 7. World
    simdjson::dom::element w;
    if (!root["world"].get(w)) {
        scene.background_color = sjsonToVec3(w["background_color"]);
    }

    // 8. Timeline (Requires nlohmann::json for now as it's complex)
    simdjson::dom::element t;
    if (!root["timeline"].get(t)) {
        scene.timeline.deserialize(sjsonToNlohmann(t));
    }
    
    // 9. Terrains
    simdjson::dom::element ts;
    if (!root["terrain_system"].get(ts)) {
        // Purge zombie chunks
        auto& objs_list = scene.world.objects;
        size_t purged_count = 0;
        for (auto it = objs_list.begin(); it != objs_list.end(); ) {
            auto tri = std::dynamic_pointer_cast<Triangle>(*it);
            if (tri && tri->nodeName.find("Terrain_") == 0 && tri->nodeName.find("_Chunk") != std::string::npos) {
                it = objs_list.erase(it);
                purged_count++;
            } else {
                ++it;
            }
        }
        if (purged_count > 0) SCENE_LOG_INFO("[SceneSerializer] Purged " + std::to_string(purged_count) + " zombie terrain chunks.");

        TerrainManager::getInstance().removeAllTerrains(scene);

        auto abs_path = std::filesystem::absolute(filepath);
        std::string terrainDir = abs_path.parent_path().string();
        
        TerrainManager::getInstance().deserialize(sjsonToNlohmann(ts), terrainDir, scene);
    }
    
    // 9.5 Modifiers
    scene.mesh_modifiers.clear();
    scene.base_mesh_cache.clear();
    simdjson::dom::element modRoot;
    if (!root["mesh_modifiers"].get(modRoot)) {
        // Convert to nlohmann::json for easy parsing using our existing function
        nlohmann::json nlohmannMods = sjsonToNlohmann(modRoot);
        for (auto it = nlohmannMods.begin(); it != nlohmannMods.end(); ++it) {
            std::string nodeName = it.key();
            MeshModifiers::ModifierStack stack;
            stack.deserialize(it.value());
            scene.mesh_modifiers[nodeName] = stack;
        }

        // We must lazily build base_mesh_cache for these modifiers, evaluate them and set new objects
        for(const auto& [nodeName, stack] : scene.mesh_modifiers) {
            std::vector<std::shared_ptr<Triangle>> baseTriangles;
            std::vector<std::shared_ptr<Hittable>> remainingObjects;

            for (const auto& obj : scene.world.objects) {
                if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
                    if (tri->getNodeName() == nodeName) {
                        baseTriangles.push_back(tri);
                    } else {
                        remainingObjects.push_back(obj);
                    }
                } else {
                    remainingObjects.push_back(obj);
                }
            }

            if (!baseTriangles.empty() && !stack.modifiers.empty()) {
                scene.base_mesh_cache[nodeName] = baseTriangles;
                auto newMesh = stack.evaluate(baseTriangles);
                for (const auto& tri : newMesh) {
                    remainingObjects.push_back(tri); // tri is a std::shared_ptr<Triangle> matching new mesh
                }
                scene.world.objects = remainingObjects;
            }
        }
    }

    // 10. Rebuild All
    renderer.rebuildBVH(scene, settings.UI_use_embree);
    
    if (backend) {
        renderer.rebuildBackendGeometry(scene);
        backend->setLights(scene.lights);
        if (scene.camera) {
            renderer.syncCameraToBackend(*scene.camera);
        }
    }

    SCENE_LOG_INFO("Scene loaded with turbo parser from: " + filepath);
    return true;
}
