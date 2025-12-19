#include "SceneSerializer.h"
#include "globals.h"
#include "Renderer.h"
#include "OptixWrapper.h"
#include "default_scene_creator.hpp"
#include "World.h"
#include <fstream>
#include <iostream>
#include "json.hpp"
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

// Helper for float3 to JSON
json float3ToJson(const float3& v) {
    return { v.x, v.y, v.z };
}

float3 jsonToFloat3(const json& j) {
    if (j.is_array() && j.size() >= 3)
        return make_float3(j[0], j[1], j[2]);
    return make_float3(0, 0, 0);
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

    // 1. Scene Info (Model Path)
    extern std::string active_model_path;
    root["model_path"] = active_model_path;

    // 2. Camera
    if (scene.camera) {
        json camJson;
        camJson["lookfrom"] = vec3ToJson(scene.camera->lookfrom);
        camJson["lookat"] = vec3ToJson(scene.camera->lookat);
        camJson["vup"] = vec3ToJson(scene.camera->vup);
        camJson["vfov"] = scene.camera->vfov;
        camJson["aperture"] = scene.camera->aperture;
        camJson["focus_dist"] = scene.camera->focus_dist;
        root["camera"] = camJson;
    }

    // 3. Lights
    json lightsJson = json::array();
    for (const auto& light : scene.lights) {
        json l;
        l["type"] = (int)light->type();
        l["position"] = vec3ToJson(light->position);
        l["color"] = vec3ToJson(light->color);
        l["intensity"] = light->intensity;
        
        if (auto dl = std::dynamic_pointer_cast<DirectionalLight>(light)) {
            l["direction"] = vec3ToJson(dl->direction);
            l["radius"] = dl->radius;
        }
        else if (auto pl = std::dynamic_pointer_cast<PointLight>(light)) {
            l["radius"] = pl->radius;
        }
        else if (auto sl = std::dynamic_pointer_cast<SpotLight>(light)) {
            l["direction"] = vec3ToJson(sl->direction);
            l["radius"] = sl->radius;
            l["angle"] = sl->getAngleDegrees();
            l["falloff"] = sl->getFalloff();
        }
        else if (auto al = std::dynamic_pointer_cast<AreaLight>(light)) {
            l["u"] = vec3ToJson(al->u);
            l["v"] = vec3ToJson(al->v);
            l["width"] = al->width;
            l["height"] = al->height;
        }
        lightsJson.push_back(l);
    }
    root["lights"] = lightsJson;

    // 4. Object Transforms (By Name) - Also save mesh_type for procedural recreation
    json objectsJson = json::array();
    for (const auto& obj : scene.world.objects) {
        auto tri = std::dynamic_pointer_cast<Triangle>(obj);
        if (tri && !tri->nodeName.empty()) {
            json o;
            o["name"] = tri->nodeName;
            // Determine mesh type: "Cube", "Plane", or "Model" (imported)
            std::string meshType = "Model";
            if (tri->nodeName == "Cube") meshType = "Cube";
            else if (tri->nodeName == "Plane") meshType = "Plane";
            o["mesh_type"] = meshType;
            
            auto th = tri->getTransformHandle();
            if (th) {
                o["transform"] = mat4ToJson(th->base);
            }
            o["material_id"] = tri->getMaterialID();
            objectsJson.push_back(o);
        }
    }
    root["objects"] = objectsJson;

    // 5. Render Settings
    json setJson;
    setJson["quality_preset"] = (int)settings.quality_preset;
    setJson["samples_per_pixel"] = settings.samples_per_pixel;
    setJson["max_bounces"] = settings.max_bounces;
    setJson["use_adaptive"] = settings.use_adaptive_sampling;
    setJson["use_denoiser"] = settings.use_denoiser;
    setJson["use_optix"] = settings.use_optix;
    root["settings"] = setJson;

    // 6. PostFX (Color Processing)
    json postfxJson;
    const auto& pp = scene.color_processor.params;
    postfxJson["exposure"] = pp.global_exposure;
    postfxJson["gamma"] = pp.global_gamma;
    postfxJson["saturation"] = pp.saturation;
    postfxJson["color_temperature"] = pp.color_temperature;
    postfxJson["tone_mapping"] = (int)pp.tone_mapping_type;
    postfxJson["vignette_enabled"] = pp.enable_vignette;
    postfxJson["vignette_strength"] = pp.vignette_strength;
    root["postfx"] = postfxJson;

    // 7. World/Environment (from Renderer - we need to get this via extern or parameter)
    // For now we'll save basic background color from scene
    json worldJson;
    worldJson["background_color"] = vec3ToJson(scene.background_color);
    // Note: Full World (HDRI, Nishita Sky) data requires access to Renderer.world
    // This can be extended when Renderer& is passed to Serialize
    root["world"] = worldJson;

    // Write file
    std::ofstream out(filepath);
    if (out.is_open()) {
        out << root.dump(4);
        out.close();
        SCENE_LOG_INFO("Scene saved to: " + filepath);
    } else {
        SCENE_LOG_ERROR("Failed to save scene: " + filepath);
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

    if (model_path == "Untitled" || model_path.empty()) {
        createDefaultScene(scene, renderer, optix_gpu);
    } else {
        renderer.create_scene(scene, optix_gpu, model_path, nullptr); 
    }

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
            
            if (lightPtr) scene.lights.push_back(lightPtr);
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
        settings.max_bounces = s.value("max_bounces", 4);
        settings.use_adaptive_sampling = s.value("use_adaptive", true);
        settings.use_denoiser = s.value("use_denoiser", false);
        settings.use_optix = s.value("use_optix", true);
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
    
    // 8. Rebuild All
    renderer.rebuildBVH(scene, settings.UI_use_embree);
    
    if (optix_gpu) {
        renderer.rebuildOptiXGeometry(scene, optix_gpu);
        optix_gpu->setLightParams(scene.lights);
        if (scene.camera) optix_gpu->setCameraParams(*scene.camera);
    }

    SCENE_LOG_INFO("Scene loaded from: " + filepath);
    return true;
}
