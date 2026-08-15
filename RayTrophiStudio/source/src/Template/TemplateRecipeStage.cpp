#include "Template/TemplateRecipeStage.h"

#include "Camera.h"
#include "PointLight.h"
#include "PrincipledBSDF.h"
#include "Transform.h"
#include "TriangleMesh.h"
#include "Vec2.h"
#include "json.hpp"

#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <utility>

namespace raytrophi::templates {
namespace {

using json = nlohmann::json;

std::shared_ptr<TriangleMesh> stageCube() {
    auto mesh = std::make_shared<TriangleMesh>();
    mesh->nodeName = "Default_Cube";
    mesh->transform = std::make_shared<Transform>();
    mesh->geometry->add_attribute<Vec3>("P");
    mesh->geometry->add_attribute<Vec3>("N");
    mesh->geometry->add_attribute<Vec3>("P_orig");
    mesh->geometry->add_attribute<Vec3>("N_orig");
    mesh->geometry->add_attribute<Vec2>("uv");
    mesh->geometry->add_attribute<uint16_t>("materialID");

    // Match Add > Cube exactly: 36 independent corners and the same 4x3 UV
    // cross atlas. This avoids sharing corners across UV seams and keeps a
    // template cube interchangeable with an authored procedural cube.
    const Vec3 corners[8] = {
        {-1,-1, 1}, { 1,-1, 1}, { 1, 1, 1}, {-1, 1, 1},
        {-1,-1,-1}, { 1,-1,-1}, { 1, 1,-1}, {-1, 1,-1}};
    const uint32_t corner_indices[36] = {
        0,1,2, 2,3,0, 1,5,6, 6,2,1, 7,6,5, 5,4,7,
        4,0,3, 3,7,4, 4,5,1, 1,0,4, 3,2,6, 6,7,3};
    constexpr float w = 0.25f;
    constexpr float h = 1.0f / 3.0f;
    const Vec2 cube_uvs[36] = {
        {w,h},{2*w,h},{2*w,2*h}, {2*w,2*h},{w,2*h},{w,h},
        {2*w,h},{3*w,h},{3*w,2*h}, {3*w,2*h},{2*w,2*h},{2*w,h},
        {4*w,2*h},{3*w,2*h},{3*w,h}, {3*w,h},{4*w,h},{4*w,2*h},
        {0,h},{w,h},{w,2*h}, {w,2*h},{0,2*h},{0,h},
        {w,0},{2*w,0},{2*w,h}, {2*w,h},{w,h},{w,0},
        {w,2*h},{2*w,2*h},{2*w,3*h}, {2*w,3*h},{w,3*h},{w,2*h}};

    std::vector<Vec3> positions;
    std::vector<Vec3> normals;
    std::vector<Vec2> uvs;
    positions.reserve(36);
    normals.reserve(36);
    uvs.reserve(36);
    auto& indices = mesh->geometry->indices;
    indices.reserve(36);
    for (uint32_t face = 0; face < 12; ++face) {
        const uint32_t base = face * 3;
        const Vec3 v0 = corners[corner_indices[base]];
        const Vec3 v1 = corners[corner_indices[base + 1]];
        const Vec3 v2 = corners[corner_indices[base + 2]];
        const Vec3 normal = (v1 - v0).cross(v2 - v0).normalize();
        for (uint32_t corner = 0; corner < 3; ++corner) {
            positions.push_back(corners[corner_indices[base + corner]]);
            normals.push_back(normal);
            uvs.push_back(cube_uvs[base + corner]);
            indices.push_back(base + corner);
        }
    }

    mesh->geometry->resize_vertices(positions.size());
    std::memcpy(mesh->geometry->get_attribute_data_mut<Vec3>("P"), positions.data(), positions.size() * sizeof(Vec3));
    std::memcpy(mesh->geometry->get_attribute_data_mut<Vec3>("P_orig"), positions.data(), positions.size() * sizeof(Vec3));
    std::memcpy(mesh->geometry->get_attribute_data_mut<Vec3>("N"), normals.data(), normals.size() * sizeof(Vec3));
    std::memcpy(mesh->geometry->get_attribute_data_mut<Vec3>("N_orig"), normals.data(), normals.size() * sizeof(Vec3));
    std::memcpy(mesh->geometry->get_attribute_data_mut<Vec2>("uv"), uvs.data(), uvs.size() * sizeof(Vec2));
    std::memset(mesh->geometry->get_attribute_data_mut<uint16_t>("materialID"), 0,
                positions.size() * sizeof(uint16_t));
    mesh->build_local_bvh();
    return mesh;
}

std::shared_ptr<Material> stageDefaultCubeMaterial() {
    auto material = std::make_shared<PrincipledBSDF>();
    material->materialName = "Default_Cube_Material";
    auto gpu = std::make_shared<GpuMaterial>();
    gpu->albedo = make_float3(0.8f, 0.8f, 0.8f);
    gpu->roughness = 0.5f;
    gpu->metallic = 0.0f;
    gpu->transmission = 0.0f;
    gpu->emission = make_float3(0.0f, 0.0f, 0.0f);
    gpu->opacity = 1.0f;
    gpu->ior = 1.5f;
    gpu->anisotropic = 0.0f;
    gpu->clearcoat = 0.0f;
    material->gpuMaterial = std::move(gpu);
    return material;
}

std::shared_ptr<Camera> stageCamera(bool general) {
    const Vec3 position = general ? Vec3(11.0f, 8.0f, 14.0f) : Vec3(5, 4, 6);
    const Vec3 target = general ? Vec3(0, 0.25f, 0) : Vec3(0, 0, 0);
    // Keep the camera's focus plane exactly on its look target. A fixed focus
    // distance left the procedural camera's center ray short of the staged
    // cube, while default.glb derives this distance from its authored camera.
    const float focus_distance = (position - target).length();
    auto camera = std::make_shared<Camera>(
        position, target, Vec3(0, 1, 0), 40.0f, 16.0f / 9.0f,
        0.0f, general ? focus_distance : 9.0f, 0);
    camera->nodeName = general ? "Camera" : "Viewport Camera";
    camera->update_camera_vectors();
    return camera;
}

void buildEmpty(TemplateRecipeStage& result) {
    result.camera = stageCamera(false);
}

void buildGeneralScene(TemplateRecipeStage& result) {
    result.mesh = stageCube();
    result.material = stageDefaultCubeMaterial();
    result.camera = stageCamera(true);
    auto light = std::make_shared<PointLight>(Vec3(4, 5, -3), Vec3(1, 1, 1), 0.25f);
    light->nodeName = "Key_Light";
    light->intensity = 100.0f;
    result.light = std::move(light);
}

struct PresetBuilder {
    const char* name;
    void (*build)(TemplateRecipeStage&);
};

// The single source of truth for "which presets exist". Both the preflight
// answer and the commit dispatch read this table; adding a row is the only
// step needed to support a new preset.
const PresetBuilder kPresetBuilders[] = {
    {"empty", &buildEmpty},
    {"general_scene", &buildGeneralScene},
};

const PresetBuilder* findPresetBuilder(const std::string& preset) {
    for (const auto& entry : kPresetBuilders) {
        if (preset == entry.name) return &entry;
    }
    return nullptr;
}

} // namespace

const std::vector<std::string>& TemplateRecipeStager::supportedPresets() {
    static const std::vector<std::string> presets = [] {
        std::vector<std::string> names;
        names.reserve(std::size(kPresetBuilders));
        for (const auto& entry : kPresetBuilders) names.emplace_back(entry.name);
        return names;
    }();
    return presets;
}

bool TemplateRecipeStager::supportsPreset(const std::string& preset) {
    return findPresetBuilder(preset) != nullptr;
}

TemplateRecipeStage TemplateRecipeStager::stage(const std::filesystem::path& recipe_path) {
    TemplateRecipeStage result;
    json recipe;
    try {
        std::ifstream stream(recipe_path);
        if (!stream) throw std::runtime_error("recipe cannot be opened");
        stream >> recipe;
    } catch (const std::exception& error) {
        result.errors.push_back(error.what());
        return result;
    }

    try {
        result.preset = recipe.value("preset", std::string());
        const PresetBuilder* builder = findPresetBuilder(result.preset);
        if (!builder) {
            result.code = "recipe_commit_not_available";
            result.errors.push_back("recipe commit is not implemented for preset: " + result.preset);
            return result;
        }
        builder->build(result);
    } catch (const std::exception& error) {
        result.code = "recipe_staging_failed";
        result.errors.push_back(error.what());
        return result;
    }
    result.ready = true;
    result.code = "ready";
    return result;
}

} // namespace raytrophi::templates
