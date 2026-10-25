#include "RtApiInternal.h"
#include "Api/GrainGeneratorApi.h"

#include "GrainGenerator.h"
#include "MaterialManager.h"
#include "ProjectManager.h"
#include "Transform.h"
#include "TriangleMesh.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <set>
#include <stdexcept>

namespace rtapi {
namespace {

class GrainLibraryCommand final : public SceneCommand {
public:
    std::vector<std::shared_ptr<TriangleMesh>> meshes;

    void execute(UIContext& ctx) override {
        for (const auto& mesh : meshes) {
            AddObjectCommand(mesh).execute(ctx);
        }

    }

    void undo(UIContext& ctx) override {
        for (auto it = meshes.rbegin(); it != meshes.rend(); ++it) {
            AddObjectCommand(*it).undo(ctx);
        }
    }

    Type getType() const override {
        return Type::Heavy;
    }
    bool isHeavyGeometry() const override {
        return true;
    }
    std::string getDescription() const override {
        return "Generate Grain / Crystal Library";
    }
    std::size_t getTriangleCount() const override {
        std::size_t count = 0;
        for (const auto& mesh : meshes) {
            count += mesh->num_triangles();
        }
        return count;
    }
};

float number(const nlohmann::json& input, const char* key, float fallback,
    float minimum, float maximum) {
    if (!input.contains(key)) {
        return fallback;
    }
    if (!input.at(key).is_number()) {
        throw std::runtime_error(std::string(key) + " must be numeric");
    }
    const float value = input.at(key).get<float>();
    if (!std::isfinite(value) || value < minimum || value > maximum) {
        throw std::runtime_error(std::string(key) + " outside allowed range");
    }
    return value;
}

uint16_t materialId(const std::string& name) {
    auto& manager = MaterialManager::getInstance();
    if (name.empty()) {
        const auto& materials = manager.getAllMaterials();
        for (std::size_t index = 0; index < materials.size(); ++index) {
            if (materials[index]) {
                return static_cast<uint16_t>(index);
            }
        }
    } else {
        const auto id = manager.getMaterialID(name);
        if (id != MaterialManager::INVALID_MATERIAL_ID && manager.getMaterial(id)) {
            return id;
        }
    }
    throw std::runtime_error(name.empty()
        ? "no scene material exists; create a material first"
        : "material not found: " + name);
}

std::shared_ptr<TriangleMesh> flatMesh(const RayTrophiSim::Fluid::GrainExemplar& shape,
    const std::string& name, uint16_t material, float diameter, const Vec3& position) {
    auto mesh = std::make_shared<TriangleMesh>();
    mesh->nodeName = name;
    mesh->transform = std::make_shared<Transform>();
    mesh->transform->setBase(Matrix4x4::translation(position));
    auto& geometry = *mesh->geometry;
    for (const char* key : {"P", "N", "P_orig", "N_orig"}) {
        geometry.add_attribute<Vec3>(key);
    }
    geometry.add_attribute<Vec2>("uv");
    geometry.add_attribute<uint16_t>("materialID");
    const auto count = shape.faces.size() * 3;
    geometry.resize_vertices(count);
    auto* points = geometry.get_attribute_data_mut<Vec3>("P");
    auto* original = geometry.get_attribute_data_mut<Vec3>("P_orig");
    auto* normals = geometry.get_attribute_data_mut<Vec3>("N");
    auto* original_normals = geometry.get_attribute_data_mut<Vec3>("N_orig");
    auto* uv = geometry.get_attribute_data_mut<Vec2>("uv");
    auto* materials = geometry.get_attribute_data_mut<uint16_t>("materialID");
    geometry.indices.resize(count);
    std::size_t index = 0;
    for (const auto& triangle : shape.faces) {
        for (int corner = 0; corner < 3; ++corner, ++index) {
            original[index] = triangle.p[corner] * diameter;
            points[index] = original[index] + position;
            normals[index] = original_normals[index] = triangle.n[corner];
            uv[index] = triangle.uv[corner];
            materials[index] = material;
            geometry.indices[index] = static_cast<uint32_t>(index);
        }
    }
    mesh->build_local_bvh();
    return mesh;
}

} // namespace

Result generateGrains(const nlohmann::json& input, nlohmann::json& output) {
    output = nlohmann::json::object();
    if (!g_ctx) {
        return notBound();
    }
    if (renderJobActive()) {
        return Result::fail("scene is locked by the final render job");
    }
    if (!g_history) {
        return Result::fail("rtapi has no SceneHistory bound");
    }
    RayTrophiSim::GrainGeneratorSettings settings;
    std::string error;
    if (!RayTrophiSim::parseGrainGeneratorSettings(input, settings, error)) {
        return Result::fail(error);
    }
    try {
        const std::string name = input.value("name", std::string("Grain"));
        if (name.empty() || name.size() > 128 || name.front() == '[') {
            return Result::fail("name must be 1..128 characters and cannot start with '['");
        }
        if (name.find("_inst_") != std::string::npos) {
            return Result::fail("name cannot contain the reserved foliage marker '_inst_'");
        }
        auto scalar = [&](const char* key, float fallback, float low, float high) {
            return number(input, key, fallback, low, high);
        };
        const float minimum = scalar("diameter_min", 0.02f, 0.00001f, 100.0f);
        const float maximum = scalar("diameter_max", minimum, minimum, 100.0f);
        const std::string distribution = input.value("size_distribution", "uniform");
        if (distribution != "uniform" && distribution != "log_uniform") {
            return Result::fail("size_distribution must be uniform or log_uniform");
        }
        Vec3 origin(0.0f);
        if (input.contains("origin")) {
            const auto& values = input.at("origin");
            if (!values.is_array() || values.size() != 3) {
                return Result::fail("origin must contain three numbers");
            }
            float* components[] = {&origin.x, &origin.y, &origin.z};
            for (int axis = 0; axis < 3; ++axis) {
                if (!values[axis].is_number()) {
                    return Result::fail("origin must be numeric");
                }
                *components[axis] = values[axis].get<float>();
                if (!std::isfinite(*components[axis]) || std::abs(*components[axis]) > 1e6f) {
                    return Result::fail("origin must be finite and within +/-1e6 metres");
                }
            }
        }
        if (input.contains("material") && input.contains("material_weights")) {
            return Result::fail("choose material or material_weights, not both");
        }
        std::vector<uint16_t> materials;
        std::vector<double> weights;
        if (input.contains("material_weights")) {
            const auto& palette = input.at("material_weights");
            if (!palette.is_array() || palette.empty() || palette.size() > 32) {
                return Result::fail("material_weights must contain 1..32 material/weight entries");
            }
            for (const auto& entry : palette) {
                if (!entry.is_object() || entry.size() != 2 ||
                    !entry.contains("material") || !entry.contains("weight")) {
                    return Result::fail("palette entry needs exactly material and weight");
                }
                materials.push_back(materialId(entry.at("material").get<std::string>()));
                weights.push_back(number(entry, "weight", 1.0f, 0.00001f, 1e6f));
            }
        } else {
            materials.push_back(materialId(input.value("material", std::string{})));
            weights.push_back(1.0);
        }
        auto command = std::make_unique<GrainLibraryCommand>();
        std::vector<RayTrophiSim::Fluid::GrainExemplar> shapes;
        if (!RayTrophiSim::generateGrainShapes(settings, shapes, error)) {
            return Result::fail(error);
        }
        std::mt19937 random(settings.seed ^ 0x51ed270bu);
        std::uniform_real_distribution<float> unit(0.0f, 1.0f);
        std::discrete_distribution<std::size_t> choose_material(weights.begin(), weights.end());
        const int columns = static_cast<int>(
            std::ceil(std::sqrt(static_cast<float>(shapes.size()))));
        std::set<std::string> reserved;
        std::vector<std::string> names;
        auto rows = nlohmann::json::array();
        for (std::size_t index = 0; index < shapes.size(); ++index) {
            std::string candidate = name + "_" + std::to_string(index + 1);
            const std::string base = candidate;
            int suffix = 1;
            while (objectExists(candidate) || reserved.count(candidate) ||
                g_ctx->scene.isEditorPendingDeleteObjectName(candidate)) {
                candidate = base + "." + std::to_string(suffix++);
            }
            reserved.insert(candidate);
            const float sample = unit(random);
            const float diameter = distribution == "uniform"
                ? minimum + sample * (maximum - minimum)
                : std::exp(std::log(minimum) + sample * std::log(maximum / minimum));
            const uint16_t material = materials[choose_material(random)];
            const Vec3 position = origin + Vec3(
                static_cast<float>(index % columns) * maximum * 1.5f,
                0, static_cast<float>(index / columns) * maximum * 1.5f);
            command->meshes.push_back(
                flatMesh(shapes[index], candidate, material, diameter, position));
            names.push_back(candidate);
            rows.push_back({{"object", candidate}, {"diameter", diameter},
                {"material_id", material}, {"triangles", shapes[index].faces.size()}});
        }
        command->execute(*g_ctx);
        g_history->record(std::move(command));
        output = {{"objects", names}, {"requested_count", settings.count},
            {"count", names.size()}, {"type", settings.type}, {"seed", settings.seed},
            {"examples", rows}};
        return Result::success();
    } catch (const std::exception& exception) {
        return Result::fail(exception.what());
    }
}

} // namespace rtapi
