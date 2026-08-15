#include "Template/TemplateLoader.h"

#include "Template/TemplateRecipeStage.h"
#include "Template/TemplateRegistry.h"
#include "json.hpp"

#include <array>
#include <cstring>
#include <fstream>
#include <utility>

namespace raytrophi::templates {
namespace {

using json = nlohmann::json;

void fail(TemplateLoadPlan& plan, const std::string& code, const std::string& error) {
    if (plan.errors.empty()) plan.code = code;
    plan.errors.push_back(error);
    plan.ready = false;
    plan.state = "invalid";
}

bool regularNonEmptyFile(const std::filesystem::path& path) {
    std::error_code ec;
    return std::filesystem::is_regular_file(path, ec) &&
           std::filesystem::file_size(path, ec) > 0 && !ec;
}

// ProjectManager::readGeometryBinary accepts these magics and then reads a
// uint32 version and a uint32 transform count. Those three reads are the only
// cheap, content-level failures we can see without loading the project, and
// they happen *after* openProject has already destroyed the active scene.
bool geometryBinaryHeaderReadable(const std::filesystem::path& path, std::string& error) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        error = "project geometry binary cannot be opened: " + path.generic_string();
        return false;
    }
    char header[12] = {};
    stream.read(header, sizeof(header));
    if (stream.gcount() < static_cast<std::streamsize>(sizeof(header))) {
        error = "project geometry binary is truncated before its header: " + path.generic_string();
        return false;
    }
    static const std::array<const char*, 6> kMagics = {"RTP3", "RTP4", "RTP5",
                                                       "RTP6", "RTP7", "RTP8"};
    bool magic_ok = false;
    for (const char* magic : kMagics) {
        if (std::memcmp(header, magic, 4) == 0) {
            magic_ok = true;
            break;
        }
    }
    if (!magic_ok) {
        error = "project geometry binary has an unrecognized format magic: " +
                path.generic_string();
        return false;
    }
    return true;
}

bool readJsonObject(const std::filesystem::path& path, json& output, std::string& error) {
    try {
        std::ifstream stream(path);
        if (!stream) {
            error = "file cannot be opened: " + path.generic_string();
            return false;
        }
        stream >> output;
        if (!output.is_object()) {
            error = "JSON root must be an object: " + path.generic_string();
            return false;
        }
        return true;
    } catch (const std::exception& exception) {
        error = std::string("JSON parse failed: ") + exception.what();
        return false;
    }
}

} // namespace

TemplateLoader& TemplateLoader::instance() {
    static TemplateLoader loader;
    return loader;
}

bool TemplateLoader::parseConflictPolicy(const std::string& value,
                                         TemplateConflictPolicy& output) {
    if (value == "reject") {
        output = TemplateConflictPolicy::Reject;
        return true;
    }
    if (value == "discard") {
        output = TemplateConflictPolicy::Discard;
        return true;
    }
    return false;
}

const char* TemplateLoader::conflictPolicyName(TemplateConflictPolicy policy) {
    return policy == TemplateConflictPolicy::Discard ? "discard" : "reject";
}

TemplateLoadPlan TemplateLoader::prepare(const std::string& template_id,
                                         bool has_unsaved_changes,
                                         TemplateConflictPolicy policy) const {
    TemplateLoadPlan plan;
    plan.template_id = template_id;
    plan.has_unsaved_changes = has_unsaved_changes;

    const TemplateMetadata* metadata = TemplateRegistry::instance().find(template_id);
    if (!metadata) {
        fail(plan, "template_not_found", "template not found: " + template_id);
        return plan;
    }
    plan.manifest_path = metadata->manifest_path;
    plan.scene_type = metadata->scene_type;
    plan.scene_path = metadata->scene_path;

    if (!metadata->valid) {
        plan.code = "invalid_template";
        plan.errors = metadata->errors;
        plan.state = "invalid";
        return plan;
    }
    if (has_unsaved_changes && policy == TemplateConflictPolicy::Reject) {
        fail(plan, "unsaved_changes",
             "the active project has unsaved changes; use conflict_policy='discard' only after explicit user confirmation");
        plan.state = "conflict";
        return plan;
    }
    plan.requires_discard = has_unsaved_changes && policy == TemplateConflictPolicy::Discard;

    if (metadata->scene_type == "project") plan = preflightProject(std::move(plan));
    else if (metadata->scene_type == "recipe") plan = preflightRecipe(std::move(plan));
    else fail(plan, "unsupported_scene_type", "unsupported scene type: " + metadata->scene_type);

    if (plan.errors.empty()) {
        plan.ready = true;
        plan.state = "ready";
        plan.code = "ready";
    }
    return plan;
}

TemplateLoadPlan TemplateLoader::preflightProject(TemplateLoadPlan plan) const {
    json project;
    std::string error;
    if (!readJsonObject(plan.scene_path, project, error)) {
        fail(plan, "invalid_project", error);
        return plan;
    }

    if (!project.contains("format_version"))
        fail(plan, "invalid_project", "project is missing format_version");

    bool has_geometry = false;
    if (project.contains("has_geometry")) {
        if (project["has_geometry"].is_boolean())
            has_geometry = project["has_geometry"].get<bool>();
        else
            fail(plan, "invalid_project", "project has_geometry must be a boolean");
    }
    std::string format_version;
    if (project.contains("format_version")) {
        if (project["format_version"].is_string())
            format_version = project["format_version"].get<std::string>();
        else if (project["format_version"].is_number())
            // Mirror openProject's numeric normalization exactly: it truncates
            // std::to_string(double) to three characters, so 3.0 becomes "3.0".
            format_version = std::to_string(project["format_version"].get<double>()).substr(0, 3);
        else
            fail(plan, "invalid_project", "project format_version has an invalid type");
    }

    plan.binary_path = std::filesystem::path(plan.scene_path.string() + ".bin");
    const bool has_binary = regularNonEmptyFile(plan.binary_path);

    // openProject only has one loading path: `is_v3 && has_geometry && has_binary`,
    // where is_v3 is an exact "3.0" match. Anything else falls into its legacy
    // branch and returns false — but only *after* newProject() has already wiped
    // the active scene. Preflight must therefore demand the same three
    // conditions, not a looser prefix match, or it reports `ready` for a project
    // whose commit is guaranteed to fail destructively.
    if (format_version != "3.0") {
        fail(plan, "unsupported_project_version",
             "project format_version must be exactly '3.0'; legacy projects cannot be "
             "opened as templates without losing the active scene (found: " +
                 (format_version.empty() ? std::string("<missing>") : format_version) + ")");
    } else if (!has_geometry) {
        fail(plan, "unsupported_project_layout",
             "project declares has_geometry=false, which openProject rejects after "
             "clearing the scene");
    } else if (!has_binary) {
        fail(plan, "missing_project_binary",
             "project geometry binary is missing or empty: " + plan.binary_path.generic_string());
    } else {
        std::string header_error;
        if (!geometryBinaryHeaderReadable(plan.binary_path, header_error))
            fail(plan, "invalid_project_binary", header_error);
    }

    plan.auxiliary_path = std::filesystem::path(plan.scene_path.string() + ".aux.json");
    if (std::filesystem::exists(plan.auxiliary_path)) {
        json auxiliary;
        if (!readJsonObject(plan.auxiliary_path, auxiliary, error))
            fail(plan, "invalid_auxiliary", error);
    } else {
        plan.auxiliary_path.clear();
    }
    return plan;
}

TemplateLoadPlan TemplateLoader::preflightRecipe(TemplateLoadPlan plan) const {
    json recipe;
    std::string error;
    if (!readJsonObject(plan.scene_path, recipe, error)) {
        fail(plan, "invalid_recipe", error);
        return plan;
    }
    const std::string recipe_version =
        recipe.contains("recipe_version") && recipe["recipe_version"].is_string()
            ? recipe["recipe_version"].get<std::string>() : std::string();
    if (recipe_version != "1.0")
        fail(plan, "unsupported_recipe_version", "recipe_version must be '1.0'");
    if (!recipe.contains("preset") || !recipe["preset"].is_string())
        fail(plan, "invalid_recipe", "recipe must contain a string preset");
    else {
        const std::string preset = recipe["preset"].get<std::string>();
        // Ask the committing stager, never a local copy of the preset list.
        // `ready` must mean "open() will accept this"; a second list here would
        // let preflight promise a commit the stager then refuses.
        if (!TemplateRecipeStager::supportsPreset(preset))
            fail(plan, "unsupported_recipe", "unsupported recipe preset: " + preset);
    }
    return plan;
}

} // namespace raytrophi::templates
