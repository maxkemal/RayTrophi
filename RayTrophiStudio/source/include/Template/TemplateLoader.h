#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace raytrophi::templates {

enum class TemplateConflictPolicy {
    Reject,
    Discard
};

struct TemplateLoadPlan {
    std::string template_id;
    std::string state = "invalid";
    std::string code = "invalid_template";
    std::string scene_type;
    std::filesystem::path manifest_path;
    std::filesystem::path scene_path;
    std::filesystem::path binary_path;
    std::filesystem::path auxiliary_path;
    bool ready = false;
    bool has_unsaved_changes = false;
    bool requires_discard = false;
    std::vector<std::string> errors;
    std::vector<std::string> warnings;
};

class TemplateLoader {
public:
    static TemplateLoader& instance();

    TemplateLoadPlan prepare(const std::string& template_id,
                             bool has_unsaved_changes,
                             TemplateConflictPolicy policy) const;

    static bool parseConflictPolicy(const std::string& value,
                                    TemplateConflictPolicy& output);
    static const char* conflictPolicyName(TemplateConflictPolicy policy);

private:
    TemplateLoadPlan preflightProject(TemplateLoadPlan plan) const;
    TemplateLoadPlan preflightRecipe(TemplateLoadPlan plan) const;
};

} // namespace raytrophi::templates
