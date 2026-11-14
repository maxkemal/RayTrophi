#pragma once

#include "Api/RtApi.h"
#include "json.hpp"

#include <optional>
#include <filesystem>
#include <string>

struct UIContext;

namespace AppRuntime {

std::filesystem::path projectPathFromUtf8(const std::string& path);
std::string projectPathToUtf8(const std::filesystem::path& path);

// Main-thread extension of the existing project.open operation. Studio keeps
// its editor finalization; Player installs its scene-viewer finalization.
class ProjectHost {
public:
    virtual ~ProjectHost() = default;
    virtual rtapi::Result openProject(const std::string& path) = 0;
    virtual nlohmann::json status() const = 0;
};

void bindProjectHost(UIContext& context, ProjectHost& host);
void unbindProjectHost(ProjectHost& host);
std::optional<rtapi::Result> tryOpenRuntimeProject(UIContext& context,
                                                const std::string& path);
nlohmann::json runtimeStatus(UIContext& context);

} // namespace AppRuntime
