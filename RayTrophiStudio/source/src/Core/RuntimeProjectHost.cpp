#include "Core/RuntimeProjectHost.h"

namespace AppRuntime {
namespace {
UIContext* host_context = nullptr;
ProjectHost* project_host = nullptr;
} // namespace

std::filesystem::path projectPathFromUtf8(const std::string& path) {
    return std::filesystem::path(std::u8string(path.begin(), path.end()));
}

std::string projectPathToUtf8(const std::filesystem::path& path) {
    const auto encoded = path.u8string();
    return std::string(encoded.begin(), encoded.end());
}

void bindProjectHost(UIContext& context, ProjectHost& host) {
    host_context = &context;
    project_host = &host;
}

void unbindProjectHost(ProjectHost& host) {
    if (project_host == &host) {
        project_host = nullptr;
        host_context = nullptr;
    }
}

std::optional<rtapi::Result> tryOpenRuntimeProject(UIContext& context,
                                                const std::string& path) {
    if (!project_host || host_context != &context) {
        return std::nullopt;
    }
    return project_host->openProject(path);
}

nlohmann::json runtimeStatus(UIContext& context) {
    if (!project_host || host_context != &context) {
        return {{"host_active", false}, {"host", "studio"}};
    }
    return project_host->status();
}

} // namespace AppRuntime
