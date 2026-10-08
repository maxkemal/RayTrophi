#pragma once

#include "scene_data.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <exception>
#include <string>

namespace ProjectSimulationCache {

inline std::string normalizedPath(const std::string& value) {
    if (value.empty()) {
        return {};
    }
    std::error_code error;
    auto path = std::filesystem::weakly_canonical(value, error);
    if (error) {
        path = std::filesystem::path(value).lexically_normal();
    }
    auto result = path.generic_string();
#ifdef _WIN32
    std::transform(result.begin(), result.end(), result.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
#endif
    return result;
}

inline bool changesCacheOwner(const std::string& source, const std::string& target) {
    return normalizedPath(SceneData::simCacheDirForProject(source)) !=
        normalizedPath(SceneData::simCacheDirForProject(target));
}

inline bool canSaveAs(const SceneData& scene, const std::string& source,
                      const std::string& target, bool as_copy) {
    if (!as_copy && changesCacheOwner(source, target) && scene.isSimulationBaking()) {
        SCENE_LOG_ERROR("Save As requires the active simulation bake to finish or cancel.");
        return false;
    }
    return true;
}

inline void afterSuccessfulSave(SceneData& scene, const std::string& source,
                                const std::string& target, bool as_copy) {
    if (as_copy || !changesCacheOwner(source, target)) {
        return;
    }
    // Detach only: never clear, move or overwrite the source project's files.
    // Delete-cache controls must no longer point at the previous project's bake.
    scene.clearSimDiskCacheBinding();
    const auto directory = SceneData::simCacheDirForProject(target);
    std::error_code error;
    if (!std::filesystem::exists(std::filesystem::path(directory) / "manifest.json", error)) {
        return;
    }
    try {
        scene.setSimDiskCache(directory);
    } catch (const std::exception& exception) {
        scene.clearSimDiskCacheBinding();
        SCENE_LOG_WARN(std::string("Save As: target simulation cache could not be bound: ") +
                       exception.what());
    }
}

} // namespace ProjectSimulationCache
