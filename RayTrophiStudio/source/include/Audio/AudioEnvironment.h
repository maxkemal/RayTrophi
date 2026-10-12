#pragma once
#include "json.hpp"
#include <filesystem>
namespace audio {
class AudioService;
nlohmann::json installEnvironment(AudioService& candidate, const std::filesystem::path& assetsRoot);
}
