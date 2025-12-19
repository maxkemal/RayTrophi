#pragma once

#include <string>
#include <vector>
#include <memory>
#include "scene_data.h"

// Forward define
struct RenderSettings;
class Renderer;
class OptixWrapper;

class SceneSerializer {
public:
    // Sahne verilerini ve ayarları .rts (JSON) dosyasına kaydeder
    static void Serialize(const SceneData& scene, const RenderSettings& settings, const std::string& filepath);

    // .rts dosyasından sahneyi yükler
    // Önce modeli yükler (eğer varsa), sonra ayarları uygular
    static bool Deserialize(SceneData& scene, RenderSettings& settings, Renderer& renderer, OptixWrapper* optix_gpu, const std::string& filepath);
};
