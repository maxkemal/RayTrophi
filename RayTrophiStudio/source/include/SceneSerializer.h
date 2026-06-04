/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          SceneSerializer.h
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once

#include <string>
#include <vector>
#include <memory>
#include "scene_data.h"

// Forward define
struct RenderSettings;
class Renderer;
namespace Backend { class IBackend; }

class SceneSerializer {
public:
    // NOTE: SceneSerializer::Serialize is currently NOT used. Saving is handled by ProjectManager.
    // Sahne verilerini ve ayarlarÄ± .rts (JSON) dosyasÄ±na kaydeder
    static void Serialize(const SceneData& scene, const RenderSettings& settings, const std::string& filepath);

    // NOTE: SceneSerializer::Deserialize is only used as a fallback legacy loader (e.g. for non-.rtp files) in scene_ui.cpp.
    // .rts dosyasÄ±ndan sahneyi yÃ¼kler
    // Ã–nce modeli yÃ¼kler (eÄŸer varsa), sonra ayarlarÄ± uygular
    static bool Deserialize(SceneData& scene, RenderSettings& settings, Renderer& renderer, Backend::IBackend* backend, const std::string& filepath);
};

