#include "TerrainSatMapNodes.h"
#include "UI/imgui_setup.h"
#include <algorithm>
#include <cmath>

namespace rtapi {
namespace nodes {

    // --- TerrainSatMapColorRampNode ---

    void TerrainSatMapColorRampNode::sortStops() {
        std::sort(stops.begin(), stops.end(), [](const Stop& a, const Stop& b) {
            return a.pos < b.pos;
        });
    }

    bool TerrainSatMapColorRampNode::evaluate(NodeSystem::EvaluationContext& ctx) {
        auto inputImg = getImageInput(0, ctx);
        if (!inputImg.isValid()) {
            return false;
        }

        sortStops();

        int w = inputImg.width;
        int h = inputImg.height;
        auto outputData = std::make_shared<std::vector<float>>(w * h * 4); // RGBA

        const float* inData = inputImg.data->data();
        float* outData = outputData->data();

        for (int i = 0; i < w * h; ++i) {
            float val = inData[i]; 
            
            // clamped value
            val = std::max(0.0f, std::min(1.0f, val));

            // Evaluate ramp
            float r = 0, g = 0, b = 0, a = 1;
            if (stops.empty()) {
                r = g = b = val;
            } else if (val <= stops.front().pos) {
                r = stops.front().r; g = stops.front().g; b = stops.front().b; a = stops.front().a;
            } else if (val >= stops.back().pos) {
                r = stops.back().r; g = stops.back().g; b = stops.back().b; a = stops.back().a;
            } else {
                for (size_t s = 0; s < stops.size() - 1; ++s) {
                    if (val >= stops[s].pos && val <= stops[s+1].pos) {
                        float t = (val - stops[s].pos) / (stops[s+1].pos - stops[s].pos);
                        r = stops[s].r + t * (stops[s+1].r - stops[s].r);
                        g = stops[s].g + t * (stops[s+1].g - stops[s].g);
                        b = stops[s].b + t * (stops[s+1].b - stops[s].b);
                        a = stops[s].a + t * (stops[s+1].a - stops[s].a);
                        break;
                    }
                }
            }

            outData[i * 4 + 0] = r;
            outData[i * 4 + 1] = g;
            outData[i * 4 + 2] = b;
            outData[i * 4 + 3] = a;
        }

        NodeSystem::Image2DData outImg;
        outImg.data = outputData;
        outImg.width = w;
        outImg.height = h;
        outImg.channels = 4;
        outImg.semantic = NodeSystem::ImageSemantic::Albedo;

        setOutputValue(0, outImg, ctx);
        return true;
    }

    json TerrainSatMapColorRampNode::saveCustomData() const {
        json j;
        json jStops = json::array();
        for (const auto& s : stops) {
            jStops.push_back({
                {"pos", s.pos},
                {"r", s.r}, {"g", s.g}, {"b", s.b}, {"a", s.a}
            });
        }
        j["stops"] = jStops;
        return j;
    }

    void TerrainSatMapColorRampNode::loadCustomData(const json& j) {
        if (j.contains("stops") && j["stops"].is_array()) {
            stops.clear();
            for (const auto& sj : j["stops"]) {
                Stop s;
                s.pos = sj.value("pos", 0.0f);
                s.r = sj.value("r", 1.0f);
                s.g = sj.value("g", 1.0f);
                s.b = sj.value("b", 1.0f);
                s.a = sj.value("a", 1.0f);
                stops.push_back(s);
            }
            sortStops();
        }
    }

    void TerrainSatMapColorRampNode::drawCustomUI() {
        ImGui::Text("Color Ramp (Edit in Properties panel)");
    }

    // --- TerrainSatMapOutputNode ---

    bool TerrainSatMapOutputNode::evaluate(NodeSystem::EvaluationContext& ctx) {
        if (!publicationEnabled) return true; // Output is skipped

        auto colorImg = getImageInput(0, ctx);
        if (!colorImg.isValid()) return false;

        NodeSystem::PinValue strVal = getInputValue(1, ctx);
        if (auto* pf = std::get_if<float>(&strVal)) {
            strength = *pf;
        }

        TerrainContext* tCtx = getTerrainContext(ctx);
        if (!tCtx || !tCtx->terrainObj) return false;

        TerrainObject* terrain = tCtx->terrainObj;

        // Convert the float RGBA array from colorImg into the macroColorMap texture
        int w = colorImg.width;
        int h = colorImg.height;

        if (!terrain->macroColorMap) {
            terrain->macroColorMap = std::make_shared<Texture>(nullptr, TextureType::Albedo, "MacroColorMap");
        }
        terrain->macroColorMap->width = w;
        terrain->macroColorMap->height = h;
        terrain->macroColorMap->channels = 4;
        
        // Populate CompactVec4 pixels for GPU upload compatibility
        auto& pixels = terrain->macroColorMap->pixels;
        pixels.resize(w * h);

        const float* inData = colorImg.data->data();
        for (int i = 0; i < w * h; ++i) {
            pixels[i].r = static_cast<uint8_t>(std::max(0.0f, std::min(1.0f, inData[i*4+0])) * 255.0f);
            pixels[i].g = static_cast<uint8_t>(std::max(0.0f, std::min(1.0f, inData[i*4+1])) * 255.0f);
            pixels[i].b = static_cast<uint8_t>(std::max(0.0f, std::min(1.0f, inData[i*4+2])) * 255.0f);
            pixels[i].a = static_cast<uint8_t>(std::max(0.0f, std::min(1.0f, inData[i*4+3])) * 255.0f);
        }

        terrain->macroColorMap->m_uid = Texture::nextUid(); // Force GPU texture reload
        terrain->macro_color_strength = strength;
        
        terrain->markPaintMapsDirty(); // Ensure the renderer catches this update

        return true;
    }

    json TerrainSatMapOutputNode::saveCustomData() const {
        json j;
        j["strength"] = strength;
        return j;
    }

    void TerrainSatMapOutputNode::loadCustomData(const json& j) {
        strength = j.value("strength", 1.0f);
    }

    void TerrainSatMapOutputNode::drawCustomUI() {
        ImGui::DragFloat("Strength", &strength, 0.01f, 0.0f, 1.0f);
    }

} // namespace nodes
} // namespace rtapi

namespace {
    NodeSystem::AutoRegisterNode<rtapi::nodes::TerrainSatMapColorRampNode> reg_SatMapColorRamp("Terrain.SatMapColorRamp");
    NodeSystem::AutoRegisterNode<rtapi::nodes::TerrainSatMapOutputNode>    reg_SatMapOutput("Terrain.SatMapOutput");
}
