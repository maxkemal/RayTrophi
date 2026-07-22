/* RayTrophi Studio - deterministic mesh paint scripting facade (Faz 5.4b) */

#include "RtApiInternal.h"
#include "Paint/MeshPaintAdapter.h"
#include "Paint/PaintLayerData.h"
#include "Renderer.h"
#include "Triangle.h"
#include "TriangleMesh.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include "stb_image_write.h"

namespace rtapi {
namespace {

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

const char* channelName(Paint::PaintChannel channel) {
    switch (channel) {
        case Paint::PaintChannel::BaseColor: return "base_color";
        case Paint::PaintChannel::Normal: return "normal";
        case Paint::PaintChannel::Roughness: return "roughness";
        case Paint::PaintChannel::Metallic: return "metallic";
        case Paint::PaintChannel::Emission: return "emission";
        case Paint::PaintChannel::Mask: return "mask";
        case Paint::PaintChannel::Transmission: return "transmission";
        case Paint::PaintChannel::Opacity: return "opacity";
    }
    return "unknown";
}

bool parseChannel(const std::string& value, Paint::PaintChannel& out) {
    const std::string key = lower(value);
    if (key == "base_color" || key == "basecolor" || key == "albedo") out = Paint::PaintChannel::BaseColor;
    else if (key == "normal") out = Paint::PaintChannel::Normal;
    else if (key == "roughness") out = Paint::PaintChannel::Roughness;
    else if (key == "metallic" || key == "metalness") out = Paint::PaintChannel::Metallic;
    else if (key == "emission" || key == "emissive") out = Paint::PaintChannel::Emission;
    else if (key == "mask" || key == "height") out = Paint::PaintChannel::Mask;
    else if (key == "transmission") out = Paint::PaintChannel::Transmission;
    else if (key == "opacity" || key == "alpha") out = Paint::PaintChannel::Opacity;
    else return false;
    return true;
}

bool parseBlend(const std::string& value, Paint::LayerBlendMode& out) {
    const std::string key = lower(value);
    if (key == "normal") out = Paint::LayerBlendMode::Normal;
    else if (key == "add") out = Paint::LayerBlendMode::Add;
    else if (key == "multiply") out = Paint::LayerBlendMode::Multiply;
    else if (key == "screen") out = Paint::LayerBlendMode::Screen;
    else if (key == "overlay") out = Paint::LayerBlendMode::Overlay;
    else return false;
    return true;
}

std::shared_ptr<Triangle> findPaintTriangle(UIContext& ctx, const std::string& object_name,
                                            int material_id) {
    for (const auto& object : ctx.scene.world.objects) {
        if (auto tri = std::dynamic_pointer_cast<Triangle>(object)) {
            if (tri->getNodeName() == object_name &&
                (material_id < 0 || tri->getMaterialID() == static_cast<uint16_t>(material_id))) return tri;
        } else if (auto mesh = std::dynamic_pointer_cast<TriangleMesh>(object)) {
            if (mesh->nodeName != object_name || !mesh->geometry) continue;
            for (size_t i = 0; i < mesh->num_triangles(); ++i) {
                auto tri = std::make_shared<Triangle>(mesh, static_cast<uint32_t>(i));
                if (material_id < 0 || tri->getMaterialID() == static_cast<uint16_t>(material_id)) return tri;
            }
        }
    }
    return nullptr;
}

Result adapterFor(const std::string& object_name, int material_id,
                  std::shared_ptr<Paint::MeshPaintAdapter>& out) {
    if (!g_ctx) return notBound();
    auto triangle = findPaintTriangle(*g_ctx, object_name, material_id);
    if (!triangle) return Result::fail("paint target mesh/material not found: " + object_name);
    out = std::make_shared<Paint::MeshPaintAdapter>(&g_ctx->scene, triangle);
    if (!out->isValid()) return Result::fail("paint target is not valid: " + object_name);
    return Result::success();
}

PaintLayerInfo describeLayer(const Paint::PaintLayerData& layer, int index) {
    PaintLayerInfo info;
    info.index = index; info.id = layer.id; info.name = layer.meta.name;
    info.visible = layer.meta.visible; info.locked = layer.meta.locked;
    info.opacity = layer.meta.opacity; info.blend_mode = lower(Paint::blendModeName(layer.meta.blend_mode));
    for (size_t i = 0; i < Paint::kPaintChannelCount; ++i) {
        const auto channel = static_cast<Paint::PaintChannel>(i);
        if (layer.hasPixels(channel)) info.channels.push_back(channelName(channel));
    }
    return info;
}

PaintTargetInfo describeTarget(Paint::MeshPaintAdapter& adapter) {
    PaintTargetInfo info;
    info.object_name = adapter.getNodeName(); info.material_id = adapter.getMaterialID();
    if (const auto* set = adapter.getTextureSet()) {
        info.resolution = set->resolution;
        for (size_t i = 0; i < Paint::kPaintChannelCount; ++i) {
            const auto channel = static_cast<Paint::PaintChannel>(i);
            if (set->getTexture(channel)) info.channels.push_back(channelName(channel));
        }
    }
    if (const auto* stack = adapter.getLayerStack()) {
        if (info.resolution == 0) info.resolution = stack->width();
        for (int i = 0; i < stack->layerCount(); ++i)
            if (const auto* layer = stack->layerAt(i)) info.layers.push_back(describeLayer(*layer, i));
    }
    return info;
}

void syncPaint(Paint::MeshPaintAdapter& adapter) {
    adapter.compositeAndUpload();
    g_ctx->renderer.updateBackendMaterials(g_ctx->scene);
    g_ctx->renderer.resetCPUAccumulation();
}

} // namespace

Result getPaintTarget(const std::string& object_name, int material_id, PaintTargetInfo& out_info) {
    std::shared_ptr<Paint::MeshPaintAdapter> adapter;
    Result r = adapterFor(object_name, material_id, adapter); if (!r.ok) return r;
    if (!adapter->getLayerStack() && !adapter->getTextureSet())
        return Result::fail("paint target has not been initialized: " + object_name);
    out_info = describeTarget(*adapter);
    return Result::success();
}

Result ensurePaintTarget(const std::string& object_name, int material_id, int resolution,
                         PaintTargetInfo& out_info) {
    if (resolution < 16 || resolution > 16384)
        return Result::fail("paint resolution must be between 16 and 16384");
    std::shared_ptr<Paint::MeshPaintAdapter> adapter;
    Result r = adapterFor(object_name, material_id, adapter); if (!r.ok) return r;
    adapter->ensureTextureSet(resolution);
    adapter->ensureLayerStack();
    out_info = describeTarget(*adapter);
    return Result::success();
}

Result addPaintLayer(const std::string& object_name, int material_id,
                     const std::string& name, int insert_at, PaintLayerInfo& out_info) {
    std::shared_ptr<Paint::MeshPaintAdapter> adapter;
    Result r = adapterFor(object_name, material_id, adapter); if (!r.ok) return r;
    auto& stack = adapter->ensureLayerStack();
    const int index = stack.addLayer(name.empty() ? "Paint Layer" : name, insert_at);
    const auto* layer = stack.layerAt(index);
    if (!layer) return Result::fail("failed to add paint layer");
    out_info = describeLayer(*layer, index);
    return Result::success();
}

Result removePaintLayer(const std::string& object_name, int material_id, int layer_index) {
    std::shared_ptr<Paint::MeshPaintAdapter> adapter;
    Result r = adapterFor(object_name, material_id, adapter); if (!r.ok) return r;
    auto* stack = adapter->getLayerStack();
    if (!stack) return Result::fail("paint target has no layer stack: " + object_name);
    if (!stack->removeLayer(layer_index)) return Result::fail("cannot remove paint layer index " + std::to_string(layer_index));
    syncPaint(*adapter);
    return Result::success();
}

Result updatePaintLayer(const std::string& object_name, int material_id, int layer_index,
                        const std::string* name, const bool* visible, const bool* locked,
                        const float* opacity, const std::string* blend_mode) {
    std::shared_ptr<Paint::MeshPaintAdapter> adapter;
    Result r = adapterFor(object_name, material_id, adapter); if (!r.ok) return r;
    auto* stack = adapter->getLayerStack();
    auto* layer = stack ? stack->layerAt(layer_index) : nullptr;
    if (!layer) return Result::fail("paint layer index not found: " + std::to_string(layer_index));
    if (opacity && (*opacity < 0.0f || *opacity > 1.0f))
        return Result::fail("paint layer opacity must be between 0 and 1");
    Paint::LayerBlendMode blend;
    if (blend_mode && !parseBlend(*blend_mode, blend)) return Result::fail("unknown paint blend mode: " + *blend_mode);
    if (name) layer->meta.name = *name;
    if (visible) layer->meta.visible = *visible;
    if (locked) layer->meta.locked = *locked;
    if (opacity) layer->meta.opacity = *opacity;
    if (blend_mode) layer->meta.blend_mode = blend;
    syncPaint(*adapter);
    return Result::success();
}

Result fillPaintLayer(const std::string& object_name, int material_id, int layer_index,
                      const std::string& channel_name, Vec3 color) {
    Paint::PaintChannel channel;
    if (!parseChannel(channel_name, channel)) return Result::fail("unknown paint channel: " + channel_name);
    std::shared_ptr<Paint::MeshPaintAdapter> adapter;
    Result r = adapterFor(object_name, material_id, adapter); if (!r.ok) return r;
    if (!adapter->getLayerStack() || !adapter->getLayerStack()->layerAt(layer_index))
        return Result::fail("paint layer index not found: " + std::to_string(layer_index));
    if (!adapter->assignTextureToChannel(channel)) return Result::fail("failed to initialize paint channel: " + channel_name);
    Paint::BrushSettings brush; brush.tool = Paint::BrushTool::Fill; brush.color = color; brush.strength = 1.0f;
    if (!adapter->fillChannel(channel, brush, layer_index)) return Result::fail("failed to fill paint channel: " + channel_name);
    g_ctx->renderer.updateBackendMaterials(g_ctx->scene);
    g_ctx->renderer.resetCPUAccumulation();
    return Result::success();
}

Result clearPaintLayerChannel(const std::string& object_name, int material_id,
                              int layer_index, const std::string& channel_name) {
    Paint::PaintChannel channel;
    if (!parseChannel(channel_name, channel)) return Result::fail("unknown paint channel: " + channel_name);
    std::shared_ptr<Paint::MeshPaintAdapter> adapter;
    Result r = adapterFor(object_name, material_id, adapter); if (!r.ok) return r;
    auto* stack = adapter->getLayerStack(); auto* layer = stack ? stack->layerAt(layer_index) : nullptr;
    if (!layer) return Result::fail("paint layer index not found: " + std::to_string(layer_index));
    layer->clearChannel(channel);
    syncPaint(*adapter);
    return Result::success();
}

Result duplicatePaintLayer(const std::string& object_name, int material_id, int layer_index,
                           PaintLayerInfo& out_info) {
    std::shared_ptr<Paint::MeshPaintAdapter> adapter;
    Result r = adapterFor(object_name, material_id, adapter); if (!r.ok) return r;
    auto* stack = adapter->getLayerStack();
    if (!stack || !stack->layerAt(layer_index))
        return Result::fail("paint layer index not found: " + std::to_string(layer_index));
    const int index = stack->duplicateLayer(layer_index);
    const auto* layer = stack->layerAt(index);
    if (index < 0 || !layer) return Result::fail("failed to duplicate paint layer");
    out_info = describeLayer(*layer, index);
    syncPaint(*adapter);
    return Result::success();
}

Result movePaintLayer(const std::string& object_name, int material_id,
                      int from_index, int to_index) {
    std::shared_ptr<Paint::MeshPaintAdapter> adapter;
    Result r = adapterFor(object_name, material_id, adapter); if (!r.ok) return r;
    auto* stack = adapter->getLayerStack();
    if (!stack || !stack->moveLayer(from_index, to_index))
        return Result::fail("failed to move paint layer from " + std::to_string(from_index) +
                            " to " + std::to_string(to_index));
    syncPaint(*adapter);
    return Result::success();
}

Result mergePaintLayerDown(const std::string& object_name, int material_id, int layer_index) {
    std::shared_ptr<Paint::MeshPaintAdapter> adapter;
    Result r = adapterFor(object_name, material_id, adapter); if (!r.ok) return r;
    auto* stack = adapter->getLayerStack();
    if (!stack || !stack->mergeDown(layer_index))
        return Result::fail("cannot merge paint layer down at index " + std::to_string(layer_index));
    syncPaint(*adapter);
    return Result::success();
}

Result flattenPaintLayers(const std::string& object_name, int material_id) {
    std::shared_ptr<Paint::MeshPaintAdapter> adapter;
    Result r = adapterFor(object_name, material_id, adapter); if (!r.ok) return r;
    auto* stack = adapter->getLayerStack();
    if (!stack) return Result::fail("paint target has no layer stack: " + object_name);
    stack->flattenAll();
    syncPaint(*adapter);
    return Result::success();
}

Result bakePaintHeightToNormal(const std::string& object_name, int material_id,
                              float strength, bool clear_height) {
    if (strength <= 0.0f || strength > 128.0f)
        return Result::fail("paint normal bake strength must be greater than 0 and at most 128");
    std::shared_ptr<Paint::MeshPaintAdapter> adapter;
    Result r = adapterFor(object_name, material_id, adapter); if (!r.ok) return r;
    if (!adapter->assignTextureToChannel(Paint::PaintChannel::Mask) ||
        !adapter->assignTextureToChannel(Paint::PaintChannel::Normal))
        return Result::fail("failed to initialize height/normal paint channels");
    if (!adapter->bakeHeightIntoNormal(strength, clear_height))
        return Result::fail("failed to bake paint height into normal");
    g_ctx->renderer.updateBackendMaterials(g_ctx->scene);
    g_ctx->renderer.resetCPUAccumulation();
    return Result::success();
}

Result importPaintChannel(const std::string& object_name, int material_id, int layer_index,
                          const std::string& channel_name, const std::string& filepath) {
    Paint::PaintChannel channel;
    if (!parseChannel(channel_name, channel)) return Result::fail("unknown paint channel: " + channel_name);
    auto imported = std::make_shared<Texture>(filepath, TextureType::Unknown);
    if (!imported || !imported->is_loaded() || imported->pixels.empty())
        return Result::fail("failed to load paint texture: " + filepath);
    std::shared_ptr<Paint::MeshPaintAdapter> adapter;
    Result r = adapterFor(object_name, material_id, adapter); if (!r.ok) return r;
    if (!adapter->assignTextureToChannel(channel))
        return Result::fail("failed to initialize paint channel: " + channel_name);
    auto* set = adapter->getTextureSet(); auto* stack = adapter->getLayerStack();
    auto* layer = stack ? stack->layerAt(layer_index) : nullptr;
    if (!set || !layer) return Result::fail("paint layer index not found: " + std::to_string(layer_index));
    if (layer->meta.locked) return Result::fail("paint layer is locked: " + std::to_string(layer_index));
    const int width = stack->width(); const int height = stack->height();
    if (width <= 0 || height <= 0) return Result::fail("paint target has invalid resolution");
    auto& pixels = layer->ensurePixels(channel);
    pixels.resize(static_cast<size_t>(width) * static_cast<size_t>(height));
    for (int y = 0; y < height; ++y) {
        const int sy = std::clamp(static_cast<int>((static_cast<float>(y) / std::max(1, height - 1)) *
                                                   static_cast<float>(imported->height - 1) + 0.5f), 0, imported->height - 1);
        for (int x = 0; x < width; ++x) {
            const int sx = std::clamp(static_cast<int>((static_cast<float>(x) / std::max(1, width - 1)) *
                                                       static_cast<float>(imported->width - 1) + 0.5f), 0, imported->width - 1);
            pixels[static_cast<size_t>(y) * width + x] =
                imported->pixels[static_cast<size_t>(sy) * imported->width + sx];
        }
    }
    set->setSourceInfo(channel, true, filepath);
    set->setSourceTexture(channel, imported);
    syncPaint(*adapter);
    return Result::success();
}

Result exportPaintChannel(const std::string& object_name, int material_id, int layer_index,
                          const std::string& channel_name, const std::string& filepath) {
    Paint::PaintChannel channel;
    if (!parseChannel(channel_name, channel)) return Result::fail("unknown paint channel: " + channel_name);
    if (filepath.empty()) return Result::fail("paint export filepath cannot be empty");
    std::shared_ptr<Paint::MeshPaintAdapter> adapter;
    Result r = adapterFor(object_name, material_id, adapter); if (!r.ok) return r;
    const CompactVec4* data = nullptr; int width = 0; int height = 0;
    if (layer_index >= 0) {
        const auto* stack = adapter->getLayerStack(); const auto* layer = stack ? stack->layerAt(layer_index) : nullptr;
        if (!layer || !layer->hasPixels(channel))
            return Result::fail("paint layer channel has no pixels: " + channel_name);
        const auto& pixels = layer->getPixels(channel);
        data = pixels.data(); width = layer->width; height = layer->height;
    } else {
        adapter->compositeAndUpload();
        const auto* set = adapter->getTextureSet(); auto texture = set ? set->getTexture(channel) : nullptr;
        if (!texture || texture->pixels.empty()) return Result::fail("paint channel has no texture: " + channel_name);
        data = texture->pixels.data(); width = texture->width; height = texture->height;
    }
    std::filesystem::path path(filepath);
    if (path.has_parent_path()) {
        std::error_code ec; std::filesystem::create_directories(path.parent_path(), ec);
        if (ec) return Result::fail("failed to create paint export directory: " + path.parent_path().string());
    }
    if (!stbi_write_png(path.string().c_str(), width, height, 4, data, width * 4))
        return Result::fail("failed to export paint channel: " + filepath);
    return Result::success();
}

Result listPaintMaskPresets(std::vector<std::string>& out_presets) {
    out_presets = {"linear_x", "linear_y", "radial", "checker", "noise", "edge_wear", "dirt"};
    return Result::success();
}

Result applyPaintMaskPreset(const std::string& object_name, int material_id, int layer_index,
                            const std::string& preset, float strength, unsigned int seed) {
    if (strength < 0.0f || strength > 1.0f)
        return Result::fail("paint mask strength must be between 0 and 1");
    const std::string key = lower(preset);
    std::vector<std::string> valid; listPaintMaskPresets(valid);
    if (std::find(valid.begin(), valid.end(), key) == valid.end())
        return Result::fail("unknown paint mask preset: " + preset);
    std::shared_ptr<Paint::MeshPaintAdapter> adapter;
    Result r = adapterFor(object_name, material_id, adapter); if (!r.ok) return r;
    if (!adapter->assignTextureToChannel(Paint::PaintChannel::Mask))
        return Result::fail("failed to initialize paint mask channel");
    auto* stack = adapter->getLayerStack(); auto* layer = stack ? stack->layerAt(layer_index) : nullptr;
    if (!layer) return Result::fail("paint layer index not found: " + std::to_string(layer_index));
    if (layer->meta.locked) return Result::fail("paint layer is locked: " + std::to_string(layer_index));
    auto hash01 = [seed](unsigned int x, unsigned int y) {
        unsigned int h = x * 374761393u + y * 668265263u + seed * 2246822519u;
        h = (h ^ (h >> 13u)) * 1274126177u; h ^= h >> 16u;
        return static_cast<float>(h & 0x00ffffffu) / static_cast<float>(0x01000000u);
    };
    const int width = stack->width(); const int height = stack->height();
    auto& pixels = layer->ensurePixels(Paint::PaintChannel::Mask);
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
        const float u = static_cast<float>(x) / std::max(1, width - 1);
        const float v = static_cast<float>(y) / std::max(1, height - 1);
        const float noise = hash01(static_cast<unsigned int>(x), static_cast<unsigned int>(y));
        float value = 0.0f;
        if (key == "linear_x") value = u;
        else if (key == "linear_y") value = v;
        else if (key == "radial") value = std::clamp(1.0f - std::sqrt((u - 0.5f)*(u - 0.5f) + (v - 0.5f)*(v - 0.5f)) * 2.0f, 0.0f, 1.0f);
        else if (key == "checker") value = ((x * 8 / std::max(1, width) + y * 8 / std::max(1, height)) & 1) ? 1.0f : 0.0f;
        else if (key == "noise") value = noise;
        else if (key == "edge_wear") { const float edge = std::min({u, v, 1.0f-u, 1.0f-v}) * 8.0f; value = std::clamp((1.0f-edge) * 0.75f + noise * 0.45f, 0.0f, 1.0f); }
        else if (key == "dirt") value = std::clamp((1.0f-v) * 0.65f + noise * 0.5f, 0.0f, 1.0f);
        const uint8_t byte = static_cast<uint8_t>(std::clamp(0.5f + (value - 0.5f) * strength, 0.0f, 1.0f) * 255.0f + 0.5f);
        pixels[static_cast<size_t>(y) * width + x] = CompactVec4(byte, byte, byte, 255);
    }
    syncPaint(*adapter);
    return Result::success();
}

} // namespace rtapi
