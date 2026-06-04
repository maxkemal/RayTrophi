#include "Paint/PaintLayerStack.h"
#include <algorithm>
#include <cmath>
#include <cassert>
#include <fstream>
#include "stb_image.h"
#include "stb_image_write.h"

namespace Paint {

// ======================== helpers ========================

namespace {

CompactVec4 defaultCompositePixel(PaintChannel channel) {
    switch (channel) {
        case PaintChannel::BaseColor:
        case PaintChannel::Emission:
            return CompactVec4(255, 255, 255, 255);
        case PaintChannel::Normal:
            return CompactVec4(128, 128, 255, 255);
        case PaintChannel::Mask:
            return CompactVec4(128, 128, 128, 255);
        case PaintChannel::Opacity:
            return CompactVec4(255, 255, 255, 255);
        case PaintChannel::Roughness:
        case PaintChannel::Metallic:
        case PaintChannel::Transmission:
            return CompactVec4(0, 0, 0, 255);
    }
    return CompactVec4(0, 0, 0, 255);
}

// Bilinear-resample a pixel buffer from (src_w, src_h) to (dst_w, dst_h).
void resamplePixels(const std::vector<CompactVec4>& src, int src_w, int src_h,
                    std::vector<CompactVec4>& dst, int dst_w, int dst_h)
{
    dst.resize(static_cast<size_t>(dst_w) * static_cast<size_t>(dst_h));
    if (src.empty() || src_w <= 0 || src_h <= 0) {
        std::fill(dst.begin(), dst.end(), CompactVec4(0, 0, 0, 0));
        return;
    }

    for (int y = 0; y < dst_h; ++y) {
        const float v = dst_h > 1 ? static_cast<float>(y) / static_cast<float>(dst_h - 1) : 0.0f;
        const float sy = v * static_cast<float>(src_h - 1);
        const int y0 = std::clamp(static_cast<int>(std::floor(sy)), 0, src_h - 1);
        const int y1 = std::clamp(y0 + 1, 0, src_h - 1);
        const float ty = sy - static_cast<float>(y0);

        for (int x = 0; x < dst_w; ++x) {
            const float u = dst_w > 1 ? static_cast<float>(x) / static_cast<float>(dst_w - 1) : 0.0f;
            const float sx = u * static_cast<float>(src_w - 1);
            const int x0 = std::clamp(static_cast<int>(std::floor(sx)), 0, src_w - 1);
            const int x1 = std::clamp(x0 + 1, 0, src_w - 1);
            const float tx = sx - static_cast<float>(x0);

            const auto& c00 = src[y0 * src_w + x0];
            const auto& c10 = src[y0 * src_w + x1];
            const auto& c01 = src[y1 * src_w + x0];
            const auto& c11 = src[y1 * src_w + x1];

            auto lerp = [](float a, float b, float t) { return a + (b - a) * t; };
            auto bilerp = [&](uint8_t a00, uint8_t a10, uint8_t a01, uint8_t a11) -> uint8_t {
                const float top = lerp(static_cast<float>(a00), static_cast<float>(a10), tx);
                const float bot = lerp(static_cast<float>(a01), static_cast<float>(a11), tx);
                return static_cast<uint8_t>(std::clamp(lerp(top, bot, ty), 0.0f, 255.0f));
            };

            dst[y * dst_w + x] = CompactVec4(
                bilerp(c00.r, c10.r, c01.r, c11.r),
                bilerp(c00.g, c10.g, c01.g, c11.g),
                bilerp(c00.b, c10.b, c01.b, c11.b),
                bilerp(c00.a, c10.a, c01.a, c11.a));
        }
    }
}

// Blend a single source pixel onto a destination pixel.
// `opacity` is the layer-level opacity [0..1].
void blendPixel(CompactVec4& dst, const CompactVec4& src,
                float opacity, LayerBlendMode mode)
{
    // Source alpha combined with layer opacity.
    const float sa = (static_cast<float>(src.a) / 255.0f) * opacity;
    if (sa <= 0.0f) return;

    const float sr = static_cast<float>(src.r);
    const float sg = static_cast<float>(src.g);
    const float sb = static_cast<float>(src.b);
    const float dr = static_cast<float>(dst.r);
    const float dg = static_cast<float>(dst.g);
    const float db = static_cast<float>(dst.b);

    float out_r, out_g, out_b;

    auto overlay_ch = [](float base, float blend) -> float {
        return base < 128.0f
            ? (2.0f * base * blend) / 255.0f
            : 255.0f - (2.0f * (255.0f - base) * (255.0f - blend)) / 255.0f;
    };

    switch (mode) {
        default:
        case LayerBlendMode::Normal:
            out_r = sr;
            out_g = sg;
            out_b = sb;
            break;
        case LayerBlendMode::Add:
            out_r = std::min(dr + sr, 255.0f);
            out_g = std::min(dg + sg, 255.0f);
            out_b = std::min(db + sb, 255.0f);
            break;
        case LayerBlendMode::Multiply:
            out_r = (dr * sr) / 255.0f;
            out_g = (dg * sg) / 255.0f;
            out_b = (db * sb) / 255.0f;
            break;
        case LayerBlendMode::Screen:
            out_r = 255.0f - ((255.0f - dr) * (255.0f - sr)) / 255.0f;
            out_g = 255.0f - ((255.0f - dg) * (255.0f - sg)) / 255.0f;
            out_b = 255.0f - ((255.0f - db) * (255.0f - sb)) / 255.0f;
            break;
        case LayerBlendMode::Overlay:
            out_r = overlay_ch(dr, sr);
            out_g = overlay_ch(dg, sg);
            out_b = overlay_ch(db, sb);
            break;
    }

    // Alpha-blend the blended color onto the destination.
    dst.r = static_cast<uint8_t>(std::clamp(dr + (out_r - dr) * sa, 0.0f, 255.0f));
    dst.g = static_cast<uint8_t>(std::clamp(dg + (out_g - dg) * sa, 0.0f, 255.0f));
    dst.b = static_cast<uint8_t>(std::clamp(db + (out_b - db) * sa, 0.0f, 255.0f));

    // Simple Porter-Duff "over" for alpha.
    const float da = static_cast<float>(dst.a) / 255.0f;
    const float out_a = sa + da * (1.0f - sa);
    dst.a = static_cast<uint8_t>(std::clamp(out_a * 255.0f, 0.0f, 255.0f));
}

void blendHeightMaskPixel(CompactVec4& dst, const CompactVec4& src, float opacity)
{
    const float sa = (static_cast<float>(src.a) / 255.0f) * opacity;
    if (sa <= 0.0f) return;

    const float neutral = 128.0f;
    const float dst_delta = static_cast<float>(dst.r) - neutral;
    const float src_delta = static_cast<float>(src.r) - neutral;
    const float out = neutral + dst_delta + src_delta * sa;
    const uint8_t value = static_cast<uint8_t>(std::clamp(out, 0.0f, 255.0f));
    dst.r = value;
    dst.g = value;
    dst.b = value;

    const float da = static_cast<float>(dst.a) / 255.0f;
    const float out_a = sa + da * (1.0f - sa);
    dst.a = static_cast<uint8_t>(std::clamp(out_a * 255.0f, 0.0f, 255.0f));
}

void blendOpacityPixel(CompactVec4& dst, const CompactVec4& src, float opacity)
{
    const float layer_alpha = std::clamp(opacity, 0.0f, 1.0f);
    if (layer_alpha <= 0.0f) return;

    const float src_mask = static_cast<float>(src.r) / 255.0f;
    const float src_coverage = static_cast<float>(src.a) / 255.0f;
    const float t = src_coverage * layer_alpha;
    if (t <= 0.0f) return;

    const float dst_mask = static_cast<float>(dst.r) / 255.0f;
    const uint8_t value = static_cast<uint8_t>(
        std::clamp((dst_mask + (src_mask - dst_mask) * t) * 255.0f + 0.5f, 0.0f, 255.0f));

    // Opacity is a scalar mask. Keep RGB and alpha identical so both shader
    // paths (read .r for grayscale or .a for RGBA-alpha textures) see the same
    // value after layer visibility/opacity changes.
    dst.r = value;
    dst.g = value;
    dst.b = value;
    dst.a = value;
}

void blendChannelPixel(PaintChannel channel,
                       CompactVec4& dst,
                       const CompactVec4& src,
                       float opacity,
                       LayerBlendMode mode)
{
    if (channel == PaintChannel::Mask) {
        blendHeightMaskPixel(dst, src, opacity);
        return;
    }
    if (channel == PaintChannel::Opacity) {
        blendOpacityPixel(dst, src, opacity);
        return;
    }
    blendPixel(dst, src, opacity, mode);
}

} // namespace

// ======================== PaintLayerData ========================

void PaintLayerData::resize(int new_width, int new_height) {
    if (new_width <= 0 || new_height <= 0) return;
    if (new_width == width && new_height == height) return;

    for (size_t i = 0; i < channel_pixels.size(); ++i) {
        auto& buf = channel_pixels[i];
        if (buf.empty()) continue;
        std::vector<CompactVec4> resized;
        resamplePixels(buf, width, height, resized, new_width, new_height);
        buf = std::move(resized);
    }
    width  = new_width;
    height = new_height;
}

// ======================== PaintLayerStack ========================

uint32_t PaintLayerStack::nextId() {
    return next_id_++;
}

PaintLayerData* PaintLayerStack::layerAt(int index) {
    if (index < 0 || index >= static_cast<int>(layers_.size())) return nullptr;
    return &layers_[index];
}

const PaintLayerData* PaintLayerStack::layerAt(int index) const {
    if (index < 0 || index >= static_cast<int>(layers_.size())) return nullptr;
    return &layers_[index];
}

PaintLayerData* PaintLayerStack::layerById(uint32_t id) {
    for (auto& layer : layers_) {
        if (layer.id == id) return &layer;
    }
    return nullptr;
}

int PaintLayerStack::indexOfId(uint32_t id) const {
    for (int i = 0; i < static_cast<int>(layers_.size()); ++i) {
        if (layers_[i].id == id) return i;
    }
    return -1;
}

// -------- mutation --------

int PaintLayerStack::addLayer(const std::string& name, int insert_at) {
    PaintLayerData layer;
    layer.meta.name = name.empty() ? ("Layer " + std::to_string(next_id_)) : name;
    layer.id = nextId();
    layer.width  = width_;
    layer.height = height_;

    if (insert_at < 0 || insert_at >= static_cast<int>(layers_.size())) {
        layers_.push_back(std::move(layer));
        return static_cast<int>(layers_.size()) - 1;
    }

    layers_.insert(layers_.begin() + insert_at, std::move(layer));
    return insert_at;
}

int PaintLayerStack::duplicateLayer(int source_index) {
    if (source_index < 0 || source_index >= static_cast<int>(layers_.size())) return -1;

    PaintLayerData copy = layers_[source_index];
    copy.id = nextId();
    copy.meta.name += " Copy";

    const int dest = source_index + 1;
    layers_.insert(layers_.begin() + dest, std::move(copy));
    return dest;
}

bool PaintLayerStack::removeLayer(int index) {
    if (layers_.size() <= 1) return false;
    if (index < 0 || index >= static_cast<int>(layers_.size())) return false;

    layers_.erase(layers_.begin() + index);
    return true;
}

bool PaintLayerStack::moveLayer(int from_index, int to_index) {
    const int count = static_cast<int>(layers_.size());
    if (from_index < 0 || from_index >= count) return false;
    if (to_index < 0 || to_index >= count) return false;
    if (from_index == to_index) return true;

    PaintLayerData tmp = std::move(layers_[from_index]);
    layers_.erase(layers_.begin() + from_index);
    layers_.insert(layers_.begin() + to_index, std::move(tmp));
    return true;
}

bool PaintLayerStack::mergeDown(int index) {
    if (index <= 0 || index >= static_cast<int>(layers_.size())) return false;

    const PaintLayerData& upper = layers_[index];
    PaintLayerData& lower = layers_[index - 1];

    if (!upper.meta.visible) {
        // Invisible layer — just remove it, nothing to merge.
        layers_.erase(layers_.begin() + index);
        return true;
    }

    for (int ch = 0; ch < static_cast<int>(kPaintChannelCount); ++ch) {
        const auto channel = static_cast<PaintChannel>(ch);
        if (!upper.hasPixels(channel)) continue;

        auto& dst_buf = lower.ensurePixels(channel);
        const auto& src_buf = upper.getPixels(channel);
        const size_t pixel_count = static_cast<size_t>(width_) * static_cast<size_t>(height_);

        for (size_t i = 0; i < pixel_count && i < src_buf.size() && i < dst_buf.size(); ++i) {
            blendChannelPixel(channel, dst_buf[i], src_buf[i], upper.meta.opacity, upper.meta.blend_mode);
        }
    }

    layers_.erase(layers_.begin() + index);
    return true;
}

void PaintLayerStack::flattenAll() {
    if (layers_.size() <= 1) return;

    const size_t pixel_count = static_cast<size_t>(width_) * static_cast<size_t>(height_);

    for (int ch = 0; ch < static_cast<int>(kPaintChannelCount); ++ch) {
        const auto channel = static_cast<PaintChannel>(ch);
        std::vector<CompactVec4> result(pixel_count, CompactVec4(0, 0, 0, 0));

        for (const auto& layer : layers_) {
            if (!layer.meta.visible || !layer.hasPixels(channel)) continue;
            const auto& src = layer.getPixels(channel);
            for (size_t i = 0; i < pixel_count && i < src.size(); ++i) {
                blendChannelPixel(channel, result[i], src[i], layer.meta.opacity, layer.meta.blend_mode);
            }
        }

        layers_[0].channel_pixels[ch] = std::move(result);
    }

    // Keep only layer 0.
    layers_.resize(1);
    layers_[0].meta.name = "Flattened";
    layers_[0].meta.opacity = 1.0f;
    layers_[0].meta.blend_mode = LayerBlendMode::Normal;
    layers_[0].meta.visible = true;
    layers_[0].meta.locked = false;
}

// -------- compositing --------

void PaintLayerStack::compositeChannel(PaintChannel channel,
                                       std::vector<CompactVec4>& dst_pixels,
                                       int w, int h) const
{
    const size_t pixel_count = static_cast<size_t>(w) * static_cast<size_t>(h);
    dst_pixels.assign(pixel_count, defaultCompositePixel(channel));

    for (const auto& layer : layers_) {
        if (!layer.meta.visible) continue;
        if (!layer.hasPixels(channel)) continue;

        const auto& src = layer.getPixels(channel);
        const float opacity = layer.meta.opacity;
        const LayerBlendMode mode = layer.meta.blend_mode;

        for (size_t i = 0; i < pixel_count && i < src.size(); ++i) {
            blendChannelPixel(channel, dst_pixels[i], src[i], opacity, mode);
        }
    }
}

bool PaintLayerStack::anyLayerHasPixels(PaintChannel channel) const {
    for (const auto& layer : layers_) {
        if (layer.hasPixels(channel)) return true;
    }
    return false;
}

void PaintLayerStack::flattenInto(PaintTextureSet& texture_set) const {
    if (width_ <= 0 || height_ <= 0) return;

    for (int ch = 0; ch < static_cast<int>(kPaintChannelCount); ++ch) {
        const auto channel = static_cast<PaintChannel>(ch);
        auto texture = texture_set.getTexture(channel);
        if (!texture) continue;

        // If no layer has pixel data for this channel, leave the texture as-is.
        // Overwriting with empty composited data would turn it black.
        if (!anyLayerHasPixels(channel)) continue;

        std::vector<CompactVec4> composited;
        compositeChannel(channel, composited, width_, height_);

        // Write composited pixels into the texture.
        if (texture->width == width_ && texture->height == height_) {
            texture->pixels = std::move(composited);
        } else {
            // Resolution mismatch — resample.
            resamplePixels(composited, width_, height_,
                           texture->pixels, texture->width, texture->height);
        }

        if (texture->isUploaded()) {
            texture->updateGPU();
        } else {
            texture->upload_to_gpu();
        }
    }
}

void PaintLayerStack::flattenChannelInto(PaintChannel channel, PaintTextureSet& texture_set) const {
    if (width_ <= 0 || height_ <= 0) return;

    auto texture = texture_set.getTexture(channel);
    if (!texture) return;
    if (!anyLayerHasPixels(channel)) return;

    std::vector<CompactVec4> composited;
    compositeChannel(channel, composited, width_, height_);

    if (texture->width == width_ && texture->height == height_) {
        texture->pixels = std::move(composited);
    } else {
        resamplePixels(composited, width_, height_,
                       texture->pixels, texture->width, texture->height);
    }

    if (texture->isUploaded()) {
        texture->updateGPU();
    } else {
        texture->upload_to_gpu();
    }
}

void PaintLayerStack::flattenChannelRegionInto(PaintChannel channel, PaintTextureSet& texture_set,
                                               const PaintDirtyRect& dirty) const {
    if (width_ <= 0 || height_ <= 0 || dirty.empty()) return;

    auto texture = texture_set.getTexture(channel);
    if (!texture) return;
    if (!anyLayerHasPixels(channel)) return;

    // If texture resolution differs from stack, fall back to full composite.
    if (texture->width != width_ || texture->height != height_) {
        flattenChannelInto(channel, texture_set);
        return;
    }

    // Clamp dirty rect to valid range.
    const int rx0 = std::max(dirty.min_x, 0);
    const int ry0 = std::max(dirty.min_y, 0);
    const int rx1 = std::min(dirty.max_x, width_ - 1);
    const int ry1 = std::min(dirty.max_y, height_ - 1);
    if (rx0 > rx1 || ry0 > ry1) return;

    // Composite only the dirty region into the texture's pixel buffer.
    for (int y = ry0; y <= ry1; ++y) {
        for (int x = rx0; x <= rx1; ++x) {
            const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(width_) + static_cast<size_t>(x);
            CompactVec4 result = defaultCompositePixel(channel);

            for (const auto& layer : layers_) {
                if (!layer.meta.visible || !layer.hasPixels(channel)) continue;
                const auto& src = layer.getPixels(channel);
                if (idx < src.size()) {
                    blendChannelPixel(channel, result, src[idx], layer.meta.opacity, layer.meta.blend_mode);
                }
            }

            texture->pixels[idx] = result;
        }
    }

    // Partial GPU upload for only the dirty region.
    if (texture->isUploaded()) {
        texture->updateGPURegion(rx0, ry0, rx1 - rx0 + 1, ry1 - ry0 + 1);
    } else {
        texture->upload_to_gpu();
    }
}

// -------- resolution --------

void PaintLayerStack::setResolution(int w, int h) {
    if (w <= 0 || h <= 0) return;
    if (w == width_ && h == height_) return;

    for (auto& layer : layers_) {
        layer.resize(w, h);
    }
    width_  = w;
    height_ = h;
}

// -------- initialisation --------

void PaintLayerStack::seedFromTextureSet(const PaintTextureSet& src) {
    ensureBaseLayer();

    PaintLayerData& base = layers_[0];

    for (int ch = 0; ch < static_cast<int>(kPaintChannelCount); ++ch) {
        const auto channel = static_cast<PaintChannel>(ch);
        auto texture = src.getTexture(channel);
        if (!texture || !texture->is_loaded() || texture->pixels.empty()) continue;

        if (width_ <= 0 || height_ <= 0) {
            width_  = texture->width;
            height_ = texture->height;
            base.width  = width_;
            base.height = height_;
        }

        auto& buf = base.ensurePixels(channel);
        if (texture->width == width_ && texture->height == height_) {
            buf = texture->pixels;
        } else {
            resamplePixels(texture->pixels, texture->width, texture->height,
                           buf, width_, height_);
        }

        // Base layer must always be fully opaque for every channel except
        // Opacity. compositeChannel starts from CompactVec4(0,0,0,0) and uses
        // Porter-Duff "over" via blendPixel; alpha < 255 on the base would
        // compose to black on the first dab. Opacity is the one channel
        // where alpha *is* the user-meaningful value (the mask itself), so
        // we leave its base layer alpha alone — the composite still works
        // because makeBrushPixel writes the same grayscale into all four
        // channels for Opacity, keeping RGB and A in lockstep.
        if (channel != PaintChannel::Opacity) {
            for (auto& p : buf) p.a = 255;
        }
    }
}

void PaintLayerStack::ensureBaseLayer() {
    if (layers_.empty()) {
        addLayer("Background");
    }
}

// ======================== serialization ========================

namespace {

void png_write_to_vector(void* context, void* data, int size) {
    auto* vec = static_cast<std::vector<char>*>(context);
    const char* d = static_cast<const char*>(data);
    vec->insert(vec->end(), d, d + size);
}

} // namespace

void PaintLayerStack::serialize(nlohmann::json& j, std::ostream& bin) const {
    j["width"]   = width_;
    j["height"]  = height_;
    j["next_id"] = next_id_;

    nlohmann::json j_layers = nlohmann::json::array();
    for (const auto& layer : layers_) {
        nlohmann::json jl;
        jl["id"]               = layer.id;
        jl["name"]             = layer.meta.name;
        jl["visible"]          = layer.meta.visible;
        jl["locked"]           = layer.meta.locked;
        jl["opacity"]          = layer.meta.opacity;
        jl["blend_mode"]       = static_cast<int>(layer.meta.blend_mode);
        jl["enabled_channels"] = layer.meta.enabled_channels;

        nlohmann::json j_channels = nlohmann::json::array();
        for (int ch = 0; ch < static_cast<int>(kPaintChannelCount); ++ch) {
            const auto& buf = layer.channel_pixels[ch];
            if (buf.empty()) continue;

            // Encode pixels as RGBA PNG into the binary stream.
            std::vector<uint8_t> raw(buf.size() * 4);
            for (size_t i = 0; i < buf.size(); ++i) {
                raw[i * 4 + 0] = buf[i].r;
                raw[i * 4 + 1] = buf[i].g;
                raw[i * 4 + 2] = buf[i].b;
                raw[i * 4 + 3] = buf[i].a;
            }

            std::vector<char> png_data;
            stbi_write_png_to_func(png_write_to_vector, &png_data,
                                   layer.width, layer.height, 4,
                                   raw.data(), layer.width * 4);
            if (png_data.empty()) continue;

            const auto offset = static_cast<int64_t>(bin.tellp());
            bin.write(png_data.data(), static_cast<std::streamsize>(png_data.size()));

            nlohmann::json jc;
            jc["ch"]     = ch;
            jc["offset"] = offset;
            jc["size"]   = static_cast<int64_t>(png_data.size());
            j_channels.push_back(jc);
        }
        jl["pixel_channels"] = j_channels;
        j_layers.push_back(jl);
    }
    j["layers"] = j_layers;
}

void PaintLayerStack::deserialize(const nlohmann::json& j, std::istream& bin) {
    layers_.clear();

    width_   = j.value("width", 0);
    height_  = j.value("height", 0);
    next_id_ = j.value("next_id", 1u);

    if (!j.contains("layers") || !j["layers"].is_array()) return;

    for (const auto& jl : j["layers"]) {
        PaintLayerData layer;
        layer.id                   = jl.value("id", 0u);
        layer.meta.name            = jl.value("name", std::string("Layer"));
        layer.meta.visible         = jl.value("visible", true);
        layer.meta.locked          = jl.value("locked", false);
        layer.meta.opacity         = jl.value("opacity", 1.0f);
        layer.meta.blend_mode      = static_cast<LayerBlendMode>(jl.value("blend_mode", 0));
        layer.meta.enabled_channels = jl.value("enabled_channels", 0xFFFFFFFFu);
        layer.width  = width_;
        layer.height = height_;

        if (jl.contains("pixel_channels") && jl["pixel_channels"].is_array()) {
            for (const auto& jc : jl["pixel_channels"]) {
                const int ch          = jc.value("ch", -1);
                const int64_t offset  = jc.value("offset", int64_t(-1));
                const int64_t size    = jc.value("size", int64_t(0));
                if (ch < 0 || ch >= static_cast<int>(kPaintChannelCount) || offset < 0 || size <= 0) continue;

                // Read PNG blob from binary stream.
                std::vector<uint8_t> png_buf(static_cast<size_t>(size));
                bin.seekg(offset);
                bin.read(reinterpret_cast<char*>(png_buf.data()), size);
                if (bin.gcount() != size) continue;

                int img_w = 0, img_h = 0, img_channels = 0;
                uint8_t* decoded = stbi_load_from_memory(
                    png_buf.data(), static_cast<int>(size),
                    &img_w, &img_h, &img_channels, 4);
                if (!decoded) continue;

                const size_t pixel_count = static_cast<size_t>(img_w) * static_cast<size_t>(img_h);
                auto& buf = layer.channel_pixels[ch];
                buf.resize(pixel_count);
                for (size_t i = 0; i < pixel_count; ++i) {
                    buf[i] = CompactVec4(
                        decoded[i * 4 + 0],
                        decoded[i * 4 + 1],
                        decoded[i * 4 + 2],
                        decoded[i * 4 + 3]);
                }
                stbi_image_free(decoded);

                // Update layer dimensions from decoded image if stack dimensions
                // are missing (robustness).
                if (layer.width <= 0)  layer.width  = img_w;
                if (layer.height <= 0) layer.height = img_h;
            }
        }

        layers_.push_back(std::move(layer));
    }

    // Fix up stack dimensions from first layer if they were zero.
    if ((width_ <= 0 || height_ <= 0) && !layers_.empty()) {
        width_  = layers_[0].width;
        height_ = layers_[0].height;
    }
}

} // namespace Paint
