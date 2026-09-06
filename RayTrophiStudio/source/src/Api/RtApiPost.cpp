#include "Api/RtApiInternal.h"
#include "PostProcess/PostService.h"
#include <algorithm>

extern bool stylize_redisplay;
#include <cmath>
#include <cctype>
namespace rtapi {
namespace {

std::string toneMapTypeName(ToneMappingType type) {
    if (auto name = rtpost::modernToneName(type)) return name;
    switch (type) {
        case ToneMappingType::AGX: return "agx";
        case ToneMappingType::ACES: return "aces";
        case ToneMappingType::Uncharted: return "uncharted";
        case ToneMappingType::Filmic: return "filmic";
        case ToneMappingType::None: return "none";
    }
    return "none";
}

bool parseToneMapType(const std::string& name, ToneMappingType& out) {
    std::string s = name;
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (rtpost::parseModernTone(s, out)) return true;
    if (s == "agx") { out = ToneMappingType::AGX; return true; }
    if (s == "aces") { out = ToneMappingType::ACES; return true; }
    if (s == "uncharted") { out = ToneMappingType::Uncharted; return true; }
    if (s == "filmic") { out = ToneMappingType::Filmic; return true; }
    if (s == "none") { out = ToneMappingType::None; return true; }
    return false;
}

void postChanged() {
    if (g_ctx) {
        rtpost::syncDisplay(g_ctx->color_processor, g_ctx->scene.camera.get(), false);
        g_ctx->apply_tonemap = true;
        g_ctx->render_settings.persistent_tonemap = true;
    }
}

void stylizeChanged() {
    if (g_ctx) {
        g_ctx->render_settings.stylize_enabled = g_ctx->renderer.stylizeMode.enabled;
        stylize_redisplay = true;
    }
}

} // namespace

Result getPost(PostState& out) {
    if (!g_ctx) return notBound();
    const auto& params = g_ctx->color_processor.params;
    out.exposure = params.global_exposure;
    out.gamma = params.global_gamma;
    out.saturation = params.saturation;
    out.color_temperature = params.color_temperature;
    out.tone_mapping = toneMapTypeName(params.tone_mapping_type);
    out.vignette_enabled = params.enable_vignette;
    out.vignette_strength = params.vignette_strength;
    out.stylize_enabled = g_ctx->renderer.stylizeMode.enabled;
    out.stylize_strength = g_ctx->renderer.stylizeMode.profile.global_strength;
    return Result::success();
}

Result setPostExposure(float exposure) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (!std::isfinite(exposure) || exposure < 0.0f || exposure > 65504.0f) return Result::fail("exposure must be finite and in [0,65504]");
    g_ctx->color_processor.params.global_exposure = exposure;
    postChanged();
    return Result::success();
}

Result setPostGamma(float gamma) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (!std::isfinite(gamma) || gamma < 0.1f || gamma > 10.0f) return Result::fail("gamma must be finite and in [0.1,10]");
    g_ctx->color_processor.params.global_gamma = gamma;
    postChanged();
    return Result::success();
}

Result setPostSaturation(float saturation) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (!std::isfinite(saturation) || saturation < 0.0f || saturation > 4.0f) return Result::fail("saturation must be finite and in [0,4]");
    g_ctx->color_processor.params.saturation = saturation;
    postChanged();
    return Result::success();
}

Result setPostColorTemperature(float temp_k) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (!std::isfinite(temp_k) || temp_k < 4000.0f || temp_k > 25000.0f) return Result::fail("white balance must be finite and in [4000,25000] K");
    g_ctx->color_processor.params.color_temperature = temp_k;
    postChanged();
    return Result::success();
}

Result setPostToneMapping(const std::string& type) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    ToneMappingType t;
    if (!parseToneMapType(type, t))
        return Result::fail("unknown tone mapping type '" + type + "' (expected agx|aces_fitted|uncharted|filmic|linear|reinhard)");
    g_ctx->color_processor.params.tone_mapping_type = t;
    postChanged();
    return Result::success();
}

Result setPostVignetteEnabled(bool enabled) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    g_ctx->color_processor.params.enable_vignette = enabled;
    postChanged();
    return Result::success();
}

Result setPostVignetteStrength(float strength) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (!std::isfinite(strength) || strength < 0.0f || strength > 2.0f) return Result::fail("vignette strength must be non-negative");
    g_ctx->color_processor.params.vignette_strength = strength;
    postChanged();
    return Result::success();
}

Result setPostStylizeEnabled(bool enabled) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    g_ctx->renderer.stylizeMode.enabled = enabled;
    stylizeChanged();
    return Result::success();
}

Result setPostStylizeStrength(float strength) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (!std::isfinite(strength) || strength < 0.0f || strength > 2.0f) return Result::fail("stylize strength must be non-negative");
    g_ctx->renderer.stylizeMode.profile.global_strength = strength;
    stylizeChanged();
    return Result::success();
}


}
