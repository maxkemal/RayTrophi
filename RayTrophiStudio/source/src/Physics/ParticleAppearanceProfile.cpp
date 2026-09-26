#include "ParticleAppearanceProfile.h"

#include <algorithm>
#include <cmath>

namespace RayTrophiSim {
namespace {

constexpr float kNeutralOpacity = 1.0f;
constexpr float kNeutralSize = 0.05f;
constexpr float kNeutralEmission = 1.0f;

// Old ParticleEmitterDesc defaults: a legacy emitter JSON that omits a key
// meant exactly these values.
constexpr float kLegacyStartSize = 0.06f;
constexpr float kLegacyEndSize = 0.02f;
constexpr float kLegacyStartOpacity = 1.0f;
constexpr float kLegacyEndOpacity = 0.0f;
const Vec3 kLegacyStartColor(1.0f, 0.85f, 0.5f);
const Vec3 kLegacyEndColor(1.0f, 0.25f, 0.08f);

float evaluateCurve(const std::vector<ParticleCurveKey>& keys, float t, float neutral) {
    if (keys.empty()) {
        return neutral;
    }
    if (t <= keys.front().t) {
        return keys.front().value;
    }
    for (std::size_t i = 1; i < keys.size(); ++i) {
        if (t <= keys[i].t) {
            const float span = keys[i].t - keys[i - 1].t;
            const float w = span > 1e-6f ? (t - keys[i - 1].t) / span : 1.0f;
            return keys[i - 1].value + (keys[i].value - keys[i - 1].value) * w;
        }
    }
    return keys.back().value;
}

Vec3 evaluateRamp(const std::vector<ParticleColorStop>& stops, float t) {
    if (stops.empty()) {
        return Vec3(1.0f, 1.0f, 1.0f);
    }
    if (t <= stops.front().t) {
        return stops.front().color;
    }
    for (std::size_t i = 1; i < stops.size(); ++i) {
        if (t <= stops[i].t) {
            const float span = stops[i].t - stops[i - 1].t;
            const float w = span > 1e-6f ? (t - stops[i - 1].t) / span : 1.0f;
            return stops[i - 1].color + (stops[i].color - stops[i - 1].color) * w;
        }
    }
    return stops.back().color;
}

bool finite(float v) {
    return std::isfinite(v);
}

std::string validateCurve(const std::vector<ParticleCurveKey>& keys, const char* label,
                          float min_value, float max_value) {
    if (keys.size() > kParticleAppearanceMaxKeys) {
        return std::string(label) + " has more than " +
               std::to_string(kParticleAppearanceMaxKeys) + " keys";
    }
    for (const auto& key : keys) {
        if (!finite(key.t) || key.t < 0.0f || key.t > 1.0f) {
            return std::string(label) + " key t must be in [0, 1]";
        }
        if (!finite(key.value) || key.value < min_value || key.value > max_value) {
            return std::string(label) + " value out of range [" +
                   std::to_string(min_value) + ", " + std::to_string(max_value) + "]";
        }
    }
    return {};
}

nlohmann::json curveToJson(const std::vector<ParticleCurveKey>& keys) {
    nlohmann::json out = nlohmann::json::array();
    for (const auto& key : keys) {
        out.push_back({key.t, key.value});
    }
    return out;
}

void curveFromJson(const nlohmann::json& j, std::vector<ParticleCurveKey>& out) {
    out.clear();
    if (!j.is_array()) {
        return;
    }
    for (const auto& item : j) {
        if (item.is_array() && item.size() >= 2) {
            out.push_back({item[0].get<float>(), item[1].get<float>()});
        }
    }
}

Vec3 legacyColor(const nlohmann::json& j, const char* key, const Vec3& fallback) {
    if (!j.contains(key)) {
        return fallback;
    }
    const auto& c = j[key];
    if (c.is_array() && c.size() >= 3) {
        return Vec3(c[0].get<float>(), c[1].get<float>(), c[2].get<float>());
    }
    if (c.is_object()) {
        return Vec3(c.value("x", fallback.x), c.value("y", fallback.y), c.value("z", fallback.z));
    }
    return fallback;
}

} // namespace

ParticleAppearanceSample evaluateParticleAppearanceCurves(
    const ParticleAppearanceProfile& profile, float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    ParticleAppearanceSample s;
    s.color = evaluateRamp(profile.color_ramp, t);
    s.opacity = evaluateCurve(profile.opacity_curve, t, kNeutralOpacity);
    s.size = evaluateCurve(profile.size_curve, t, kNeutralSize);
    s.emission = evaluateCurve(profile.emission_curve, t, kNeutralEmission);
    return s;
}

void bakeParticleAppearanceLut(const ParticleAppearanceProfile& profile,
                               std::vector<float>& row) {
    row.assign(kParticleAppearanceLutFloatsPerRow, 0.0f);
    for (int i = 0; i < kParticleAppearanceLutSamples; ++i) {
        const float t = static_cast<float>(i) /
                        static_cast<float>(kParticleAppearanceLutSamples - 1);
        const ParticleAppearanceSample s = evaluateParticleAppearanceCurves(profile, t);
        float* out = row.data() + static_cast<std::size_t>(i) *
                                      kParticleAppearanceLutFloatsPerSample;
        out[0] = s.color.x;
        out[1] = s.color.y;
        out[2] = s.color.z;
        out[3] = s.opacity;
        out[4] = s.size;
        out[5] = s.emission;
    }
}

ParticleAppearanceSample sampleParticleAppearanceLut(const float* row, float t) {
    ParticleAppearanceSample s;
    if (!row) {
        return s;
    }
    if (!std::isfinite(t)) {
        t = 0.0f;
    }
    const float x = std::clamp(t, 0.0f, 1.0f) *
                    static_cast<float>(kParticleAppearanceLutSamples - 1);
    const int i0 = std::min(static_cast<int>(x), kParticleAppearanceLutSamples - 1);
    const int i1 = std::min(i0 + 1, kParticleAppearanceLutSamples - 1);
    const float w = x - static_cast<float>(i0);
    const float* a = row + static_cast<std::size_t>(i0) * kParticleAppearanceLutFloatsPerSample;
    const float* b = row + static_cast<std::size_t>(i1) * kParticleAppearanceLutFloatsPerSample;
    auto mix = [w](float p, float q) { return p + (q - p) * w; };
    s.color = Vec3(mix(a[0], b[0]), mix(a[1], b[1]), mix(a[2], b[2]));
    s.opacity = mix(a[3], b[3]);
    s.size = mix(a[4], b[4]);
    s.emission = mix(a[5], b[5]);
    return s;
}

std::string normalizeParticleAppearanceProfile(ParticleAppearanceProfile& profile) {
    auto byT = [](const auto& a, const auto& b) { return a.t < b.t; };
    std::stable_sort(profile.color_ramp.begin(), profile.color_ramp.end(), byT);
    std::stable_sort(profile.opacity_curve.begin(), profile.opacity_curve.end(), byT);
    std::stable_sort(profile.size_curve.begin(), profile.size_curve.end(), byT);
    std::stable_sort(profile.emission_curve.begin(), profile.emission_curve.end(), byT);

    if (profile.name.empty()) {
        return "appearance profile name must not be empty";
    }
    if (profile.color_ramp.size() > kParticleAppearanceMaxKeys) {
        return "color_ramp has more than " + std::to_string(kParticleAppearanceMaxKeys) +
               " stops";
    }
    for (const auto& stop : profile.color_ramp) {
        if (!finite(stop.t) || stop.t < 0.0f || stop.t > 1.0f) {
            return "color_ramp stop t must be in [0, 1]";
        }
        if (!finite(stop.color.x) || !finite(stop.color.y) || !finite(stop.color.z) ||
            stop.color.x < 0.0f || stop.color.y < 0.0f || stop.color.z < 0.0f) {
            return "color_ramp colour must be finite and >= 0";
        }
    }
    if (auto e = validateCurve(profile.opacity_curve, "opacity_curve", 0.0f, 1.0f); !e.empty()) {
        return e;
    }
    if (auto e = validateCurve(profile.size_curve, "size_curve", 0.0f, 1000.0f); !e.empty()) {
        return e;
    }
    if (auto e = validateCurve(profile.emission_curve, "emission_curve", 0.0f, 1000.0f);
        !e.empty()) {
        return e;
    }
    return {};
}

const ParticleAppearanceProfile& fallbackParticleAppearance() {
    static const ParticleAppearanceProfile profile = [] {
        ParticleAppearanceProfile p;
        p.id = 0;
        p.name = "Fallback";
        p.blend = ParticleAppearanceBlend::Additive;
        p.color_ramp = {{0.0f, Vec3(1.0f, 1.0f, 1.0f)}};
        p.opacity_curve = {{0.0f, 1.0f}, {1.0f, 0.0f}};
        p.size_curve = {{0.0f, 1.0f}};
        p.emission_curve = {{0.0f, 1.0f}};
        return p;
    }();
    return profile;
}

ParticleAppearanceProfile makeTwoKeyParticleAppearance(
    const std::string& name, ParticleAppearanceBlend blend,
    float start_size, float end_size,
    float start_opacity, float end_opacity,
    const Vec3& start_color, const Vec3& end_color,
    float emission) {
    ParticleAppearanceProfile p;
    p.name = name;
    p.blend = blend;
    p.color_ramp = {{0.0f, start_color}, {1.0f, end_color}};
    p.opacity_curve = {{0.0f, std::clamp(start_opacity, 0.0f, 1.0f)},
                       {1.0f, std::clamp(end_opacity, 0.0f, 1.0f)}};
    p.size_curve = {{0.0f, std::max(0.0f, start_size)}, {1.0f, std::max(0.0f, end_size)}};
    p.emission_curve = {{0.0f, std::max(0.0f, emission)}};
    return p;
}

const char* particleAppearanceBlendName(ParticleAppearanceBlend blend) {
    return blend == ParticleAppearanceBlend::Alpha ? "alpha" : "additive";
}

bool parseParticleAppearanceBlend(const std::string& name, ParticleAppearanceBlend& out) {
    if (name == "additive" || name == "add") {
        out = ParticleAppearanceBlend::Additive;
        return true;
    }
    if (name == "alpha") {
        out = ParticleAppearanceBlend::Alpha;
        return true;
    }
    return false;
}

nlohmann::json serializeParticleAppearanceProfile(const ParticleAppearanceProfile& profile) {
    nlohmann::json j;
    j["id"] = profile.id;
    j["name"] = profile.name;
    j["blend"] = particleAppearanceBlendName(profile.blend);
    nlohmann::json ramp = nlohmann::json::array();
    for (const auto& stop : profile.color_ramp) {
        ramp.push_back({stop.t, stop.color.x, stop.color.y, stop.color.z});
    }
    j["color_ramp"] = std::move(ramp);
    j["opacity_curve"] = curveToJson(profile.opacity_curve);
    j["size_curve"] = curveToJson(profile.size_curve);
    j["emission_curve"] = curveToJson(profile.emission_curve);
    return j;
}

bool deserializeParticleAppearanceProfile(const nlohmann::json& j,
                                          ParticleAppearanceProfile& out) {
    if (!j.is_object() || !j.contains("id")) {
        return false;
    }
    out = ParticleAppearanceProfile{};
    out.id = j.value("id", 0u);
    out.name = j.value("name", out.name);
    ParticleAppearanceBlend blend = ParticleAppearanceBlend::Additive;
    if (parseParticleAppearanceBlend(j.value("blend", std::string("additive")), blend)) {
        out.blend = blend;
    }
    if (j.contains("color_ramp") && j["color_ramp"].is_array()) {
        for (const auto& item : j["color_ramp"]) {
            if (item.is_array() && item.size() >= 4) {
                out.color_ramp.push_back({item[0].get<float>(),
                                          Vec3(item[1].get<float>(), item[2].get<float>(),
                                               item[3].get<float>())});
            }
        }
    }
    if (j.contains("opacity_curve")) curveFromJson(j["opacity_curve"], out.opacity_curve);
    if (j.contains("size_curve")) curveFromJson(j["size_curve"], out.size_curve);
    if (j.contains("emission_curve")) curveFromJson(j["emission_curve"], out.emission_curve);
    return out.id != 0;
}

ParticleAppearanceProfile migrateLegacyEmitterAppearance(
    const nlohmann::json& emitter_json, int legacy_blend,
    const std::string& emitter_name) {
    const auto num = [&](const char* key, float fallback) {
        return emitter_json.contains(key) && emitter_json[key].is_number()
            ? emitter_json[key].get<float>() : fallback;
    };
    return makeTwoKeyParticleAppearance(
        emitter_name.empty() ? std::string("Emitter Appearance")
                             : emitter_name + " Appearance",
        legacy_blend == 1 ? ParticleAppearanceBlend::Alpha : ParticleAppearanceBlend::Additive,
        num("start_size", kLegacyStartSize), num("end_size", kLegacyEndSize),
        num("start_opacity", kLegacyStartOpacity), num("end_opacity", kLegacyEndOpacity),
        legacyColor(emitter_json, "start_color", kLegacyStartColor),
        legacyColor(emitter_json, "end_color", kLegacyEndColor));
}

} // namespace RayTrophiSim
