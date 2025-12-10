#include "AtmosphericEffects.h"
#include <algorithm>
#include <cmath>


AtmosphericEffects::AtmosphericEffects(
    float fog_start, float fog_base, float fog_factor, float haze,
    const Vec3& fog_col, const Vec3& haze_col, const Vec3& bg_col)
    : fog_start_distance(fog_start), fog_base_density(fog_base),
    fog_distance_factor(fog_factor), haze_density(haze),
    fog_color(fog_col), haze_color(haze_col),
    background_color(bg_col), use_background_texture(false) {}

void AtmosphericEffects::setFogTexture(const std::string& texture_path) {
    if (!texture_path.empty()) {
//        fog_texture.emplace(texture_path);
    }
    else {
        fog_texture.reset();
    }
}

void AtmosphericEffects::setHazeTexture(const std::string& texture_path) {
    if (!texture_path.empty()) {
//        haze_texture.emplace(texture_path);
    }
    else {
        haze_texture.reset();
    }
}

void AtmosphericEffects::setBackgroundTexture(const std::string& texture_path) {
    if (!texture_path.empty()) {
       // background_texture = Texture(texture_path); // Doðrudan atama
        use_background_texture = true;
    }
    else {
        background_texture.reset(); // Boþ deðer
        use_background_texture = false;
    }
}



void AtmosphericEffects::setBackgroundColor(const Vec3& color) {
    background_color = color;
    use_background_texture = false;
    background_texture.reset();
}

Vec3 AtmosphericEffects::getBackgroundColor(float u, float v) const {
    if (use_background_texture && background_texture.has_value()) {
        return background_texture->get_color(u, v);
    }
    return background_color;
}

void AtmosphericEffects::setFogStartDistance(float distance) {
    fog_start_distance = distance;
}

void AtmosphericEffects::setFogBaseDensity(float density) {
    fog_base_density = density;
}

void AtmosphericEffects::setFogDistanceFactor(float factor) {
    fog_distance_factor = factor;
}

void AtmosphericEffects::setHazeDensity(float density) {
    haze_density = density;
}

void AtmosphericEffects::setFogColor(const Vec3& color) {
    fog_color = color;
}

void AtmosphericEffects::setHazeColor(const Vec3& color) {
    haze_color = color;
}

float AtmosphericEffects::getFogStartDistance() const {
    return fog_start_distance;
}

float AtmosphericEffects::getFogBaseDensity() const {
    return fog_base_density;
}

float AtmosphericEffects::getFogDistanceFactor() const {
    return fog_distance_factor;
}

float AtmosphericEffects::getHazeDensity() const {
    return haze_density;
}

Vec3 AtmosphericEffects::getFogColor() const {
    return fog_color;
}

Vec3 AtmosphericEffects::getHazeColor() const {
    return haze_color;
}

float AtmosphericEffects::calculateFogFactor(float distance) const {
    if (distance <= fog_start_distance) return 0.0f;
    float fog_distance = distance - fog_start_distance;
    return std::min(1.0f, fog_base_density + fog_distance * fog_distance_factor);
}

float AtmosphericEffects::calculateHazeFactor(float distance) const {
    return std::min(1.0f, distance * haze_density);
}
Vec3 AtmosphericEffects::attenuateSegment(const Vec3& color, float start_distance, float end_distance) const {
    if (!atmosferic_effect_enabled) return color;

    float start_fog_factor = calculateFogFactor(start_distance);
    float end_fog_factor = calculateFogFactor(end_distance);
    float start_haze_factor = calculateHazeFactor(start_distance);
    float end_haze_factor = calculateHazeFactor(end_distance);

    float avg_fog_factor = (start_fog_factor + end_fog_factor) * 0.5f;
    float avg_haze_factor = (start_haze_factor + end_haze_factor) * 0.5f;

    Vec3 fog_contribution = fog_texture.has_value() ? fog_texture->get_color(0.5f, 0.5f) : Vec3(fog_color);
    Vec3 haze_contribution = haze_texture.has_value() ? haze_texture->get_color(0.5f, 0.5f) : Vec3(haze_color);

    return color * (1.0f - avg_fog_factor) * (1.0f - avg_haze_factor) +
        fog_contribution * avg_fog_factor +
        haze_contribution * avg_haze_factor;
}

Vec3 AtmosphericEffects::calculateSegmentContribution(float start_distance, float end_distance) const {
    if (!atmosferic_effect_enabled) return Vec3(0.0f, 0.0f, 0.0f);

    float start_fog_factor = calculateFogFactor(start_distance);
    float end_fog_factor = calculateFogFactor(end_distance);
    float start_haze_factor = calculateHazeFactor(start_distance);
    float end_haze_factor = calculateHazeFactor(end_distance);

    float fog_contribution = end_fog_factor - start_fog_factor;
    float haze_contribution = end_haze_factor - start_haze_factor;

    Vec3 fog_color_contribution = fog_texture.has_value() ? fog_texture->get_color(0.5f, 0.5f) : Vec3(fog_color);
    Vec3 haze_color_contribution = haze_texture.has_value() ? haze_texture->get_color(0.5f, 0.5f) : Vec3(haze_color);

    return fog_color_contribution * fog_contribution + haze_color_contribution * haze_contribution;
}
Vec3 AtmosphericEffects::applyAtmosphericEffects(const Vec3& color, float distance, float u, float v) const {
    if (!atmosferic_effect_enabled) return color;

    float fog_factor = calculateFogFactor(distance);
    float haze_factor = calculateHazeFactor(distance);

    Vec3 fog_contribution = fog_texture.has_value() ? fog_texture->get_color(u, v) : Vec3(fog_color);
    Vec3 haze_contribution = haze_texture.has_value() ? haze_texture->get_color(u, v) : haze_color;
    Vec3 background_contribution = getBackgroundColor(u, v);

    Vec3 result = color * (1.0f - fog_factor) * (1.0f - haze_factor) +
        fog_contribution * fog_factor +
        haze_contribution * haze_factor;

    // Uzak mesafelerde arka plan rengine/texture'ýna geçiþ yap
    float background_blend = std::min(1.0f, distance / 1000.0f);  // 1000 birim mesafede tam arka plana geç
    result = result * (1.0f - background_blend) + background_contribution * background_blend;

    return result;
}