#pragma once
#include "Vec3.h"
#include "Texture.h"
#include <string>
#include <optional>
#include <globals.h>

class AtmosphericEffects {
private:
    float fog_start_distance;
    float fog_base_density;
    float fog_distance_factor;
    float haze_density;
    Vec3 fog_color;
    Vec3 haze_color;
    std::optional<Texture> fog_texture;
    std::optional<Texture> haze_texture;
    std::optional<Texture> background_texture;
    Vec3 background_color;
    //bool use_background_texture;

public:
    bool use_background_texture;
    Vec3 attenuateSegment(const Vec3& color, float start_distance, float end_distance) const;
    Vec3 calculateSegmentContribution(float start_distance, float end_distance) const;

    AtmosphericEffects(
        float fog_start = 0.0f,
        float fog_base = 0.0f,
        float fog_factor = 0.0f,
        float haze = 0.0f,
        const Vec3& fog_col = Vec3(0.5f, 0.5f, 0.5f),
        const Vec3& haze_col = Vec3(0.7f, 0.8f, 1.0f),
        const Vec3& bg_col = Vec3(0.0f, 0.0f, 0.0f));

    void setFogTexture(const std::string& texture_path);
    void setHazeTexture(const std::string& texture_path);
    void setBackgroundTexture(const std::string& texture_path);
    void setBackgroundColor(const Vec3& color);
    Vec3 getBackgroundColor(float u = 0.0f, float v = 0.0f) const;

    void setFogStartDistance(float distance);
    void setFogBaseDensity(float density);
    void setFogDistanceFactor(float factor);
    void setHazeDensity(float density);
    void setFogColor(const Vec3& color);
    void setHazeColor(const Vec3& color);

    float getFogStartDistance() const;
    float getFogBaseDensity() const;
    float getFogDistanceFactor() const;
    float getHazeDensity() const;
    Vec3 getFogColor() const;
    Vec3 getHazeColor() const;

    float calculateFogFactor(float distance) const;
    float calculateHazeFactor(float distance) const;
    Vec3 applyAtmosphericEffects(const Vec3& color, float distance, float u = 0.0f, float v = 0.0f) const;
};