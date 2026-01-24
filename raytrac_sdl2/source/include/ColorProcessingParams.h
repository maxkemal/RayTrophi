/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          ColorProcessingParams.h
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once
#include <iostream>
#include "Vec3.h"
#include "globals.h"

enum class ToneMappingType {
    AGX,
    ACES,
    Uncharted,
    Filmic,
    None
};


class ColorProcessor {
public:
    
    struct ColorProcessingParams {
        float global_exposure = exposure;
        float global_gamma = gamma;
        bool use_adaptive_exposure = false;
        bool use_local_tone_mapping = true;
        float key_value = 0.18f;
        float local_contrast = 0.02f;
        float saturation = 1.0;  // Yeni: Renk doygunluğu faktörü (1.0 = değişiklik yok)
        float color_temperature = 6500.0f; // Yeni: Kelvin cinsinden renk sıcaklığı
        ToneMappingType tone_mapping_type = ToneMappingType::None;
        bool enable_vignette = true;                // <<< yeni eklendi
        float vignette_strength = 0.0f;             // <<< isteğe bağlı yoğunluk kontrolü
    };
    // Improved LogCTransform helper function
    float LogCTransform(float x) {
        // Avoid division by zero or negative values
        x = std::max(x, 1e-10f);

        // Adjusted LogC transformation parameters for better color preservation
        constexpr float cut = 0.010591f;
        constexpr float a = 5.555556f;
        constexpr float b = 0.052272f;
        constexpr float c = 0.247190f;
        constexpr float d = 0.385537f;
        constexpr float e = 5.367655f;
        constexpr float f = 0.092809f;

        if (x > cut)
            return c * log10(a * x + b) + d;
        else
            return e * x + f;
    }
    ColorProcessingParams params;
    // Improved AGX tonemapping with better color preservation
    Vec3 AGXToneMapping(const Vec3& color) {
        // Find maximum value to preserve color ratios
        float maxVal = std::max(std::max(color.x, color.y), color.z);
        maxVal = std::max(maxVal, 1e-5f); // Avoid division by zero

        // Preserve color ratios by normalizing
        Vec3 normalizedColor(
            color.x / maxVal,
            color.y / maxVal,
            color.z / maxVal
        );

        // AGX constants with adjusted range for more vibrant colors
        const float minEv = -12.0f;  // Slightly adjusted for better shadows
        const float maxEv = 6.0f;    // Increased for better highlights 

        // Apply logC transform to luminance only (preserves color better)
        float luminance = 0.2126f * color.x + 0.7152f * color.y + 0.0722f * color.z;
        float logLuminance = LogCTransform(luminance);

        // Apply the AGX S-curve remapping to luminance
        auto agxSCurve = [minEv, maxEv](float x) -> float {
            // Normalize to 0-1 range
            x = (x - minEv) / (maxEv - minEv);
            // Apply modified sigmoid curve for more vibrant output
            x = std::clamp(x, 0.0f, 1.0f);

            // More vibrant curve parameters
            float s = 0.65f;  // Adjusted shoulder position
            float t = 0.95f;  // Higher contrast

            float y;
            if (x < s) {
                y = (x / s) * t;
            }
            else {
                y = t + ((x - s) / (1.0f - s)) * (1.0f - t);
            }
            return y;
            };

        // Apply S-curve to luminance
        float remappedLuminance = agxSCurve(logLuminance);

        // Scale color by new luminance but preserve original color ratios
        Vec3 remappedColor = normalizedColor * remappedLuminance * maxVal;

        // Enhanced color grading for more vibrancy - boost saturation
        Vec3 gradedColor;
        gradedColor.x = remappedColor.x * 1.15f;  // More red
        gradedColor.y = remappedColor.y * 1.05f;  // Slightly more green
        gradedColor.z = remappedColor.z * 1.0f;   // Keep blue

        // Apply final clamp
        return Vec3(
            std::clamp(gradedColor.x, 0.0f, 1.0f),
            std::clamp(gradedColor.y, 0.0f, 1.0f),
            std::clamp(gradedColor.z, 0.0f, 1.0f)
        );
    }
    void resize(int w, int h) {
        width = w;
        height = h;
		// Resize luminance map to match new dimensions
		luminance_map.resize(w * h, 0.0f);

    }

private:
  
    std::vector<float> luminance_map;
    int width, height;

    float calculateAdaptiveLuminance(const std::vector<Vec3>& frame_buffer) {
        float log_avg_luminance = 0.0f;
        for (const auto& color : frame_buffer) {
            float luminance = 0.2126f * color.x + 0.7152f * color.y + 0.0722f * color.z;
            log_avg_luminance += std::log(luminance + 1e-6f);
        }
        log_avg_luminance = std::exp(log_avg_luminance / frame_buffer.size());
        return params.key_value / log_avg_luminance;
    }

    Vec3 applyLocalToneMapping(const Vec3& color, int x, int y) {
        float local_luminance = luminance_map[y * width + x];
        float local_exposure = params.local_contrast / (local_luminance + 0.05f);
        return color * local_exposure;
    }

    // Yeni: Renk doygunluğunu ayarlama
    Vec3 adjustSaturation(const Vec3& color, float saturation_factor) {
        float luminance = 0.2126f * color.x + 0.7152f * color.y + 0.0722f * color.z;
        return Vec3(
            luminance + (color.x - luminance) * saturation_factor,
            luminance + (color.y - luminance) * saturation_factor,
            luminance + (color.z - luminance) * saturation_factor
        );
    }

    // Yeni: Renk sıcaklığını ayarlama
    Vec3 adjustColorTemperature(const Vec3& color, float temperature) {
        float t = (temperature - 6500.0f) / 1000.0f;  // 6500K nötr kabul edilir
        float r_factor = 1.0f + t * 0.1f;
        float b_factor = 1.0f - t * 0.1f;
        return Vec3(
            std::clamp(color.x * r_factor, 0.0f, 1.0f),
            color.y,
            std::clamp(color.z * b_factor, 0.0f, 1.0f)
        );
    }
    Vec3 FilmicTonemap(const Vec3& x) {
        const float A = 2.0f;
        return x / (x + Vec3(A));
    }
    Vec3 UnchartedFilmic(const Vec3& x) {
        const float A = 0.15f;
        const float B = 0.50f;
        const float C = 0.10f;
        const float D = 0.20f;
        const float E = 0.02f;
        const float F = 0.30f;

        auto tonemap = [&](float v) {
            return ((v * (A * v + C * B) + D * E) / (v * (A * v + B) + D * F)) - E / F;
            };

        return Vec3(
            std::clamp(tonemap(x.x), 0.0f, 1.0f),
            std::clamp(tonemap(x.y), 0.0f, 1.0f),
            std::clamp(tonemap(x.z), 0.0f, 1.0f)
        );
    }

public:
    ColorProcessor() : width(0), height(0) {}
    ColorProcessor(int w, int h) : width(w), height(h) {
        luminance_map.resize(w * h, 0.0f);
    }
    float linearToSRGB(float linear) {
        return (linear <= 0.0031308f) ? 12.92f * linear : 1.055f * std::pow(linear, 1.0f /2.4f) - 0.055f;
    }
 

    Vec3 ACESFilmicToneMapping(const Vec3& color) {
        float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
        return Vec3(
            std::clamp((color.x * (a * color.x + b)) / (color.x * (c * color.x + d) + e), 0.0f, 1.0f),
            std::clamp((color.y * (a * color.y + b)) / (color.y * (c * color.y + d) + e), 0.0f, 1.0f),
            std::clamp((color.z * (a * color.z + b)) / (color.z * (c * color.z + d) + e), 0.0f, 1.0f)
        );
    }

    void setParams(const ColorProcessingParams& new_params) {
        params = new_params;
    }

    void preprocess(const std::vector<Vec3>& frame_buffer) {
        if (params.use_adaptive_exposure) {
            params.global_exposure = calculateAdaptiveLuminance(frame_buffer);
            params.global_exposure = std::clamp(params.global_exposure, 0.1f, 5.0f);
        }

        if (params.use_local_tone_mapping) {
            for (int i = 0; i < frame_buffer.size(); ++i) {
                const auto& color = frame_buffer[i];
                luminance_map[i] = 0.2126f * color.x + 0.7152f * color.y + 0.0722f * color.z;
            }
        }
    }

    Vec3 processColor(const Vec3& color, int x, int y) {
        Vec3 processed_color = color * params.global_exposure;

        // AGX Filmic ton eşleme
        //  processed_color = AGXToneMapping(processed_color);
       // ACES filmic ton eşleştirme 
       // processed_color = ACESFilmicToneMapping(processed_color);
       // UnchartedFilmic ton eşleştirme
       // processed_color = UnchartedFilmic(processed_color);
        // Filmic ton eşleştirme
       // processed_color = FilmicTonemap(processed_color);
        // Renk sıcaklığı ayarı
        processed_color = adjustColorTemperature(processed_color, params.color_temperature);

        // Renk doygunluğu ayarı
        processed_color = adjustSaturation(processed_color, params.saturation);
        switch (params.tone_mapping_type) {
        case ToneMappingType::AGX:
            processed_color = AGXToneMapping(processed_color);
            break;
        case ToneMappingType::ACES:
            processed_color = ACESFilmicToneMapping(processed_color);
            break;
        case ToneMappingType::Uncharted:
            processed_color = UnchartedFilmic(processed_color);
            break;
        case ToneMappingType::Filmic:
            processed_color = FilmicTonemap(processed_color);
            break;
        case ToneMappingType::None:
            // no tonemapping
            break;
        }
        // Gamma düzeltmesi
        float gamma_adjust = 1.0f / params.global_gamma;
        processed_color = Vec3(
            std::pow(processed_color.x, gamma_adjust),
            std::pow(processed_color.y, gamma_adjust),
            std::pow(processed_color.z, gamma_adjust)
        );
        
        // SADECE EN SON: 0-1 clamp (ekrana gidecek, mecbur)
        return Vec3::clamp(processed_color, 0.0f, 1.0f);
    }
};

