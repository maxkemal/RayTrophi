/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          PhysicalCamera.h
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// PHYSICAL CAMERA SYSTEM - RayTrophi Engine
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// Tam fiziksel kamera simÃ¼lasyonu:
//   - GerÃ§ek lens optiÄŸi (distortion, CA, vignette, flare, breathing)
//   - Fiziksel sensÃ¶r modeli (exposure, dynamic range, noise davranÄ±ÅŸÄ±)
//   - Fizik tabanlÄ± kamera hareketleri (atalet, sÃ¶nÃ¼mleme, sarsÄ±ntÄ±)
//   - Sinematik rig sistemleri (dolly, crane, steadicam)
//
// TasarÄ±m felsefesi:
//   - ISO noise EKLEMEZ, var olan Monte Carlo varyansÄ±nÄ± gÃ¶rÃ¼nÃ¼r yapar
//   - Shutter speed motion blur'u kontrol eder (temporal sampling)
//   - TÃ¼m hareketler fizik yasalarÄ±na uygun (F=ma, sÃ¶nÃ¼mleme)
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

#include "Vec3.h"
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace PhysicalCameraSystem {

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// PHYSICAL CONSTANTS (SI Units)
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
namespace Constants {
    constexpr float SPEED_OF_LIGHT = 299792458.0f;      // m/s
    constexpr float PLANCK_CONSTANT = 6.62607e-34f;    // JÂ·s
    
    // Film/Sensor reference sizes (mm)
    constexpr float FULL_FRAME_WIDTH = 36.0f;
    constexpr float FULL_FRAME_HEIGHT = 24.0f;
    constexpr float SUPER35_WIDTH = 24.89f;
    constexpr float SUPER35_HEIGHT = 18.66f;
    constexpr float APSC_CANON_WIDTH = 22.3f;
    constexpr float APSC_CANON_HEIGHT = 14.9f;
    
    // Reference exposure (ISO 100, f/8, 1/125s = EV 13, sunny daylight)
    constexpr float REFERENCE_EV = 13.0f;
    constexpr float REFERENCE_LUMINANCE = 1000.0f;  // cd/mÂ² for EV 13
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// CAMERA MODE - Auto, Pro, Cinema
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
enum class CameraMode {
    Auto,       // Otomatik - basit kullanÄ±m, kÄ±sÄ±tlÄ± kontrol
    Pro,        // Profesyonel - manuel kontrol, temiz render
    Cinema      // Sinematik - tam fiziksel simÃ¼lasyon, tÃ¼m kusurlar
};

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// SENSOR TYPE
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
enum class SensorType {
    FullFrame,          // 36x24mm - crop 1.0x
    Super35,            // 24.89x18.66mm - crop 1.39x (Cinema standard)
    APSC_Canon,         // 22.3x14.9mm - crop 1.6x
    APSC_Sony,          // 23.6x15.6mm - crop 1.5x
    MicroFourThirds,    // 17.3x13mm - crop 2.0x
    MediumFormat,       // 43.8x32.9mm - crop 0.79x
    LargeFormat,        // 87x67mm - crop 0.41x
    IMAX                // 70.4x52.6mm - crop 0.48x
};

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// PHYSICAL SENSOR
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
struct PhysicalSensor {
    // === Fiziksel Boyutlar ===
    SensorType type = SensorType::FullFrame;
    float width_mm = 36.0f;
    float height_mm = 24.0f;
    float crop_factor = 1.0f;
    
    // === Elektriksel Ã–zellikler ===
    int native_iso = 100;               // SensÃ¶rÃ¼n doÄŸal ISO'su
    float dynamic_range_stops = 14.0f;  // Toplam dinamik aralÄ±k
    float max_well_capacity = 60000.0f; // Maksimum elektron kapasitesi (full well)
    float read_noise_electrons = 3.0f;  // Okuma gÃ¼rÃ¼ltÃ¼sÃ¼ (elektron)
    float dark_current = 0.01f;         // KaranlÄ±k akÄ±m (elektron/piksel/saniye)
    
    // === Quantization ===
    int bit_depth = 14;                 // ADC bit derinliÄŸi
    
    // === Hesaplanan DeÄŸerler ===
    float getPixelPitch(int resolution_width) const {
        return width_mm * 1000.0f / resolution_width; // mikrometreler
    }
    
    float getDiagonal() const {
        return sqrtf(width_mm * width_mm + height_mm * height_mm);
    }
    
    // Crop factor'e gÃ¶re sensÃ¶r tipini ayarla
    void setFromType(SensorType t) {
        type = t;
        switch (t) {
            case SensorType::FullFrame:
                width_mm = 36.0f; height_mm = 24.0f; crop_factor = 1.0f; break;
            case SensorType::Super35:
                width_mm = 24.89f; height_mm = 18.66f; crop_factor = 1.39f; break;
            case SensorType::APSC_Canon:
                width_mm = 22.3f; height_mm = 14.9f; crop_factor = 1.6f; break;
            case SensorType::APSC_Sony:
                width_mm = 23.6f; height_mm = 15.6f; crop_factor = 1.5f; break;
            case SensorType::MicroFourThirds:
                width_mm = 17.3f; height_mm = 13.0f; crop_factor = 2.0f; break;
            case SensorType::MediumFormat:
                width_mm = 43.8f; height_mm = 32.9f; crop_factor = 0.79f; break;
            case SensorType::LargeFormat:
                width_mm = 87.0f; height_mm = 67.0f; crop_factor = 0.41f; break;
            case SensorType::IMAX:
                width_mm = 70.4f; height_mm = 52.6f; crop_factor = 0.48f; break;
        }
    }
};

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// PHYSICAL LENS
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
struct PhysicalLens {
    // === Temel Optik Parametreler ===
    float focal_length_mm = 50.0f;          // Odak uzunluÄŸu
    float max_aperture = 1.4f;              // Maksimum aÃ§Ä±klÄ±k (f/1.4)
    float min_aperture = 22.0f;             // Minimum aÃ§Ä±klÄ±k (f/22)
    float current_aperture = 2.8f;          // Mevcut f-stop
    float min_focus_distance_m = 0.45f;     // Minimum odak mesafesi
    float max_focus_distance_m = 10000.0f;  // Sonsuz odak (pratik limit)
    int aperture_blades = 9;                // Diyafram yaprak sayÄ±sÄ±
    
    // === Cinema Lens (T-Stop) ===
    bool use_tstop = false;                 // T-stop kullanÄ±mÄ± (sinema lensleri iÃ§in)
    float light_transmission = 0.95f;       // IÅŸÄ±k geÃ§irgenliÄŸi (T = f / sqrt(transmission))
    
    // === Optik Kusurlar ===
    // Distortion (Brown-Conrady Model)
    float distortion_k1 = 0.0f;             // Radyal distortion (1. derece)
    float distortion_k2 = 0.0f;             // Radyal distortion (2. derece)
    float distortion_k3 = 0.0f;             // Radyal distortion (3. derece)
    float distortion_p1 = 0.0f;             // TeÄŸetsel distortion
    float distortion_p2 = 0.0f;             // TeÄŸetsel distortion
    
    // Chromatic Aberration
    float lateral_ca = 0.0f;                // Yanal renk sapmasÄ± (0-1)
    float longitudinal_ca = 0.0f;           // Eksenel renk sapmasÄ±
    
    // Vignetting (Natural + Mechanical)
    float natural_vignette = 0.0f;          // cos^4 dÃ¼ÅŸÃ¼ÅŸÃ¼ (0-1)
    float mechanical_vignette = 0.0f;       // Lens barrel engellemesi
    float vignette_falloff = 2.0f;          // DÃ¼ÅŸÃ¼ÅŸ eÄŸrisi Ã¼ssÃ¼
    
    // === Lens Flare ===
    float flare_threshold = 0.9f;           // Flare baÅŸlangÄ±Ã§ parlaklÄ±ÄŸÄ±
    float flare_intensity = 0.5f;           // Flare yoÄŸunluÄŸu
    float ghost_intensity = 0.3f;           // Ghost (yansÄ±ma) yoÄŸunluÄŸu
    int flare_elements = 6;                 // Ä°Ã§ yansÄ±ma element sayÄ±sÄ±
    bool anamorphic_flare = false;          // Anamorfik yatay flare
    
    // === Focus Breathing ===
    bool enable_breathing = false;          // Odak solumasÄ±
    float breathing_amount = 0.05f;         // FOV deÄŸiÅŸim oranÄ± (odak deÄŸiÅŸiminde)
    
    // === Bokeh Karakteri ===
    float spherical_aberration = 0.0f;      // Sferikal sapma (-1 = busy, 0 = neutral, +1 = bubble)
    float cats_eye_amount = 0.0f;           // Kenar bokeh ovalliÄŸi
    float swirl_amount = 0.0f;              // Helios-tarzÄ± dÃ¶ngÃ¼ bokeh
    float onion_ring = 0.0f;                // SoÄŸan halkalarÄ± (asferik lenslerde)
    
    // === Coating ===
    float coating_quality = 0.9f;           // Lens kaplama kalitesi (0 = flare dolu, 1 = temiz)
    
    // === Hesaplamalar ===
    // Efektif T-stop (Ä±ÅŸÄ±k geÃ§irgenliÄŸini hesaba katar)
    float getEffectiveTStop() const {
        if (use_tstop) {
            return current_aperture / sqrtf(light_transmission);
        }
        return current_aperture;
    }
    
    // Fiziksel aÃ§Ä±klÄ±k Ã§apÄ± (mm)
    float getApertureDiameter() const {
        return focal_length_mm / current_aperture;
    }
    
    // SensÃ¶re baÄŸlÄ± FOV hesaplama
    float getVerticalFOV(float sensor_height_mm) const {
        return 2.0f * atanf(sensor_height_mm / (2.0f * focal_length_mm)) * 180.0f / (float)M_PI;
    }
    
    float getHorizontalFOV(float sensor_width_mm) const {
        return 2.0f * atanf(sensor_width_mm / (2.0f * focal_length_mm)) * 180.0f / (float)M_PI;
    }
    
    // Focal length'e gÃ¶re otomatik distortion hesapla
    void calculateAutoDistortion() {
        if (focal_length_mm < 24.0f) {
            // Ultra geniÅŸ aÃ§Ä± - belirgin varil distorsiyonu
            float ratio = (24.0f - focal_length_mm) / 24.0f;
            distortion_k1 = -0.3f * ratio * ratio;
            distortion_k2 = -0.05f * ratio;
        } else if (focal_length_mm < 50.0f) {
            // GeniÅŸ aÃ§Ä± - hafif varil
            float ratio = (50.0f - focal_length_mm) / 26.0f;
            distortion_k1 = -0.1f * ratio;
        } else if (focal_length_mm > 85.0f) {
            // Telefoto - hafif yastÄ±k distorsiyonu
            float ratio = std::min((focal_length_mm - 85.0f) / 200.0f, 1.0f);
            distortion_k1 = 0.02f * ratio;
        } else {
            // Normal lens (50-85mm) - minimal distortion
            distortion_k1 = 0.0f;
        }
    }
    
    // Aperture'a gÃ¶re vignetting hesapla
    void calculateAutoVignette() {
        // GeniÅŸ aperture = daha fazla vignette
        float aperture_factor = 1.0f - ((current_aperture - max_aperture) / (min_aperture - max_aperture));
        natural_vignette = 0.3f * aperture_factor;
        
        // GeniÅŸ aÃ§Ä± lensler daha fazla vignette yapar
        if (focal_length_mm < 35.0f) {
            natural_vignette += 0.2f * (35.0f - focal_length_mm) / 35.0f;
        }
    }
};

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// PHYSICAL SHUTTER
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
struct PhysicalShutter {
    // === Shutter Tipi ===
    enum class Type {
        MechanicalFocalPlane,   // DSLR/Mirrorless - vertikal/horizontal travel
        MechanicalLeaf,         // Medium format - merkezi deklanÅŸÃ¶r
        Electronic,             // Tamamen elektronik (rolling shutter olabilir)
        GlobalElectronic        // Global shutter (tÃ¼m sensÃ¶r aynÄ± anda)
    };
    Type type = Type::MechanicalFocalPlane;
    
    // === Exposure SÃ¼resi ===
    float shutter_speed_seconds = 1.0f / 250.0f;    // 1/250s
    float shutter_angle_degrees = 180.0f;           // Cinema: 180Â° = %50 duty cycle
    
    // === Motion Blur KontrolÃ¼ ===
    bool enable_motion_blur = true;
    int motion_blur_samples = 8;                     // Temporal sample sayÄ±sÄ±
    
    // === Mekanik Etkileri ===
    float shutter_shock = 0.0f;                      // DeklanÅŸÃ¶r titreÅŸimi
    float curtain_travel_time_ms = 3.0f;             // Perde hareket sÃ¼resi
    
    // === Rolling Shutter ===
    bool rolling_shutter = false;
    float rolling_shutter_skew = 0.0f;               // EÄŸrilik miktarÄ±
    
    // === Hesaplamalar ===
    // Cinema: Shutter angle'dan exposure sÃ¼resini hesapla
    float getExposureFromAngle(float fps) const {
        return (shutter_angle_degrees / 360.0f) / fps;
    }
    
    // Motion blur iÃ§in temporal sample aralÄ±ÄŸÄ±
    float getSampleTimeStep() const {
        return shutter_speed_seconds / (float)motion_blur_samples;
    }
    
    // Motion blur yoÄŸunluÄŸu (normalize)
    float getMotionBlurFactor() const {
        // 1/1000s = minimal blur, 1/15s = heavy blur
        return std::clamp(shutter_speed_seconds * 60.0f, 0.0f, 1.0f);
    }
};

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// PHYSICAL EXPOSURE (Exposure ÃœÃ§geni)
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
struct PhysicalExposure {
    // === ISO ===
    int iso = 100;
    int native_iso = 100;
    bool dual_native_iso = false;           // Dual native ISO desteÄŸi
    int second_native_iso = 800;            // Ä°kinci native ISO
    
    // === Auto Exposure ===
    bool auto_exposure = false;
    float target_exposure = 0.18f;          // 18% gray target
    float ev_compensation = 0.0f;           // EV kompanzasyonu (-3 to +3)
    
    // === Metering ===
    enum class MeteringMode {
        Matrix,         // TÃ¼m kare
        CenterWeighted, // Merkez aÄŸÄ±rlÄ±klÄ±
        Spot,           // Nokta Ã¶lÃ§Ã¼m
        Highlight       // Highlight koruma
    };
    MeteringMode metering = MeteringMode::Matrix;
    
    // === Hesaplamalar ===
    // ISO amplifikasyon faktÃ¶rÃ¼ (noise'u da Ã§arpar!)
    float getISOGain() const {
        return (float)iso / (float)native_iso;
    }
    
    // Dual ISO modunda hangi native'e yakÄ±nsak o kullanÄ±lÄ±r
    int getEffectiveNativeISO() const {
        if (!dual_native_iso) return native_iso;
        // Ä°kisinin ortasÄ±na yakÄ±nsa hangisi daha yakÄ±nsa
        int mid = (native_iso + second_native_iso) / 2;
        return (iso < mid) ? native_iso : second_native_iso;
    }
    
    // Exposure Value (EV) hesaplama
    // EV = log2(NÂ² / t) - log2(ISO/100)
    float calculateEV(float aperture, float shutter_seconds) const {
        float ev = log2f((aperture * aperture) / shutter_seconds);
        ev -= log2f((float)iso / 100.0f);
        return ev + ev_compensation;
    }
    
    // Sahne parlaklÄ±ÄŸÄ± iÃ§in gerekli exposure multiplier
    // Bu deÄŸer render sonucunu Ã‡ARPAR (sinyali + varyansÄ± birlikte)
    float getExposureMultiplier(float aperture, float shutter_seconds) const {
        // Fiziksel exposure = IÅŸÄ±k toplama kapasitesi
        // Aperture: IÅŸÄ±k miktarÄ± âˆ 1/NÂ²
        // Shutter: IÅŸÄ±k miktarÄ± âˆ sÃ¼re
        // ISO: Amplifikasyon
        
        float aperture_factor = 1.0f / (aperture * aperture);          // f/1.4 = 0.51, f/8 = 0.016
        float shutter_factor = shutter_seconds / (1.0f / 250.0f);       // 1/250s referans
        float iso_factor = getISOGain();
        
        // Toplam exposure (referans ayarlÄ±)
        const float reference_scale = 2.5f; // Sahne parlaklÄ±ÄŸÄ±na gÃ¶re ayarlanabilir
        return aperture_factor * shutter_factor * iso_factor * reference_scale;
    }
};

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// CAMERA BODY PHYSICS (Fiziksel KÃ¼tle ve Hareket)
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
struct CameraBodyPhysics {
    // === Fiziksel KÃ¼tle ===
    float body_mass_kg = 0.8f;              // GÃ¶vde kÃ¼tlesi
    float lens_mass_kg = 0.5f;              // Lens kÃ¼tlesi
    float total_mass_kg() const { return body_mass_kg + lens_mass_kg; }
    
    Vec3 center_of_gravity = Vec3(0, 0, 0.05f); // AÄŸÄ±rlÄ±k merkezi (lensten Ã¶tÃ¼rÃ¼ Ã¶nde)
    
    // === Atalet Momenti (dÃ¶ndÃ¼rme direnci) ===
    float inertia_x = 0.01f;                // Pitch (yukarÄ±-aÅŸaÄŸÄ±) atalet
    float inertia_y = 0.01f;                // Yaw (saÄŸ-sol) atalet
    float inertia_z = 0.005f;               // Roll ataleti
    
    // === Hareket Durumu (SimÃ¼lasyon tarafÄ±ndan gÃ¼ncellenir) ===
    Vec3 linear_velocity = Vec3(0);         // DoÄŸrusal hÄ±z (m/s)
    Vec3 angular_velocity = Vec3(0);        // AÃ§Ä±sal hÄ±z (rad/s)
    Vec3 linear_acceleration = Vec3(0);     // DoÄŸrusal ivme
    Vec3 angular_acceleration = Vec3(0);    // AÃ§Ä±sal ivme
    
    // === SÃ¶nÃ¼mleme (Damping) ===
    float linear_damping = 5.0f;            // DoÄŸrusal sÃ¶nÃ¼mleme
    float angular_damping = 10.0f;          // AÃ§Ä±sal sÃ¶nÃ¼mleme
    
    // === Stabilizasyon (IBIS/OIS) ===
    bool ibis_enabled = false;              // In-Body Image Stabilization
    float ibis_effectiveness_stops = 5.0f;  // Stabilizasyon etkinliÄŸi
    float ibis_response_rate = 0.1f;        // Tepki hÄ±zÄ± (dÃ¼ÅŸÃ¼k = yumuÅŸak)
    
    // === Fizik GÃ¼ncellemesi ===
    void update(float dt, const Vec3& applied_force, const Vec3& applied_torque) {
        // F = ma -> a = F/m
        linear_acceleration = applied_force / total_mass_kg();
        
        // Ï„ = IÎ± -> Î± = Ï„/I (her eksen iÃ§in ayrÄ±)
        angular_acceleration.x = applied_torque.x / inertia_x;
        angular_acceleration.y = applied_torque.y / inertia_y;
        angular_acceleration.z = applied_torque.z / inertia_z;
        
        // HÄ±z gÃ¼ncelle (yarÄ±-implicit Euler)
        linear_velocity = linear_velocity + linear_acceleration * dt;
        angular_velocity = angular_velocity + angular_acceleration * dt;
        
        // SÃ¶nÃ¼mleme uygula
        linear_velocity = linear_velocity * expf(-linear_damping * dt);
        angular_velocity = angular_velocity * expf(-angular_damping * dt);
        
        // IBIS kompanzasyonu
        if (ibis_enabled) {
            float compensation = 1.0f / powf(2.0f, ibis_effectiveness_stops);
            // Angular velocity'yi yavaÅŸÃ§a sÄ±fÄ±ra Ã§ek
            angular_velocity = angular_velocity * (1.0f - ibis_response_rate);
        }
    }
    
    // Pozisyon ve rotasyon deÄŸiÅŸimi al
    Vec3 getPositionDelta(float dt) const {
        return linear_velocity * dt;
    }
    
    Vec3 getRotationDelta(float dt) const {
        return angular_velocity * dt; // radyan
    }
};

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// HANDHELD SIMULATION (El SarsÄ±ntÄ±sÄ± FiziÄŸi)
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
struct HandheldSimulation {
    bool enabled = false;
    
    // === SarsÄ±ntÄ± Parametreleri ===
    float shake_intensity = 1.0f;           // Genel yoÄŸunluk Ã§arpanÄ±
    
    // El titremesi (yÃ¼ksek frekans)
    float hand_tremor_frequency = 8.0f;     // Hz (doÄŸal el titremesi 8-12 Hz)
    float hand_tremor_amplitude = 0.001f;   // metre
    
    // VÃ¼cut sallanmasÄ± (dÃ¼ÅŸÃ¼k frekans)
    float body_sway_frequency = 0.5f;       // Hz
    float body_sway_amplitude = 0.005f;     // metre
    
    // Nefes alma
    float breathing_frequency = 0.25f;      // Hz (~15 nefes/dakika)
    float breathing_amplitude = 0.003f;     // metre (dikey hareket)
    
    // Kalp atÄ±ÅŸÄ±
    float heartbeat_frequency = 1.2f;       // Hz (~72 bpm)
    float heartbeat_amplitude = 0.0005f;    // metre
    
    // === OperatÃ¶r Becerisi ===
    enum class OperatorSkill {
        Amateur,        // AmatÃ¶r - maksimum sarsÄ±ntÄ±
        Intermediate,   // Orta - %60 sarsÄ±ntÄ±
        Professional,   // Profesyonel - %30 sarsÄ±ntÄ±
        Expert          // Uzman (sniper eÄŸitimi) - %10 sarsÄ±ntÄ±
    };
    OperatorSkill skill = OperatorSkill::Professional;
    
    float getSkillMultiplier() const {
        switch (skill) {
            case OperatorSkill::Amateur: return 1.0f;
            case OperatorSkill::Intermediate: return 0.6f;
            case OperatorSkill::Professional: return 0.3f;
            case OperatorSkill::Expert: return 0.1f;
            default: return 1.0f;
        }
    }
    
    // === DuruÅŸ Pozisyonu ===
    enum class Stance {
        Standing,       // Ayakta - maksimum sarsÄ±ntÄ±
        Kneeling,       // Diz Ã§Ã¶kmÃ¼ÅŸ - %60
        Sitting,        // Oturarak - %40
        Prone,          // Yere yatmÄ±ÅŸ - %20
        Braced          // Destekli (duvar/tripod) - %10
    };
    Stance stance = Stance::Standing;
    
    float getStanceMultiplier() const {
        switch (stance) {
            case Stance::Standing: return 1.0f;
            case Stance::Kneeling: return 0.6f;
            case Stance::Sitting: return 0.4f;
            case Stance::Prone: return 0.2f;
            case Stance::Braced: return 0.1f;
            default: return 1.0f;
        }
    }
    
    // === Perlin Noise tabanlÄ± sarsÄ±ntÄ± hesaplama ===
    // Not: noise3D fonksiyonu harici olarak saÄŸlanmalÄ±dÄ±r
    Vec3 calculateShake(float time) const {
        if (!enabled) return Vec3(0);
        
        float skill_mult = getSkillMultiplier();
        float stance_mult = getStanceMultiplier();
        float total_mult = shake_intensity * skill_mult * stance_mult;
        
        Vec3 shake(0);
        
        // El titremesi (yÃ¼ksek frekans, dÃ¼ÅŸÃ¼k amplitÃ¼d)
        float tremor_phase = time * hand_tremor_frequency;
        shake.x += sinf(tremor_phase * 1.0f) * hand_tremor_amplitude;
        shake.y += sinf(tremor_phase * 1.3f + 1.5f) * hand_tremor_amplitude;
        shake.z += sinf(tremor_phase * 0.7f + 3.0f) * hand_tremor_amplitude * 0.3f;
        
        // VÃ¼cut sallanmasÄ± (dÃ¼ÅŸÃ¼k frekans, yÃ¼ksek amplitÃ¼d)
        float sway_phase = time * body_sway_frequency;
        shake.x += sinf(sway_phase * 1.0f) * body_sway_amplitude;
        shake.y += sinf(sway_phase * 0.7f + 2.0f) * body_sway_amplitude * 0.5f;
        
        // Nefes alma (dikey hareket)
        float breath_phase = time * breathing_frequency * 2.0f * (float)M_PI;
        shake.y += sinf(breath_phase) * breathing_amplitude;
        
        // Kalp atÄ±ÅŸÄ± (hÄ±zlÄ±, kÃ¼Ã§Ã¼k)
        float heart_phase = time * heartbeat_frequency * 2.0f * (float)M_PI;
        shake += Vec3(sinf(heart_phase), cosf(heart_phase), 0) * heartbeat_amplitude;
        
        return shake * total_mult;
    }
    
    // AÃ§Ä±sal sarsÄ±ntÄ± (rotasyon)
    Vec3 calculateAngularShake(float time) const {
        if (!enabled) return Vec3(0);
        
        float total_mult = shake_intensity * getSkillMultiplier() * getStanceMultiplier();
        
        // AÃ§Ä±sal sarsÄ±ntÄ± (radyan cinsinden)
        float angular_scale = 0.005f; // ~0.3 derece maksimum
        
        float phase = time * hand_tremor_frequency;
        Vec3 angular_shake;
        angular_shake.x = sinf(phase * 1.1f) * angular_scale;       // Pitch
        angular_shake.y = sinf(phase * 0.9f + 1.0f) * angular_scale; // Yaw
        angular_shake.z = sinf(phase * 0.5f + 2.0f) * angular_scale * 0.3f; // Roll (daha az)
        
        return angular_shake * total_mult;
    }
};

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// CAMERA RIG SYSTEM (Sinematik Hareket Sistemleri)
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
struct CameraRig {
    enum class RigType {
        Static,         // Sabit tripod
        Handheld,       // El ile (shake simÃ¼lasyonu)
        Dolly,          // Ray Ã¼zerinde lineer hareket
        Crane,          // Jib kolu
        Orbit,          // Hedef etrafÄ±nda yÃ¶rÃ¼nge
        Steadicam,      // Stabilize hareket
        Drone,          // Hava Ã§ekimi (6 DoF)
        Cablecam,       // Kablo sistemi
        Gimbal          // Motorlu gimbal
    };
    RigType type = RigType::Static;
    
    // === Dolly Parametreleri ===
    struct DollyParams {
        Vec3 track_start = Vec3(0);
        Vec3 track_end = Vec3(5, 0, 0);
        float position = 0.0f;              // 0-1 arasÄ± track pozisyonu
        float speed = 1.0f;                 // m/s
        bool smooth_start_stop = true;      // YumuÅŸak baÅŸla/dur
        float acceleration = 2.0f;          // m/sÂ²
    } dolly;
    
    // === Crane/Jib Parametreleri ===
    struct CraneParams {
        Vec3 pivot_point = Vec3(0);
        float arm_length = 3.0f;            // Kol uzunluÄŸu (m)
        float arm_angle = 0.0f;             // Yatay aÃ§Ä± (derece)
        float boom_angle = 0.0f;            // Dikey aÃ§Ä± (derece)
        float max_boom_up = 45.0f;
        float max_boom_down = 30.0f;
        float rotation_speed = 30.0f;       // derece/s
    } crane;
    
    // === Orbit Parametreleri ===
    struct OrbitParams {
        Vec3 target = Vec3(0);              // YÃ¶rÃ¼nge merkezi
        float radius = 5.0f;                // YÃ¶rÃ¼nge yarÄ±Ã§apÄ±
        float angle = 0.0f;                 // Mevcut aÃ§Ä± (derece)
        float height_offset = 0.0f;         // Dikey offset
        float angular_speed = 30.0f;        // derece/s
        bool look_at_target = true;         // Hedefe bak
    } orbit;
    
    // === Steadicam Parametreleri ===
    struct SteadicamParams {
        float smoothing_position = 0.9f;    // Pozisyon yumuÅŸatma (0-1)
        float smoothing_rotation = 0.95f;   // Rotasyon yumuÅŸatma (0-1)
        Vec3 filtered_position = Vec3(0);
        Vec3 filtered_rotation = Vec3(0);
        float boom_height = 1.7f;           // OperatÃ¶r gÃ¶ÄŸÃ¼s yÃ¼ksekliÄŸi
        
        // YumuÅŸatÄ±lmÄ±ÅŸ pozisyon hesapla
        Vec3 smooth(const Vec3& target_pos, float dt) {
            float alpha = 1.0f - powf(1.0f - smoothing_position, dt * 60.0f);
            filtered_position = filtered_position + (target_pos - filtered_position) * alpha;
            return filtered_position;
        }
    } steadicam;
    
    // === Drone Parametreleri ===
    struct DroneParams {
        float max_speed = 15.0f;            // m/s
        float max_acceleration = 5.0f;      // m/sÂ²
        float altitude = 10.0f;             // Mevcut irtifa
        float gimbal_pitch = 0.0f;          // Gimbal aÃ§Ä±sÄ±
        float gimbal_smoothing = 0.9f;      // Gimbal stabilizasyonu
        bool gps_stabilized = true;         // GPS pozisyon kilidi
    } drone;
    
    // === Gimbal Parametreleri ===
    struct GimbalParams {
        bool enabled = false;
        int axes = 3;                       // 2-axis veya 3-axis
        float max_pitch = 90.0f;
        float max_roll = 45.0f;
        float max_yaw = 180.0f;
        float motor_strength = 0.9f;        // Motor gÃ¼cÃ¼ (stabilizasyon)
        float response_speed = 0.1f;        // Tepki hÄ±zÄ±
    } gimbal;
    
    // === Rig pozisyonunu hesapla ===
    Vec3 calculatePosition(float time) const {
        switch (type) {
            case RigType::Dolly: {
                float t = dolly.position;
                if (dolly.smooth_start_stop) {
                    // Smoothstep
                    t = t * t * (3.0f - 2.0f * t);
                }
                return dolly.track_start + (dolly.track_end - dolly.track_start) * t;
            }
            
            case RigType::Crane: {
                float arm_rad = crane.arm_angle * (float)M_PI / 180.0f;
                float boom_rad = crane.boom_angle * (float)M_PI / 180.0f;
                
                float x = crane.arm_length * cosf(boom_rad) * cosf(arm_rad);
                float y = crane.arm_length * sinf(boom_rad);
                float z = crane.arm_length * cosf(boom_rad) * sinf(arm_rad);
                
                return crane.pivot_point + Vec3(x, y, z);
            }
            
            case RigType::Orbit: {
                float angle_rad = orbit.angle * (float)M_PI / 180.0f;
                float x = orbit.radius * cosf(angle_rad);
                float z = orbit.radius * sinf(angle_rad);
                return orbit.target + Vec3(x, orbit.height_offset, z);
            }
            
            default:
                return Vec3(0);
        }
    }
};

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// COMPLETE PHYSICAL CAMERA
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
struct PhysicalCamera {
    // === Mod ===
    CameraMode mode = CameraMode::Pro;
    
    // === Ana BileÅŸenler ===
    PhysicalSensor sensor;
    PhysicalLens lens;
    PhysicalShutter shutter;
    PhysicalExposure exposure;
    CameraBodyPhysics body_physics;
    HandheldSimulation handheld;
    CameraRig rig;
    
    // === Pozisyon ve Oryantasyon ===
    Vec3 position = Vec3(0, 1.7f, 5);       // DÃ¼nya koordinatlarÄ±
    Vec3 rotation = Vec3(0);                 // Euler aÃ§Ä±larÄ± (pitch, yaw, roll)
    Vec3 target = Vec3(0);                   // BaktÄ±ÄŸÄ± nokta
    bool use_look_at = true;                 // Target'a bak modu
    
    // === Odaklama ===
    float focus_distance = 5.0f;             // Odak mesafesi (m)
    bool auto_focus = false;
    enum class AFMode { Single, Continuous, Manual };
    AFMode af_mode = AFMode::Manual;
    
    // === Zaman ===
    float simulation_time = 0.0f;
    
    // === Mod bazlÄ± Ã¶zellik kontrolÃ¼ ===
    bool isFeatureEnabled(const char* feature) const {
        if (mode == CameraMode::Auto) {
            // Auto modda minimal Ã¶zellik
            return false;
        } else if (mode == CameraMode::Pro) {
            // Pro modda optik kusurlar opsiyonel
            return false; // VarsayÄ±lan kapalÄ±
        } else { // Cinema
            // Cinema modda her ÅŸey aÃ§Ä±k
            return true;
        }
    }
    
    // === Fizik gÃ¼ncellemesi ===
    void update(float dt) {
        simulation_time += dt;
        
        // Body physics gÃ¼ncelle
        Vec3 force(0), torque(0);
        body_physics.update(dt, force, torque);
        
        // Handheld shake uygula
        if (handheld.enabled) {
            Vec3 shake_offset = handheld.calculateShake(simulation_time);
            Vec3 shake_rotation = handheld.calculateAngularShake(simulation_time);
            
            // Pozisyon ve rotasyona ekle (geÃ§ici olarak, render sÄ±rasÄ±nda)
            // Bu deÄŸerler ana pozisyonu DEÄÄ°ÅTÄ°RMEZ, sadece render'da eklenir
        }
        
        // Rig pozisyonunu gÃ¼ncelle
        if (rig.type != CameraRig::RigType::Static && 
            rig.type != CameraRig::RigType::Handheld) {
            position = rig.calculatePosition(simulation_time);
        }
    }
    
    // === Render iÃ§in efektif deÄŸerleri al ===
    Vec3 getEffectivePosition() const {
        Vec3 pos = position;
        
        // Rig pozisyonu
        if (rig.type != CameraRig::RigType::Static && 
            rig.type != CameraRig::RigType::Handheld) {
            pos = rig.calculatePosition(simulation_time);
        }
        
        // Handheld shake
        if (handheld.enabled) {
            pos = pos + handheld.calculateShake(simulation_time);
        }
        
        // Body physics offset
        pos = pos + body_physics.getPositionDelta(1.0f / 60.0f);
        
        return pos;
    }
    
    Vec3 getEffectiveRotation() const {
        Vec3 rot = rotation;
        
        if (handheld.enabled) {
            rot = rot + handheld.calculateAngularShake(simulation_time);
        }
        
        rot = rot + body_physics.getRotationDelta(1.0f / 60.0f);
        
        return rot;
    }
    
    // === FOV hesaplama ===
    float getVerticalFOV() const {
        float base_fov = lens.getVerticalFOV(sensor.height_mm);
        
        // Focus breathing etkisi
        if (lens.enable_breathing && mode == CameraMode::Cinema) {
            float focus_normalized = focus_distance / 10.0f; // 10m referans
            base_fov *= (1.0f + lens.breathing_amount * (1.0f - focus_normalized));
        }
        
        return base_fov;
    }
    
    float getHorizontalFOV() const {
        return lens.getHorizontalFOV(sensor.width_mm);
    }
    
    // === Exposure Ã§arpanÄ± (render sonucu bu ile Ã§arpÄ±lÄ±r) ===
    float getExposureMultiplier() const {
        return exposure.getExposureMultiplier(
            lens.current_aperture, 
            shutter.shutter_speed_seconds
        );
    }
    
    // === Recommended sample sayÄ±sÄ± (ISO'ya gÃ¶re) ===
    int getRecommendedSamples(int base_samples = 64) const {
        // YÃ¼ksek ISO = daha fazla sample gerekli
        float iso_factor = log2f((float)exposure.iso / 100.0f);
        return base_samples * (int)powf(2.0f, std::max(0.0f, iso_factor * 0.5f));
    }
    
    // === Preset uygulama ===
    void applyAutoMode() {
        mode = CameraMode::Auto;
        exposure.auto_exposure = true;
        exposure.iso = 400;
        lens.current_aperture = 5.6f;
        shutter.shutter_speed_seconds = 1.0f / 125.0f;
        shutter.enable_motion_blur = false;
        handheld.enabled = false;
        af_mode = AFMode::Continuous;
    }
    
    void applyProMode() {
        mode = CameraMode::Pro;
        exposure.auto_exposure = false;
        shutter.enable_motion_blur = false;
        handheld.enabled = false;
        lens.natural_vignette = 0.0f;
        lens.lateral_ca = 0.0f;
    }
    
    void applyCinemaMode() {
        mode = CameraMode::Cinema;
        exposure.auto_exposure = false;
        shutter.enable_motion_blur = true;
        shutter.shutter_angle_degrees = 180.0f;
        lens.use_tstop = true;
        lens.enable_breathing = true;
        lens.calculateAutoDistortion();
        lens.calculateAutoVignette();
    }
};

} // namespace PhysicalCameraSystem

