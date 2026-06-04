/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          CameraPresets.h
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// CAMERA PRESETS - Professional Camera Body, Lens, and Aspect Ratio Data
// For RayTrophi Renderer - Modular camera simulation system
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

#include <cstddef>

namespace CameraPresets {

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// SENSOR TYPES
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
enum class SensorType {
    FullFrame,        // 36x24mm
    APSC_Canon,       // 22.3x14.9mm (1.6x crop)
    APSC_Sony,        // 23.6x15.6mm (1.5x crop)
    Super35,          // 24.9x18.7mm (1.39x crop) - Cinema
    MicroFourThirds,  // 17.3x13mm (2.0x crop)
    MediumFormat      // 43.8x32.9mm (0.79x crop)
};

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// CAMERA BODY PRESETS
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
struct CameraBody {
    const char* name;
    const char* brand;
    SensorType sensor;
    float sensor_width_mm;  // Physical width
    float sensor_height_mm; // Physical height
    float crop_factor;
    int min_iso;            // Native Min ISO
    int max_iso;            // Native Max ISO
    int resolution_mp;      // Megapixels
    const char* description;
};

inline constexpr CameraBody CAMERA_BODIES[] = {
    // Custom (manual settings)
    {"Custom", "Manual", SensorType::FullFrame, 36.0f, 24.0f, 1.0f, 50, 409600, 0, "Manual crop factor"},
    
    // Full Frame (36x24mm)
    {"Generic Full Frame", "Generic", SensorType::FullFrame, 36.0f, 24.0f, 1.0f, 100, 25600, 24, "Standard 35mm equivalent"},
    {"Canon EOS R5", "Canon", SensorType::FullFrame, 36.0f, 24.0f, 1.0f, 100, 51200, 45, "High-res mirrorless"},
    {"Canon EOS R6 II", "Canon", SensorType::FullFrame, 36.0f, 24.0f, 1.0f, 100, 102400, 24, "Versatile mirrorless"},
    {"Sony A7 IV", "Sony", SensorType::FullFrame, 35.9f, 23.9f, 1.0f, 100, 51200, 33, "Hybrid mirrorless"},
    {"Sony A7R V", "Sony", SensorType::FullFrame, 35.7f, 23.8f, 1.0f, 100, 32000, 61, "High-res mirrorless"},
    {"Nikon Z8", "Nikon", SensorType::FullFrame, 35.9f, 23.9f, 1.0f, 64, 25600, 45, "Pro mirrorless"},
    {"Pentax K-1 Mark II", "Pentax", SensorType::FullFrame, 35.9f, 24.0f, 1.0f, 100, 819200, 36, "Full frame DSLR legend"},
    
    // APS-C (Canon: 22.3x14.9, Sony/Others: 23.5x15.6)
    {"Canon EOS R7", "Canon", SensorType::APSC_Canon, 22.3f, 14.9f, 1.6f, 100, 32000, 32, "APS-C mirrorless"},
    {"Canon EOS 90D", "Canon", SensorType::APSC_Canon, 22.3f, 14.9f, 1.6f, 100, 25600, 32, "APS-C DSLR"},
    {"Sony a6700", "Sony", SensorType::APSC_Sony, 23.3f, 15.5f, 1.5f, 100, 32000, 26, "APS-C mirrorless"},
    {"Fujifilm X-T5", "Fujifilm", SensorType::APSC_Sony, 23.5f, 15.6f, 1.5f, 125, 12800, 40, "X-Trans sensor"},
    {"Pentax K-3 III", "Pentax", SensorType::APSC_Sony, 23.3f, 15.5f, 1.5f, 100, 1600000, 26, "APS-C DSLR flagship"}, // Pentax high ISO logic
    {"Pentax KF", "Pentax", SensorType::APSC_Sony, 23.5f, 15.6f, 1.5f, 100, 102400, 24, "Weather-sealed APS-C"},
    
    // Cinema
    {"RED Komodo 6K", "RED", SensorType::Super35, 27.03f, 14.25f, 1.33f, 250, 12800, 19, "Cinema 6K S35"},
    {"RED V-Raptor", "RED", SensorType::FullFrame, 40.96f, 21.60f, 0.88f, 250, 12800, 35, "Cinema 8K VV"}, // Wider than FF
    {"ARRI Alexa Mini LF", "ARRI", SensorType::FullFrame, 36.70f, 25.54f, 0.98f, 160, 3200, 9, "Large format cinema"},
    {"ARRI Alexa Mini", "ARRI", SensorType::Super35, 28.17f, 18.13f, 1.27f, 160, 3200, 3, "Super 35 cinema"}, // 4:3 mode sensor
    {"Blackmagic Pocket 6K", "BMD", SensorType::Super35, 23.10f, 12.99f, 1.56f, 100, 25600, 6, "Affordable cinema"},
    
    // Micro 4/3 (17.3 x 13.0)
    {"Panasonic GH6", "Panasonic", SensorType::MicroFourThirds, 17.3f, 13.0f, 2.0f, 100, 25600, 25, "Video-focused M43"},
    {"OM System OM-1", "OM", SensorType::MicroFourThirds, 17.4f, 13.0f, 2.0f, 200, 102400, 20, "Compact M43"},
    
    // Medium Format (GFX: 43.8x32.9)
    {"Hasselblad X2D", "Hasselblad", SensorType::MediumFormat, 43.8f, 32.9f, 0.79f, 64, 25600, 100, "100MP medium format"},
    {"Fujifilm GFX 100S", "Fujifilm", SensorType::MediumFormat, 43.8f, 32.9f, 0.79f, 100, 102400, 102, "Medium format mirrorless"},
    {"Pentax 645Z", "Pentax", SensorType::MediumFormat, 43.8f, 32.8f, 0.79f, 100, 204800, 51, "Medium format DSLR"},
};

inline constexpr size_t CAMERA_BODY_COUNT = sizeof(CAMERA_BODIES) / sizeof(CameraBody);

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// LENS CATEGORY
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
enum class LensCategory {
    UltraWide,    // <24mm
    Wide,         // 24-35mm
    Standard,     // 35-60mm
    Portrait,     // 70-135mm
    Telephoto,    // 135mm+
    Cinema        // Cine primes
};

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// PROFESSIONAL LENS PRESETS
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
struct LensPreset {
    const char* name;
    const char* brand;
    LensCategory category;
    float focal_mm;
    bool is_zoom;         // True if zoom lens
    float min_mm;         // Min focal length
    float max_mm;         // Max focal length
    float fov_deg;        // Vertical FOV on full frame (at focal_mm)
    float max_aperture;   // f-number (e.g., 1.4, 2.8) - WIDEST
    float min_aperture;   // f-number (e.g., 16, 22, 32) - TIGHTEST
    int blade_count;      // Aperture blades
    const char* description;
};

inline constexpr LensPreset LENS_PRESETS[] = {
    // Custom (manual)
    {"Custom", "Manual", LensCategory::Standard, 50.0f, true, 1.0f, 2000.0f, 46.8f, 0.0f, 128.0f, 0, "Manual FOV control"},
    
    // Ultra Wide
    {"Canon RF 14-35mm f/4L", "Canon", LensCategory::UltraWide, 14.0f, true, 14.0f, 35.0f, 104.0f, 4.0f, 22.0f, 9, "Ultra wide zoom"},
    {"Sony FE 12-24mm f/2.8 GM", "Sony", LensCategory::UltraWide, 12.0f, true, 12.0f, 24.0f, 111.0f, 2.8f, 22.0f, 9, "Fast ultra wide data"},
    {"Pentax HD DA 15mm f/4 Ltd", "Pentax", LensCategory::UltraWide, 15.0f, false, 15.0f, 15.0f, 100.0f, 4.0f, 22.0f, 7, "Pancake ultra wide"},
    
    // Wide
    {"Canon RF 24-70mm f/2.8L", "Canon", LensCategory::Wide, 24.0f, true, 24.0f, 70.0f, 84.0f, 2.8f, 22.0f, 9, "Standard zoom"},
    {"Sony FE 24-70mm f/2.8 GM II", "Sony", LensCategory::Wide, 24.0f, true, 24.0f, 70.0f, 84.0f, 2.8f, 22.0f, 11, "Versatile zoom"},
    {"Sigma 24mm f/1.4 DG DN", "Sigma", LensCategory::Wide, 24.0f, false, 24.0f, 24.0f, 84.0f, 1.4f, 16.0f, 11, "Fast wide prime"},
    {"Pentax FA 31mm f/1.8 Ltd", "Pentax", LensCategory::Wide, 31.0f, false, 31.0f, 31.0f, 72.0f, 1.8f, 22.0f, 9, "The legendary Limited"},
    {"Pentax HD FA 21mm f/2.4 Ltd", "Pentax", LensCategory::Wide, 21.0f, false, 21.0f, 21.0f, 91.0f, 2.4f, 22.0f, 9, "Ultra wide Limited"},
    
    // Standard
    {"Canon RF 50mm f/1.2L", "Canon", LensCategory::Standard, 50.0f, false, 50.0f, 50.0f, 46.8f, 1.2f, 16.0f, 10, "Ultimate 50mm"},
    {"Sony FE 50mm f/1.2 GM", "Sony", LensCategory::Standard, 50.0f, false, 50.0f, 50.0f, 46.8f, 1.2f, 16.0f, 11, "Fast 50 prime"},
    {"Sigma 35mm f/1.4 DG DN Art", "Sigma", LensCategory::Standard, 35.0f, false, 35.0f, 35.0f, 63.0f, 1.4f, 16.0f, 11, "35mm Art"},
    {"Zeiss Otus 55mm f/1.4", "Zeiss", LensCategory::Standard, 55.0f, false, 55.0f, 55.0f, 43.0f, 1.4f, 16.0f, 9, "Reference lens"},
    {"Pentax FA 43mm f/1.9 Ltd", "Pentax", LensCategory::Standard, 43.0f, false, 43.0f, 43.0f, 54.0f, 1.9f, 22.0f, 9, "Perfect normal - pancake"},
    {"Pentax HD FA 35mm f/2", "Pentax", LensCategory::Standard, 35.0f, false, 35.0f, 35.0f, 63.0f, 2.0f, 22.0f, 9, "Classic 35mm"},
    
    // Portrait
    {"Canon RF 85mm f/1.2L", "Canon", LensCategory::Portrait, 85.0f, false, 85.0f, 85.0f, 28.6f, 1.2f, 16.0f, 10, "Bokeh master"},
    {"Sony FE 85mm f/1.4 GM", "Sony", LensCategory::Portrait, 85.0f, false, 85.0f, 85.0f, 28.6f, 1.4f, 16.0f, 11, "Portrait king"},
    {"Sigma 105mm f/1.4 DG HSM Art", "Sigma", LensCategory::Portrait, 105.0f, false, 105.0f, 105.0f, 23.3f, 1.4f, 16.0f, 9, "Bokeh monster"},
    {"Pentax FA 77mm f/1.8 Ltd", "Pentax", LensCategory::Portrait, 77.0f, false, 77.0f, 77.0f, 31.5f, 1.8f, 22.0f, 9, "Portrait legend - creamy bokeh"},
    {"Pentax DFA* 85mm f/1.4", "Pentax", LensCategory::Portrait, 85.0f, false, 85.0f, 85.0f, 28.6f, 1.4f, 16.0f, 9, "Star series flagship"},
    
    // Telephoto
    {"Canon RF 70-200mm f/2.8L", "Canon", LensCategory::Telephoto, 70.0f, true, 70.0f, 200.0f, 34.0f, 2.8f, 32.0f, 9, "Pro tele zoom"},
    {"Sony FE 70-200mm f/2.8 GM II", "Sony", LensCategory::Telephoto, 70.0f, true, 70.0f, 200.0f, 34.0f, 2.8f, 22.0f, 11, "Compact tele"},
    {"Sony FE 135mm f/1.8 GM", "Sony", LensCategory::Telephoto, 135.0f, false, 135.0f, 135.0f, 18.0f, 1.8f, 22.0f, 11, "Fast tele prime"},
    {"Canon RF 100-500mm f/4.5-7.1L", "Canon", LensCategory::Telephoto, 100.0f, true, 100.0f, 500.0f, 24.0f, 5.6f, 32.0f, 9, "Wildlife zoom"},
    {"Pentax DFA* 70-200mm f/2.8", "Pentax", LensCategory::Telephoto, 70.0f, true, 70.0f, 200.0f, 34.0f, 2.8f, 22.0f, 9, "Pro tele zoom"},
    
    // Super Telephoto (Wildlife, Sports, Astro)
    {"Canon RF 400mm f/2.8L", "Canon", LensCategory::Telephoto, 400.0f, false, 400.0f, 400.0f, 6.2f, 2.8f, 32.0f, 9, "Super tele - sports/wildlife"},
    {"Sony FE 400mm f/2.8 GM", "Sony", LensCategory::Telephoto, 400.0f, false, 400.0f, 400.0f, 6.2f, 2.8f, 22.0f, 11, "Pro wildlife prime"},
    {"Nikon 500mm f/5.6E PF", "Nikon", LensCategory::Telephoto, 500.0f, false, 500.0f, 500.0f, 5.0f, 5.6f, 32.0f, 9, "Lightweight super tele"},
    {"Canon RF 600mm f/4L", "Canon", LensCategory::Telephoto, 600.0f, false, 600.0f, 600.0f, 4.1f, 4.0f, 32.0f, 9, "Flagship wildlife"},
    {"Sony FE 600mm f/4 GM", "Sony", LensCategory::Telephoto, 600.0f, false, 600.0f, 600.0f, 4.1f, 4.0f, 22.0f, 11, "Pro nature/sports"},
    {"Canon RF 800mm f/5.6L", "Canon", LensCategory::Telephoto, 800.0f, false, 800.0f, 800.0f, 3.1f, 5.6f, 32.0f, 9, "Extreme reach"},
    {"Nikon 800mm f/6.3 VR S", "Nikon", LensCategory::Telephoto, 800.0f, false, 800.0f, 800.0f, 3.1f, 6.3f, 32.0f, 9, "Z-mount super tele"},
    {"Canon RF 1200mm f/8L", "Canon", LensCategory::Telephoto, 1200.0f, false, 1200.0f, 1200.0f, 2.1f, 8.0f, 32.0f, 9, "Maximum reach - astro/wildlife"},
    
    // Cinema
    {"Cooke S4/i 50mm T2", "Cooke", LensCategory::Cinema, 50.0f, false, 50.0f, 50.0f, 46.8f, 2.0f, 22.0f, 8, "Cooke Look"},
    {"Cooke S4/i 32mm T2", "Cooke", LensCategory::Cinema, 32.0f, false, 32.0f, 32.0f, 65.0f, 2.0f, 22.0f, 8, "Wide cinema"},
    {"ARRI Master Prime 35mm", "ARRI", LensCategory::Cinema, 35.0f, false, 35.0f, 35.0f, 63.0f, 1.4f, 22.0f, 15, "Master quality"},
    {"ARRI Master Prime 50mm", "ARRI", LensCategory::Cinema, 50.0f, false, 50.0f, 50.0f, 46.8f, 1.4f, 22.0f, 15, "Reference cine"},
    {"Zeiss Supreme 85mm", "Zeiss", LensCategory::Cinema, 85.0f, false, 85.0f, 85.0f, 28.6f, 1.5f, 22.0f, 11, "Supreme bokeh"},
};
inline constexpr size_t LENS_PRESET_COUNT = sizeof(LENS_PRESETS) / sizeof(LensPreset);

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// ASPECT RATIO PRESETS
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
struct AspectRatioPreset {
    const char* name;
    float ratio;          // Width / Height
    const char* usage;
    bool is_cinema;       // Cinema format (letterbox)
};

inline constexpr AspectRatioPreset ASPECT_RATIOS[] = {
    {"Native", 0.0f, "Use render resolution", false},
    {"16:9 HD", 1.778f, "TV, YouTube, Streaming", false},
    {"4:3 Academy", 1.333f, "Classic TV, vintage", false},
    {"1:1 Square", 1.0f, "Instagram, social", false},
    {"3:2 Photo", 1.5f, "Photography standard", false},
    {"1.85:1 Flat", 1.85f, "Cinema flat/theatrical", true},
    {"2.00:1 Univisium", 2.0f, "Netflix, streaming films", true},
    {"2.35:1 Scope", 2.35f, "Anamorphic classic", true},
    {"2.39:1 Scope", 2.39f, "Modern cinemascope", true},
    {"2.76:1 Ultra Panavision", 2.76f, "Epic widescreen", true},
    {"9:16 Vertical", 0.5625f, "TikTok, Reels, Shorts", false},
};

inline constexpr size_t ASPECT_RATIO_COUNT = sizeof(ASPECT_RATIOS) / sizeof(AspectRatioPreset);

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// F-STOP PRESETS
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
struct FStopPreset {
    const char* name;
    float aperture_value;   // Internal aperture (for DOF simulation)
    float f_number;         // Real f-stop number
    const char* description;
};

inline constexpr FStopPreset FSTOP_PRESETS[] = {
    {"Custom", 0.0f, 0.0f, "Manual aperture"},
    {"f/1.2", 3.0f, 1.2f, "Maximum blur - ultra fast"},
    {"f/1.4", 2.5f, 1.4f, "Cinematic bokeh"},
    {"f/1.8", 2.0f, 1.8f, "Shallow DOF - portraits"},
    {"f/2.0", 1.6f, 2.0f, "Cinema standard"},
    {"f/2.8", 1.2f, 2.8f, "Pro zoom max aperture"},
    {"f/4.0", 0.7f, 4.0f, "General purpose"},
    {"f/5.6", 0.4f, 5.6f, "Landscape standard"},
    {"f/8.0", 0.2f, 8.0f, "Sweet spot sharpness"},
    {"f/11", 0.1f, 11.0f, "Deep DOF"},
    {"f/16", 0.05f, 16.0f, "Maximum sharpness"},
    {"f/22", 0.02f, 22.0f, "Diffraction limited"},
    {"f/32", 0.015f, 32.0f, "Deep focus - Macro"},
    {"f/45", 0.01f, 45.0f, "Large format standard"},
    {"f/64", 0.008f, 64.0f, "Group f/64"},
    {"f/90", 0.005f, 90.0f, "Pinhole like"},
    {"f/128", 0.004f, 128.0f, "Extreme depth"},
};

inline constexpr size_t FSTOP_PRESET_COUNT = sizeof(FSTOP_PRESETS) / sizeof(FStopPreset);

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// ISO PRESETS (Affects exposure brightness and noise simulation)
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
struct ISOPreset {
    const char* name;
    int iso_value;
    float exposure_multiplier;  // Relative to ISO 100
    float noise_factor;         // Noise intensity (0 = clean, 1 = very noisy)
    const char* description;
};

inline constexpr ISOPreset ISO_PRESETS[] = {
    {"ISO 50", 50, 0.5f, 0.0f, "Extended low - studio"},
    {"ISO 100", 100, 1.0f, 0.0f, "Native - cleanest"},
    {"ISO 200", 200, 2.0f, 0.02f, "Daylight standard"},
    {"ISO 400", 400, 4.0f, 0.05f, "Overcast/indoor"},
    {"ISO 800", 800, 8.0f, 0.1f, "Indoor standard"},
    {"ISO 1600", 1600, 16.0f, 0.2f, "Low light"},
    {"ISO 3200", 3200, 32.0f, 0.35f, "Night/concert"},
    {"ISO 6400", 6400, 64.0f, 0.5f, "Extreme low light"},
    {"ISO 12800", 12800, 128.0f, 0.7f, "Emergency only"},
    {"ISO 25600", 25600, 256.0f, 0.9f, "Very noisy"},
};

inline constexpr size_t ISO_PRESET_COUNT = sizeof(ISO_PRESETS) / sizeof(ISOPreset);

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// SHUTTER SPEED PRESETS (Affects motion blur amount)
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
struct ShutterSpeedPreset {
    const char* name;
    float speed_seconds;       // Exposure time in seconds
    float motion_blur_factor;  // 0 = frozen, 1 = max blur
    const char* description;
};

inline constexpr ShutterSpeedPreset SHUTTER_SPEED_PRESETS[] = {
    {"1/8000s", 0.000125f, 0.0f, "Sports freeze"},
    {"1/4000s", 0.00025f, 0.01f, "Action freeze"},
    {"1/2000s", 0.0005f, 0.02f, "Fast action"},
    {"1/1000s", 0.001f, 0.05f, "General action"},
    {"1/500s", 0.002f, 0.1f, "Walking subjects"},
    {"1/250s", 0.004f, 0.15f, "General purpose"},
    {"1/125s", 0.008f, 0.2f, "Portraits"},
    {"1/60s", 0.0167f, 0.3f, "Handheld limit"},
    {"1/30s", 0.0333f, 0.4f, "Slight motion blur"},
    {"1/15s", 0.0667f, 0.5f, "Motion blur visible"},
    {"1/8s", 0.125f, 0.6f, "Panning shots"},
    {"1/4s", 0.25f, 0.7f, "Waterfalls"},
    {"1/2s", 0.5f, 0.8f, "Light trails"},
    {"1s", 1.0f, 0.9f, "Long exposure"},
    {"2s", 2.0f, 0.95f, "Night scenes"},
    {"30s", 30.0f, 1.0f, "Star trails/astrophoto"},
};

inline constexpr size_t SHUTTER_SPEED_PRESET_COUNT = sizeof(SHUTTER_SPEED_PRESETS) / sizeof(ShutterSpeedPreset);

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// HELPER FUNCTIONS
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

// Get effective FOV after applying crop factor
inline float getEffectiveFOV(float base_fov, float crop_factor) {
    // FOV narrows with crop: atan(tan(fov/2) / crop) * 2
    // Simplified approximation: base_fov / crop_factor
    return base_fov / crop_factor;
}

// Get lens category name
inline const char* getLensCategoryName(LensCategory cat) {
    switch (cat) {
        case LensCategory::UltraWide: return "Ultra Wide";
        case LensCategory::Wide: return "Wide";
        case LensCategory::Standard: return "Standard";
        case LensCategory::Portrait: return "Portrait";
        case LensCategory::Telephoto: return "Telephoto";
        case LensCategory::Cinema: return "Cinema";
        default: return "Unknown";
    }
}

// Get sensor type name
inline const char* getSensorTypeName(SensorType sensor) {
    switch (sensor) {
        case SensorType::FullFrame: return "Full Frame";
        case SensorType::APSC_Canon: return "APS-C (Canon)";
        case SensorType::APSC_Sony: return "APS-C";
        case SensorType::Super35: return "Super 35";
        case SensorType::MicroFourThirds: return "Micro 4/3";
        case SensorType::MediumFormat: return "Medium Format";
        default: return "Unknown";
    }
}

} // namespace CameraPresets

