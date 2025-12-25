#pragma once

// ═══════════════════════════════════════════════════════════════════════════════════
// CAMERA PRESETS - Professional Camera Body, Lens, and Aspect Ratio Data
// For RayTrophi Renderer - Modular camera simulation system
// ═══════════════════════════════════════════════════════════════════════════════════

#include <cstddef>

namespace CameraPresets {

// ═══════════════════════════════════════════════════════════════════════════════════
// SENSOR TYPES
// ═══════════════════════════════════════════════════════════════════════════════════
enum class SensorType {
    FullFrame,        // 36x24mm
    APSC_Canon,       // 22.3x14.9mm (1.6x crop)
    APSC_Sony,        // 23.6x15.6mm (1.5x crop)
    Super35,          // 24.9x18.7mm (1.39x crop) - Cinema
    MicroFourThirds,  // 17.3x13mm (2.0x crop)
    MediumFormat      // 43.8x32.9mm (0.79x crop)
};

// ═══════════════════════════════════════════════════════════════════════════════════
// CAMERA BODY PRESETS
// ═══════════════════════════════════════════════════════════════════════════════════
struct CameraBody {
    const char* name;
    const char* brand;
    SensorType sensor;
    float crop_factor;
    int resolution_mp;  // Megapixels
    const char* description;
};

inline constexpr CameraBody CAMERA_BODIES[] = {
    // Custom (manual settings)
    {"Custom", "Manual", SensorType::FullFrame, 1.0f, 0, "Manual crop factor"},
    
    // Full Frame
    {"Generic Full Frame", "Generic", SensorType::FullFrame, 1.0f, 24, "Standard 35mm equivalent"},
    {"Canon EOS R5", "Canon", SensorType::FullFrame, 1.0f, 45, "High-res mirrorless"},
    {"Canon EOS R6 II", "Canon", SensorType::FullFrame, 1.0f, 24, "Versatile mirrorless"},
    {"Sony A7 IV", "Sony", SensorType::FullFrame, 1.0f, 33, "Hybrid mirrorless"},
    {"Sony A7R V", "Sony", SensorType::FullFrame, 1.0f, 61, "High-res mirrorless"},
    {"Nikon Z8", "Nikon", SensorType::FullFrame, 1.0f, 45, "Pro mirrorless"},
    {"Pentax K-1 Mark II", "Pentax", SensorType::FullFrame, 1.0f, 36, "Full frame DSLR legend"},
    
    // APS-C
    {"Canon EOS R7", "Canon", SensorType::APSC_Canon, 1.6f, 32, "APS-C mirrorless"},
    {"Canon EOS 90D", "Canon", SensorType::APSC_Canon, 1.6f, 32, "APS-C DSLR"},
    {"Sony a6700", "Sony", SensorType::APSC_Sony, 1.5f, 26, "APS-C mirrorless"},
    {"Fujifilm X-T5", "Fujifilm", SensorType::APSC_Sony, 1.5f, 40, "X-Trans sensor"},
    {"Pentax K-3 III", "Pentax", SensorType::APSC_Sony, 1.5f, 26, "APS-C DSLR flagship"},
    {"Pentax KF", "Pentax", SensorType::APSC_Sony, 1.5f, 24, "Weather-sealed APS-C"},
    
    // Cinema
    {"RED Komodo 6K", "RED", SensorType::Super35, 1.39f, 19, "Cinema 6K"},
    {"RED V-Raptor", "RED", SensorType::FullFrame, 1.0f, 35, "Cinema 8K VV"},
    {"ARRI Alexa Mini LF", "ARRI", SensorType::FullFrame, 1.0f, 9, "Large format cinema"},
    {"ARRI Alexa Mini", "ARRI", SensorType::Super35, 1.39f, 3, "Super 35 cinema"},
    {"Blackmagic Pocket 6K", "BMD", SensorType::Super35, 1.39f, 6, "Affordable cinema"},
    
    // Micro 4/3
    {"Panasonic GH6", "Panasonic", SensorType::MicroFourThirds, 2.0f, 25, "Video-focused M43"},
    {"OM System OM-1", "OM", SensorType::MicroFourThirds, 2.0f, 20, "Compact M43"},
    
    // Medium Format
    {"Hasselblad X2D", "Hasselblad", SensorType::MediumFormat, 0.79f, 100, "100MP medium format"},
    {"Fujifilm GFX 100S", "Fujifilm", SensorType::MediumFormat, 0.79f, 102, "Medium format mirrorless"},
    {"Pentax 645Z", "Pentax", SensorType::MediumFormat, 0.79f, 51, "Medium format DSLR"},
};

inline constexpr size_t CAMERA_BODY_COUNT = sizeof(CAMERA_BODIES) / sizeof(CameraBody);

// ═══════════════════════════════════════════════════════════════════════════════════
// LENS CATEGORY
// ═══════════════════════════════════════════════════════════════════════════════════
enum class LensCategory {
    UltraWide,    // <24mm
    Wide,         // 24-35mm
    Standard,     // 35-60mm
    Portrait,     // 70-135mm
    Telephoto,    // 135mm+
    Cinema        // Cine primes
};

// ═══════════════════════════════════════════════════════════════════════════════════
// PROFESSIONAL LENS PRESETS
// ═══════════════════════════════════════════════════════════════════════════════════
struct LensPreset {
    const char* name;
    const char* brand;
    LensCategory category;
    float focal_mm;
    float fov_deg;        // Vertical FOV on full frame
    float max_aperture;   // f-number (e.g., 1.4, 2.8)
    int blade_count;      // Aperture blades
    const char* description;
};

inline constexpr LensPreset LENS_PRESETS[] = {
    // Custom (manual)
    {"Custom", "Manual", LensCategory::Standard, 0.0f, 0.0f, 0.0f, 0, "Manual FOV control"},
    
    // Ultra Wide
    {"Canon RF 14-35mm f/4L", "Canon", LensCategory::UltraWide, 14.0f, 104.0f, 4.0f, 9, "Ultra wide zoom"},
    {"Sony FE 12-24mm f/2.8 GM", "Sony", LensCategory::UltraWide, 12.0f, 111.0f, 2.8f, 9, "Fast ultra wide"},
    {"Pentax HD DA 15mm f/4 Ltd", "Pentax", LensCategory::UltraWide, 15.0f, 100.0f, 4.0f, 7, "Pancake ultra wide"},
    
    // Wide
    {"Canon RF 24-70mm f/2.8L", "Canon", LensCategory::Wide, 24.0f, 84.0f, 2.8f, 9, "Standard zoom"},
    {"Sony FE 24-70mm f/2.8 GM II", "Sony", LensCategory::Wide, 24.0f, 84.0f, 2.8f, 11, "Versatile zoom"},
    {"Sigma 24mm f/1.4 DG DN", "Sigma", LensCategory::Wide, 24.0f, 84.0f, 1.4f, 11, "Fast wide prime"},
    {"Pentax FA 31mm f/1.8 Ltd", "Pentax", LensCategory::Wide, 31.0f, 72.0f, 1.8f, 9, "The legendary Limited"},
    {"Pentax HD FA 21mm f/2.4 Ltd", "Pentax", LensCategory::Wide, 21.0f, 91.0f, 2.4f, 9, "Ultra wide Limited"},
    
    // Standard
    {"Canon RF 50mm f/1.2L", "Canon", LensCategory::Standard, 50.0f, 46.8f, 1.2f, 10, "Ultimate 50mm"},
    {"Sony FE 50mm f/1.2 GM", "Sony", LensCategory::Standard, 50.0f, 46.8f, 1.2f, 11, "Fast 50 prime"},
    {"Sigma 35mm f/1.4 DG DN Art", "Sigma", LensCategory::Standard, 35.0f, 63.0f, 1.4f, 11, "35mm Art"},
    {"Zeiss Otus 55mm f/1.4", "Zeiss", LensCategory::Standard, 55.0f, 43.0f, 1.4f, 9, "Reference lens"},
    {"Pentax FA 43mm f/1.9 Ltd", "Pentax", LensCategory::Standard, 43.0f, 54.0f, 1.9f, 9, "Perfect normal - pancake"},
    {"Pentax HD FA 35mm f/2", "Pentax", LensCategory::Standard, 35.0f, 63.0f, 2.0f, 9, "Classic 35mm"},
    
    // Portrait
    {"Canon RF 85mm f/1.2L", "Canon", LensCategory::Portrait, 85.0f, 28.6f, 1.2f, 10, "Bokeh master"},
    {"Sony FE 85mm f/1.4 GM", "Sony", LensCategory::Portrait, 85.0f, 28.6f, 1.4f, 11, "Portrait king"},
    {"Sigma 105mm f/1.4 DG HSM Art", "Sigma", LensCategory::Portrait, 105.0f, 23.3f, 1.4f, 9, "Bokeh monster"},
    {"Pentax FA 77mm f/1.8 Ltd", "Pentax", LensCategory::Portrait, 77.0f, 31.5f, 1.8f, 9, "Portrait legend - creamy bokeh"},
    {"Pentax DFA* 85mm f/1.4", "Pentax", LensCategory::Portrait, 85.0f, 28.6f, 1.4f, 9, "Star series flagship"},
    
    // Telephoto
    {"Canon RF 70-200mm f/2.8L", "Canon", LensCategory::Telephoto, 135.0f, 18.0f, 2.8f, 9, "Pro tele zoom"},
    {"Sony FE 70-200mm f/2.8 GM II", "Sony", LensCategory::Telephoto, 135.0f, 18.0f, 2.8f, 11, "Compact tele"},
    {"Sony FE 135mm f/1.8 GM", "Sony", LensCategory::Telephoto, 135.0f, 18.0f, 1.8f, 11, "Fast tele prime"},
    {"Canon RF 100-500mm f/4.5-7.1L", "Canon", LensCategory::Telephoto, 200.0f, 12.0f, 5.6f, 9, "Wildlife zoom"},
    {"Pentax DFA* 70-200mm f/2.8", "Pentax", LensCategory::Telephoto, 135.0f, 18.0f, 2.8f, 9, "Pro tele zoom"},
    
    // Cinema
    {"Cooke S4/i 50mm T2", "Cooke", LensCategory::Cinema, 50.0f, 46.8f, 2.0f, 8, "Cooke Look"},
    {"Cooke S4/i 32mm T2", "Cooke", LensCategory::Cinema, 32.0f, 65.0f, 2.0f, 8, "Wide cinema"},
    {"ARRI Master Prime 35mm", "ARRI", LensCategory::Cinema, 35.0f, 63.0f, 1.4f, 15, "Master quality"},
    {"ARRI Master Prime 50mm", "ARRI", LensCategory::Cinema, 50.0f, 46.8f, 1.4f, 15, "Reference cine"},
    {"Zeiss Supreme 85mm", "Zeiss", LensCategory::Cinema, 85.0f, 28.6f, 1.5f, 11, "Supreme bokeh"},
};

inline constexpr size_t LENS_PRESET_COUNT = sizeof(LENS_PRESETS) / sizeof(LensPreset);

// ═══════════════════════════════════════════════════════════════════════════════════
// ASPECT RATIO PRESETS
// ═══════════════════════════════════════════════════════════════════════════════════
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

// ═══════════════════════════════════════════════════════════════════════════════════
// F-STOP PRESETS
// ═══════════════════════════════════════════════════════════════════════════════════
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
};

inline constexpr size_t FSTOP_PRESET_COUNT = sizeof(FSTOP_PRESETS) / sizeof(FStopPreset);

// ═══════════════════════════════════════════════════════════════════════════════════
// ISO PRESETS (Affects exposure brightness and noise simulation)
// ═══════════════════════════════════════════════════════════════════════════════════
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

// ═══════════════════════════════════════════════════════════════════════════════════
// SHUTTER SPEED PRESETS (Affects motion blur amount)
// ═══════════════════════════════════════════════════════════════════════════════════
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

// ═══════════════════════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════════

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
