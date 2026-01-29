/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          Camera.h
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#ifndef CAMERA_H
#define CAMERA_H

#include <vector>
#include "Vec3.h"
#include "Matrix4x4.h"
#include "Ray.h"
#include "AABB.h"
#include "ThreadLocalRNG.h"

// ═══════════════════════════════════════════════════════════════════════════════
// CAMERA MODE - Controls feature availability and physical simulation level
// ═══════════════════════════════════════════════════════════════════════════════
enum class CameraMode {
    Auto,       // Amatör - Otomatik ayarlar, kısıtlı kontrol, kusurlar kapalı
    Pro,        // Profesyonel - Manuel kontrol, temiz görüntü, opsiyonel kusurlar  
    Cinema      // Sinematik - Tam fiziksel simülasyon, tüm lens/sensör kusurları
};
class Camera {
private:
    struct Plane {
        Vec3 normal;
        float distance;

        Plane() : normal(Vec3()), distance(0.0f) {}
        Plane(const Vec3& n, const Vec3& point) : normal(n.normalize()) {
            distance = -Vec3::dot(normal, point);
        }

        float distanceToPoint(const Vec3& point) const {
            return Vec3::dot(normal, point) + distance;
        }
    };

public:
    Vec3 initialLookDirection;
    std::string nodeName;
    bool visible = true;
    int blade_count = 6;
    float aperture = 0.0f;
    float focus_dist = 10.0f;
    Vec3 origin;
    Vec3 u, v, w;
    Vec3 lookfrom;
    Vec3 lookat;
    Vec3 vup;
    float aspect = 1.7777f;
    float near_dist = 0.01f;
    float far_dist = 1000.0f;
    float fov = 45.0f;
    float aspect_ratio = 1.7777f;
    float vfov = 45.0f;
    // Camera(Vec3 lookfrom, Vec3 lookat, Vec3 vup, double vfov, double aspect, double aperture, double focus_dist);

    Camera(Vec3 lookfrom, Vec3 lookat, Vec3 vup, float vfov, float aspect, float aperture, float focus_dist, int blade_count);
    Camera();
    Ray get_ray(float s, float t) const;

    int random_int(int min, int max) const;

    void update_camera_vectors();

    void moveToTargetLocked(const Vec3& new_position);

    void setLookDirection(const Vec3& direction_normalized);

    Vec3 random_in_unit_polygon(int sides) const;

    float calculate_bokeh_intensity(const Vec3& point) const;

    Vec3 create_bokeh_shape(const Vec3& color, float intensity) const;
    void reset();
    void save_initial_state();
    bool isPointInFrustum(const Vec3& point, float size) const;
    Matrix4x4 getRotationMatrix() const;
    bool isAABBInFrustum(const AABB& aabb) const;
    std::vector<AABB> performFrustumCulling(const std::vector<AABB>& objects) const;
    Vec3 lower_left_corner;
    Vec3 horizontal;
    Vec3 vertical;
    float lens_radius = 0.0f;
    
    // ═══════════════════════════════════════════════════════════════════════════
    // PROFESSIONAL EXPOSURE SETTINGS
    // ═══════════════════════════════════════════════════════════════════════════
    int iso = 100;                     // Current ISO value
    float shutter_speed = 250.0f;      // Shutter speed as 1/x (e.g., 250 = 1/250s)
    int iso_preset_index = 1;          // Default: ISO 100
    int shutter_preset_index = 1;      // Default: 1/4000s
    int fstop_preset_index = 4;        // Default: f/2.8
    int lens_preset_index = 0;         // Default: Custom/Manual
    int body_preset_index = 1;         // Default: Generic Full Frame
    bool auto_exposure = true;         // Default to manual to use above settings
    float ev_compensation = 0.0f;      // EV compensation (-2 to +2)
    float calculated_ev = 0.0f;        // Calculated exposure value (output)
    
    // Aspect Ratio for Output (syncs with final render)
    int output_aspect_index = 2;       // Default: 16:9 (index into CameraPresets::ASPECT_RATIOS)

    // PHYSICAL LENS SETTINGS
    // ═══════════════════════════════════════════════════════════════════════════
    bool use_physical_lens = false;    // Toggle between basic FOV and Physical Lens
    float focal_length_mm = 50.0f;     // Focal length in mm (e.g. 24, 35, 50, 85)
    float sensor_width_mm = 36.0f;     // Sensor width (Full Frame = 36mm)
    float sensor_height_mm = 24.0f;    // Sensor height (Full Frame = 24mm)
    bool enable_motion_blur = false;   // Enable Camera Motion Blur (requires velocity calculation)
    float distortion = 0.0f;           // Lens Distortion (-0.5 to 0.5): Negative=Barrel, Positive=Pincushion
    
    // ═══════════════════════════════════════════════════════════════════════════
    // CAMERA MODE - Auto/Pro/Cinema
    // ═══════════════════════════════════════════════════════════════════════════
    CameraMode camera_mode = CameraMode::Pro;
    
    // ═══════════════════════════════════════════════════════════════════════════
    // CINEMA MODE - Lens Imperfections (only active when camera_mode == Cinema)
    // ═══════════════════════════════════════════════════════════════════════════
    
    // Lens Quality (affects all optical aberrations)
    // 0.0 = Vintage/Budget lens (more aberrations)
    // 1.0 = Perfect optical design (minimal aberrations)
    float lens_quality = 0.7f;
    
    // Auto-calculate lens characteristics (true = physics-based, false = manual)
    bool auto_lens_characteristics = false;
    
    // Chromatic Aberration (Renk Sapması)
    bool enable_chromatic_aberration = false;
    float chromatic_aberration = 0.0f;      // 0-1: Lateral CA amount
    float chromatic_aberration_r = 1.002f;  // Red channel offset multiplier
    float chromatic_aberration_b = 0.998f;  // Blue channel offset multiplier
    
    // Vignetting (Köşe Kararması)
    bool enable_vignetting = false;
    float vignetting_amount = 0.0f;         // 0-1: Vignette strength
    float vignetting_falloff = 2.0f;        // Falloff curve exponent (1.5-4.0)
    
    // Calculate lens characteristics from physical properties
    // Call this when focal_length, aperture, or lens_quality changes
    void calculateLensCharacteristics() {
        if (!auto_lens_characteristics || camera_mode != CameraMode::Cinema) return;
        
        // Get current f-stop
        float f_number = 2.8f;
        if (fstop_preset_index > 0 && fstop_preset_index < 12) {
            const float fstops[] = {0, 1.2f, 1.4f, 1.8f, 2.0f, 2.8f, 4.0f, 5.6f, 8.0f, 11.0f, 16.0f, 22.0f};
            f_number = fstops[fstop_preset_index];
        }
        
        // ─────────────────────────────────────────────────────────────────────
        // VIGNETTING: Based on focal length and aperture
        // - Wide angle = more mechanical vignetting
        // - Wide aperture = more optical vignetting
        // - Stopping down reduces vignetting significantly
        // ─────────────────────────────────────────────────────────────────────
        float focal_factor = 1.0f;
        if (focal_length_mm < 24.0f) focal_factor = 1.5f;       // Ultra wide
        else if (focal_length_mm < 35.0f) focal_factor = 1.2f;  // Wide
        else if (focal_length_mm < 50.0f) focal_factor = 1.0f;  // Normal
        else if (focal_length_mm < 85.0f) focal_factor = 0.8f;  // Portrait
        else focal_factor = 0.6f;                                // Telephoto
        
        // Aperture effect: wide open = more vignetting
        float aperture_vignette = 1.0f / (f_number * 0.5f);     // f/1.4 = 1.43, f/8 = 0.25
        aperture_vignette = std::min(aperture_vignette, 1.0f);
        
        // Quality reduces vignetting
        float quality_reduction = 1.0f - (lens_quality * 0.6f);
        
        vignetting_amount = focal_factor * aperture_vignette * quality_reduction * 0.4f;
        vignetting_amount = std::clamp(vignetting_amount, 0.0f, 0.8f);
        vignetting_falloff = 2.0f + (1.0f - lens_quality);
        enable_vignetting = (vignetting_amount > 0.02f);
        
        // ─────────────────────────────────────────────────────────────────────
        // CHROMATIC ABERRATION: Based on lens quality and aperture
        // - Budget lenses have more CA
        // - Wide aperture = more visible CA
        // - Stopping down reduces CA
        // ─────────────────────────────────────────────────────────────────────
        float ca_base = (1.0f - lens_quality) * 0.02f;  // 0 to 0.02 based on quality
        
        // Aperture effect: wide open = more CA
        float aperture_ca = (2.8f / f_number);  // f/1.4 = 2, f/8 = 0.35
        aperture_ca = std::clamp(aperture_ca, 0.2f, 2.0f);
        
        chromatic_aberration = ca_base * aperture_ca;
        chromatic_aberration = std::clamp(chromatic_aberration, 0.0f, 0.03f);
        
        // R/B channel scales
        chromatic_aberration_r = 1.0f + chromatic_aberration * 0.5f;  // Red bends outward
        chromatic_aberration_b = 1.0f - chromatic_aberration * 0.5f;  // Blue bends inward
        
        enable_chromatic_aberration = (chromatic_aberration > 0.001f);
        
        // ─────────────────────────────────────────────────────────────────────
        // AUTO DISTORTION: Based on focal length
        // ─────────────────────────────────────────────────────────────────────
        if (focal_length_mm < 24.0f) {
            distortion = -0.15f * (1.0f - lens_quality);  // Barrel
        } else if (focal_length_mm > 100.0f) {
            distortion = 0.05f * (1.0f - lens_quality);   // Pincushion
        } else {
            distortion = 0.0f;  // Normal range - minimal distortion
        }
    }
    
    // Focus Breathing (Odak Soluması - FOV changes with focus)
    bool enable_focus_breathing = false;
    float focus_breathing_amount = 0.05f;   // % FOV change per focus distance change
    
    // Lens Flare
    bool enable_lens_flare = false;
    float lens_flare_intensity = 0.5f;
    float lens_flare_threshold = 0.9f;      // Brightness threshold to trigger flare
    bool anamorphic_flare = false;          // Horizontal blue streak (cinema style)
    
    
    // ═══════════════════════════════════════════════════════════════════════════
    // CAMERA SHAKE / HANDHELD SIMULATION
    // ═══════════════════════════════════════════════════════════════════════════
    bool enable_camera_shake = false;
    float shake_intensity = 0.03f;          // Overall shake multiplier (0-1), 0.03 = Professional default
    float shake_frequency = 8.0f;           // Hz (hand tremor ~8-12Hz)
    
    // Handheld physics
    float handheld_sway_amplitude = 0.005f;   // Body sway (meters)
    float handheld_sway_frequency = 0.5f;     // Hz
    float breathing_amplitude = 0.003f;       // Breathing motion (meters)
    float breathing_frequency = 0.25f;        // ~15 breaths/minute
    
    // Focus Drift (shake-induced focus variation)
    bool enable_focus_drift = true;           // Focus follows shake movement
    float focus_drift_amount = 0.1f;          // Max focus distance variation (meters)
    
    // Operator skill (affects shake reduction)
    enum class OperatorSkill { Amateur, Intermediate, Professional, Expert };
    OperatorSkill operator_skill = OperatorSkill::Professional;
    
    // IBIS (In-Body Image Stabilization)
    bool ibis_enabled = false;
    float ibis_effectiveness = 5.0f;        // Stops of stabilization (typically 3-8 stops)
    
    // ═══════════════════════════════════════════════════════════════════════════
    // PHYSICAL EXPOSURE (Fiziksel Pozlama)
    // ═══════════════════════════════════════════════════════════════════════════
    int native_iso = 100;                   // Sensor native ISO (for noise calculation)
    bool use_physical_exposure = false;     // Use physical exposure calculation
    
    // Shutter Angle (Cinema style) - Alternative to shutter speed
    bool use_shutter_angle = false;
    float shutter_angle = 180.0f;           // Degrees (180 = 50% duty cycle)
    
    // Get physical exposure multiplier
    // This multiplies the render result (signal + variance together!)
    float getPhysicalExposureMultiplier() const {
        if (!use_physical_exposure) return 1.0f;
        
        // Get f-stop from aperture or preset
        float f_number = 2.8f; // default
        if (fstop_preset_index > 0) {
            // Will use CameraPresets later
            switch(fstop_preset_index) {
                case 1: f_number = 1.2f; break;
                case 2: f_number = 1.4f; break;
                case 3: f_number = 1.8f; break;
                case 4: f_number = 2.0f; break;
                case 5: f_number = 2.8f; break;
                case 6: f_number = 4.0f; break;
                case 7: f_number = 5.6f; break;
                case 8: f_number = 8.0f; break;
                case 9: f_number = 11.0f; break;
                case 10: f_number = 16.0f; break;
                case 11: f_number = 22.0f; break;
            }
        }
        
        // Shutter time in seconds
        float shutter_seconds = 1.0f / shutter_speed;
        
        // ISO amplification factor (relative to native ISO)
        // THIS IS THE KEY: ISO amplifies signal AND variance together!
        float iso_gain = static_cast<float>(iso) / static_cast<float>(native_iso);
        
        // Aperture factor: Light ∝ 1/N²
        float aperture_factor = 1.0f / (f_number * f_number);
        
        // Shutter factor: Light ∝ time (normalized to 1/250s)
        float shutter_factor = shutter_seconds / (1.0f / 250.0f);
        
        // Total exposure = Light gathered × Amplification
        const float reference_scale = 2.5f; // Scene brightness calibration
        return aperture_factor * shutter_factor * iso_gain * reference_scale;
    }
    
    // Get recommended sample count based on ISO
    // Higher ISO = more samples needed to reduce visible variance
    int getRecommendedSamples(int base_samples = 64) const {
        if (iso <= native_iso) return base_samples;
        
        float iso_factor = std::log2(static_cast<float>(iso) / 100.0f);
        return base_samples * static_cast<int>(std::pow(2.0f, std::max(0.0f, iso_factor * 0.5f)));
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // CAMERA RIG SYSTEM (Dolly, Crane, Orbit, Handheld, Steadicam)
    // ═══════════════════════════════════════════════════════════════════════════
    enum class RigMode { Static, Dolly, Crane, Orbit, Handheld, Steadicam };
    RigMode rig_mode = RigMode::Static;
    
    // Dolly - Linear track movement
    float dolly_position = 0.0f;       // Position along track (units)
    float dolly_speed = 1.0f;          // Movement speed multiplier
    Vec3 dolly_start_pos;              // Initial position when dolly started
    Vec3 dolly_end_pos;                // End position for dolly track
    
    // Crane - Arm with boom
    float crane_arm = 5.0f;            // Arm length
    float crane_height = 2.0f;         // Base height
    float crane_boom = 0.0f;           // Boom angle (-45 to +45)
    
    // Orbit - Around target
    float orbit_angle = 0.0f;          // Current angle
    float orbit_radius = 5.0f;         // Distance from target
    Vec3 orbit_target;                 // Point to orbit around
    
    // Steadicam - Smoothed movement
    float steadicam_smoothing = 0.9f;  // Position smoothing (0-1)
    
    // Camera physics (for realistic motion)
    float camera_mass_kg = 1.3f;       // Body + lens mass
    float camera_damping = 5.0f;       // Movement damping
    Vec3 camera_velocity;              // Current velocity (m/s)
    Vec3 camera_angular_velocity;      // Current angular velocity (rad/s)
    
    // ═══════════════════════════════════════════════════════════════════════════
    // STATE MANAGEMENT (Dirty Flag Architecture)
    // ═══════════════════════════════════════════════════════════════════════════
    bool is_dirty = false;
    
    void markDirty() {
        is_dirty = true;
    }

    bool checkDirty() {
        bool was_dirty = is_dirty;
        is_dirty = false;
        return was_dirty;
    }

private:
    // Initial state for reset
    Vec3 init_lookfrom;
    Vec3 init_lookat;
    Vec3 init_vup;
    float init_vfov = 45.0f;
    float init_aperture = 0.0f;
    float init_focus_dist = 10.0f;
    void updateFrustumPlanes();

    Vec3 getViewDirection() const;

    // Frustum culling i�in ek alanlar

    Plane frustum_planes[6];
};

#endif // CAMERA_H




