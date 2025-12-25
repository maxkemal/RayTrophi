#ifndef CAMERA_H
#define CAMERA_H

#include <vector>
#include "Vec3.h"
#include "Matrix4x4.h"
#include "Ray.h"
#include "AABB.h"
#include "ThreadLocalRNG.h"
class Camera {
private:
    struct Plane {
        Vec3 normal;
        double distance;

        Plane() : normal(Vec3()), distance(0) {}
        Plane(const Vec3& n, const Vec3& point) : normal(n.normalize()) {
            distance = -Vec3::dot(normal, point);
        }

        double distanceToPoint(const Vec3& point) const {
            return Vec3::dot(normal, point) + distance;
        }
    };

public:
    Vec3 initialLookDirection;
    std::string nodeName;
    int blade_count;
    float aperture;
    float focus_dist;
    Vec3 origin;
    Vec3 u, v, w;
    Vec3 lookfrom;
    Vec3 lookat;
    Vec3 vup;
    float aspect;
    float near_dist;
    float far_dist;
    float fov;
    float aspect_ratio;
    float vfov;
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
    float lens_radius;
    
    // ═══════════════════════════════════════════════════════════════════════════
    // PROFESSIONAL EXPOSURE SETTINGS
    // ═══════════════════════════════════════════════════════════════════════════
    int iso = 100;                     // Current ISO value
    float shutter_speed = 250.0f;      // Shutter speed as 1/x (e.g., 250 = 1/250s)
    int iso_preset_index = 1;          // Default: ISO 100
    int shutter_preset_index = 1;      // Default: 1/4000s
    int fstop_preset_index = 4;        // Default: f/2.8
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
    
    // ═══════════════════════════════════════════════════════════════════════════
    // CAMERA RIG SYSTEM (Dolly, Crane, Orbit)
    // ═══════════════════════════════════════════════════════════════════════════
    enum class RigMode { Static, Dolly, Crane, Orbit };
    RigMode rig_mode = RigMode::Static;
    
    // Dolly - Linear track movement
    float dolly_position = 0.0f;       // Position along track (units)
    float dolly_speed = 1.0f;          // Movement speed multiplier
    Vec3 dolly_start_pos;              // Initial position when dolly started
    
    // Crane - Arm with boom
    float crane_arm = 5.0f;            // Arm length
    float crane_height = 2.0f;         // Base height
    float crane_boom = 0.0f;           // Boom angle (-45 to +45)
    
    // Orbit - Around target
    float orbit_angle = 0.0f;          // Current angle
    float orbit_radius = 5.0f;         // Distance from target
    Vec3 orbit_target;                 // Point to orbit around
    
private:
    // Initial state for reset
    Vec3 init_lookfrom;
    Vec3 init_lookat;
    Vec3 init_vup;
    float init_vfov;
    float init_aperture;
    float init_focus_dist;
    void updateFrustumPlanes();

    Vec3 getViewDirection() const;

    // Frustum culling i�in ek alanlar

    Plane frustum_planes[6];
};

#endif // CAMERA_H

