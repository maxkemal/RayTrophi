/**
 * @file unified_types.h
 * @brief Platform-agnostic unified types for CPU/GPU rendering parity
 * 
 * This header provides common data structures that can be used on both
 * CPU and GPU without any CUDA dependencies. When compiled with CUDA,
 * it provides seamless conversion to/from CUDA types.
 * 
 * Design inspired by Blender Cycles' shared kernel approach.
 */
#pragma once

#include <cmath>
#include <algorithm>

// Forward declaration of CPU Vec3 class
class Vec3;

// =============================================================================
// PLATFORM DETECTION & MACROS
// =============================================================================

#ifdef __CUDACC__
    #define UNIFIED_FUNC __host__ __device__ inline
    #define UNIFIED_CONST __device__ constexpr
    // CUDA has fmaxf/fminf as built-ins
#else
    #define UNIFIED_FUNC inline
    #define UNIFIED_CONST constexpr
    // MSVC compatibility - define fmaxf/fminf if not available
    #ifndef fmaxf
        inline float fmaxf(float a, float b) { return (a > b) ? a : b; }
    #endif
    #ifndef fminf
        inline float fminf(float a, float b) { return (a < b) ? a : b; }
    #endif
#endif

// =============================================================================
// UNIFIED CONSTANTS (Same on CPU and GPU)
// =============================================================================

namespace UnifiedConstants {
    constexpr float PI = 3.14159265358979323846f;
    constexpr float INV_PI = 0.31830988618379067154f;
    constexpr float EPSILON = 1e-6f;
    
    // =========================================================================
    // INDUSTRY STANDARD BRDF CONSTANTS (Matches Cycles, Disney, PBRT)
    // =========================================================================
    
    // Minimum alpha (roughness^2) to avoid singularities in specular
    // Disney/Cycles use ~0.0001, PBRT uses 0.001
    constexpr float MIN_ALPHA = 0.0001f;
    
    // Minimum dot product clamp to avoid division by zero
    constexpr float MIN_DOT = 0.0001f;
    
    // =========================================================================
    // RENDERING CONSTANTS
    // =========================================================================
    constexpr float MAX_CONTRIBUTION = 100.0f;      // Firefly clamp threshold
    constexpr float SHADOW_BIAS = 1e-3f;           // Shadow ray offset
    
    // Russian Roulette parameters (standard path tracer values)
    constexpr int RR_START_BOUNCE = 3;             // Start RR after bounce 3
    constexpr float RR_MIN_PROB = 0.05f;           // Minimum survival probability
    constexpr float RR_MAX_PROB = 0.95f;           // Maximum survival probability
    
    // Background falloff
    constexpr float BG_FALLOFF_MIN = 0.1f;         // Minimum background contribution
    constexpr float BG_FALLOFF_RATE = 0.5f;        // Falloff rate per bounce
}

// =============================================================================
// VEC3F - Platform-Agnostic 3D Vector
// =============================================================================

struct Vec3f {
    float x, y, z;
    
    // Constructors
    UNIFIED_FUNC Vec3f() : x(0.0f), y(0.0f), z(0.0f) {}
    UNIFIED_FUNC Vec3f(float v) : x(v), y(v), z(v) {}
    UNIFIED_FUNC Vec3f(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    
    // CUDA float3 conversion (only available when CUDA is present)
    #ifdef __CUDACC__
    UNIFIED_FUNC Vec3f(const float3& v) : x(v.x), y(v.y), z(v.z) {}
    UNIFIED_FUNC operator float3() const { return make_float3(x, y, z); }
    #endif
    
    // Basic operations
    UNIFIED_FUNC Vec3f operator+(const Vec3f& b) const { return Vec3f(x + b.x, y + b.y, z + b.z); }
    UNIFIED_FUNC Vec3f operator-(const Vec3f& b) const { return Vec3f(x - b.x, y - b.y, z - b.z); }
    UNIFIED_FUNC Vec3f operator*(const Vec3f& b) const { return Vec3f(x * b.x, y * b.y, z * b.z); }
    UNIFIED_FUNC Vec3f operator/(const Vec3f& b) const { return Vec3f(x / b.x, y / b.y, z / b.z); }
    
    UNIFIED_FUNC Vec3f operator*(float s) const { return Vec3f(x * s, y * s, z * s); }
    UNIFIED_FUNC Vec3f operator/(float s) const { float inv = 1.0f / s; return Vec3f(x * inv, y * inv, z * inv); }
    
    UNIFIED_FUNC Vec3f& operator+=(const Vec3f& b) { x += b.x; y += b.y; z += b.z; return *this; }
    UNIFIED_FUNC Vec3f& operator*=(const Vec3f& b) { x *= b.x; y *= b.y; z *= b.z; return *this; }
    UNIFIED_FUNC Vec3f& operator*=(float s) { x *= s; y *= s; z *= s; return *this; }
    UNIFIED_FUNC Vec3f& operator/=(float s) { float inv = 1.0f / s; x *= inv; y *= inv; z *= inv; return *this; }
    
    UNIFIED_FUNC Vec3f operator-() const { return Vec3f(-x, -y, -z); }
    
    // Utility functions
    UNIFIED_FUNC float length_squared() const { return x*x + y*y + z*z; }
    UNIFIED_FUNC float length() const { return sqrtf(length_squared()); }
    
    UNIFIED_FUNC Vec3f normalize() const {
        float len = length();
        if (len > UnifiedConstants::EPSILON)
            return *this / len;
        return Vec3f(0.0f);
    }
    
    UNIFIED_FUNC float max_component() const {
        return fmaxf(x, fmaxf(y, z));
    }
    
    UNIFIED_FUNC float luminance() const {
        return 0.2126f * x + 0.7152f * y + 0.0722f * z;
    }
    
    UNIFIED_FUNC bool is_valid() const {
        return isfinite(x) && isfinite(y) && isfinite(z);
    }
    
    UNIFIED_FUNC Vec3f clamp(float min_val, float max_val) const {
        return Vec3f(
            fminf(fmaxf(x, min_val), max_val),
            fminf(fmaxf(y, min_val), max_val),
            fminf(fmaxf(z, min_val), max_val)
        );
    }
};

// Free functions for Vec3f
UNIFIED_FUNC Vec3f operator*(float s, const Vec3f& v) { return v * s; }

UNIFIED_FUNC float dot(const Vec3f& a, const Vec3f& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

UNIFIED_FUNC Vec3f cross(const Vec3f& a, const Vec3f& b) {
    return Vec3f(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

UNIFIED_FUNC Vec3f normalize(const Vec3f& v) {
    return v.normalize();
}

UNIFIED_FUNC Vec3f lerp(const Vec3f& a, const Vec3f& b, float t) {
    return a * (1.0f - t) + b * t;
}

UNIFIED_FUNC Vec3f reflect(const Vec3f& v, const Vec3f& n) {
    return v - 2.0f * dot(v, n) * n;
}

UNIFIED_FUNC Vec3f fminf(const Vec3f& a, const Vec3f& b) {
    return Vec3f(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

UNIFIED_FUNC Vec3f fmaxf(const Vec3f& a, const Vec3f& b) {
    return Vec3f(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

UNIFIED_FUNC Vec3f exp_vec(const Vec3f& v) {
    return Vec3f(expf(v.x), expf(v.y), expf(v.z));
}

// =============================================================================
// VEC2F - Platform-Agnostic 2D Vector
// =============================================================================

struct Vec2f {
    float x, y;
    
    UNIFIED_FUNC Vec2f() : x(0.0f), y(0.0f) {}
    UNIFIED_FUNC Vec2f(float v) : x(v), y(v) {}
    UNIFIED_FUNC Vec2f(float x_, float y_) : x(x_), y(y_) {}
    
    #ifdef __CUDACC__
    UNIFIED_FUNC Vec2f(const float2& v) : x(v.x), y(v.y) {}
    UNIFIED_FUNC operator float2() const { return make_float2(x, y); }
    #endif
    
    UNIFIED_FUNC Vec2f clamp(float min_val, float max_val) const {
        return Vec2f(
            fminf(fmaxf(x, min_val), max_val),
            fminf(fmaxf(y, min_val), max_val)
        );
    }
};

// =============================================================================
// UNIFIED MATERIAL - Same structure for CPU and GPU
// =============================================================================

/**
 * @brief Unified Material structure - identical on CPU and GPU
 * 
 * This struct uses Vec3f instead of float3 to be platform-agnostic.
 * Size: 96 bytes, 16-byte aligned for optimal cache access.
 */
struct alignas(16) UnifiedMaterial {
    // Block 1: Albedo + opacity (16 bytes)
    Vec3f albedo;                     // 12 bytes - base color
    float opacity;                    // 4 bytes  - alpha/opacity
    
    // Block 2: PBR core properties (16 bytes)
    float roughness;                  // 4 bytes
    float metallic;                   // 4 bytes
    float clearcoat;                  // 4 bytes
    float transmission;               // 4 bytes
    
    // Block 3: Emission + IOR (16 bytes)
    Vec3f emission;                   // 12 bytes - emission color
    float ior;                        // 4 bytes  - index of refraction
    
    // Block 4: Subsurface (16 bytes)
    Vec3f subsurface_color;           // 12 bytes
    float subsurface;                 // 4 bytes
    
    // Block 5: Additional properties (16 bytes)
    float artistic_albedo_response;   // 4 bytes
    float anisotropic;                // 4 bytes
    float sheen;                      // 4 bytes
    float sheen_tint;                 // 4 bytes
    
    // Block 6: Texture IDs (16 bytes) - resolved at runtime
    int albedo_tex_id;                // 4 bytes
    int normal_tex_id;                // 4 bytes
    int roughness_tex_id;             // 4 bytes
    int metallic_tex_id;              // 4 bytes
    
    // Block 7: More texture IDs (16 bytes)
    int emission_tex_id;              // 4 bytes
    int opacity_tex_id;               // 4 bytes
    int transmission_tex_id;          // 4 bytes
    int _padding;                     // 4 bytes - alignment padding
    
    // Default constructor with sensible defaults
    UNIFIED_FUNC UnifiedMaterial() :
        albedo(0.8f, 0.8f, 0.8f),
        opacity(1.0f),
        roughness(0.5f),
        metallic(0.0f),
        clearcoat(0.0f),
        transmission(0.0f),
        emission(0.0f),
        ior(1.45f),
        subsurface_color(0.0f),
        subsurface(0.0f),
        artistic_albedo_response(0.0f),
        anisotropic(0.0f),
        sheen(0.0f),
        sheen_tint(0.0f),
        albedo_tex_id(-1),
        normal_tex_id(-1),
        roughness_tex_id(-1),
        metallic_tex_id(-1),
        emission_tex_id(-1),
        opacity_tex_id(-1),
        transmission_tex_id(-1),
        _padding(0)
    {}
};

// =============================================================================
// UNIFIED LIGHT TYPE
// =============================================================================

enum class UnifiedLightType : int {
    Point = 0,
    Directional = 1,
    Area = 2,
    Spot = 3
};

// =============================================================================
// UNIFIED LIGHT - Same structure for CPU and GPU
// =============================================================================

/**
 * @brief Unified Light structure - identical on CPU and GPU
 * 
 * Matches the GPU's LightGPU structure but uses Vec3f for platform independence.
 * Size: 80 bytes
 */
struct alignas(16) UnifiedLight {
    Vec3f position;                   // 12 bytes
    float intensity;                  // 4 bytes
    
    Vec3f direction;                  // 12 bytes - for directional, spot, area
    float radius;                     // 4 bytes - soft shadow radius
    
    Vec3f color;                      // 12 bytes - normalized color [0-1]
    int type;                         // 4 bytes - 0=point, 1=dir, 2=area, 3=spot
    
    // Spot light parameters
    float inner_cone_cos;             // 4 bytes
    float outer_cone_cos;             // 4 bytes
    
    // Area light parameters
    float area_width;                 // 4 bytes
    float area_height;                // 4 bytes
    
    Vec3f area_u;                     // 12 bytes - area light U vector
    float _padding1;                  // 4 bytes
    
    Vec3f area_v;                     // 12 bytes - area light V vector
    float _padding2;                  // 4 bytes
    
    // Default constructor
    UNIFIED_FUNC UnifiedLight() :
        position(0.0f),
        intensity(1.0f),
        direction(0.0f, -1.0f, 0.0f),
        radius(0.1f),
        color(1.0f),
        type(0),
        inner_cone_cos(0.9f),
        outer_cone_cos(0.8f),
        area_width(1.0f),
        area_height(1.0f),
        area_u(1.0f, 0.0f, 0.0f),
        _padding1(0.0f),
        area_v(0.0f, 0.0f, 1.0f),
        _padding2(0.0f)
    {}
    
    UNIFIED_FUNC UnifiedLightType getType() const {
        return static_cast<UnifiedLightType>(type);
    }
};

// =============================================================================
// UNIFIED HIT RESULT - Common hit record structure
// =============================================================================

/**
 * @brief Unified hit result - for passing hit info to shading functions
 */
struct UnifiedHitResult {
    Vec3f position;                   // Hit point
    Vec3f normal;                     // Shading normal (may include normal map)
    Vec3f geometric_normal;           // Geometric normal (from triangle)
    Vec2f uv;                         // Texture coordinates
    int material_id;                  // Index into material array
    float t;                          // Ray parameter at hit
    bool front_face;                  // True if ray hit front face
    
    // Texture object handles (resolved differently on CPU vs GPU)
    // On GPU: cudaTextureObject_t, on CPU: texture ID for lookup
    int albedo_tex;
    int normal_tex;
    int roughness_tex;
    int metallic_tex;
    int emission_tex;
    int opacity_tex;
    int transmission_tex;
    
    bool has_albedo_tex;
    bool has_normal_tex;
    bool has_roughness_tex;
    bool has_metallic_tex;
    bool has_emission_tex;
    bool has_opacity_tex;
    bool has_transmission_tex;
    
    UNIFIED_FUNC UnifiedHitResult() :
        position(0.0f),
        normal(0.0f, 1.0f, 0.0f),
        geometric_normal(0.0f, 1.0f, 0.0f),
        uv(0.0f),
        material_id(0),
        t(0.0f),
        front_face(true),
        albedo_tex(-1),
        normal_tex(-1),
        roughness_tex(-1),
        metallic_tex(-1),
        emission_tex(-1),
        opacity_tex(-1),
        transmission_tex(-1),
        has_albedo_tex(false),
        has_normal_tex(false),
        has_roughness_tex(false),
        has_metallic_tex(false),
        has_emission_tex(false),
        has_opacity_tex(false),
        has_transmission_tex(false)
    {}
};
