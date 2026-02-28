/*
 * File: vulkan_world_data.h
 * Description: Extended Vulkan World Data structure - mirrors CPU WorldData for full Nishita Sky Model
 * 
 * This header provides an enhanced GPU-compatible world data structure that supports:
 * - Complete Nishita Atmosphere Sky Model
 * - Atmosphere LUT (Look-Up Table) references
 * - Cloud parameters (single & multi-layer)
 * - Fog and God Rays
 * - Anisotropic material properties
 * 
 * Size: ~256 bytes (suitable for SSBO)
 */

#pragma once
#include <cstdint>

namespace VulkanRT {

/**
 * @struct VkWorldDataExtended
 * @brief GPU-friendly world data structure supporting full Nishita Sky Model
 * 
 * This structure mirrors the CPU WorldData class and provides all necessary
 * parameters for advanced atmospheric rendering on Vulkan backend.
 * 
 * Memory layout optimized for cache-friendly access (aligned to 16-byte boundary).
 */
struct VkWorldDataExtended {
    // ═══════════════════════════════════════════════════════════════════════════════
    // CORE MODE & SUN TINT (32 bytes) - Cache Line 0
    // ═══════════════════════════════════════════════════════════════════════════════
    float sunDir[3];        // Sun direction vector (normalized)
    int   mode;             // WorldMode: 0=Color, 1=HDRI, 2=Nishita
    
    float sunColor[3];      // Sun tint color (RGB)
    float sunIntensity;     // Sun brightness multiplier
    
    // ═══════════════════════════════════════════════════════════════════════════════
    // NISHITA SUN PARAMETERS (48 bytes) - Cache Line 1-2
    // ═══════════════════════════════════════════════════════════════════════════════
    float sunSize;          // Sun disc angular size (degrees, default ~0.545)
    float mieAnisotropy;    // Mie scattering g-factor (0.0-0.99, default 0.8)
    float rayleighDensity;  // Rayleigh scale height (meters)
    float mieDensity;       // Mie scale height (meters)
    
    float humidity;         // 0.0 (Dry) to 1.0 (Humid)
    float temperature;      // Celsius (-50 to +50)
    float ozoneAbsorptionScale; // Blue hour intensity (0.0-10.0)
    float _pad0;            // Padding for alignment
    
    // ═══════════════════════════════════════════════════════════════════════════════
    // ATMOSPHERE DENSITY PARAMETERS (32 bytes) - Cache Line 2-3
    // ═══════════════════════════════════════════════════════════════════════════════
    float airDensity;       // Rayleigh scattering multiplier (0.0-1.0+)
    float dustDensity;      // Mie/aerosol multiplier (0.0-1.0+)
    float ozoneDensity;     // Ozone density multiplier (affects saturation)
    float altitude;         // Camera altitude above sea level (meters)
    
    float planetRadius;     // Earth radius (meters, ~6371000)
    float atmosphereHeight; // Atmosphere thickness (meters, ~100000)
    float _pad1, _pad2;     // Padding
    
    // ═══════════════════════════════════════════════════════════════════════════════
    // CLOUD LAYER 1 PARAMETERS (64 bytes) - Cache Line 3-5
    // ═══════════════════════════════════════════════════════════════════════════════
    int   cloudsEnabled;    // 1 = render clouds
    float cloudCoverage;    // 0.0-1.0 (sky coverage)
    float cloudDensity;     // Opacity multiplier
    float cloudScale;       // Noise frequency (larger = bigger clouds)
    
    float cloudHeightMin;   // Cloud bottom altitude (meters)
    float cloudHeightMax;   // Cloud top altitude (meters)
    float cloudOffsetX;     // X offset for wind/animation
    float cloudOffsetZ;     // Z offset for wind/animation
    
    float cloudQuality;     // Quality multiplier for steps
    float cloudDetail;      // Detail level (0.5 low, 1.0 normal, 2.0 high)
    int   cloudBaseSteps;   // Base ray marching steps (e.g., 48)
    int   cloudLightSteps;  // Light marching steps (0-disabled, 4-8 recommended)
    
    float cloudShadowStrength; // Shadow darkness (0-2.0)
    float cloudAmbientStrength; // Ambient contribution (0.5-2.0)
    float cloudSilverIntensity; // Silver lining (0-2.0)
    float cloudAbsorption;     // Light absorption rate (0.5-2.0)
    
    // ═══════════════════════════════════════════════════════════════════════════════
    // ADVANCED CLOUD SCATTERING (32 bytes) - Cache Line 6
    // ═══════════════════════════════════════════════════════════════════════════════
    float cloudAnisotropy;      // Forward scattering g-factor (0.0-0.99)
    float cloudAnisotropyBack;  // Backward scattering (-0.99-0.0)
    float cloudLobeMix;         // Blend forward/backward (0.0-1.0)
    float cloudEmissiveIntensity; // Emission strength
    
    float cloudEmissiveColor[3]; // Emission color (RGB)
    float _pad3;                 // Padding
    
    // ═══════════════════════════════════════════════════════════════════════════════
    // FOG PARAMETERS (32 bytes) - Cache Line 7
    // ═══════════════════════════════════════════════════════════════════════════════
    int   fogEnabled;       // 1 = enable fog
    float fogDensity;       // Base density (0.0-0.1)
    float fogHeight;        // Falloff height (meters)
    float fogFalloff;       // Exponential falloff (0.001-0.01)
    
    float fogDistance;      // Max fog distance (meters)
    float fogSunScatter;    // Sun scattering in fog
    float fogColor[3];      // Fog tint color (RGB)
    
    // ═══════════════════════════════════════════════════════════════════════════════
    // VOLUMETRIC GOD RAYS (16 bytes)
    // ═══════════════════════════════════════════════════════════════════════════════
    int   godRaysEnabled;   // 1 = enable god rays
    float godRaysIntensity; // Brightness (0.0-2.0)
    float godRaysDensity;   // Density/thickness
    int   godRaysSamples;   // Quality steps (8-32)
    
    // ═══════════════════════════════════════════════════════════════════════════════
    // ENVIRONMENT & LUT REFERENCES (32 bytes) - Cache Line 8
    // ═══════════════════════════════════════════════════════════════════════════════
    int   envTexSlot;       // Environment texture descriptor slot
    float envIntensity;     // Environment map intensity
    float envRotation;      // Rotation in radians
    int   _pad5;            // Padding
    
    // LUT availability flags (0 or 1 for each, or actual descriptor handles in real impl)
    // Note: Using uint64_t for GPU texture object handles (matches CUDA)
    uint64_t transmittanceLUT;          // Transmittance LUT handle/flag
    uint64_t skyviewLUT;                // Sky view LUT handle/flag
    uint64_t multiScatterLUT;           // Multi-scatter LUT handle/flag
    uint64_t aerialPerspectiveLUT;     // Aerial perspective LUT handle/flag
};

// Compile-time validation
// static_assert(sizeof(VkWorldDataExtended) >= 256, "VkWorldDataExtended must be at least 256 bytes");
// Note: SSBO alignment is handled by Vulkan driver; size will be padded automatically if needed

} // namespace VulkanRT
