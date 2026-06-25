/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          HairBSDF.h
 * Author:        Kemal Demirtaş
 * Description:   Physically-based hair shading model
 *                Implements Marschner et al. 2003 with d'Eon improvements
 * 
 * References:
 *   - "Light Scattering from Human Hair Fibers" (Marschner 2003)
 *   - "An Energy-Conserving Hair Reflectance Model" (d'Eon 2011)
 *   - "A Practical and Controllable Hair and Fur Model" (Chiang 2016, Disney)
 * =========================================================================
 */
#ifndef HAIR_BSDF_H
#define HAIR_BSDF_H

#include "Vec3.h"
#include <cmath>
#include "json.hpp"
#include "Texture.h"
#include "material_gpu.h"

namespace Hair {

/**
 * @brief Hair material properties (matches Blender Principled Hair BSDF)
 */
struct HairMaterialParams {
    // Color modes
    enum class ColorMode : uint8_t {
        DIRECT_COLORING,    // Use albedo directly
        MELANIN,            // Physical melanin pigmentation
        ABSORPTION,         // Explicit absorption coefficient
        ROOT_UV_MAP         // Inherit color from scalp mesh UVs
    };
    
    ColorMode colorMode = ColorMode::DIRECT_COLORING;
    
    // Direct coloring - Medium brown default (more visible)
    Vec3 color = Vec3(0.4f, 0.25f, 0.15f);  // Medium brown default
    
    // Melanin model (realistic human hair)
    float melanin = 0.8f;               // 0=blonde, 0.5=brown, 1=black
    float melaninRedness = 0.5f;        // Red vs brown pigment ratio
    float tint = 0.0f;                  // Additional color tint strength
    Vec3 tintColor = Vec3(1, 1, 1);     // Tint color
    
    // Absorption coefficient (advanced)
    Vec3 absorptionCoefficient = Vec3(0.245531f, 0.52f, 1.365f);
    
    // Surface roughness
    float roughness = 0.3f;             // Longitudinal roughness (0-1)
    float radialRoughness = 0.3f;       // Azimuthal roughness (0-1)
    
    // Cuticle (scale tilt)
    float cuticleAngle = 2.0f;          // Degrees, typically 2-3°
    
    // Internal structure
    float ior = 1.55f;                  // Index of refraction (1.55 for human hair)
    float randomHue = 0.0f;             // Per-strand color variation
    float randomValue = 0.0f;           // Per-strand brightness variation
    
    // Coat (animal fur outer layer)
    float coat = 0.0f;                  // Coat strength (0=off)
    Vec3 coatTint = Vec3(1, 1, 1);      // Coat reflection tint
    
    // Emission (for bioluminescent effects, stylized)
    Vec3 emission = Vec3(0, 0, 0);
    float emissionStrength = 0.0f;
    
    // --- NEW: Root-to-Tip Color Gradient ---
    // Blend absorption towards tipColor at the tip end
    bool  enableRootTipGradient = false;
    Vec3  tipColor = Vec3(0.6f, 0.4f, 0.25f); // Lighter tip default
    float rootTipBalance = 0.5f;               // 0=all root color, 1=all tip color at tip
    
    // --- NEW: Specular Tint ---
    // Tints the R (primary specular) highlight  
    float specularTint = 0.0f;          // 0=white highlight (physical), 1=tinted by hair color
    
    // --- NEW: Diffuse Softness (Multiple Scattering Weight) ---
    float diffuseSoftness = 0.5f;       // 0=hard specular only, 1=strong diffuse/MS component
    
    // Custom Textures (Optional overrides)
    std::shared_ptr<Texture> customAlbedoTexture = nullptr;
    std::shared_ptr<Texture> customRoughnessTexture = nullptr;
};

void to_json(nlohmann::json& j, const HairMaterialParams& p);
void from_json(const nlohmann::json& j, HairMaterialParams& p);


/**
 * @brief Hair BSDF implementation
 * 
 * Uses Marschner model with 3 lobes:
 *   R  - Primary specular reflection
 *   TT - Transmission through fiber (soft glow)
 *   TRT - Internal reflection (secondary highlight)
 * 
 * Plus optional:
 *   TRRT - Additional internal bounce (for thick fibers)
 */
class HairBSDF {
public:
    /**
     * @brief Evaluate BSDF for given directions
     * 
     * @param wo Outgoing direction (toward camera)
     * @param wi Incoming direction (toward light)
     * @param tangent Hair tangent direction
     * @param params Material parameters
     * @param v Parametric position along strand (0=root, 1=tip)
     * @param h Offset from hair center (-1 to 1)
     * @return RGB BSDF value
     */
    static Vec3 evaluate(
        const Vec3& wo,
        const Vec3& wi,
        const Vec3& tangent,
        const HairMaterialParams& params,
        float v = 0.5f,
        float h = 0.0f
    );
    
    /**
     * @brief Sample incoming direction for path tracing
     * 
     * @param wo Outgoing direction
     * @param tangent Hair tangent
     * @param params Material parameters
     * @param random1 Random value [0,1)
     * @param random2 Random value [0,1)
     * @param outWi Sampled incoming direction
     * @param outPdf Probability density
     * @return BSDF value / pdf
     */
    static Vec3 sample(
        const Vec3& wo,
        const Vec3& tangent,
        const HairMaterialParams& params,
        float random1,
        float random2,
        Vec3& outWi,
        float& outPdf
    );
    
    /**
     * @brief Get PDF for given sample
     */
    static float pdf(
        const Vec3& wo,
        const Vec3& wi,
        const Vec3& tangent,
        const HairMaterialParams& params
    );
    
    /**
     * @brief Convert melanin to absorption coefficient
     */
    static Vec3 melaninToAbsorption(float melanin, float redness);
    
    /**
     * @brief Compute Fresnel for hair fiber
     */
    static float fresnel(float cosTheta, float ior);
    
    /**
     * @brief Convert to GPU-friendly material
     */
    static GpuHairMaterial convertToGpu(const HairMaterialParams& params);
    
private:
    // Marschner lobe evaluations
    static float evalM(float roughness, float sinThetaI, float sinThetaO, float cosThetaD);
    static float evalN(float roughness, float phi, float phiTarget);
    static Vec3 evalAbsorption(const Vec3& sigma, float cosGammaO, float cosGammaT);
    
    // Azimuthal functions
    static float phi_R(float gammaO, float gammaT);
    static float phi_TT(float gammaO, float gammaT);
    static float phi_TRT(float gammaO, float gammaT);
    
    // Gaussian distribution
    static float gaussian(float x, float stddev);
    
    // Logistic distribution (faster than Gaussian, used in Chiang 2016)
    static float logisticCDF(float x, float s);
    static float logisticPDF(float x, float s);
    static float sampleLogistic(float s, float u);


};

} // namespace Hair

#endif // HAIR_BSDF_H
