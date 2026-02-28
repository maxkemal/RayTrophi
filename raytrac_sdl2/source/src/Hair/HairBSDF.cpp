/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          HairBSDF.cpp
 * Author:        Kemal Demirtaş
 * Description:   Marschner Hair BSDF Implementation (CPU)
 *                Physically-based hair/fur shading model
 * 
 * References:
 *   - Marschner et al. 2003 "Light Scattering from Human Hair Fibers"
 *   - d'Eon et al. 2011 "An Energy-Conserving Hair Reflectance Model"
 *   - Chiang et al. 2016 (Disney) "A Practical Hair Scattering Model"
 * =========================================================================
 */

#include "Hair/HairBSDF.h"
#include <cmath>
#include <algorithm>
#include <filesystem>
#include "ProjectManager.h"

namespace Hair {

    // ============================================================================
    // Constants
    // ============================================================================

    constexpr float PI = 3.14159265358979323846f;
    constexpr float TWO_PI = 6.28318530717958647692f;
    constexpr float INV_PI = 0.31830988618379067154f;
    constexpr float INV_TWO_PI = 0.15915494309189533577f;
    constexpr float SQRT_PI_INV = 0.56418958354775628695f;

    // ============================================================================
    // Helper Functions
    // ============================================================================

    static inline float sqr(float x) { return x * x; }

    static inline float safeSqrt(float x) { return std::sqrt(std::max(0.0f, x)); }

    static inline float safeAsin(float x) {
        return std::asin(std::clamp(x, -1.0f, 1.0f));
    }

    static inline float safePow(float x, float y) {
        return std::pow(std::max(0.0f, x), y);
    }

    // ============================================================================
    // Fresnel
    // ============================================================================

    float HairBSDF::fresnel(float cosTheta, float ior) {
        float sinThetaI = safeSqrt(1.0f - cosTheta * cosTheta);
        float sinThetaT = sinThetaI / ior;

        if (sinThetaT >= 1.0f) return 1.0f; // Total internal reflection

        float cosThetaT = safeSqrt(1.0f - sinThetaT * sinThetaT);

        float rs = (ior * cosTheta - cosThetaT) / (ior * cosTheta + cosThetaT);
        float rp = (cosTheta - ior * cosThetaT) / (cosTheta + ior * cosThetaT);

        return 0.5f * (rs * rs + rp * rp);
    }

    // ============================================================================
    // Gaussian & Logistic Distributions
    // ============================================================================

    float HairBSDF::gaussian(float x, float stddev) {
        float variance = stddev * stddev;
        return std::exp(-x * x / (2.0f * variance)) / (stddev * std::sqrt(2.0f * PI));
    }

    float HairBSDF::logisticCDF(float x, float s) {
        return 1.0f / (1.0f + std::exp(-x / s));
    }

    float HairBSDF::logisticPDF(float x, float s) {
        float expX = std::exp(-std::abs(x) / s);
        return expX / (s * sqr(1.0f + expX));
    }

    float HairBSDF::sampleLogistic(float s, float u) {
        return -s * std::log(1.0f / u - 1.0f);
    }

    // ============================================================================
    // Marschner Lobe Components
    // ============================================================================

    // Longitudinal scattering function (M term)
    float HairBSDF::evalM(float roughness, float sinThetaI, float sinThetaO, float cosThetaD) {
        float v = sqr(roughness);
        float sinSum = sinThetaI + sinThetaO;
        // Standard Gaussian distribution for longitudinal scattering
        return gaussian(sinSum, std::sqrt(v) + 1e-4f);
    }

    // Azimuthal scattering function (N term)
    float HairBSDF::evalN(float roughness, float phi, float phiTarget) {
        // Component width (s) relates to roughness
        float s = std::max(roughness * 0.5f, 0.01f);

        float diff = phi - phiTarget;
        while (diff > PI) diff -= TWO_PI;
        while (diff < -PI) diff += TWO_PI;

        // Energy conservation: Normalize the logistic distribution over [-PI, PI]
        // The integral of logistic PDF from -PI to PI is (CDF(PI) - CDF(-PI))
        float norm = (1.0f / (1.0f + std::exp(-PI / s))) - (1.0f / (1.0f + std::exp(PI / s)));
        
        // Logistic distribution for azimuthal scattering
        return logisticPDF(diff, s) / std::max(norm, 0.1f);
    }

    // Absorption for transmission through fiber
    Vec3 HairBSDF::evalAbsorption(const Vec3& sigma, float cosGammaT, float cosThetaD) {
        // Physical path length inside fiber: 2 * cos(gammaT) / cos(thetaD)
        float pathLength = 2.0f * std::abs(cosGammaT) / std::max(cosThetaD, 0.1f);

        return Vec3(
            std::exp(-sigma.x * pathLength),
            std::exp(-sigma.y * pathLength),
            std::exp(-sigma.z * pathLength)
        );
    }

    // ============================================================================
    // Azimuthal Angles for Each Lobe
    // ============================================================================

    float HairBSDF::phi_R(float gammaO, float gammaT) {
        (void)gammaT;
        return -2.0f * gammaO;
    }

    float HairBSDF::phi_TT(float gammaO, float gammaT) {
        return PI + 2.0f * gammaT - 2.0f * gammaO;
    }

    float HairBSDF::phi_TRT(float gammaO, float gammaT) {
        return 4.0f * gammaT - 2.0f * gammaO;
    }

    // ============================================================================
    // Melanin to Absorption Coefficient
    // ============================================================================

    Vec3 HairBSDF::melaninToAbsorption(float melanin, float redness) {
        // Eumelanin is the primary darkness. Redness introduces pheomelanin.
        // Instead of (1 - redness), use a softer blend so high melanin stays dark.
        float eumelanin = melanin * (1.0f - redness * 0.5f); 
        float pheomelanin = melanin * redness; 

        // Base absorption coefficients per mm (based on physical measurements)
        Vec3 eumelaninSigma(0.419f, 0.697f, 1.37f);     // Brownish
        Vec3 pheomelaninSigma(0.187f, 0.4f, 1.05f);     // Reddish

        // Total absorption (scaled for fiber scale)
        // Adjusted multipliers to match visual expectations
        return eumelaninSigma * eumelanin * 8.0f + pheomelaninSigma * pheomelanin * 8.0f;
    }

    // ============================================================================
    // Main BSDF Evaluation
    // ============================================================================

    Vec3 HairBSDF::evaluate(
        const Vec3& wo,
        const Vec3& wi,
        const Vec3& tangent,
        const HairMaterialParams& params,
        float u,
        float h
    ) {
        // Longitudinal angles (Theta: angle from normal to tangent)
        float sinThetaO = std::clamp(Vec3::dot(wo, tangent), -0.999f, 0.999f);
        float sinThetaI = std::clamp(Vec3::dot(wi, tangent), -0.999f, 0.999f);

        float cosThetaO = safeSqrt(1.0f - sinThetaO * sinThetaO);
        float cosThetaI = safeSqrt(1.0f - sinThetaI * sinThetaI);

        // Difference angle (Theta_D) and Half angle (Theta_H)
        float thetaO = std::asin(sinThetaO);
        float thetaI = std::asin(sinThetaI);
        float thetaD = (thetaO - thetaI) * 0.5f;
        float cosThetaD = std::cos(thetaD);
        float cosSqThetaD = cosThetaD * cosThetaD;

        // Material parameters
        float alpha = params.cuticleAngle * PI / 180.0f;
        float baseRoughness = std::max(params.roughness, 0.08f);
        float eta = params.ior;

        // Get absorption coefficient (root color)
        Vec3 sigma;
        switch (params.colorMode) {
            case HairMaterialParams::ColorMode::DIRECT_COLORING:
            case HairMaterialParams::ColorMode::ROOT_UV_MAP:
                sigma = Vec3(
                    -std::log(std::max(0.001f, params.color.x)),
                    -std::log(std::max(0.001f, params.color.y)),
                    -std::log(std::max(0.001f, params.color.z))
                ) * 0.5f; 
                break;
            case HairMaterialParams::ColorMode::MELANIN:
                sigma = melaninToAbsorption(params.melanin, params.melaninRedness);
                break;
            case HairMaterialParams::ColorMode::ABSORPTION:
                sigma = params.absorptionCoefficient;
                break;
        }

        // --- Root-to-Tip Gradient ---
        if (params.enableRootTipGradient) {
            Vec3 tipSigma(
                -std::log(std::max(0.001f, params.tipColor.x)) * 0.5f,
                -std::log(std::max(0.001f, params.tipColor.y)) * 0.5f,
                -std::log(std::max(0.001f, params.tipColor.z)) * 0.5f
            ) ;
            float t = u * params.rootTipBalance; // u is 0 at root, 1 at tip
            sigma = sigma * (1.0f - t) + tipSigma * t;
        }

        // h is the azimuthal offset [-1, 1]
        float gammaO = safeAsin(h);
        float gammaT = safeAsin(h / eta);

        float cosGammaO = std::cos(gammaO);
        float cosGammaT = std::cos(gammaT);

        // Calculate actual azimuthal difference phi between wi and wo
        Vec3 woPerp = (wo - tangent * sinThetaO).normalize();
        Vec3 wiPerp = (wi - tangent * sinThetaI).normalize();
        
        float cosPhi = std::clamp(Vec3::dot(woPerp, wiPerp), -1.0f, 1.0f);
        float sinPhi = Vec3::dot(Vec3::cross(woPerp, wiPerp), tangent);
        float phi = std::atan2(sinPhi, cosPhi);

        // ========================================================================
        // Fresnel
        // ========================================================================
        float F_R = fresnel(cosThetaO, eta);

        // R: Reflection Lobe (Primary Specular)
        float M_R = evalM(baseRoughness * 1.2f, sinThetaI, sinThetaO - 2.0f * alpha, cosThetaO);
        float N_R = evalN(baseRoughness * 1.8f, phi, phi_R(gammaO, gammaT));
        
        // --- Specular Tint: blend white highlight with hair body color ---
        Vec3 hairBodyColor(
            std::exp(-sigma.x * 0.5f),
            std::exp(-sigma.y * 0.5f),
            std::exp(-sigma.z * 0.5f)
        );
        Vec3 specColor = Vec3(1, 1, 1) * (1.0f - params.specularTint) + hairBodyColor * params.specularTint;
        Vec3 R = specColor * (F_R * M_R * N_R);

        // TT: Transmission Lobe (Through hair)
        Vec3 A = evalAbsorption(sigma, cosGammaT, cosThetaD);
        float F_TT = (1.0f - F_R) * (1.0f - fresnel(cosGammaT, 1.0f / eta));
        float M_TT = evalM(baseRoughness * 0.707f, sinThetaI, sinThetaO + alpha, cosThetaO);
        float N_TT = evalN(baseRoughness * 0.7f, phi, phi_TT(gammaO, gammaT));
        Vec3 TT = A * (F_TT * M_TT * N_TT);

        // TRT: Internal Reflection (Secondary highlight)
        float F_TRT = (1.0f - F_R) * fresnel(cosGammaT, 1.0f / eta) * (1.0f - fresnel(cosGammaT, 1.0f / eta));
        float M_TRT = evalM(baseRoughness * 1.414f, sinThetaI, sinThetaO - 4.0f * alpha, cosThetaO);
        float N_TRT = evalN(baseRoughness * 2.2f, phi, phi_TRT(gammaO, gammaT));
        Vec3 TRT = (A * A) * (F_TRT * M_TRT * N_TRT);

        // MS: Multiple Scattering / Cortex Diffusion (Bulk Body Color)
        float s_ms = std::max(baseRoughness * 0.7f * 10.0f, 0.2f);
        float N_MS = evalN(s_ms, phi, 0.0f); 
        // --- Diffuse Softness controls MS weight ---
        float msWeight = params.diffuseSoftness * 1.2f; // 0.5 default -> 0.6 (close to previous 0.6*0.8=0.48)
        Vec3 MS = (A * A) * (msWeight * N_MS);

        // Final result: Marschner model lobes
        Vec3 bsdf = R + TT + TRT + MS;

        // --- Apply Artistic Tint ---
        if (params.tint > 0.0f) {
            Vec3 tinted = bsdf * params.tintColor;
            bsdf = bsdf * (1.0f - params.tint) + tinted * params.tint;
        }

        // Apply coat layer (wet look / fur gloss)
        // Coat = additional specular R-like lobe from water/gel film on hair surface
        if (params.coat > 0.0f) {
            float coatIOR = 1.33f; // Water film IOR
            float coatFresnel = fresnel(cosThetaO, coatIOR) * params.coat;
            
            // Coat specular: narrow Gaussian lobe (smooth water surface)
            float coatRoughness = std::max(params.roughness * 0.3f, 0.02f); // Much smoother than hair
            float M_coat = evalM(coatRoughness, sinThetaI, sinThetaO, cosThetaD);
            float N_coat = evalN(coatRoughness * 0.8f, phi, phi_R(gammaO, gammaT));
            
            Vec3 coatSpec = params.coatTint * (coatFresnel * M_coat * N_coat);
            
            // Energy conservation: dim base BSDF by coat reflection
            bsdf = bsdf * (1.0f - coatFresnel) + coatSpec;
        }

        // Add emission
        if (params.emissionStrength > 0.0f) {
            bsdf = bsdf + params.emission * params.emissionStrength;
        }

        // Final normalization
        float denominator = std::max(cosSqThetaD, 0.001f);
        bsdf = bsdf / denominator;
        
        // Firefly clamp
        float maxVal = 100.0f;
        bsdf.x = std::min(bsdf.x, maxVal);
        bsdf.y = std::min(bsdf.y, maxVal);
        bsdf.z = std::min(bsdf.z, maxVal);

        return bsdf;
    }


// ============================================================================
// Importance Sampling
// ============================================================================

Vec3 HairBSDF::sample(
    const Vec3& wo,
    const Vec3& tangent,
    const HairMaterialParams& params,
    float random1,
    float random2,
    Vec3& outWi,
    float& outPdf
) {
    // Choose lobe
    float lobeWeights[3] = {0.4f, 0.3f, 0.3f};  // R, TT, TRT
    
    int lobe;
    if (random1 < lobeWeights[0]) {
        lobe = 0;
        random1 /= lobeWeights[0];
    } else if (random1 < lobeWeights[0] + lobeWeights[1]) {
        lobe = 1;
        random1 = (random1 - lobeWeights[0]) / lobeWeights[1];
    } else {
        lobe = 2;
        random1 = (random1 - lobeWeights[0] - lobeWeights[1]) / lobeWeights[2];
    }
    
    float baseRoughness = std::max(params.roughness, 0.08f);
    float alpha = params.cuticleAngle * M_PI / 180.0f;
    
    // Get longitudinal parameters for chosen lobe
    float variance, alphaShift;
    switch (lobe) {
        case 0: variance = sqr(baseRoughness * 1.2f); alphaShift = -2.0f * alpha; break;
        case 1: variance = 0.5f * sqr(baseRoughness); alphaShift = alpha; break;
        case 2: variance = 2.0f * sqr(baseRoughness); alphaShift = -4.0f * alpha; break;
        default: variance = sqr(baseRoughness); alphaShift = 0.0f;
    }
    
    // Sample longitudinal
    float sinThetaO = Vec3::dot(wo, tangent);
    float sinThetaI = sinThetaO + alphaShift + std::sqrt(variance) * (2.0f * random1 - 1.0f);
    sinThetaI = std::clamp(sinThetaI, -1.0f, 1.0f);
    float cosThetaI = Hair::safeSqrt(1.0f - sinThetaI * sinThetaI);
    
    // Sample azimuthal
    float phiTarget = (lobe == 1) ? M_PI : 0.0f;
    float s;
    if (lobe == 0) s = 0.9f * baseRoughness;
    else if (lobe == 1) s = 0.35f * baseRoughness;
    else s = 1.1f * baseRoughness;
    
    float phi = Hair::HairBSDF::sampleLogistic(s, random2) + phiTarget;
    
    // Reconstruct wi
    float cosThetaO = safeSqrt(1.0f - sinThetaO * sinThetaO);
    Vec3 woPerp = (wo - tangent * sinThetaO);
    float perpLen = woPerp.length();
    if (perpLen > 1e-4f) woPerp = woPerp / perpLen;
    else woPerp = Vec3(1, 0, 0);
    
    Vec3 bitangent = Vec3::cross(tangent, woPerp).normalize();
    
    outWi = tangent * sinThetaI + 
            woPerp * (cosThetaI * std::cos(phi)) +
            bitangent * (cosThetaI * std::sin(phi));
    outWi = outWi.normalize();
    
    // Compute PDF (with azimuthal normalization)
    float M = gaussian(sinThetaI - sinThetaO - alphaShift, std::sqrt(variance));
    float N = logisticPDF(phi - phiTarget, s);
    float norm = (1.0f / (1.0f + std::exp(-PI / s))) - (1.0f / (1.0f + std::exp(PI / s)));
    outPdf = M * (N / std::max(norm, 0.1f)) * lobeWeights[lobe];
    
    // Return BSDF value
    return evaluate(wo, outWi, tangent, params, 0.5f, 0.0f);
}

// ============================================================================
// PDF Evaluation
// ============================================================================

float HairBSDF::pdf(
    const Vec3& wo,
    const Vec3& wi,
    const Vec3& tangent,
    const HairMaterialParams& params
) {
    float sinThetaO = Vec3::dot(wo, tangent);
    float sinThetaI = Vec3::dot(wi, tangent);
    
    Vec3 woPerp = (wo - tangent * sinThetaO).normalize();
    Vec3 wiPerp = (wi - tangent * sinThetaI).normalize();
    
    float cosPhi = Vec3::dot(woPerp, wiPerp);
    float phi = std::acos(std::clamp(cosPhi, -1.0f, 1.0f));
    
    float baseRoughness = std::max(params.roughness, 0.08f);
    float alpha = params.cuticleAngle * M_PI / 180.0f;
    
    float lobeWeights[3] = {0.4f, 0.3f, 0.3f};
    float totalPdf = 0.0f;
    
    for (int lobe = 0; lobe < 3; ++lobe) {
        float variance, alphaShift, s;
        float phiTarget = (lobe == 1) ? M_PI : 0.0f; // Determine phiTarget based on lobe
        switch (lobe) {
            case 0: variance = sqr(baseRoughness * 1.2f); alphaShift = -2.0f * alpha; s = 0.9f * baseRoughness; break;
            case 1: variance = 0.5f * sqr(baseRoughness); alphaShift = alpha; s = 0.35f * baseRoughness; break;
            case 2: variance = 2.0f * sqr(baseRoughness); alphaShift = -4.0f * alpha; s = 1.1f * baseRoughness;  break;
            default: variance = sqr(baseRoughness); alphaShift = 0.0f; s = 0.5f * baseRoughness; break; // Should not happen
        }
        
        float M = gaussian(sinThetaI - sinThetaO - alphaShift, std::sqrt(variance));
        float N_val = logisticPDF(phi - phiTarget, s);
        float norm = (1.0f / (1.0f + std::exp(-M_PI / s))) - (1.0f / (1.0f + std::exp(M_PI / s))); // Use M_PI
        totalPdf += M * (N_val / std::max(norm, 0.1f)) * lobeWeights[lobe];
    }
    
    return totalPdf;
}



    GpuHairMaterial HairBSDF::convertToGpu(const HairMaterialParams& params) {
        GpuHairMaterial gpu = {};
        
        // 1. Color Mode & Base Color
        gpu.colorMode = static_cast<int>(params.colorMode);
        gpu.color = make_float3(params.color.x, params.color.y, params.color.z);
        gpu.roughness = params.roughness;
        
        // 2. Physical Constants
        gpu.melanin = params.melanin;
        gpu.melaninRedness = params.melaninRedness;
        gpu.ior = params.ior;
        gpu.cuticleAngle = params.cuticleAngle * PI / 180.0f;
        
        // 3. Styling & Tint
        gpu.tintColor = make_float3(params.tintColor.x, params.tintColor.y, params.tintColor.z);
        gpu.tint = params.tint;
        gpu.radialRoughness = params.radialRoughness;
        gpu.randomHue = params.randomHue;
        gpu.randomValue = params.randomValue;
        
        // 4. Emission
        gpu.emission = make_float3(params.emission.x, params.emission.y, params.emission.z);
        gpu.emissionStrength = params.emissionStrength;
        
        // 5. Calculate Root Absorption (Sigma_a)
        Vec3 sigma;
        switch (params.colorMode) {
            case HairMaterialParams::ColorMode::DIRECT_COLORING:
            case HairMaterialParams::ColorMode::ROOT_UV_MAP:
                sigma = Vec3(
                    -std::log(std::max(0.001f, params.color.x)),
                    -std::log(std::max(0.001f, params.color.y)),
                    -std::log(std::max(0.001f, params.color.z))
                ) * 0.5f; 
                break;
            case HairMaterialParams::ColorMode::MELANIN:
                sigma = Hair::HairBSDF::melaninToAbsorption(params.melanin, params.melaninRedness);
                break;
            case HairMaterialParams::ColorMode::ABSORPTION:
                sigma = params.absorptionCoefficient;
                break;
        }
        gpu.sigma_a = make_float3(sigma.x, sigma.y, sigma.z);
        
        // 6. Precompute Lobe Variances (Longitudinal)
        float baseR = std::max(params.roughness, 0.08f);
        gpu.v_R   = sqr(baseR * 1.2f);
        gpu.v_TT  = sqr(baseR);
        gpu.v_TRT = sqr(baseR * 1.0f);
        
        // 7. Precompute Azimuthal Widths (Logistic scale s)
        gpu.s_R   = baseR * 1.8f * 0.5f;
        gpu.s_TT  = baseR * 0.7f * 0.5f;
        gpu.s_TRT = baseR * 2.2f * 0.5f;
        gpu.s_MS  = std::max(baseR * 0.7f * 10.0f, 0.2f) * 0.5f;
        
        // 8. Textures
        gpu.albedo_tex = params.customAlbedoTexture ? params.customAlbedoTexture->get_cuda_texture() : 0;
        gpu.roughness_tex = params.customRoughnessTexture ? params.customRoughnessTexture->get_cuda_texture() : 0;
        
        // 9. NEW: Coat parameters
        gpu.coat = params.coat;
        gpu.coatTint = make_float3(params.coatTint.x, params.coatTint.y, params.coatTint.z);
        
        // 10. NEW: Specular Tint & Diffuse Softness
        gpu.specularTint = params.specularTint;
        gpu.diffuseSoftness = params.diffuseSoftness;
        
        // 11. NEW: Root-Tip Gradient
        gpu.enableRootTipGradient = params.enableRootTipGradient ? 1 : 0;
        gpu.rootTipBalance = params.rootTipBalance;
        Vec3 tipSigma(
            -std::log(std::max(0.001f, params.tipColor.x)) * 0.5f,
            -std::log(std::max(0.001f, params.tipColor.y)) * 0.5f,
            -std::log(std::max(0.001f, params.tipColor.z)) * 0.5f
        );
        gpu.tipSigma = make_float3(tipSigma.x, tipSigma.y, tipSigma.z);
        
        // 12. Padding
        gpu.pad1 = 0.0f;
        gpu.pad2 = 0.0f;
        gpu.pad3 = 0.0f;
        
        return gpu;
    }

    // ============================================================================
    // Serialization
    // ============================================================================

    void to_json(nlohmann::json& j, const HairMaterialParams& p) {
        j = nlohmann::json{
            {"colorMode", static_cast<int>(p.colorMode)},
            {"color", {p.color.x, p.color.y, p.color.z}},
            {"melanin", p.melanin},
            {"melaninRedness", p.melaninRedness},
            {"tint", p.tint},
            {"tintColor", {p.tintColor.x, p.tintColor.y, p.tintColor.z}},
            {"absorptionCoefficient", {p.absorptionCoefficient.x, p.absorptionCoefficient.y, p.absorptionCoefficient.z}},
            {"roughness", p.roughness},
            {"radialRoughness", p.radialRoughness},
            {"cuticleAngle", p.cuticleAngle},
            {"ior", p.ior},
            {"randomHue", p.randomHue},
            {"randomValue", p.randomValue},
            {"coat", p.coat},
            {"coatTint", {p.coatTint.x, p.coatTint.y, p.coatTint.z}},
            {"emission", {p.emission.x, p.emission.y, p.emission.z}},
            {"emissionStrength", p.emissionStrength},
            // NEW parameters
            {"enableRootTipGradient", p.enableRootTipGradient},
            {"tipColor", {p.tipColor.x, p.tipColor.y, p.tipColor.z}},
            {"rootTipBalance", p.rootTipBalance},
            {"specularTint", p.specularTint},
            {"diffuseSoftness", p.diffuseSoftness}
        };
        if (p.customAlbedoTexture && !p.customAlbedoTexture->name.empty()) {
            j["customAlbedoTexture"] = p.customAlbedoTexture->name;
        }
        if (p.customRoughnessTexture && !p.customRoughnessTexture->name.empty()) {
            j["customRoughnessTexture"] = p.customRoughnessTexture->name;
        }
    }

    void from_json(const nlohmann::json& j, HairMaterialParams& p) {
        if (j.contains("colorMode")) p.colorMode = static_cast<HairMaterialParams::ColorMode>(j["colorMode"].get<int>());
        if (j.contains("color")) p.color = Vec3(j["color"][0], j["color"][1], j["color"][2]);
        p.melanin = j.value("melanin", 0.8f);
        p.melaninRedness = j.value("melaninRedness", 0.5f);
        p.tint = j.value("tint", 0.0f);
        if (j.contains("tintColor")) p.tintColor = Vec3(j["tintColor"][0], j["tintColor"][1], j["tintColor"][2]);
        if (j.contains("absorptionCoefficient")) p.absorptionCoefficient = Vec3(j["absorptionCoefficient"][0], j["absorptionCoefficient"][1], j["absorptionCoefficient"][2]);
        p.roughness = j.value("roughness", 0.3f);
        p.radialRoughness = j.value("radialRoughness", 0.3f);
        p.cuticleAngle = j.value("cuticleAngle", 2.0f);
        p.ior = j.value("ior", 1.55f);
        p.randomHue = j.value("randomHue", 0.0f);
        p.randomValue = j.value("randomValue", 0.0f);
        p.coat = j.value("coat", 0.0f);
        if (j.contains("coatTint")) p.coatTint = Vec3(j["coatTint"][0], j["coatTint"][1], j["coatTint"][2]);
        if (j.contains("emission")) p.emission = Vec3(j["emission"][0], j["emission"][1], j["emission"][2]);
        p.emissionStrength = j.value("emissionStrength", 0.0f);

        // NEW parameters (backward compatible defaults)
        p.enableRootTipGradient = j.value("enableRootTipGradient", false);
        if (j.contains("tipColor")) p.tipColor = Vec3(j["tipColor"][0], j["tipColor"][1], j["tipColor"][2]);
        p.rootTipBalance = j.value("rootTipBalance", 0.5f);
        p.specularTint = j.value("specularTint", 0.0f);
        p.diffuseSoftness = j.value("diffuseSoftness", 0.5f);

        // Texture Restoration
        auto loadTex = [&](const std::string& key, std::shared_ptr<Texture>& tex, TextureType type) {
            if (j.contains(key) && !j[key].get<std::string>().empty()) {
                std::string path = j[key].get<std::string>();
                
                // 1. Check embedded
                auto& pm = ProjectManager::getInstance();
                auto* embedded = pm.getEmbeddedTexture(path);
                if (embedded) {
                    tex = std::make_shared<Texture>(embedded->data, type, path);
                    return;
                }
                
                // 2. Check disk
                if (std::filesystem::exists(path)) {
                    tex = std::make_shared<Texture>(path, type);
                } else {
                    // Try relative to project file
                    std::string projectPath = pm.getCurrentFilePath();
                    if (!projectPath.empty()) {
                        std::string dir = std::filesystem::path(projectPath).parent_path().string();
                        std::string fullPath = dir + "/" + path;
                        if (std::filesystem::exists(fullPath)) {
                            tex = std::make_shared<Texture>(fullPath, type);
                        }
                    }
                }
            }
        };

        loadTex("customAlbedoTexture", p.customAlbedoTexture, TextureType::Albedo);
        loadTex("customRoughnessTexture", p.customRoughnessTexture, TextureType::Roughness);
    }

} // namespace Hair

