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

        // Logistic distribution for azimuthal scattering
        return logisticPDF(diff, s);
    }

    // Absorption for transmission through fiber
    Vec3 HairBSDF::evalAbsorption(const Vec3& sigma, float cosGammaO, float cosGammaT) {
        // Physical path length: 2 * R * cos(gammaT)
        float pathLength = 2.0f * std::abs(cosGammaT);

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
        return PI - 2.0f * gammaT;
    }

    float HairBSDF::phi_TRT(float gammaO, float gammaT) {
        return PI - 2.0f * (gammaO - gammaT);
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
        // wo/wi are directions TO light/eye.
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
        float alpha = params.cuticleAngle * PI / 180.0f;  // Slant angle
        
        // Use a slightly wider base roughness to avoid the "fast pop to matte"
        float baseRoughness = std::max(params.roughness, 0.08f);
        float eta = params.ior;

        // Get absorption coefficient
        Vec3 sigma;
        switch (params.colorMode) {
            case HairMaterialParams::ColorMode::DIRECT_COLORING:
                sigma = Vec3(
                    -std::log(std::max(0.001f, params.color.x)),
                    -std::log(std::max(0.001f, params.color.y)),
                    -std::log(std::max(0.001f, params.color.z))
                ) * 2.5f; 
                break;
            case HairMaterialParams::ColorMode::MELANIN:
                sigma = melaninToAbsorption(params.melanin, params.melaninRedness);
                break;
            case HairMaterialParams::ColorMode::ABSORPTION:
                sigma = params.absorptionCoefficient;
                break;
        }

        // h is the azimuthal offset [-1, 1]
        float gammaO = safeAsin(h);
        float gammaT = safeAsin(h / eta);

        float cosGammaO = std::cos(gammaO);
        float cosGammaT = std::cos(gammaT);

        // Calculate actual azimuthal difference phi between wi and wo
        Vec3 woPerp = (wo - tangent * sinThetaO).normalize();
        Vec3 wiPerp = (wi - tangent * sinThetaI).normalize();
        
        // Signed angle around tangent
        float cosPhi = std::clamp(Vec3::dot(woPerp, wiPerp), -1.0f, 1.0f);
        float sinPhi = Vec3::dot(Vec3::cross(woPerp, wiPerp), tangent);
        float phi = std::atan2(sinPhi, cosPhi);

        // ========================================================================
        // Component Weights (Frensel)
        // ========================================================================
        float F_R = fresnel(cosThetaO, eta);

        // R: Reflection Lobe (Primary Specular)
        float M_R = evalM(baseRoughness * 1.2f, sinThetaI, sinThetaO - 2.0f * alpha, cosThetaO);
        float N_R = evalN(baseRoughness * 1.8f, phi, phi_R(gammaO, gammaT));
        Vec3 R = Vec3(F_R * M_R * N_R, F_R * M_R * N_R, F_R * M_R * N_R);

        // TT: Transmission Lobe (Through hair)
        Vec3 A = evalAbsorption(sigma, cosGammaO, cosGammaT);
        float F_TT = (1.0f - F_R) * (1.0f - fresnel(cosGammaT, 1.0f / eta));
        float M_TT = evalM(baseRoughness * 0.6f, sinThetaI, sinThetaO + alpha, cosThetaO);
        float N_TT = evalN(baseRoughness * 1.2f, phi, phi_TT(gammaO, gammaT));
        Vec3 TT = A * (F_TT * M_TT * N_TT);

        // TRT: Internal Reflection (Secondary highlight)
        float F_TRT = (1.0f - F_R) * F_R * (1.0f - fresnel(cosGammaT, 1.0f / eta));
        float M_TRT = evalM(baseRoughness * 2.2f, sinThetaI, sinThetaO - 4.0f * alpha, cosThetaO);
        float N_TRT = evalN(baseRoughness * 2.2f, phi, phi_TRT(gammaO, gammaT));
        Vec3 TRT = (A * A) * (F_TRT * M_TRT * N_TRT);

        // Final result: Marschner model lobes
        // Standard normalization for cylindrical fibers involves 1/cos^2(theta_d)
        Vec3 bsdf = TT * 2.1f + TRT * 1.5f + R * 1.3f;

        // Apply coat layer (fur gloss)
        if (params.coat > 0.0f) {
            float coatF = fresnel(cosThetaO, 1.35f) * params.coat;
            bsdf = bsdf * (1.0f - coatF) + params.coatTint * coatF * 0.5f;
        }

        // Add emission
        if (params.emissionStrength > 0.0f) {
            bsdf = bsdf + params.emission * params.emissionStrength;
        }

        // Final normalization and protection
        // Cylindrical hair BSDF is normalized by cos^2(theta_d) * cos(theta_i) 
        // We use a robust clamp to avoid division by zero at grazing angles
        float denominator = std::max(cosSqThetaD * cosThetaI, 0.01f);
        bsdf = bsdf / denominator;
        
        // Firefly clamp (increased for high-range HDR)
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
    
    float roughness = params.roughness;
    float alpha = params.cuticleAngle * M_PI / 180.0f;
    
    // Get longitudinal parameters for chosen lobe
    float variance, alphaShift;
    switch (lobe) {
    case 0: variance = Hair::sqr(roughness); alphaShift = -2.0f * alpha; break;
        case 1: variance = Hair::sqr(roughness * 0.5f); alphaShift = alpha; break;
        case 2: variance = Hair::sqr(roughness * 2.0f); alphaShift = -4.0f * alpha; break;
        default: variance = Hair::sqr(roughness); alphaShift = 0.0f;
    }
    
    // Sample longitudinal
    float sinThetaO = Vec3::dot(wo, tangent);
    float sinThetaI = sinThetaO + alphaShift + std::sqrt(variance) * (2.0f * random1 - 1.0f);
    sinThetaI = std::clamp(sinThetaI, -1.0f, 1.0f);
    float cosThetaI = Hair::safeSqrt(1.0f - sinThetaI * sinThetaI);
    
    // Sample azimuthal
    float phiTarget = (lobe == 1) ? M_PI : 0.0f;
    float s = roughness * ((lobe == 2) ? 1.0f : 0.5f);
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
    
    // Compute PDF
    float M = gaussian(sinThetaI - sinThetaO - alphaShift, std::sqrt(variance));
    float N = logisticPDF(phi - phiTarget, s);
    outPdf = M * N * lobeWeights[lobe];
    
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
    
    float roughness = params.roughness;
    float alpha = params.cuticleAngle * PI / 180.0f;
    
    float lobeWeights[3] = {0.4f, 0.3f, 0.3f};
    float totalPdf = 0.0f;
    
    // Sum PDF over all lobes
    for (int lobe = 0; lobe < 3; ++lobe) {
        float variance, alphaShift, phiTarget, s;
        switch (lobe) {
            case 0: 
                variance = sqr(roughness); 
                alphaShift = -2.0f * alpha;
                phiTarget = 0.0f;
                s = roughness * 0.5f;
                break;
            case 1: 
                variance = sqr(roughness * 0.5f); 
                alphaShift = alpha;
                phiTarget = PI;
                s = roughness * 0.5f;
                break;
            case 2: 
                variance = sqr(roughness * 2.0f); 
                alphaShift = -4.0f * alpha;
                phiTarget = 0.0f;
                s = roughness;
                break;
            default:
                continue;
        }
        
        float M = gaussian(sinThetaI - sinThetaO - alphaShift, std::sqrt(variance));
        float N = logisticPDF(phi - phiTarget, s);
        totalPdf += M * N * lobeWeights[lobe];
    }
    
    return totalPdf;
}



    GpuHairMaterial HairBSDF::convertToGpu(const HairMaterialParams& params) {
        GpuHairMaterial gpu;
        
        // 1. Color Mode & Base Color
        gpu.colorMode = static_cast<int>(params.colorMode);
        gpu.color = make_float3(params.color.x, params.color.y, params.color.z);
        gpu.roughness = params.roughness;
        
        // 2. Physical Constants
        gpu.melanin = params.melanin;
        gpu.melaninRedness = params.melaninRedness;
        gpu.ior = params.ior;
        gpu.cuticleAngle = params.cuticleAngle * PI / 180.0f; // Slant to radians
        
        // 3. Styling & Tint
        gpu.tintColor = make_float3(params.tintColor.x, params.tintColor.y, params.tintColor.z);
        gpu.tint = params.tint;
        gpu.radialRoughness = params.radialRoughness;
        gpu.randomHue = params.randomHue;
        gpu.randomValue = params.randomValue;
        
        // 4. Emission
        gpu.emission = make_float3(params.emission.x, params.emission.y, params.emission.z);
        gpu.emissionStrength = params.emissionStrength;
        
        // 5. Calculate Absorption (Sigma_a)
        Vec3 sigma;
        switch (params.colorMode) {
            case HairMaterialParams::ColorMode::DIRECT_COLORING:
            case HairMaterialParams::ColorMode::ROOT_UV_MAP:
                sigma = Vec3(
                    -std::log(std::max(0.001f, params.color.x)),
                    -std::log(std::max(0.001f, params.color.y)),
                    -std::log(std::max(0.001f, params.color.z))
                ) * 2.5f; 
                break;
            case HairMaterialParams::ColorMode::MELANIN:
                sigma = Hair::HairBSDF::melaninToAbsorption(params.melanin, params.melaninRedness);
                break;
            case HairMaterialParams::ColorMode::ABSORPTION:
                sigma = params.absorptionCoefficient;
                break;
        }
        gpu.sigma_a = make_float3(sigma.x, sigma.y, sigma.z);
        
        // 6. Precompute Lobe Variances (Derived from longitudinal roughness)
        // Matches CPU evalM usages
        gpu.v_R = sqr(std::max(params.roughness, 0.08f) * 1.2f);
        gpu.v_TT = sqr(std::max(params.roughness, 0.08f) * 0.6f);
        gpu.v_TRT = sqr(std::max(params.roughness, 0.08f) * 2.2f);
        
        // Azimuthal width: logistic scale s
        gpu.s = std::max(params.roughness * 0.5f, 0.01f);
        
        // 7. Textures
        gpu.albedo_tex = params.customAlbedoTexture ? params.customAlbedoTexture->get_cuda_texture() : 0;
        gpu.roughness_tex = params.customRoughnessTexture ? params.customRoughnessTexture->get_cuda_texture() : 0;
        
        gpu.pad0 = 0.0f;
        
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
            {"emissionStrength", p.emissionStrength}
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

