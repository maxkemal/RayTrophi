/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          PrincipledBSDF.h
* Author:        Kemal Demirtas
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once

#include "Material.h"
#include "Texture.h"
#include "Vec2.h"
#include "Ray.h"
#include "Hittable.h"
#include <memory>

class PrincipledBSDF : public Material {
public:
    struct SurfaceDepositionSettings {
        bool enabled = false;
        float thickness_scale = 1.0f;
        float max_thickness = 0.05f;
        float hit_offset_scale = 1.0f;
    };

    struct TextureTransform {
        Vec2 scale;
        float rotation_degrees;
        Vec2 translation;
        Vec2 tilingFactor;
        WrapMode wrapMode;

        TextureTransform(
            const Vec2& scale = Vec2(1.0, 1.0),
            float rotation = 0.0f,
            const Vec2& translation = Vec2(0.0, 0.0),
            const Vec2& tiling = Vec2(1.0, 1.0),
            WrapMode wrap = WrapMode::Repeat
        ) : scale(scale), rotation_degrees(rotation), translation(translation),
            tilingFactor(tiling), wrapMode(wrap) {
        }
    };


    // PrincipledBSDF(std::shared_ptr<Texture> tex) : texture(tex) {}
     // Existing constructors
    PrincipledBSDF(
        const Vec3& albedo = Vec3(1, 1, 1),
        float roughness = 0.5f,
        float metallic = 0.0f,
        const std::shared_ptr<Texture>& albedoTexture = nullptr,
        const std::shared_ptr<Texture>& roughnessTexture = nullptr,
        const std::shared_ptr<Texture>& metallicTexture = nullptr,
        const std::shared_ptr<Texture>& normalTexture = nullptr,
        const std::shared_ptr<Texture>& heightTexture = nullptr,
        const std::shared_ptr<Texture>& opacityTexture = nullptr,
        const TextureTransform& transform = TextureTransform(),
        const Vec3& emission = Vec3(0.0f, 0.0f, 0.0f),
        float subsurface_in = 0.0f,
        const Vec3& subsurfaceColor_in = Vec3(1.0f, 0.8f, 0.6f),
        float subsurfaceRadius_in = 0.0f,
        float clearcoat_in = 0.0f,
        float transmission_in = 0.0f,
        float translucent_in = 0.0f,
        float clearcoatRoughness_in = 0.03f
    ) : transmission(transmission_in),
        translucent(translucent_in),
        textureTransform(transform)
    {
        if (subsurface_in > 0.0f || subsurfaceRadius_in > 0.0f) {
            sss_ext = std::make_unique<SubsurfaceExtension>();
            sss_ext->subsurface = subsurface_in;
            sss_ext->subsurfaceColor = subsurfaceColor_in;
            sss_ext->subsurfaceRadius = Vec3(subsurfaceRadius_in);
        }
        if (clearcoat_in > 0.0f || clearcoatRoughness_in != 0.03f) {
            clearcoat_ext = std::make_unique<ClearcoatExtension>();
            clearcoat_ext->clearcoat = clearcoat_in;
            clearcoat_ext->clearcoatRoughness = clearcoatRoughness_in;
        }
        // Initialize base class properties
        albedoProperty = MaterialProperty(albedo, 1.0f, albedoTexture);
        roughnessProperty = MaterialProperty(Vec3(roughness), 1.0f, roughnessTexture);
        metallicProperty = MaterialProperty(Vec3(metallic), metallic, metallicTexture);
        normalProperty = MaterialProperty(Vec3(0.5f, 0.5f, 1.0f), 1.0f, normalTexture);
        specularProperty = MaterialProperty(Vec3(1.0f), 0.5f);
        heightProperty = MaterialProperty(Vec3(0.5f), 1.0f, heightTexture); // 0.5 = mid-gray
        opacityProperty = MaterialProperty(Vec3(1.0f), 1.0f, opacityTexture, 1.0f);
        emissionProperty = MaterialProperty(emission, 0.0f);
        
        static std::once_flag flag;
        std::call_once(flag, precomputeLUT);
    }

    PrincipledBSDF(const PrincipledBSDF& other);
    PrincipledBSDF& operator=(const PrincipledBSDF& other);

    virtual bool hasTexture() const override;
    virtual float getIndexOfRefraction() const override;
    virtual std::shared_ptr<Texture> getTexture() const override;
    bool useSmartUVProjection = false; // Yeni üye
    MaterialType type() const override {
        return MaterialType::PrincipledBSDF;
    }

    bool has_normal_map() const override { return normalProperty.texture != nullptr; }
    Vec3 get_normal_from_map(float u, float v) const override;
    float get_normal_strength() const override { return normalStrength; }
    virtual void set_normal_strength(float norm) override {
        normalStrength = norm;
    }
    void setTextureTransform(const TextureTransform& transform);
    void setTilingFactor(const Vec2& factor) { tilingFactor = factor; }
    void set_normal_map(std::shared_ptr<Texture> normalMap, float normalStrength = 1.0f);

    // New methods for PBR
    void setSpecular(const Vec3& specular, float intensity = 1.0f);
    void setSpecularTexture(const std::shared_ptr<Texture>& tex, float intensity = 1.0f);
    void setEmission(const Vec3& emission, const float intensity);
    void setEmissionTexture(const std::shared_ptr<Texture>& tex, float intensity);
    void setAnisotropic(float anisotropic, const Vec3& anisotropicDirection);
    float getTransmission(const Vec2& uv) const;
    void setSubsurfaceScattering(const Vec3& sssColor, Vec3 sssRadius);
    Vec3 getTextureColor(float u, float v) const;
    virtual bool hasOpacityTexture() const override;
    Vec2 applyTextureTransform(float u, float v) const override;
    void setMetallic(float metallic, float intensity = 1.0f);
    void setMetallicTexture(const std::shared_ptr<Texture>& tex, float intensity = 1.0f);
    Vec3 evalSpecular(const Vec3& N, const Vec3& V, const Vec3& L, const Vec3& F0, float roughness) const;
    // Yeni metodlar
    void setOpacityTexture(const std::shared_ptr<Texture>& tex, float intensity = 1.0f);
    float get_roughness(float u, float v) const;
    float getRoughnessValue(const Vec2& uv) const;
    float getMetallicValue(const Vec2& uv) const;
    float getScalarRoughness() const;
    float getScalarMetallic() const;
    void setTransmission(float transmission, float ior);
   
    float getIOR() const;
    bool isEmissive() const;
    virtual float get_opacity(const Vec2& uv) const override;
    virtual bool scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered, bool& is_specular) const override;
    float pdf(const HitRecord& rec, const Vec3& incoming, const Vec3& outgoing) const override;
    float get_scattering_factor() const override {
        // Example formula incorporating reflectivity and roughness
        return 0.01f;
    }
    Vec3 getEmission(const Vec2& uv, const Vec3& p) const {
        (void)p;
        // Vulkan: payload.radiance = emColor * emStrength
        // OptiX:  gpuMat.emission = color * intensity (pre-multiplied at upload)
        // CPU must match: return color * intensity
        Vec2 sampleUv = applyTextureTransform(uv.u, uv.v);
        if (emissionProperty.texture) {
            return emissionProperty.texture->get_color_bilinear(sampleUv.u, sampleUv.v) * emissionProperty.intensity;
        }
        return emissionProperty.color * emissionProperty.intensity;
    }
    TextureTransform textureTransform;
    int selected_uv_set = 0;
    // MaterialProperty transmissionProperty; // Shadowing removed
    Vec3 albedoValue;
    Vec3 getPropertyValue(const MaterialProperty& prop, const Vec2& uv) const;
    float getSpecularValue(const Vec2& uv) const;
    Vec3 computeFresnel(const Vec3& F0, float cosTheta) const;
    Vec3 computeClearcoat(const Vec3& R, const Vec3& L, const Vec3& N) const;
    
    // Scalar BRDF Helpers
    float DistributionGGX(const Vec3& N, const Vec3& H, float roughness) const;
    float GeometrySchlickGGX(float NdotV, float roughness) const;
    float GeometrySmith(const Vec3& N, const Vec3& V, const Vec3& L, float roughness) const;
    Vec3 fresnelSchlick(float cosTheta, const Vec3& F0) const;
    Vec3 fresnelSchlickRoughness(float cosTheta, const Vec3& F0, float roughness) const;

    // GPU-parity scatter lobes
    bool clearcoat_scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered) const;
    bool sss_random_walk_scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered) const;
    bool translucent_scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered) const;

    // float normalStrength = 1.0f; // Shadowing removed
        // Override virtual getters and setters for memory-saving extensions
    float getClearcoat() const override;
    void setClearcoat(float cc, float roughness = 0.03f) override;
    float getClearcoatRoughness() const override;
    float getClearcoatIridescence() const override;
    void setClearcoatIridescence(float val) override;
    float getClearcoatFilmThickness() const override;
    void setClearcoatFilmThickness(float val) override;

    float getSubsurface() const override;
    void setSubsurface(float val) override;
    Vec3 getSubsurfaceColor() const override;
    void setSubsurfaceColor(const Vec3& val) override;
    Vec3 getSubsurfaceRadius() const override;
    void setSubsurfaceRadius(const Vec3& val) override;
    float getSubsurfaceScale() const override;
    void setSubsurfaceScale(float val) override;
    float getSubsurfaceAnisotropy() const override;
    void setSubsurfaceAnisotropy(float val) override;
    float getSubsurfaceIOR() const override;
    void setSubsurfaceIOR(float val) override;
    bool getUseRandomWalkSSS() const override;
    void setUseRandomWalkSSS(bool val) override;
    int getSssMaxSteps() const override;
    void setSssMaxSteps(int val) override;

    float getTransmissionDensity() const override;
    void setTransmissionDensity(float val) override;
    Vec3 getResinColor() const override;
    void setResinColor(const Vec3& val) override;
    float getResinRoughness() const override;
    void setResinRoughness(float val) override;
    float getResinInclusion() const override;
    void setResinInclusion(float val) override;
    float getResinDirt() const override;
    void setResinDirt(float val) override;
    float getResinInclusionScale() const override;
    void setResinInclusionScale(float val) override;
    Vec3 getResinDirtColor() const override;
    void setResinDirtColor(const Vec3& val) override;
    bool getGlassMarbleVolume() const override;
    void setGlassMarbleVolume(bool val) override;

    bool getIsBubble() const override;
    void setIsBubble(bool val) override;
    float getBubbleIor() const override;
    void setBubbleIor(float val) override;
    float getBubbleFilm() const override;
    void setBubbleFilm(float val) override;

    // Advanced Material extensions (lazy allocated)
    struct SubsurfaceExtension {
        Vec3 subsurfaceColor = Vec3(1.0f, 0.8f, 0.6f);     // SSS tint color
        Vec3 subsurfaceRadius = Vec3(1.0f, 0.2f, 0.1f);    // Per-channel scatter distance
        float subsurfaceScale = 0.05f;                      // Global radius multiplier
        float subsurfaceAnisotropy = 0.0f;                  // Scatter direction bias
        float subsurfaceIOR = 1.4f;                         // Internal SSS IOR
        float subsurface = 0.0f;                            // SSS amount (0-1)
        bool useRandomWalkSSS = true;                       // Enable bounded SSS
        int sssMaxSteps = 6;                                // Max internal scattering steps
    };
    
    struct ClearcoatExtension {
        float clearcoat = 0.0f;                             // Clear coat amount (0-1)
        float clearcoatRoughness = 0.03f;                   // Clear coat roughness
        float clearcoatIridescence = 0.0f;                  // thin-film tint strength (0-1)
        float clearcoatFilmThickness = 0.55f;               // hue cycle / film thickness
    };

    struct ResinExtension {
        float transmissionDensity = 0.0f;                  // interior absorption density
        Vec3 resinColor = Vec3(1.0f, 1.0f, 1.0f);           // absorption color
        float resinRoughness = 0.1f;                        // resin coat gloss
        float resinInclusion = 0.0f;                        // dust cloudiness amount
        float resinDirt = 0.0f;                             // opaque dirt-speck amount
        float resinInclusionScale = 8.0f;                  // procedural feature size
        Vec3 resinDirtColor = Vec3(0.18f, 0.14f, 0.10f);    // speck tint
        bool glassMarbleVolume = false;                     // Glass Marble FULL VOLUME
    };

    struct BubbleExtension {
        bool isBubble = false;                              // Thin-shell BUBBLE mode
        float bubbleIor = 1.33f;                            // rim Fresnel IOR
        float bubbleFilm = 0.0f;                            // thin-film interference strength
    };

    std::unique_ptr<SubsurfaceExtension> sss_ext;
    std::unique_ptr<ClearcoatExtension> clearcoat_ext;
    std::unique_ptr<ResinExtension> resin_ext;
    std::unique_ptr<BubbleExtension> bubble_ext;

    // Base PBR properties kept directly in the class for fast access
    float translucent = 0.0f;                           // Translucency amount (0-1)
    float anisotropic = 0.0f;                           // Surface anisotropy
    float transmission = 0.0f;                          // Glass/water transmission
    float dispersion = 0.0f;                            // Spectral dispersion (Abbe-style, 0 = off)
    // Packed-texture channel selection for the metallic/roughness texture slots:
    // 0 = Auto (ORM convention: roughness .g / metallic .b; BC4/R8 uploads .r),
    // 1 = R, 2 = G, 3 = B. Non-ORM packings (RMA/MRA, DirectX metal-in-R maps)
    // read the wrong channel under Auto — classic symptom: surface shades as
    // full metal even where the authored metallic value is 0.
    int metallic_tex_channel = 0;
    int roughness_tex_channel = 0;
    float sheen = 0.0f;                                 // Sheen amount (used as IS_WATER flag)
    float sheen_tint = 0.5f;                            // Sheen color tint
    SurfaceDepositionSettings surface_deposition;
    Vec3 anisotropicDirection;
    float opacityAlpha = 1.0f;

    // Procedural surface detail
    float micro_detail_strength = 0.0f;  // 0 = disabled
    float micro_detail_scale    = 2.0f;  // World-space frequency
    float tile_break_strength   = 0.0f;  // UV tile-break
    Vec3 sampleGGXVNDF(const Vec3& N, const Vec3& V, float roughness, float u1, float u2) const;
private:
  

    // Helper methods
    float max(float a, float b) const { return a > b ? a : b; }
    UVData transformUV(float u, float v) const;
    Vec2 applyWrapMode(const UVData& uvData) const;
    Vec2 applyPlanarWrapping(double u, double v) const;

    Vec2 applyRepeatWrapping(const Vec2& uv) const;
    Vec2 applyMirrorWrapping(const Vec2& uv) const;
    Vec2 applyClampWrapping(const Vec2& uv) const;
    Vec2 applyPlanarWrapping(const Vec2& uv) const;
    Vec2 applyCubicWrapping(const Vec2& uv) const;

  

    Vec3 evaluateBRDF(const Vec3& N, const Vec3& V, const Vec3& L, float roughness, float metallic, const Vec3& baseColor) const;

    float calculate_sss_density(float distance) const;

    float calculate_sss_absorption(float distance) const;

    Vec3 calculate_sss_attenuation(float distance) const;

    Vec3 sample_henyey_greenstein(const Vec3& wi, float g) const;
   

    Vec3 computeScatterDirection(const Vec3& N, const Vec3& T, const Vec3& B, float roughness) const;
    void createCoordinateSystem(const Vec3& N, Vec3& T, Vec3& B) const; 
    

  
    Vec2 applyTiling(float u, float v) const;
    Vec3 importanceSampleGGX(float u1, float u2, float roughness, const Vec3& N) const;
  

  

    // New helper methods for PBR
    Vec3 computeSubsurfaceScattering(const Vec3& N, const Vec3& V, const Vec3& subsurfaceRadius, float thickness) const;
    Vec3 evaluateIBLSpecular(const Vec3& N, const Vec3& V, float roughness, float metallic, const Vec3& baseColor, const Vec3& irradiance, const Vec3& prefilteredColor, float brdfLUT) const;
    Vec3 computeAnisotropicDirection(const Vec3& N, const Vec3& T, const Vec3& B, float roughness, float anisotropy) const;
    static float sqrtTable[256];
    static float cosTable[256];
    static float sinTable[256];

    static bool lutInitialized;  // İlk başlatma kontrolü

    static void precomputeLUT(); // LUT'yi başlatan fonksiyon
};



// Refreshed
