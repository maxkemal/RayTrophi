#pragma once

#include "Material.h"
#include "Texture.h"
#include "Vec2.h"
#include "Vec3SIMD.h"
#include "Ray.h"
#include "Hittable.h"
#include <memory>

class PrincipledBSDF : public Material {
public:
    struct TextureTransform {
        Vec2 scale;
        double rotation_degrees;
        Vec2 translation;
        Vec2 tilingFactor;
        WrapMode wrapMode;

        TextureTransform(
            const Vec2& scale = Vec2(1.0, 1.0),
            double rotation = 0.0,
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
        float roughness = 0.001f,
        float metallic = 0.0f,
        const std::shared_ptr<Texture>& albedoTexture = nullptr,
        const std::shared_ptr<Texture>& roughnessTexture = nullptr,
        const std::shared_ptr<Texture>& metallicTexture = nullptr,
        const std::shared_ptr<Texture>& normalTexture = nullptr,
        const std::shared_ptr<Texture>& opacityTexture = nullptr,
        const TextureTransform& transform = TextureTransform(),
        const Vec3& emission = Vec3(0.0f, 0.0f, 0.0f),
        const Vec3& subsurfaceColor = Vec3(0.0, 0.0, 0.0),
        float subsurfaceRadius = 0.0f,
        float clearcoat = 0.0f,
        float transmission = 0.0f,  // 💡 yeni parametre
        float clearcoatRoughness = 0.0f
    ) : albedoProperty(albedo, 1.0f, albedoTexture),
        roughnessProperty(Vec3(roughness), 1.0f, roughnessTexture),
        metallicProperty(Vec3(metallic), 1.0f, metallicTexture),
        normalProperty(Vec3(0.5f, 0.5f, 1.0f), 1.0f, normalTexture),
        opacityProperty(Vec3(1.0f), 1.0f, opacityTexture, opacityAlpha)        ,
        emissionProperty(emission, 0.0f),
        subsurfaceColor(subsurfaceColor),
        subsurfaceRadius(subsurfaceRadius),
        clearcoat(clearcoat),
        transmission(transmission),  // Eski sabit değer (isteğe bağlı)
        clearcoatRoughness(clearcoatRoughness),
        textureTransform(transform)
    {
        static std::once_flag flag;
        std::call_once(flag, precomputeLUT);
    }


    virtual bool hasTexture() const override;
    virtual double getIndexOfRefraction() const override;
    virtual std::shared_ptr<Texture> getTexture() const override;
    bool useSmartUVProjection = false; // Yeni üye
    MaterialType type() const override {
        return MaterialType::PrincipledBSDF;
    }

    bool has_normal_map() const override { return normalProperty.texture != nullptr; }
    Vec3 get_normal_from_map(double u, double v) const override;
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
    void setClearcoat(float clearcoat, float clearcoatRoughness = 0.1f);
    void setAnisotropic(float anisotropic, const Vec3& anisotropicDirection);
    float getTransmission(const Vec2& uv) const;
    void setSubsurfaceScattering(const Vec3& sssColor, Vec3 sssRadius);
    Vec3 getTextureColor(double u, double v) const;
    virtual bool hasOpacityTexture() const override;
    Vec2 applyTextureTransform(double u, double v) const;
    void setMetallic(float metallic, float intensity = 1.0f);
    void setMetallicTexture(const std::shared_ptr<Texture>& tex, float intensity = 1.0f);
    Vec3 evalSpecular(const Vec3& N, const Vec3& V, const Vec3& L, const Vec3& F0, float roughness) const;
    // Yeni metodlar
    void setOpacityTexture(const std::shared_ptr<Texture>& tex, float intensity = 1.0f);
    float get_roughness(float u, float v) const;
    void setTransmission(float transmission, float ior);
   
    float getIOR() const;
    bool isEmissive() const;
    virtual float get_opacity(const Vec2& uv) const override;
    virtual bool scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered) const override;
    float pdf(const HitRecord& rec, const Vec3& incoming, const Vec3& outgoing) const override;
    float get_scattering_factor() const override {
        // Example formula incorporating reflectivity and roughness
        return 0.01f;
    }
    Vec3 getEmission(const Vec2& uv, const Vec3& p) const {
        return getPropertyValue(emissionProperty, uv);
    }
    MaterialProperty opacityProperty;
    MaterialProperty albedoProperty;
    MaterialProperty roughnessProperty;
    MaterialProperty metallicProperty;
    MaterialProperty normalProperty;
    Vec2 tilingFactor;
    TextureTransform textureTransform;
    MaterialProperty specularProperty;
    MaterialProperty emissionProperty;
    MaterialProperty transmissionProperty;
    Vec3 albedoValue;
    Vec3 getPropertyValue(const MaterialProperty& prop, const Vec2& uv) const;
    Vec3 computeFresnel(const Vec3& F0, float cosTheta) const;
    Vec3 computeClearcoat(const Vec3& R, const Vec3& L, const Vec3& N) const;
    float DistributionGGX(const Vec3& N, const Vec3& H, float roughness) const;
    float GeometrySchlickGGX(float NdotV, float roughness) const;
    float GeometrySmith(const Vec3& N, const Vec3& V, const Vec3& L, float roughness) const;
    Vec3 fresnelSchlick(float cosTheta, const Vec3& F0) const;

    Vec3 fresnelSchlickRoughness(float cosTheta, const Vec3& F0, float roughness) const;
private:
    float normalStrength = 1.0f;
    Vec3 subsurfaceRadius = 0.0f;
    float clearcoat = 1.0f;
    float clearcoatRoughness = 0.0f;
    float anisotropic = 0;
    float transmission = 0;
    Vec3 anisotropicDirection;
    Vec3 subsurfaceColor;
    float opacityAlpha = 1.0f;

    // Helper methods
    float max(float a, float b) const { return a > b ? a : b; }
    UVData transformUV(double u, double v) const;
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

    Vec3 calculate_sss_attenuation(double distance) const;

    Vec3 sample_henyey_greenstein(const Vec3& wi, double g) const;
   

    Vec3 computeScatterDirection(const Vec3& N, const Vec3& T, const Vec3& B, float roughness) const;
    void createCoordinateSystem(const Vec3& N, Vec3& T, Vec3& B) const; 
    

  
    Vec2 applyTiling(double u, double v) const;
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
