#ifndef METAL_H
#define METAL_H

#include "Material.h"
#include "Texture.h"
#include "Vec2.h"
#include "Vec3SIMD.h"
#include <memory>




class Metal : public Material {
public:
    struct TextureTransform {
        Vec2 scale{ 1.0, 1.0 };
        double rotation_degrees{ 0.0 };
        Vec2 translation{ 0.0, 0.0 };
        Vec2 tilingFactor{ 1.0, 1.0 };
        WrapMode wrapMode{ WrapMode::Repeat };
    };

    Metal(const Vec3SIMD& albedo, float roughness, float metallic, float fuzz, float clearcoat);
        
    Metal(const std::shared_ptr<Texture>& albedoTexture, float roughness, float metallic, float fuzz, float clearcoat);
      

    MaterialType type() const override;
    virtual Vec3 getEmission(const Vec2& uv, const Vec3& p) const override;
   
    virtual bool scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered) const override;
    virtual double getIndexOfRefraction() const override;
    virtual float get_opacity(const Vec2& uv) const override;

   

    void setTextureTransform(const TextureTransform& transform);
    void setTilingFactor(const Vec2& factor) { tilingFactor = factor; }
    void set_normal_map(std::shared_ptr<Texture> normalMap, float normalStrength = 1.0f);

    

    // New methods for PBR
    void setSpecular(const Vec3SIMD& specular, float intensity = 1.0f);
    void setMetallic(float metallic, float intensity);
    void setSpecularTexture(const std::shared_ptr<Texture>& tex, float intensity = 1.0f);
    void setClearcoat(float clearcoat, float clearcoatRoughness = 0.1f, const Vec3SIMD& clearcoatColor = Vec3SIMD(1.0f, 1.0f, 1.0f));
    void setMetallic(float metallic, float intensity, const Vec3SIMD& color);
    void setAnisotropic(float anisotropic, const Vec3SIMD& anisotropicDirection = Vec3SIMD(1, 0, 0));
   
    void setMetallicTexture(const std::shared_ptr<Texture>& tex, float intensity = 1.0f);
    void setEmission(const Vec3SIMD& emission, float intensity = 0.0f);
    void setEmissionTexture(const std::shared_ptr<Texture>& tex, float intensity = 1.0f);
    Vec2 applyTextureTransform(double u, double v) const;
    float get_scattering_factor() const override {
        // Example formula incorporating reflectivity and roughness
        return  0.01f;
    }
    // New properties for PBR
    MaterialProperty specularProperty;
    Vec3SIMD metallicColor;
    Vec3SIMD albedo;
    double fuzz;
    Vec3SIMD specularColor;    // Specular rengi
    float specularIntensity; // Specular yoðunluðu
    float clearcoat; // Clearcoat etkisi
    float clearcoatRoughness; // Clearcoat pürüzlülüðü
    float anisotropic; // Anizotropik etkiler
    Vec3SIMD anisotropicDirection; // Anizotropik yön
    float metallic; // Metalik özellik
    float roughness;
    Vec3SIMD clearcoatColor;
    MaterialProperty albedoProperty;
    MaterialProperty roughnessProperty;
    MaterialProperty metallicProperty;
    MaterialProperty normalProperty;
    MaterialProperty emissionProperty;
    Vec2 tilingFactor;
    TextureTransform textureTransform;
private:
   

  
    // Helper methods
    float max(float a, float b) const { return a > b ? a : b; }
    Vec2 applyWrapMode(double u, double v) const;
    float computeAmbientOcclusion(const Vec3SIMD& point, const Vec3SIMD& normal) const;
    Vec3SIMD computeClearcoat(const Vec3SIMD& reflected, const Vec3SIMD& normal) const;
    Vec3SIMD computeScatterDirection(const Vec3SIMD& N, const Vec3SIMD& T, const Vec3SIMD& B, float roughness) const;
    void createCoordinateSystem(const Vec3SIMD& N, Vec3SIMD& T, Vec3SIMD& B) const;
   
   
    Vec3SIMD computeFresnel(const Vec3SIMD& F0, float cosTheta) const;
    Vec3SIMD getPropertyValue(const MaterialProperty& prop, const Vec2& uv) const;
    Vec3SIMD applyTiling(double u, double v) const;
   
    // GGX BRDF için yeni yardýmcý fonksiyonlar
    float DistributionGGX(const Vec3SIMD& N, const Vec3SIMD& H, float roughness) const;
    float GeometrySchlickGGX(float NdotV, float roughness) const;
    float GeometrySmith(const Vec3SIMD& N, const Vec3SIMD& V, const Vec3SIMD& L, float roughness) const;
    Vec3SIMD fresnelSchlick(float cosTheta, const Vec3SIMD& F0) const;
    // New helper methods for PBR
    Vec3SIMD computeAnisotropicDirection(const Vec3SIMD& N, const Vec3SIMD& T, const Vec3SIMD& B, float roughness, float anisotropy) const;
};

#endif // METAL_H
