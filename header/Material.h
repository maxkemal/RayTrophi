# pragma once

#include "Ray.h"
#include "Vec3.h"
#include "Hittable.h"
#include <memory>
#include "Texture.h"
#include "ParallelBVHNode.h"
#include "material_gpu.h"
enum class MaterialType {
    PrincipledBSDF,
    Metal,
    Dielectric,
    Volumetric,   
   
};
struct UVData {
    Vec2 original;
    Vec2 transformed;
};

enum class WrapMode {
    Repeat,
    Mirror,
    Clamp,
    Planar,
    Cubic
};
struct MaterialProperty {
    Vec3 color;
    float intensity=0.0f;
    float alpha=1.0f;  // Alfa kanal� i�in ek alan
    std::shared_ptr<Texture> texture;
    operator float() const {
        return intensity;  // veya ba�ka bir uygun de�er
    }
    // Texture varsa UV'den �rnek al, yoksa sabit de�eri d�nd�r
    Vec3 evaluate(const Vec2& uv) const {
        if (texture) {
            return texture->getColor(uv) * intensity;  // Texture �rne�ini yo�unlukla �arp
        }
        return color * intensity;
    }
    float evaluateOpacity(const Vec2& uv) const {
        if (texture) {
            return texture->get_alpha(uv.x, uv.y) * alpha;
        }
        return alpha;
    }


    // Veya
    float getValue() const {
        return intensity;  // veya ba�ka bir uygun de�er
    }
    // Yeni yap�c�
    MaterialProperty(std::shared_ptr<Texture> tex, float i = 1.0f, float a = 1.0f)
        : color(1.0f, 1.0f, 1.0f), intensity(i), texture(tex), alpha(a) {}
    MaterialProperty(const Vec3& c = Vec3(1, 1, 1), float i = 1.0f, std::shared_ptr<Texture> tex = nullptr)
        : color(c), intensity(i), texture(tex) {}
    MaterialProperty(const Vec3& c, float i, std::shared_ptr<Texture> tex, float a)
        : color(c), intensity(i), texture(tex), alpha(a) {
    }

};

class Material {
public:
    virtual float pdf(const HitRecord& rec, const Vec3& incoming, const Vec3& outgoing) const {
        // Default: Cosine-weighted hemisphere
        float cos_theta = std::max(Vec3::dot(rec.normal, outgoing), 0.0);
        return cos_theta / M_PI;
    }
  
    std::string materialName;
    std::shared_ptr<GpuMaterial> gpuMaterial;
    Vec3 getPropertyValue(const MaterialProperty& prop, const Vec2& uv) const {
        if (prop.texture) {
            return prop.texture->get_color(uv.u, uv.v) * prop.intensity;
        }
        return prop.color * prop.intensity;
    }
    // Materyal t�r�n� d�nd�ren sanal metot
    virtual MaterialType type() const = 0;

    // Maksimum derinlik de�erini d�nd�ren metot
    int get_max_depth() const {
        static const std::unordered_map<MaterialType, int> max_depths = {
            {MaterialType::Dielectric,16},
            {MaterialType::Volumetric, 12},
            {MaterialType::PrincipledBSDF,16}
            // Di�er materyal t�rleri ve derinlik de�erleri...
        };

        // Materyal t�r�ne g�re derinlik de�erini d�nd�r
        auto it = max_depths.find(type());
        return (it != max_depths.end()) ? it->second : 1; // Varsay�lan de�er: 1
    }
    virtual double getIndexOfRefraction() const = 0;
    float get_roughness(float u, float v) const {
        return getPropertyValue(roughnessProperty, Vec2(u, v)).y;
    }

    bool useSmartUVProjection = false; // Yeni �ye
    MaterialProperty albedoProperty;
    MaterialProperty roughnessProperty;
    MaterialProperty metallicProperty;
    MaterialProperty normalProperty;
    MaterialProperty opacityProperty;
	MaterialProperty transmissionProperty;
    virtual Vec2 applyTextureTransform(double u, double v) const {
        return Vec2(u, v);  // Varsay�lan olarak d�n���m uygulamaz
    }
    virtual std::shared_ptr<Texture> getTexture() const {
        if (albedoProperty.texture) return albedoProperty.texture;
        if (roughnessProperty.texture) return roughnessProperty.texture;
        if (metallicProperty.texture) return metallicProperty.texture;
        if (normalProperty.texture) return normalProperty.texture;
        if (transmissionProperty.texture) return transmissionProperty.texture;
        return nullptr;
    }
   

    virtual ~Material() = default;
    MaterialProperty shininess;
    MaterialProperty metallic;
    MaterialProperty materyalproperty;  
    virtual bool scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered) const = 0;
    virtual float get_metallic() const { return metallic.getValue(); }
    virtual Vec3 getEmission(double u, double v, const Vec3& p) const {
        return emissionColor* materyalproperty.intensity;
    }
    virtual bool isEmissive() const { return false; }
    // Texture handling
    virtual bool hasTexture() const { return texture != nullptr; }
    virtual bool hasOpacityTexture() const {
        return opacityProperty.texture != nullptr;
    }
    virtual float get_opacity(const Vec2& uv) const {
        return 1.0f; // Varsay�lan olarak tam opak
    }

    void setTexture(std::shared_ptr<Texture> tex) { texture = tex; }

   
    // Material.h (varsay�lan baz s�n�f)
    virtual float getTransmission(const Vec2& uv) const { return 0.0f; }
	virtual void setTransmission(float transmission, float ior) {
		transmissionProperty.intensity = transmission;
		ior = ior;
	}
    virtual float getIOR() const { return ior; } // Yeni: K�r�lma indeksi
    virtual Vec3 getF0() const { return f0; } // Yeni: Fresnel yans�ma katsay�s�
    virtual bool has_normal_map() const { return false; }
    virtual Vec3 get_normal_from_map(double u, double v) const { return Vec3(0, 0, 1); }
    virtual float get_normal_strength() const { return normalStrength; }
    virtual void set_normal_strength(float norm)  {
        normalStrength = norm;
    }
    virtual float get_shininess() const {
        return shininess.intensity*128 ;  // veya shininess.intensity
    }
   
    // Tiling
    Vec2 tilingFactor = Vec2(1, 1);
    void setTilingFactor(const Vec2& factor) { tilingFactor = factor; }
    Vec2 getTilingFactor() const { return tilingFactor; }

    // Advanced scattering methods
    virtual bool volumetric_scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered) const;
    virtual bool sss_scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered) const;

    // Yeni: Anizotropik malzemeler i�in
    virtual Vec3 getAnisotropicDirection() const;
    virtual float getAnisotropy() const;

    // Yeni: Emisyon �zelli�i
    virtual Vec3 getEmission() const;
    virtual float get_scattering_factor() const = 0;
    Vec3 albedo;
    float artistic_albedo_response = 0.50f; // default fiziksel
    float ior = 1.5f; // Yeni: Varsay�lan k�r�lma indeksi
protected:
   
    float normalStrength;
    float roughness = 0.0f;
   
    Vec3 f0 = Vec3(0.04f); // Yeni: Varsay�lan Fresnel yans�ma katsay�s�
    std::shared_ptr<Texture> texture;
    Vec3 emissionColor = 0;
    // Yeni: Malzeme �zelliklerini ayarlamak i�in yard�mc� metotlar
    void setAlbedo(const Vec3& a) { albedo = a; }
   // void setMetallic(float m) { metallic = m; }
    void setRoughness(float r) { roughness = r; }
    void setIOR(float i) { ior = i; }
    void setF0(const Vec3& f) { f0 = f; }
};