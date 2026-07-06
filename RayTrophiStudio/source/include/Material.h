/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          Material.h
* Author:        Kemal Demirtas
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
# pragma once

#include "Ray.h"
#include "Vec3.h"
#include "Vec2.h"
#include "Hittable.h"
#include <memory>
#include "Texture.h"
#include "material_gpu.h"
class ParallelBVHNode; // Forward declaration if needed
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
    float alpha=1.0f;  // Alfa kanalı için ek alan
    std::shared_ptr<Texture> texture;
    operator float() const {
        return intensity;  // veya başka bir uygun değer
    }
    // Texture varsa UV'den örnek al, yoksa sabit değeri döndür
    Vec3 evaluate(const Vec2& uv) const {
        if (texture) {
            return texture->get_color_bilinear(uv.u, uv.v);  // CPU path should match GPU linear filtering
        }
        return color * intensity;
    }
    float evaluateOpacity(const Vec2& uv) const {
        if (texture) {
            float val = 0.0f;
            if (texture->has_alpha) {
                val = texture->get_alpha_bilinear(uv.u, uv.v);
            } else {
                val = texture->get_color_bilinear(uv.u, uv.v).x;
            }
            // Cutoff for noise/compression artifacts
            return (val < 0.1f) ? 0.0f : val * alpha;
        }
        return alpha;
    }


    // Veya
    float getValue() const {
        return intensity;  // veya başka bir uygun değer
    }
    // Yeni yapıcı
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
        float cos_theta = std::fmax(Vec3::dot(rec.normal, outgoing), 0.0);
        return cos_theta / M_PI;
    }
  
    std::string materialName;
    std::shared_ptr<GpuMaterial> gpuMaterial;
    Vec3 getPropertyValue(const MaterialProperty& prop, const Vec2& uv) const {
        const Vec2 sampleUv = applyTextureTransform(uv.u, uv.v);
        if (prop.texture) {
            return prop.texture->get_color_bilinear(sampleUv.u, sampleUv.v) * prop.intensity;
        }
        return prop.color * prop.intensity;
    }
    // Materyal türünü döndüren sanal metot
    virtual MaterialType type() const = 0;

    // Maksimum derinlik değerini döndüren metot
    int get_max_depth() const {
        static const std::unordered_map<MaterialType, int> max_depths = {
            {MaterialType::Dielectric,16},
            {MaterialType::Volumetric, 12},
            {MaterialType::PrincipledBSDF,16}
            // Diğer materyal türleri ve derinlik değerleri...
        };

        // Materyal türüne göre derinlik değerini döndür
        auto it = max_depths.find(type());
        return (it != max_depths.end()) ? it->second : 1; // Varsayılan değer: 1
    }
    virtual float getIndexOfRefraction() const = 0;
    float get_roughness(float u, float v) const {
        return getPropertyValue(roughnessProperty, Vec2(u, v)).y;
    }

    bool useSmartUVProjection = false; // Yeni üye
    MaterialProperty albedoProperty;
    MaterialProperty roughnessProperty;
    MaterialProperty metallicProperty;
    MaterialProperty specularProperty;
    MaterialProperty normalProperty;
    MaterialProperty opacityProperty;
    MaterialProperty transmissionProperty;
    MaterialProperty emissionProperty;
    MaterialProperty heightProperty; // Displacement / Height map
    virtual Vec2 applyTextureTransform(float u, float v) const {
        return Vec2(u, v);  // Varsayılan olarak dönüşüm uygulamaz
    }
    virtual std::shared_ptr<Texture> getTexture() const {
        if (albedoProperty.texture) return albedoProperty.texture;
        if (roughnessProperty.texture) return roughnessProperty.texture;
        if (metallicProperty.texture) return metallicProperty.texture;
        if (specularProperty.texture) return specularProperty.texture;
        if (normalProperty.texture) return normalProperty.texture;
        if (transmissionProperty.texture) return transmissionProperty.texture;
        if (opacityProperty.texture) return opacityProperty.texture;
        if (emissionProperty.texture) return emissionProperty.texture;
        return nullptr;
    }
   

    virtual ~Material() = default;
    virtual bool scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered, bool& is_specular) const = 0;
    
    virtual float get_metallic() const { return metallicProperty.intensity; }
    virtual Vec3 getEmission(const Vec2& uv, const Vec3& p) const {
        (void)uv;
        (void)p;
        return emissionProperty.color * emissionProperty.intensity;
    }
    virtual bool isEmissive() const { return false; }
    // Texture handling
    virtual bool hasTexture() const { return getTexture() != nullptr; }
    virtual bool hasOpacityTexture() const {
        return opacityProperty.texture != nullptr;
    }
    virtual bool isTransparent() const {
        return opacityProperty.alpha < 0.999f || opacityProperty.texture != nullptr;
    }
    virtual float get_opacity(const Vec2& uv) const {
        const Vec2 sampleUv = applyTextureTransform(uv.u, uv.v);
        return opacityProperty.evaluateOpacity(sampleUv);
    }

    void setTexture(std::shared_ptr<Texture> tex) { albedoProperty.texture = tex; }

   
    // Material.h (varsayılan baz sınıf)
    virtual float getTransmission(const Vec2& uv) const { return 0.0f; }
	virtual void setTransmission(float transmission, float ior) {
		transmissionProperty.intensity = transmission;
		this->ior = ior;
	}
    virtual float getIOR() const { return ior; } // Yeni: Kırılma indeksi
    virtual Vec3 getF0() const { return f0; } // Yeni: Fresnel yansıma katsayısı
    virtual bool has_normal_map() const { return false; }
    virtual Vec3 get_normal_from_map(float u, float v) const { return Vec3(0, 0, 1); }
    virtual float get_normal_strength() const { return normalStrength; }
    virtual void set_normal_strength(float norm)  {
        normalStrength = norm;
    }
    virtual float get_shininess() const {
        return (1.0f - roughnessProperty.intensity) * 128.0f; // Derived from roughness
    }

    // Advanced Material extensions (lazy allocated in PrincipledBSDF)
    virtual float getClearcoat() const { return 0.0f; }
    virtual void setClearcoat(float cc, float roughness = 0.03f) {}
    virtual float getClearcoatRoughness() const { return 0.03f; }
    virtual float getClearcoatIridescence() const { return 0.0f; }
    virtual void setClearcoatIridescence(float val) {}
    virtual float getClearcoatFilmThickness() const { return 0.55f; }
    virtual void setClearcoatFilmThickness(float val) {}

    virtual float getSubsurface() const { return 0.0f; }
    virtual void setSubsurface(float val) {}
    virtual Vec3 getSubsurfaceColor() const { return Vec3(1.0f, 0.8f, 0.6f); }
    virtual void setSubsurfaceColor(const Vec3& val) {}
    virtual Vec3 getSubsurfaceRadius() const { return Vec3(1.0f, 0.2f, 0.1f); }
    virtual void setSubsurfaceRadius(const Vec3& val) {}
    virtual float getSubsurfaceScale() const { return 0.05f; }
    virtual void setSubsurfaceScale(float val) {}
    virtual float getSubsurfaceAnisotropy() const { return 0.0f; }
    virtual void setSubsurfaceAnisotropy(float val) {}
    virtual float getSubsurfaceIOR() const { return 1.4f; }
    virtual void setSubsurfaceIOR(float val) {}
    virtual bool getUseRandomWalkSSS() const { return true; }
    virtual void setUseRandomWalkSSS(bool val) {}
    virtual int getSssMaxSteps() const { return 6; }
    virtual void setSssMaxSteps(int val) {}

    virtual float getTransmissionDensity() const { return 0.0f; }
    virtual void setTransmissionDensity(float val) {}
    virtual Vec3 getResinColor() const { return Vec3(1.0f, 1.0f, 1.0f); }
    virtual void setResinColor(const Vec3& val) {}
    virtual float getResinRoughness() const { return 0.1f; }
    virtual void setResinRoughness(float val) {}
    virtual float getResinInclusion() const { return 0.0f; }
    virtual void setResinInclusion(float val) {}
    virtual float getResinDirt() const { return 0.0f; }
    virtual void setResinDirt(float val) {}
    virtual float getResinInclusionScale() const { return 8.0f; }
    virtual void setResinInclusionScale(float val) {}
    virtual Vec3 getResinDirtColor() const { return Vec3(0.18f, 0.14f, 0.10f); }
    virtual void setResinDirtColor(const Vec3& val) {}
    virtual float getResinShard() const { return 0.0f; }
    virtual void setResinShard(float val) {}
    virtual float getResinShardHue() const { return -1.0f; }   // <0 = rainbow palette
    virtual void setResinShardHue(float val) {}
    virtual bool getResinObjectSpace() const { return true; }  // interior anchored to the object
    virtual void setResinObjectSpace(bool val) {}
    virtual int getDustStyle() const { return 0; }             // 0=Nebula 1=Billow 2=Wispy 3=Paint swirl
    virtual void setDustStyle(int val) {}
    virtual Vec3 getDustColorA() const { return Vec3(1.0f, 1.0f, 1.0f); }
    virtual void setDustColorA(const Vec3& val) {}
    virtual Vec3 getDustColorB() const { return Vec3(1.0f, 1.0f, 1.0f); }
    virtual void setDustColorB(const Vec3& val) {}
    virtual int getShardShape() const { return 0; }            // 0=chips 1=crystals
    virtual void setShardShape(int val) {}
    virtual bool getGlassMarbleVolume() const { return false; }
    virtual void setGlassMarbleVolume(bool val) {}

    virtual bool getIsBubble() const { return false; }
    virtual void setIsBubble(bool val) {}
    virtual float getBubbleIor() const { return 1.33f; }
    virtual void setBubbleIor(float val) {}
    virtual float getBubbleFilm() const { return 0.0f; }
    virtual void setBubbleFilm(float val) {}
   
    // Tiling
    Vec2 tilingFactor = Vec2(1, 1);
    void setTilingFactor(const Vec2& factor) { tilingFactor = factor; }
    Vec2 getTilingFactor() const { return tilingFactor; }

    // Advanced scattering methods
    virtual bool volumetric_scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered) const;
    virtual bool sss_scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered) const;
    virtual bool sss_random_walk_scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered) const;
    virtual bool clearcoat_scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered) const;
    virtual bool translucent_scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered) const;

    // Yeni: Anizotropik malzemeler için
    virtual Vec3 getAnisotropicDirection() const;
    virtual float getAnisotropy() const;

    // Yeni: Emisyon özelliği
    virtual Vec3 getEmission() const;
    virtual float get_scattering_factor() const = 0;
    Vec3 albedo;
    float artistic_albedo_response = 0.50f; // default fiziksel
    float ior = 1.5f; // Yeni: Varsayılan kırılma indeksi
    float normalStrength = 1.0f;
    float roughness = 0.0f;
    void setRoughness(float r) { roughness = r; }
protected:   
   
  
   
    Vec3 f0 = Vec3(0.04f); // Yeni: Varsayılan Fresnel yansıma katsayısı
    
    Vec3 emissionColor = Vec3(0.0f);
    // Yeni: Malzeme özelliklerini ayarlamak için yardımcı metotlar
    void setAlbedo(const Vec3& a) { albedo = a; }
   // void setMetallic(float m) { metallic = m; }
   
    void setIOR(float i) { ior = i; }
    void setF0(const Vec3& f) { f0 = f; }
};


// Refreshed
