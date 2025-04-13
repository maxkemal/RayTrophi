﻿#include "PrincipledBSDF.h"
#include <cmath>
#include "Matrix4x4.h"
#include "HittableList.h"
#include <globals.h>
double PrincipledBSDF::getIndexOfRefraction() const {
    // Metaller için genellikle kompleks kırılma indeksi kullanılır,
    // ancak basitlik için sabit bir değer döndürebiliriz.
    return 0.0; // veya başka bir uygun değer
}
bool PrincipledBSDF::hasTexture() const {
    return albedoProperty.texture != nullptr ||
        roughnessProperty.texture != nullptr ||
        metallicProperty.texture != nullptr ||
        opacityProperty.texture != nullptr ||
        normalProperty.texture != nullptr;
}
void PrincipledBSDF::set_normal_map(std::shared_ptr<Texture> normalMap, float normalStrength) {
    normalProperty.texture = normalMap;
    normalProperty.intensity = normalStrength;
}
Vec3 PrincipledBSDF::get_normal_from_map(double u, double v) const {
    if (normalProperty.texture) {
        Vec2 Transform = applyTextureTransform(u, v);

        // std::cout << "Normal Map: (" << Normalmap.x << ", " << Normalmap.y << ", " << Normalmap.z << ")" << std::endl;
        return normalProperty.texture->get_color(Transform.u, Transform.v);
    }
    return Vec3(0, 0, 1);
}
void PrincipledBSDF::setSpecular(const Vec3& specular, float intensity) {
    specularProperty = MaterialProperty(specular, intensity);
}
void PrincipledBSDF::setMetallic(float metallic, float intensity) {
    metallicProperty.color = Vec3(metallic, metallic, metallic);
    metallicProperty.intensity = intensity;
    metallicProperty.texture = nullptr;
}

void PrincipledBSDF::setMetallicTexture(const std::shared_ptr<Texture>& tex, float intensity) {
    metallicProperty.texture = tex;
    metallicProperty.intensity = intensity;
}
void PrincipledBSDF::setSpecularTexture(const std::shared_ptr<Texture>& tex, float intensity) {
    specularProperty = MaterialProperty(Vec3(1, 1, 1), intensity, tex);
}

void PrincipledBSDF::setEmission(const Vec3& emission, float intensity) {
    emissionProperty = MaterialProperty(emission, intensity);
}

void PrincipledBSDF::setEmissionTexture(const std::shared_ptr<Texture>& tex, float intensity = 0.0f) {
    emissionProperty = MaterialProperty(Vec3(0, 0, 0), intensity, tex);
}

void PrincipledBSDF::setClearcoat(float clearcoat, float clearcoatRoughness) {
    this->clearcoat = clearcoat;
    this->clearcoatRoughness = clearcoatRoughness;
}

void PrincipledBSDF::setAnisotropic(float anisotropic = 0.0f, const Vec3& anisotropicDirection = Vec3(0, 0, 0)) {
    this->anisotropic = anisotropic;
    this->anisotropicDirection = anisotropicDirection.normalize();
}

void PrincipledBSDF::setSubsurfaceScattering(const Vec3& sssColor, Vec3 sssRadius) {
    this->subsurfaceColor = sssColor;
    this->subsurfaceRadius = sssRadius;
}
std::shared_ptr<Texture> PrincipledBSDF::getTexture() const {
    if (albedoProperty.texture) return albedoProperty.texture;
    if (roughnessProperty.texture) return roughnessProperty.texture;
    if (metallicProperty.texture) return metallicProperty.texture;
    if (normalProperty.texture) return normalProperty.texture;
    return nullptr;
}

Vec3 PrincipledBSDF::getTextureColor(double u, double v) const {
    std::cout << "Original UV: (" << u << ", " << v << ")" << std::endl;
    UVData uvData = transformUV(u, v);
    std::cout << "Transformed UV: (" << uvData.transformed.u << ", " << uvData.transformed.v << ")" << std::endl;
    Vec2 finalUV = applyWrapMode(uvData);
    std::cout << "Final UV: (" << finalUV.u << ", " << finalUV.v << ")" << std::endl;
    return texture->get_color(finalUV.u, finalUV.v);
}

bool PrincipledBSDF::hasOpacityTexture() const {
    return opacityProperty.texture != nullptr;
}

void PrincipledBSDF::setOpacityTexture(const std::shared_ptr<Texture>& tex, float intensity) {
    opacityProperty.texture = tex;
    opacityProperty.intensity = intensity;
}

float PrincipledBSDF::get_opacity(const Vec2& uv) const {
    // Eğer texture yoksa, opaklığı 1.0 olarak döndür
    if (!opacityProperty.texture) {
        return 1.0f;
    }

    const Texture& opacityTexture = *opacityProperty.texture;
    Vec2 updated_uv = uv;  // Geçici bir değişken oluştur
    if (updated_uv.x < 0.0f || updated_uv.x > 1.0f || updated_uv.y < 0.0f || updated_uv.y > 1.0f) {
        // UV koordinatlarını sınırlar içinde sıkıştır
        updated_uv.x = std::clamp(updated_uv.x, 0.0, 1.0);
        updated_uv.y = std::clamp(updated_uv.y, 0.0, 1.0);
    }
    // Alpha değerini al
    float alpha = opacityTexture.get_alpha(uv.x, uv.y);
    // Alpha kanalını doğrudan kullanarak opaklık değeri hesapla
    return alpha * opacityProperty.intensity;
}
float PrincipledBSDF::get_roughness(float u, float v) const {
    return getPropertyValue(roughnessProperty, Vec2(u, v)).y;
}

void PrincipledBSDF::setTransmission(float transmission, float ior) {
    this->transmission = transmission;
    this->ior = ior;
}

float PrincipledBSDF::getTransmission() const {
    return transmission;
}

float PrincipledBSDF::getIOR() const {
    return ior;
}
bool PrincipledBSDF::isEmissive() const {
    return emissionProperty.intensity > 0.0f;
}
bool refract(const Vec3& uv, const Vec3& n, float etai_over_etat, Vec3& refracted) {
    double cos_theta = std::clamp(-Vec3::dot(uv, n), -1.0, 1.0);
    Vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    float k = 1.0f - r_out_perp.length_squared();

    if (k < 0) {
        return false;  // Total internal reflection oldu, kırılma yok.
    }

    Vec3 r_out_parallel = -sqrt(k) * n;
    refracted = r_out_perp + r_out_parallel;
    return true;
}
Vec3 applyDisneyMultiScattering(const Vec3& specular, const Vec3& F0,
    float roughness, float metallic) {
    // 1. Ortalama Fresnel yansımasını hesapla (RGB-aware)
    Vec3 F_avg = F0 + (Vec3(1.0) - F0) / 21.0;

    // 2. Enerji kaybını roughness'a bağlı olarak tahmin et
    Vec3 energyLoss = (Vec3(1.0) - F_avg) * (Vec3(1.0) - F_avg) * roughness;

    // 3. Metalik malzemelerde tazminatı devre dışı bırak (F0 ≈ 1 olduğu için)
    Vec3 compensation = Vec3(1.0) + energyLoss * (1.0 - metallic);

    // 4. Enerji oranı hesapla (Fresnel bazlı)
    float E = std::max(1.0f - (F0.x * 0.9f), 0.01f);  // Enerji korunumu için  

    // 5. Speküler enerjiyi artır ve enerji oranıyla dengele
    return Vec3::min(specular * compensation / E, specular * 2.0);
}

bool PrincipledBSDF::scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered) const {
    Vec2 transformedUV = useSmartUVProjection ? Vec2(rec.u, rec.v) : applyTextureTransform(rec.u, rec.v);
    Vec3 albedoValue = (getPropertyValue(albedoProperty, transformedUV));
    albedoValue = Vec3::max(albedoValue, Vec3(0.05f));
    float roughness = getPropertyValue(roughnessProperty, transformedUV).y;
    float metallic = getPropertyValue(metallicProperty, transformedUV).z;
    Vec3 specularValue = getPropertyValue(specularProperty, transformedUV);
    Vec3 emissionValue = getPropertyValue(emissionProperty, transformedUV);
    float transmissionValue = transmission; // Sabit değer veya texture'dan alınabilir    	
    Vec3 N = rec.interpolated_normal;
    Vec3 V = -r_in.direction.normalize();   
    Vec3 L = scattered.direction.normalize();
    // Normali kamera yönüne doğru bük (N•V < 0.1 ise)
    const float threshold = 0.1f; // Daha geniş bir açı aralığı
    float NdotV = Vec3::dot(N, V);
    if (NdotV < threshold) {
        float blend = pow(1.0f - NdotV / threshold, 2.0f) * 0.7f; // Quadratic falloff & max 70% blend
        N = Vec3::mix(N, V, blend).normalize();
        NdotV = Vec3::dot(N, V); // Güncellenmiş değer
    }
    Vec3 H;
 
	// Normali düzelt
    N = N.normalize();
    roughness = std::clamp(roughness, 0.0f, 1.0f);
    metallic = std::clamp(metallic, 0.0f, 1.0f);
    attenuation = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 F0 = Vec3::lerp(Vec3(0.04), albedoValue, metallic);
    Vec3 F ;
    float opacity = get_opacity(rec.uv);
    if (opacity < 1.0f) {
        attenuation = Vec3(1.0f); // ışık kaybı olmasın
        scattered = Ray(rec.point + r_in.direction * 0.001f, r_in.direction); // ışığı geç
        return true;
    }

    // --- YENİ SAÇILMA YÖNÜ HESAPLA ---
    Vec3 scatteredDirection;
   if (emissionValue.length() > 0.0f) {
    attenuation = emissionValue;
    scatteredDirection = N;
    return true;
}
   if (transmissionValue > 0.0f) {
       float cosTheta = Vec3::dot(N, V);
       float eta = (cosTheta > 0.0f) ? (1.0f / ior) : ior;

       Vec3 refractedDir;
       if (refract(V, N, eta, refractedDir)) {
           // Fresnel hesabı
           Vec3 F0 = Vec3::mix(Vec3(0.04f), albedoValue, metallic);
           Vec3 F = fresnelSchlick(fabs(cosTheta), F0);

           // Enerji dağılımı
           float reflectance = (F.x + F.y + F.z) / 3.0f;
           float transmittance = 1.0f - reflectance;

           // Kırılan ışın
           scattered = Ray(rec.point, refractedDir.normalize());

           // Yüzey katkısını engelle
           attenuation = albedoValue * transmissionValue * transmittance;

           return true;
       }
       else {
           // Total internal reflection varsa, yansıma yap
           Vec3 reflected = Vec3::reflect(V, N).normalize();
           scattered = Ray(rec.point, reflected);
           attenuation = albedoValue * transmissionValue;
           return true;
       }
   }

    else if (anisotropic > 0) {
        Vec3 T, B;
        createCoordinateSystem(N, T, B);
        scatteredDirection = computeAnisotropicDirection(N, T, B, roughness, anisotropic);
    }
    else {       
        // GGX ile mikro - yüzey normali(H) örnekle
        if (roughness > 0.8f) {
            scatteredDirection = Vec3::random_cosine_direction(N).normalize(); // GGX yerine klasik cosine-weighted
        }
        else {
            H = importanceSampleGGX(Vec3::random_double(), Vec3::random_double(), roughness, N);
            scatteredDirection = Vec3::reflect(-V, H).normalize();
        }

       // Vec3 diffuse_dir = Vec3::lerp(metal_dir,Vec3::random_cosine_direction(N),roughness).normalize();
   //     if(roughness<0.1f)
			//diffuse_dir = metal_dir; // Diffuse yönü düzelt
        // Fresnel-Schlick-Roughness (RGB -> sadece R kanalını kullan)
       // float cosTheta = std::fmax(V.dot(H), 1e-6f);       
        // F = fresnelSchlickRoughness(cosTheta, F0, roughness);
        // Stokastik seçim (speküler/difüz)
       // scatteredDirection = (Vec3::random_double() < F.x) ? metal_dir : diffuse_dir;
      
      
    }

    // Nümerik hata kontrolü
    if (scatteredDirection.near_zero()) {
        scatteredDirection = rec.normal;
    }

    // --- BRDF HESAPLAMALARI (L KULLANILARAK) ---
    Vec3 kD = (1.0 - metallic);	
    // Difüz + Speküler katkı
    Vec3 specular = evalSpecular(N, V, L, F0, roughness);
    // evalSpecular() sonrasında Enerji Tazminatı
    // Multi-scattering enerji faktörü (fiziksel tabanlı)
    // Disney multi-scattering uygula
    specular += applyDisneyMultiScattering(specular, F0, roughness, metallic); 
    
	Vec3 diffuse = kD * albedoValue ; // Difüz katkı
    if (NdotV < 0.0f) { // Backface
        float backfaceIntensity = std::max(1.5f - roughness, 1.0f); // Roughness'a bağlı azalma
        diffuse += albedoValue * backfaceIntensity * pow(abs(NdotV), 2.0f);
    }
   
    // Toplam BRDF'yi enerji koruyacak şekilde ayarla
    Vec3 brdf = diffuse + specular;
   // float energy = Vec3::dot(brdf, Vec3(1.0)); // Toplam enerji (luminance)
   // if (energy > 1.0) {
    //    float metalFactor = metallic * 0.5 + 0.5; // Metaliklik arttıkça daha az enerji normalizasyonu
     //    brdf = brdf / (energy * metalFactor + (1.0 - metalFactor));

     //    brdf = brdf / energy; // Normalize et
    //}
    attenuation = brdf;

    // Yeni saçılmış ışını güncelle
    scattered = Ray(rec.point, scatteredDirection);

    if (clearcoat > 0.0) {
        Vec3 clearcoatReflection = computeClearcoat(V, L, N).normalize();

        // Clearcoat etkisini roughness ile ölçekle
        double clearcoatFactor = clearcoat * (1.0 - clearcoatRoughness);

        attenuation = Vec3::lerp(attenuation, clearcoatReflection, clearcoatFactor);
    }


    if (subsurfaceRadius.x > 0 || subsurfaceRadius.y > 0 || subsurfaceRadius.z > 0) {
        float avgThickness = Vec3::average(subsurfaceRadius) * 4.0f;

        Vec3 subsurfaceContribution = computeSubsurfaceScattering(N, V, subsurfaceRadius, avgThickness);

        // Sadece metalik olmayan yüzeylerde katkı
        float sssBlend = 1.0f - metallic;

        // Eğer Fresnel'e benzer azalma isteniyorsa, ayrı bir 'viewFactor' tanımlanabilir
        float viewFactor = std::pow(1.0f - std::abs(Vec3::dot(N, V)), 2.0f); // kenarlarda daha güçlü
        sssBlend *= viewFactor;

        subsurfaceContribution *= albedoValue;
        attenuation = Vec3::lerp(attenuation, subsurfaceContribution, sssBlend);
    }

    return true;
}
float PrincipledBSDF::GeometrySchlickGGX(float NdotV, float roughness) const {
    float r = (roughness * roughness);
    float k = (r * r) / 8.0f; // Changed from /2.0 to /8.0 for better approximation

    return NdotV / (NdotV * (1.0f - k) + k);
}

float PrincipledBSDF::DistributionGGX(const Vec3& N, const Vec3& H, float roughness) const {
    float alpha =(roughness * roughness);
    float alpha2 = alpha * alpha;
    float NdotH = fmax(Vec3::dot(N, H), 0.0001f);
    float NdotH2 = NdotH * NdotH;
    float denom = M_PI * (NdotH2 * (alpha2 - 1.0f) + 1.0f);
    float denom_rcp = 1.0f / (denom * denom);  // Bölme yerine ters çevirme
    return alpha2 * denom_rcp;
}

float PrincipledBSDF::GeometrySmith(const Vec3& N, const Vec3& V, const Vec3& L, float roughness) const {

    float k = (roughness * roughness) /4.0f;
    float NdotV = fmax(Vec3::dot(N, V), 0.0001f);
    float NdotL = fmax(Vec3::dot(N, L), 0.0001f);

    float denomV = NdotV * (1.0f - k) + k;
    float denomL = NdotL * (1.0f - k) + k;
    return (NdotV / denomV) * (NdotL / denomL);

}
// Optimization 5: Use importance sampling for more efficient Monte Carlo integration
Vec3 PrincipledBSDF::importanceSampleGGX(float u1, float u2, float roughness, const Vec3& N) const {
    float alpha = (roughness * roughness);
    float phi = 2.0f * M_PI * u1;
    float cosTheta = std::sqrt((1.0f - u2) / (1.0f + (alpha * alpha - 1.0f) * u2));
    float sinTheta = std::sqrt(1.0f - cosTheta * cosTheta);
    Vec3 H(sinTheta * std::cos(phi), sinTheta * std::sin(phi), cosTheta);
    Vec3 up = std::abs(N.z) > 0.999f ? Vec3(1.0f, 0.0f, 0.0f) : Vec3(0.0f, 0.0f, 1.0f);
    Vec3 tangentX = up.cross(N);
    Vec3 tangentY = N.cross(tangentX);
    return (tangentX * H.x + tangentY * H.y + N * H.z).normalize();
}

Vec3 PrincipledBSDF::evalSpecular(const Vec3& N, const Vec3& V, const Vec3& L, const Vec3& F0, float roughness) const {
    Vec3 H = (V + L).normalize();
    float NDF = DistributionGGX(N, H, roughness);
    float G = GeometrySmith(N, V, L, roughness);
    Vec3 F = fresnelSchlickRoughness(max(Vec3::dot(H, V), 0.0f), F0, roughness);
    float NdotV = fmax(Vec3::dot(N, V), 0.0001f);
    float NdotL = fmax(Vec3::dot(N, L), 0.0001f);
    float denominator = 4.0 * NdotV * NdotL;
    return (NDF * G * F) / denominator;
}
Vec3 PrincipledBSDF::fresnelSchlickRoughness(float cosTheta, const Vec3& F0, float roughness) const {
    cosTheta = fmax(cosTheta, 0.0001f);  // Küçük değerler için koruma
    float cosTheta1m = (1.0f - cosTheta);
    float exponent = cosTheta1m * cosTheta1m * cosTheta1m * cosTheta1m * cosTheta1m;
    float alpha = roughness * roughness;  // Mikrofacet modeline uygun
    Vec3 Fmax = Vec3::mix(F0, Vec3(1.0), alpha);

    return F0 + (Fmax - F0) * exponent;  // Doğru formül
}


Vec3 PrincipledBSDF::fresnelSchlick(float cosTheta, const Vec3& F0) const {
    return F0 + (Vec3(1.0) - F0) * std::pow(1.0 - cosTheta, 5.0);
}
// LUT dizilerinin başlangıçta boş olması gerekiyor.
float PrincipledBSDF::sqrtTable[256];
float PrincipledBSDF::cosTable[256];
float PrincipledBSDF::sinTable[256];
bool PrincipledBSDF::lutInitialized = false;
// LUT Hesaplama Metodu
void PrincipledBSDF::precomputeLUT() {
    if (lutInitialized) return;  // Zaten hesaplandıysa tekrar yapma

    for (int i = 0; i < 256; ++i) {
        float u = i / 255.0f;
        sqrtTable[i] = std::sqrt(u);
        cosTable[i] = std::cos(2.0f * M_PI * u);
        sinTable[i] = std::sin(2.0f * M_PI * u);
    }
    lutInitialized = true;
}
Vec3 PrincipledBSDF::computeAnisotropicDirection(const Vec3& N, const Vec3& T, const Vec3& B, float roughness, float anisotropy) const {
    float r1 = random_double();
    float r2 = random_double();

    float phi = 2 * M_PI * r1;
    float cosTheta = std::sqrt((1 - r2) / (1 + (anisotropy * anisotropy - 1) * r2));
    float sinTheta = std::sqrt(1 - cosTheta * cosTheta);

    float x = sinTheta * std::cos(phi);
    float y = sinTheta * std::sin(phi);
    float z = cosTheta;

    return (x * T + y * B + z * N);
}


void PrincipledBSDF::createCoordinateSystem(const Vec3& N, Vec3& T, Vec3& B) const {
    if (std::fabs(N.x) > std::fabs(N.y)) {
        T = Vec3(N.z, 0, -N.x);
    }
    else {
        T = Vec3(0, -N.z, N.y);
    }
    B = Vec3::cross(N, T);
}

Vec3 PrincipledBSDF::computeFresnel(const Vec3& F0, float cosTheta) const {
    float p = std::pow(1.0f - cosTheta, 5.0f);
    return F0 + (Vec3(1.0f, 1.0f, 1.0f) - F0) * p;
}
Vec3 PrincipledBSDF::computeClearcoat(const Vec3& V, const Vec3& L, const Vec3& N) const {
    Vec3 clearcoatColor = Vec3(1.0);
    float clearcoatRough = clearcoatRoughness;

    Vec3 H = (V + L).normalize();  // H vektörü yansıyan ışık değil, göz ve ışık yönü olmalı
    float clearcoatNDF = DistributionGGX(N, H, clearcoatRough);
    float clearcoatG = GeometrySmith(N, V, L, clearcoatRough);
    Vec3 clearcoatF = fresnelSchlick(std::max(Vec3::dot(H, V), 0.0), clearcoatColor);

    float clearcoatDenom = 4.0f * std::max(Vec3::dot(N, V), 0.0) * max(Vec3::dot(N, L), 0.0f) + 0.0001f;

    return (clearcoatNDF * clearcoatG * clearcoatF) / clearcoatDenom;
}


Vec3 PrincipledBSDF::computeSubsurfaceScattering(const Vec3& N, const Vec3& V, const Vec3& subsurfaceRadius, float thickness) const {
    Vec3 scatteredLight(0.0f);

    int numSamples = 5; // Ne kadar çok olursa o kadar yumuşak
    for (int i = 0; i < numSamples; i++) {
        // Işığın içte ne kadar ilerlediğini rastgele belirle
        Vec3 randomOffset = Vec3(
            random_double() * subsurfaceRadius.x,
            random_double() * subsurfaceRadius.y,
            random_double() * subsurfaceRadius.z
        );

        // Yeni bir yön belirle (Henyey-Greenstein kullanabiliriz)
        Vec3 newDirection = sample_henyey_greenstein(V, 0.5f);

        // Yeni ışık vektörünü hesapla
        Vec3 lightContribution = Vec3::exp(-randomOffset / thickness);

        scatteredLight += lightContribution;
    }

    scatteredLight /= static_cast<float>(numSamples);
    return subsurfaceColor * scatteredLight;
}


Vec3 PrincipledBSDF::sample_henyey_greenstein(const Vec3& wi, double g) const {
    double rand1 = random_double();
    double rand2 = random_double();

    // HG (Henyey-Greenstein) phase function için cos(theta) hesapla
    double cos_theta;
    if (std::abs(g) < 1e-3) {
        cos_theta = 1.0 - 2.0 * rand1;  // g  0 için isotropik dağılım
    }
    else {
        double square = (1.0 - g * g) / (1.0 + g - 2.0 * g * rand1);
        cos_theta = (1.0 + g * g - square * square) / (2.0 * g);
    }

    // Sin(θ) hesapla
    double sin_theta = sqrt(std::max(0.0, 1.0 - cos_theta * cos_theta));

    // (phi) rastgele bir açı seç
    double phi = 2.0 * M_PI * rand2;

    // Başlangıç vektörü w olarak al
    Vec3 w = wi.normalize();

    // w'den ortogonal bir u vektörü oluştur
    Vec3 u = (fabs(w.x) > 0.9) ? Vec3(0, 1, 0) : Vec3(1, 0, 0);
    u = (u.cross(w)).normalize();

    // v vektörünü hesapla (u ve w'nin çapraz çarpımı)
    Vec3 v = w.cross(u);

    // Saçılım yönünü oluştur
    Vec3 scatter_direction =
        u * (sin_theta * cos(phi)) +
        v * (sin_theta * sin(phi)) +
        w * cos_theta;

    return scatter_direction.normalize();
}


Vec3 PrincipledBSDF::getPropertyValue(const MaterialProperty& prop, const Vec2& uv) const {
    if (prop.texture) {
        return prop.texture->get_color(uv.u, uv.v) * prop.intensity;
    }
    return prop.color * prop.intensity;
}

Vec2 PrincipledBSDF::applyTiling(double u, double v) const {
    return Vec2(fmod(u * tilingFactor.u, 1.0),
        fmod(v * tilingFactor.v, 1.0));
}

Vec2 PrincipledBSDF::applyTextureTransform(double u, double v) const {
    // Merkezi (0.5, 0.5) olarak kabul edelim
    u -= 0.5;
    v -= 0.5;
    // Ölçeklendirme
    u *= textureTransform.scale.u;
    v *= textureTransform.scale.v;
    // Döndürme (dereceyi radyana çevir)
    double rotation_radians = textureTransform.rotation_degrees * M_PI / 180.0;
    double cosTheta = std::cos(rotation_radians);
    double sinTheta = std::sin(rotation_radians);
    double newU = u * cosTheta - v * sinTheta;
    double newV = u * sinTheta + v * cosTheta;
    u = newU;
    v = newV;
    // Merkezi geri taşı
    u += 0.5;
    v += 0.5;
    // Öteleme uygula
    u += textureTransform.translation.u;
    v += textureTransform.translation.v;
    // Tiling uygula
    u *= textureTransform.tilingFactor.u;
    v *= textureTransform.tilingFactor.v;
    // Sarma modunu uygula
    UVData uvData = transformUV(u, v); // Orijinal ve dönüşmüş UV'leri almak için çağır
    return applyWrapMode(uvData);
}

UVData PrincipledBSDF::transformUV(double u, double v) const {
    UVData uvData;
    uvData.original = Vec2(u, v);

    // Tüm dönüşümleri uygula
    u -= 0.5;
    v -= 0.5;

    u *= textureTransform.scale.u;
    v *= textureTransform.scale.v;

    double rotation_radians = textureTransform.rotation_degrees * M_PI / 180.0;
    double cosTheta = std::cos(rotation_radians);
    double sinTheta = std::sin(rotation_radians);
    double newU = u * cosTheta - v * sinTheta;
    double newV = u * sinTheta + v * cosTheta;
    u = newU;
    v = newV;

    u += 0.5;
    v += 0.5;

    u += textureTransform.translation.u;
    v += textureTransform.translation.v;

    u *= textureTransform.tilingFactor.u;
    v *= textureTransform.tilingFactor.v;

    uvData.transformed = Vec2(u, v);
    return uvData;
}


Vec2 PrincipledBSDF::applyWrapMode(const UVData& uvData) const {
    switch (textureTransform.wrapMode) {
    case WrapMode::Repeat:
        return applyRepeatWrapping(uvData.transformed);
    case WrapMode::Mirror:
        return applyMirrorWrapping(uvData.transformed);
    case WrapMode::Clamp:
        return applyClampWrapping(uvData.transformed);
    case WrapMode::Planar:
        return applyPlanarWrapping(uvData.original);
    case WrapMode::Cubic:
        return applyCubicWrapping(uvData.transformed);
    }
    return uvData.transformed;
}
Vec2 PrincipledBSDF::applyRepeatWrapping(const Vec2& uv) const {
    double u = std::fmod(uv.u, 1.0);
    double v = std::fmod(uv.v, 1.0);
    if (u < 0) u += 1.0;
    if (v < 0) v += 1.0;
    return Vec2(u, v);
}

Vec2 PrincipledBSDF::applyMirrorWrapping(const Vec2& uv) const {
    double u = std::fmod(uv.u, 2.0);
    double v = std::fmod(uv.v, 2.0);
    if (u < 0) u += 2.0;
    if (v < 0) v += 2.0;
    if (u > 1.0) u = 2.0 - u;
    if (v > 1.0) v = 2.0 - v;
    return Vec2(u, v);
}

Vec2 PrincipledBSDF::applyClampWrapping(const Vec2& uv) const {
    double u = (uv.u < 0.0) ? 0.0 : ((uv.u > 1.0) ? 1.0 : uv.u);
    double v = (uv.v < 0.0) ? 0.0 : ((uv.v > 1.0) ? 1.0 : uv.v);
    return Vec2(u, v);
}

Vec2 PrincipledBSDF::applyPlanarWrapping(const Vec2& uv) const {
    // Planer sarma için orijinal UV'leri kullan
    return uv;
}

Vec2 PrincipledBSDF::applyCubicWrapping(const Vec2& uv) const {
    // UV koordinatlarını 0-3 aralığına genişlet
    double u_scaled = uv.u * 3.0;
    double v_scaled = uv.v * 3.0;

    // Hangi yüzeyde olduğumuzu belirle
    int face = static_cast<int>(u_scaled) + 3 * static_cast<int>(v_scaled);

    // Yüzey içindeki lokal koordinatları hesapla
    double u_local = std::fmod(u_scaled, 1.0);
    double v_local = std::fmod(v_scaled, 1.0);

    // Yüzeye göre koordinatları ayarla
    switch (face % 6) {  // 6'ya göre mod alarak taşmaları önlüyoruz
    case 0: // Ön yüz
        return Vec2(u_local, v_local);
    case 1: // Sağ yüz
        return Vec2(v_local, 1.0 - u_local);
    case 2: // Arka yüz
        return Vec2(1.0 - u_local, v_local);
    case 3: // Sol yüz
        return Vec2(1.0 - v_local, 1.0 - u_local);
    case 4: // Üst yüz
        return Vec2(u_local, 1.0 - v_local);
    case 5: // Alt yüz
        return Vec2(u_local, v_local);
    }

    // Bu noktaya asla ulaşılmamalı, ama güvenlik için ekliyoruz
    return uv;
}
void PrincipledBSDF::setTextureTransform(const TextureTransform& transform) {
    textureTransform = transform;
}
