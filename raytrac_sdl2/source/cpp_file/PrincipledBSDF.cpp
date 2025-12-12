#include "PrincipledBSDF.h"
#include <cmath>
#include "Matrix4x4.h"
#include "HittableList.h"
#include <globals.h>
#include <Dielectric.h>
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
float PrincipledBSDF::getTransmission(const Vec2& uv) const {
    if (transmissionProperty.texture) {
        return transmissionProperty.evaluate(uv).x * transmissionProperty.intensity;
    }
    return transmission;
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
    return opacityProperty.evaluateOpacity(uv);
}

float PrincipledBSDF::get_roughness(float u, float v) const {
    return getPropertyValue(roughnessProperty, Vec2(u, v)).y;
}

void PrincipledBSDF::setTransmission(float transmission, float ior) {
    this->transmission = transmission;
    this->ior = ior;
}

float PrincipledBSDF::getIOR() const {
    return ior;
}
bool PrincipledBSDF::isEmissive() const {
    return emissionProperty.intensity > 0.0f;
}
bool refract(const Vec3& uv, const Vec3& n, float etai_over_etat, Vec3& refracted) {
    double cos_theta = std::clamp(-Vec3::dot(uv, n), -1.0f, 1.0f);
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


bool PrincipledBSDF::scatter(
    const Ray& r_in,
    const HitRecord& rec,
    Vec3& attenuation,
    Ray& scattered
) const {
    // UV koordinatları, texture dönüşümünü uygula
    // Apply UV transformation (if needed)
    Vec2 uv = applyTextureTransform(rec.u, rec.v);

    // Albedo, roughness, metallic gibi değerleri al (texture olabilir)
    // Fetch BRDF properties (may be textured)
    Vec3 albedo = getPropertyValue(albedoProperty, uv);
   // albedo = Vec3::max(albedo, Vec3(0.05f)); // Tam siyahı engelle (Avoid total black)
    float roughness = getPropertyValue(roughnessProperty, uv).y;
    float metallic = getPropertyValue(metallicProperty, uv).z;
    Vec3 emission = getPropertyValue(emissionProperty, uv);
    float transmissionValue = transmission;
    float opacity = get_opacity(uv);
	float ior = getIOR();
    Vec3 N = rec.interpolated_normal.normalize(); // Geometri normali
    Vec3 V = -r_in.direction.normalize();         // Gelen ışığın yönü (view vector)
    Vec3 L = scattered.direction;
    // 1. Opaklık kontrolü – ışığın geçip geçmeyeceğine karar ver
    // Opacity check – stochastic transparency
    if (opacity < 1.0f) {
        if (Vec3::random_float() < opacity) {
            // Opaque kısım - normal reflection
            attenuation = albedo;
        }
        else {
            // Işık geçiyor, bu yüzden ışık kaynağına geri döndür
            scattered = Ray(rec.point + r_in.direction * 0.001f, r_in.direction);
            attenuation = Vec3(1.0f); // Tam geçiş
            return true; // Işık geçiyor
        }
    }
    // 2. Emissive malzeme varsa doğrudan ışık katkısı verir
   // If material emits light, return its emission
    if (emission.length() > 0.01f) {
        attenuation = emission;
        return true;
    }


 
    // 3. Transmission varsa cam/şeffaf gibi davran
    // Transmission check (e.g. glass-like material)
    if (Vec3::random_float() < transmissionValue) {
        Dielectric dielectricMat(
            ior, albedo, 1.0, 1.0, roughness, 0
        ); // IOR vs. daha gelişmiş hale getirilebilir
        return dielectricMat.scatter(r_in, rec, attenuation, scattered);
    }

    // 4. Mikrofacet yönü için Half-Vector oluştur
    // Sample a GGX-based half-vector for microfacet reflection
    Vec3 H = importanceSampleGGX(Vec3::random_float(), Vec3::random_float(), roughness, N);
    Vec3 metal_dir = Vec3::reflect(-V, H).normalize();
    Vec3 cosine_dir = Vec3::random_cosine_direction(N);

    // Geçiş katsayısı daha net kontrol edilsin
   
    Vec3 diffuse_dir = Vec3::lerp(metal_dir, cosine_dir, roughness).normalize();
    //Vec3 diffuse_dir = cosine_dir.normalize();
    // Fresnel term determines reflectance vs diffuse ratio
    Vec3 F0 = Vec3::lerp(Vec3(0.04f), albedo, metallic);
    float cosTheta = std::fmax(Vec3::dot(V, L), 1e-4);
    Vec3 F = fresnelSchlickRoughness(cosTheta, F0, roughness);
    Vec3 F_avg = F0 + (Vec3(1.0f) - F0) / 21.0f;
    // GPU ile uyumlu: k_d = (1 - F_avg) * (1 - metallic)
    Vec3 k_d = (Vec3(1.0f) - F_avg) * (1.0f - metallic);
    // 7. Stokastik seçim: F.x oranında speküler, geri kalanı difüz
    // Stochastic selection based on Fresnel reflectance
    Vec3 sampled_dir;
    float randVal = Vec3::random_float();
    Vec3 specular = evalSpecular(N, V, sampled_dir, F0, roughness);
    // Attenuation ayır
    if (randVal < F.x) {
    // 
    // yeni yön şimdilik metal_dr yönü alınsın
        sampled_dir = metal_dir;
       
     }
    else {
       sampled_dir = diffuse_dir;
       
    }
    // 8. Near-zero kontrolü
    // Check for near-zero direction to avoid NaNs
    if (sampled_dir.near_zero()) {
        sampled_dir = N;
    }	
    attenuation += (k_d * albedo)+specular;
    scattered = Ray(rec.point + N * 0.0001f, sampled_dir);
    return true;
}
float PrincipledBSDF::pdf(const HitRecord& rec, const Vec3& incoming, const Vec3& outgoing) const  {
    float metallic = getPropertyValue(metallicProperty, Vec2(rec.u, rec.v)).z;
    float roughness = getPropertyValue(roughnessProperty, Vec2(rec.u, rec.v)).y;
    float cos_theta = std::fmax(Vec3::dot(rec.normal, outgoing), 0.0);

    // GGX sample: assume isotropic
    // GPU ile uyumlu D terimi
    float alpha = max(roughness * roughness, 0.001f);
    float alpha2 = alpha * alpha;
    float NdotH = max(Vec3::dot(rec.normal, (outgoing + incoming)), 0.001);
    float NdotH2 = NdotH * NdotH;
    float denom = (NdotH2 * (alpha2 - 1.0f) + 1.0f);
    float D = alpha2 / (M_PI * denom * denom);

    float pdf_specular = D * NdotH / (4.0f * std::fmax(Vec3::dot(outgoing, (incoming + outgoing)), 0.001));
    float pdf_diffuse = cos_theta / M_PI;

    float fresnel_weight = metallic; // çok basit: sadece metal oranı
    return fresnel_weight * pdf_specular + (1.0f - fresnel_weight) * pdf_diffuse;
}

float PrincipledBSDF::GeometrySchlickGGX(float NdotV, float roughness) const {
    // GPU ile uyumlu: k = roughness^2 / 2 (eski: roughness^4 / 2 idi)
    float k = max(roughness * roughness, 0.01f) / 2.0f;
    return NdotV / (NdotV * (1.0f - k) + k);
}

float PrincipledBSDF::DistributionGGX(const Vec3& N, const Vec3& H, float roughness) const {
    // GPU ile uyumlu GGX NDF
    float alpha = max(roughness * roughness, 0.001f);
    float alpha2 = alpha * alpha;
    float NdotH = fmax(Vec3::dot(N, H), 0.0001f);
    float NdotH2 = NdotH * NdotH;
    // GPU formülü: alpha2 / (PI * denom^2)
    float denom = (NdotH2 * (alpha2 - 1.0f) + 1.0f);
    return alpha2 / (M_PI * denom * denom);
}

float PrincipledBSDF::GeometrySmith(const Vec3& N, const Vec3& V, const Vec3& L, float roughness) const {
    // GPU ile uyumlu formül
    float k = max(roughness * roughness, 0.01f) / 2.0f;
    float NdotV = fmax(Vec3::dot(N, V), 0.0001f);
    float NdotL = fmax(Vec3::dot(N, L), 0.0001f);

    float G1_V = NdotV / (NdotV * (1.0f - k) + k);
    float G1_L = NdotL / (NdotL * (1.0f - k) + k);
    return G1_V * G1_L;
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
    // GPU ile uyumlu Fresnel formülü
    cosTheta = fmax(cosTheta, 0.0001f);
    float oneMinusCos = 1.0f - cosTheta;
    float pow5 = oneMinusCos * oneMinusCos * oneMinusCos * oneMinusCos * oneMinusCos;
    
    // GPU formülüyle aynı: F0 + (max(1-roughness, F0) - F0) * pow5
    Vec3 Fmax = Vec3::max(Vec3(1.0f - roughness), F0);
    return F0 + (Fmax - F0) * pow5;
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
    float r1 = Vec3::random_float();
    float r2 = Vec3::random_float();

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
    Vec3 clearcoatF = fresnelSchlick(std::fmax(Vec3::dot(H, V), 0.0), clearcoatColor);

    float clearcoatDenom = 4.0f * std::fmax(Vec3::dot(N, V), 0.0) * max(Vec3::dot(N, L), 0.0f) + 0.0001f;

    return (clearcoatNDF * clearcoatG * clearcoatF) / clearcoatDenom;
}


Vec3 PrincipledBSDF::computeSubsurfaceScattering(const Vec3& N, const Vec3& V, const Vec3& subsurfaceRadius, float thickness) const {
    Vec3 scatteredLight(0.0f);

    int numSamples = 5; // Ne kadar çok olursa o kadar yumuşak
    for (int i = 0; i < numSamples; i++) {
        // Işığın içte ne kadar ilerlediğini rastgele belirle
        Vec3 randomOffset = Vec3(
            Vec3::random_float() * subsurfaceRadius.x,
            Vec3::random_float() * subsurfaceRadius.y,
            Vec3::random_float() * subsurfaceRadius.z
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
    double rand1 = Vec3::random_float();
    double rand2 = Vec3::random_float();

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
