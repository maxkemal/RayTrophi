#include "PrincipledBSDF.h"
#include <cmath>
#include "Matrix4x4.h"
#include "HittableList.h"
#include <globals.h>
#include <Dielectric.h>
float PrincipledBSDF::getIndexOfRefraction() const {
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
Vec3 PrincipledBSDF::get_normal_from_map(float u, float v) const {
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

__m256 PrincipledBSDF::scatter_packet(const RayPacket& r_in, const HitRecordPacket& rec, Vec3Packet& attenuation, RayPacket& scattered) const {
    // 1. Gather Material Properties for 8 lanes 
    alignas(32) float alb_x[8], alb_y[8], alb_z[8];
    alignas(32) float rough[8], metal[8], trans[8];
    alignas(32) int mat_ids[8];
    _mm256_store_si256((__m256i*)mat_ids, rec.mat_id);

    for (int i = 0; i < 8; ++i) {
        Vec2 uv(rec.u.get(i), rec.v.get(i));
        Vec3 alb = albedoProperty.evaluate(uv);
        alb_x[i] = alb.x; alb_y[i] = alb.y; alb_z[i] = alb.z;
        rough[i] = std::max(roughnessProperty.evaluate(uv).y, 0.05f);
        metal[i] = metallicProperty.evaluate(uv).x;
        trans[i] = transmission; 
    }
    
    Vec3SIMD albedo_x(_mm256_load_ps(alb_x)), albedo_y(_mm256_load_ps(alb_y)), albedo_z(_mm256_load_ps(alb_z));
    __m256 roughness = _mm256_load_ps(rough);
    __m256 metallic = _mm256_load_ps(metal);
    __m256 transmission_v = _mm256_load_ps(trans);
    __m256 inv_transmission = _mm256_sub_ps(_mm256_set1_ps(1.0f), transmission_v);
    
    Vec3SIMD V_x = -r_in.dir_x, V_y = -r_in.dir_y, V_z = -r_in.dir_z;
    __m256 NdotV = _mm256_max_ps(Vec3SIMD::dot_product_8x(rec.normal_x, rec.normal_y, rec.normal_z, V_x, V_y, V_z), _mm256_set1_ps(1e-4f));

    // 2. Lobe Selection 
    __m256 r_lobe = Vec3SIMD::random_float_8x();
    __m256 is_trans = _mm256_cmp_ps(r_lobe, transmission_v, _CMP_LT_OQ);
    
    __m256 p_spec = _mm256_add_ps(_mm256_set1_ps(0.5f), _mm256_mul_ps(_mm256_set1_ps(0.5f), metallic));
    __m256 is_spec = _mm256_andnot_ps(is_trans, _mm256_cmp_ps(r_lobe, _mm256_add_ps(transmission_v, _mm256_mul_ps(inv_transmission, p_spec)), _CMP_LT_OQ));
    
    __m256 is_diff = _mm256_andnot_ps(is_trans, _mm256_andnot_ps(is_spec, _mm256_set1_ps(1.0f)));

    // 3. Sampling
    __m256 r1 = Vec3SIMD::random_float_8x(), r2 = Vec3SIMD::random_float_8x();
    
    // GGX Specular Lobe
    __m256 alpha = _mm256_mul_ps(roughness, roughness);
    __m256 alpha2 = _mm256_mul_ps(alpha, alpha);
    __m256 denom_ggx = _mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(alpha2, _mm256_set1_ps(1.0f)), r2), _mm256_set1_ps(1.0f));
    __m256 cosTheta_ggx = _mm256_sqrt_ps(_mm256_max_ps(_mm256_div_ps(_mm256_sub_ps(_mm256_set1_ps(1.0f), r2), denom_ggx), _mm256_setzero_ps()));
    __m256 sinTheta_ggx = _mm256_sqrt_ps(_mm256_max_ps(_mm256_sub_ps(_mm256_set1_ps(1.0f), _mm256_mul_ps(cosTheta_ggx, cosTheta_ggx)), _mm256_setzero_ps()));
    
    // Diffuse Lobe
    __m256 cosTheta_diff = _mm256_sqrt_ps(r2);
    __m256 sinTheta_diff = _mm256_sqrt_ps(_mm256_max_ps(_mm256_sub_ps(_mm256_set1_ps(1.0f), r2), _mm256_setzero_ps()));
    
    // Combine
    __m256 cosTheta = _mm256_blendv_ps(cosTheta_diff, cosTheta_ggx, _mm256_or_ps(is_spec, is_trans));
    __m256 sinTheta = _mm256_blendv_ps(sinTheta_diff, sinTheta_ggx, _mm256_or_ps(is_spec, is_trans));
    __m256 phi = _mm256_mul_ps(_mm256_set1_ps(2.0f * M_PI), r1), sinPhi, cosPhi;
    Vec3SIMD::sincos_256(phi, &sinPhi, &cosPhi);
    
    Vec3Packet H_local(Vec3SIMD(_mm256_mul_ps(sinTheta, cosPhi)), Vec3SIMD(_mm256_mul_ps(sinTheta, sinPhi)), Vec3SIMD(cosTheta));
    
    // Tangent Space Transform
    Vec3SIMD T_x, T_y, T_z, B_x, B_y, B_z;
    __m256 mask_z = _mm256_cmp_ps(_mm256_and_ps(rec.normal_z.data, _mm256_castsi256_ps(_mm256_set1_epi32(~0x80000000))), _mm256_set1_ps(0.999f), _CMP_LT_OQ);
    Vec3Packet up(Vec3SIMD(_mm256_andnot_ps(mask_z, _mm256_set1_ps(1.0f))), Vec3SIMD(0.0f), Vec3SIMD(mask_z));
    Vec3SIMD::cross_8x(up.x, up.y, up.z, rec.normal_x, rec.normal_y, rec.normal_z, T_x, T_y, T_z);
    Vec3SIMD::normalize_8x(T_x, T_y, T_z, T_x, T_y, T_z);
    Vec3SIMD::cross_8x(rec.normal_x, rec.normal_y, rec.normal_z, T_x, T_y, T_z, B_x, B_y, B_z);
    
    Vec3SIMD H_x, H_y, H_z;
    H_x.data = _mm256_add_ps(_mm256_mul_ps(T_x.data, H_local.x.data), _mm256_add_ps(_mm256_mul_ps(B_x.data, H_local.y.data), _mm256_mul_ps(rec.normal_x.data, H_local.z.data)));
    H_y.data = _mm256_add_ps(_mm256_mul_ps(T_y.data, H_local.x.data), _mm256_add_ps(_mm256_mul_ps(B_y.data, H_local.y.data), _mm256_mul_ps(rec.normal_y.data, H_local.z.data)));
    H_z.data = _mm256_add_ps(_mm256_mul_ps(T_z.data, H_local.x.data), _mm256_add_ps(_mm256_mul_ps(B_z.data, H_local.y.data), _mm256_mul_ps(rec.normal_z.data, H_local.z.data)));
    
    Vec3SIMD L_refl_x, L_refl_y, L_refl_z, L_refr_x, L_refr_y, L_refr_z;
    Vec3SIMD::reflect_8x(V_x, V_y, V_z, H_x, H_y, H_z, L_refl_x, L_refl_y, L_refl_z);
    
    // Simple refraction for now
    L_refr_x.data = _mm256_sub_ps(_mm256_setzero_ps(), V_x.data); 
    L_refr_y.data = _mm256_sub_ps(_mm256_setzero_ps(), V_y.data); 
    L_refr_z.data = _mm256_sub_ps(_mm256_setzero_ps(), V_z.data); 

    scattered.dir_x.data = _mm256_blendv_ps(H_x.data, L_refl_x.data, is_spec);
    scattered.dir_y.data = _mm256_blendv_ps(H_y.data, L_refl_y.data, is_spec);
    scattered.dir_z.data = _mm256_blendv_ps(H_z.data, L_refl_z.data, is_spec);
    
    scattered.dir_x.data = _mm256_blendv_ps(scattered.dir_x.data, L_refr_x.data, is_trans);
    scattered.dir_y.data = _mm256_blendv_ps(scattered.dir_y.data, L_refr_y.data, is_trans);
    scattered.dir_z.data = _mm256_blendv_ps(scattered.dir_z.data, L_refr_z.data, is_trans);
    
    __m256 NdotL = Vec3SIMD::dot_product_8x(rec.normal_x, rec.normal_y, rec.normal_z, scattered.dir_x, scattered.dir_y, scattered.dir_z);
    __m256 valid = _mm256_or_ps(is_trans, _mm256_cmp_ps(NdotL, _mm256_setzero_ps(), _CMP_GT_OQ));
    
    scattered.orig_x = rec.p_x; scattered.orig_y = rec.p_y; scattered.orig_z = rec.p_z;
    
    // 4. Weighting
    __m256 VdotH = _mm256_max_ps(Vec3SIMD::dot_product_8x(V_x, V_y, V_z, H_x, H_y, H_z), _mm256_setzero_ps());
    Vec3Packet F0(Vec3SIMD(_mm256_blendv_ps(_mm256_set1_ps(0.04f), albedo_x.data, metallic)),
                  Vec3SIMD(_mm256_blendv_ps(_mm256_set1_ps(0.04f), albedo_y.data, metallic)),
                  Vec3SIMD(_mm256_blendv_ps(_mm256_set1_ps(0.04f), albedo_z.data, metallic)));
    
    Vec3SIMD F_x, F_y, F_z; 
    fresnelSchlick_SIMD(VdotH, F0.x, F0.y, F0.z, F_x, F_y, F_z); 
    
    __m256 G = GeometrySmith_SIMD(rec.normal_x, rec.normal_y, rec.normal_z, V_x, V_y, V_z, scattered.dir_x, scattered.dir_y, scattered.dir_z, roughness);
    __m256 NdotH = _mm256_max_ps(Vec3SIMD::dot_product_8x(rec.normal_x, rec.normal_y, rec.normal_z, H_x, H_y, H_z), _mm256_set1_ps(1e-4f));
    __m256 spec_w = _mm256_div_ps(_mm256_mul_ps(G, VdotH), _mm256_mul_ps(NdotH, NdotV));
    
    attenuation.x.data = _mm256_blendv_ps(albedo_x.data, _mm256_mul_ps(F_x.data, spec_w), is_spec);
    attenuation.y.data = _mm256_blendv_ps(albedo_y.data, _mm256_mul_ps(F_y.data, spec_w), is_spec);
    attenuation.z.data = _mm256_blendv_ps(albedo_z.data, _mm256_mul_ps(F_z.data, spec_w), is_spec);
    
    attenuation.x.data = _mm256_blendv_ps(attenuation.x.data, albedo_x.data, is_trans);
    attenuation.y.data = _mm256_blendv_ps(attenuation.y.data, albedo_y.data, is_trans);
    attenuation.z.data = _mm256_blendv_ps(attenuation.z.data, albedo_z.data, is_trans);

    __m256 prob = _mm256_blendv_ps(_mm256_mul_ps(inv_transmission, _mm256_sub_ps(_mm256_set1_ps(1.0f), p_spec)), _mm256_mul_ps(inv_transmission, p_spec), is_spec);
    prob = _mm256_blendv_ps(prob, transmission_v, is_trans);
    
    attenuation.x.data = _mm256_div_ps(attenuation.x.data, _mm256_max_ps(prob, _mm256_set1_ps(1e-4f)));
    attenuation.y.data = _mm256_div_ps(attenuation.y.data, _mm256_max_ps(prob, _mm256_set1_ps(1e-4f)));
    attenuation.z.data = _mm256_div_ps(attenuation.z.data, _mm256_max_ps(prob, _mm256_set1_ps(1e-4f)));
    
    return _mm256_and_ps(valid, rec.mask);
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

Vec3 PrincipledBSDF::getTextureColor(float u, float v) const {
    std::cout << "Original UV: (" << u << ", " << v << ")" << std::endl;
    UVData uvData = transformUV(u, v);
    std::cout << "Transformed UV: (" << uvData.transformed.u << ", " << uvData.transformed.v << ")" << std::endl;
    Vec2 finalUV = applyWrapMode(uvData);
    std::cout << "Final UV: (" << finalUV.u << ", " << finalUV.v << ")" << std::endl;
    if (albedoProperty.texture) {
        return albedoProperty.texture->get_color(finalUV.u, finalUV.v);
    }
    return albedoProperty.color;
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
    // Check if emission color is non-zero (matches getEmission behavior)
    return emissionProperty.color.length_squared() > 0.0001f || emissionProperty.texture != nullptr;
}
bool refract(const Vec3& uv, const Vec3& n, float etai_over_etat, Vec3& refracted) {
    float cos_theta = std::clamp(-Vec3::dot(uv, n), -1.0f, 1.0f);
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
    Vec3 F_avg = F0 + (Vec3(1.0f) - F0) / 21.0;

    // 2. Enerji kaybını roughness'a bağlı olarak tahmin et
    Vec3 energyLoss = (Vec3(1.0f) - F_avg) * (Vec3(1.0f) - F_avg) * roughness;

    // 3. Metalik malzemelerde tazminatı devre dışı bırak (F0 ≈ 1 olduğu için)
    Vec3 compensation = Vec3(1.0f) + energyLoss * (1.0f - metallic);

    // 4. Enerji oranı hesapla (Fresnel bazlı)
    float E = std::max(1.0f - (F0.x * 0.9f), 0.01f);  // Enerji korunumu için  

    // 5. Speküler enerjiyi artır ve enerji oranıyla dengele
    return Vec3::min(specular * compensation / E, specular * 2.0f);
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
    float transmissionValue = transmission;
    float opacity = get_opacity(uv);
	float ior = getIOR();

    Vec3 N = rec.interpolated_normal.normalize(); // Geometri normali
    Vec3 V = -r_in.direction.normalize();         // Gelen ışığın yönü (view vector)
    Vec3 L = scattered.direction;
    attenuation = Vec3(0.0f);
    // 1. Opacity check - now handled in ray_color (bounce refund logic)

  
     
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
     L = Vec3::reflect(-V, H).normalize();
    
    // Ensure L is in the upper hemisphere
    if (Vec3::dot(N, L) < 0.0f) L = -L; // Or just return false? Usually geometric normal check handles this.
    // For safety, flip if needed or clamp.
    float NdotL = std::fmax(Vec3::dot(N, L), 0.001f);
    float NdotV = std::fmax(Vec3::dot(N, V), 0.001f);
    float NdotH = std::fmax(Vec3::dot(N, H), 0.001f);
    float VdotH = std::fmax(Vec3::dot(V, H), 0.001f);

    // Fresnel (Schlick)
    Vec3 F0 = Vec3::lerp(Vec3(0.04f), albedo, metallic);
    Vec3 F = fresnelSchlickRoughness(VdotH, F0, roughness);

    // Diffuse Term (Lambertian)
    Vec3 F_avg = F0 + (Vec3(1.0f) - F0) / 21.0f;
    Vec3 k_d = (Vec3(1.0f) - F_avg) * (1.0f - metallic);

    // Optimization: Algebraic Simplification of Weight = Weight_Spec + Weight_Diff
    // Weight_Spec = (F * G * VdotH) / (NdotV * NdotH)  [D cancels out]
    // Weight_Diff = (k_d * albedo) * NdotL * (denom^2 * 4 * VdotH) / (alpha2 * NdotH) [PI cancels out]

    float alpha = roughness * roughness;
    float alpha2 = std::max(alpha * alpha, 0.001f);
    float denom = (NdotH * NdotH) * (alpha2 - 1.0f) + 1.0f;

    // 1. Specular Weight
    float G = GeometrySmith(N, V, L, roughness);
    Vec3 spec_weight = (F * G * VdotH) / (NdotV * NdotH);

    // 2. Diffuse Weight
    // Note: Implicit PDF is (D * NdotH) / (4 * VdotH). Inverse PDF has (4 * VdotH) in numerator.
    // Diffuse term has 1/PI. PDF has 1/PI. They cancel.
    // Explicitly: (k_d * albedo / PI) * NdotL * (4 * VdotH) / (D * NdotH)
    // Substitute D = alpha2 / (PI * denom^2)
    // Result: (k_d * albedo) * NdotL * (denom^2 * 4 * VdotH) / (alpha2 * NdotH)
    float pdf_inverse_factor = (denom * denom * 4.0f * VdotH) / (alpha2 * NdotH);
    Vec3 diff_weight = (k_d * albedo)/M_PI * (NdotL * pdf_inverse_factor);

    // Final Attenuation
    attenuation = diff_weight + spec_weight;

    // Sanity check
    if (std::isnan(attenuation.x) || std::isnan(attenuation.y) || std::isnan(attenuation.z)) {
         attenuation = Vec3(0.0f);
    }
    
    scattered = Ray(rec.point + N * 0.001f, L);
    return true;
}
float PrincipledBSDF::pdf(const HitRecord& rec, const Vec3& incoming, const Vec3& outgoing) const  {
    float metallic = getPropertyValue(metallicProperty, Vec2(rec.u, rec.v)).z;
    float roughness = getPropertyValue(roughnessProperty, Vec2(rec.u, rec.v)).y;
    float cos_theta = std::fmax(Vec3::dot(rec.normal, outgoing), 0.0f);

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
    // GPU ile uyumlu: k = (r+1)^2 / 8 (Direct Lighting) vs k = r^2 / 2 (IBL)
    // We stick to the IBL/PathTracing convention: k = alpha^2 / 2 = roughness^4 / 2 ??
    // The previous code used k = max(r^2, 0.01) / 2.0. Let's optimize and stabilize.
    
    // Optimization: Precomputed k passed? No, local calculation.
    float r = roughness + 1.0f;
    float k = (r*r) / 8.0f; // Standard Schlick-GGX for direct lighting (often used in PT too for consistency)
    // Or previous: float k = alpha / 2.0f; 
    
    // Let's stick to the simpler one used before but optimized
    float alpha = roughness * roughness;
    float k_ibl = std::max(alpha, 0.001f) * 0.5f; 
    
    float denom = NdotV * (1.0f - k_ibl) + k_ibl;
    return NdotV / std::max(denom, 0.0001f);
}

float PrincipledBSDF::DistributionGGX(const Vec3& N, const Vec3& H, float roughness) const {
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;
    float NdotH = std::fmax(Vec3::dot(N, H), 0.0f);
    float NdotH2 = NdotH * NdotH;
    
    float denom = (NdotH2 * (alpha2 - 1.0f) + 1.0f);
    // Optimization: Avoid double PI division later if possible, but here return D.
    // Safety: max(denom, epsilon)
    return alpha2 / (M_PI * std::max(denom * denom, 0.0001f));
}

float PrincipledBSDF::GeometrySmith(const Vec3& N, const Vec3& V, const Vec3& L, float roughness) const {
    float NdotV = std::fmax(Vec3::dot(N, V), 0.0f);
    float NdotL = std::fmax(Vec3::dot(N, L), 0.0f);
    // Optimized: Inline GeometrySchlickGGX to avoid repetitive k calc?
    // Let's keep it separate for readability but ensure k is consistent.
    
    // Note: Smith G is often G1(V) * G1(L)
    // If we want high perf, we can combine:
    // G = 0.5 / (lerp(2*NdotL*NdotV, NdotL+NdotV, alpha)) ?? (Visibility term)
    // But sticking to separate G1 calls for now to match previous logic logic.
    return GeometrySchlickGGX(NdotV, roughness) * GeometrySchlickGGX(NdotL, roughness);
}

// Optimization 5: Use importance sampling for more efficient Monte Carlo integration
Vec3 PrincipledBSDF::importanceSampleGGX(float u1, float u2, float roughness, const Vec3& N) const {
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha; // GGX usually uses alpha = roughness^2
    
    float phi = 2.0f * M_PI * u1;
    // Optimization: fast cos/sin if available? Standard is fine.
    
    // GGX importance sampling formula:
    // cosTheta = sqrt( (1 - u2) / (1 + (alpha2 - 1) * u2) )
    float denom = (1.0f - u2) / (1.0f + (alpha2 - 1.0f) * u2);
    float cosTheta = std::sqrt(std::max(denom, 0.0f));
    float sinTheta = std::sqrt(std::max(1.0f - cosTheta * cosTheta, 0.0f));

    // Spherical to Cartesian
    float x = sinTheta * std::cos(phi);
    float y = sinTheta * std::sin(phi);
    float z = cosTheta;
    
    // Tangent Space to World Space
    Vec3 up = std::abs(N.z) < 0.999f ? Vec3(0.0f, 0.0f, 1.0f) : Vec3(1.0f, 0.0f, 0.0f);
    Vec3 tangentX = Vec3::cross(up, N).normalize();
    Vec3 tangentY = Vec3::cross(N, tangentX);
    
    return (tangentX * x + tangentY * y + N * z).normalize();
}

Vec3 PrincipledBSDF::evalSpecular(const Vec3& N, const Vec3& V, const Vec3& L, const Vec3& F0, float roughness) const {
    // Note: This method is largely superseded by the algebraic optimization in scatter,
    // but kept for other lobes or fallbacks.
    Vec3 H = (V + L).normalize();
    float D = DistributionGGX(N, H, roughness);
    float G = GeometrySmith(N, V, L, roughness);
    Vec3 F = fresnelSchlickRoughness(std::max(Vec3::dot(H, V), 0.0f), F0, roughness);
    
    float NdotV = std::fmax(Vec3::dot(N, V), 0.001f);
    float NdotL = std::fmax(Vec3::dot(N, L), 0.001f);
    
    return (F * D * G) / (4.0f * NdotV * NdotL);
}

Vec3 PrincipledBSDF::fresnelSchlickRoughness(float cosTheta, const Vec3& F0, float roughness) const {
    // Optimization: Replace pow(x, 5) with explicit multiplication
    float x = std::clamp(1.0f - cosTheta, 0.0f, 1.0f);
    float x2 = x * x;
    float x5 = x2 * x2 * x; // Faster than std::pow(x, 5)
    
    Vec3 F90 = Vec3::max(Vec3(1.0f - roughness), F0);
    return F0 + (F90 - F0) * x5;
}


Vec3 PrincipledBSDF::fresnelSchlick(float cosTheta, const Vec3& F0) const {
    return F0 + (Vec3(1.0f) - F0) * std::pow(1.0f - cosTheta, 5.0);
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
    Vec3 clearcoatColor = Vec3(1.0f);
    float clearcoatRough = clearcoatRoughness;

    Vec3 H = (V + L).normalize();  // H vektörü yansıyan ışık değil, göz ve ışık yönü olmalı
    float clearcoatNDF = DistributionGGX(N, H, clearcoatRough);
    float clearcoatG = GeometrySmith(N, V, L, clearcoatRough);
    Vec3 clearcoatF = fresnelSchlick(std::fmax(Vec3::dot(H, V), 0.0f), clearcoatColor);

    float clearcoatDenom = 4.0f * std::fmax(Vec3::dot(N, V), 0.0f) * max(Vec3::dot(N, L), 0.0f) + 0.0001f;

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


Vec3 PrincipledBSDF::sample_henyey_greenstein(const Vec3& wi, float g) const {
    float rand1 = Vec3::random_float();
    float rand2 = Vec3::random_float();

    // HG (Henyey-Greenstein) phase function için cos(theta) hesapla
    float cos_theta;
    if (std::abs(g) < 1e-3) {
        cos_theta = 1.0f - 2.0f * rand1;  // g  0 için isotropik dağılım
    }
    else {
        float square = (1.0f - g * g) / (1.0f + g - 2.0f * g * rand1);
        cos_theta = (1.0f + g * g - square * square) / (2.0f * g);
    }

    // Sin(θ) hesapla
    float sin_theta = sqrt(std::fmax(0.0f, 1.0f - cos_theta * cos_theta));

    // (phi) rastgele bir açı seç
    float phi = 2.0f * static_cast<float>(M_PI) * rand2;

    // Başlangıç vektörü w olarak al
    Vec3 w = wi.normalize();

    // w'den ortogonal bir u vektörü oluştur
    Vec3 u = (fabs(w.x) > 0.9f) ? Vec3(0, 1, 0) : Vec3(1, 0, 0);
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

Vec2 PrincipledBSDF::applyTiling(float u, float v) const {
    return Vec2(fmod(u * tilingFactor.u, 1.0f),
        fmod(v * tilingFactor.v, 1.0f));
}

Vec2 PrincipledBSDF::applyTextureTransform(float u, float v) const {
    // Merkezi (0.5, 0.5f) olarak kabul edelim
    u -= 0.5f;
    v -= 0.5f;
    // Ölçeklendirme
    u *= textureTransform.scale.u;
    v *= textureTransform.scale.v;
    // Döndürme (dereceyi radyana çevir)
    float rotation_radians = textureTransform.rotation_degrees * M_PI / 180.0f;
    float cosTheta = std::cos(rotation_radians);
    float sinTheta = std::sin(rotation_radians);
    float newU = u * cosTheta - v * sinTheta;
    float newV = u * sinTheta + v * cosTheta;
    u = newU;
    v = newV;
    // Merkezi geri taşı
    u += 0.5f;
    v += 0.5f;
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

UVData PrincipledBSDF::transformUV(float u, float v) const {
    UVData uvData;
    uvData.original = Vec2(u, v);

    // Tüm dönüşümleri uygula
    u -= 0.5f;
    v -= 0.5f;

    u *= textureTransform.scale.u;
    v *= textureTransform.scale.v;

    float rotation_radians = textureTransform.rotation_degrees * M_PI / 180.0f;
    float cosTheta = std::cos(rotation_radians);
    float sinTheta = std::sin(rotation_radians);
    float newU = u * cosTheta - v * sinTheta;
    float newV = u * sinTheta + v * cosTheta;
    u = newU;
    v = newV;

    u += 0.5f;
    v += 0.5f;

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
    float u = std::fmod(uv.u, 1.0f);
    float v = std::fmod(uv.v, 1.0f);
    if (u < 0) u += 1.0f;
    if (v < 0) v += 1.0f;
    return Vec2(u, v);
}

Vec2 PrincipledBSDF::applyMirrorWrapping(const Vec2& uv) const {
    float u = std::fmod(uv.u, 2.0f);
    float v = std::fmod(uv.v, 2.0f);
    if (u < 0) u += 2.0f;
    if (v < 0) v += 2.0f;
    if (u > 1.0f) u = 2.0f - u;
    if (v > 1.0f) v = 2.0f - v;
    return Vec2(u, v);
}

Vec2 PrincipledBSDF::applyClampWrapping(const Vec2& uv) const {
    float u = (uv.u < 0.0f) ? 0.0f : ((uv.u > 1.0f) ? 1.0f : uv.u);
    float v = (uv.v < 0.0f) ? 0.0f : ((uv.v > 1.0f) ? 1.0f : uv.v);
    return Vec2(u, v);
}

Vec2 PrincipledBSDF::applyPlanarWrapping(const Vec2& uv) const {
    // Planer sarma için orijinal UV'leri kullan
    return uv;
}

Vec2 PrincipledBSDF::applyCubicWrapping(const Vec2& uv) const {
    // UV koordinatlarını 0-3 aralığına genişlet
    float u_scaled = uv.u * 3.0f;
    float v_scaled = uv.v * 3.0f;

    // Hangi yüzeyde olduğumuzu belirle
    int face = static_cast<int>(u_scaled) + 3 * static_cast<int>(v_scaled);

    // Yüzey içindeki lokal koordinatları hesapla
    float u_local = std::fmod(u_scaled, 1.0f);
    float v_local = std::fmod(v_scaled, 1.0f);

    // Yüzeye göre koordinatları ayarla
    switch (face % 6) {  // 6'ya göre mod alarak taşmaları önlüyoruz
    case 0: // Ön yüz
        return Vec2(u_local, v_local);
    case 1: // Sağ yüz
        return Vec2(v_local, 1.0f - u_local);
    case 2: // Arka yüz
        return Vec2(1.0f - u_local, v_local);
    case 3: // Sol yüz
        return Vec2(1.0f - v_local, 1.0f - u_local);
    case 4: // Üst yüz
        return Vec2(u_local, 1.0f - v_local);
    case 5: // Alt yüz
        return Vec2(u_local, v_local);
    }

    // Bu noktaya asla ulaşılmamalı, ama güvenlik için ekliyoruz
    return uv;
}
void PrincipledBSDF::setTextureTransform(const TextureTransform& transform) {
    textureTransform = transform;
}

// ============================================================================
// SIMD Vectorized BRDF Implementation (Phase 3)
// ============================================================================

__m256 PrincipledBSDF::DistributionGGX_SIMD(const Vec3SIMD& N_x, const Vec3SIMD& N_y, const Vec3SIMD& N_z,
                                            const Vec3SIMD& H_x, const Vec3SIMD& H_y, const Vec3SIMD& H_z,
                                            const __m256& roughness) const {
    __m256 alpha = _mm256_mul_ps(roughness, roughness);
    __m256 alpha2 = _mm256_mul_ps(alpha, alpha);
    
    __m256 NdotH = _mm256_max_ps(Vec3SIMD::dot_product_8x(N_x, N_y, N_z, H_x, H_y, H_z), _mm256_setzero_ps());
    __m256 NdotH2 = _mm256_mul_ps(NdotH, NdotH);
    
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 denom_inner = _mm256_add_ps(_mm256_mul_ps(NdotH2, _mm256_sub_ps(alpha2, one)), one);
    __m256 denom = _mm256_mul_ps(_mm256_set1_ps(M_PI), _mm256_max_ps(_mm256_mul_ps(denom_inner, denom_inner), _mm256_set1_ps(0.0001f)));
    
    return _mm256_div_ps(alpha2, denom);
}

__m256 PrincipledBSDF::GeometrySmith_SIMD(const Vec3SIMD& N_x, const Vec3SIMD& N_y, const Vec3SIMD& N_z,
                                          const Vec3SIMD& V_x, const Vec3SIMD& V_y, const Vec3SIMD& V_z,
                                          const Vec3SIMD& L_x, const Vec3SIMD& L_y, const Vec3SIMD& L_z,
                                          const __m256& roughness) const {
    auto G1 = [&](const Vec3SIMD& dir_x, const Vec3SIMD& dir_y, const Vec3SIMD& dir_z) {
        __m256 NdotD = _mm256_max_ps(Vec3SIMD::dot_product_8x(N_x, N_y, N_z, dir_x, dir_y, dir_z), _mm256_setzero_ps());
        __m256 alpha = _mm256_mul_ps(roughness, roughness);
        __m256 k_ibl = _mm256_mul_ps(_mm256_max_ps(alpha, _mm256_set1_ps(0.001f)), _mm256_set1_ps(0.5f));
        
        __m256 one = _mm256_set1_ps(1.0f);
        __m256 denom = _mm256_add_ps(_mm256_mul_ps(NdotD, _mm256_sub_ps(one, k_ibl)), k_ibl);
        return _mm256_div_ps(NdotD, _mm256_max_ps(denom, _mm256_set1_ps(0.0001f)));
    };
    
    return _mm256_mul_ps(G1(V_x, V_y, V_z), G1(L_x, L_y, L_z));
}

void PrincipledBSDF::fresnelSchlick_SIMD(const __m256& cosTheta,
                                         const Vec3SIMD& F0_x, const Vec3SIMD& F0_y, const Vec3SIMD& F0_z,
                                         Vec3SIMD& out_x, Vec3SIMD& out_y, Vec3SIMD& out_z) const {
    __m256 x = _mm256_max_ps(_mm256_min_ps(_mm256_sub_ps(_mm256_set1_ps(1.0f), cosTheta), _mm256_set1_ps(1.0f)), _mm256_setzero_ps());
    __m256 x2 = _mm256_mul_ps(x, x);
    __m256 x5 = _mm256_mul_ps(_mm256_mul_ps(x2, x2), x);
    
    __m256 one = _mm256_set1_ps(1.0f);
    out_x.data = _mm256_add_ps(F0_x.data, _mm256_mul_ps(_mm256_sub_ps(one, F0_x.data), x5));
    out_y.data = _mm256_add_ps(F0_y.data, _mm256_mul_ps(_mm256_sub_ps(one, F0_y.data), x5));
    out_z.data = _mm256_add_ps(F0_z.data, _mm256_mul_ps(_mm256_sub_ps(one, F0_z.data), x5));
}
