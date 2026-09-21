#include "Backend/VulkanBackend.h"
#include "RayFusion/ProbeBounce.h"
#include "RayFusion/RayFusionHairBounce.h"
#include <algorithm>
#include <chrono>
#include <cmath>

namespace Backend {

// ★★★★★ DOLAYLI GUNES. Bu fonksiyona kadar sicrama isik tablosu yalnizca
//   `m_cachedLights` uzerinde donuyordu, yani Physical Sky GUNESI dolayli
//   aydinlatmaya HIC katilmiyordu -- durum metni bunu zaten soyluyordu:
//   "No ... analytic sky-sun bounce".
//
// ★★★★ Olculdu 2026-09-15, ayni sahne ayni ayar, lineer uzay:
//     tavan  RayFusion 0.44/0.71/0.90  vs  RT 0.14/0.10/0.08
//   Oran R/G/B = 3.2 / 7.0 / 12.0 -- KANALLAR ARASI ESIT DEGIL. Skaler bir
//   cift uygulama uc kanalda da ayni sayiyi verirdi; kirmizidan maviye buyuyen
//   bu profil Rayleigh imzasidir. RT'de tavan SICAK (gunesli zeminden sicrayan
//   isik), RayFusion'da MAVI (yalnizca gokyuzu). Eksik olan tam olarak buydu.
//
// ★★★ Ton, dogrudan aydinlatmanin kullandigi `canonicalWorldSunRadiance` ile
//   AYNI formulden gelir. Ayri bir formul, bir yuzeyin aldigi dogrudan gunes
//   ile ondan sicrayan gunesi farkli renk yapardi ve kimse buna "iki farkli
//   gunes" demezdi -- "GI biraz soguk" derdi.
bool VulkanBackendAdapter::worldSunBounceRadiance(float outRgb[3], bool& tintFromLut) const {
    tintFromLut = false;
    outRgb[0] = outRgb[1] = outRgb[2] = 0.0f;
    if (m_cachedWorld.mode != WORLD_MODE_NISHITA) return false;
    const float intensity = m_cachedWorld.nishita.sun_intensity;
    if (!(intensity > 0.0f)) return false;
    const auto& d = m_cachedWorld.nishita.sun_direction;
    const float len = std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
    if (!(len > 1e-4f)) return false;
    const float sunY = d.y / len;

    // Shader ile BIREBIR ayni u: clamp((max(0.01, sunDir.y) + 0.2) / 1.2, 0, 1)
    float tint[3] = {1.0f, 0.95f, 0.86f};   // shader'in LUT'suz yedegi
    if (m_atmosphereTransmittanceRow0.size() >= static_cast<size_t>(TRANSMITTANCE_LUT_W) * 3u) {
        const float u = std::min(std::max(((std::max)(0.01f, sunY) + 0.2f) / 1.2f, 0.0f), 1.0f);
        // Bilinear, cunku GPU ornekleyicisi de bilinear: en yakin texel'i almak
        // alcak gunes acilarinda gorunur bir renk basamagi uretirdi.
        const float x = u * static_cast<float>(TRANSMITTANCE_LUT_W - 1);
        const int i0 = static_cast<int>(x);
        const int i1 = std::min(i0 + 1, TRANSMITTANCE_LUT_W - 1);
        const float f = x - static_cast<float>(i0);
        for (int c = 0; c < 3; ++c)
            tint[c] = m_atmosphereTransmittanceRow0[i0 * 3 + c] * (1.0f - f) +
                      m_atmosphereTransmittanceRow0[i1 * 3 + c] * f;
        tintFromLut = true;
    }
    for (int c = 0; c < 3; ++c) outRgb[c] = (std::max)(tint[c], 0.0f) * intensity;
    return true;
}

class RayFusionBounceResources {
public:
    RayFusion::BounceStatus status;
    VulkanRT::BufferHandle buffers[4]{};  // hits, materials, lights, emissive triangles
    // ★★★★★ EMISSIVE TARAMASI ONBELLEGI. Olculdu 2026-09-15: bu tarama tek
    //   basina 8,89 ms/kare -- toplam hazirligin %96,5'i -- ve bu sahnede
    //   SIFIR emissive ucgen buluyordu. Her karede butun ucgenleri gezip
    //   "hicbiri" demenin bedeli buydu.
    // ★★★ Neden mevcut imza kapisi bunu kurtaramiyordu: o imza KURULAN
    //   dizilerin hash'i, yani onu ureten isi yapisal olarak atlayamaz.
    //   Bu kapi GIRDILERDEN kuruluyor ve is yapilmadan ONCE bakiliyor.
    std::vector<RayFusion::EmissiveTriangle> cachedEmissives;
    std::vector<uint8_t> cachedMaterialInNee;
    uint64_t emissiveInputSignature = 0;
    bool emissiveCacheValid = false;
    // Taramanin status'e yazdigi sayaclar da onbellege girer, yoksa cache
    // isabetinde SIFIRLANIRLAR ve panel "emissive yok" der -- olculmus bir
    // degeri olcumun YOKLUGUNA cevirmek tam olarak kacindigimiz hata.
    uint32_t cachedEmissiveTriangles = 0, cachedEmissiveDropped = 0;
    uint32_t cachedEmissiveSkippedIndexed = 0, cachedEmissiveRejectedTransparent = 0;
    float cachedEmissiveArea = 0.0f;
};
namespace {
uint64_t hashBytes(uint64_t h, const void* data, size_t bytes) {
    const auto* p = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < bytes; ++i) h = (h ^ p[i]) * 1099511628211ull;
    return h;
}
float finitePositive(float f) { return std::isfinite(f) ? std::max(f, 0.0f) : 0.0f; }
}

bool VulkanBackendAdapter::setRayFusionProbeBounce(bool enabled) {
    if (!m_device || !m_device->hasHardwareRT()) return false;
    if (!m_rayFusionBounce) m_rayFusionBounce = std::make_shared<RayFusionBounceResources>();
    m_rayFusionBounce->status.requested = enabled;
    // ★★★★★ RASTER VIEWPORT'UN KAPISI `m_interactiveViewport.dirty`, ve bu
    //   satirin yoklugu 2026-09-15'te OLCULDU: `set_probe_bounce` "applied:
    //   True, requested: True" donuyor, sonra HICBIR SEY olmuyordu --
    //   `traced_publishes` 1201'de, ekran GI isimasi bes ondaliga kadar ayni,
    //   `bounce_active` False. Bir ayar uygulanmis ve kimse cizmemisti.
    //
    // ★★★★ Onceki partide bu, API katmaninda `g_ctx->start_render` ile
    //   "duzeltilmisti" -- YANLIS KOL. O bayrak yol izleyiciyi surer;
    //   raster viewport'un kendi kapisi `needsViewportRender()` ona hic
    //   bakmaz. Iki farkli "yeniden ciz" kavramini ayni isimle dusunmek,
    //   duzeltmenin kendisini gorunmez bir no-op yapti.
    //
    // ★★★ Tek kare de YETMEZ ama gerekli: alan butceyle sinirli izlendigi
    //   icin yakinsamasi cok kare surer. Onu `pump` tasiyor (pending > 0
    //   oldugu surece viewport cizmeye devam eder) -- ama pump ancak bir kare
    //   cizilip alan gecersiz kilindiktan SONRA kalkar. Bu satir o ilk kareyi
    //   uretir; zincirin geri kalani zaten calisiyordu.
    m_interactiveViewport.dirty = true;
    return true;
}

uint64_t VulkanBackendAdapter::prepareRayFusionBounce() {
    if (!m_rayFusionBounce) m_rayFusionBounce = std::make_shared<RayFusionBounceResources>();
    auto& s = *m_rayFusionBounce;
    auto& status = s.status;
    status.active = false;
    if (!m_device || !m_device->hasHardwareRT()) return 0;
    // Timed because this runs on EVERY raster frame and scales with the scene,
    // not with what the bounce is asked to do. An unmeasured per-frame cost is
    // how this path came to own the frame without appearing anywhere.
    const auto prepareStart = std::chrono::steady_clock::now();

    // Faz saatleri. Tek bir toplam "pahali" der, hangi fazin kapiya ihtiyaci
    // oldugunu soylemez -- ve yanlis faza kapi koymak bayatlama uretir.
    const auto phaseInstances = std::chrono::steady_clock::now();
    std::vector<RayFusion::HitInstance> instances;
    const bool geometryReady = getRayFusionHitInstances(instances);
    status.prepareInstancesMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - phaseInstances).count();
    // ★★★★★ Emissive UCGEN listesi MALZEME TABLOSUNDAN ONCE kurulur, cunku
    //   ciktisi tabloya GIRIYOR: `materialInNee`, hangi malzemenin emission'inin
    //   listede temsil edildigini soyler ve shader yarim kure isininda o
    //   emission'i EKLEMEZ. Sira ters olsa bayrak bir kare geriden gelirdi.
    //
    // ★★★ Bu dilim eksik olan bir yetenegi eklemiyor -- emissive yuzeylerin
    //   yakin alani aydinlatmasi zaten vardi (`rfBounceRadiance` emission
    //   donduruyor). Duzelttigi sey KESTIRICI: emissive geometri orneklenebilir
    //   bir isik olmadigi icin bir lamba ancak yarim kure isininin SANSINA
    //   bulunuyordu, ve belirtisi titreme DEGIL sabit bir noktali desendi.
    const auto phaseEmissive = std::chrono::steady_clock::now();
    std::vector<RayFusion::EmissiveTriangle> emissives;
    std::vector<uint8_t> materialInNee;
    // ★★★ Girdi imzasi: taramanin bagli oldugu HER SEY, ve hepsi ucuz.
    //   - geometry_signature : hangi mesh'ler var (ucgenler)
    //   - instance_signature : nerede duruyorlar (emissive ucgenler DUNYA
    //                          uzayinda tutuluyor, yani transform onemli)
    //   - malzeme baytlari   : hangi malzeme yayiyor ve ne kadar
    //   Ucunu de atlamak, "isik eski yerinden aydinlatiyor" seklinde sessizce
    //   bayat bir tablo demekti; o 8,89 ms'den beterdir.
    RayFusionSceneASStatus emissiveAs{};
    getRayFusionSceneASStatus(emissiveAs);
    uint64_t emissiveInputs = hashBytes(1469598103934665603ull,
                                        &emissiveAs.geometry_signature,
                                        sizeof(emissiveAs.geometry_signature));
    emissiveInputs = hashBytes(emissiveInputs, &emissiveAs.instance_signature,
                               sizeof(emissiveAs.instance_signature));
    if (!m_cachedGpuMaterials.empty())
        emissiveInputs = hashBytes(emissiveInputs, m_cachedGpuMaterials.data(),
                                   m_cachedGpuMaterials.size() * sizeof(m_cachedGpuMaterials[0]));
    const bool emissiveCacheHit =
        s.emissiveCacheValid && emissiveInputs == s.emissiveInputSignature;
    if (emissiveCacheHit) {
        emissives = s.cachedEmissives;
        materialInNee = s.cachedMaterialInNee;
        status.emissiveTriangles = s.cachedEmissiveTriangles;
        status.emissiveDropped = s.cachedEmissiveDropped;
        status.emissiveSkippedIndexed = s.cachedEmissiveSkippedIndexed;
        status.emissiveRejectedTransparent = s.cachedEmissiveRejectedTransparent;
        status.emissiveArea = s.cachedEmissiveArea;
    } else {
        getRayFusionEmissiveTriangles(emissives, materialInNee, status);
        s.cachedEmissives = emissives;
        s.cachedMaterialInNee = materialInNee;
        s.cachedEmissiveTriangles = status.emissiveTriangles;
        s.cachedEmissiveDropped = status.emissiveDropped;
        s.cachedEmissiveSkippedIndexed = status.emissiveSkippedIndexed;
        s.cachedEmissiveRejectedTransparent = status.emissiveRejectedTransparent;
        s.cachedEmissiveArea = status.emissiveArea;
        s.emissiveInputSignature = emissiveInputs;
        s.emissiveCacheValid = true;
    }
    status.emissiveCached = emissiveCacheHit;
    status.prepareEmissiveMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - phaseEmissive).count();
    const auto phaseMaterials = std::chrono::steady_clock::now();

    std::vector<RayFusion::BounceMaterial> materials;
    status.supportedMaterials = 0;
    status.unsupportedMaterials = 0;
    status.rejectedTextured = 0;
    status.rejectedTransparent = 0;
    status.rejectedLayered = 0;
    status.rejectedFlagged = 0;
    status.rejectedFlagBits = 0;
    for (const auto& m : m_cachedGpuMaterials) {
        RayFusion::BounceMaterial packed{};
        // This slice shades opaque (or alpha-masked) diffuse/emissive surfaces
        // and reads their albedo, emission, opacity, metallic and specular
        // textures. Unsupported hits remain occluders, never fake albedo.
        //
        // ★ Each clause is counted separately. The single "unsupported" count
        //   this started as could not be checked against the scene at all: a
        //   scene of 31 textured materials and a gate that rejects everything
        //   print the same number. Counting the CLAUSE makes the two differ.
        //
        // ★ Slots deliberately NOT disqualifying, and why they are honest
        //   omissions rather than silent approximations:
        //     roughness_tex - the bounce has a diffuse lobe only; roughness
        //                     never enters the result.
        //     normal_tex    - the probe texel is a cosine average over a
        //                     hemisphere; a tangent-space perturbation is below
        //                     its frequency. Geometric normal is documented.
        //     height_tex    - parallax/displacement, same argument.
        //   transmission_tex still disqualifies: it is transparency, and this
        //   slice cannot continue a ray through a surface.
        auto textureResident = [&](uint32_t slot) {
            if (slot == 0u) return true;
            const auto found = m_uploadedImages.find(slot);
            return found != m_uploadedImages.end() && found->second.view && found->second.sampler;
        };
        const bool textured = m.transmission_tex != 0u ||
            !textureResident(m.albedo_tex) || !textureResident(m.emission_tex) ||
            !textureResident(m.opacity_tex) || !textureResident(m.metallic_tex) ||
            !textureResident(m.specular_tex);
        // Negated rather than rewritten: a NaN must stay UNsupported, and
        // `x > 0.001f` is false for NaN while `!(x <= 0.001f)` is true.
        // An opacity TEXTURE is fine (binary alpha masking is implemented);
        // a scalar opacity below 1 is not, because uniform semi-transparency
        // needs the ray continued rather than cut.
        const bool cutout = (m.flags & MATERIAL_FLAGS_PREVIEW_CUTOUT) != 0u;
        const bool transparent = !std::isfinite(m.opacity) ||
            (!cutout && !(m.opacity >= 0.999f)) || !(m.transmission <= 0.001f);
        // Thin cutout foliage uses a low-frequency, two-sided diffuse
        // approximation. Keep refractive transmission and special materials
        // outside the subset; an alpha flag does not turn glass into a leaf.
        const bool thinFoliage = cutout && std::isfinite(m.subsurface_amount) &&
            std::isfinite(m.translucent) && std::isfinite(m.clearcoat);
        const bool layered = !thinFoliage && (!(m.subsurface_amount <= 0.001f) ||
            !(m.translucent <= 0.001f) || !(m.clearcoat <= 0.001f));
        // ★★★ NOT `m.flags != 0u`. That rejected EVERY material in EVERY
        //   scene: `flags` is not a set of material features, it is a word
        //   with three different meanings packed into it. Bit 21
        //   (RESIN_OBJ_SPACE) is the resin interior's COORDINATE SPACE and
        //   defaults to true on every material ever authored, so the whole
        //   word is never zero. Bits 8-15 are texture channel decode/selector
        //   bits, which the shader now READS rather than treating as a
        //   disqualifier. Only the bits that would actually change how this
        //   slice has to shade a hit disqualify it, and they are named rather
        //   than inferred from "non-zero".
        constexpr uint32_t kDisqualifyingFlags =
            VulkanRT::VK_MAT_FLAG_TERRAIN | VulkanRT::VK_MAT_FLAG_WATER |
            VulkanRT::VK_MAT_FLAG_WATER_FFT_READY | VulkanRT::VK_MAT_FLAG_BUBBLE |
            VulkanRT::VK_MAT_FLAG_MARBLE_VOLUME | VulkanRT::VK_MAT_FLAG_WATER_LAKE |
            VulkanRT::VK_MAT_FLAG_WATER_RIVER | VulkanRT::VK_MAT_FLAG_VOLUME;
        const uint32_t disqualifying = m.flags & kDisqualifyingFlags;
        const bool flagged = disqualifying != 0u;
        const bool supported = !textured && !transparent && !layered && !flagged;
        if (!supported) {
            if (textured) ++status.rejectedTextured;
            if (transparent) ++status.rejectedTransparent;
            if (layered) ++status.rejectedLayered;
            // Report the bits that MATTERED, not the whole word: the word
            // always carries bits this slice deliberately ignores.
            if (flagged) { ++status.rejectedFlagged; status.rejectedFlagBits |= disqualifying; }
        }
        // ★ The energy split moved to the SHADER. It used to be folded in here
        //   from the scalar metallic/specular, which was correct only while no
        //   texture could change them per texel; a metallic map would have been
        //   multiplied by the wrong constant with nothing to show for it.
        //   diffuse.rgb is now the raw tint the albedo texture multiplies.
        packed.diffuse[0] = std::min(finitePositive(m.albedo_r), 1.0f);
        packed.diffuse[1] = std::min(finitePositive(m.albedo_g), 1.0f);
        packed.diffuse[2] = std::min(finitePositive(m.albedo_b), 1.0f);
        packed.diffuse[3] = supported ? 1.0f : 0.0f;
        packed.emission[0] = finitePositive(m.emission_r) * finitePositive(m.emission_strength);
        packed.emission[1] = finitePositive(m.emission_g) * finitePositive(m.emission_strength);
        packed.emission[2] = finitePositive(m.emission_b) * finitePositive(m.emission_strength);
        packed.emission[3] = std::isfinite(m.opacity) ? std::clamp(m.opacity, 0.0f, 1.0f) : 1.0f;
        // ★★★★★ TEK SAYIM. Bu malzemenin emission'i emissive ucgen listesinde
        //   tam olarak temsil ediliyorsa shader onu yarim kure isininda
        //   EKLEMEZ -- NEE zaten bir kez ekliyor. Bayrak olmadan ayni isik iki
        //   kez sayilirdi ve belirtisi "lambalar iki kat parlak" degil, "GI
        //   biraz fazla sicak" olurdu. Kimse buna cift sayim demez.
        packed.emissive[0] = materials.size() < materialInNee.size() &&
            materialInNee[materials.size()] ? 1.0f : 0.0f;
        packed.textures[0] = m.albedo_tex;
        packed.textures[1] = m.emission_tex;
        // An unresident descriptor is a placeholder image, not the authored
        // alpha. Mark it out of range so coverage conservatively occludes.
        packed.textures[2] = textureResident(m.opacity_tex) ? m.opacity_tex : UINT32_MAX;
        packed.textures[3] = m.metallic_tex;
        packed.textures2[0] = m.specular_tex;
        // The WHOLE flags word travels, unmasked: the shader decodes the
        // packed-channel and opacity-in-alpha bits from it with the same
        // helpers the RT closest-hit uses. Masking here would fork the policy.
        packed.textures2[1] = m.flags;
        packed.textures2[2] = m.uv_wrap_mode;
        // uv_scale/uv_tiling of 0 would collapse every lookup onto one texel.
        // The raster path already guards this the same way; a bounce that
        // disagreed with it would read a different texel than the visible
        // surface shows, which is the "producer != consumer" failure class.
        auto nonZero = [](float v) { return (std::isfinite(v) && v != 0.0f) ? v : 1.0f; };
        packed.uvScaleOffset[0] = nonZero(m.uv_scale_x);
        packed.uvScaleOffset[1] = nonZero(m.uv_scale_y);
        packed.uvScaleOffset[2] = std::isfinite(m.uv_offset_x) ? m.uv_offset_x : 0.0f;
        packed.uvScaleOffset[3] = std::isfinite(m.uv_offset_y) ? m.uv_offset_y : 0.0f;
        packed.uvTiling[0] = nonZero(m.uv_tiling_x);
        packed.uvTiling[1] = nonZero(m.uv_tiling_y);
        packed.uvTiling[2] = std::isfinite(m.uv_rotation_degrees) ? m.uv_rotation_degrees : 0.0f;
        packed.uvTiling[3] = std::clamp(m.metallic, 0.0f, 1.0f);
        packed.scalars[0] = std::clamp(finitePositive(m.specular), 0.0f, 4.0f);
        // Binary cutout, not a dissolve: the slice either occludes or does not.
        packed.scalars[1] = 0.5f;
        packed.scalars[2] = thinFoliage ? std::clamp(m.translucent, 0.0f, 1.0f) : 0.0f;
        // Reserve the remaining scalar for the coat's diffuse energy loss.
        packed.scalars[3] = thinFoliage ? std::clamp(m.clearcoat, 0.0f, 1.0f) : 0.0f;
        packed.textures2[3] = thinFoliage ? 1u : 0u;
        if (supported) ++status.supportedMaterials; else ++status.unsupportedMaterials;
        materials.push_back(packed);
    }
    // ── Hair materyallerini bounce tablosuna ekle ─────────────────────────
    // Hair, RayFusion TLAS'ında AABB occluder olarak yer alır. Shader'da
    // AABB hit'inin customIndex'i >= hairMaterialOffset ise bu bir saç
    // hit'idir ve diffuse albedosu bu tablodan okunur.
    status.hairMaterials = 0;
    const uint32_t hairMaterialOffset = static_cast<uint32_t>(materials.size());
    if (!m_cachedHairGpuMaterials.empty()) {
        auto hairBounce = RayFusion::hairMaterialsToBounce(
            m_cachedHairGpuMaterials);
        for (auto& hm : hairBounce) {
            ++status.supportedMaterials;
            materials.push_back(hm);
        }
        status.hairMaterials = static_cast<uint32_t>(hairBounce.size());
    }
    std::vector<RayFusion::BounceLight> lights;
    status.unsupportedLights = 0;
    status.sunInBounce = false;
    status.sunTintFromLut = false;
    for (const auto& light : m_cachedLights) {
        if (!light || !light->visible) continue;
        const bool directional = light->type() == LightType::Directional;
        if ((!directional && light->type() != LightType::Point) || lights.size() >= 64u) {
            ++status.unsupportedLights;
            continue;
        }
        RayFusion::BounceLight packed{};
        packed.position[0] = light->position.x;
        packed.position[1] = light->position.y;
        packed.position[2] = light->position.z;
        packed.position[3] = directional ? 1.0f : 0.0f;
        packed.radiance[0] = finitePositive(light->color.x) * finitePositive(light->intensity);
        packed.radiance[1] = finitePositive(light->color.y) * finitePositive(light->intensity);
        packed.radiance[2] = finitePositive(light->color.z) * finitePositive(light->intensity);
        packed.direction[0] = -light->direction.x;
        packed.direction[1] = -light->direction.y;
        packed.direction[2] = -light->direction.z;
        lights.push_back(packed);
    }
    // ★★★★★ Physical Sky gunesi tabloya YONLU bir isik olarak girer. Sicrama
    //   shader'i zaten `position.w > 0.5` ile yonlu isigi ve golge isinini
    //   dogru isliyor; eksik olan sey isigin KENDISIYDI, yolu degil.
    {
        float sunRgb[3];
        bool fromLut = false;
        if (lights.size() < 64u && worldSunBounceRadiance(sunRgb, fromLut)) {
            const auto& sd = m_cachedWorld.nishita.sun_direction;
            const float len = std::sqrt(sd.x * sd.x + sd.y * sd.y + sd.z * sd.z);
            RayFusion::BounceLight packed{};
            packed.position[0] = packed.position[1] = packed.position[2] = 0.0f;
            packed.position[3] = 1.0f;   // yonlu
            packed.radiance[0] = sunRgb[0];
            packed.radiance[1] = sunRgb[1];
            packed.radiance[2] = sunRgb[2];
            // `direction` alani sahne isiklarinda `-light->direction` olarak
            // paketleniyor (isiga DOGRU vektor). worldSun.xyz zaten gunese
            // dogru baktigi icin burada isaret cevirmesi YOK -- bu iki farkli
            // sozlesme ve karistirmak golgeyi ters cevirirdi.
            packed.direction[0] = sd.x / len;
            packed.direction[1] = sd.y / len;
            packed.direction[2] = sd.z / len;
            lights.push_back(packed);
            status.sunInBounce = true;
            status.sunTintFromLut = fromLut;
        }
    }
    status.instances = static_cast<uint32_t>(instances.size());
    status.materials = static_cast<uint32_t>(materials.size());
    status.lights = static_cast<uint32_t>(lights.size());
    // Empty arrays still need valid bound storage descriptors on the alpha path.
    if (instances.empty()) instances.emplace_back();
    if (materials.empty()) materials.emplace_back();
    if (lights.empty()) lights.emplace_back();
    // Bos dizi de gecerli bir storage descriptor'a ihtiyac duyar. ★ Bos girisin
    //   alani 0 ve CDF'si 0: GPU tarafi `emissiveCount == 0` ile kapiyor, ama
    //   kapi kacsa bile sifir alanli bir ucgen isik tasimaz.
    if (emissives.empty()) emissives.emplace_back();
    uint64_t signature = hashBytes(1469598103934665603ull, instances.data(), instances.size() * sizeof(instances[0]));
    signature = hashBytes(signature, materials.data(), materials.size() * sizeof(materials[0]));
    signature = hashBytes(signature, lights.data(), lights.size() * sizeof(lights[0]));
    signature = hashBytes(signature, &status.requested, sizeof(status.requested));
    // Gunes yonu/siddeti imzaya girer: yoksa gunes hareket ettiginde sicrama
    // tablosu bayat kalir ve dolayli isik ESKI gunesten gelmeye devam ederdi.
    signature = hashBytes(signature, &m_cachedWorld.nishita.sun_direction,
                          sizeof(m_cachedWorld.nishita.sun_direction));
    signature = hashBytes(signature, &m_cachedWorld.nishita.sun_intensity,
                          sizeof(m_cachedWorld.nishita.sun_intensity));
    signature = hashBytes(signature, &geometryReady, sizeof(geometryReady));
    signature = hashBytes(signature, &status.lights, sizeof(status.lights));
    // ★ Emissive tablo imzaya KATILIR: bir lambanin emission'i degistiginde ya
    //   da bir obje tasindiginda (dunya uzayinda ucgenler) tablo bayat kalirdi
    //   ve belirtisi "isik eski yerinden aydinlatiyor" olurdu.
    signature = hashBytes(signature, emissives.data(), emissives.size() * sizeof(emissives[0]));
    status.prepareMaterialsMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - phaseMaterials).count();
    const auto phaseUpload = std::chrono::steady_clock::now();
    status.uploadSkipped = (signature == status.signature) && s.buffers[0].buffer;
    const void* data[4] = {instances.data(), materials.data(), lights.data(), emissives.data()};
    const uint64_t bytes[4] = {instances.size() * sizeof(instances[0]), materials.size() * sizeof(materials[0]), lights.size() * sizeof(lights[0]), emissives.size() * sizeof(emissives[0])};
    if (signature != status.signature || !s.buffers[0].buffer) {
        drainInteractiveViewportInFlight();
        for (uint32_t i = 0; i < 4; ++i) {
            if (!s.buffers[i].buffer || s.buffers[i].size < bytes[i]) {
                if (s.buffers[i].buffer) m_device->destroyBuffer(s.buffers[i]);
                VulkanRT::BufferCreateInfo ci;
                ci.size = bytes[i];
                ci.usage = VulkanRT::BufferUsage::STORAGE;
                ci.location = VulkanRT::MemoryLocation::CPU_TO_GPU;
                s.buffers[i] = m_device->createBuffer(ci);
            }
            if (!s.buffers[i].buffer) {
                status.ready = false;
                status.signature = 0;
                status.reason = "bounce table allocation failed";
                status.prepareMs = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - prepareStart).count();
                return 0;
            }
            m_device->uploadBuffer(s.buffers[i], data[i], bytes[i]);
        }
        status.signature = signature;
    }
    status.prepareUploadMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - phaseUpload).count();
    status.ready = geometryReady && status.materials > 0;
    if (!status.ready) {
        status.reason = "hit geometry/material table is not ready";
    } else if (status.supportedMaterials == 0u && status.materials > 0u) {
        // ★★★ The honest sentence when the slice matches nothing. "N materials
        //   unsupported" reads like a defect; what it actually means is that
        //   the bounce cannot change a single pixel of THIS scene, and that is
        //   what has to be said out loud -- otherwise enabling the checkbox
        //   produces an image identical to 1b-alpha and looks like a bug.
        status.reason = "No material in this scene is inside the 1b-beta subset, "
            "so no covered hit returns bounced light. Cutout holes still pass rays. "
            "Supported: textured opaque and thin cutout diffuse/emission.";
    } else {
        status.reason = status.sunInBounce
            ? "Textured opaque and two-sided cutout diffuse/emission; cutout SSS approximated by diffuse, translucency by a thin diffuse lobe, coat by energy attenuation. The Physical Sky sun IS carried as a directional bounce light. No refractive transmission, volumes, terrain layers, or area/spot bounce."
            : "Textured opaque and two-sided cutout diffuse/emission; cutout SSS approximated by diffuse, translucency by a thin diffuse lobe, coat by energy attenuation. No refractive transmission, volumes, terrain layers, area/spot or analytic sky-sun bounce.";
    }
    status.prepareMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - prepareStart).count();
    return status.requested ? status.signature : 0;
}

RayFusion::BounceStatus VulkanBackendAdapter::rayFusionBounceStatus() const {
    RayFusion::BounceStatus out =
        m_rayFusionBounce ? m_rayFusionBounce->status : RayFusion::BounceStatus{};
    // The GPU counters live with the trace resources, not with the tables --
    // they answer a different question and must not be inferred from the table
    // sizes. Merged here so one read reports both what the slice CAN shade and
    // what it ACTUALLY shaded.
    const RayFusion::BounceCounters counters = rayFusionBounceCounters();
    out.hits = counters.hits;
    out.shadedHits = counters.shaded;
    out.alphaTested = counters.alphaTested;
    out.alphaOccluded = counters.alphaPassed;
    out.backFaceShaded = counters.backFaceShaded;
    out.skippedBounceDisabled = counters.skippedBounceDisabled;
    out.rejectedUnresolved = counters.rejectedUnresolved;
    out.rejectedUnsupported = counters.rejectedUnsupported;
    out.rejectedDegenerate = counters.rejectedDegenerate;
    return out;
}
bool VulkanBackendAdapter::rayFusionBounceBuffers(VulkanRT::BufferHandle (&buffers)[4]) const {
    if (!m_rayFusionBounce) return false;
    for (uint32_t i = 0; i < 4; ++i) {
        buffers[i] = m_rayFusionBounce->buffers[i];
        if (!buffers[i].buffer) return false;
    }
    return true;
}
void VulkanBackendAdapter::destroyRayFusionBounce() {
    if (m_rayFusionBounce && m_device) {
        drainInteractiveViewportInFlight();
        for (auto& b : m_rayFusionBounce->buffers) if (b.buffer) m_device->destroyBuffer(b);
    }
    m_rayFusionBounce.reset();
}
}
