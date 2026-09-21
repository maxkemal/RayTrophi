// RayFusionHairBounce.h — Hair materyal verilerini RayFusion bounce tablosuna
// dönüştüren yardımcı arayüz. Hair'in RayFusion TLAS'ına opak bir AABB occluder
// olarak eklenmesini ve basit diffuse albedo ile golgelenmesini sağlar.
//
// Vulkan RT'deki referans: hair_closesthit.rchit + binding 10/11
// RayFusion'da: AABB BLAS → TLAS instance, HairGpuMaterial → BounceMaterial
#pragma once

#include "RayFusion/ProbeBounce.h"
#include "Backend/VulkanBackend.h"
#include <vector>
#include <cmath>
#include <algorithm>

namespace RayFusion {

// ★★★★★ Hair materyalini bounce tablosuna "basit diffuse" olarak dönüştürür.
//   Tam Marschner BSDF rayQuery compute shader'da pratik değildir (procedural
//   intersection yok); ama saçın dolaylı aydınlatmadan tamamen kaybolması da
//   kabul edilemez. Bu dönüşüm, saçın görünür rengini (baseColor ya da
//   melanin'den türetilen renk) diffuse albedo olarak taşır ve saçı bir opak
//   occluder yapar.
//
// Renk modu dönüşümü (hair_closesthit.rchit ile BIREBIR):
//   0 (Direct) → baseColor
//   1 (Melanin) → exp(-melanin_to_absorption(melanin, redness) * 0.5)
//   2 (Absorption) → exp(-absorption * 0.5)
//   3 (Root UV Map) → baseColor (UV mapping yok, skalar fallback)

inline BounceMaterial hairToBounce(
    const VulkanRT::HairGpuMaterial& hair)
{
    BounceMaterial m{};

    // ── Derive visible hair color ────────────────────────────────────────
    // Matches hair_closesthit.rchit color derivation (mode 0/1/2/3).
    float r = 0.0f, g = 0.0f, b = 0.0f;

    if (hair.colorMode == 1u) {
        // Melanin → absorption → color
        // melanin_to_absorption from hair_bsdf.cuh / hair_closesthit.rchit:
        //   eumelanin = melanin * (1 - melaninRedness)
        //   pheomelanin = melanin * melaninRedness
        //   sigma_a = eumelanin * vec3(0.506, 0.841, 1.653)
        //           + pheomelanin * vec3(0.343, 0.733, 1.924)
        float mel = std::clamp(hair.melanin, 0.0f, 1.0f);
        float red = std::clamp(hair.melaninRedness, 0.0f, 1.0f);
        float eu  = mel * (1.0f - red);
        float ph  = mel * red;
        float sa_r = eu * 0.506f + ph * 0.343f;
        float sa_g = eu * 0.841f + ph * 0.733f;
        float sa_b = eu * 1.653f + ph * 1.924f;
        r = std::exp(-sa_r * 0.5f);
        g = std::exp(-sa_g * 0.5f);
        b = std::exp(-sa_b * 0.5f);
    } else if (hair.colorMode == 2u) {
        // Explicit absorption
        r = std::exp(-hair.absorption[0] * 0.5f);
        g = std::exp(-hair.absorption[1] * 0.5f);
        b = std::exp(-hair.absorption[2] * 0.5f);
    } else {
        // Direct color (0) or Root UV Map (3) — use baseColor directly
        r = (std::max)(hair.baseColor[0], 0.001f);
        g = (std::max)(hair.baseColor[1], 0.001f);
        b = (std::max)(hair.baseColor[2], 0.001f);
    }

    // Apply tint (same as hair_closesthit.rchit tint logic)
    if (hair.tint > 0.001f) {
        float t = std::clamp(hair.tint, 0.0f, 1.0f);
        r = r * (1.0f - t) + hair.tintColor[0] * t;
        g = g * (1.0f - t) + hair.tintColor[1] * t;
        b = b * (1.0f - t) + hair.tintColor[2] * t;
    }

    // ── Pack into BounceMaterial ──────────────────────────────────────────
    m.diffuse[0] = std::clamp(r, 0.0f, 1.0f);
    m.diffuse[1] = std::clamp(g, 0.0f, 1.0f);
    m.diffuse[2] = std::clamp(b, 0.0f, 1.0f);
    m.diffuse[3] = 1.0f;  // supported = true

    // Emission
    float eStr = std::isfinite(hair.emissionStrength)
        ? (std::max)(hair.emissionStrength, 0.0f) : 0.0f;
    m.emission[0] = (std::max)(hair.emission[0], 0.0f) * eStr;
    m.emission[1] = (std::max)(hair.emission[1], 0.0f) * eStr;
    m.emission[2] = (std::max)(hair.emission[2], 0.0f) * eStr;
    m.emission[3] = 1.0f;  // opacity = 1 (saç opak)

    // No textures for hair bounce (procedural shading)
    // textures[0..3] = 0 (no albedo/emission/opacity/metallic texture)
    // textures2[0..3] = 0 (no specular/flags/wrap/thin)

    // UV identity (unused, but well-formed)
    m.uvScaleOffset[0] = 1.0f;
    m.uvScaleOffset[1] = 1.0f;
    m.uvScaleOffset[2] = 0.0f;
    m.uvScaleOffset[3] = 0.0f;
    m.uvTiling[0] = 1.0f;
    m.uvTiling[1] = 1.0f;
    m.uvTiling[2] = 0.0f;
    m.uvTiling[3] = 0.0f;  // metallic = 0 (saç metalik değil)

    // Scalars
    m.scalars[0] = 0.5f;  // specular (modest)
    m.scalars[1] = 0.5f;  // alpha cutoff (unused)
    m.scalars[2] = 0.0f;  // thin transmission = 0
    m.scalars[3] = 0.0f;  // coat = 0

    // Not in emissive NEE list
    m.emissive[0] = 0.0f;

    return m;
}

// Toplu dönüşüm. Her HairGpuMaterial girişi bir BounceMaterial çıkışına dönüşür.
inline std::vector<BounceMaterial> hairMaterialsToBounce(
    const std::vector<VulkanRT::HairGpuMaterial>& hairMats)
{
    std::vector<BounceMaterial> out;
    out.reserve(hairMats.size());
    for (const auto& h : hairMats) {
        out.push_back(hairToBounce(h));
    }
    return out;
}

} // namespace RayFusion
