#ifndef MATERIAL_TEXTURE_ARRAYS_GLSL
#define MATERIAL_TEXTURE_ARRAYS_GLSL

#ifdef MATERIAL_TEXTURE_BINDLESS

// RT entry points declare materialTextures at binding 6. Expand at the call
// site to preserve their implicit/explicit LOD and non-uniform descriptor ABI.
#define materialTextureValid(slot) ((slot) != 0u)
#define materialTextureMissing(slot) ((slot) == 0u)
#define materialTextureFftValid(slot) ((slot) > 0u)
#define sampleMaterialTexture(slot, uv) texture(materialTextures[nonuniformEXT(slot)], uv)
#define sampleMaterialTextureLod(slot, uv, lod) textureLod(materialTextures[nonuniformEXT(slot)], uv, lod)

#else

// Each branch selects one constant descriptor. Only the array layer varies
// per invocation. Keep the page count and reference packing in sync with
// VulkanMaterialTexturePages.h / VulkanMaterialTextureBinding.h.
//
// Callers pass the RT material SLOT id stored in the shared material buffer (the
// same value the bindless path indexes with). The host's slot->ref table maps it
// to a packed (page << 16 | layer) reference; 0xffffffff = not paged / missing.
const uint MATERIAL_TEXTURE_PAGE_COUNT = 32u;

#ifndef MATERIAL_TEXTURE_REF_BINDING
#error "MATERIAL_TEXTURE_REF_BINDING must be defined with MATERIAL_TEXTURE_BINDING"
#endif
layout(set = 0, binding = MATERIAL_TEXTURE_REF_BINDING, std430)
readonly buffer MaterialTextureRefTable { uint materialTextureSlotToRef[]; };

uint materialTextureRef(uint slot) {
    return (slot != 0u && slot < uint(materialTextureSlotToRef.length()))
        ? materialTextureSlotToRef[slot] : 0xffffffffu;
}

bool materialTextureValid(uint slot) {
    return (materialTextureRef(slot) >> 16u) < MATERIAL_TEXTURE_PAGE_COUNT;
}

bool materialTextureMissing(uint slot) {
    return !materialTextureValid(slot);
}

#define materialTextureFftValid(ref) materialTextureValid(ref)

layout(set = 0, binding = MATERIAL_TEXTURE_BINDING)
uniform sampler2DArray materialTexturePages[32];

#define MATERIAL_TEXTURE_CASES(M) \
    M(0) \
    M(1) \
    M(2) \
    M(3) \
    M(4) \
    M(5) \
    M(6) \
    M(7) \
    M(8) \
    M(9) \
    M(10) \
    M(11) \
    M(12) \
    M(13) \
    M(14) \
    M(15) \
    M(16) \
    M(17) \
    M(18) \
    M(19) \
    M(20) \
    M(21) \
    M(22) \
    M(23) \
    M(24) \
    M(25) \
    M(26) \
    M(27) \
    M(28) \
    M(29) \
    M(30) \
    M(31)

vec4 sampleMaterialTextureLod(uint slot, vec2 uv, float lod) {
    const uint ref = materialTextureRef(slot);
    vec3 coord = vec3(uv, float(ref & 0xffffu));
    switch (ref >> 16u) {
#ifdef RT_SPEC_HAS_TEX_PAGES
#define SAMPLE_PAGE(i) case i: return (uint(i) < RT_SPEC_TEX_PAGES) ? textureLod(materialTexturePages[i], coord, lod) : vec4(1.0);
#else
#define SAMPLE_PAGE(i) case i: return textureLod(materialTexturePages[i], coord, lod);
#endif
        MATERIAL_TEXTURE_CASES(SAMPLE_PAGE)
#undef SAMPLE_PAGE
    default: return vec4(1.0);
    }
}

#ifdef MATERIAL_TEXTURE_FRAGMENT
vec4 sampleMaterialTextureGrad(uint slot, vec2 uv, vec2 dx, vec2 dy) {
    const uint ref = materialTextureRef(slot);
    vec3 coord = vec3(uv, float(ref & 0xffffu));
    switch (ref >> 16u) {
#ifdef RT_SPEC_HAS_TEX_PAGES
#define SAMPLE_PAGE(i) case i: return (uint(i) < RT_SPEC_TEX_PAGES) ? textureGrad(materialTexturePages[i], coord, dx, dy) : vec4(1.0);
#else
#define SAMPLE_PAGE(i) case i: return textureGrad(materialTexturePages[i], coord, dx, dy);
#endif
        MATERIAL_TEXTURE_CASES(SAMPLE_PAGE)
#undef SAMPLE_PAGE
    default: return vec4(1.0);
    }
}

vec4 sampleMaterialTexture(uint slot, vec2 uv) {
    // Compute derivatives before branching on a per-invocation page.
    return sampleMaterialTextureGrad(slot, uv, dFdx(uv), dFdy(uv));
}
#else
vec4 sampleMaterialTexture(uint slot, vec2 uv) {
    return sampleMaterialTextureLod(slot, uv, 0.0);
}
#endif

#undef MATERIAL_TEXTURE_CASES
#endif
#endif
