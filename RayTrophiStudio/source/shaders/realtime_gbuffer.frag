// ===========================================================================
// KULLANILMIYOR - REFERANS (2026-08-31)
// Ayri "Realtime" deferred viewport modu sokuldu; bu shader'i yukleyen kod
// artik yok. Gerekce: docs/dev/REALTIME_RENDERER_ROADMAP.md
// Deferred'a donuldugunde baslangic noktasi olsun diye birakildi.
// UYARI: eski hali depth image'i input attachment olarak okuyordu, ama
// depthImage INPUT_ATTACHMENT usage biti olmadan olusturuluyor -- geri
// baglayan kisi once orayi duzeltmeli.
// ===========================================================================
#version 450

// Material struct matching the C++ layout
struct MaterialData {
    vec4 baseColor;
    vec4 emission;
    float roughness;
    float metallic;
    float specular;
    float transmission;
    float ior;
    float clearcoat;
    float clearcoatRoughness;
    float opacity;
    int albedoMap;
    int normalMap;
    int roughnessMap;
    int metallicMap;
    int emissionMap;
    int alphaMap;
    uint pad1;
    uint pad2;
};

// SSBO for Materials
layout(set = 0, binding = 0, std430) readonly buffer MaterialSSBO {
    MaterialData materials[];
};

// Textures
layout(set = 0, binding = 1) uniform sampler2D textures[1024];

layout(location = 0) in vec3 vWorldNormal;
layout(location = 1) flat in uint vMaterialID;
layout(location = 2) in vec2 vTexCoord;
layout(location = 3) in vec3 vWorldPos;

layout(location = 0) out vec4 outAlbedo;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outMaterial;

void main() {
    MaterialData mat = materials[vMaterialID];
    
    vec4 baseColor = mat.baseColor;
    if (mat.albedoMap >= 0) {
        vec4 texColor = texture(textures[mat.albedoMap], vTexCoord);
        baseColor.rgb *= texColor.rgb;
        baseColor.a *= texColor.a;
    }
    if (baseColor.a < 0.1) discard;

    vec3 normal = normalize(vWorldNormal);
    if (mat.normalMap >= 0) {
        vec3 tNormal = texture(textures[mat.normalMap], vTexCoord).xyz * 2.0 - 1.0;
        // Basic tangent space (approximate if no tangent vector provided)
        vec3 dp1 = dFdx(vWorldPos);
        vec3 dp2 = dFdy(vWorldPos);
        vec2 duv1 = dFdx(vTexCoord);
        vec2 duv2 = dFdy(vTexCoord);
        vec3 N = normal;
        vec3 dp2perp = cross(dp2, N);
        vec3 dp1perp = cross(N, dp1);
        vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
        vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;
        float invmax = inversesqrt(max(dot(T,T), dot(B,B)));
        mat3 tbn = mat3(T * invmax, B * invmax, N);
        normal = normalize(tbn * tNormal);
    }

    float roughness = mat.roughness;
    if (mat.roughnessMap >= 0) roughness *= texture(textures[mat.roughnessMap], vTexCoord).r;
    
    float metallic = mat.metallic;
    if (mat.metallicMap >= 0) metallic *= texture(textures[mat.metallicMap], vTexCoord).r;
    
    vec3 emission = mat.emission.rgb;
    if (mat.emissionMap >= 0) emission *= texture(textures[mat.emissionMap], vTexCoord).rgb;

    outAlbedo = vec4(baseColor.rgb, 1.0);
    // Encode emission magnitude in normal w, or just pack it in another target.
    // Let's use outNormal.xyz for normal, w for emission magnitude.
    float emMag = max(emission.r, max(emission.g, emission.b));
    outNormal = vec4(normal, emMag);
    
    // outMaterial: R = roughness, G = metallic, B = specular
    outMaterial = vec4(roughness, metallic, mat.specular, 1.0);
}
