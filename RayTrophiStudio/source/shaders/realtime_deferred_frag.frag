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

layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput gbufferAlbedo;
layout(input_attachment_index = 1, set = 0, binding = 1) uniform subpassInput gbufferNormal;
layout(input_attachment_index = 2, set = 0, binding = 2) uniform subpassInput gbufferMaterial;
layout(input_attachment_index = 3, set = 0, binding = 3) uniform subpassInput gbufferDepth;

layout(location = 0) in vec2 vTexCoord;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform DeferredPushConstants {
    mat4 viewProjInverse;
    vec4 cameraPos;
    vec4 lightDir0; // xyz=dir, w=intensity
    vec4 lightDir1;
    vec4 lightDir2;
    // can add more lights or point lights here
} pc;

const float PI = 3.14159265359;

// GGX / Cook-Torrance BRDF (simplified)
float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness*roughness;
    float a2 = a*a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;
    float num = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
    return num / denom;
}

float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;
    float num = NdotV;
    float denom = NdotV * (1.0 - k) + k;
    return num / denom;
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2  = GeometrySchlickGGX(NdotV, roughness);
    float ggx1  = GeometrySchlickGGX(NdotL, roughness);
    return ggx1 * ggx2;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

void main() {
    float depth = subpassLoad(gbufferDepth).r;
    if (depth == 1.0) {
        // Background
        outColor = vec4(0.1, 0.1, 0.1, 1.0); // Simple dark background
        return;
    }

    vec4 albedo = subpassLoad(gbufferAlbedo);
    vec4 normEm = subpassLoad(gbufferNormal);
    vec4 mat = subpassLoad(gbufferMaterial);
    
    vec3 N = normalize(normEm.xyz);
    float roughness = mat.r;
    float metallic = mat.g;
    
    // Reconstruct world position from depth
    vec2 clipSpace = vTexCoord * 2.0 - 1.0;
    clipSpace.y = -clipSpace.y; // Vulkan Y is down
    vec4 clipPos = vec4(clipSpace, depth, 1.0);
    vec4 worldPosH = pc.viewProjInverse * clipPos;
    vec3 worldPos = worldPosH.xyz / worldPosH.w;
    
    vec3 V = normalize(pc.cameraPos.xyz - worldPos);
    
    vec3 F0 = vec3(0.04); 
    F0 = mix(F0, albedo.rgb, metallic);
    
    vec3 Lo = vec3(0.0);
    
    // Process 3 directional lights for now (like material preview, but we can feed scene lights from C++)
    vec4 lights[3] = vec4[](pc.lightDir0, pc.lightDir1, pc.lightDir2);
    
    for(int i = 0; i < 3; ++i) {
        if (lights[i].w <= 0.0) continue;
        
        vec3 L = normalize(lights[i].xyz);
        vec3 H = normalize(V + L);
        vec3 radiance = vec3(lights[i].w); // white light for now
        
        float NDF = DistributionGGX(N, H, roughness);
        float G   = GeometrySmith(N, V, L, roughness);
        vec3 F    = fresnelSchlick(max(dot(H, V), 0.0), F0);
        
        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;
        kD *= 1.0 - metallic;
        
        vec3 numerator    = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
        vec3 specular     = numerator / denominator;
        
        float NdotL = max(dot(N, L), 0.0);
        Lo += (kD * albedo.rgb / PI + specular) * radiance * NdotL;
    }
    
    vec3 ambient = vec3(0.03) * albedo.rgb;
    vec3 color = ambient + Lo;
    
    // Tone mapping
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0/2.2));
    
    outColor = vec4(color, 1.0);
}
