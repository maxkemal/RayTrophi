#version 450

layout(location = 0) in vec3 vWorldPos;
layout(location = 1) in vec3 vWorldNormal;
layout(location = 2) in vec2 vUV;
layout(location = 3) flat in uint vMaterialID;

layout(location = 0) out vec4 outColor;

// Material buffer (matches VkGpuMaterial layout — 15 blocks x 16 bytes = 240 bytes)
struct GpuMaterial {
    // Block 1
    float albedo_r, albedo_g, albedo_b, opacity;
    // Block 2
    float emission_r, emission_g, emission_b, emission_strength;
    // Block 3
    float roughness, metallic, ior, transmission;
    // Block 4
    float subsurface_r, subsurface_g, subsurface_b, subsurface_amount;
    // Block 5
    float subsurface_radius_r, subsurface_radius_g, subsurface_radius_b, subsurface_scale;
    // Block 6
    float clearcoat, clearcoat_roughness, translucent, subsurface_anisotropy;
    // Block 7
    float anisotropic, sheen, sheen_tint;
    uint flags;
    // Block 8
    float fft_amplitude, fft_time_scale, micro_detail_strength, micro_detail_scale;
    // Block 9
    float foam_threshold, fft_ocean_size, fft_choppiness, fft_wind_speed;
    // Block 10
    float micro_anim_speed, micro_morph_speed, foam_noise_scale, fft_wind_direction;
    // Block 11
    float uv_scale_x, uv_scale_y, uv_offset_x, uv_offset_y;
    // Block 12
    float uv_rotation_degrees, uv_tiling_x, uv_tiling_y;
    uint uv_wrap_mode;
    // Block 13
    uint albedo_tex, normal_tex, roughness_tex, metallic_tex;
    // Block 14
    uint emission_tex, height_tex, opacity_tex, transmission_tex;
    // Block 15
    float subsurface_ior;
    uint _terrain_layer_idx;
    float normal_strength;
    uint _reserved;
};

layout(set = 0, binding = 0, std430) readonly buffer MaterialBuffer {
    GpuMaterial materials[];
};

layout(push_constant) uniform MaterialPreviewPushConstants {
    mat4 viewProj;
    mat4 view;
    vec4 cameraPos;
    vec4 lightDir0;   // key light
    vec4 lightDir1;   // fill light
    vec4 lightDir2;   // rim light
} pc;

// ── PBR Helpers ──

const float PI = 3.14159265359;

float DistributionGGX(float NdotH, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom + 1e-7);
}

float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k + 1e-7);
}

float GeometrySmith(float NdotV, float NdotL, float roughness) {
    return GeometrySchlickGGX(NdotV, roughness) * GeometrySchlickGGX(NdotL, roughness);
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(max(1.0 - cosTheta, 0.0), 5.0);
}

// Simple tone mapping (ACES approximation)
vec3 acesTonemap(vec3 x) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// Linear to sRGB
vec3 linearToSRGB(vec3 c) {
    return pow(c, vec3(1.0 / 2.2));
}

void main() {
    vec3 N = normalize(vWorldNormal);
    vec3 V = normalize(pc.cameraPos.xyz - vWorldPos);

    // Fetch material
    GpuMaterial mat = materials[vMaterialID];

    vec3 albedo = vec3(mat.albedo_r, mat.albedo_g, mat.albedo_b);
    float roughness = clamp(mat.roughness, 0.04, 1.0);
    float metallic = clamp(mat.metallic, 0.0, 1.0);
    vec3 emission = vec3(mat.emission_r, mat.emission_g, mat.emission_b) * mat.emission_strength;
    float opacity = mat.opacity;

    // F0 (base reflectance)
    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    vec3 Lo = vec3(0.0);

    // 3-point light setup
    vec3 lightDirs[3] = vec3[](
        normalize(pc.lightDir0.xyz),
        normalize(pc.lightDir1.xyz),
        normalize(pc.lightDir2.xyz)
    );
    float lightIntensities[3] = float[](
        pc.lightDir0.w,
        pc.lightDir1.w,
        pc.lightDir2.w
    );
    vec3 lightColors[3] = vec3[](
        vec3(1.0, 0.98, 0.95),   // key: warm white
        vec3(0.75, 0.82, 0.95),  // fill: cool blue
        vec3(0.90, 0.90, 0.95)   // rim: neutral
    );

    for (int i = 0; i < 3; ++i) {
        vec3 L = lightDirs[i];
        vec3 H = normalize(V + L);

        float NdotL = max(dot(N, L), 0.0);
        float NdotV = max(dot(N, V), 0.001);
        float NdotH = max(dot(N, H), 0.0);
        float HdotV = max(dot(H, V), 0.0);

        // Cook-Torrance specular BRDF
        float D = DistributionGGX(NdotH, roughness);
        float G = GeometrySmith(NdotV, NdotL, roughness);
        vec3 F = fresnelSchlick(HdotV, F0);

        vec3 specular = (D * G * F) / (4.0 * NdotV * NdotL + 0.0001);

        // Diffuse (energy conservation)
        vec3 kD = (1.0 - F) * (1.0 - metallic);
        vec3 diffuse = kD * albedo / PI;

        vec3 radiance = lightColors[i] * lightIntensities[i];
        Lo += (diffuse + specular) * radiance * NdotL;
    }

    // Ambient (simple hemisphere)
    vec3 ambientUp = vec3(0.15, 0.17, 0.22);    // sky
    vec3 ambientDown = vec3(0.08, 0.06, 0.05);  // ground
    float ambientBlend = N.y * 0.5 + 0.5;
    vec3 ambient = mix(ambientDown, ambientUp, ambientBlend) * albedo * (1.0 - metallic * 0.5);

    // Clearcoat (simplified)
    if (mat.clearcoat > 0.0) {
        vec3 H_key = normalize(V + lightDirs[0]);
        float ccNdotH = max(dot(N, H_key), 0.0);
        float ccD = DistributionGGX(ccNdotH, mat.clearcoat_roughness);
        float ccF = 0.04 + 0.96 * pow(1.0 - max(dot(H_key, V), 0.0), 5.0);
        Lo += vec3(ccD * ccF * mat.clearcoat * max(dot(N, lightDirs[0]), 0.0) * lightIntensities[0]);
    }

    vec3 color = ambient + Lo + emission;

    // Tone map and gamma correct
    color = acesTonemap(color);
    color = linearToSRGB(color);

    outColor = vec4(color, opacity);
}
