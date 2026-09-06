#version 450

layout(location = 0) in vec2 vNdc;
layout(location = 0) out vec4 outColor;

// ★★★ Gokyuzu ile nesneler AYNI goruntuleme donusumunden gecmek zorunda.
//   Eskiden gokyuzu ACES + pow(1/2.2), nesneler ACES + pow(1/2.2) idi ama
//   ikisi ayri kopyalardi ve biri degistiginde oteki sessizce geride kalirdi.
#include "post_chain.glsl"

layout(set = 0, binding = 6, std430) readonly buffer PreviewSceneGlobalsBuffer {
    uint sceneLightCount;
    uint sceneFlags;
    uint shadowedLightCount;
    uint worldMode;
    vec4 worldColor;
    vec4 worldParams;
    vec4 worldSun;
    vec4 atmosphereA;      // multi enabled, factor, mie anisotropy, mie density
    vec4 atmosphereB;      // planet radius, atmosphere height, reserved
    // ★★ material_preview_frag.frag ile AYNI buffer. Duzen orada da yazili;
    //   birini degistirip otekini birakmak gokyuzunu nesnelerden farkli
    //   pozlar ve bu ekranda "sanatsal" gorunur.
    vec4 postA;            // x=exposure y=gamma z=saturation w=colorTemperature
    vec4 postB;            // x=vignetteStrength y=toneMappingType z=vignetteEnabled
    vec4 postC;            // x=viewportWidth y=viewportHeight
};

RtPostParams previewPostParams() {
    RtPostParams p;
    p.exposure         = postA.x;
    p.gamma            = postA.y;
    p.saturation       = postA.z;
    p.colorTemperature = postA.w;
    p.vignetteStrength = postB.x;
    p.toneMapping      = uint(postB.y + 0.5);
    p.vignetteEnabled  = uint(postB.z + 0.5);
    // ★ postB.w eskiden bostu; kamera pozlamasi oraya girdi, yani globals
    //   ABI'si (112 bayt) DEGISMEDI.
    p.cameraExposure   = postB.w;
    return p;
}
layout(set = 0, binding = 9) uniform sampler2D worldEnvironment;
layout(set = 0, binding = 10) uniform sampler2D atmosphereTransmittance;
layout(set = 0, binding = 11) uniform sampler2D atmosphereSkyView;
layout(set = 0, binding = 12) uniform sampler2D atmosphereMultiScatter;

layout(push_constant) uniform MaterialPreviewPushConstants {
    mat4 viewProj;
    mat4 view;
    vec4 cameraPos;       // w = viewport aspect for the sky pass
    vec4 lightDir0;       // w = tan(vertical fov / 2) for the sky pass
    vec4 lightDir1;
    vec4 lightDir2;
    uvec4 materialMeta;
} pc;

const float PI = 3.14159265359;

vec2 environmentUV(vec3 d, float rotation) {
    float phi = atan(d.z, d.x) - rotation;
    return vec2(fract(phi / (2.0 * PI) + 0.5),
                acos(clamp(d.y, -1.0, 1.0)) / PI);
}

vec3 blendOverlay(vec3 base, vec3 sampled) {
    float strength = max(worldParams.y, 0.0);
    float amount = min(strength, 1.0);
    vec3 overlay = sampled * strength;
    int mode = int(worldParams.w + 0.5);
    if (mode == 1) return base * mix(vec3(1.0), sampled, amount);
    if (mode == 2) return base + overlay;
    if (mode == 3) return overlay;
    return mix(base, overlay, amount);
}

vec3 nishitaSky(vec3 d) {
    vec3 sky;
    if ((sceneFlags & 8u) != 0u) {
        float azimuth = atan(d.z, d.x) / (2.0 * PI);
        if (azimuth < 0.0) azimuth += 1.0;
        sky = texture(atmosphereSkyView,
                      vec2(azimuth, (1.0 - clamp(d.y, -1.0, 1.0)) * 0.5)).rgb;
        if (atmosphereA.x > 0.5) {
            vec3 scatteringAlbedo = vec3(0.8, 0.85, 0.9);
            vec3 secondOrder = sky * scatteringAlbedo * 0.5 * exp(-0.5 * 0.3);
            vec3 thirdOrder = secondOrder * scatteringAlbedo * 0.25 * exp(-0.5 * 0.1);
            sky += secondOrder * atmosphereA.y + thirdOrder * (atmosphereA.y * 0.5);
        }
    } else {
        float up = clamp(d.y * 0.5 + 0.5, 0.0, 1.0);
        vec3 horizon = vec3(0.42, 0.53, 0.68);
        vec3 zenith = vec3(0.09, 0.24, 0.52);
        vec3 ground = max(worldColor.rgb, vec3(0.025));
        sky = mix(ground, mix(horizon, zenith, pow(up, 0.65)),
                  smoothstep(0.0, 0.12, d.y));
        sky *= max(worldParams.z / 10.0, 0.0);
    }

    vec3 sunDir = dot(worldSun.xyz, worldSun.xyz) > 1e-8
        ? normalize(worldSun.xyz) : vec3(0.0, 1.0, 0.0);
    float sunSize = max(worldColor.w, 0.05);
    float elevation = degrees(asin(clamp(sunDir.y, -1.0, 1.0)));
    if (elevation < 15.0)
        sunSize *= 1.0 + (15.0 - max(elevation, -10.0)) * 0.04;
    float radius = radians(sunSize * 0.5);
    float mu = dot(d, sunDir);
    if (mu > cos(radius) && worldSun.w > 0.0) {
        float radial = acos(clamp(mu, -1.0, 1.0)) / max(radius, 1e-6);
        float limb = 1.0 - 0.6 * (1.0 - sqrt(max(0.0, 1.0 - radial * radial)));
        float edge = 1.0 - smoothstep(0.85, 1.0, radial);
        vec3 transSun = vec3(1.0);
        if ((sceneFlags & 8u) != 0u) {
            float u = clamp((max(0.01, sunDir.y) + 0.2) / 1.2, 0.0, 1.0);
            float planetRadius = max(atmosphereB.x, 1.0);
            float altitude = max(0.0, length(pc.cameraPos.xyz + vec3(0.0, planetRadius, 0.0)) - planetRadius);
            float v = clamp(altitude / max(atmosphereB.y, 1.0), 0.0, 1.0);
            transSun = texture(atmosphereTransmittance, vec2(u, v)).rgb;
        }
        sky += transSun * worldSun.w * 80000.0 * limb * edge;
    }
    if ((sceneFlags & 4u) != 0u)
        sky = blendOverlay(sky, texture(worldEnvironment, environmentUV(d, worldParams.x)).rgb);
    return sky;
}

void main() {
    float tanHalfFov = max(pc.lightDir0.w, 0.001);
    vec3 viewDir = normalize(vec3(vNdc.x * max(pc.cameraPos.w, 0.001) * tanHalfFov,
                                  -vNdc.y * tanHalfFov, -1.0));
    vec3 d = normalize(transpose(mat3(pc.view)) * viewDir);

    vec3 sky;
    if (worldMode == 1u && (sceneFlags & 2u) != 0u)
        sky = texture(worldEnvironment, environmentUV(d, worldParams.x)).rgb * max(worldParams.y, 0.0);
    else if (worldMode == 2u)
        sky = nishitaSky(d);
    else
        sky = max(worldColor.rgb * worldColor.w, vec3(0.0));

    outColor = vec4(rtApplyPost(sky, previewPostParams(),
                                gl_FragCoord.xy / max(postC.xy, vec2(1.0))), 1.0);
}
