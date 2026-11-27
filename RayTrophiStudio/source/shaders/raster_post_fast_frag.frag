#version 450

#include "post_chain.glsl"

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D inHDR;

layout(push_constant) uniform Push {
    uint  width;
    uint  height;

    float postExposure;
    float postGamma;
    float postSaturation;
    float postColorTemperature;
    float postVignetteStrength;
    uint  postToneMapping;
    uint  postVignetteEnabled;
    float postCameraExposure;

    float focusDist;
    float lensRadius;
    float tanHalfFovY;
    float depthA;
    float depthB;
    float depthC;
    float maxCoC;
    uint  maxTaps;

    uint  aerialEnabled;
    float fogDistance;
} pc;

void main() {
    vec4 center = texture(inHDR, vUV);

    RtPostParams p;
    p.exposure         = pc.postExposure;
    p.gamma            = pc.postGamma;
    p.saturation       = pc.postSaturation;
    p.colorTemperature = pc.postColorTemperature;
    p.vignetteStrength = pc.postVignetteStrength;
    p.toneMapping      = pc.postToneMapping;
    p.vignetteEnabled  = pc.postVignetteEnabled;
    p.cameraExposure   = pc.postCameraExposure;

    vec3 srgb = rtApplyPost(center.rgb, p, vUV);
    outColor = vec4(srgb, clamp(center.a, 0.0, 1.0));
}
