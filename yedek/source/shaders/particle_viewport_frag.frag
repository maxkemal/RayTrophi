#version 450

layout(location = 0) in vec2 vUV;
layout(location = 1) in vec4 vColor;
layout(location = 0) out vec4 outColor;

void main() {
    // Round soft sprite: discard outside unit disc, smooth radial falloff.
    float d2 = dot(vUV, vUV);
    if (d2 > 1.0) {
        discard;
    }
    float falloff = 1.0 - d2;          // soft edge toward the rim
    falloff *= falloff;                // a touch sharper core
    outColor = vec4(vColor.rgb, vColor.a * falloff);
}
