#version 450

layout(location = 0) in vec2 vUV;
layout(location = 1) in vec4 vColor;
layout(location = 2) flat in vec4 vShade;   // roughness, lit (particle_appearance_lut.glsl)
layout(location = 3) flat in vec3 vL1;      // strongest light, billboard frame
layout(location = 4) flat in vec3 vC1;      // its relative radiance
layout(location = 5) flat in vec3 vL2;
layout(location = 6) flat in vec3 vC2;
layout(location = 7) flat in vec3 vAmb;     // the other lights + bounce
layout(location = 8) flat in vec3 vUp;      // world up, billboard frame
layout(location = 0) out vec4 outColor;

const float PI = 3.14159265;

// GGX highlight of one light, coloured by it.
vec3 lightSpec(vec3 n, vec3 L, vec3 C, float rough) {
    if (dot(C, C) <= 0.0) return vec3(0.0);
    vec3 V = vec3(0.0, 0.0, 1.0);
    float a = rough * rough;
    float a2 = a * a;
    float ndl = max(dot(n, L), 0.0);
    float ndv = max(n.z, 1e-3);
    vec3 H = normalize(L + V);
    float ndh = max(dot(n, H), 0.0);
    float dd = ndh * ndh * (a2 - 1.0) + 1.0;
    float D = a2 / (PI * dd * dd);
    float k = (rough + 1.0) * (rough + 1.0) / 8.0;
    float G = (ndl / (ndl * (1.0 - k) + k)) * (ndv / (ndv * (1.0 - k) + k));
    float F = 0.04 + 0.96 * pow(1.0 - max(dot(H, V), 0.0), 5.0);
    return C * min(D * G * F / (4.0 * ndl * ndv + 1e-4) * ndl, 10.0);
}

void main() {
    float d2 = dot(vUV, vUV);
    if (d2 > 1.0) {
        discard;
    }
    if (vShade.y < 0.5) {
        // Flat look (emissive systems): round soft sprite, radial falloff.
        float falloff = 1.0 - d2;
        falloff *= falloff;
        outColor = vec4(vColor.rgb, vColor.a * falloff);
        return;
    }
    // Lit liquid bead (2026-11-18). The normal is a FLATTENED sphere: a full
    // hemisphere read as a hard ball and broke the look of flowing paint; a
    // lens-like bead keeps the volume cue and the highlight without the
    // dark terminator. Lighting from the scene's lights weighed per particle
    // in the vertex stage (two direct + coloured ambient), relative to the room.
    vec3 n = normalize(vec3(vUV * 0.8, sqrt(max(0.0, 1.0 - d2)) + 0.35));
    vec3 V = vec3(0.0, 0.0, 1.0);
    vec3 up = normalize(vUp);
    vec3 albedo = vColor.rgb;
    float rough = clamp(vShade.x, 0.03, 1.0);
    vec3 L1 = normalize(vL1);
    vec3 L2 = normalize(vL2);

    float w1 = max((dot(n, L1) + 0.5) / 1.5, 0.0);   // wrap: liquid, not chalk
    float w2 = max((dot(n, L2) + 0.5) / 1.5, 0.0);
    float hemi = 0.5 + 0.5 * dot(n, up);
    vec3 diffuse = albedo * (vAmb * mix(0.75, 1.15, hemi) + vC1 * w1 + vC2 * w2 + 0.06);

    vec3 spec = lightSpec(n, L1, vC1, rough) + lightSpec(n, L2, vC2, rough);

    float ndv = max(n.z, 1e-3);
    float Fv = 0.04 + 0.96 * pow(1.0 - ndv, 5.0);
    vec3 R = reflect(-V, n);
    float sky = 0.5 + 0.5 * dot(R, up);
    float level = clamp(dot(vC1 + vC2 + vAmb, vec3(0.333)), 0.3, 1.6);
    vec3 env = mix(vec3(0.10), vec3(0.85), sky) * level * Fv * (1.0 - 0.7 * rough);

    vec3 color = min(diffuse * (1.0 - Fv) + spec + env, vec3(4.0));
    // Hard liquid edge, antialiased over about one pixel.
    float w = max(fwidth(d2), 1e-4);
    float edge = 1.0 - smoothstep(1.0 - 1.5 * w, 1.0, d2);
    outColor = vec4(color, vColor.a * edge);
}
