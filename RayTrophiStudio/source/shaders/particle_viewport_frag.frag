#version 450

layout(location = 0) in vec2 vUV;
layout(location = 1) in vec4 vColor;
layout(location = 2) flat in vec4 vShade;   // roughness or flat sprite style, lit
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

vec4 fireworkSprite(float d2, bool hasRays, float streak) {
    float radius = sqrt(min(d2, 1.0));
    float core = 0.86 * (1.0 - smoothstep(0.07, 0.32, radius));
    float halo = 0.24 * (1.0 - radius) * (1.0 - radius);
    float rays = 0.0;
    if (hasRays) {
        vec2 axis = abs(vec2(vUV.x + vUV.y, vUV.x - vUV.y)) * 0.70710678;
        float across = min(axis.x, axis.y);
        float along = max(axis.x, axis.y);
        float width = mix(0.095, 0.030, clamp(along / 0.78, 0.0, 1.0));
        float rayMask = (1.0 - smoothstep(width * 0.25, width, across)) *
                        (1.0 - smoothstep(0.10, 0.78, along));
        rays = 0.30 * rayMask * smoothstep(0.56, 0.72, vColor.a);
    }

    float strength = max(max(vColor.r, vColor.g), vColor.b);
    vec3 hue = vColor.rgb / max(strength, 1e-5);
    vec3 coreTint = mix(hue, vec3(1.0), 0.50);
    vec3 haloTint = hue * hue;
    vec3 rayTint = mix(hue, vec3(1.0), 0.24);
    vec3 streakTint = mix(hue, vec3(1.0), 0.10);
    float weight = core + halo + rays + streak;
    vec3 color = (core * coreTint + halo * haloTint + rays * rayTint +
                  streak * streakTint) / max(weight, 1e-5);
    return vec4(color * strength, vColor.a * min(weight, 1.0));
}

float confettiOutline(vec2 p, float style) {
    if (style < 3.5) {
        return max(-p.y, 0.8660254 * abs(p.x) + 0.5 * p.y) - 0.41;
    }
    if (style < 4.5) {
        vec2 diamond = abs(vec2(p.x + p.y, p.x - p.y)) * 0.70710678;
        return max(diamond.x, diamond.y) - 0.55;
    }
    return max(p.y, max(0.951057 * abs(p.x) + 0.309017 * p.y,
                       0.587785 * abs(p.x) - 0.809017 * p.y)) - 0.59;
}

vec4 foilSprite() {
    float outline = confettiOutline(vUV, vShade.x);
    float aa = clamp(fwidth(outline) * 0.65, 0.008, 0.22);
    float coverage = 1.0 - smoothstep(-aa, aa, outline);
    float bodyAlpha = vColor.a * coverage;

    // Keep the surface readable even at peak flash. The streak occupies
    // the uncovered part of the sprite and tapers away from the foil edge.
    vec3 foil = vColor.rgb * (0.86 + 0.14 * smoothstep(-0.55, 0.55, vUV.y));
    float edgeStreak = 0.0;
    if (vShade.z > 0.0) {
        vec2 axis = abs(vUV - vec2(0.0, 0.14));
        float width = max(0.085, min(0.26, fwidth(vUV.y)));
        float arms = (1.0 - smoothstep(0.24, vShade.w, axis.x)) *
                     (1.0 - smoothstep(0.0, width, axis.y));
        edgeStreak = 0.30 * vColor.a * vShade.z * arms * (1.0 - coverage);
    }
    float alpha = bodyAlpha + edgeStreak;
    vec3 streakTint = mix(vColor.rgb, vec3(1.0, 0.98, 0.94), 0.68);
    vec3 color = foil * bodyAlpha + streakTint * edgeStreak;
    return vec4(color / max(alpha, 1e-5), alpha);
}

void main() {
    if (vShade.y < 0.5 && vShade.x >= 3.0) {
        outColor = foilSprite();
        if (outColor.a <= 0.0) {
            discard;
        }
        return;
    }
    float d2 = dot(vUV, vUV);
    bool firework = vShade.y < 0.5 && vShade.x > 0.5;
    float streak = 0.0;
    if (firework && vShade.z > 0.0) {
        vec2 axis = abs(vUV);
        float reach = vShade.w;
        float lengthMask = 1.0 - smoothstep(0.20, reach, axis.x);
        float width = mix(0.14, 0.06, clamp(axis.x / reach, 0.0, 1.0));
        // Keep the thin streak visible when it approaches a pixel in height.
        float pixelWidth = min(0.38, 0.95 * fwidth(vUV.y));
        float crossMask = 1.0 - smoothstep(0.0, max(width, pixelWidth), axis.y);
        streak = 0.56 * vShade.z * lengthMask * crossMask;
    }
    if (d2 > 1.0 && streak <= 0.0) {
        discard;
    }
    if (firework) {
        outColor = fireworkSprite(d2, vShade.x > 1.5, streak);
        return;
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
