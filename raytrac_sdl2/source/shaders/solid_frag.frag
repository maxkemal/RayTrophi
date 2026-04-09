#version 450

layout(location = 0) in vec3 vNormal;
layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform sampler2D matcapSampler;

layout(push_constant) uniform SolidParams {
    mat4 viewProj;
    mat4 view;
    int useMatcap; // -1 = flat color, 0 = solid clay, 1 = texture, 2..9 = procedural
    float overrideR, overrideG, overrideB; // flat color (useMatcap == -1)
} solidParams;

// Shared matcap helper: hemisphere shading from colors + specular
vec3 matcapShade(vec2 uv, vec3 highlight, vec3 midtone, vec3 shadow, float specPow, float specStr, vec2 specPos) {
    // Diagonal gradient for warm-cool blend
    float t = uv.y * 0.6 + uv.x * 0.4;
    vec3 base = mix(shadow, midtone, smoothstep(0.0, 0.5, t));
    base = mix(base, highlight, smoothstep(0.5, 1.0, t));

    // Specular highlight
    float specDist = length(uv - specPos);
    float spec = pow(max(1.0 - specDist * 3.0, 0.0), specPow) * specStr;

    // Rim/edge falloff
    float rim = 1.0 - length(uv - vec2(0.5));
    rim = smoothstep(0.0, 0.5, rim);

    return clamp(base * rim + vec3(spec), 0.0, 1.0);
}

void main() {
    vec3 n = normalize(vNormal);
    vec2 uv = n.xy * 0.5 + 0.5;
    int preset = solidParams.useMatcap;

    // -1: Flat color override (grid axes, wireframe, etc.)
    if (preset == -1) {
        outColor = vec4(solidParams.overrideR, solidParams.overrideG, solidParams.overrideB, 1.0);
        return;
    }

    // 0: Solid clay (flat lambert, not matcap)
    if (preset == 0) {
        vec3 lightDir = normalize(vec3(0.45, 0.8, 0.35));
        float lambert = max(dot(n, lightDir), 0.0);
        float fill = 0.35 + lambert * 0.65;
        vec3 clay = vec3(0.74, 0.76, 0.80);
        outColor = vec4(clay * fill, 1.0);
        return;
    }

    // 1: User-loaded matcap texture
    if (preset == 1) {
        vec2 texUV = uv;
        texUV.y = 1.0 - texUV.y;
        outColor = vec4(texture(matcapSampler, texUV).rgb, 1.0);
        return;
    }

    // Procedural matcap presets (2..9)
    vec3 color;
    switch (preset) {
        case 2: // Default — warm-cool hemisphere
            color = matcapShade(uv,
                vec3(0.95, 0.90, 0.82),  // highlight: warm cream
                vec3(0.65, 0.67, 0.72),  // midtone: neutral gray
                vec3(0.25, 0.30, 0.45),  // shadow: cool blue
                6.0, 0.55, vec2(0.62, 0.72));
            break;

        case 3: // Clay — warm terracotta
            color = matcapShade(uv,
                vec3(0.92, 0.78, 0.65),  // highlight: warm sand
                vec3(0.72, 0.50, 0.38),  // midtone: terracotta
                vec3(0.35, 0.20, 0.15),  // shadow: dark brown
                4.0, 0.30, vec2(0.60, 0.70));
            break;

        case 4: // Silver — chrome metallic
            color = matcapShade(uv,
                vec3(0.95, 0.95, 0.97),  // highlight: bright white
                vec3(0.55, 0.58, 0.62),  // midtone: steel gray
                vec3(0.12, 0.13, 0.16),  // shadow: near black
                12.0, 0.85, vec2(0.58, 0.75));
            // Add secondary reflection
            float sec = pow(max(1.0 - length(uv - vec2(0.38, 0.35)) * 4.0, 0.0), 6.0) * 0.25;
            color += vec3(sec);
            break;

        case 5: // Pearl — soft white iridescent
            color = matcapShade(uv,
                vec3(0.98, 0.96, 0.94),  // highlight: off-white
                vec3(0.82, 0.80, 0.85),  // midtone: lavender gray
                vec3(0.55, 0.52, 0.58),  // shadow: muted purple
                5.0, 0.45, vec2(0.60, 0.72));
            // Subtle iridescence
            float irid = sin(uv.x * 12.0 + uv.y * 8.0) * 0.04;
            color += vec3(irid, -irid * 0.5, irid * 0.8);
            break;

        case 6: // Jade — green stone
            color = matcapShade(uv,
                vec3(0.72, 0.88, 0.70),  // highlight: light green
                vec3(0.28, 0.52, 0.32),  // midtone: jade green
                vec3(0.08, 0.18, 0.10),  // shadow: dark green
                8.0, 0.50, vec2(0.60, 0.73));
            break;

        case 7: // Copper — warm metal
            color = matcapShade(uv,
                vec3(0.95, 0.82, 0.65),  // highlight: bright copper
                vec3(0.72, 0.45, 0.22),  // midtone: copper
                vec3(0.25, 0.12, 0.06),  // shadow: dark bronze
                10.0, 0.70, vec2(0.60, 0.74));
            break;

        case 8: // Obsidian — dark glossy
            color = matcapShade(uv,
                vec3(0.50, 0.50, 0.55),  // highlight: dim gray
                vec3(0.15, 0.15, 0.18),  // midtone: very dark
                vec3(0.03, 0.03, 0.05),  // shadow: near black
                16.0, 0.90, vec2(0.58, 0.76));
            break;

        case 9: // Skin — warm subsurface
            color = matcapShade(uv,
                vec3(0.95, 0.85, 0.78),  // highlight: bright skin
                vec3(0.82, 0.62, 0.52),  // midtone: warm skin
                vec3(0.45, 0.25, 0.20),  // shadow: deep warm
                4.0, 0.25, vec2(0.62, 0.70));
            // Subtle subsurface warmth at edges
            float edge = 1.0 - smoothstep(0.3, 0.5, length(uv - vec2(0.5)));
            color = mix(color, color * vec3(1.1, 0.85, 0.80), edge * 0.4);
            break;

        default:
            color = matcapShade(uv,
                vec3(0.95, 0.90, 0.82),
                vec3(0.65, 0.67, 0.72),
                vec3(0.25, 0.30, 0.45),
                6.0, 0.55, vec2(0.62, 0.72));
            break;
    }

    outColor = vec4(clamp(color, 0.0, 1.0), 1.0);
}
