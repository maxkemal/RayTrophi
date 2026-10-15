#ifndef LENS_WATER_COMPOSITE_GLSL
#define LENS_WATER_COMPOSITE_GLSL
layout(std430, set = 1, binding = 0) readonly buffer LensDisplayState {
    vec4 lensNearState[8];
    uvec4 lensMeta;
};
layout(set = 1, binding = 1, rgba16f) uniform readonly image2D lensWeatherField;
layout(set = 1, binding = 2, rgba16f) uniform readonly image2D lensWeatherSolidField;
vec2 lensSolidAt(vec2 uv) {
    ivec2 size = imageSize(lensWeatherSolidField);
    vec2 p = clamp(uv * vec2(size) - 0.5, vec2(0.0), vec2(size - 1));
    ivec2 a = ivec2(floor(p));
    ivec2 b = min(a + 1, size - 1);
    vec2 blend = fract(p);
    return mix(mix(imageLoad(lensWeatherSolidField, a).xy,
                   imageLoad(lensWeatherSolidField, ivec2(b.x, a.y)).xy, blend.x),
               mix(imageLoad(lensWeatherSolidField, ivec2(a.x, b.y)).xy,
                   imageLoad(lensWeatherSolidField, b).xy, blend.x), blend.y);
}
vec4 lensWeatherAt(vec2 uv) {
    if (lensMeta.y == 0u) {
        return vec4(0.0);
    }
    ivec2 size = imageSize(lensWeatherField);
    if (size.x <= 1) {
        return vec4(0.0);
    }
    vec2 p = clamp(uv * vec2(size) - 0.5, vec2(0.0), vec2(size - 1));
    ivec2 a = ivec2(floor(p));
    ivec2 b = min(a + 1, size - 1);
    vec2 blend = fract(p);
    return mix(mix(imageLoad(lensWeatherField, a),
                   imageLoad(lensWeatherField, ivec2(b.x, a.y)), blend.x),
               mix(imageLoad(lensWeatherField, ivec2(a.x, b.y)),
                   imageLoad(lensWeatherField, b), blend.x), blend.y);
}
// Each backend supplies only its scene-linear sampling adapter. All glass
// refraction, blur and transmission arithmetic is shared here.
vec3 lensWeatherScene(vec2 uv);
vec3 lensWeatherComposite(vec3 center, vec2 uv, bool enabled, vec2 outputSize) {
    if (!enabled) {
        return center;
    }
    vec4 lens = lensWeatherAt(uv);
    vec2 solid = lensMeta.y > 0u ? lensSolidAt(uv) : vec2(0.0);
    if (max(lens.w, max(solid.x, solid.y)) <= 0.0001) {
        return center;
    }
    vec3 refracted = vec3(0.0);
    float rootAspect = sqrt(max(outputSize.x, 1.0) / max(outputSize.y, 1.0));
    vec2 blurScale = vec2(1.0 / rootAspect, rootAspect);
    for (int tap = 0; tap < 5; ++tap) {
        float angle = float(tap) * 2.39996323;
        vec2 source = clamp(uv + lens.xy + vec2(cos(angle), sin(angle)) * blurScale * lens.z *
                            sqrt(float(tap) / 4.0), vec2(0.0), vec2(1.0));
        refracted += lensWeatherScene(source);
    }
    vec3 result = mix(center, refracted * 0.188, lens.w);
    if (max(solid.x, solid.y) > 0.0001) {
        vec3 scattered = vec3(0.0);
        for (int tap = 0; tap < 5; ++tap) {
            float angle = float(tap) * 2.39996323;
            vec2 source = clamp(uv + vec2(cos(angle), sin(angle)) * blurScale *
                (0.015 + lens.z) * sqrt(float(tap) / 4.0), vec2(0.0), vec2(1.0));
            scattered += lensWeatherScene(source) * 0.2;
        }
        float luminance = dot(scattered, vec3(0.2126, 0.7152, 0.0722));
        // Broad, mostly achromatic scattering response for compacted snow;
        // intensity follows scene radiance rather than a fixed white overlay.
        vec3 snow = mix(scattered, vec3(luminance), 0.85) * 1.35;
        result = mix(result, mix(scattered, vec3(luminance), 0.4) * 0.92, solid.y);
        result = mix(result, snow, solid.x);
    }
    return result;
}
#endif
