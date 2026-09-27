#version 450
layout(location = 0) in vec4 centerRadius;
layout(location = 0) noperspective out vec3 sphereNear;
layout(location = 1) noperspective out vec3 sphereFar;
layout(location = 2) flat out vec4 sphereCenterRadius;
layout(push_constant) uniform SolidParams {
    mat4 viewProj;
    mat4 view;
    int useMatcap;
    float overrideR, overrideG, overrideB;
    float fadeCenterX, fadeCenterY, fadeCenterZ;
    float fadeStart, fadeEnd;
    float overrideA;
} pc;

void main() {
    // Project an enclosing box, not a camera-facing diameter: the latter clips
    // silhouettes near the edges of perspective views.
    vec2 lower = vec2(1.0);
    vec2 upper = vec2(-1.0);
    bool crossesEye = false;
    bool anyInFront = false;
    for (int i = 0; i < 8; ++i) {
        vec3 corner = vec3((i & 1) != 0 ? 1.0 : -1.0,
                           (i & 2) != 0 ? 1.0 : -1.0,
                           (i & 4) != 0 ? 1.0 : -1.0);
        vec4 clip = pc.viewProj * vec4(centerRadius.xyz + corner * centerRadius.w, 1.0);
        crossesEye = crossesEye || clip.w <= 0.00001;
        anyInFront = anyInFront || clip.w > 0.00001;
        if (clip.w > 0.00001) {
            lower = min(lower, clip.xy / clip.w);
            upper = max(upper, clip.xy / clip.w);
        }
    }
    if (crossesEye) {
        lower = vec2(-1.0);
        upper = vec2(1.0);
    }
    lower = clamp(lower, vec2(-1.0), vec2(1.0));
    upper = clamp(upper, vec2(-1.0), vec2(1.0));
    const vec2 corners[6] = vec2[6](vec2(0, 0), vec2(1, 0), vec2(1, 1),
                                   vec2(0, 0), vec2(1, 1), vec2(0, 1));
    vec2 ndc = mix(lower, upper, corners[gl_VertexIndex]);
    mat4 inverseVP = inverse(pc.viewProj);
    vec4 nearPoint = inverseVP * vec4(ndc, 0.0, 1.0);
    vec4 farPoint = inverseVP * vec4(ndc, 0.9999, 1.0);
    sphereNear = nearPoint.xyz / nearPoint.w;
    sphereFar = farPoint.xyz / farPoint.w;
    sphereCenterRadius = centerRadius;
    gl_Position = anyInFront ? vec4(ndc, 0.0, 1.0) : vec4(2, 2, 0, 1);
}
