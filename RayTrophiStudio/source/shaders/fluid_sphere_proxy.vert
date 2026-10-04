#version 450

struct PackedPosition {
    float x;
    float y;
    float z;
};

layout(set = 1, binding = 0, std430) readonly buffer CarrierPositions {
    PackedPosition positions[];
};

layout(location = 0) noperspective out vec3 sphereNear;
layout(location = 1) noperspective out vec3 sphereFar;
layout(location = 2) flat out vec4 sphereCenterRadius;

layout(push_constant) uniform FluidSphereParams {
    mat4 viewProj;
    mat4 view;
    int useMatcap;
    float overrideR, overrideG, overrideB;
    float fadeCenterX, fadeCenterY, fadeCenterZ;
    float fadeStart, fadeEnd;
    float overrideA;
    float childRadius;
    float spreadRadius;
    uint parentCount;
    uint childrenPerParent;
    float sizeVariation;
} pc;

uint proxyHash(uint value) {
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    return value ^ (value >> 16u);
}

float proxyUnitFloat(uint value) {
    return float(proxyHash(value) & 0xffffu) / 65535.0;
}

vec3 proxyPairDirection(uint parent, uint pair) {
    const uint seed = proxyHash(parent ^ (pair * 0x85ebca6bu) ^ 0x9e3779b9u);
    vec3 direction = vec3(
        proxyUnitFloat(seed ^ 0x68bc21ebu) * 2.0 - 1.0,
        proxyUnitFloat(seed ^ 0x02e5be93u) * 2.0 - 1.0,
        proxyUnitFloat(seed ^ 0x967a889bu) * 2.0 - 1.0);
    const float lengthSquared = dot(direction, direction);
    return lengthSquared > 1e-6
        ? direction * inversesqrt(lengthSquared)
        : vec3(1.0, 0.0, 0.0);
}

vec3 childOffset(uint parent, uint child, uint count) {
    if (count <= 1u || pc.spreadRadius <= 0.0) return vec3(0.0);
    child %= count;
    const uint pair = child / 2u;
    const float signValue = (child & 1u) != 0u ? 1.0 : -1.0;
    vec3 direction = proxyPairDirection(parent, pair) * signValue;
    if ((count & 1u) != 0u && child == count - 1u) {
        direction = proxyPairDirection(parent, pair + 1u);
    }

    const vec3 parentPosition = vec3(
        positions[parent].x,
        positions[parent].y,
        positions[parent].z);
    const float supportRadius = pc.spreadRadius * 2.5;
    vec3 weightedDelta = vec3(0.0);
    float weightSum = 0.0;
    for (uint slot = 0u; slot < 5u; ++slot) {
        const uint step = 1u << slot;
        for (uint side = 0u; side < 2u; ++side) {
            if ((side == 0u && parent < step) ||
                (side == 1u && parent + step >= pc.parentCount)) {
                continue;
            }
            const uint candidate = side == 0u ? parent - step : parent + step;
            const vec3 candidatePosition = vec3(
                positions[candidate].x,
                positions[candidate].y,
                positions[candidate].z);
            const vec3 delta = candidatePosition - parentPosition;
            const float distanceSquared = dot(delta, delta);
            if (distanceSquared <= 1e-8 ||
                distanceSquared >= supportRadius * supportRadius) {
                continue;
            }
            const float distanceValue = sqrt(distanceSquared);
            const float alignment = max(
                0.0,
                dot(direction, delta / distanceValue));
            const float kernel = 1.0 - distanceValue / supportRadius;
            const float weight = kernel * kernel * (0.15 + 0.85 * alignment);
            weightedDelta += delta * weight;
            weightSum += weight;
        }
    }
    if (weightSum > 1e-6) {
        vec3 offset = weightedDelta * (0.52 / weightSum);
        const float offsetLength = length(offset);
        const float maxLength = pc.spreadRadius * 1.25;
        if (offsetLength > maxLength && maxLength > 0.0) {
            offset *= maxLength / offsetLength;
        }
        return offset;
    }

    const float radialScale = 0.72 + 0.28 * proxyUnitFloat(
        parent ^ (pair * 0xc2b2ae35u) ^ 0x27d4eb2fu);
    return direction * (pc.spreadRadius * radialScale);
}

float childVariationScale(uint parent, uint child, uint count) {
    const float variation = clamp(pc.sizeVariation, 0.0, 0.75);
    if (count <= 1u || variation <= 0.0) return 1.0;
    child %= count;
    if ((count & 1u) != 0u && child == count - 1u) return 1.0;
    const uint pair = child / 2u;
    const float signedRandom = proxyUnitFloat(
        parent ^ (pair * 0x165667b1u) ^ 0xd3a2646cu) * 2.0 - 1.0;
    const float delta = variation * signedRandom;
    const float smaller = 1.0 - delta;
    const float larger = 1.0 + delta;
    const float normalization = pow(
        2.0 / (smaller * smaller * smaller + larger * larger * larger),
        1.0 / 3.0);
    return normalization * ((child & 1u) != 0u ? larger : smaller);
}

void main() {
    const uint children = max(pc.childrenPerParent, 1u);
    const uint parent = uint(gl_InstanceIndex) / children;
    const uint child = uint(gl_InstanceIndex) - parent * children;
    if (parent >= pc.parentCount) {
        gl_Position = vec4(2.0, 2.0, 0.0, 1.0);
        return;
    }
    const PackedPosition packed = positions[parent];
    const vec3 center = vec3(packed.x, packed.y, packed.z) +
        childOffset(parent, child, children);
    const float childRadius = pc.childRadius *
        childVariationScale(parent, child, children);
    const vec4 centerRadius = vec4(center, childRadius);

    vec2 lower = vec2(1.0);
    vec2 upper = vec2(-1.0);
    bool crossesEye = false;
    bool anyInFront = false;
    for (int i = 0; i < 8; ++i) {
        vec3 corner = vec3((i & 1) != 0 ? 1.0 : -1.0,
                           (i & 2) != 0 ? 1.0 : -1.0,
                           (i & 4) != 0 ? 1.0 : -1.0);
        vec4 clip = pc.viewProj * vec4(center + corner * childRadius, 1.0);
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
