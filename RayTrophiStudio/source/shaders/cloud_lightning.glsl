#ifndef CLOUD_LIGHTNING_GLSL
#define CLOUD_LIGHTNING_GLSL

float cloudLightningHash(float value) {
    uint bits = uint(value * 4096.0 + 11.0);
    bits ^= bits >> 16;
    bits *= 0x7feb352du;
    bits ^= bits >> 15;
    bits *= 0x846ca68bu;
    bits ^= bits >> 16;
    return float(bits & 0x00ffffffu) / 16777216.0;
}

float cloudLightningActivity(CloudLightningG event) {
    if (event.start.w <= 0.0 || !cloudRtEnabled()) return 0.0;
    // Reject flashes in clear portions of the authored storm footprint.
    float activity = 0.0;
    for (int i = 0; i < 3; ++i) {
        CloudLayerG layer = cloudParams.layers[i];
        if (layer.b.w < 0.5 || layer.b.x <= 0.0 || layer.a.w < 0.6 || layer.a.z <= 0.0) continue;
        vec4 weather = cloudWeatherAt(layer, event.start.xyz);
        float covered = clamp(cloudRemap(weather.r, 1.0 - layer.a.z, 1.0, 0.0, 1.0),
                              0.0, 1.0);
        activity = max(activity, smoothstep(0.02, 0.25, covered));
    }
    return activity;
}

vec4 cloudLightningActivities() {
    return vec4(cloudLightningActivity(cloudParams.lightning[0]),
                cloudLightningActivity(cloudParams.lightning[1]),
                cloudLightningActivity(cloudParams.lightning[2]),
                cloudLightningActivity(cloudParams.lightning[3]));
}

vec3 cloudLightningSource(vec3 p, vec4 activity) {
    vec3 source = vec3(0.0);
    for (int i = 0; i < 4; ++i) {
        CloudLightningG event = cloudParams.lightning[i];
        if (event.start.w <= 0.0 || activity[i] <= 0.0) continue;
        vec3 span = event.end.xyz - event.start.xyz;
        float u = clamp(dot(p - event.start.xyz, span) / max(dot(span, span), 1e-6),
                        0.0, event.stage.x);
        vec3 offset = p - (event.start.xyz + span * u);
        float radius = event.stage.w;
        float spread = exp(-dot(offset, offset) / (radius * radius));
        if (spread < 0.001) continue;
        source += event.color.rgb * event.start.w * spread * activity[i];
    }
    return source;
}

vec3 cloudBoltPoint(CloudLightningG event, int point) {
    float u = float(point) / 24.0;
    vec3 p = mix(event.start.xyz, event.end.xyz, u);
    float seed = event.color.w + float(point) * 17.0;
    float spread = length(event.end.xyz - event.start.xyz) * 0.075 * (4.0 * u * (1.0 - u));
    p += vec3(cloudLightningHash(seed) - 0.5,
              (cloudLightningHash(seed + 1.0) - 0.5) * 0.3,
              cloudLightningHash(seed + 2.0) - 0.5) * spread;
    return p;
}

float cloudBoltSegment(vec3 origin, vec3 direction, float limit, vec3 a, vec3 b,
                      float radius, out float hitDistance) {
    vec3 v = b - a;
    vec3 w = origin - a;
    float e = max(dot(v, v), 1e-6);
    float dv = dot(direction, v);
    float dw = dot(direction, w);
    float vw = dot(v, w);
    float denominator = e - dv * dv;
    float t = denominator > 1e-6 ? (dv * vw - dw * e) / denominator : -dw;
    float u = clamp((vw + dv * max(t, 0.0)) / e, 0.0, 1.0);
    vec3 closest = a + v * u;
    t = dot(closest - origin, direction);
    hitDistance = t;
    if (t <= 0.0 || t >= limit) return 0.0;
    float separation = length(origin + direction * t - closest);
    // A minimum angular footprint keeps the bright core visible at distance.
    float width = max(radius, t * 0.00012);
    float core = 1.0 - smoothstep(width * 0.35, width * 1.5, separation);
    float glow = exp(-separation / max(width * 6.0, 0.001)) * 0.04;
    return core + glow;
}

vec3 cloudApplyLightning(vec3 color, vec3 origin, vec3 direction, float limit) {
    if (!cloudRtEnabled()) return color;
    for (int i = 0; i < 4; ++i) {
        CloudLightningG event = cloudParams.lightning[i];
        if (event.start.w <= 0.0) continue;
        vec3 center = (event.start.xyz + event.end.xyz) * 0.5;
        float bound = length(event.end.xyz - event.start.xyz) * 0.75 + event.end.w * 12.0;
        vec3 offset = center - origin;
        float projected = dot(offset, direction);
        if (projected + bound <= 0.0 || projected - bound >= limit ||
            dot(offset, offset) - projected * projected > bound * bound) continue;
        float activity = cloudLightningActivity(event);
        if (activity <= 0.0) continue;
        float brightness = 0.0;
        float nearest = limit;
        vec3 a = cloudBoltPoint(event, 0);
        vec3 axis = normalize(event.end.xyz - event.start.xyz);
        vec3 side = cross(axis, vec3(0.0, 1.0, 0.0));
        side = dot(side, side) > 1e-6 ? normalize(side) : vec3(1.0, 0.0, 0.0);
        float spanLength = length(event.end.xyz - event.start.xyz);
        for (int segment = 0; segment < 24; ++segment) {
            float growth = clamp(event.stage.x * 24.0 - float(segment), 0.0, 1.0);
            if (growth <= 0.0) break;
            vec3 fullEnd = cloudBoltPoint(event, segment + 1);
            vec3 b = mix(a, fullEnd, growth);
            float distance;
            float value = cloudBoltSegment(origin, direction, limit, a, b, event.end.w, distance);
            if (value > brightness) {
                brightness = value;
                nearest = distance;
            }
            if (segment == 6 || segment == 10 || segment == 14 || segment == 18) {
                float seed = event.color.w + float(segment) * 23.0;
                float branchGrowth = clamp((event.stage.x * 24.0 - float(segment) - 1.0) *
                                           0.25, 0.0, 1.0);
                if (cloudLightningHash(seed + 3.0) < event.stage.y && branchGrowth > 0.0) {
                    float directionSign = cloudLightningHash(seed) < 0.5 ? -1.0 : 1.0;
                    vec3 reach = (axis * 0.14 + side * directionSign *
                                  (0.10 + cloudLightningHash(seed + 2.0) * 0.12) +
                                  vec3(0.0, event.stage.z > 0.5 ? 0.025 : -0.08, 0.0)) *
                                 spanLength;
                    vec3 elbow = b + reach * 0.45 + side * directionSign * spanLength * 0.03;
                    vec3 tip = b + reach;
                    vec3 branchStart = b;
                    for (int twig = 0; twig < 2; ++twig) {
                        float extension = clamp(branchGrowth * 2.0 - float(twig), 0.0, 1.0);
                        if (extension <= 0.0) break;
                        vec3 branchEnd = mix(branchStart, twig == 0 ? elbow : tip, extension);
                        value = cloudBoltSegment(origin, direction, limit, branchStart, branchEnd,
                                                event.end.w * 0.5, distance) * 0.45;
                        if (value > brightness) {
                            brightness = value;
                            nearest = distance;
                        }
                        branchStart = branchEnd;
                    }
                }
            }
            a = fullEnd;
        }
        if (brightness < 0.0001) continue;
        // Only live visible bolts pay up to four cheap samples per crossed
        // cloud-shell interval; no random
        // walk, scene lights or dynamic mesh/TLAS updates are introduced.
        float visibility = exp(-cloudOpticalDepth(origin, direction, nearest, 4, 0.5, false));
        color += event.color.rgb * event.start.w * brightness * activity * visibility;
    }
    return color;
}

#endif
