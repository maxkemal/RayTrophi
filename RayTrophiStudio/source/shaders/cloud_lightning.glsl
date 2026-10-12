#ifndef CLOUD_LIGHTNING_GLSL
#define CLOUD_LIGHTNING_GLSL

float cloudLightningActivity(CloudLightningG event) {
    if (event.start.w <= 0.0 || !cloudRtEnabled()) return 0.0;
    // Reject flashes in clear portions of the authored storm footprint.
    float activity = 0.0;
    for (int i = 0; i < 3; ++i) {
        CloudLayerG layer = cloudParams.layers[i];
        // Storm cells raise the genus locally: a flash in a cell needs no
        // convective layer-wide genus.
        bool cellHost = cloudParams.storms.w > 0.5;
        if (layer.b.w < 0.5 || layer.b.x <= 0.0 || (layer.a.w < 0.6 && !cellHost) ||
            layer.a.z <= 0.0) continue;
        vec4 weather = cloudWeatherAt(layer, event.start.xyz);
        float covered = clamp(cloudRemap(weather.r, 1.0 - layer.a.z, 1.0, 0.0, 1.0),
                              0.0, 1.0);
        covered = max(covered, cloudStormCellAt(event.start.xyz).r);
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

// Diffusion attenuation of the flash through cloud (CPU: kCloudFlashDiffusion).
const float CLOUD_FLASH_DIFFUSION = 0.11;

float cloudFlashMeanExtinction() {
    float e = 0.0;
    for (int i = 0; i < 3; ++i)
        if (cloudParams.layers[i].b.w >= 0.5) e = max(e, cloudParams.layers[i].b.x);
    return e;
}

// ── Channel network (CPU: buildCloudBolt) ───────────────────────────────────
// The visible part of segment k of event i at the current leader progress:
// it starts growing at its `appear` progress and takes ~1/16 of the leader
// to reach full length. Returns false while it has not started.
bool cloudBoltSegment(int i, int k, CloudLightningG event, out vec3 a, out vec3 b,
                      out float weight) {
    CloudBoltSegG s = cloudParams.bolt[i * CLOUD_BOLT_SEGMENTS + k];
    float grown = clamp((event.stage.x - s.b.w) * 16.0, 0.0, 1.0);
    a = s.a.xyz;
    b = mix(s.a.xyz, s.b.xyz, grown);
    weight = s.a.w;
    return grown > 0.0;
}

// Brightness of a segment. After a downward ground strike connects (return
// stroke) the current runs down the main channel only: it dominates and the
// branches that did not connect fade. An in-cloud flash keeps its network,
// and so does upward lightning (stage.z 2): its continuous current feeds
// every branch that reached the cloud.
float cloudBoltSegmentBrightness(CloudLightningG event, float weight) {
    bool returnStroke = event.stage.z < 0.5 && event.stage.x >= 1.0;
    if (weight >= 0.99) return 1.0;
    return returnStroke ? weight * 0.35 : weight;
}

vec3 cloudClosestOnSegment(vec3 p, vec3 a, vec3 b) {
    vec3 ab = b - a;
    float u = clamp(dot(p - a, ab) / max(dot(ab, ab), 1e-6), 0.0, 1.0);
    return a + ab * u;
}

// In-cloud flash source at p. The light comes out of the lit channel network
// and diffuses THROUGH the cloud, so (1) it follows the network: the nearest
// grown segment, weighted by its brightness, sets the source -- the cell
// lights in patches along the branches as the leader crawls; (2) it is
// attenuated by the optical depth between that segment and p (two cheap
// base-shape taps, ~1/(1 + k tau)): dense lobes shade the rest, which is the
// cloud's inner structure. Segment distances are ALU only; the two density
// taps are paid once per event, not per segment.
vec3 cloudLightningSource(vec3 p, vec4 activity, bool occlude) {
    vec3 source = vec3(0.0);
    for (int i = 0; i < 4; ++i) {
        CloudLightningG event = cloudParams.lightning[i];
        if (event.start.w <= 0.0 || activity[i] <= 0.0) continue;
        float radius = event.stage.w;
        // Cheap reject: the whole network lies within ~0.6 span of the main
        // start->end line (branches and spiders included).
        vec3 axisPoint = cloudClosestOnSegment(p, event.start.xyz, event.end.xyz);
        float span = length(event.end.xyz - event.start.xyz);
        if (length(p - axisPoint) > 4.0 * radius + 0.6 * max(span, radius)) continue;
        int count = min(int(event.color.w + 0.5), CLOUD_BOLT_SEGMENTS);
        float best = 0.0;
        vec3 bestPoint = axisPoint;
        float bestDistance = 0.0;
        for (int k = 0; k < count; ++k) {
            vec3 a, b;
            float weight;
            if (!cloudBoltSegment(i, k, event, a, b, weight)) continue;
            vec3 c = cloudClosestOnSegment(p, a, b);
            float d = length(p - c);
            float value = cloudFlashProfile(d, radius * mix(0.5, 1.0, weight)) *
                          cloudBoltSegmentBrightness(event, weight);
            if (value > best) {
                best = value;
                bestPoint = c;
                bestDistance = d;
            }
        }
        if (best < 0.002) continue;
        // Secondary rays skip the taps (the flash reaches what they light
        // through the surface NEE; their view of the lit cloud is averaged)
        // and use the mean: half the densest layer's extinction.
        vec3 offset = p - bestPoint;
        float tau = occlude
            ? 0.5 * (cloudDensity(bestPoint + offset * 0.3, false) +
                     cloudDensity(bestPoint + offset * 0.7, false)) * bestDistance
            : 0.5 * cloudFlashMeanExtinction() * bestDistance;
        best /= 1.0 + CLOUD_FLASH_DIFFUSION * tau;
        source += event.color.rgb * event.start.w * best * activity[i];
    }
    return source;
}

// Fluence-like light of the lit cloud bases at a point of the rain curtain
// (no surface normal). Same CPU-built emitters as the surface NEE; the
// activity gate is the one the in-cloud flash uses.
vec3 cloudFlashMediumLight(vec3 p, vec4 activity) {
    vec3 light = vec3(0.0);
    for (int i = 0; i < 4; ++i) {
        CloudFlashLightG f = cloudParams.flash[i];
        if (f.position.w <= 0.0) continue;
        float a = activity[int(f.radiance.w + 0.5) & 3];
        if (a <= 0.0) continue;
        vec3 wi;
        light += cloudFlashIrradiance(f, p, vec3(0.0), wi) * a;
    }
    return light;
}

bool cloudFlashAny() {
    for (int i = 0; i < 4; ++i)
        if (cloudParams.flash[i].position.w > 0.0) return true;
    return false;
}

// Ray vs one channel segment: core + halo, as seen along the view ray.
float cloudBoltRaySegment(vec3 origin, vec3 direction, float limit, vec3 a, vec3 b,
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
    // Halo: light scattered by the air/rain around the channel. Two scales,
    // a tight bright sheath and a wide faint glow, so a near bolt reads as
    // glowing, not as a drawn line.
    float halo = exp(-separation / max(width * 6.0, 0.001)) * 0.05 +
                 exp(-separation / max(width * 60.0, 0.01)) * 0.008;
    return core + halo;
}

vec3 cloudApplyLightning(vec3 color, vec3 origin, vec3 direction, float limit) {
    if (!cloudRtEnabled()) return color;
    for (int i = 0; i < 4; ++i) {
        CloudLightningG event = cloudParams.lightning[i];
        if (event.start.w <= 0.0) continue;
        vec3 center = (event.start.xyz + event.end.xyz) * 0.5;
        float span = length(event.end.xyz - event.start.xyz);
        float bound = span * 1.1 + event.end.w * 120.0;
        vec3 offset = center - origin;
        float projected = dot(offset, direction);
        if (projected + bound <= 0.0 || projected - bound >= limit ||
            dot(offset, offset) - projected * projected > bound * bound) continue;
        float activity = cloudLightningActivity(event);
        if (activity <= 0.0) continue;
        float brightness = 0.0;
        float nearest = limit;
        int count = min(int(event.color.w + 0.5), CLOUD_BOLT_SEGMENTS);
        for (int k = 0; k < count; ++k) {
            vec3 a, b;
            float weight;
            if (!cloudBoltSegment(i, k, event, a, b, weight)) continue;
            float distance;
            // Branches are thinner and taper toward their twigs.
            float value = cloudBoltRaySegment(origin, direction, limit, a, b,
                                              event.end.w * mix(0.35, 1.0, weight), distance) *
                          cloudBoltSegmentBrightness(event, weight);
            if (value > brightness) {
                brightness = value;
                nearest = distance;
            }
        }
        if (brightness < 0.0001) continue;
        // Only live visible bolts pay up to four cheap samples per crossed
        // cloud-shell interval; no random walk, scene lights or dynamic
        // mesh/TLAS updates are introduced.
        float visibility = exp(-cloudOpticalDepth(origin, direction, nearest, 4, 0.5, false));
        // The channel core is far brighter than the cloud it lights
        // (lightning.channel_brightness, stage.y).
        color += event.color.rgb * event.start.w * max(event.stage.y, 0.0) * brightness *
                 activity * visibility;
    }
    return color;
}

#endif
