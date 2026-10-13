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
// Parent-dependent intervals and arc-length travel are built by the CPU.
// Hold, then extend the tip quickly: a stepped leader rather than uniform wiping.
bool cloudBoltSegment(int i, int k, CloudLightningG event, out vec3 a, out vec3 b,
                      out float weight) {
    CloudBoltSegG s = cloudParams.bolt[i * CLOUD_BOLT_SEGMENTS + k];
    float phase = clamp((event.stage.x - s.b.w) / max(s.growth.x - s.b.w, 1e-6), 0.0, 1.0);
    float grown = smoothstep(0.35, 0.90, phase);
    a = s.a.xyz;
    b = mix(s.a.xyz, s.b.xyz, grown);
    weight = mix(s.a.w, s.growth.w, grown);
    return grown > 0.0;
}

// Brightness of a segment. After a downward ground strike connects (return
// stroke) the main channel dominates while its unconnected branches still
// illuminate at reduced strength. An in-cloud flash keeps its network,
// and so does upward lightning (stage.z 2): its continuous current feeds
// every branch that reached the cloud.
float cloudBoltSegmentBrightness(CloudLightningG event, float weight) {
    bool returnStroke = event.stage.z < 0.5 && event.stage.x >= 1.0;
    if (weight >= 0.99) return 1.0;
    return returnStroke ? weight * 0.75 : weight;
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

// ── Fine channel structure (view only) ───────────────────────────────────────
// The CPU tree has 32 main pieces over a channel several kilometres long, so
// a piece is a 100-300 m straight run: from up close the bolt read as a few
// ruler strokes (2026-10-13). A piece the view sees large is split here into
// 2^L sub-pieces by midpoint displacement, continuing the CPU tree's rule one
// scale down: level l puts a random offset on its new knots, linear between,
// with a constant knot angle (amplitude halves as spacing halves) -- real
// channels are equally kinked at every scale. Deterministic per segment, so
// the shape never changes between frames; a growing segment reveals the same
// fine path. Only pixels near the coarse piece pay for it.
const int   CLOUD_BOLT_FINE_MAX_LEVELS = 5;      // up to 32 sub-pieces
const float CLOUD_BOLT_FINE_AMPLITUDE = 0.15;    // level-1 offset / piece length
const float CLOUD_BOLT_FINE_PIECE_ANGLE = 0.006; // target sub-piece angular size (rad)
const float CLOUD_BOLT_FINE_MIN_PIECE_M = 4.0;

uint cloudBoltHash(uint v) {
    v ^= v >> 16;
    v *= 0x7feb352du;
    v ^= v >> 15;
    v *= 0x846ca68bu;
    return v ^ (v >> 16);
}

// Offset of knot j (of n = 2^levels) along a piece, in the piece's frame
// (s1, s2 perpendicular to it): the sum over levels of each level's hat
// function. Odd knots of a level carry its random offset, even ones 0.
vec2 cloudBoltFineOffset(int j, int levels, uint seed) {
    vec2 o = vec2(0.0);
    float amp = CLOUD_BOLT_FINE_AMPLITUDE;
    int n = 1 << levels;
    for (int l = 1; l <= levels; ++l) {
        int spacing = n >> l;          // finest-grid knots per level-l step
        int k0 = j / spacing;
        float f = float(j - k0 * spacing) / float(spacing);
        vec2 v0 = vec2(0.0), v1 = vec2(0.0);
        if ((k0 & 1) == 1) {
            uint h = cloudBoltHash(seed ^ uint(l * 7919 + k0 * 104729));
            v0 = vec2(float(h & 0xffffu), float(h >> 16)) / 32767.5 - 1.0;
        }
        if (((k0 + 1) & 1) == 1 && f > 0.0) {
            uint h = cloudBoltHash(seed ^ uint(l * 7919 + (k0 + 1) * 104729));
            v1 = vec2(float(h & 0xffffu), float(h >> 16)) / 32767.5 - 1.0;
        }
        o += mix(v0, v1, f) * amp;
        amp *= 0.5;
    }
    return o;
}

// Ray vs one channel piece: distance along the ray and closest separation.
float cloudBoltRaySeparation(vec3 origin, vec3 direction, vec3 a, vec3 b, out float t) {
    vec3 v = b - a;
    vec3 w = origin - a;
    float e = max(dot(v, v), 1e-6);
    float dv = dot(direction, v);
    float dw = dot(direction, w);
    float vw = dot(v, w);
    float denominator = e - dv * dv;
    float tr = denominator > 1e-6 ? (dv * vw - dw * e) / denominator : -dw;
    float u = clamp((vw + dv * max(tr, 0.0)) / e, 0.0, 1.0);
    vec3 closest = a + v * u;
    t = dot(closest - origin, direction);
    return length(origin + direction * t - closest);
}

// Core + tight sheath of segment k of event i along the view ray, on the fine
// path when the piece is seen large; the wide faint glow stays on the coarse
// piece (it is blurrier than the displacement).
float cloudBoltRayFine(vec3 origin, vec3 direction, float limit, int i, int k,
                       CloudLightningG event, float radius, out float hitDistance) {
    CloudBoltSegG s = cloudParams.bolt[i * CLOUD_BOLT_SEGMENTS + k];
    float phase = clamp((event.stage.x - s.b.w) / max(s.growth.x - s.b.w, 1e-6), 0.0, 1.0);
    float grown = smoothstep(0.35, 0.90, phase);
    hitDistance = limit;
    if (grown <= 0.0) return 0.0;
    vec3 a = s.a.xyz;
    vec3 full = s.b.xyz;
    vec3 b = mix(a, full, grown);
    float len = length(full - a);

    float t;
    float separation = cloudBoltRaySeparation(origin, direction, a, b, t);
    if (t <= 0.0 || t >= limit) return 0.0;
    float width = max(radius, t * 0.00012);
    float glow = exp(-separation / max(width * 60.0, 0.01)) * 0.008;
    // The fine path stays within the level sum (2 x the level-1 amplitude)
    // on each of the two perpendicular axes: sqrt(2) x 2 x amplitude < 3 x.
    float envelope = 3.0 * CLOUD_BOLT_FINE_AMPLITUDE * len;
    int levels = 0;
    float angular = len / max(t, 1.0);
    while (levels < CLOUD_BOLT_FINE_MAX_LEVELS &&
           angular / float(1 << levels) > CLOUD_BOLT_FINE_PIECE_ANGLE &&
           len / float(2 << levels) >= CLOUD_BOLT_FINE_MIN_PIECE_M) ++levels;
    float near = 0.0;
    if (levels == 0 || separation > envelope + width * 30.0) {
        near = 1.0 - smoothstep(width * 0.35, width * 1.5, separation);
        near += exp(-separation / max(width * 6.0, 0.001)) * 0.05;
        hitDistance = t;
        return near + glow;
    }
    vec3 axis = (full - a) / max(len, 1e-6);
    vec3 ref = abs(axis.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 s1 = normalize(cross(axis, ref));
    vec3 s2 = cross(axis, s1);
    // Static per segment for the life of the event (its birth/completion
    // progress come from the geometry seed), not tied to the moving position.
    uint seed = cloudBoltHash(floatBitsToUint(s.b.w) ^ cloudBoltHash(floatBitsToUint(s.growth.x)) ^
                              uint(k) * 2654435761u);
    int n = 1 << levels;
    float best = 0.0;
    vec3 prev = a;
    for (int j = 1; j <= n; ++j) {
        float u0 = float(j - 1) / float(n);
        if (u0 >= grown) break;
        float u1 = min(float(j) / float(n), grown);
        vec2 o = (j < n) ? cloudBoltFineOffset(j, levels, seed) * len : vec2(0.0);
        vec3 knot = mix(a, full, float(j) / float(n)) + s1 * o.x + s2 * o.y;
        // A partially grown last piece reveals its fine path, not a stretch.
        vec3 next = mix(prev, knot, (u1 - u0) * float(n));
        float tj;
        float sj = cloudBoltRaySeparation(origin, direction, prev, next, tj);
        if (tj > 0.0 && tj < limit) {
            float wj = max(radius, tj * 0.00012);
            float v = 1.0 - smoothstep(wj * 0.35, wj * 1.5, sj);
            v += exp(-sj / max(wj * 6.0, 0.001)) * 0.05;
            if (v > best) {
                best = v;
                hitDistance = tj;
            }
        }
        prev = knot;
    }
    if (best <= 0.0) hitDistance = t;
    return best + glow;
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
            float value = cloudBoltRayFine(origin, direction, limit, i, k, event,
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
