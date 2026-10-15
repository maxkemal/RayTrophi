// Near-camera precipitation (docs/dev/ATMOSPHERE_NEAR_PRECIP.md).
//
// The atmosphere's precipitation AT THE CAMERA, drawn as individual drops and
// flakes out to cloudNearPrecipRange(). Nothing here is authored: the rate and
// the form mix are cloudPrecipSampleForm() at the camera (the function the
// curtain uses), the light is the curtain's sample light (cloudPrecipMarch),
// the motion is fall speed + climate surface wind over the frame exposure.
//
// Two halves:
//   state  -- needs cloud_common.glsl only (cloud_sample.comp measures it);
//   draw   -- define NEAR_PRECIP_DRAW, include after cloud_rt.glsl with
//             CLOUD_RT_LIGHTING (cloud_callable.rcall, cloud_scene.comp).
//
// Mean preserved: a band's drawn streaks plus its veil cover, on average,
// N * pi r^2 * depth -- the curtain's optical depth over the same metres
// (sigma = 2 pi r^2 N, Q_ext = 2, half of it diffraction kept in the beam).
// The curtain march of camera rays starts at the range (cloud_rt.glsl).
#ifndef NEAR_PRECIP_GLSL
#define NEAR_PRECIP_GLSL

// Per form, in CloudPrecipForm order (a = rain, drizzle, snow, sleet;
// b = graupel, hail, freezing rain, ice pellets):
//   radius of the drawn drop / flake, m (equivalent sphere: a snowflake is an
//   aggregate of mm-size crystals);
//   flutter amplitude, share of the cell margin (snow sways, hail does not).
const vec4 kNearRadiusA = vec4(1.0e-3, 0.25e-3, 2.5e-3, 1.2e-3);
const vec4 kNearRadiusB = vec4(1.5e-3, 6.0e-3, 1.0e-3, 1.2e-3);
const vec4 kNearSwayA   = vec4(0.0, 0.25, 1.0, 0.3);
const vec4 kNearSwayB   = vec4(0.15, 0.0, 0.0, 0.1);
// Liquid share of the particle surface: a liquid drop is a lens, an ice
// particle a diffuse white scatterer (sleet: half melted, graupel: rimed).
const vec4 kNearLiquidA = vec4(1.0, 1.0, 0.0, 0.5);
const vec4 kNearLiquidB = vec4(0.0, 0.0, 1.0, 0.0);

// A liquid drop is a lens with a ~165 deg view (Garg & Nayar 2004): it shows
// the scene around the ray, inverted, at ~0.94 of its radiance (Fresnel
// loss). Roughly half of that view is the scene behind the drop -- the pixel
// itself -- the other half the surroundings (sky dome, ground), here the
// curtain's ambient. Written as result = acc + T * behind, a drop of
// coverage alpha then keeps kNearLensKeep of the background instead of
// replacing it: rain brightens dark backgrounds and greys bright skies, it
// never paints a dark curtain over the image.
const float kNearLensTau  = 0.94;
const float kNearLensKeep = 0.47;   // tau x share of the view behind the drop

float nearFormPick(vec4 a, vec4 b, int i) { return i < 4 ? a[i] : b[i - 4]; }

struct NearPrecipForm {
    int   id;        // PRECIP_* (1..8), 0 = none
    float share;     // mass share in the mix
    float radius;    // m
    float fall;      // terminal speed m/s
    float density;   // drops per m^3 at the camera's rate
    float forward;   // forward-lobe share (phase)
    float sway;      // flutter, 0..1
};

NearPrecipForm nearFormNone() {
    NearPrecipForm f;
    f.id = 0; f.share = 0.0; f.radius = 1e-3; f.fall = 1.0;
    f.density = 0.0; f.forward = 0.0; f.sway = 0.0;
    return f;
}

void nearFormFill(inout NearPrecipForm f, float rate) {
    if (f.id <= 0) return;
    int i = f.id - 1;
    f.radius = nearFormPick(kNearRadiusA, kNearRadiusB, i);
    f.fall = max(nearFormPick(kPrecipFallA, kPrecipFallB, i), 0.3);
    float sigma = nearFormPick(kPrecipSigmaA, kPrecipSigmaB, i) *
                  pow(rate, nearFormPick(kPrecipExpA, kPrecipExpB, i));
    // sigma = Q_ext pi r^2 N with Q_ext = 2 (large particles).
    f.density = f.share * sigma / (2.0 * CLOUD_PI * f.radius * f.radius);
    f.forward = nearFormPick(kPrecipFwdA, kPrecipFwdB, i);
    f.sway = nearFormPick(kNearSwayA, kNearSwayB, i);
}

// The rate at p and the two largest forms of its mix. Two, because a mix
// rarely holds more (rain + graupel / hail under a core, rain + sleet or snow
// + sleet at the melting band); the rest of the mass is not drawn.
float nearPrecipForms(vec3 p, out NearPrecipForm f0, out NearPrecipForm f1) {
    f0 = nearFormNone();
    f1 = nearFormNone();
    vec3 fallLine;
    CloudPrecipForm form;
    float rate = cloudPrecipSampleForm(p, fallLine, form);
    if (rate <= 0.0) return 0.0;
    for (int i = 0; i < 8; ++i) {
        float s = nearFormPick(form.a, form.b, i);
        if (s > f0.share) {
            f1 = f0;
            f0.id = i + 1;
            f0.share = s;
        } else if (s > f1.share) {
            f1.id = i + 1;
            f1.share = s;
        }
    }
    nearFormFill(f0, rate);
    nearFormFill(f1, rate);
    return rate;
}

// ── Per-frame camera state (cloud_sample.comp mode 7) ──────────────────────
// Everything the drops need that is the SAME for every pixel: the rate and
// form mix at the camera and the cloud shadow on it. Computed ONCE per cloud
// update into cloudParams.nearState (Backend::CloudNearStateGPU); the draw
// half only reads it. Evaluating it per pixel was ~6 heavy cloud lookups
// x 1.6 M pixels per RayFusion frame.
//   s[0] = rate mm/h, first form id, second form id (0 = not drawn), sun shade
//   s[1] = first form share, fall m/s, drops / m^3, radius m
//   s[2] = first form forward-lobe share, sway
//   s[3], s[4] = the same for the second form
void nearStatePack(NearPrecipForm f, out vec4 a, out vec4 b) {
    a = vec4(f.share, f.fall, f.density, f.radius);
    b = vec4(f.forward, f.sway, 0.0, 0.0);
}

NearPrecipForm nearStateUnpack(float id, vec4 a, vec4 b) {
    NearPrecipForm f;
    f.id = int(id + 0.5);
    f.share = a.x; f.fall = a.y; f.density = a.z; f.radius = a.w;
    f.forward = b.x; f.sway = b.y;
    return f;
}

// Sun transmittance of the cloud field above `p`: base-shape density along
// the sun direction (one thread per frame, so 48 samples are cheap).
float nearSunShade(vec3 p, vec3 sunDir) {
    if (sunDir.y < -0.05) return 0.0;
    float len = min(40000.0, 16000.0 / max(sunDir.y, 0.05));
    float dt = len / 48.0;
    float tau = 0.0;
    for (int i = 0; i < 48; ++i) {
        tau += cloudDensity(p + sunDir * ((float(i) + 0.5) * dt), false) * dt;
        if (tau > 20.0) break;
    }
    return exp(-tau);
}

void nearStateCompute(vec3 camera, vec3 sunDir, out vec4 s[6]) {
    for (int i = 0; i < 6; ++i) s[i] = vec4(0.0);
    NearPrecipForm f0, f1;
    float rate = nearPrecipForms(camera, f0, f1);
    if (rate <= 0.0 || f0.id <= 0) return;
    bool second = f1.id > 0 && f1.share > 0.05;
    s[0] = vec4(rate, float(f0.id), second ? float(f1.id) : 0.0, nearSunShade(camera, sunDir));
    nearStatePack(f0, s[1], s[2]);
    if (second) nearStatePack(f1, s[3], s[4]);
}

#ifdef NEAR_PRECIP_DRAW

// A drawn streak is at most this opaque; the rest of a band's mean coverage
// goes into its veil. Without the cap a far band's few representative
// streaks turn into thick opaque strings.
const float kNearAlphaMax = 0.35;
// Mean of the streak's cross profile (1 - smoothstep(0.5, 1, x)) over |x| < 1.
const float kNearProfileMean = 0.75;

uint nearHash(ivec3 c, uint salt) {
    uint h = uint(c.x) * 0x8da6b343u ^ uint(c.y) * 0xd8163841u ^ uint(c.z) * 0xcb1ab31fu ^ salt;
    h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15; h *= 0x846ca68bu; h ^= h >> 16;
    return h;
}

float nearRand(inout uint h) {
    h = h * 747796405u + 2891336453u;
    uint w = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    return float((w >> 22u) ^ w) * (1.0 / 4294967296.0);
}

// Closest approach of the ray s*d (s in [0, sMax], d unit, origin 0) and the
// segment a..b. Returns the distance; sOut = the ray parameter there.
float nearSegmentDistance(vec3 d, float sMax, vec3 a, vec3 b, out float sOut) {
    vec3 u = b - a;
    vec3 w = -a;
    float uu = max(dot(u, u), 1e-12);
    float du = dot(d, u), dw = dot(d, w), uw = dot(u, w);
    float den = uu - du * du;
    float k = den > 1e-10 * uu ? clamp((uw - du * dw) / den, 0.0, 1.0) : 0.0;
    float s = clamp(du * k - dw, 0.0, sMax);
    k = clamp((uw + du * s) / uu, 0.0, 1.0);
    sOut = s;
    return length(w + d * s - u * k);
}

// One form's drops along the camera ray: a lattice moving with the drops
// (q = p - v t), one drop at most per cell, cells stretched along v so the
// whole exposure streak stays inside its own cell (one test, no neighbours).
// Six depth bands [0.25, 0.5, 1, 2, 4, 8, 16) m, cells growing with depth so
// each band costs about the same number of steps.
// Lc = the particle's own radiance (not counting the background it passes),
// keep = share of the background a covering particle lets through.
void nearPrecipLayer(NearPrecipForm f, uint salt, vec3 o, vec3 d, float tEnd, float tHit,
                     float pixelAngle, vec3 Lc, float keep, inout vec3 acc, inout float T,
                     inout int budget) {
    if (f.id <= 0 || f.density <= 0.0) return;
    float time = cloudNearPrecipTime();
    float shutter = cloudShutter();
    vec3 v = cloudSurfaceWind() + vec3(0.0, -f.fall, 0.0);
    float speed = length(v);
    vec3 e1 = v / speed;
    vec3 e2 = normalize(cross(e1, abs(e1.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0)));
    vec3 e3 = cross(e1, e2);
    float streak = speed * shutter;               // metres swept in one exposure
    float r = f.radius;
    float sinView = length(cross(e1, d));         // streak foreshortening
    vec3 q0 = o - v * time;                       // camera in the drops' frame
    float absorb = 1.0 - keep;                    // background share a drop removes

    float z0 = 0.25;
    for (int band = 0; band < 6 && z0 < tEnd && budget > 0; ++band) {
        float z1 = min(z0 * 2.0, tEnd);
        float h = z0 * 0.2;
        float hv = max(h, streak * 1.25 + 0.05 * h);
        float w = f.density * h * h * hv;         // expected drops per cell
        float occ = min(w, 1.0);
        float weight = max(w, 1.0);               // drops one drawn streak stands for
        uint bandSalt = salt ^ (uint(band + 1) * 0x85ebca6bu);
        // Lattice coordinates: x along v (cell hv), y/z across (cell h).
        vec3 inv = vec3(1.0 / hv, 1.0 / h, 1.0 / h);
        vec3 Qo = vec3(dot(q0, e1), dot(q0, e2), dot(q0, e3)) * inv;
        vec3 Qd = vec3(dot(d, e1), dot(d, e2), dot(d, e3)) * inv;
        vec3 Pin = Qo + Qd * z0;
        ivec3 cell = ivec3(floor(Pin));
        ivec3 stp = ivec3(sign(Qd));
        vec3 tNext, tDelta;
        for (int k = 0; k < 3; ++k) {
            if (abs(Qd[k]) < 1e-9) {
                tNext[k] = 1e30;
                tDelta[k] = 1e30;
            } else {
                float edge = float(cell[k]) + (Qd[k] > 0.0 ? 1.0 : 0.0);
                tNext[k] = z0 + (edge - Pin[k]) / Qd[k];
                tDelta[k] = abs(1.0 / Qd[k]);
            }
        }
        float Ls = streak / hv;                   // streak length in cells
        float rhoMax = 0.2 * h;                   // stays inside the cross margin
        float t = z0;
        while (t < z1 && budget > 0) {
            --budget;
            uint hs = nearHash(cell, bandSalt);
            if (nearRand(hs) < occ) {
                float a1 = mix(Ls + 0.02, 0.98, nearRand(hs));
                float a2 = mix(0.3, 0.7, nearRand(hs));
                float a3 = mix(0.3, 0.7, nearRand(hs));
                // Flutter (snow): ~5 cm, kept inside the cross margin, at
                // both ends of the exposure.
                float ph = nearRand(hs) * 6.2831853;
                float om = mix(1.2, 2.6, nearRand(hs));
                float amp = min(f.sway * 0.05 / h, 0.08);
                vec2 sEnd = amp * vec2(sin(om * time + ph), cos(0.7 * om * time + 1.3 * ph));
                vec2 sBeg = amp * vec2(sin(om * (time - shutter) + ph),
                                       cos(0.7 * om * (time - shutter) + 1.3 * ph));
                vec3 base = vec3(cell) - Qo;
                vec3 headQ = base + vec3(a1, a2 + sEnd.x, a3 + sEnd.y);
                vec3 tailQ = base + vec3(a1 - Ls, a2 + sBeg.x, a3 + sBeg.y);
                // Camera-relative metres (the frame offset v t cancels: both
                // the camera and the drop were moved into the same frame).
                vec3 head = e1 * (headQ.x * hv) + e2 * (headQ.y * h) + e3 * (headQ.z * h);
                vec3 tail = e1 * (tailQ.x * hv) + e2 * (tailQ.y * h) + e3 * (tailQ.z * h);
                float s;
                // Surfaces hide the drops behind them: the ray ends at the hit.
                float dist = nearSegmentDistance(d, tHit, tail, head, s);
                float rho = clamp(max(r, 0.5 * pixelAngle * s), r, max(rhoMax, r));
                if (dist < rho) {
                    float lPerp = length(cross(head - tail, d));
                    float area = 2.0 * rho * (lPerp + 2.0 * rho);
                    float alpha = min(weight * CLOUD_PI * r * r / (area * kNearProfileMean), kNearAlphaMax);
                    alpha *= 1.0 - smoothstep(0.5 * rho, rho, dist);
                    acc += T * alpha * Lc;
                    T *= 1.0 - alpha * absorb;
                }
            }
            if (tNext.x < tNext.y && tNext.x < tNext.z) {
                cell.x += stp.x; t = tNext.x; tNext.x += tDelta.x;
            } else if (tNext.y < tNext.z) {
                cell.y += stp.y; t = tNext.y; tNext.y += tDelta.y;
            } else {
                cell.z += stp.z; t = tNext.z; tNext.z += tDelta.z;
            }
        }
        // Veil: the share of the band's mean coverage the capped streaks
        // could not carry (estimated at the band's middle depth), spread
        // evenly. Only the metres in front of the surface.
        float zm = 0.5 * (z0 + z1);
        float rhoM = clamp(max(r, 0.5 * pixelAngle * zm), r, max(rhoMax, r));
        float areaM = 2.0 * rhoM * (streak * sinView + 2.0 * rhoM);
        float need = weight * CLOUD_PI * r * r / (areaM * kNearProfileMean);
        float carried = need > kNearAlphaMax ? kNearAlphaMax / need : 1.0;
        float depth = max(min(z1, tHit) - z0, 0.0);
        float veil = 1.0 - exp(-f.density * CLOUD_PI * r * r * depth * (1.0 - carried));
        acc += T * veil * Lc;
        T *= 1.0 - veil * absorb;
        z0 = z1;
    }
}

// The particle's own light and the share of the background it passes.
//   ice:    the curtain's sample light, albedo 0.95, opaque (keep 0);
//   liquid: a lens -- the sun glint (forward lobe, the same phase) plus the
//           surroundings half of its view (kNearLensTau x ambient x 0.5);
//           the pixel half is passed through (kNearLensKeep).
// Mixed forms (sleet) blend the two by their liquid share.
void nearParticleLight(NearPrecipForm f, vec3 sunLit, float mu, vec3 ambient,
                       out vec3 Lc, out float keep) {
    float pIso = 1.0 / (4.0 * CLOUD_PI);
    float pFwd = cloudPhaseHG(0.6, mu);
    vec3 glint = sunLit * mix(pIso, pFwd, f.forward);
    float liquid = f.id > 0 ? nearFormPick(kNearLiquidA, kNearLiquidB, f.id - 1) : 0.0;
    vec3 Lice = (glint + ambient) * 0.95;
    vec3 Llens = glint * kNearLensTau + ambient * (kNearLensTau - kNearLensKeep);
    Lc = mix(Lice, Llens, liquid);
    keep = kNearLensKeep * liquid;
}

// Drops in front of whatever the camera ray (o = camera, d unit) sees up to
// tHit: result = acc + T * behind. pixelAngle = radians per pixel (a drop is
// never drawn thinner than half a pixel; thinner ones fade by area).
void nearPrecipComposite(vec3 o, vec3 d, float tHit, float pixelAngle, out vec3 acc, out float T) {
    acc = vec3(0.0);
    T = 1.0;
    float range = cloudNearPrecipRange();
    if (range <= 0.0 || !cloudRtEnabled()) return;
    float tEnd = min(tHit, range);
    if (tEnd <= 0.25) return;
    // The camera's state, computed once per cloud update (nearStateCompute).
    vec4 s0 = cloudParams.nearState.state[0];
    if (s0.x <= 0.0 || s0.y < 0.5) return;
    NearPrecipForm f0 = nearStateUnpack(s0.y, cloudParams.nearState.state[1], cloudParams.nearState.state[2]);
    NearPrecipForm f1 = nearStateUnpack(s0.z, cloudParams.nearState.state[3], cloudParams.nearState.state[4]);

    // Light: the curtain's sample light at the camera (cloudPrecipMarch):
    // sun through the cloud column above (the per-frame shade), sky / ground
    // ambient scaled by that shade, the lit cloud base during a flash.
    vec3 sunDir = normalize(worldData.w.sunDir);
    float mu = dot(d, sunDir);
    vec3 Es = cloudSunIrradiance(o, sunDir);
    float shade = s0.w;
    vec3 ambTop = cloudSkyAmbient(normalize(vec3(0.0, 1.0, 0.0) + d * 0.2));
    vec3 ambient = ambTop * (0.275 + 0.35 * shade);
    if (cloudFlashAny())
        ambient += cloudFlashMediumLight(o, cloudLightningActivities()) * (1.0 / (4.0 * CLOUD_PI));

    uint seed = uint(cloudParams.misc.y) * 0x9e3779b9u;
    int budget = 200;
    vec3 Lc0;
    float keep0;
    nearParticleLight(f0, Es * shade, mu, ambient, Lc0, keep0);
    nearPrecipLayer(f0, seed ^ (uint(f0.id) * 0x27d4eb2du), o, d, tEnd, tHit, pixelAngle,
                    Lc0, keep0, acc, T, budget);
    if (f1.id > 0) {
        vec3 Lc1;
        float keep1;
        nearParticleLight(f1, Es * shade, mu, ambient, Lc1, keep1);
        nearPrecipLayer(f1, seed ^ (uint(f1.id) * 0x27d4eb2du), o, d, tEnd, tHit, pixelAngle,
                        Lc1, keep1, acc, T, budget);
    }
}

#endif // NEAR_PRECIP_DRAW
#endif // NEAR_PRECIP_GLSL
