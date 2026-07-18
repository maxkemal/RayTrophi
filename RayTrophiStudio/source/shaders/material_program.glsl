// =============================================================================
// material_program.glsl — Faz 2b GPU port of the CPU MaterialProgram VM.
// =============================================================================
// Direct mirror of include/MaterialProgram.h (evalMaterialProgram) and the
// noise in include/MaterialProceduralMath.h. A material node graph compiled by
// compileMaterialProgram (MaterialNodesV2.h) is flattened into ONE storage
// buffer (binding 23) and interpreted here per shading point, so procedural
// chains (Noise/Voronoi/Checker/ColorRamp/Mix/Math) shade PER PIXEL in the
// Vulkan RT final render — matching the CPU reference exactly.
//
// Buffer layout (all uint words; floats stored bit-reinterpreted):
//   [0]              materialCount M
//   [1 .. M]         procOffset[materialIndex]   (0xFFFFFFFF = no program)
//   [1+M]            constBaseWord  (word index where the float const pool begins)
//   [1+M+1 ..]       instruction stream: 8 words/instr, programs concatenated,
//                    each program terminated by an End instr (op = 0xFFFF)
//   [constBaseWord..] const pool floats (uintBitsToFloat)
//
// Instruction word layout (matches MatInstr):
//   w+0 op   w+1 outReg   w+2..4 inReg[0..2]   w+5 constOff   w+6 aux   w+7 iparam
// Signed fields (-1) are stored as their two's-complement bits; int(word) recovers them.
//
// NOTE: programs that sample textures (TexColor/TexAlpha ops) are now fully supported.
// The CPU flattener resolves their texture handles into bindless GPU indices via the IBackend interface.
#ifndef MATERIAL_PROGRAM_GLSL
#define MATERIAL_PROGRAM_GLSL

layout(set = 0, binding = 23, std430) readonly buffer MatProgramBuffer { uint w[]; } matprog;

const uint MATPROG_NONE = 0xFFFFFFFFu;
const uint MP_END       = 0xFFFFu;

// MatSlot bits (index i -> bit 1<<i), mirrors MatSlot enum order.
const uint MP_SLOT_BASECOLOR        = 1u << 0;
const uint MP_SLOT_METALLIC         = 1u << 1;
const uint MP_SLOT_ROUGHNESS        = 1u << 2;
const uint MP_SLOT_SPECULAR         = 1u << 3;
const uint MP_SLOT_TRANSMISSION     = 1u << 4;
const uint MP_SLOT_EMISSIONCOLOR    = 1u << 5;
const uint MP_SLOT_EMISSIONSTRENGTH = 1u << 6;
const uint MP_SLOT_OPACITY          = 1u << 7;
const uint MP_SLOT_IOR              = 1u << 8;
const uint MP_SLOT_NORMAL           = 1u << 9;   // tangent-space perturbed normal (Bump)

struct MatProgOut {
    vec3  baseColor;
    float metallic, roughness, specular, transmission;
    vec3  emissionColor;
    float emissionStrength, opacity, ior;
    vec3  normal;      // tangent-space (Bump); default (0,0,1)
    bool  normalWorld; // StoreWorldNormal (Bevel): `normal` is WORLD-space — no TBN
    uint  written;
};

// ---- shared procedural math (port of MaterialProceduralMath.h) --------------
uint mp_pcgHash(uint x) {
    x = x * 747796405u + 2891336453u;
    x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    return (x >> 22u) ^ x;
}
uint mp_hash2i(int xi, int yi, uint seed) {
    return mp_pcgHash(mp_pcgHash(uint(xi) + seed * 0x9E3779B9u) + uint(yi));
}
float mp_hash2f(int xi, int yi, uint seed) {
    return float(mp_hash2i(xi, yi, seed)) * (1.0 / 4294967295.0);
}
float mp_smoothFade(float t) { return t * t * t * (t * (t * 6.0 - 15.0) + 10.0); }

// Object Info -> Random. Mirrors objectRandom01 (MaterialProceduralMath.h) exactly:
// the float BITS of the object's world origin through the PCG avalanche chain, with
// -0.0 folded to +0.0 first. Hashing the origin (not an instance id) is what lets the
// CPU and this shader agree — see the CPU header for the full reasoning.
float mp_objectRandom01(vec3 o) {
    if (o.x == 0.0) o.x = 0.0;
    if (o.y == 0.0) o.y = 0.0;
    if (o.z == 0.0) o.z = 0.0;
    uint h = mp_pcgHash(mp_pcgHash(mp_pcgHash(floatBitsToUint(o.x)) ^ floatBitsToUint(o.y)) ^ floatBitsToUint(o.z));
    return float(h) * (1.0 / 4294967296.0);   // [0,1)
}

float mp_valueNoise2D(float x, float y, uint seed) {
    int xi = int(floor(x));
    int yi = int(floor(y));
    float fx = x - float(xi);
    float fy = y - float(yi);
    float v00 = mp_hash2f(xi,     yi,     seed);
    float v10 = mp_hash2f(xi + 1, yi,     seed);
    float v01 = mp_hash2f(xi,     yi + 1, seed);
    float v11 = mp_hash2f(xi + 1, yi + 1, seed);
    float tx = mp_smoothFade(fx), ty = mp_smoothFade(fy);
    float a = v00 + (v10 - v00) * tx;
    float b = v01 + (v11 - v01) * tx;
    return a + (b - a) * ty;
}
float mp_fbm2D(float x, float y, int octaves, float gain, uint seed) {
    float sum = 0.0, amp = 1.0, norm = 0.0, freq = 1.0;
    octaves = clamp(octaves, 1, 8);
    for (int i = 0; i < octaves; ++i) {
        sum += mp_valueNoise2D(x * freq, y * freq, seed + uint(i) * 101u) * amp;
        norm += amp; amp *= gain; freq *= 2.0;
    }
    return (norm > 0.0) ? sum / norm : 0.0;
}
float mp_ridge2D(float x, float y, int octaves, float gain, uint seed) {
    float sum = 0.0, amp = 1.0, norm = 0.0, freq = 1.0;
    octaves = clamp(octaves, 1, 8);
    for (int i = 0; i < octaves; ++i) {
        float n = 1.0 - abs(2.0 * mp_valueNoise2D(x * freq, y * freq, seed + uint(i) * 101u) - 1.0);
        sum += n * n * amp;
        norm += amp; amp *= gain; freq *= 2.0;
    }
    return (norm > 0.0) ? sum / norm : 0.0;
}
float mp_billow2D(float x, float y, int octaves, float gain, uint seed) {
    float sum = 0.0, amp = 1.0, norm = 0.0, freq = 1.0;
    octaves = clamp(octaves, 1, 8);
    for (int i = 0; i < octaves; ++i) {
        sum += abs(2.0 * mp_valueNoise2D(x * freq, y * freq, seed + uint(i) * 101u) - 1.0) * amp;
        norm += amp; amp *= gain; freq *= 2.0;
    }
    return (norm > 0.0) ? sum / norm : 0.0;
}
float mp_voronoiF1(float px, float py, float randomness, uint s, out uint cellHash) {
    int xi = int(floor(px));
    int yi = int(floor(py));
    float best = 1e30;
    cellHash = 0u;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int cx = xi + dx, cy = yi + dy;
            uint h = mp_hash2i(cx, cy, s);
            float jx = (float(mp_pcgHash(h ^ 0x1u)) * (1.0 / 4294967295.0) - 0.5) * randomness;
            float jy = (float(mp_pcgHash(h ^ 0x2u)) * (1.0 / 4294967295.0) - 0.5) * randomness;
            float fx = float(cx) + 0.5 + jx - px;
            float fy = float(cy) + 0.5 + jy - py;
            float d = fx * fx + fy * fy;
            if (d < best) { best = d; cellHash = h; }
        }
    }
    return sqrt(best);
}

// ---- 3D variants (position-driven, seamless solid texturing) ----------------
uint mp_hash3i(int xi, int yi, int zi, uint seed) {
    return mp_pcgHash(mp_pcgHash(mp_pcgHash(uint(xi) + seed * 0x9E3779B9u) + uint(yi)) + uint(zi));
}
float mp_hash3f(int xi, int yi, int zi, uint seed) {
    return float(mp_hash3i(xi, yi, zi, seed)) * (1.0 / 4294967295.0);
}
float mp_valueNoise3D(float x, float y, float z, uint seed) {
    int xi = int(floor(x)); int yi = int(floor(y)); int zi = int(floor(z));
    float fx = mp_smoothFade(x - float(xi));
    float fy = mp_smoothFade(y - float(yi));
    float fz = mp_smoothFade(z - float(zi));
    float c000 = mp_hash3f(xi, yi, zi, seed),     c100 = mp_hash3f(xi+1, yi, zi, seed);
    float c010 = mp_hash3f(xi, yi+1, zi, seed),   c110 = mp_hash3f(xi+1, yi+1, zi, seed);
    float c001 = mp_hash3f(xi, yi, zi+1, seed),   c101 = mp_hash3f(xi+1, yi, zi+1, seed);
    float c011 = mp_hash3f(xi, yi+1, zi+1, seed), c111 = mp_hash3f(xi+1, yi+1, zi+1, seed);
    float x00 = mix(c000, c100, fx), x10 = mix(c010, c110, fx);
    float x01 = mix(c001, c101, fx), x11 = mix(c011, c111, fx);
    return mix(mix(x00, x10, fy), mix(x01, x11, fy), fz);
}
float mp_fbm3D(float x, float y, float z, int octaves, float gain, uint seed) {
    float sum = 0.0, amp = 1.0, norm = 0.0, freq = 1.0;
    octaves = clamp(octaves, 1, 8);
    for (int i = 0; i < octaves; ++i) {
        sum += mp_valueNoise3D(x*freq, y*freq, z*freq, seed + uint(i)*101u) * amp;
        norm += amp; amp *= gain; freq *= 2.0;
    }
    return (norm > 0.0) ? sum / norm : 0.0;
}
float mp_ridge3D(float x, float y, float z, int octaves, float gain, uint seed) {
    float sum = 0.0, amp = 1.0, norm = 0.0, freq = 1.0;
    octaves = clamp(octaves, 1, 8);
    for (int i = 0; i < octaves; ++i) {
        float n = 1.0 - abs(2.0 * mp_valueNoise3D(x*freq, y*freq, z*freq, seed + uint(i)*101u) - 1.0);
        sum += n * n * amp;
        norm += amp; amp *= gain; freq *= 2.0;
    }
    return (norm > 0.0) ? sum / norm : 0.0;
}
float mp_billow3D(float x, float y, float z, int octaves, float gain, uint seed) {
    float sum = 0.0, amp = 1.0, norm = 0.0, freq = 1.0;
    octaves = clamp(octaves, 1, 8);
    for (int i = 0; i < octaves; ++i) {
        sum += abs(2.0 * mp_valueNoise3D(x*freq, y*freq, z*freq, seed + uint(i)*101u) - 1.0) * amp;
        norm += amp; amp *= gain; freq *= 2.0;
    }
    return (norm > 0.0) ? sum / norm : 0.0;
}
float mp_voronoi3D_F1(float px, float py, float pz, float randomness, uint s, out uint cellHash) {
    int xi = int(floor(px)); int yi = int(floor(py)); int zi = int(floor(pz));
    float best = 1e30; cellHash = 0u;
    for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
        int cx = xi+dx, cy = yi+dy, cz = zi+dz;
        uint h = mp_hash3i(cx, cy, cz, s);
        float jx = (float(mp_pcgHash(h ^ 0x1u)) * (1.0/4294967295.0) - 0.5) * randomness;
        float jy = (float(mp_pcgHash(h ^ 0x2u)) * (1.0/4294967295.0) - 0.5) * randomness;
        float jz = (float(mp_pcgHash(h ^ 0x3u)) * (1.0/4294967295.0) - 0.5) * randomness;
        float fx = float(cx) + 0.5 + jx - px;
        float fy = float(cy) + 0.5 + jy - py;
        float fz = float(cz) + 0.5 + jz - pz;
        float d = fx*fx + fy*fy + fz*fz;
        if (d < best) { best = d; cellHash = h; }
    }
    return sqrt(best);
}

// ---- HSV (port of rgbToHsv / hsvToRgb in MaterialProceduralMath.h) ----------
// Same branch structure as the CPU on purpose, including "grey => hue 0": a branchless
// rewrite here is exactly how the two backends end up disagreeing on the grey axis.
vec3 mp_rgbToHsv(vec3 c) {
    float mx = max(c.r, max(c.g, c.b));
    float mn = min(c.r, min(c.g, c.b));
    float d  = mx - mn;
    float v = mx;
    float s = (mx > 1e-8) ? (d / mx) : 0.0;
    float h = 0.0;
    if (d >= 1e-8) {
        if (mx == c.r)      h = (c.g - c.b) / d + ((c.g < c.b) ? 6.0 : 0.0);
        else if (mx == c.g) h = (c.b - c.r) / d + 2.0;
        else                h = (c.r - c.g) / d + 4.0;
        h *= 1.0 / 6.0;
    }
    return vec3(h, s, v);
}
vec3 mp_hsvToRgb(vec3 hsv) {
    float h = hsv.x - floor(hsv.x);
    float s = clamp(hsv.y, 0.0, 1.0);
    float v = hsv.z;
    float i = floor(h * 6.0);
    float f = h * 6.0 - i;
    float p = v * (1.0 - s);
    float q = v * (1.0 - s * f);
    float t = v * (1.0 - s * (1.0 - f));
    int idx = int(mod(i, 6.0));
    if (idx == 0) return vec3(v, t, p);
    if (idx == 1) return vec3(q, v, p);
    if (idx == 2) return vec3(p, v, t);
    if (idx == 3) return vec3(p, q, v);
    if (idx == 4) return vec3(t, p, v);
    return vec3(v, p, q);
}

// ---- register helpers -------------------------------------------------------
float mp_rscalar(vec3 v) { return (v.x + v.y + v.z) * (1.0 / 3.0); }

/// Cosine between two register vectors, normalizing BOTH (a Normal socket can be fed any
/// vector). Mirrors detail::cosBetween in MaterialProgram.h. Degenerate -> 1 (head-on).
float mp_cosBetween(vec3 a, vec3 b) {
    float la = length(a);
    float lb = length(b);
    if (la < 1e-8 || lb < 1e-8) return 1.0;
    return clamp(dot(a, b) / (la * lb), -1.0, 1.0);
}

uint matProgramOffset(uint matIndex) {
    uint count = matprog.w[0];
    if (matIndex >= count) return MATPROG_NONE;
    return matprog.w[1u + matIndex];
}

MatProgOut mp_defaultOut() {
    MatProgOut o;
    o.baseColor = vec3(0.8); o.metallic = 0.0; o.roughness = 0.5; o.specular = 0.5;
    o.transmission = 0.0; o.emissionColor = vec3(0.0); o.emissionStrength = 0.0;
    o.opacity = 1.0; o.ior = 1.45; o.normal = vec3(0.0, 0.0, 1.0); o.normalWorld = false;
    o.written = 0u;
    return o;
}

// Named per-vertex attribute slots (Attribute node). MUST equal kMatAttribSlots in
// MaterialProgram.h: the mesh uploads one interleaved block of this many floats per
// vertex and the compiler bakes slot indices into the instruction stream.
const int MP_ATTRIB_SLOTS = 4;

// Run the compiled program starting at word `pc`. `gpos`/`gnrm` are the world-
// space shading point (for Geometry / 3D noise), `gpoint` the barycentric
// pointiness at that point (0.5 = flat), `gobj` the hit object's world origin
// (gl_ObjectToWorldEXT[3].xyz) for Object Info, `gattr` the named per-vertex
// channels already blended at the hit, and `gview` the unit vector from the shading
// point TOWARD the viewer (-gl_WorldRayDirectionEXT) for Fresnel / Layer Weight.
// Immutable buffer; a local register file (per invocation) keeps it re-entrant.
//
// AO: a stage that CAN trace rays (closesthit) defines MP_HAS_AO and provides
//     float mp_traceAO(vec3 p, vec3 n, float dist, int samples, bool inside);
// declaring the prototype BEFORE including this file. Stages where tracing is illegal
// (shadow any-hit) or meaningless (ray-gen) simply leave it undefined -> AO reads 1.0.
MatProgOut evalMaterialProgram(uint pc, vec2 uv, vec3 gpos, vec3 gnrm, float gpoint, vec3 gobj,
                               float gattr[MP_ATTRIB_SLOTS], vec3 gobjp, vec3 gview) {
    MatProgOut o = mp_defaultOut();
    uint count = matprog.w[0];
    uint constBase = matprog.w[1u + count];

    // Register file. Dynamically indexed => lives in per-thread scratch memory,
    // so its size is a direct occupancy cost on every invocation. The host
    // compiler compacts register live ranges and deactivates any program whose
    // PEAK LIVE register count exceeds kMatMaxRegs (MaterialProgram.h) — this
    // size MUST equal that constant.
    const int MP_MAX_REGS = 32;
    vec3 regs[MP_MAX_REGS];

    for (uint guard = 0u; guard < 8192u; ++guard) {
        uint op = matprog.w[pc];
        if (op == MP_END) break;
        int outReg = int(matprog.w[pc + 1u]);
        int in0    = int(matprog.w[pc + 2u]);
        int in1    = int(matprog.w[pc + 3u]);
        int in2    = int(matprog.w[pc + 4u]);
        int cOff   = int(matprog.w[pc + 5u]);
        int aux    = int(matprog.w[pc + 6u]);
        int ip     = int(matprog.w[pc + 7u]);
        vec3 r = vec3(0.0);

        if (op == 0u) {            // Const
            uint b = constBase + uint(cOff);
            r = vec3(uintBitsToFloat(matprog.w[b]), uintBitsToFloat(matprog.w[b + 1u]), uintBitsToFloat(matprog.w[b + 2u]));
        } else if (op == 1u) {     // UV
            r = vec3(uv, 0.0);
        } else if (op == 2u) {     // Mapping
            vec3 t = regs[in0];
            uint b = constBase + uint(cOff);
            float sx = uintBitsToFloat(matprog.w[b]);
            float sy = uintBitsToFloat(matprog.w[b + 1u]);
            float ox = uintBitsToFloat(matprog.w[b + 2u]);
            float oy = uintBitsToFloat(matprog.w[b + 3u]);
            float rot = uintBitsToFloat(matprog.w[b + 4u]);
            float rad = rot * 3.14159265 / 180.0;
            float cr = cos(rad), sr = sin(rad);
            float cx = t.x - 0.5, cy = t.y - 0.5;
            float rx = cx * cr - cy * sr;
            float ry = cx * sr + cy * cr;
            r = vec3((rx + 0.5) * sx + ox, (ry + 0.5) * sy + oy, 0.0);
        } else if (op == 3u) {     // TexColor
            // The VM runs in the CPU's raw mesh-UV space (V up). Texture::get_color_bilinear
            // flips V inside the fetch; do the same here so the two backends sample the
            // same texel from the same register value. Every op upstream of this one then
            // sees identical inputs on CPU and GPU.
            vec3 t = regs[in0];
            uint texID = aux;
            if (texID != 0u) {
                r = texture(materialTextures[nonuniformEXT(texID)], vec2(t.x, 1.0 - t.y)).rgb;
            } else {
                r = vec3(0.8, 0.0, 0.8);
            }
        } else if (op == 4u) {     // TexAlpha
            vec3 t = regs[in0];
            uint texID = aux;
            if (texID != 0u) {
                r = vec3(texture(materialTextures[nonuniformEXT(texID)], vec2(t.x, 1.0 - t.y)).a);
            } else {
                r = vec3(1.0);
            }
        } else if (op == 5u) {     // Noise (kind in const[0])
            vec3 t = regs[in0];
            uint b = constBase + uint(cOff);
            int   kind   = int(uintBitsToFloat(matprog.w[b]));
            float scale  = uintBitsToFloat(matprog.w[b + 1u]);
            int   detail = int(uintBitsToFloat(matprog.w[b + 2u]));
            float rough  = uintBitsToFloat(matprog.w[b + 3u]);
            float rand   = uintBitsToFloat(matprog.w[b + 4u]);
            float dist   = uintBitsToFloat(matprog.w[b + 5u]);
            int   seed   = int(uintBitsToFloat(matprog.w[b + 6u]));
            uint s = (kind == 4)
                ? uint(seed) * 0x68bc21ebu + 0x2545F491u
                : uint(seed) * 0x51633e2du + 0x9E3779B9u;
            uint cellHash = 0u;
            float fac;
            if (aux == 3) {          // 3D (position-driven, seamless)
                float x = t.x * scale, y = t.y * scale, z = t.z * scale;
                if (kind == 1)      fac = mp_ridge3D(x, y, z, detail, rough, s);
                else if (kind == 2) fac = mp_billow3D(x, y, z, detail, rough, s);
                else if (kind == 3) {
                    float wx = mp_fbm3D(x + 17.3, y + 9.1, z + 3.7, detail, rough, s ^ 0xA511u) - 0.5;
                    float wy = mp_fbm3D(x - 11.7, y + 4.9, z - 8.2, detail, rough, s ^ 0x3C6Fu) - 0.5;
                    float wz = mp_fbm3D(x + 5.1, y - 6.3, z + 12.9, detail, rough, s ^ 0x77A5u) - 0.5;
                    fac = mp_fbm3D(x + wx*dist*8.0, y + wy*dist*8.0, z + wz*dist*8.0, detail, rough, s);
                } else if (kind == 4) fac = clamp(mp_voronoi3D_F1(x, y, z, rand, s, cellHash), 0.0, 1.0);
                else if (kind == 5) {
                    int cx = int(floor(x)); int cy = int(floor(y)); int cz = int(floor(z));
                    fac = (((cx + cy + cz) & 1) != 0) ? 1.0 : 0.0;
                } else fac = mp_fbm3D(x, y, z, detail, rough, s);
            } else {                 // 2D (UV-driven)
                float x = t.x * scale, y = t.y * scale;
                if (kind == 1)      fac = mp_ridge2D(x, y, detail, rough, s);
                else if (kind == 2) fac = mp_billow2D(x, y, detail, rough, s);
                else if (kind == 3) {
                    float wx = mp_fbm2D(x + 17.3, y + 9.1, detail, rough, s ^ 0xA511u) - 0.5;
                    float wy = mp_fbm2D(x - 11.7, y + 4.9, detail, rough, s ^ 0x3C6Fu) - 0.5;
                    fac = mp_fbm2D(x + wx * dist * 8.0, y + wy * dist * 8.0, detail, rough, s);
                } else if (kind == 4) fac = clamp(mp_voronoiF1(x, y, rand, s, cellHash), 0.0, 1.0);
                else if (kind == 5) {
                    int cx = int(floor(x)); int cy = int(floor(y));
                    fac = (((cx + cy) & 1) != 0) ? 1.0 : 0.0;
                } else fac = mp_fbm2D(x, y, detail, rough, s);
            }

            if (ip == 1) {          // Voronoi per-cell color
                r = vec3(float(mp_pcgHash(cellHash ^ 0x11u)) * (1.0 / 4294967295.0),
                         float(mp_pcgHash(cellHash ^ 0x22u)) * (1.0 / 4294967295.0),
                         float(mp_pcgHash(cellHash ^ 0x33u)) * (1.0 / 4294967295.0));
            } else if (ip == 2) {   // lerp(Color1, Color2, fac)
                r = mix(regs[in1], regs[in2], fac);
            } else {                // Fac / grayscale
                r = vec3(fac);
            }
        } else if (op == 6u) {     // ColorRamp
            float fac = clamp(mp_rscalar(regs[in0]), 0.0, 1.0);
            int n = aux;
            uint b = constBase + uint(cOff);
            if (n <= 0) { r = vec3(0.0); }
            else {
                float p0 = uintBitsToFloat(matprog.w[b]);
                float pN = uintBitsToFloat(matprog.w[b + uint((n - 1) * 4)]);
                if (fac <= p0) {
                    r = vec3(uintBitsToFloat(matprog.w[b + 1u]), uintBitsToFloat(matprog.w[b + 2u]), uintBitsToFloat(matprog.w[b + 3u]));
                } else if (fac >= pN) {
                    uint e = b + uint((n - 1) * 4);
                    r = vec3(uintBitsToFloat(matprog.w[e + 1u]), uintBitsToFloat(matprog.w[e + 2u]), uintBitsToFloat(matprog.w[e + 3u]));
                } else {
                    bool done = false;
                    for (int i = 0; i + 1 < n; ++i) {
                        uint bi = b + uint(i * 4);
                        float pa = uintBitsToFloat(matprog.w[bi]);
                        float pb = uintBitsToFloat(matprog.w[bi + 4u]);
                        if (fac <= pb) {
                            vec3 ca = vec3(uintBitsToFloat(matprog.w[bi + 1u]), uintBitsToFloat(matprog.w[bi + 2u]), uintBitsToFloat(matprog.w[bi + 3u]));
                            if (ip == 1) { r = ca; }   // Constant
                            else {
                                vec3 cb = vec3(uintBitsToFloat(matprog.w[bi + 5u]), uintBitsToFloat(matprog.w[bi + 6u]), uintBitsToFloat(matprog.w[bi + 7u]));
                                float span = pb - pa;
                                float tt = (span > 1e-6) ? (fac - pa) / span : 0.0;
                                r = mix(ca, cb, tt);
                            }
                            done = true;
                            break;
                        }
                    }
                    if (!done) {
                        uint e = b + uint((n - 1) * 4);
                        r = vec3(uintBitsToFloat(matprog.w[e + 1u]), uintBitsToFloat(matprog.w[e + 2u]), uintBitsToFloat(matprog.w[e + 3u]));
                    }
                }
            }
        } else if (op == 7u) {     // MixColor
            float fac = clamp(mp_rscalar(regs[in0]), 0.0, 1.0);
            vec3 a = regs[in1], b = regs[in2];
            vec3 m;
            if (ip == 1)      m = a + b;
            else if (ip == 2) m = a * b;
            else if (ip == 3) m = a - b;
            else if (ip == 4) m = vec3(1.0) - (vec3(1.0) - a) * (vec3(1.0) - b);
            else if (ip == 5) m = mix(2.0 * a * b, vec3(1.0) - 2.0 * (vec3(1.0) - a) * (vec3(1.0) - b), step(vec3(0.5), a));
            else              m = b;   // Mix
            r = mix(a, m, fac);
        } else if (op == 8u) {     // Invert
            float fac = clamp(mp_rscalar(regs[in0]), 0.0, 1.0);
            vec3 c = regs[in1];
            r = mix(c, vec3(1.0) - c, fac);
        } else if (op == 9u) {     // Gamma
            vec3 c = regs[in0];
            float inv = 1.0 / max(uintBitsToFloat(matprog.w[constBase + uint(cOff)]), 0.01);
            r = pow(max(c, vec3(0.0)), vec3(inv));
        } else if (op == 10u) {    // Math
            float a = mp_rscalar(regs[in0]);
            float b = mp_rscalar(regs[in1]);
            float m;
            if (ip == 1)      m = a - b;
            else if (ip == 2) m = a * b;
            else if (ip == 3) m = (abs(b) > 1e-8) ? a / b : 0.0;
            else if (ip == 4) m = pow(max(a, 0.0), b);
            else if (ip == 5) m = sqrt(max(a, 0.0));
            else if (ip == 6) m = abs(a);
            else if (ip == 7) m = min(a, b);
            else if (ip == 8) m = max(a, b);
            else if (ip == 9) m = clamp(a, 0.0, 1.0);
            // Appended ops (MathNode::Op ids are serialized, so new ones go at the end).
            else if (ip == 10) m = sin(a);
            else if (ip == 11) m = cos(a);
            else if (ip == 12) m = fract(a);
            else if (ip == 13) m = floor(a);
            else if (ip == 14) m = ceil(a);
            else if (ip == 15) m = (abs(b) > 1e-8) ? mod(a, b) : 0.0;
            else if (ip == 16) m = smoothstep(0.0, 1.0, a);
            else if (ip == 17) m = (a > b) ? 1.0 : 0.0;
            else if (ip == 18) m = (a < b) ? 1.0 : 0.0;
            else              m = a + b;
            r = vec3(m);
        } else if (op == 18u) {    // MatMapping (material-panel UV transform)
            // Mirrors PrincipledBSDF::applyTextureTransform exactly — see MaterialProgram.h.
            vec3 t = regs[in0];
            uint b = constBase + uint(cOff);
            float sx  = uintBitsToFloat(matprog.w[b]);
            float sy  = uintBitsToFloat(matprog.w[b + 1u]);
            float ox  = uintBitsToFloat(matprog.w[b + 2u]);
            float oy  = uintBitsToFloat(matprog.w[b + 3u]);
            float rot = uintBitsToFloat(matprog.w[b + 4u]);
            float tu  = uintBitsToFloat(matprog.w[b + 5u]);
            float tv  = uintBitsToFloat(matprog.w[b + 6u]);
            float u = t.x;
            float v = 1.0 - t.y;
            u -= 0.5; v -= 0.5;
            u *= sx;  v *= sy;
            float rad = rot * 3.14159265 / 180.0;
            float cr = cos(rad), sr = sin(rad);
            float nu = u * cr - v * sr;
            float nv = u * sr + v * cr;
            u = nu + 0.5 + ox;
            v = nv + 0.5 + oy;
            u *= tu;  v *= tv;
            r = vec3(u, 1.0 - v, 0.0);
        } else if (op == 19u) {    // CurveLUT
            float fac = clamp(mp_rscalar(regs[in0]), 0.0, 1.0);
            int n = aux;
            uint b = constBase + uint(cOff);
            float y;
            if (n <= 1) {
                y = (n == 1) ? uintBitsToFloat(matprog.w[b]) : 0.0;
            } else {
                float s = fac * float(n - 1);
                int i0 = min(int(s), n - 2);
                float f = s - float(i0);
                float y0 = uintBitsToFloat(matprog.w[b + uint(i0)]);
                float y1 = uintBitsToFloat(matprog.w[b + uint(i0) + 1u]);
                y = mix(y0, y1, f);
            }
            r = vec3(y);
        } else if (op == 11u) {    // Swizzle
            vec3 c = regs[in0];
            float ch = (ip == 1) ? c.y : (ip == 2) ? c.z : c.x;
            r = vec3(ch);
        } else if (op == 12u) {    // Combine
            r = vec3(mp_rscalar(regs[in0]), mp_rscalar(regs[in1]), mp_rscalar(regs[in2]));
        } else if (op == 14u) {    // GeoPosition
            r = gpos;
        } else if (op == 15u) {    // GeoNormal
            r = gnrm;
        } else if (op == 16u) {    // GeoPointiness
            // Per-vertex attribute (MeshPointiness.h), barycentric-interpolated by the
            // caller: 0.5 flat, >0.5 convex ridge, <0.5 concave crease.
            r = vec3(gpoint);
        } else if (op == 20u) {    // ObjLocation
            r = gobj;
        } else if (op == 21u) {    // ObjRandom
            r = vec3(mp_objectRandom01(gobj));
        } else if (op == 22u) {    // Attribute (aux = interned slot)
            // 0 when the slot is out of range — the caller already reads 0 for a mesh with
            // no attribute block, and the CPU VM does the same. An unpainted mask must read
            // 0 (nothing selected), never 1 (everything selected).
            float av = (aux >= 0 && aux < MP_ATTRIB_SLOTS) ? gattr[aux] : 0.0;
            r = vec3(av);
        } else if (op == 27u) {    // GeoPositionObj (object-space shading point)
            r = gobjp;
        } else if (op == 23u) {    // Wave
            vec3 t = regs[in0];
            uint b = constBase + uint(cOff);
            int   type    = int(uintBitsToFloat(matprog.w[b]));
            int   dir     = int(uintBitsToFloat(matprog.w[b + 1u]));
            int   profile = int(uintBitsToFloat(matprog.w[b + 2u]));
            float scale   = uintBitsToFloat(matprog.w[b + 3u]);
            float distort = uintBitsToFloat(matprog.w[b + 4u]);
            int   detail  = int(uintBitsToFloat(matprog.w[b + 5u]));
            float dScale  = uintBitsToFloat(matprog.w[b + 6u]);
            float phase   = uintBitsToFloat(matprog.w[b + 7u]);
            float x = t.x * scale, y = t.y * scale, z = t.z * scale;

            float n;
            if (type == 1) {          // Rings
                float rx = x, ry = y, rz = z;
                if (dir == 0) rx = 0.0; else if (dir == 1) ry = 0.0; else if (dir == 2) rz = 0.0;
                n = sqrt(rx * rx + ry * ry + rz * rz) * 20.0;
            } else {                  // Bands
                float d = (dir == 0) ? x : (dir == 1) ? y : (dir == 2) ? z : (x + y + z);
                n = d * 20.0;
            }
            n += phase;
            if (distort != 0.0 && detail > 0) {
                float w = (aux == 3)
                    ? mp_fbm3D(x * dScale, y * dScale, z * dScale, detail, 0.5, 0x9E37u)
                    : mp_fbm2D(x * dScale, y * dScale, detail, 0.5, 0x9E37u);
                n += distort * (w * 2.0 - 1.0);
            }

            float fac;
            const float TWO_PI = 6.2831853;
            if (profile == 1)      { float s = n / TWO_PI; fac = s - floor(s); }
            else if (profile == 2) { float s = n / TWO_PI; fac = abs(2.0 * (s - floor(s)) - 1.0); }
            else                   { fac = 0.5 + 0.5 * sin(n - 1.5707963); }
            r = vec3(clamp(fac, 0.0, 1.0));
        } else if (op == 24u) {    // Gradient
            vec3 t = regs[in0];
            float fac;
            if (ip == 1)      { float g = max(t.x, 0.0); fac = g * g; }
            else if (ip == 2) { float g = clamp(t.x, 0.0, 1.0); fac = g * g * (3.0 - 2.0 * g); }
            else if (ip == 3) { fac = (t.x + t.y) * 0.5; }
            else if (ip == 4) { fac = atan(t.y, t.x) / 6.2831853 + 0.5; }
            else if (ip == 5) { float g = max(1.0 - length(t), 0.0); fac = g * g; }
            else if (ip == 6) { fac = max(1.0 - length(t), 0.0); }
            else              { fac = t.x; }
            r = vec3(clamp(fac, 0.0, 1.0));
        } else if (op == 25u) {    // VectorMath
            vec3 a = regs[in0];
            vec3 b = regs[in1];
            if (ip == 1)      r = a - b;
            else if (ip == 2) r = a * b;
            else if (ip == 3) r = vec3((abs(b.x) > 1e-8) ? a.x / b.x : 0.0,
                                       (abs(b.y) > 1e-8) ? a.y / b.y : 0.0,
                                       (abs(b.z) > 1e-8) ? a.z / b.z : 0.0);
            else if (ip == 4) r = cross(a, b);
            else if (ip == 5) r = vec3(dot(a, b));
            else if (ip == 6) { float l = length(a); r = (l > 1e-8) ? (a / l) : vec3(0.0); }
            else if (ip == 7) r = vec3(length(a));
            else if (ip == 8) r = vec3(length(a - b));
            else if (ip == 9) { float l = length(b);
                                if (l > 1e-8) { vec3 nb = b / l; r = a - 2.0 * dot(a, nb) * nb; }
                                else r = a; }
            else if (ip == 10) { float s = uintBitsToFloat(matprog.w[constBase + uint(cOff)]); r = a * s; }
            else if (ip == 11) r = abs(a);
            else if (ip == 12) r = min(a, b);
            else if (ip == 13) r = max(a, b);
            else              r = a + b;
        } else if (op == 26u) {    // HSV
            vec3 col = regs[in0];
            vec3 adj = regs[in1];           // (hue, saturation, value)
            float fac = clamp(mp_rscalar(regs[in2]), 0.0, 1.0);
            vec3 hsv = mp_rgbToHsv(col);
            hsv.x += adj.x - 0.5;           // Blender: Hue 0.5 = no shift
            hsv.x = hsv.x - floor(hsv.x);
            hsv.y = clamp(hsv.y * adj.y, 0.0, 1.0);
            hsv.z *= adj.z;
            r = mix(col, mp_hsvToRgb(hsv), fac);
        } else if (op == 17u) {    // Fresnel
            float ior = mp_rscalar(regs[in0]);
            vec3 n = regs[in1];
            // Real viewing angle (see MaterialProgram.h): this read n.z before, which is
            // the normal's world-Z tilt, not N.V. abs() so a back-facing hit still reflects.
            float ndotv = abs(mp_cosBetween(n, gview));
            float r0 = (1.0 - ior) / (1.0 + ior);
            r0 = r0 * r0;
            float fc = pow(1.0 - ndotv, 5.0);
            r = vec3(r0 + (1.0 - r0) * fc);
        } else if (op == 28u) {    // GeoIncoming (view vector, toward the viewer)
            r = gview;
        } else if (op == 29u) {    // AmbientOcclusion
            // The VM does not trace: the includer supplies mp_traceAO (closesthit does,
            // via the SAME shadow payload/miss the NEE shadow ray uses). Where tracing is
            // illegal or pointless — shadow any-hit, ray-gen — MP_HAS_AO is undefined and
            // this collapses to 1.0 (unoccluded), the value an open surface has anyway.
            uint b = constBase + uint(cOff);
            int  samples = int(uintBitsToFloat(matprog.w[b]));
            bool inside  = uintBitsToFloat(matprog.w[b + 1u]) != 0.0;
#ifdef MP_HAS_AO
            r = vec3(mp_traceAO(gpos, gnrm, max(1e-4, regs[in0].x), samples, inside));
#else
            r = vec3(1.0);
#endif
        } else if (op == 30u) {    // Bevel (rounded-edge WORLD normal)
            // Second tracing op, same includer contract as AO: a stage that can trace
            // defines MP_HAS_BEVEL and provides
            //   vec3 mp_traceBevel(vec3 p, vec3 n, float radius, int samples);
            // anywhere else the op is the identity (the plain shading normal).
            uint b = constBase + uint(cOff);
            int samples = int(uintBitsToFloat(matprog.w[b]));
            float radius = max(1e-5, mp_rscalar(regs[in0]));
#ifdef MP_HAS_BEVEL
            r = mp_traceBevel(gpos, gnrm, radius, samples);
#else
            r = normalize(gnrm);
#endif
        } else if (op == 31u) {    // StoreWorldNormal (Bevel -> Normal slot, no TBN)
            o.normal = regs[in0];
            o.normalWorld = true;
            o.written |= (1u << 9u);   // MatSlot::Normal
            pc += 8u;
            continue;
        } else if (op == 13u) {    // Store
            vec3 src = regs[in0];
            if (aux == 0)      { o.baseColor = src; }
            else if (aux == 1) { o.metallic = mp_rscalar(src); }
            else if (aux == 2) { o.roughness = mp_rscalar(src); }
            else if (aux == 3) { o.specular = mp_rscalar(src); }
            else if (aux == 4) { o.transmission = mp_rscalar(src); }
            else if (aux == 5) { o.emissionColor = src; }
            else if (aux == 6) { o.emissionStrength = mp_rscalar(src); }
            else if (aux == 7) { o.opacity = mp_rscalar(src); }
            else if (aux == 8) { o.ior = mp_rscalar(src); }
            else if (aux == 9) { o.normal = src; }
            o.written |= (1u << uint(aux));
            pc += 8u;
            continue;
        }

        if (outReg >= 0) regs[outReg] = r;
        pc += 8u;
    }
    return o;
}

#endif // MATERIAL_PROGRAM_GLSL
