/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          MaterialProgram.h
* Author:        Kemal Demirtas
* Date:          July 2026
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once

#include "MaterialProceduralMath.h"
#include "Texture.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

// =============================================================================
// MATERIAL PROGRAM — Faz 2a per-pixel procedural runtime (SVM-lite)
// =============================================================================
// A material node graph, once authored, compiles (compileMaterialProgram in
// MaterialNodesV2.h) to this flat instruction stream + constant pool. The
// interpreter below runs at a shading point (u,v) and produces the material
// inputs scatter() reads — so a Noise/Voronoi/ColorRamp chain finally shades
// PER PIXEL instead of Faz 1's 5-sample average.
//
// Why a VM and not "just run the graph per pixel":
//   * thread-safety — the program is immutable, the register file is a local
//     stack array, so N render threads run the same program with no shared
//     mutable state (the graph's dirty/cache flags are single-threaded).
//   * portability — POD instructions + a float pool + plain math (the noise
//     lives in MaterialProceduralMath.h). Faz 2b/2c flatten `textures` to
//     bindless handles and emit the same opcodes as GLSL/CUDA; the opcode
//     semantics here are the reference.
//
// Design: SSA-ish register machine. Every node output writes one register
// (Vec3; scalars are stored broadcast, so read-as-scalar = channel average and
// read-as-color = passthrough, exactly matching the node getFloatIn/getVec3In
// convention). Unconnected inputs are pre-baked as Const instructions, so every
// instruction reads registers only.

namespace MaterialNodesV2 {

    // ---- Output slots the CPU scatter path consumes (Faz 2a scope). Slots not
    // listed here keep the material's Faz-1 constant / bound texture. -----------
    enum class MatSlot : uint16_t {
        BaseColor = 0,   // vec3
        Metallic,        // float
        Roughness,       // float
        Specular,        // float
        Transmission,    // float
        EmissionColor,   // vec3
        EmissionStrength,// float
        Opacity,         // float
        IOR,             // float
        Normal,          // vec3 (TANGENT-space perturbed normal from Bump; consumed
                         //       by apply_normal_map / closesthit, not folded)
        Count
    };

    enum class MatOp : uint16_t {
        Const = 0,   // constOff[0..2]                       -> out (rgb)
        UV,          // (u, v, 0)                            -> out
        Mapping,     // in0=uv ; const: sx,sy,ox,oy,rotDeg   -> out (uv)
        TexColor,    // in0=uv ; aux=texIndex                -> out (rgb)
        TexAlpha,    // in0=uv ; aux=texIndex                -> out (a,a,a)
        Noise,       // in0=uv,in1=c1,in2=c2 ; const: kind,scale,detail,rough,rand,dist,seed
                     //   iparam: 0=Fac/grayscale(splat) 1=VoronoiCellColor 2=CheckerLerp(c1,c2) -> out
        ColorRamp,   // in0=fac ; iparam=interp ; aux=stopCount ; const: (pos,r,g,b)* sorted -> out (rgb)
        MixColor,    // in0=fac,in1=a,in2=b ; iparam=mode    -> out (rgb)
        Invert,      // in0=fac,in1=color                    -> out (rgb)
        Gamma,       // in0=color ; const: gamma             -> out (rgb)
        Math,        // in0=a,in1=b ; iparam=op              -> out (broadcast)
        Swizzle,     // in0=color ; iparam=channel(0/1/2)    -> out (broadcast of that channel)
        Combine,     // in0=r,in1=g,in2=b (each read scalar) -> out (rgb)
        Store,       // in0=src ; aux=MatSlot                -> writes an output slot
        GeoPosition, // -> out = shading-point position (for 3D/solid procedural)
        GeoNormal,   // -> out = shading-point normal
        GeoPointiness, // -> out = shading-point pointiness
        Fresnel,     // in0=ior, in1=normal -> out (float reflectance)
        MatMapping,  // in0=uv ; const: sx,sy,ox,oy,rotDeg,tileU,tileV -> out (uv)
                     //   The MATERIAL PANEL's UV transform, not the Mapping node's.
                     //   The two are genuinely different chains: the node scales about
                     //   the UV origin (Blender semantics), the material scales about
                     //   the UV CENTER, then rotates, then tiles, in a V-flipped space.
                     //   A Material Ref's own texture has to reproduce the panel chain
                     //   byte-for-byte or the texture a Mix Material blends in would
                     //   wrap differently than the same texture bound directly.
                     //   See PrincipledBSDF::applyTextureTransform — this op IS that
                     //   function, minus the wrap mode (Repeat is what both the CPU
                     //   bilinear fetch and the Vulkan sampler already do).
        CurveLUT,    // in0=fac ; aux=N ; const: N uniformly-spaced y samples -> out (y,y,y)
                     //   Float Curve / RGB Curves bake to this. The curve's shape
                     //   (linear, constant, monotone-cubic) is resolved at COMPILE time,
                     //   so the VM never learns about splines and CPU/GPU cannot drift:
                     //   both do the same single lerp between two table entries.
        ObjLocation, // -> out = the hit object's world-space origin
        ObjRandom,   // -> out = stable per-object random in [0,1), broadcast
                     //   Both read the object origin the runtime hands the VM: on GPU
                     //   gl_ObjectToWorldEXT[3].xyz, on CPU HitRecord::object_origin.
                     //   See objectRandom01 (MaterialProceduralMath.h) for why the RANDOM
                     //   hashes the origin instead of an instance id.
        Attribute,   // aux = attribute SLOT (not a name) -> out = value, broadcast
                     //   A named per-vertex float channel (sculpt mask, Geo-DAG mask,
                     //   paint layer, vertex group) barycentric-blended at the hit. The
                     //   NAME is resolved to a slot at COMPILE time (materialAttributeSlot
                     //   below) because a shader cannot look up strings; the runtime only
                     //   ever sees a small integer.
        Wave,        // in0=vector ; aux=dims(2|3)
                     //   const: type, direction, profile, scale, distortion, detail, detailScale, phase
                     //   -> out (fac, broadcast).  Wood grain / marble banding.
        Gradient,    // in0=vector ; iparam=type -> out (fac, broadcast)
        VectorMath,  // in0=a, in1=b, in2=c ; iparam=op ; const: scale -> out (vec3, or a
                     //   broadcast scalar for Dot/Length/Distance)
        HSV,         // in0=color, in1=(h,s,v), in2=fac -> out (rgb)
        GeoPositionObj, // -> out = shading point in OBJECT space (the same space the BLAS
                       //   vertices live in). Free on both sides: the GPU already
                       //   interpolates its object-space verts, the CPU its P_orig.
                       //   Lets a 3D noise stick to the object instead of swimming
                       //   through it as the object moves.
        GeoIncoming, // -> out = unit vector from the shading point TOWARD the viewer
                     //   (-rayDir). Fresnel / Layer Weight are meaningless without it:
                     //   before this op existed both ops used the world normal's Z as
                     //   N.V, which measures "does the normal point along world +Z",
                     //   not the viewing angle. Falls back to the normal (N.V = 1,
                     //   head-on) when the caller has no ray — the emission path.
        AmbientOcclusion, // in0 = distance ; const: [samples, inside] -> out = AO fac, broadcast
                     //   The ONLY op that traces rays. It does NOT trace inside the VM:
                     //   the interpreter calls out to the host (GPU: mp_traceAO in
                     //   closesthit.rchit, the same shadow payload/miss the NEE shadow
                     //   ray already uses; CPU: MatAOContext below). Where tracing is
                     //   illegal or unavailable (shadow any-hit, ray-gen, no BVH bound)
                     //   the op returns 1.0 = unoccluded, which is also what a surface
                     //   in open space evaluates to.
                     //
                     //   HONEST LIMIT: this is the one op that is not bit-identical
                     //   across backends. It is a STOCHASTIC estimate and the CPU and
                     //   the GPU draw their hemisphere samples from different RNGs, so
                     //   at low sample counts the two differ per pixel; they converge to
                     //   the same mean. Every other op is exact and matches.
        Bevel,       // in0 = radius ; const: [samples] -> out = WORLD-space rounded-edge
                     //   normal. The second (and last) tracing op: probes the neighborhood
                     //   with short rays (host hook, like AO — GPU mp_traceBevel via the
                     //   shadow any-hit's probe mode, CPU MatAOContext::probe) and blends
                     //   nearby surface normals in, so a hard edge SHADES as if filleted
                     //   while the silhouette stays sharp. No hook -> the plain shading
                     //   normal (identity: the node quietly does nothing where it can't
                     //   trace — editor preview, fold, OptiX). Same stochastic caveat as AO.
        StoreWorldNormal, // in0 = src -> writes the Normal slot AND marks it WORLD-space.
                     //   Bump stores a TANGENT-space normal that the consumers push through
                     //   the mesh TBN; a Bevel normal is already world-space and shoving it
                     //   through a TBN would twist it by the UV layout. The flag rides in
                     //   MatProgramOutputs::normalIsWorld, and both consumers (closesthit,
                     //   apply_normal_map) branch on it.
        VolumeDensity,     // current volume sample density (broadcast)
        VolumeTemperature, // current volume sample temperature (broadcast)
        VolumeFlame,       // current volume sample flame (broadcast)
        VolumeFuel,        // current volume sample fuel (broadcast)
        VolumeVelocity,    // current volume sample velocity
        VolumePosition,    // current volume sample position
        Blackbody,         // in0 = Kelvin -> physically plausible RGB
        VolumeColor,
        VolumeEmission,
        VolumeVoxelSize,
        VolumeObjectPosition,
        Time,               // deterministic timeline seconds (broadcast)
        CloudShape          // in0=position,in1=time,in2=wind; const cloud controls
    };

    // ---- Ambient Occlusion: the host's occlusion hook (CPU side) ----------------
    // MaterialProgram.h stays free of scene types on purpose (it is included by the
    // material/UI/compiler layers as well), so the AO op reaches the BVH through a POD
    // callback the renderer installs instead of a Hittable*.
    //
    // thread_local because each render thread traces its own rays; a null context (viewport
    // preview, editor fold, any consumer that never installed one) means AO = 1.0 rather
    // than a crash or a silent black surface.
    struct MatAOContext {
        /// true when the segment origin + dir * [eps, tmax] is blocked.
        bool (*occluded)(const void* user, const float o[3], const float d[3], float tmax) = nullptr;
        /// Bevel probe: CLOSEST hit in [tmin, tmax], writing that hit's STORED (outward)
        /// normal — NOT face-forwarded (the Bevel op's chords cross closed geometry from
        /// inside as often as outside; flipping against the ray would tilt the average
        /// the wrong way) — and its distance; returns false on a miss. The Bevel op calls
        /// this repeatedly with an advancing tmin to enumerate EVERY surface crossing
        /// along a probe chord, so no filtering happens here: the op's area estimator
        /// needs all of them. Appended for the Bevel op — an occlusion boolean cannot
        /// answer "whose normal is next to me".
        bool (*probe)(const void* user, const float o[3], const float d[3],
                      float tmin, float tmax,
                      float outNormal[3], float* outT) = nullptr;
        const void* user = nullptr;
        uint32_t    seed = 0u;   ///< per-sample decorrelation; averages out over accumulation
    };
    inline thread_local const MatAOContext* g_matAOContext = nullptr;

    // ---- Named per-vertex attributes (Attribute node) --------------------------
    // A material graph reads a per-vertex float channel BY NAME, but the GPU can only be
    // handed numbers, so names are interned into a small fixed set of slots and the mesh
    // uploads one interleaved block of `kMatAttribSlots` floats per vertex
    // (attribs[vertexId * kMatAttribSlots + slot]). Interleaving is what keeps the GPU
    // side to a SINGLE new address in VkGeometryData instead of one per slot — the stride
    // is a compile-time constant shared with the shader, so no vertex count is needed.
    static constexpr int kMatAttribSlots = 4;

    /// The interned slot table (index = slot, value = attribute name). Scene-wide and
    /// APPEND-ONLY within a compile generation: a slot index is baked into compiled
    /// instruction streams and into the per-vertex blocks meshes upload, so re-ordering it
    /// under a live program would silently swap one mask for another.
    inline std::vector<std::string>& materialAttributeSlots() {
        static std::vector<std::string> slots;
        return slots;
    }

    /// Look a name up WITHOUT interning it. This is what the editor-side evaluate/fold and
    /// the UI warnings use: they run every frame, so if they interned, one typo in the name
    /// field would permanently burn a slot out of the budget of 4.
    inline int findMaterialAttributeSlot(const std::string& name) {
        const auto& slots = materialAttributeSlots();
        for (size_t i = 0; i < slots.size(); ++i) {
            if (slots[i] == name) return static_cast<int>(i);
        }
        return -1;
    }

    /// Intern a name -> slot. Returns -1 when the name is empty or all slots are taken
    /// (the compiler then drops the chain to the Faz-1 fold rather than reading a wrong
    /// channel). ONLY the compiler may call this — see findMaterialAttributeSlot.
    inline int materialAttributeSlot(const std::string& name) {
        if (name.empty()) return -1;
        auto& slots = materialAttributeSlots();
        for (size_t i = 0; i < slots.size(); ++i) {
            if (slots[i] == name) return static_cast<int>(i);
        }
        if (static_cast<int>(slots.size()) >= kMatAttribSlots) return -1;
        slots.push_back(name);
        return static_cast<int>(slots.size() - 1);
    }

    /// Drop every interning. ONLY legal from a WHOLE-SCENE recompile (project load), where
    /// every program is rebuilt right after and every mesh block is rebuilt after that — a
    /// single-material applyGraph must never call this or the OTHER materials' already
    /// compiled slot indices would now point at different names.
    inline void resetMaterialAttributeSlots() { materialAttributeSlots().clear(); }

    struct MatInstr {
        uint16_t op = 0;
        int16_t  outReg = -1;
        int16_t  inReg[3] = { -1, -1, -1 };
        int32_t  constOff = -1;   // offset into MaterialProgram::constPool
        int32_t  aux = 0;         // texIndex / stopCount / slot id
        int32_t  iparam = 0;      // mode / op / kind color-flag / interp
    };

    struct MaterialProgram {
        std::vector<MatInstr> instrs;
        std::vector<float>    constPool;
        std::vector<std::shared_ptr<Texture>> textures;
        int      regCount = 0;
        bool     active = false;      // false => scatter uses the Faz-1 fast path
        uint32_t drivenSlots = 0;     // bitmask over MatSlot
        // Set when the program reads the Geometry node's Pointiness output. Gates the
        // per-vertex pointiness precompute (a full mesh weld + 1-ring pass, see
        // MeshPointiness.h) so a scene that never uses it pays nothing.
        bool     usesPointiness = false;
        // Set when the program reads an Attribute node. Gates the per-vertex attribute
        // block the same way: no graph reads one -> no mesh cache, no GPU block, and
        // sampleMaterialAttributes() collapses to a null check in the hot path.
        bool     usesAttributes = false;
        // Set when the program contains an Ambient Occlusion node. Nothing gates on it in
        // the VM (the op checks its own hook), but it makes the cost VISIBLE: this is the
        // only node that multiplies the ray count per shading call, so the UI warns on it.
        bool     usesAO = false;
        // Same story for the Bevel node (the other tracing op).
        bool     usesBevel = false;
        // Drives the deterministic backend timeline only when a compiled graph
        // actually reads Time (Cloud Shape's implicit default counts).
        bool     usesTime = false;
    };

    // Interpreter register cap (stack-allocated per shading call — no heap, no
    // shared state). The compiler marks a program inactive if it would exceed
    // this, so huge graphs fall back to the Faz-1 fold instead of misbehaving.
    // regCount is PEAK LIVE registers (the compiler compacts live ranges after
    // emission), so 32 means an expression ~32 deep, not a 32-node graph — in
    // practice unreachable. Kept small because the GPU VM mirrors it with a
    // dynamically indexed local array (per-thread scratch, an occupancy tax on
    // every closesthit): MP_MAX_REGS in material_program.glsl MUST match.
    static constexpr int kMatMaxRegs = 32;

    struct MatProgramOutputs {
        float baseColor[3] = { 0.8f, 0.8f, 0.8f };
        float metallic = 0.0f, roughness = 0.5f, specular = 0.5f, transmission = 0.0f;
        float emissionColor[3] = { 0.0f, 0.0f, 0.0f };
        float emissionStrength = 0.0f, opacity = 1.0f, ior = 1.45f;
        float normal[3] = { 0.0f, 0.0f, 1.0f };   // tangent-space (Bump); z-up flat default
        // Set by StoreWorldNormal (Bevel): `normal` is a WORLD-space normal, not a
        // tangent-space one — consumers must NOT push it through the mesh TBN.
        bool normalIsWorld = false;
        uint32_t written = 0;
        bool has(MatSlot s) const { return (written >> static_cast<uint32_t>(s)) & 1u; }
    };

    namespace detail {
        struct MatVal { float x, y, z; };
        inline float rscalar(const MatVal& v) { return (v.x + v.y + v.z) * (1.0f / 3.0f); }
        inline MatVal splat(float f) { return { f, f, f }; }
        inline float clamp01(float f) { return f < 0.0f ? 0.0f : (f > 1.0f ? 1.0f : f); }

        /// Cosine between two register vectors, normalizing BOTH: a Normal socket can be fed
        /// any vector the user wires into it, and an unnormalized one would push N.V past 1
        /// and break the Fresnel/Layer-Weight curve. Degenerate input -> 1 (head-on).
        inline float cosBetween(const MatVal& a, const MatVal& b) {
            const float la = std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
            const float lb = std::sqrt(b.x * b.x + b.y * b.y + b.z * b.z);
            if (la < 1e-8f || lb < 1e-8f) return 1.0f;
            const float d = (a.x * b.x + a.y * b.y + a.z * b.z) / (la * lb);
            return d < -1.0f ? -1.0f : (d > 1.0f ? 1.0f : d);
        }
    }

    /// True if the program samples any texture — those are NOT sent to the GPU
    /// (the GLSL interpreter has no bindless texture-index table yet), so the
    /// flattener leaves such a material's proc_offset = NONE and it keeps the
    /// folded average on GPU while the CPU still shades it per pixel.
    inline bool programUsesTexture(const MaterialProgram& prog) {
        for (const MatInstr& in : prog.instrs) {
            const MatOp op = static_cast<MatOp>(in.op);
            if (op == MatOp::TexColor || op == MatOp::TexAlpha) return true;
        }
        return false;
    }

    /// Flatten a per-material-index program list into the single uint buffer the
    /// Vulkan interpreter (shaders/material_program.glsl) reads at binding 23.
    /// Layout (all uint words; floats stored bit-reinterpreted):
    ///   [0]          materialCount M
    ///   [1..M]       procOffset[i]  (0xFFFFFFFF = no program)
    ///   [1+M]        constBaseWord
    ///   [2+M..]      instruction stream (8 words/instr; End sentinel op=0xFFFF)
    ///   [constBaseWord..] const pool floats
    /// perMaterial[i] may be null (no program). Texture-using programs are skipped.
    inline std::vector<uint32_t> flattenMaterialPrograms(
        const std::vector<const MaterialProgram*>& perMaterial,
        std::function<uint32_t(Texture*)> getTexIDFn = nullptr) {
        auto ibits = [](int v) -> uint32_t { return static_cast<uint32_t>(v); };
        auto fbits = [](float f) -> uint32_t { uint32_t u; std::memcpy(&u, &f, 4); return u; };

        const uint32_t M = static_cast<uint32_t>(perMaterial.size());
        std::vector<uint32_t> words;
        words.push_back(M);
        words.resize(1u + M, 0xFFFFFFFFu);   // proc offset table, all NONE
        const size_t constBaseSlot = words.size();
        words.push_back(0u);                  // placeholder for constBaseWord

        std::vector<float> consts;
        for (uint32_t i = 0; i < M; ++i) {
            const MaterialProgram* p = perMaterial[i];
            if (!p || !p->active || (!getTexIDFn && programUsesTexture(*p))) continue;
            words[1u + i] = static_cast<uint32_t>(words.size());  // program start word
            const int constBaseForProg = static_cast<int>(consts.size());
            for (const MatInstr& in : p->instrs) {
                const int cOff = (in.constOff >= 0) ? (in.constOff + constBaseForProg) : -1;
                uint32_t auxWord = static_cast<uint32_t>(in.aux);
                const MatOp opEnum = static_cast<MatOp>(in.op);
                if (getTexIDFn && (opEnum == MatOp::TexColor || opEnum == MatOp::TexAlpha)) {
                    if (in.aux >= 0 && in.aux < static_cast<int>(p->textures.size())) {
                        auxWord = getTexIDFn(p->textures[in.aux].get());
                    } else {
                        auxWord = 0;
                    }
                }
                words.push_back(in.op);
                words.push_back(ibits(in.outReg));
                words.push_back(ibits(in.inReg[0]));
                words.push_back(ibits(in.inReg[1]));
                words.push_back(ibits(in.inReg[2]));
                words.push_back(ibits(cOff));
                words.push_back(auxWord);
                words.push_back(ibits(in.iparam));
            }
            words.push_back(0xFFFFu);          // End sentinel (single word)
            consts.insert(consts.end(), p->constPool.begin(), p->constPool.end());
        }

        words[constBaseSlot] = static_cast<uint32_t>(words.size());  // constBaseWord
        for (float f : consts) words.push_back(fbits(f));
        return words;
    }

    /// Run the compiled program at a shading point. `pos`/`nrm` (3 floats each,
    /// may be null) feed Geometry / 3D-noise ops; UV feeds 2D. `objOrigin` (3 floats,
    /// may be null) is the hit object's world-space origin, feeding Object Info.
    /// `attribs` (kMatAttribSlots floats, may be null) are the named per-vertex channels
    /// already barycentric-blended at the hit, indexed by interned slot. `objPos` (3 floats,
    /// may be null) is the shading point in OBJECT space, for the object-space procedural
    /// toggle; it falls back to the world point so an absent value never swaps spaces
    /// silently. `viewDir` (3 floats, may be null) points from the shading point TOWARD the
    /// viewer (-ray.direction) and feeds Fresnel / Layer Weight; absent, it falls back to
    /// the normal, i.e. a head-on view. Immutable + thread-safe.
    inline MatProgramOutputs evalMaterialProgram(const MaterialProgram& prog, float u, float v,
                                                 const float* pos = nullptr, const float* nrm = nullptr, float pointiness = 0.5f,
                                                 const float* objOrigin = nullptr, const float* attribs = nullptr,
                                                 const float* objPos = nullptr, const float* viewDir = nullptr) {
        using detail::MatVal;
        using detail::rscalar;
        using detail::splat;

        MatProgramOutputs outp;
        if (!prog.active || prog.regCount > kMatMaxRegs) return outp;

        const MatVal gPos = pos ? MatVal{ pos[0], pos[1], pos[2] } : MatVal{ u, v, 0.0f };
        const MatVal gNrm = nrm ? MatVal{ nrm[0], nrm[1], nrm[2] } : MatVal{ 0.0f, 0.0f, 1.0f };
        const MatVal gObj = objOrigin ? MatVal{ objOrigin[0], objOrigin[1], objOrigin[2] } : MatVal{ 0.0f, 0.0f, 0.0f };
        const MatVal gObjP = objPos ? MatVal{ objPos[0], objPos[1], objPos[2] } : gPos;
        // No ray => head-on. Falling back to (0,0,1) instead would make Fresnel report the
        // normal's world-Z tilt as a viewing angle, which is the bug this vector fixes.
        const MatVal gView = viewDir ? MatVal{ viewDir[0], viewDir[1], viewDir[2] } : gNrm;

        MatVal regs[kMatMaxRegs];
        const float* C = prog.constPool.empty() ? nullptr : prog.constPool.data();

        for (const MatInstr& ins : prog.instrs) {
            const MatOp op = static_cast<MatOp>(ins.op);
            MatVal r{ 0.0f, 0.0f, 0.0f };

            switch (op) {
                case MatOp::Const: {
                    const float* c = C + ins.constOff;
                    r = { c[0], c[1], c[2] };
                    break;
                }
                case MatOp::UV: {
                    r = { u, v, 0.0f };
                    break;
                }
                case MatOp::Mapping: {
                    const MatVal& uv = regs[ins.inReg[0]];
                    const float* c = C + ins.constOff;  // sx,sy,ox,oy,rotDeg
                    const float rad = c[4] * 3.14159265f / 180.0f;
                    const float cr = std::cos(rad), sr = std::sin(rad);
                    const float cx = uv.x - 0.5f, cy = uv.y - 0.5f;
                    const float rx = cx * cr - cy * sr;
                    const float ry = cx * sr + cy * cr;
                    r = { (rx + 0.5f) * c[0] + c[2], (ry + 0.5f) * c[1] + c[3], 0.0f };
                    break;
                }
                case MatOp::TexColor: {
                    const MatVal& uv = regs[ins.inReg[0]];
                    if (ins.aux < 0 || ins.aux >= static_cast<int>(prog.textures.size())) {
                        r = { 0.8f, 0.0f, 0.8f };
                        break;
                    }
                    const auto& tex = prog.textures[ins.aux];
                    if (tex) {
                        const Vec3 c = tex->get_color_bilinear(uv.x, uv.y);
                        r = { static_cast<float>(c.x), static_cast<float>(c.y), static_cast<float>(c.z) };
                    } else {
                        r = { 0.8f, 0.0f, 0.8f };  // magenta = missing texture
                    }
                    break;
                }
                case MatOp::TexAlpha: {
                    const MatVal& uv = regs[ins.inReg[0]];
                    if (ins.aux < 0 || ins.aux >= static_cast<int>(prog.textures.size())) {
                        r = splat(1.0f);
                        break;
                    }
                    const auto& tex = prog.textures[ins.aux];
                    float a = 1.0f;
                    if (tex && tex->has_alpha) a = tex->get_alpha_bilinear(uv.x, uv.y);
                    r = splat(a);
                    break;
                }
                case MatOp::Noise: {
                    const MatVal& uv = regs[ins.inReg[0]];
                    const float* c = C + ins.constOff;  // kind,scale,detail,rough,rand,dist,seed
                    const int   kind    = static_cast<int>(c[0]);
                    const float scale   = c[1];
                    const int   detail  = static_cast<int>(c[2]);
                    const float rough   = c[3];
                    const float rand    = c[4];
                    const float dist    = c[5];
                    const int   seed    = static_cast<int>(c[6]);
                    const uint32_t s = (kind == 4)  // 4 = Voronoi
                        ? static_cast<uint32_t>(seed) * 0x68bc21ebu + 0x2545F491u
                        : static_cast<uint32_t>(seed) * 0x51633e2du + 0x9E3779B9u;
                    uint32_t cellHash = 0;
                    float fac;
                    if (ins.aux == 3) {   // 3D (position-driven, seamless)
                        const float x = uv.x * scale, y = uv.y * scale, z = uv.z * scale;
                        switch (kind) {
                            case 1: fac = ridge3D(x, y, z, detail, rough, s); break;
                            case 2: fac = billow3D(x, y, z, detail, rough, s); break;
                            case 3: {
                                const float wx = fbm3D(x + 17.3f, y + 9.1f, z + 3.7f, detail, rough, s ^ 0xA511u) - 0.5f;
                                const float wy = fbm3D(x - 11.7f, y + 4.9f, z - 8.2f, detail, rough, s ^ 0x3C6Fu) - 0.5f;
                                const float wz = fbm3D(x + 5.1f, y - 6.3f, z + 12.9f, detail, rough, s ^ 0x77A5u) - 0.5f;
                                fac = fbm3D(x + wx * dist * 8.0f, y + wy * dist * 8.0f, z + wz * dist * 8.0f, detail, rough, s);
                                break;
                            }
                            case 4: fac = detail::clamp01(voronoi3D_F1(x, y, z, rand, s, &cellHash)); break;
                            case 5: {
                                const int cx = static_cast<int>(std::floor(x));
                                const int cy = static_cast<int>(std::floor(y));
                                const int cz = static_cast<int>(std::floor(z));
                                fac = (((cx + cy + cz) & 1) != 0) ? 1.0f : 0.0f;
                                break;
                            }
                            default: fac = fbm3D(x, y, z, detail, rough, s); break;
                        }
                    } else {              // 2D (UV-driven)
                        const float x = uv.x * scale, y = uv.y * scale;
                        switch (kind) {
                            case 1: fac = ridge2D(x, y, detail, rough, s); break;   // Ridge
                            case 2: fac = billow2D(x, y, detail, rough, s); break;  // Billow
                            case 3: {                                               // Warped
                                const float wx = fbm2D(x + 17.3f, y + 9.1f, detail, rough, s ^ 0xA511u) - 0.5f;
                                const float wy = fbm2D(x - 11.7f, y + 4.9f, detail, rough, s ^ 0x3C6Fu) - 0.5f;
                                fac = fbm2D(x + wx * dist * 8.0f, y + wy * dist * 8.0f, detail, rough, s);
                                break;
                            }
                            case 4: fac = detail::clamp01(voronoiF1(x, y, rand, s, &cellHash)); break;  // Voronoi
                            case 5: {                                               // Checker
                                const int cx = static_cast<int>(std::floor(x));
                                const int cy = static_cast<int>(std::floor(y));
                                fac = (((cx + cy) & 1) != 0) ? 1.0f : 0.0f;
                                break;
                            }
                            default: fac = fbm2D(x, y, detail, rough, s); break;    // FBM
                        }
                    }
                    if (ins.iparam == 1) {          // Voronoi Color = per-cell random
                        r = { static_cast<float>(pcgHash(cellHash ^ 0x11u)) * (1.0f / 4294967295.0f),
                              static_cast<float>(pcgHash(cellHash ^ 0x22u)) * (1.0f / 4294967295.0f),
                              static_cast<float>(pcgHash(cellHash ^ 0x33u)) * (1.0f / 4294967295.0f) };
                    } else if (ins.iparam == 2) {   // Color = lerp(Color1, Color2, fac) (non-Voronoi kinds)
                        const MatVal& a = regs[ins.inReg[1]];
                        const MatVal& b = regs[ins.inReg[2]];
                        r = { a.x + (b.x - a.x) * fac, a.y + (b.y - a.y) * fac, a.z + (b.z - a.z) * fac };
                    } else {                        // Fac / grayscale color
                        r = splat(fac);
                    }
                    break;
                }
                case MatOp::ColorRamp: {
                    const float fac = detail::clamp01(rscalar(regs[ins.inReg[0]]));
                    const int n = ins.aux;                 // stop count
                    const float* st = C + ins.constOff;    // (pos,r,g,b)* pre-sorted by pos
                    if (n <= 0) { r = { 0, 0, 0 }; break; }
                    if (fac <= st[0]) { r = { st[1], st[2], st[3] }; break; }
                    if (fac >= st[(n - 1) * 4]) { r = { st[(n - 1) * 4 + 1], st[(n - 1) * 4 + 2], st[(n - 1) * 4 + 3] }; break; }
                    bool done = false;
                    for (int i = 0; i + 1 < n; ++i) {
                        const float p0 = st[i * 4], p1 = st[(i + 1) * 4];
                        if (fac <= p1) {
                            if (ins.iparam == 1) {  // Constant
                                r = { st[i * 4 + 1], st[i * 4 + 2], st[i * 4 + 3] };
                            } else {
                                const float span = p1 - p0;
                                const float t = (span > 1e-6f) ? (fac - p0) / span : 0.0f;
                                r = { st[i * 4 + 1] + (st[(i + 1) * 4 + 1] - st[i * 4 + 1]) * t,
                                      st[i * 4 + 2] + (st[(i + 1) * 4 + 2] - st[i * 4 + 2]) * t,
                                      st[i * 4 + 3] + (st[(i + 1) * 4 + 3] - st[i * 4 + 3]) * t };
                            }
                            done = true;
                            break;
                        }
                    }
                    if (!done) r = { st[(n - 1) * 4 + 1], st[(n - 1) * 4 + 2], st[(n - 1) * 4 + 3] };
                    break;
                }
                case MatOp::MixColor: {
                    const float fac = detail::clamp01(rscalar(regs[ins.inReg[0]]));
                    const MatVal& a = regs[ins.inReg[1]];
                    const MatVal& b = regs[ins.inReg[2]];
                    const float av[3] = { a.x, a.y, a.z }, bv[3] = { b.x, b.y, b.z };
                    float rv[3];
                    for (int i = 0; i < 3; ++i) {
                        float m;
                        switch (ins.iparam) {
                            case 1: m = av[i] + bv[i]; break;                                  // Add
                            case 2: m = av[i] * bv[i]; break;                                  // Multiply
                            case 3: m = av[i] - bv[i]; break;                                  // Subtract
                            case 4: m = 1.0f - (1.0f - av[i]) * (1.0f - bv[i]); break;         // Screen
                            case 5: m = (av[i] < 0.5f) ? 2.0f * av[i] * bv[i]                  // Overlay
                                                       : 1.0f - 2.0f * (1.0f - av[i]) * (1.0f - bv[i]); break;
                            default: m = bv[i]; break;                                         // Mix
                        }
                        rv[i] = av[i] + (m - av[i]) * fac;
                    }
                    r = { rv[0], rv[1], rv[2] };
                    break;
                }
                case MatOp::Invert: {
                    const float fac = detail::clamp01(rscalar(regs[ins.inReg[0]]));
                    const MatVal& c = regs[ins.inReg[1]];
                    r = { c.x + ((1.0f - c.x) - c.x) * fac,
                          c.y + ((1.0f - c.y) - c.y) * fac,
                          c.z + ((1.0f - c.z) - c.z) * fac };
                    break;
                }
                case MatOp::Gamma: {
                    const MatVal& c = regs[ins.inReg[0]];
                    const float inv = 1.0f / std::max(C[ins.constOff], 0.01f);
                    r = { std::pow(std::max(c.x, 0.0f), inv),
                          std::pow(std::max(c.y, 0.0f), inv),
                          std::pow(std::max(c.z, 0.0f), inv) };
                    break;
                }
                case MatOp::Math: {
                    const float a = rscalar(regs[ins.inReg[0]]);
                    const float b = rscalar(regs[ins.inReg[1]]);
                    float m;
                    switch (ins.iparam) {
                        case 1: m = a - b; break;
                        case 2: m = a * b; break;
                        case 3: m = (std::fabs(b) > 1e-8f) ? a / b : 0.0f; break;
                        case 4: m = std::pow(std::max(a, 0.0f), b); break;
                        case 5: m = std::sqrt(std::max(a, 0.0f)); break;
                        case 6: m = std::fabs(a); break;
                        case 7: m = std::min(a, b); break;
                        case 8: m = std::max(a, b); break;
                        case 9: m = detail::clamp01(a); break;
                        // Appended (MathNode::Op keeps its serialized ids, so new ops
                        // may only be added at the END). These are what Wave/Gradient
                        // style patterns are built out of.
                        case 10: m = std::sin(a); break;
                        case 11: m = std::cos(a); break;
                        case 12: m = a - std::floor(a); break;          // Fraction
                        case 13: m = std::floor(a); break;
                        case 14: m = std::ceil(a); break;
                        case 15: m = (std::fabs(b) > 1e-8f) ? a - b * std::floor(a / b) : 0.0f;  // Modulo (floored, matches GLSL mod)
                                 break;
                        case 16: { const float t = detail::clamp01(a); m = t * t * (3.0f - 2.0f * t); } break;  // Smooth Step
                        case 17: m = (a > b) ? 1.0f : 0.0f; break;      // Greater Than
                        case 18: m = (a < b) ? 1.0f : 0.0f; break;      // Less Than
                        default: m = a + b; break;
                    }
                    r = splat(m);
                    break;
                }
                case MatOp::MatMapping: {
                    // PrincipledBSDF::applyTextureTransform, verbatim (see MatOp docs).
                    const MatVal& t = regs[ins.inReg[0]];
                    const float* c = C + ins.constOff;   // sx,sy,ox,oy,rotDeg,tileU,tileV
                    float u = t.x;
                    float v = 1.0f - t.y;               // into the GPU-facing UV space
                    u -= 0.5f; v -= 0.5f;
                    u *= c[0];  v *= c[1];
                    const float rad = c[4] * 3.14159265f / 180.0f;
                    const float cr = std::cos(rad), sr = std::sin(rad);
                    const float nu = u * cr - v * sr;
                    const float nv = u * sr + v * cr;
                    u = nu + 0.5f + c[2];
                    v = nv + 0.5f + c[3];
                    u *= c[5];  v *= c[6];
                    r = { u, 1.0f - v, 0.0f };         // back to CPU sampler space
                    break;
                }
                case MatOp::CurveLUT: {
                    const float fac = detail::clamp01(rscalar(regs[ins.inReg[0]]));
                    const int   n   = ins.aux;
                    const float* c  = C + ins.constOff;
                    float y;
                    if (n <= 1) {
                        y = (n == 1) ? c[0] : 0.0f;
                    } else {
                        const float s  = fac * static_cast<float>(n - 1);
                        const int   i0 = std::min(static_cast<int>(s), n - 2);
                        const float f  = s - static_cast<float>(i0);
                        y = c[i0] + (c[i0 + 1] - c[i0]) * f;
                    }
                    r = splat(y);
                    break;
                }
                case MatOp::Swizzle: {
                    const MatVal& c = regs[ins.inReg[0]];
                    const float ch = (ins.iparam == 1) ? c.y : (ins.iparam == 2) ? c.z : c.x;
                    r = splat(ch);
                    break;
                }
                case MatOp::Combine: {
                    r = { rscalar(regs[ins.inReg[0]]), rscalar(regs[ins.inReg[1]]), rscalar(regs[ins.inReg[2]]) };
                    break;
                }
                case MatOp::GeoPosition: { r = gPos; break; }
                case MatOp::GeoNormal:   { r = gNrm; break; }
                case MatOp::GeoPointiness: { r = splat(pointiness); break; }
                case MatOp::ObjLocation: { r = gObj; break; }
                case MatOp::ObjRandom:   { r = splat(objectRandom01(gObj.x, gObj.y, gObj.z)); break; }
                case MatOp::Attribute: {
                    // Absent block (mesh has no such channel, or nothing is uploaded) reads 0 —
                    // an unpainted mask, which is the same thing the GPU's null-address path
                    // returns. A mask that silently read 1 would paint the whole object.
                    const float a = (attribs && ins.aux >= 0 && ins.aux < kMatAttribSlots)
                                        ? attribs[ins.aux] : 0.0f;
                    r = splat(a);
                    break;
                }
                case MatOp::GeoPositionObj: { r = gObjP; break; }
                case MatOp::GeoIncoming:    { r = gView; break; }

                case MatOp::AmbientOcclusion: {
                    // Cosine-weighted hemisphere around the shading normal, N shadow rays.
                    // Unhooked (no BVH installed on this thread: editor fold, viewport
                    // preview, unit test) -> 1.0. Same value an unoccluded surface gets, so
                    // an AO chain degrades to "no dirt" rather than to a black object.
                    const MatAOContext* ao = g_matAOContext;
                    if (!ao || !ao->occluded) { r = splat(1.0f); break; }

                    const float dist = std::max(1e-4f, regs[ins.inReg[0]].x);
                    const float* c = C + ins.constOff;                     // samples, inside
                    const int samples = std::max(1, std::min(64, static_cast<int>(c[0])));
                    const bool inside = (c[1] != 0.0f);

                    float nx = gNrm.x, ny = gNrm.y, nz = gNrm.z;
                    const float nl = std::sqrt(nx * nx + ny * ny + nz * nz);
                    if (nl < 1e-8f) { r = splat(1.0f); break; }
                    nx /= nl; ny /= nl; nz /= nl;
                    if (inside) { nx = -nx; ny = -ny; nz = -nz; }          // occlusion of the cavity BEHIND the surface

                    // Duff et al. branchless ONB — the naive "cross with up unless nearly
                    // parallel" basis flips handedness across the pole and would put a seam
                    // straight through the AO of any sphere.
                    const float sg = std::copysign(1.0f, nz);
                    const float a = -1.0f / (sg + nz);
                    const float b = nx * ny * a;
                    const float t0x = 1.0f + sg * nx * nx * a, t0y = sg * b, t0z = -sg * nx;
                    const float t1x = b, t1y = sg + ny * ny * a, t1z = -ny;

                    // Seed from the SHADING POINT (not the pixel): stable under camera
                    // motion, so AO does not crawl over a static surface. ctx->seed adds the
                    // per-sample decorrelation that lets the estimate average out.
                    uint32_t seed = pcgHash(
                        pcgHash(floatBitsToU32(gPos.x) ^ 0x9E3779B9u) ^
                        pcgHash(floatBitsToU32(gPos.y) ^ 0x85EBCA6Bu) ^
                        pcgHash(floatBitsToU32(gPos.z) ^ 0xC2B2AE35u) ^ ao->seed);

                    const float origin[3] = { gPos.x + nx * 1e-4f, gPos.y + ny * 1e-4f, gPos.z + nz * 1e-4f };
                    int hits = 0;
                    for (int s = 0; s < samples; ++s) {
                        seed = pcgHash(seed);
                        const float r1 = static_cast<float>(seed) * (1.0f / 4294967296.0f);
                        seed = pcgHash(seed);
                        const float r2 = static_cast<float>(seed) * (1.0f / 4294967296.0f);

                        const float phi = 6.2831853f * r1;
                        const float sq = std::sqrt(r2);                    // cosine-weighted
                        const float dx = std::cos(phi) * sq;
                        const float dy = std::sin(phi) * sq;
                        const float dz = std::sqrt(std::max(0.0f, 1.0f - r2));

                        const float dir[3] = {
                            t0x * dx + t1x * dy + nx * dz,
                            t0y * dx + t1y * dy + ny * dz,
                            t0z * dx + t1z * dy + nz * dz
                        };
                        if (ao->occluded(ao->user, origin, dir, dist)) ++hits;
                    }
                    r = splat(1.0f - static_cast<float>(hits) / static_cast<float>(samples));
                    break;
                }

                case MatOp::Bevel: {
                    // Rounded-edge shading normal = the AREA-AVERAGE of the surface normal
                    // over the part of the scene inside a sphere of `radius` around the
                    // shading point, distance-weighted. Estimated with `samples` random
                    // CHORDS through that sphere (uniform direction + uniform disk offset
                    // perpendicular to it), accumulating EVERY surface crossing along each
                    // chord. Rays cast FROM the shading point were shipped first and are
                    // structurally unable to be continuous across an edge: they see the
                    // face they stand on in half of all directions but the edge's neighbor
                    // face in only a quarter, so the blend over-rotates past the mid normal
                    // on BOTH sides and jumps at the exact crest line. The area estimator
                    // is continuous in the shading point and lands on the mid normal at the
                    // crest by symmetry. Far from any edge every crossing is the surface
                    // itself (normal == N) -> exactly N. Unhooked -> identity too.
                    // MIRROR: mp_traceBevel in closesthit.rchit — same estimator, only the
                    // RNG differs (stochastic; the two converge in the mean, like AO).
                    const MatAOContext* ao = g_matAOContext;
                    float nx = gNrm.x, ny = gNrm.y, nz = gNrm.z;
                    const float nl = std::sqrt(nx * nx + ny * ny + nz * nz);
                    if (nl < 1e-8f) { r = { 0.0f, 0.0f, 1.0f }; break; }
                    nx /= nl; ny /= nl; nz /= nl;
                    if (!ao || !ao->probe) { r = { nx, ny, nz }; break; }

                    const float radius = std::max(1e-5f, regs[ins.inReg[0]].x);
                    const float* c = C + ins.constOff;                    // samples
                    const int samples = std::max(1, std::min(16, static_cast<int>(c[0])));

                    // Same shading-point seeding as AO (camera-stable), different salt so a
                    // graph using both doesn't correlate the two estimates.
                    uint32_t seed = pcgHash(
                        pcgHash(floatBitsToU32(gPos.x) ^ 0xB5297A4Du) ^
                        pcgHash(floatBitsToU32(gPos.y) ^ 0x68E31DA4u) ^
                        pcgHash(floatBitsToU32(gPos.z) ^ 0x1B56C4E9u) ^ ao->seed);

                    // Tiny N seed only breaks the tie when every chord misses (a needle
                    // tip); anywhere normal it is noise-floor against the real weights.
                    float ax = nx * 0.05f, ay = ny * 0.05f, az = nz * 0.05f;
                    for (int s = 0; s < samples; ++s) {
                        seed = pcgHash(seed);
                        const float r1 = static_cast<float>(seed) * (1.0f / 4294967296.0f);
                        seed = pcgHash(seed);
                        const float r2 = static_cast<float>(seed) * (1.0f / 4294967296.0f);
                        const float z = 1.0f - 2.0f * r1;                  // chord axis: uniform sphere
                        const float phi = 6.2831853f * r2;
                        const float sxy = std::sqrt(std::max(0.0f, 1.0f - z * z));
                        const float dx = std::cos(phi) * sxy, dy = std::sin(phi) * sxy, dz = z;
                        const float dir[3] = { dx, dy, dz };

                        // Duff branchless ONB around the axis (same trick as the AO op).
                        const float sg = std::copysign(1.0f, dz);
                        const float ba = -1.0f / (sg + dz);
                        const float bb = dx * dy * ba;
                        const float e0x = 1.0f + sg * dx * dx * ba, e0y = sg * bb, e0z = -sg * dx;
                        const float e1x = bb, e1y = sg + dy * dy * ba, e1z = -dy;

                        seed = pcgHash(seed);
                        const float r3 = static_cast<float>(seed) * (1.0f / 4294967296.0f);
                        seed = pcgHash(seed);
                        const float r4 = static_cast<float>(seed) * (1.0f / 4294967296.0f);
                        const float diskR = radius * std::sqrt(r3) * 0.999f;   // uniform disk; keep h > 0
                        const float ph2 = 6.2831853f * r4;
                        const float h = std::sqrt(std::max(radius * radius - diskR * diskR, 0.0f));
                        const float ox = std::cos(ph2) * diskR, oy = std::sin(ph2) * diskR;
                        const float origin[3] = {
                            gPos.x + e0x * ox + e1x * oy - dx * h,
                            gPos.y + e0y * ox + e1y * oy - dy * h,
                            gPos.z + e0z * ox + e1z * oy - dz * h
                        };

                        // March every surface crossing along the chord (closest-from-tmin,
                        // advance, repeat). 8 crossings inside one bevel radius is already
                        // pathological geometry.
                        float tcur = 0.0f;
                        for (int hop = 0; hop < 8; ++hop) {
                            float hn[3]; float ht = 0.0f;
                            if (!ao->probe(ao->user, origin, dir, tcur, 2.0f * h, hn, &ht)) break;
                            const float dt = ht - h;
                            const float distP = std::sqrt(diskR * diskR + dt * dt);
                            const float w = std::max(0.0f, 1.0f - distP / radius);
                            ax += hn[0] * w; ay += hn[1] * w; az += hn[2] * w;
                            tcur = ht + 1e-4f;
                        }
                    }
                    const float al = std::sqrt(ax * ax + ay * ay + az * az);
                    r = (al > 1e-8f) ? MatVal{ ax / al, ay / al, az / al } : MatVal{ nx, ny, nz };
                    break;
                }

                case MatOp::Wave: {
                    const MatVal& t = regs[ins.inReg[0]];
                    const float* c = C + ins.constOff;   // type,dir,profile,scale,distortion,detail,detailScale,phase
                    const int   type    = static_cast<int>(c[0]);   // 0 Bands, 1 Rings
                    const int   dir     = static_cast<int>(c[1]);   // 0 X, 1 Y, 2 Z, 3 Diagonal
                    const int   profile = static_cast<int>(c[2]);   // 0 Sine, 1 Saw, 2 Triangle
                    const float scale   = c[3];
                    const float distort = c[4];
                    const int   detail  = static_cast<int>(c[5]);
                    const float dScale  = c[6];
                    const float phase   = c[7];
                    const float x = t.x * scale, y = t.y * scale, z = t.z * scale;

                    float n;
                    if (type == 1) {   // Rings: distance from the axis/origin
                        float rx = x, ry = y, rz = z;
                        if (dir == 0) rx = 0.0f; else if (dir == 1) ry = 0.0f; else if (dir == 2) rz = 0.0f;
                        n = std::sqrt(rx * rx + ry * ry + rz * rz) * 20.0f;
                    } else {           // Bands
                        const float d = (dir == 0) ? x : (dir == 1) ? y : (dir == 2) ? z : (x + y + z);
                        n = d * 20.0f;
                    }
                    n += phase;
                    if (distort != 0.0f && detail > 0) {
                        // The distortion is what turns clean bands into WOOD — without it this is
                        // just a sine and nobody needs a node for that.
                        const float w = (ins.aux == 3)
                            ? fbm3D(x * dScale, y * dScale, z * dScale, detail, 0.5f, 0x9E37u)
                            : fbm2D(x * dScale, y * dScale, detail, 0.5f, 0x9E37u);
                        n += distort * (w * 2.0f - 1.0f);
                    }

                    float fac;
                    const float TWO_PI = 6.2831853f;
                    if (profile == 1) {          // Saw
                        const float s = n / TWO_PI;
                        fac = s - std::floor(s);
                    } else if (profile == 2) {   // Triangle
                        const float s = n / TWO_PI;
                        fac = std::fabs(2.0f * (s - std::floor(s)) - 1.0f);
                    } else {                     // Sine
                        fac = 0.5f + 0.5f * std::sin(n - 1.5707963f);
                    }
                    r = splat(detail::clamp01(fac));
                    break;
                }

                case MatOp::Gradient: {
                    const MatVal& t = regs[ins.inReg[0]];
                    float fac;
                    switch (ins.iparam) {
                        case 1: { const float g = std::max(t.x, 0.0f); fac = g * g; } break;   // Quadratic
                        case 2: { const float g = detail::clamp01(t.x); fac = g * g * (3.0f - 2.0f * g); } break;  // Easing
                        case 3: fac = (t.x + t.y) * 0.5f; break;                               // Diagonal
                        case 4: fac = std::atan2(t.y, t.x) / 6.2831853f + 0.5f; break;         // Radial
                        case 5: { const float len = std::sqrt(t.x * t.x + t.y * t.y + t.z * t.z);
                                  const float g = std::max(1.0f - len, 0.0f); fac = g * g; } break;  // Quadratic Sphere
                        case 6: { const float len = std::sqrt(t.x * t.x + t.y * t.y + t.z * t.z);
                                  fac = std::max(1.0f - len, 0.0f); } break;                   // Spherical
                        default: fac = t.x; break;                                             // Linear
                    }
                    r = splat(detail::clamp01(fac));
                    break;
                }

                case MatOp::VectorMath: {
                    const MatVal& a = regs[ins.inReg[0]];
                    const MatVal& b = regs[ins.inReg[1]];
                    switch (ins.iparam) {
                        case 1: r = { a.x - b.x, a.y - b.y, a.z - b.z }; break;                 // Subtract
                        case 2: r = { a.x * b.x, a.y * b.y, a.z * b.z }; break;                 // Multiply
                        case 3: r = { (std::fabs(b.x) > 1e-8f) ? a.x / b.x : 0.0f,              // Divide
                                      (std::fabs(b.y) > 1e-8f) ? a.y / b.y : 0.0f,
                                      (std::fabs(b.z) > 1e-8f) ? a.z / b.z : 0.0f }; break;
                        case 4: r = { a.y * b.z - a.z * b.y,                                    // Cross
                                      a.z * b.x - a.x * b.z,
                                      a.x * b.y - a.y * b.x }; break;
                        case 5: r = splat(a.x * b.x + a.y * b.y + a.z * b.z); break;            // Dot
                        case 6: { const float l = std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z); // Normalize
                                  r = (l > 1e-8f) ? MatVal{ a.x / l, a.y / l, a.z / l } : MatVal{ 0.0f, 0.0f, 0.0f }; } break;
                        case 7: r = splat(std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z)); break; // Length
                        case 8: { const float dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;   // Distance
                                  r = splat(std::sqrt(dx * dx + dy * dy + dz * dz)); } break;
                        case 9: { // Reflect a about the normalized b
                            const float l = std::sqrt(b.x * b.x + b.y * b.y + b.z * b.z);
                            if (l > 1e-8f) {
                                const float nx = b.x / l, ny = b.y / l, nz = b.z / l;
                                const float d = 2.0f * (a.x * nx + a.y * ny + a.z * nz);
                                r = { a.x - d * nx, a.y - d * ny, a.z - d * nz };
                            } else r = a;
                        } break;
                        case 10: { const float s = C[ins.constOff];                             // Scale
                                   r = { a.x * s, a.y * s, a.z * s }; } break;
                        case 11: r = { std::fabs(a.x), std::fabs(a.y), std::fabs(a.z) }; break; // Absolute
                        case 12: r = { std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z) }; break;
                        case 13: r = { std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z) }; break;
                        default: r = { a.x + b.x, a.y + b.y, a.z + b.z }; break;                // Add
                    }
                    break;
                }

                case MatOp::HSV: {
                    const MatVal& col = regs[ins.inReg[0]];
                    const MatVal& adj = regs[ins.inReg[1]];   // (hue, saturation, value)
                    const float fac = detail::clamp01(rscalar(regs[ins.inReg[2]]));
                    float h, s, vv;
                    rgbToHsv(col.x, col.y, col.z, h, s, vv);
                    // Blender's convention: Hue 0.5 = no shift, so it maps to a +/- half turn.
                    h += adj.x - 0.5f;
                    h = h - std::floor(h);
                    s = detail::clamp01(s * adj.y);
                    vv *= adj.z;
                    float rr, gg, bb;
                    hsvToRgb(h, s, vv, rr, gg, bb);
                    r = { col.x + (rr - col.x) * fac,
                          col.y + (gg - col.y) * fac,
                          col.z + (bb - col.z) * fac };
                    break;
                }

                case MatOp::Fresnel: {
                    float ior = regs[ins.inReg[0]].x;
                    const MatVal& n = regs[ins.inReg[1]];
                    // The REAL viewing angle. This used to read n.z and call it N.V, on the
                    // assumption that the view is (0,0,1) — true in a 2D preview, false at
                    // every actual hit: it turned Fresnel into "how far does the normal tilt
                    // toward world +Z", so a wall and a floor got different reflectance for
                    // no reason and rotating the camera changed nothing.
                    // fabs, not max(0,...): a two-sided surface seen from behind has a
                    // negative cosine and must still reflect, not go black.
                    float ndotv = std::fabs(detail::cosBetween(n, gView));
                    float r0 = (1.0f - ior) / (1.0f + ior);
                    r0 = r0 * r0;
                    float fc = std::pow(1.0f - ndotv, 5.0f);
                    r.x = r.y = r.z = r0 + (1.0f - r0) * fc;
                    break;
                }
                case MatOp::Store: {
                    const MatVal& src = regs[ins.inReg[0]];
                    switch (static_cast<MatSlot>(ins.aux)) {
                        case MatSlot::BaseColor:        outp.baseColor[0] = src.x; outp.baseColor[1] = src.y; outp.baseColor[2] = src.z; break;
                        case MatSlot::Metallic:         outp.metallic = rscalar(src); break;
                        case MatSlot::Roughness:        outp.roughness = rscalar(src); break;
                        case MatSlot::Specular:         outp.specular = rscalar(src); break;
                        case MatSlot::Transmission:     outp.transmission = rscalar(src); break;
                        case MatSlot::EmissionColor:    outp.emissionColor[0] = src.x; outp.emissionColor[1] = src.y; outp.emissionColor[2] = src.z; break;
                        case MatSlot::EmissionStrength: outp.emissionStrength = rscalar(src); break;
                        case MatSlot::Opacity:          outp.opacity = rscalar(src); break;
                        case MatSlot::IOR:              outp.ior = rscalar(src); break;
                        case MatSlot::Normal:           outp.normal[0] = src.x; outp.normal[1] = src.y; outp.normal[2] = src.z; break;
                        default: break;
                    }
                    outp.written |= (1u << ins.aux);
                    continue;  // Store writes no register
                }
                case MatOp::Blackbody: {
                    const float temp = std::clamp(rscalar(regs[ins.inReg[0]]), 800.0f, 40000.0f) / 100.0f;
                    if (temp <= 66.0f) {
                        r.x = 1.0f;
                        r.y = std::clamp(0.39008158f * std::log(std::max(temp, 1.0f)) - 0.63184144f, 0.0f, 1.0f);
                    } else {
                        r.x = std::clamp(1.29293619f * std::pow(temp - 60.0f, -0.13320476f), 0.0f, 1.0f);
                        r.y = std::clamp(1.12989086f * std::pow(temp - 60.0f, -0.07551485f), 0.0f, 1.0f);
                    }
                    r.z = temp >= 66.0f ? 1.0f :
                        (temp <= 19.0f ? 0.0f :
                         std::clamp(0.54320679f * std::log(temp - 10.0f) - 1.19625409f, 0.0f, 1.0f));
                    break;
                }
                case MatOp::StoreWorldNormal: {
                    const MatVal& src = regs[ins.inReg[0]];
                    outp.normal[0] = src.x; outp.normal[1] = src.y; outp.normal[2] = src.z;
                    outp.normalIsWorld = true;   // consumers skip the TBN — see MatOp comment
                    outp.written |= (1u << static_cast<uint32_t>(MatSlot::Normal));
                    continue;
                }
                default: break;
            }
            if (ins.outReg >= 0 && ins.outReg < prog.regCount) regs[ins.outReg] = r;
        }
        return outp;
    }

} // namespace MaterialNodesV2
