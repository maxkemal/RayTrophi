/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          MaterialNodesV2.h
* Author:        Kemal Demirtas
* Date:          July 2026
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once

/**
 * @file MaterialNodesV2.h
 * @brief Material node graph (Faz 1) on the V2 NodeSystem.
 *
 * Design decision (2026-07-10): NO texture baking, NO per-edit GPU churn.
 * The graph is a FRONTEND over the existing uber-material (PrincipledBSDF +
 * GpuMaterial): Apply evaluates the graph on the CPU and folds the result into
 * the material's existing parameter/texture slots, then the material flows to
 * all three backends through the same PBRMaterialSnapshot / updateBackendMaterial
 * choke point every slider edit already uses. One runtime material representation,
 * zero shader change, identical result on CPU / OptiX / Vulkan by construction.
 *
 * What folds losslessly today:
 *   - constant chains (Value/Color/Math/Mix with constant inputs) -> scalar params
 *   - an Image Texture wired (optionally through Mapping) into a slot -> texture
 *     binding on that slot (+ material-wide TextureTransform from Mapping)
 *   - Material Ref / Mix Material with constant Fac -> full param-set blend
 *
 * What folds LOSSILY (flagged as a warning in the editor): spatially-varying
 * chains (Noise/Voronoi/Checker/Ramp driving a param) are averaged over a few
 * UV samples. Per-pixel evaluation is Faz 2 (SVM-lite instruction stream
 * interpreted in the shaders â€” see docs/material node roadmap); the node
 * compute() functions here are already written per-UV so the same graph
 * semantics carry over unchanged.
 */

#include "NodeSystem/NodeCore.h"
#include "NodeSystem/Node.h"
#include "NodeSystem/Graph.h"
#include "NodeSystem/EvaluationContext.h"
#include "NodeSystem/NodeRegistry.h"
#include "PrincipledBSDF.h"
#include "MaterialManager.h"
#include "MaterialProceduralMath.h"
#include "MaterialProgram.h"
#include "Texture.h"
#include "Vec2.h"
#include "Vec3.h"
#include "json.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

/**
 * @brief Resolve a texture that exists ONLY inside the currently loaded project — an
 * embedded blob with no file on disk. Defined in ProjectManager.cpp.
 *
 * Declared here rather than included: ProjectManager.h pulls in scene_data.h, which pulls
 * in this header, so an include would close a cycle. Returns null when the project has no
 * embedded texture under that name (or nothing is loaded).
 */
std::shared_ptr<Texture> resolveEmbeddedProjectTexture(const std::string& name);

namespace MaterialNodesV2 {

    // ============================================================================
    // SHADE STATE â€” the Material pin payload
    // ============================================================================

    /**
     * @brief Full CPU-side shading parameter set + texture bindings.
     *
     * This is what flows through DataType::Material pins and what Apply writes
     * into the target PrincipledBSDF. Field set mirrors the artist-facing part
     * of the uber-material; texture members are session shared_ptrs (zero-copy).
     */
    struct ShadeState {
        Vec3  baseColor        = Vec3(0.8f, 0.8f, 0.8f);
        float metallic         = 0.0f;
        float roughness        = 0.5f;
        float specular         = 0.5f;
        Vec3  emissionColor    = Vec3(0.0f, 0.0f, 0.0f);
        float emissionStrength = 0.0f;
        float transmission     = 0.0f;
        float ior              = 1.45f;
        float opacity          = 1.0f;
        float translucent      = 0.0f;
        float clearcoat        = 0.0f;
        float clearcoatRoughness = 0.03f;
        float subsurface       = 0.0f;
        Vec3  subsurfaceColor  = Vec3(1.0f, 0.8f, 0.6f);
        float normalStrength   = 1.0f;

        // Extended set (panel-editable; wireable pins can be appended in Faz 2).
        // NOTE: sheen is deliberately NOT exposed â€” PrincipledBSDF uses it as the
        // IS_WATER flag, a graph setting it would silently flip water shading.
        float anisotropic          = 0.0f;
        float dispersion           = 0.0f;   ///< spectral dispersion (0 = off)
        float clearcoatIridescence = 0.0f;   ///< thin-film tint strength
        float clearcoatFilmThickness = 0.55f;
        Vec3  subsurfaceRadius     = Vec3(1.0f, 0.2f, 0.1f);
        float subsurfaceScale      = 0.05f;
        float subsurfaceAnisotropy = 0.0f;
        float subsurfaceIOR        = 1.4f;
        float tileBreakStrength    = 0.0f;   ///< UV tile-break (0 = off)

        // Interior Volume (resin) â€” mirrors PrincipledBSDF::ResinExtension
        float transmissionDensity  = 0.0f;   ///< interior absorption density
        Vec3  resinColor           = Vec3(1.0f, 1.0f, 1.0f);
        float resinRoughness       = 0.1f;
        float resinInclusion       = 0.0f;   ///< dust cloudiness amount
        float resinDirt            = 0.0f;   ///< opaque dirt-speck amount
        float resinInclusionScale  = 8.0f;
        Vec3  resinDirtColor       = Vec3(0.18f, 0.14f, 0.10f);
        float resinShard           = 0.0f;   ///< colored glass-shard amount
        float resinShardHue        = -1.0f;  ///< base hue 0..1; <0 = rainbow
        bool  resinObjectSpace     = true;
        int   dustStyle            = 0;      ///< 0=Nebula 1=Billow 2=Wispy 3=Paint swirl
        Vec3  dustColorA           = Vec3(1.0f, 1.0f, 1.0f);
        Vec3  dustColorB           = Vec3(1.0f, 1.0f, 1.0f);
        int   shardShape           = 0;      ///< 0=chips 1=crystals
        bool  glassMarbleVolume    = false;

        // Bubble (thin-shell) â€” mirrors PrincipledBSDF::BubbleExtension
        bool  isBubble             = false;
        float bubbleIor            = 1.33f;
        float bubbleFilm           = 0.0f;

        // Texture bindings (nullptr = slot unbound)
        std::shared_ptr<Texture> baseColorTex;
        std::shared_ptr<Texture> roughnessTex;
        std::shared_ptr<Texture> metallicTex;
        std::shared_ptr<Texture> normalTex;
        std::shared_ptr<Texture> opacityTex;
        std::shared_ptr<Texture> emissionTex;

        // Material-wide UV transform (captured from a Mapping node feeding an
        // Image Texture; PrincipledBSDF has ONE TextureTransform per material,
        // so per-slot mappings can't be represented â€” first one wins + warning).
        Vec2  uvScale  = Vec2(1.0f, 1.0f);
        Vec2  uvOffset = Vec2(0.0f, 0.0f);
        float uvRotationDeg = 0.0f;
        bool  hasUvTransform = false;
    };

    /// Snapshot an existing material into a ShadeState (MaterialRef node, Output defaults).
    inline ShadeState makeShadeStateFromMaterial(const PrincipledBSDF& m) {
        ShadeState s;
        s.baseColor         = m.albedoProperty.color;
        s.metallic          = m.metallicProperty.intensity;   // UI slider semantics (scene_ui_materials.cpp)
        s.roughness         = static_cast<float>(m.roughnessProperty.color.x);
        s.specular          = m.specularProperty.intensity;
        s.emissionColor     = m.emissionProperty.color;
        s.emissionStrength  = m.emissionProperty.intensity;
        s.transmission      = m.transmission;
        s.ior               = m.ior;
        s.opacity           = m.opacityProperty.alpha;
        s.translucent       = m.translucent;
        s.clearcoat         = m.getClearcoat();
        s.clearcoatRoughness= m.getClearcoatRoughness();
        s.subsurface        = m.getSubsurface();
        s.subsurfaceColor   = m.getSubsurfaceColor();
        s.normalStrength    = m.get_normal_strength();
        s.anisotropic       = m.anisotropic;
        s.dispersion        = m.dispersion;
        s.clearcoatIridescence   = m.getClearcoatIridescence();
        s.clearcoatFilmThickness = m.getClearcoatFilmThickness();
        s.subsurfaceRadius       = m.getSubsurfaceRadius();
        s.subsurfaceScale        = m.getSubsurfaceScale();
        s.subsurfaceAnisotropy   = m.getSubsurfaceAnisotropy();
        s.subsurfaceIOR          = m.getSubsurfaceIOR();
        s.tileBreakStrength      = m.tile_break_strength;
        s.transmissionDensity    = m.getTransmissionDensity();
        s.resinColor             = m.getResinColor();
        s.resinRoughness         = m.getResinRoughness();
        s.resinInclusion         = m.getResinInclusion();
        s.resinDirt              = m.getResinDirt();
        s.resinInclusionScale    = m.getResinInclusionScale();
        s.resinDirtColor         = m.getResinDirtColor();
        s.resinShard             = m.getResinShard();
        s.resinShardHue          = m.getResinShardHue();
        s.resinObjectSpace       = m.getResinObjectSpace();
        s.dustStyle              = m.getDustStyle();
        s.dustColorA             = m.getDustColorA();
        s.dustColorB             = m.getDustColorB();
        s.shardShape             = m.getShardShape();
        s.glassMarbleVolume      = m.getGlassMarbleVolume();
        s.isBubble               = m.getIsBubble();
        s.bubbleIor              = m.getBubbleIor();
        s.bubbleFilm             = m.getBubbleFilm();
        s.baseColorTex      = m.albedoProperty.texture;
        s.roughnessTex      = m.roughnessProperty.texture;
        s.metallicTex       = m.metallicProperty.texture;
        s.normalTex         = m.normalProperty.texture;
        s.opacityTex        = m.opacityProperty.texture;
        s.emissionTex       = m.emissionProperty.texture;
        s.uvScale           = Vec2(m.textureTransform.scale.u, m.textureTransform.scale.v);
        s.uvOffset          = Vec2(m.textureTransform.translation.u, m.textureTransform.translation.v);
        s.uvRotationDeg     = m.textureTransform.rotation_degrees;
        s.hasUvTransform    = false; // only set when a Mapping node explicitly drives it
        return s;
    }

    /**
     * @brief Write a ShadeState into the target material's existing slots.
     * @returns true if any TEXTURE binding changed (caller must refresh
     *          triangle texture bundles / descriptor state, same as the
     *          material panel's texture_changed path).
     */
    inline bool applyShadeStateToMaterial(const ShadeState& s, PrincipledBSDF& m) {
        const bool texChanged =
            m.albedoProperty.texture    != s.baseColorTex ||
            m.roughnessProperty.texture != s.roughnessTex ||
            m.metallicProperty.texture  != s.metallicTex ||
            m.normalProperty.texture    != s.normalTex ||
            m.opacityProperty.texture   != s.opacityTex ||
            m.emissionProperty.texture  != s.emissionTex;

        m.albedoProperty.color      = s.baseColor;
        m.albedoProperty.intensity  = 1.0f;
        m.albedoProperty.texture    = s.baseColorTex;

        m.roughnessProperty.color   = Vec3(s.roughness);
        m.roughnessProperty.texture = s.roughnessTex;

        m.metallicProperty.color     = Vec3(s.metallic);
        m.metallicProperty.intensity = s.metallic;
        m.metallicProperty.texture   = s.metallicTex;

        m.specularProperty.intensity = s.specular;

        m.emissionProperty.color     = s.emissionColor;
        m.emissionProperty.intensity = s.emissionStrength;
        m.emissionProperty.texture   = s.emissionTex;

        m.setTransmission(s.transmission, s.ior);
        m.translucent = s.translucent;

        m.opacityProperty.alpha   = s.opacity;
        m.opacityProperty.texture = s.opacityTex;
        m.opacityAlpha = s.opacity;

        m.setClearcoat(s.clearcoat, s.clearcoatRoughness);
        m.setClearcoatIridescence(s.clearcoatIridescence);
        m.setClearcoatFilmThickness(s.clearcoatFilmThickness);
        m.setSubsurface(s.subsurface);
        m.setSubsurfaceColor(s.subsurfaceColor);
        m.setSubsurfaceRadius(s.subsurfaceRadius);
        m.setSubsurfaceScale(s.subsurfaceScale);
        m.setSubsurfaceAnisotropy(s.subsurfaceAnisotropy);
        m.setSubsurfaceIOR(s.subsurfaceIOR);

        m.anisotropic = s.anisotropic;
        m.dispersion = s.dispersion;
        m.tile_break_strength = s.tileBreakStrength;

        m.setTransmissionDensity(s.transmissionDensity);
        m.setResinColor(s.resinColor);
        m.setResinRoughness(s.resinRoughness);
        m.setResinInclusion(s.resinInclusion);
        m.setResinDirt(s.resinDirt);
        m.setResinInclusionScale(s.resinInclusionScale);
        m.setResinDirtColor(s.resinDirtColor);
        m.setResinShard(s.resinShard);
        m.setResinShardHue(s.resinShardHue);
        m.setResinObjectSpace(s.resinObjectSpace);
        m.setDustStyle(s.dustStyle);
        m.setDustColorA(s.dustColorA);
        m.setDustColorB(s.dustColorB);
        m.setShardShape(s.shardShape);
        m.setGlassMarbleVolume(s.glassMarbleVolume);
        m.setIsBubble(s.isBubble);
        m.setBubbleIor(s.bubbleIor);
        m.setBubbleFilm(s.bubbleFilm);

        m.normalProperty.texture = s.normalTex;
        m.set_normal_strength(s.normalStrength);

        // UV transform: the GRAPH is authoritative. A Mapping node feeding an Image Texture
        // sets it; NO Mapping node means IDENTITY.
        //
        // The else-branch is not cosmetic. The material's TextureTransform is only ever
        // applied by the direct-bind sampling path (PrincipledBSDF::applyTextureTransform);
        // the compiled per-pixel program samples raw mesh UV. So a transform left over in
        // the material — from an import, the material panel, or a Mapping node the user has
        // since deleted — kept warping one path and not the other, and the SAME texture then
        // wrapped differently depending on whether a manipulation node happened to sit in
        // its chain. Leaving it stale is what made a re-assigned texture look mis-mapped
        // after a project reload.
        {
            PrincipledBSDF::TextureTransform t = m.textureTransform;
            if (s.hasUvTransform) {
                t.scale = s.uvScale;
                t.translation = s.uvOffset;
                t.rotation_degrees = s.uvRotationDeg;
            } else {
                t.scale = Vec2(1.0f, 1.0f);
                t.translation = Vec2(0.0f, 0.0f);
                t.rotation_degrees = 0.0f;
            }
            // tilingFactor has no node representation at all (Mapping's scale is the graph's
            // tiling), so it must not survive either — it is pure legacy state from the
            // material panel / importer.
            t.tilingFactor = Vec2(1.0f, 1.0f);
            m.setTextureTransform(t);
        }
        return texChanged;
    }

    // ============================================================================
    // SESSION TEXTURE ENUMERATION
    // ============================================================================

    struct SessionTexture {
        std::string name;                 ///< Texture::name (usually the file path)
        std::string displayName;          ///< filename tail for combo display
        std::shared_ptr<Texture> tex;
    };

    /// Every unique texture currently referenced by ANY session material â€” the
    /// Image Texture node's picker source. No new registry to maintain: the
    /// material slots ARE the session texture set.
    inline std::vector<SessionTexture> collectSessionTextures() {
        std::vector<SessionTexture> out;
        std::unordered_set<std::string> seen;
        for (const auto& mat : MaterialManager::getInstance().getAllMaterials()) {
            if (!mat) continue;
            const MaterialProperty* props[] = {
                &mat->albedoProperty, &mat->roughnessProperty, &mat->metallicProperty,
                &mat->specularProperty, &mat->normalProperty, &mat->opacityProperty,
                &mat->transmissionProperty, &mat->emissionProperty, &mat->heightProperty
            };
            for (const auto* p : props) {
                if (!p->texture) continue;
                const std::string& n = p->texture->name;
                if (n.empty() || !seen.insert(n).second) continue;
                SessionTexture st;
                st.name = n;
                const size_t slash = n.find_last_of("/\\");
                st.displayName = (slash == std::string::npos) ? n : n.substr(slash + 1);
                st.tex = p->texture;
                out.push_back(std::move(st));
            }
        }
        std::sort(out.begin(), out.end(),
                  [](const SessionTexture& a, const SessionTexture& b) { return a.displayName < b.displayName; });
        return out;
    }

    /// Resolve a serialized texture reference by name (= Texture::name, its path).
    ///
    /// Order matters: the live session first (zero-copy, and it keeps the graph node and
    /// the material slot pointing at ONE Texture object), then the project's embedded
    /// blob, then disk. The embedded step is not optional — a graph texture that feeds a
    /// slot THROUGH a manipulation node is bound to no material slot, so nothing else in
    /// the project references it, and if it was embedded there is no file on disk to fall
    /// back to. Without it the node comes back empty after a reload.
    ///
    /// The weak registry keeps repeated resolves of the same name sharing one Texture:
    /// before it, every node (and every failed-then-retried resolve) minted its own copy
    /// of the same image — the "why are there suddenly so many textures" bloat.
    inline std::shared_ptr<Texture> resolveTextureByName(const std::string& name) {
        if (name.empty()) return nullptr;

        static std::mutex regMutex;
        static std::unordered_map<std::string, std::weak_ptr<Texture>> registry;
        {
            std::lock_guard<std::mutex> lock(regMutex);
            auto it = registry.find(name);
            if (it != registry.end()) {
                if (auto alive = it->second.lock()) return alive;
                registry.erase(it);
            }
        }

        auto remember = [&name](const std::shared_ptr<Texture>& tex) {
            if (tex) {
                std::lock_guard<std::mutex> lock(regMutex);
                registry[name] = tex;
            }
            return tex;
        };

        for (const auto& st : collectSessionTextures()) {
            if (st.name == name) return remember(st.tex);
        }
        if (auto embedded = resolveEmbeddedProjectTexture(name)) return remember(embedded);

        auto tex = std::make_shared<Texture>(name, TextureType::Albedo);
        if (tex->is_loaded()) return remember(tex);
        return nullptr;
    }

    // ============================================================================
    // EVALUATION CONTEXT (domain)
    // ============================================================================

    /**
     * @brief Domain context: the material this graph is bound to + the UV point
     * being evaluated. Nodes are written per-UV so the same compute() semantics
     * transfer to the Faz 2 per-pixel runtime unchanged.
     */
    struct MaterialEvalContext {
        PrincipledBSDF* boundMaterial = nullptr;
        float u = 0.5f;
        float v = 0.5f;
        // Shading-point geometry (world space) for the Geometry node / 3D noise.
        // In the editor preview there is no real surface, so these fall back to
        // (u, v, 0) / +Z â€” enough for a representative on-node preview.
        float px = 0.5f, py = 0.5f, pz = 0.0f;
        float nx = 0.0f, ny = 0.0f, nz = 1.0f;
        float pointiness = 0.5f;
        // Object Info: the hit object's world-space origin. The editor preview has no
        // object either, so it stays at the world origin there — the on-node preview of
        // an Object Info chain is necessarily "one arbitrary object", not the scatter.
        float ox = 0.0f, oy = 0.0f, oz = 0.0f;
        // Attribute node: named per-vertex channels. There is no surface in the editor
        // preview, so these read 0 (unpainted) there — the same value the render gives a
        // mesh that does not carry the channel.
        float attribs[kMatAttribSlots] = { 0.0f, 0.0f, 0.0f, 0.0f };
        // Object-space shading point (Geometry > Object Position, object-space 3D noise).
        // The preview has no object, so it mirrors the world fallback.
        float opx = 0.5f, opy = 0.5f, opz = 0.0f;
        // View vector (toward the viewer) for Fresnel / Layer Weight. The editor preview
        // has no camera, so it looks straight down the default normal (+Z) — i.e. head-on,
        // which is what the flat preview thumbnail should show.
        float vx = 0.0f, vy = 0.0f, vz = 1.0f;
        // Ambient Occlusion: the editor has no scene to trace, so the preview reads fully
        // unoccluded. The node says so on its face rather than drawing a fake dirt pattern.
        float ao = 1.0f;
    };

    // Procedural hash/noise primitives (pcgHash, valueNoise2D, fbm2D, ridge2D,
    // billow2D, voronoiF1) live in MaterialProceduralMath.h â€” shared verbatim
    // with the per-pixel runtime interpreter (MaterialProgram.h) so the editor
    // preview and the render sample the identical function. Included at the top
    // of this file.

    /// Small live preview grid drawn with the window draw list â€” used inside
    /// node bodies (inline content) and the properties panel alike. sample(u, v,
    /// r, g, b) fills one cell's color.
    template<typename SampleFn>
    inline void drawPreviewGrid(float width, float height, int cols, int rows, SampleFn&& sample) {
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const ImVec2 p = ImGui::GetCursorScreenPos();
        const float cw = width / static_cast<float>(cols);
        const float ch = height / static_cast<float>(rows);
        for (int yi = 0; yi < rows; ++yi) {
            for (int xi = 0; xi < cols; ++xi) {
                float r = 0, g = 0, b = 0;
                sample((xi + 0.5f) / static_cast<float>(cols), (yi + 0.5f) / static_cast<float>(rows), r, g, b);
                const ImU32 c = IM_COL32(
                    static_cast<int>(std::clamp(r, 0.0f, 1.0f) * 255.0f),
                    static_cast<int>(std::clamp(g, 0.0f, 1.0f) * 255.0f),
                    static_cast<int>(std::clamp(b, 0.0f, 1.0f) * 255.0f), 255);
                dl->AddRectFilled(ImVec2(p.x + xi * cw, p.y + yi * ch),
                                  ImVec2(p.x + (xi + 1) * cw, p.y + (yi + 1) * ch), c);
            }
        }
        dl->AddRect(p, ImVec2(p.x + width, p.y + height), IM_COL32(70, 70, 80, 255));
        ImGui::Dummy(ImVec2(width, height + 3.0f));
    }

    // ============================================================================
    // NODE TYPE ENUM
    // ============================================================================

    enum class NodeType {
        Output = 0,
        MaterialRef,
        MixMaterial,
        Value,
        Color,
        TextureCoordinate,
        ImageTexture,
        Mapping,
        Noise,
        Voronoi,
        Checker,
        ColorRamp,
        MixColor,
        Invert,
        Gamma,
        Math,
        SeparateColor,
        CombineColor,
        Geometry,
        Fresnel,
        Clamp,
        MapRange,
        BrightContrast,
        FloatCurve,
        Bump,
        ObjectInfo,  // append only — these ids are serialized
        Attribute,
        Wave,
        Gradient,
        VectorMath,
        HueSaturation,
        RGBCurves,
        LayerWeight,
        AmbientOcclusion,
        Bevel
    };

    // ============================================================================
    // NODE BASE
    // ============================================================================

    class MaterialNodeBase : public NodeSystem::NodeBase {
    public:
        NodeType materialNodeType = NodeType::Value;

        /// Material nodes edit their parameters directly ON the node body
        /// (colors, texture pickers, previews) â€” the properties panel remains
        /// as a mirror. Material Output opts back out (its grouped editor is
        /// panel-sized).
        bool wantsInlineContent() const override { return true; }

        /// Same parameter-persistence contract as GeometryNodeBase: unknown keys
        /// ignored, missing keys keep constructor defaults.
        virtual void serializeParams(nlohmann::json& j) const { (void)j; }
        virtual void deserializeParams(const nlohmann::json& j) { (void)j; }

        MaterialEvalContext* getMaterialContext(NodeSystem::EvaluationContext& ctx) const {
            if (ctx.hasDomainContext<MaterialEvalContext>()) {
                return ctx.getDomainContext<MaterialEvalContext>();
            }
            return nullptr;
        }

        /// Float input with float<->vec3 conversion (vec3 -> channel average).
        float getFloatIn(int index, NodeSystem::EvaluationContext& ctx, float fallback) {
            const NodeSystem::PinValue val = getInputValue(index, ctx);
            if (const auto* f = std::get_if<float>(&val)) return *f;
            if (const auto* i = std::get_if<int>(&val)) return static_cast<float>(*i);
            if (const auto* v3 = std::get_if<std::array<float, 3>>(&val)) {
                return ((*v3)[0] + (*v3)[1] + (*v3)[2]) / 3.0f;
            }
            return fallback;
        }

        /// Color/vector input with float->splat conversion.
        std::array<float, 3> getVec3In(int index, NodeSystem::EvaluationContext& ctx,
                                       const std::array<float, 3>& fallback) {
            const NodeSystem::PinValue val = getInputValue(index, ctx);
            if (const auto* v3 = std::get_if<std::array<float, 3>>(&val)) return *v3;
            if (const auto* f = std::get_if<float>(&val)) return { *f, *f, *f };
            return fallback;
        }

        /// UV input (Vector2) falling back to the context's current sample point.
        std::array<float, 2> getUVIn(int index, NodeSystem::EvaluationContext& ctx) {
            const NodeSystem::PinValue val = getInputValue(index, ctx);
            if (const auto* v2 = std::get_if<std::array<float, 2>>(&val)) return *v2;
            if (const auto* mctx = getMaterialContext(ctx)) return { mctx->u, mctx->v };
            return { 0.5f, 0.5f };
        }

        bool isInputConnected(int index, NodeSystem::EvaluationContext& ctx) const {
            NodeSystem::GraphBase* g = ctx.getGraph();
            if (!g || index < 0 || index >= static_cast<int>(inputs.size())) return false;
            return g->getInputSource(inputs[index].id) != nullptr;
        }
    };

    // ---- Shared tone-curve core (Float Curve + RGB Curves) ---------------------
    // One implementation, because two curve editors that "look the same" but evaluate
    // slightly differently is a bug you only notice after an hour of grading.

    struct CurvePoint { float x; float y; };

    /**
     * @brief THE curve. The UI preview, the fold, and the compiler's LUT bake all go
     * through here, so what you draw is what renders.
     *
     * Smooth is a MONOTONE cubic (Fritsch–Carlson), not Catmull-Rom: a plain cubic
     * overshoots between points, and a curve driving Roughness or Metallic that dips below
     * 0 or climbs past 1 is an artifact, not a look. Monotone cubic cannot overshoot —
     * between two points it never leaves their [y0,y1] range.
     *
     * O(n) with n <= 16, and it never runs per pixel: the render path evaluates a baked LUT
     * (MatOp::CurveLUT), which is O(1) and identical on CPU and GPU.
     */
    inline float evalCurvePoints(const std::vector<CurvePoint>& points, int interpolation, float x) {
        const int n = static_cast<int>(points.size());
        if (n == 0) return 0.0f;
        if (n == 1) return points[0].y;
        if (x <= points.front().x) return points.front().y;
        if (x >= points.back().x)  return points.back().y;

        int i = 0;
        for (; i + 1 < n; ++i) if (x <= points[i + 1].x) break;
        if (i + 1 >= n) return points.back().y;

        if (interpolation == 1) return points[i].y;   // Constant

        const float x0 = points[i].x, x1 = points[i + 1].x;
        const float y0 = points[i].y, y1 = points[i + 1].y;
        const float h = x1 - x0;
        if (h <= 1e-6f) return y1;
        const float t = (x - x0) / h;

        if (interpolation != 2) return y0 + (y1 - y0) * t;   // Linear

        auto secant = [&](int k) -> float {
            const float dx = points[k + 1].x - points[k].x;
            return (dx > 1e-6f) ? (points[k + 1].y - points[k].y) / dx : 0.0f;
        };
        auto tangent = [&](int k) -> float {
            if (k == 0)     return secant(0);
            if (k == n - 1) return secant(n - 2);
            return 0.5f * (secant(k - 1) + secant(k));
        };
        float m0 = tangent(i), m1 = tangent(i + 1);
        const float s = secant(i);
        if (std::fabs(s) < 1e-8f) {
            m0 = m1 = 0.0f;   // flat segment: kill the tangents or the cubic bulges out of it
        } else {
            const float a = m0 / s, b = m1 / s;
            const float mag = std::sqrt(a * a + b * b);
            if (mag > 3.0f) {                 // Fritsch–Carlson monotonicity clamp
                const float k = 3.0f / mag;
                m0 = k * a * s;
                m1 = k * b * s;
            }
        }
        const float t2 = t * t, t3 = t2 * t;
        return (2.0f * t3 - 3.0f * t2 + 1.0f) * y0
             + (t3 - 2.0f * t2 + t) * h * m0
             + (-2.0f * t3 + 3.0f * t2) * y1
             + (t3 - t2) * h * m1;
    }

    /// Interactive curve widget (click = add point, drag = move, right-click = delete).
    /// Returns true when the curve changed. Shared so both curve nodes behave identically.
    ///
    /// `dragOwner` is NOT optional state — it is the fix for a real bug. drawContent() runs
    /// TWICE per frame for the selected node: once inline in the node body and once in the
    /// properties panel, both against the same member `points`. Without an owner, the copy
    /// the user is NOT dragging still ran the drag branch, remapped the mouse into ITS OWN
    /// rect (far away, so the position clamped to an edge) and slammed the point to the
    /// bottom of the curve. That is the "points spawn at the bottom" behaviour. Only the
    /// widget whose InvisibleButton is ImGui-active may move a point; ImGui guarantees at
    /// most one active item, so one owner id is enough. ColorRampNode already does this —
    /// this widget was written later and missed the pattern.
    inline bool drawCurveWidget(std::vector<CurvePoint>& points, int interpolation,
                                float width, float height,
                                int& selectedPoint, int& draggingPoint, ImGuiID& dragOwner) {
        bool changed = false;
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const ImVec2 p = ImGui::GetCursorScreenPos();

        ImGui::InvisibleButton("##curvewidget", ImVec2(width, height));
        const ImGuiID myId = ImGui::GetItemID();
        const bool active = ImGui::IsItemActive();
        const bool clicked = ImGui::IsItemClicked(ImGuiMouseButton_Left);
        const bool rightClicked = ImGui::IsItemClicked(ImGuiMouseButton_Right);
        const ImVec2 mouse = ImGui::GetIO().MousePos;
        const float mx = std::clamp((mouse.x - p.x) / width, 0.0f, 1.0f);
        const float my = std::clamp(1.0f - (mouse.y - p.y) / height, 0.0f, 1.0f);

        auto sortAndRefind = [&](float kx, float ky) {
            std::sort(points.begin(), points.end(),
                      [](const CurvePoint& a, const CurvePoint& b) { return a.x < b.x; });
            for (int i = 0; i < static_cast<int>(points.size()); ++i) {
                if (points[i].x == kx && points[i].y == ky) { selectedPoint = i; draggingPoint = i; return; }
            }
        };

        if (clicked) {
            dragOwner = myId;
            bool hit = false;
            for (int i = 0; i < static_cast<int>(points.size()); ++i) {
                const ImVec2 scr(p.x + points[i].x * width, p.y + (1.0f - points[i].y) * height);
                if (std::fabs(scr.x - mouse.x) < 6.0f && std::fabs(scr.y - mouse.y) < 6.0f) {
                    selectedPoint = i; draggingPoint = i; hit = true; break;
                }
            }
            if (!hit && points.size() < 16) {
                points.push_back({ mx, my });
                sortAndRefind(mx, my);
                changed = true;
            }
        }
        // The two endpoints are structural — deleting them would leave the curve undefined
        // outside the remaining span, so they stay.
        if (rightClicked && points.size() > 2) {
            for (int i = 1; i + 1 < static_cast<int>(points.size()); ++i) {
                const ImVec2 scr(p.x + points[i].x * width, p.y + (1.0f - points[i].y) * height);
                if (std::fabs(scr.x - mouse.x) < 6.0f && std::fabs(scr.y - mouse.y) < 6.0f) {
                    points.erase(points.begin() + i);
                    selectedPoint = -1; draggingPoint = -1;
                    changed = true;
                    break;
                }
            }
        }
        if (dragOwner == myId) {
            if (!active) {
                draggingPoint = -1;
                dragOwner = 0;
            } else if (draggingPoint >= 0 && draggingPoint < static_cast<int>(points.size())) {
                points[draggingPoint].x = mx;
                points[draggingPoint].y = my;
                if (draggingPoint == 0) points[draggingPoint].x = 0.0f;
                if (draggingPoint == static_cast<int>(points.size()) - 1) points[draggingPoint].x = 1.0f;
                sortAndRefind(mx, my);
                changed = true;
            }
        }

        dl->AddRectFilled(p, ImVec2(p.x + width, p.y + height), IM_COL32(28, 28, 32, 255));
        // Quarter grid: a tone curve is unreadable without one — you cannot see where the
        // identity diagonal is, and every adjustment becomes guesswork.
        for (int g = 1; g < 4; ++g) {
            const float f = static_cast<float>(g) * 0.25f;
            dl->AddLine(ImVec2(p.x + f * width, p.y), ImVec2(p.x + f * width, p.y + height), IM_COL32(255, 255, 255, 16));
            dl->AddLine(ImVec2(p.x, p.y + f * height), ImVec2(p.x + width, p.y + f * height), IM_COL32(255, 255, 255, 16));
        }
        dl->AddLine(ImVec2(p.x, p.y + height), ImVec2(p.x + width, p.y), IM_COL32(255, 255, 255, 24));  // identity
        dl->AddRect(p, ImVec2(p.x + width, p.y + height), IM_COL32(70, 70, 80, 255));
        const int kSegments = 48;
        for (int i = 0; i < kSegments; ++i) {
            const float xa = static_cast<float>(i) / kSegments;
            const float xb = static_cast<float>(i + 1) / kSegments;
            const float ya = std::clamp(evalCurvePoints(points, interpolation, xa), 0.0f, 1.0f);
            const float yb = std::clamp(evalCurvePoints(points, interpolation, xb), 0.0f, 1.0f);
            dl->AddLine(ImVec2(p.x + xa * width, p.y + (1.0f - ya) * height),
                        ImVec2(p.x + xb * width, p.y + (1.0f - yb) * height),
                        IM_COL32(220, 220, 230, 255), 1.5f);
        }
        for (int i = 0; i < static_cast<int>(points.size()); ++i) {
            const ImVec2 scr(p.x + points[i].x * width, p.y + (1.0f - points[i].y) * height);
            dl->AddCircleFilled(scr, 3.5f,
                                (i == selectedPoint) ? IM_COL32(255, 190, 60, 255)
                                                     : IM_COL32(200, 200, 210, 255));
        }
        ImGui::Dummy(ImVec2(0.0f, 2.0f));
        return changed;
    }

    /// The selected point's exact X/Y + the hint line. Shared by both curve nodes so the
    /// two editors stay the same thing (they had drifted into two different layouts).
    inline bool drawCurvePointFields(std::vector<CurvePoint>& points, int selectedPoint, float itemWidth) {
        bool changed = false;
        if (selectedPoint >= 0 && selectedPoint < static_cast<int>(points.size())) {
            const bool isEnd = (selectedPoint == 0) ||
                               (selectedPoint == static_cast<int>(points.size()) - 1);
            ImGui::BeginDisabled(isEnd);   // endpoint X is pinned to 0 / 1 by construction
            ImGui::SetNextItemWidth(itemWidth);
            if (ImGui::SliderFloat("X", &points[selectedPoint].x, 0.0f, 1.0f, "%.3f")) changed = true;
            ImGui::EndDisabled();
            ImGui::SameLine();
            ImGui::SetNextItemWidth(itemWidth);
            if (ImGui::SliderFloat("Y", &points[selectedPoint].y, 0.0f, 1.0f, "%.3f")) changed = true;
        } else {
            ImGui::TextDisabled("click: add  |  right-click: remove");
        }
        return changed;
    }

    // Shared header color per category (kept close to the terrain/geo palette)
    namespace HeaderColors {
        inline ImU32 input()   { return IM_COL32(56, 130, 70, 255); }
        inline ImU32 texture() { return IM_COL32(160, 110, 40, 255); }
        inline ImU32 color()   { return IM_COL32(150, 120, 50, 255); }
        inline ImU32 convert() { return IM_COL32(70, 110, 150, 255); }
        inline ImU32 vector()  { return IM_COL32(90, 80, 160, 255); }
        inline ImU32 shader()  { return IM_COL32(170, 60, 90, 255); }
    }

    // ============================================================================
    // INPUT NODES
    // ============================================================================

    class ValueNode : public MaterialNodeBase {
    public:
        float value = 0.5f;

        ValueNode() {
            name = "Value";
            materialNodeType = NodeType::Value;
            outputs.push_back(NodeSystem::Pin::createOutput("Value", NodeSystem::DataType::Float));
            metadata.displayName = "Value";
            metadata.category = "Input";
            metadata.description = "Constant float value.";
            metadata.headerColor = HeaderColors::input();
        }
        std::string getTypeId() const override { return "MatV2.Value"; }
        void serializeParams(nlohmann::json& j) const override { j["value"] = value; }
        void deserializeParams(const nlohmann::json& j) override { value = j.value("value", 0.5f); }
        void drawContent() override {
            ImGui::SetNextItemWidth(120);
            if (ImGui::DragFloat("Value", &value, 0.01f)) dirty = true;
        }
        NodeSystem::PinValue compute(int, NodeSystem::EvaluationContext&) override { return value; }
    };

    class ColorNode : public MaterialNodeBase {
    public:
        float color[3] = { 0.8f, 0.8f, 0.8f };

        ColorNode() {
            name = "Color";
            materialNodeType = NodeType::Color;
            outputs.push_back(NodeSystem::Pin::createOutput("Color", NodeSystem::DataType::Vector3));
            metadata.displayName = "Color";
            metadata.category = "Input";
            metadata.description = "Constant RGB color.";
            metadata.headerColor = HeaderColors::input();
        }
        std::string getTypeId() const override { return "MatV2.Color"; }
        void serializeParams(nlohmann::json& j) const override { j["rgb"] = { color[0], color[1], color[2] }; }
        void deserializeParams(const nlohmann::json& j) override {
            if (j.contains("rgb") && j["rgb"].is_array() && j["rgb"].size() >= 3) {
                color[0] = j["rgb"][0].get<float>();
                color[1] = j["rgb"][1].get<float>();
                color[2] = j["rgb"][2].get<float>();
            }
        }
        void drawContent() override {
            if (ImGui::ColorEdit3("##col", color, ImGuiColorEditFlags_NoInputs)) dirty = true;
        }
        NodeSystem::PinValue compute(int, NodeSystem::EvaluationContext&) override {
            return std::array<float, 3>{ color[0], color[1], color[2] };
        }
    };

    class TextureCoordinateNode : public MaterialNodeBase {
    public:
        TextureCoordinateNode() {
            name = "Texture Coordinate";
            materialNodeType = NodeType::TextureCoordinate;
            outputs.push_back(NodeSystem::Pin::createOutput("UV", NodeSystem::DataType::Vector2));
            metadata.displayName = "Texture Coordinate";
            metadata.category = "Input";
            metadata.description = "The surface UV being shaded.";
            metadata.headerColor = HeaderColors::input();
        }
        std::string getTypeId() const override { return "MatV2.TextureCoordinate"; }
        NodeSystem::PinValue compute(int, NodeSystem::EvaluationContext& ctx) override {
            if (const auto* mctx = getMaterialContext(ctx)) {
                return std::array<float, 2>{ mctx->u, mctx->v };
            }
            return std::array<float, 2>{ 0.5f, 0.5f };
        }
    };

    /**
     * @brief Shading-point geometry as a node: world-space Position and Normal.
     * Feed Position into a Noise Texture (3D mode) for seamless SOLID texturing
     * (no UV seams), or into Math/Separate chains for position-based gradients.
     * (Per-pixel on CPU + Vulkan RT; the runtime supplies the real hit point.)
     */
    class GeometryNode : public MaterialNodeBase {
    public:
        GeometryNode() {
            name = "Geometry";
            materialNodeType = NodeType::Geometry;
            outputs.push_back(NodeSystem::Pin::createOutput("Position", NodeSystem::DataType::Vector3));
            outputs.push_back(NodeSystem::Pin::createOutput("Normal", NodeSystem::DataType::Vector3));
            outputs.push_back(NodeSystem::Pin::createOutput("Pointiness", NodeSystem::DataType::Float));
            // Appended LAST: output indices are serialized in the link list, so a new socket
            // may only go at the end or every saved graph's wires shift by one.
            outputs.push_back(NodeSystem::Pin::createOutput("Object Position", NodeSystem::DataType::Vector3));
            outputs.push_back(NodeSystem::Pin::createOutput("Incoming", NodeSystem::DataType::Vector3));
            metadata.displayName = "Geometry";
            metadata.category = "Input";
            metadata.description = "Shading-point Position / Normal / Pointiness / Object Position / Incoming (view).";
            metadata.headerColor = HeaderColors::input();
        }
        std::string getTypeId() const override { return "MatV2.Geometry"; }
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override {
            if (const auto* mctx = getMaterialContext(ctx)) {
                if (outputIndex == 1) return std::array<float, 3>{ mctx->nx, mctx->ny, mctx->nz };
                if (outputIndex == 2) return mctx->pointiness;
                if (outputIndex == 3) return std::array<float, 3>{ mctx->opx, mctx->opy, mctx->opz };
                if (outputIndex == 4) return std::array<float, 3>{ mctx->vx, mctx->vy, mctx->vz };
                return std::array<float, 3>{ mctx->px, mctx->py, mctx->pz };
            }
            if (outputIndex == 1) return std::array<float, 3>{ 0.0f, 0.0f, 1.0f };
            if (outputIndex == 2) return 0.5f;
            if (outputIndex == 4) return std::array<float, 3>{ 0.0f, 0.0f, 1.0f };
            return std::array<float, 3>{ 0.5f, 0.5f, 0.0f };
        }
    };

    /**
     * @brief Per-object inputs: world-space Location and a stable per-object Random.
     *
     * The point of this node is ONE material driving many objects: 10k scattered rocks
     * sharing a single material, each with its own tint / roughness / pattern phase.
     * Drive a ColorRamp or a Hue shift from Random, or offset a 3D noise by Location.
     *
     * Random is hashed from the object's world ORIGIN, not from an instance id. The id
     * looks like the natural key and is the one thing that cannot work here: the CPU
     * numbers instances in Embree geomID order and the GPU in TLAS order, two orderings
     * built independently of each other, so the same rock would be a different color in
     * the CPU render than in the Vulkan RT render. The origin is a physical quantity both
     * backends already hold, bit-for-bit (Matrix4x4 is float32 and the TLAS transform is
     * a verbatim copy). Consequence to know about: two objects sitting at the EXACT same
     * origin get the same Random — for scattered instances that never happens, and for a
     * hand-placed pair it is fixed by nudging one of them.
     *
     * There is deliberately no Object Index output: a stable index needs a real id
     * plumbed into VkInstanceData and the shaders that read binding 5, which is a shader
     * ABI change. Location + Random is what per-object variation actually wants.
     */
    class ObjectInfoNode : public MaterialNodeBase {
    public:
        ObjectInfoNode() {
            name = "Object Info";
            materialNodeType = NodeType::ObjectInfo;
            outputs.push_back(NodeSystem::Pin::createOutput("Location", NodeSystem::DataType::Vector3));
            outputs.push_back(NodeSystem::Pin::createOutput("Random", NodeSystem::DataType::Float));
            metadata.displayName = "Object Info";
            metadata.category = "Input";
            metadata.description = "Per-object world Location + stable Random - one material, many objects (Scatter).";
            metadata.headerColor = HeaderColors::input();
        }
        std::string getTypeId() const override { return "MatV2.ObjectInfo"; }
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override {
            if (const auto* mctx = getMaterialContext(ctx)) {
                if (outputIndex == 1) return objectRandom01(mctx->ox, mctx->oy, mctx->oz);
                return std::array<float, 3>{ mctx->ox, mctx->oy, mctx->oz };
            }
            if (outputIndex == 1) return objectRandom01(0.0f, 0.0f, 0.0f);
            return std::array<float, 3>{ 0.0f, 0.0f, 0.0f };
        }
    };


    /// Names of the per-vertex float channels the LIVE scene's meshes carry, for the
    /// Attribute node's picker. Refreshed by the material-node panel each frame (it is the
    /// only place that holds a SceneData); the node itself must not reach for the scene.
    /// Same shape as the session texture list the Image Texture node reads.
    inline std::vector<std::string>& availableVertexAttributeNames() {
        static std::vector<std::string> names;
        return names;
    }

    /**
     * @brief A named per-vertex float channel as a material input.
     *
     * This is the node that connects the tools to the shader: a sculpt mask, a Geo-DAG
     * mask (Mask / Mask Remap), a paint layer or an imported vertex group is just a named
     * float attribute on the mesh's GeometryDetail, and this reads it at the shading point.
     * Wire it into a Mix Color's Fac to blend two materials along a painted seam, into
     * Roughness for worn edges, into a ColorRamp for anything.
     *
     * The name is resolved to a small integer SLOT at compile time (materialAttributeSlot),
     * because the GPU cannot look up strings; the mesh then uploads those slots as one
     * interleaved per-vertex block. There are kMatAttribSlots (4) slots scene-wide — the
     * cost of a slot is 4 bytes per vertex per mesh, so this is deliberately a small,
     * fixed budget rather than an open-ended one. Past the budget the compiler drops the
     * chain to the fold and warns.
     *
     * A mesh that does not carry the named channel reads 0 = UNPAINTED (never 1, which
     * would select everything), and that is also what the editor preview shows: there is
     * no surface there.
     */
    class AttributeNode : public MaterialNodeBase {
    public:
        std::string attributeName;

        AttributeNode() {
            name = "Attribute";
            materialNodeType = NodeType::Attribute;
            outputs.push_back(NodeSystem::Pin::createOutput("Fac", NodeSystem::DataType::Float));
            outputs.push_back(NodeSystem::Pin::createOutput("Color", NodeSystem::DataType::Vector3));
            metadata.displayName = "Attribute";
            metadata.category = "Input";
            metadata.description = "Named per-vertex channel (sculpt/Geo-DAG mask, paint layer, vertex group).";
            metadata.headerColor = HeaderColors::input();
        }

        std::string getTypeId() const override { return "MatV2.Attribute"; }
        float getCustomWidth() const override { return 190.0f; }

        void serializeParams(nlohmann::json& j) const override { j["attr"] = attributeName; }
        void deserializeParams(const nlohmann::json& j) override {
            if (j.contains("attr")) attributeName = j.value("attr", std::string());
        }

        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override {
            float a = 0.0f;
            if (const auto* mctx = getMaterialContext(ctx)) {
                // Look up only — compute() runs on every fold/preview frame, and interning
                // here would let a half-typed name eat a slot. The COMPILER interns.
                const int slot = findMaterialAttributeSlot(attributeName);
                if (slot >= 0 && slot < kMatAttribSlots) a = mctx->attribs[slot];
            }
            if (outputIndex == 1) return std::array<float, 3>{ a, a, a };
            return a;
        }

        void drawContent() override {
            ImGui::PushItemWidth(-1.0f);
            const auto& avail = availableVertexAttributeNames();
            const std::string preview = attributeName.empty() ? std::string("(none)") : attributeName;
            if (ImGui::BeginCombo("##attr", preview.c_str())) {
                if (avail.empty()) {
                    ImGui::TextDisabled("no vertex attributes in scene");
                }
                for (const auto& n : avail) {
                    const bool sel = (n == attributeName);
                    if (ImGui::Selectable(n.c_str(), sel)) {
                        attributeName = n;
                        dirty = true;
                    }
                    if (sel) ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
            // Free-text too: a channel can be authored by a Geo-DAG node that has not been
            // applied yet, so it would not show up in the scan above.
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%s", attributeName.c_str());
            if (ImGui::InputText("##attrName", buf, sizeof(buf))) {
                attributeName = buf;
                dirty = true;
            }
            ImGui::PopItemWidth();
        }
    };

    /**
     * @brief Wave texture — bands / rings with a distorted phase. Wood and marble.
     *
     * Distortion is the whole point: without it these are a clean sine and nobody needs a
     * node for that. The distortion term is an fbm on the same coordinate, which is what
     * turns regular bands into grain.
     */
    class WaveTextureNode : public MaterialNodeBase {
    public:
        int   waveType = 0;     ///< 0 Bands, 1 Rings
        int   direction = 0;    ///< 0 X, 1 Y, 2 Z, 3 Diagonal
        int   profile = 0;      ///< 0 Sine, 1 Saw, 2 Triangle
        float scale = 5.0f;
        float distortion = 2.0f;
        int   detail = 2;
        float detailScale = 1.0f;
        float phase = 0.0f;
        int   dimensions = 2;   ///< 2 = UV, 3 = position (solid)
        bool  objectSpace = false;

        WaveTextureNode() {
            name = "Wave Texture";
            materialNodeType = NodeType::Wave;
            inputs.push_back(NodeSystem::Pin::createInput("Vector", NodeSystem::DataType::Vector3,
                                                          NodeSystem::ImageSemantic::Generic, true));
            outputs.push_back(NodeSystem::Pin::createOutput("Fac", NodeSystem::DataType::Float));
            outputs.push_back(NodeSystem::Pin::createOutput("Color", NodeSystem::DataType::Vector3));
            metadata.displayName = "Wave Texture";
            metadata.category = "Texture";
            metadata.description = "Bands / rings with distorted phase - wood grain, marble.";
            metadata.headerColor = HeaderColors::texture();
        }
        std::string getTypeId() const override { return "MatV2.Wave"; }
        float getCustomWidth() const override { return 190.0f; }

        float sampleFac(float x, float y, float z) const {
            const float sx = x * scale, sy = y * scale, sz = z * scale;
            float n;
            if (waveType == 1) {
                float rx = sx, ry = sy, rz = sz;
                if (direction == 0) rx = 0.0f; else if (direction == 1) ry = 0.0f; else if (direction == 2) rz = 0.0f;
                n = std::sqrt(rx * rx + ry * ry + rz * rz) * 20.0f;
            } else {
                const float d = (direction == 0) ? sx : (direction == 1) ? sy
                              : (direction == 2) ? sz : (sx + sy + sz);
                n = d * 20.0f;
            }
            n += phase;
            if (distortion != 0.0f && detail > 0) {
                const float w = (dimensions == 3)
                    ? fbm3D(sx * detailScale, sy * detailScale, sz * detailScale, detail, 0.5f, 0x9E37u)
                    : fbm2D(sx * detailScale, sy * detailScale, detail, 0.5f, 0x9E37u);
                n += distortion * (w * 2.0f - 1.0f);
            }
            const float TWO_PI = 6.2831853f;
            float fac;
            if (profile == 1)      { const float s = n / TWO_PI; fac = s - std::floor(s); }
            else if (profile == 2) { const float s = n / TWO_PI; fac = std::fabs(2.0f * (s - std::floor(s)) - 1.0f); }
            else                   { fac = 0.5f + 0.5f * std::sin(n - 1.5707963f); }
            return std::clamp(fac, 0.0f, 1.0f);
        }

        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override {
            float x = 0.5f, y = 0.5f, z = 0.0f;
            if (const auto* mctx = getMaterialContext(ctx)) {
                if (dimensions == 3) {
                    if (objectSpace) { x = mctx->opx; y = mctx->opy; z = mctx->opz; }
                    else             { x = mctx->px;  y = mctx->py;  z = mctx->pz;  }
                } else { x = mctx->u; y = mctx->v; z = 0.0f; }
            }
            const float f = sampleFac(x, y, z);
            if (outputIndex == 1) return std::array<float, 3>{ f, f, f };
            return f;
        }

        void serializeParams(nlohmann::json& j) const override {
            j["wtype"] = waveType; j["dir"] = direction; j["profile"] = profile;
            j["scale"] = scale; j["distortion"] = distortion; j["detail"] = detail;
            j["dscale"] = detailScale; j["phase"] = phase;
            j["dims"] = dimensions; j["objspace"] = objectSpace;
        }
        void deserializeParams(const nlohmann::json& j) override {
            waveType = j.value("wtype", 0); direction = j.value("dir", 0); profile = j.value("profile", 0);
            scale = j.value("scale", 5.0f); distortion = j.value("distortion", 2.0f);
            detail = j.value("detail", 2); detailScale = j.value("dscale", 1.0f);
            phase = j.value("phase", 0.0f);
            dimensions = j.value("dims", 2); objectSpace = j.value("objspace", false);
        }

        void drawContent() override {
            drawPreviewGrid(150.0f, 60.0f, 50, 20, [&](float u, float v, float& r, float& g, float& b) {
                const float f = sampleFac(u, v * 0.4f, 0.0f);   // preview is 2.5:1
                r = g = b = f;
            });
            static const char* typeNames[]    = { "Bands", "Rings" };
            static const char* dirNames[]     = { "X", "Y", "Z", "Diagonal" };
            static const char* profileNames[] = { "Sine", "Saw", "Triangle" };
            ImGui::SetNextItemWidth(120);
            if (ImGui::Combo("Type", &waveType, typeNames, 2)) dirty = true;
            ImGui::SetNextItemWidth(120);
            if (ImGui::Combo("Direction", &direction, dirNames, 4)) dirty = true;
            ImGui::SetNextItemWidth(120);
            if (ImGui::Combo("Profile", &profile, profileNames, 3)) dirty = true;
            ImGui::SetNextItemWidth(120);
            if (ImGui::DragFloat("Scale", &scale, 0.05f, 0.01f, 1000.0f)) dirty = true;
            ImGui::SetNextItemWidth(120);
            if (ImGui::SliderFloat("Distortion", &distortion, 0.0f, 10.0f)) dirty = true;
            ImGui::SetNextItemWidth(120);
            if (ImGui::SliderInt("Detail", &detail, 0, 8)) dirty = true;
            ImGui::SetNextItemWidth(120);
            if (ImGui::DragFloat("Detail Scale", &detailScale, 0.05f, 0.0f, 20.0f)) dirty = true;
            ImGui::SetNextItemWidth(120);
            if (ImGui::DragFloat("Phase", &phase, 0.05f)) dirty = true;
            {
                const char* dimNames[] = { "2D (UV)", "3D (Position)" };
                int di = (dimensions == 3) ? 1 : 0;
                ImGui::SetNextItemWidth(120);
                if (ImGui::Combo("Space", &di, dimNames, 2)) { dimensions = di ? 3 : 2; dirty = true; }
                if (dimensions == 3 && ImGui::Checkbox("Object Space", &objectSpace)) dirty = true;
            }
        }
    };

    /**
     * @brief Gradient texture — the standard ramp shapes over the input vector.
     */
    class GradientTextureNode : public MaterialNodeBase {
    public:
        /// 0 Linear, 1 Quadratic, 2 Easing, 3 Diagonal, 4 Radial, 5 Quadratic Sphere, 6 Spherical
        int   gradType = 0;
        int   dimensions = 2;
        bool  objectSpace = false;

        GradientTextureNode() {
            name = "Gradient Texture";
            materialNodeType = NodeType::Gradient;
            inputs.push_back(NodeSystem::Pin::createInput("Vector", NodeSystem::DataType::Vector3,
                                                          NodeSystem::ImageSemantic::Generic, true));
            outputs.push_back(NodeSystem::Pin::createOutput("Fac", NodeSystem::DataType::Float));
            outputs.push_back(NodeSystem::Pin::createOutput("Color", NodeSystem::DataType::Vector3));
            metadata.displayName = "Gradient Texture";
            metadata.category = "Texture";
            metadata.description = "Linear / radial / spherical ramps over the input vector.";
            metadata.headerColor = HeaderColors::texture();
        }
        std::string getTypeId() const override { return "MatV2.Gradient"; }
        float getCustomWidth() const override { return 190.0f; }

        float sampleFac(float x, float y, float z) const {
            float fac;
            switch (gradType) {
                case 1: { const float g = std::max(x, 0.0f); fac = g * g; } break;
                case 2: { const float g = std::clamp(x, 0.0f, 1.0f); fac = g * g * (3.0f - 2.0f * g); } break;
                case 3: fac = (x + y) * 0.5f; break;
                case 4: fac = std::atan2(y, x) / 6.2831853f + 0.5f; break;
                case 5: { const float len = std::sqrt(x * x + y * y + z * z);
                          const float g = std::max(1.0f - len, 0.0f); fac = g * g; } break;
                case 6: { const float len = std::sqrt(x * x + y * y + z * z);
                          fac = std::max(1.0f - len, 0.0f); } break;
                default: fac = x; break;
            }
            return std::clamp(fac, 0.0f, 1.0f);
        }

        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override {
            float x = 0.5f, y = 0.5f, z = 0.0f;
            if (const auto* mctx = getMaterialContext(ctx)) {
                if (dimensions == 3) {
                    if (objectSpace) { x = mctx->opx; y = mctx->opy; z = mctx->opz; }
                    else             { x = mctx->px;  y = mctx->py;  z = mctx->pz;  }
                } else { x = mctx->u; y = mctx->v; z = 0.0f; }
            }
            const float f = sampleFac(x, y, z);
            if (outputIndex == 1) return std::array<float, 3>{ f, f, f };
            return f;
        }

        void serializeParams(nlohmann::json& j) const override {
            j["gtype"] = gradType; j["dims"] = dimensions; j["objspace"] = objectSpace;
        }
        void deserializeParams(const nlohmann::json& j) override {
            gradType = j.value("gtype", 0);
            dimensions = j.value("dims", 2);
            objectSpace = j.value("objspace", false);
        }

        void drawContent() override {
            drawPreviewGrid(150.0f, 60.0f, 50, 20, [&](float u, float v, float& r, float& g, float& b) {
                const float f = sampleFac(u, v * 0.4f, 0.0f);
                r = g = b = f;
            });
            static const char* typeNames[] = { "Linear", "Quadratic", "Easing", "Diagonal",
                                               "Radial", "Quadratic Sphere", "Spherical" };
            ImGui::SetNextItemWidth(140);
            if (ImGui::Combo("Type", &gradType, typeNames, 7)) dirty = true;
            {
                const char* dimNames[] = { "2D (UV)", "3D (Position)" };
                int di = (dimensions == 3) ? 1 : 0;
                ImGui::SetNextItemWidth(120);
                if (ImGui::Combo("Space", &di, dimNames, 2)) { dimensions = di ? 3 : 2; dirty = true; }
                if (dimensions == 3 && ImGui::Checkbox("Object Space", &objectSpace)) dirty = true;
            }
        }
    };

    /**
     * @brief Vector Math — the vector ops the masking chains keep needing.
     *
     * Dot with Geometry > Normal is the "snow on upward-facing surfaces" mask; Length /
     * Distance drive falloffs; Reflect and Cross show up in stylised shading.
     *
     * The Value output exists for the ops that genuinely produce a scalar (Dot, Length,
     * Distance). For the vector-valued ops the VM stores the result as a vec3 and reading it
     * as a scalar gives the channel average — Blender greys the socket out for those, we
     * simply let it read what the register holds.
     */
    class VectorMathNode : public MaterialNodeBase {
    public:
        /// 0 Add, 1 Subtract, 2 Multiply, 3 Divide, 4 Cross, 5 Dot, 6 Normalize, 7 Length,
        /// 8 Distance, 9 Reflect, 10 Scale, 11 Absolute, 12 Minimum, 13 Maximum
        int   op = 0;
        float scaleValue = 1.0f;   ///< only used by Scale

        VectorMathNode() {
            name = "Vector Math";
            materialNodeType = NodeType::VectorMath;
            inputs.push_back(NodeSystem::Pin::createInput("A", NodeSystem::DataType::Vector3,
                                                          NodeSystem::ImageSemantic::Generic, true));
            inputs.push_back(NodeSystem::Pin::createInput("B", NodeSystem::DataType::Vector3,
                                                          NodeSystem::ImageSemantic::Generic, true));
            outputs.push_back(NodeSystem::Pin::createOutput("Vector", NodeSystem::DataType::Vector3));
            outputs.push_back(NodeSystem::Pin::createOutput("Value", NodeSystem::DataType::Float));
            metadata.displayName = "Vector Math";
            metadata.category = "Converter";
            metadata.description = "dot / cross / normalize / length / reflect - masks from geometry.";
            metadata.headerColor = HeaderColors::vector();
        }
        std::string getTypeId() const override { return "MatV2.VectorMath"; }
        float getCustomWidth() const override { return 180.0f; }

        void serializeParams(nlohmann::json& j) const override { j["op"] = op; j["scale"] = scaleValue; }
        void deserializeParams(const nlohmann::json& j) override {
            op = j.value("op", 0); scaleValue = j.value("scale", 1.0f);
        }

        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override {
            const std::array<float, 3> a = getVec3In(0, ctx, { 0.0f, 0.0f, 0.0f });
            const std::array<float, 3> b = getVec3In(1, ctx, { 0.0f, 0.0f, 0.0f });
            std::array<float, 3> r{ 0.0f, 0.0f, 0.0f };
            auto len = [](const std::array<float, 3>& v) {
                return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
            };
            switch (op) {
                case 1: r = { a[0] - b[0], a[1] - b[1], a[2] - b[2] }; break;
                case 2: r = { a[0] * b[0], a[1] * b[1], a[2] * b[2] }; break;
                case 3: for (int i = 0; i < 3; ++i) r[i] = (std::fabs(b[i]) > 1e-8f) ? a[i] / b[i] : 0.0f; break;
                case 4: r = { a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0] }; break;
                case 5: { const float d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; r = { d, d, d }; } break;
                case 6: { const float l = len(a);
                          if (l > 1e-8f) r = { a[0] / l, a[1] / l, a[2] / l }; } break;
                case 7: { const float l = len(a); r = { l, l, l }; } break;
                case 8: { const std::array<float, 3> d{ a[0] - b[0], a[1] - b[1], a[2] - b[2] };
                          const float l = len(d); r = { l, l, l }; } break;
                case 9: { const float l = len(b);
                          if (l > 1e-8f) {
                              const std::array<float, 3> n{ b[0] / l, b[1] / l, b[2] / l };
                              const float d = 2.0f * (a[0] * n[0] + a[1] * n[1] + a[2] * n[2]);
                              r = { a[0] - d * n[0], a[1] - d * n[1], a[2] - d * n[2] };
                          } else r = a;
                        } break;
                case 10: r = { a[0] * scaleValue, a[1] * scaleValue, a[2] * scaleValue }; break;
                case 11: r = { std::fabs(a[0]), std::fabs(a[1]), std::fabs(a[2]) }; break;
                case 12: r = { std::min(a[0], b[0]), std::min(a[1], b[1]), std::min(a[2], b[2]) }; break;
                case 13: r = { std::max(a[0], b[0]), std::max(a[1], b[1]), std::max(a[2], b[2]) }; break;
                default: r = { a[0] + b[0], a[1] + b[1], a[2] + b[2] }; break;
            }
            if (outputIndex == 1) return (r[0] + r[1] + r[2]) / 3.0f;
            return r;
        }

        void drawContent() override {
            static const char* opNames[] = { "Add", "Subtract", "Multiply", "Divide", "Cross Product",
                                             "Dot Product", "Normalize", "Length", "Distance", "Reflect",
                                             "Scale", "Absolute", "Minimum", "Maximum" };
            ImGui::SetNextItemWidth(150);
            if (ImGui::Combo("Op", &op, opNames, 14)) dirty = true;
            if (op == 10) {
                ImGui::SetNextItemWidth(120);
                if (ImGui::DragFloat("Scale", &scaleValue, 0.01f)) dirty = true;
            }
        }
    };

    /**
     * @brief Hue / Saturation / Value.
     *
     * Blender's convention: Hue 0.5 means "no shift", so the slider is a +/- half turn
     * around the wheel. Both VMs implement the same convention (and the same grey-axis
     * behaviour, where the hue is undefined and pinned to 0) — a "cleaner" branchless
     * rewrite on one side is precisely how the two backends drift apart.
     */
    class HueSaturationNode : public MaterialNodeBase {
    public:
        HueSaturationNode() {
            name = "Hue/Saturation";
            materialNodeType = NodeType::HueSaturation;
            {
                NodeSystem::Pin h = NodeSystem::Pin::createInput("Hue", NodeSystem::DataType::Float,
                                                                 NodeSystem::ImageSemantic::Generic, true);
                h.defaultValue = 0.5f;
                inputs.push_back(h);
                NodeSystem::Pin s = NodeSystem::Pin::createInput("Saturation", NodeSystem::DataType::Float,
                                                                 NodeSystem::ImageSemantic::Generic, true);
                s.defaultValue = 1.0f;
                inputs.push_back(s);
                NodeSystem::Pin v = NodeSystem::Pin::createInput("Value", NodeSystem::DataType::Float,
                                                                 NodeSystem::ImageSemantic::Generic, true);
                v.defaultValue = 1.0f;
                inputs.push_back(v);
                NodeSystem::Pin f = NodeSystem::Pin::createInput("Fac", NodeSystem::DataType::Float,
                                                                 NodeSystem::ImageSemantic::Generic, true);
                f.defaultValue = 1.0f;
                inputs.push_back(f);
                NodeSystem::Pin c = NodeSystem::Pin::createInput("Color", NodeSystem::DataType::Vector3,
                                                                 NodeSystem::ImageSemantic::Generic, true);
                c.defaultValue = std::array<float, 3>{ 0.8f, 0.8f, 0.8f };
                inputs.push_back(c);
            }
            outputs.push_back(NodeSystem::Pin::createOutput("Color", NodeSystem::DataType::Vector3));
            metadata.displayName = "Hue/Saturation";
            metadata.category = "Color";
            metadata.description = "Shift hue, scale saturation and value (Hue 0.5 = no shift).";
            metadata.headerColor = HeaderColors::color();
        }
        std::string getTypeId() const override { return "MatV2.HueSaturation"; }

        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override {
            (void)outputIndex;
            const float hue = getFloatIn(0, ctx, 0.5f);
            const float sat = getFloatIn(1, ctx, 1.0f);
            const float val = getFloatIn(2, ctx, 1.0f);
            const float fac = std::clamp(getFloatIn(3, ctx, 1.0f), 0.0f, 1.0f);
            const std::array<float, 3> c = getVec3In(4, ctx, { 0.8f, 0.8f, 0.8f });

            float h, s, v;
            rgbToHsv(c[0], c[1], c[2], h, s, v);
            h += hue - 0.5f;
            h = h - std::floor(h);
            s = std::clamp(s * sat, 0.0f, 1.0f);
            v *= val;
            float r, g, b;
            hsvToRgb(h, s, v, r, g, b);
            return std::array<float, 3>{ c[0] + (r - c[0]) * fac,
                                         c[1] + (g - c[1]) * fac,
                                         c[2] + (b - c[2]) * fac };
        }
    };

    /**
     * @brief RGB Curves — a combined curve plus one per channel.
     *
     * No new opcode: this lowers to Separate -> CurveLUT -> Combine using the ops that
     * already exist. The curves are baked into uniform LUTs at COMPILE time (see
     * MatOp::CurveLUT), so the VM never learns about splines and the two backends cannot
     * disagree about the interpolation.
     */
    class RGBCurvesNode : public MaterialNodeBase {
    public:
        /// Four curves: 0 = combined (C, applied to every channel first), 1..3 = R, G, B.
        /// Same point/interp representation as FloatCurveNode so evalCurveOn() is shared.
        struct Curve {
            std::vector<CurvePoint> points{ { 0.0f, 0.0f }, { 1.0f, 1.0f } };
            int interpolation = 0;   ///< 0 linear, 1 constant, 2 smooth (monotone cubic)
        };
        Curve curves[4];
        int   editing = 0;          ///< which curve the inline editor shows (UI only)
        int   selectedPoint = -1;   ///< UI-only widget state
        int   draggingPoint = -1;
        ImGuiID dragOwner = 0;      ///< which of the two drawContent() instances owns the drag

        RGBCurvesNode() {
            name = "RGB Curves";
            materialNodeType = NodeType::RGBCurves;
            {
                NodeSystem::Pin f = NodeSystem::Pin::createInput("Fac", NodeSystem::DataType::Float,
                                                                 NodeSystem::ImageSemantic::Generic, true);
                f.defaultValue = 1.0f;
                inputs.push_back(f);
                NodeSystem::Pin c = NodeSystem::Pin::createInput("Color", NodeSystem::DataType::Vector3,
                                                                 NodeSystem::ImageSemantic::Generic, true);
                c.defaultValue = std::array<float, 3>{ 0.8f, 0.8f, 0.8f };
                inputs.push_back(c);
            }
            outputs.push_back(NodeSystem::Pin::createOutput("Color", NodeSystem::DataType::Vector3));
            metadata.displayName = "RGB Curves";
            metadata.category = "Color";
            metadata.description = "Per-channel tone curves (C, R, G, B).";
            metadata.headerColor = HeaderColors::color();
        }
        std::string getTypeId() const override { return "MatV2.RGBCurves"; }
        float getCustomWidth() const override { return 200.0f; }

        void serializeParams(nlohmann::json& j) const override {
            nlohmann::json arr = nlohmann::json::array();
            for (const Curve& c : curves) {
                nlohmann::json cj;
                cj["interp"] = c.interpolation;
                nlohmann::json pts = nlohmann::json::array();
                for (const auto& p : c.points) pts.push_back({ p.x, p.y });
                cj["points"] = pts;
                arr.push_back(cj);
            }
            j["curves"] = arr;
        }
        void deserializeParams(const nlohmann::json& j) override {
            if (!j.contains("curves") || !j["curves"].is_array()) return;
            const auto& arr = j["curves"];
            for (size_t i = 0; i < 4 && i < arr.size(); ++i) {
                Curve& c = curves[i];
                c.interpolation = arr[i].value("interp", 0);
                if (!arr[i].contains("points") || !arr[i]["points"].is_array()) continue;
                c.points.clear();
                for (const auto& p : arr[i]["points"]) {
                    if (p.is_array() && p.size() >= 2)
                        c.points.push_back({ p[0].get<float>(), p[1].get<float>() });
                }
                if (c.points.size() < 2) c.points = { { 0.0f, 0.0f }, { 1.0f, 1.0f } };
                std::sort(c.points.begin(), c.points.end(),
                          [](const CurvePoint& a, const CurvePoint& b) { return a.x < b.x; });
            }
        }

        /// Apply C then the per-channel curve — the same order the compiler emits, so the
        /// preview, the fold and the render cannot disagree about it.
        float applyChannel(int ch, float x) const {
            const float c = evalCurvePoints(curves[0].points, curves[0].interpolation, x);
            return evalCurvePoints(curves[ch + 1].points, curves[ch + 1].interpolation, c);
        }

        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override {
            (void)outputIndex;
            const float fac = std::clamp(getFloatIn(0, ctx, 1.0f), 0.0f, 1.0f);
            const std::array<float, 3> col = getVec3In(1, ctx, { 0.8f, 0.8f, 0.8f });
            std::array<float, 3> out{};
            for (int i = 0; i < 3; ++i) {
                const float m = applyChannel(i, std::clamp(col[i], 0.0f, 1.0f));
                out[i] = col[i] + (m - col[i]) * fac;
            }
            return out;
        }

        void drawContent() override {
            static const char* chNames[] = { "C", "R", "G", "B" };
            ImGui::SetNextItemWidth(80);
            if (ImGui::Combo("##ch", &editing, chNames, 4)) {
                selectedPoint = draggingPoint = -1;   // point ids belong to the old curve
                dragOwner = 0;
                dirty = true;
            }
            static const char* interpNames[] = { "Linear", "Constant", "Smooth" };
            ImGui::SetNextItemWidth(90);
            if (ImGui::Combo("##interp", &curves[editing].interpolation, interpNames, 3)) dirty = true;
            if (drawCurveWidget(curves[editing].points, curves[editing].interpolation,
                                180.0f, 90.0f, selectedPoint, draggingPoint, dragOwner)) {
                dirty = true;
            }
            if (drawCurvePointFields(curves[editing].points, selectedPoint, 70.0f)) dirty = true;
        }
    };

    class FresnelNode : public MaterialNodeBase {
    public:
        float ior = 1.45f;   ///< Authority when the IOR pin is unconnected (same pattern
                             ///< as BevelNode::radius — a pin without a widget is a dead
                             ///< parameter, editable only by wiring a Value node).

        FresnelNode() {
            name = "Fresnel";
            materialNodeType = NodeType::Fresnel;
            {
                NodeSystem::Pin ior = NodeSystem::Pin::createInput("IOR", NodeSystem::DataType::Float,
                                                                   NodeSystem::ImageSemantic::Generic, true);
                ior.defaultValue = 1.45f;
                inputs.push_back(ior);
            }
            {
                NodeSystem::Pin nrm = NodeSystem::Pin::createInput("Normal", NodeSystem::DataType::Vector3,
                                                                   NodeSystem::ImageSemantic::Generic, true);
                inputs.push_back(nrm);
            }
            outputs.push_back(NodeSystem::Pin::createOutput("Fac", NodeSystem::DataType::Float));
            metadata.displayName = "Fresnel";
            metadata.category = "Input";
            metadata.description = "Reflectance based on viewing angle and IOR.";
            metadata.headerColor = HeaderColors::input();
        }
        std::string getTypeId() const override { return "MatV2.Fresnel"; }

        void serializeParams(nlohmann::json& j) const override { j["ior"] = ior; }
        void deserializeParams(const nlohmann::json& j) override {
            ior = std::clamp(j.value("ior", 1.45f), 1.0f, 4.0f);
            inputs[0].defaultValue = ior;   // keep compute()'s pin fallback in sync
        }

        void drawContent() override {
            ImGui::SetNextItemWidth(110);
            if (ImGui::DragFloat("IOR", &ior, 0.01f, 1.0f, 4.0f, "%.3f")) {
                ior = std::clamp(ior, 1.0f, 4.0f);
                inputs[0].defaultValue = ior;
                dirty = true;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("A connected IOR pin overrides this.");
            }
        }

        /// Schlick against the REAL viewing angle. Shared with LayerWeightNode and mirrored
        /// by MatOp::Fresnel (CPU) / op 17 (GLSL) — one formula, three call sites.
        static float schlick(float ior, const std::array<float, 3>& n, const MaterialEvalContext* mc) {
            const std::array<float, 3> v = mc ? std::array<float, 3>{ mc->vx, mc->vy, mc->vz }
                                              : std::array<float, 3>{ 0.0f, 0.0f, 1.0f };
            const float ln = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
            const float lv = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
            float ndotv = 1.0f;
            if (ln > 1e-8f && lv > 1e-8f) {
                ndotv = std::fabs((n[0] * v[0] + n[1] * v[1] + n[2] * v[2]) / (ln * lv));
                ndotv = std::min(1.0f, ndotv);
            }
            float r0 = (1.0f - ior) / (1.0f + ior);
            r0 = r0 * r0;
            return r0 + (1.0f - r0) * std::pow(1.0f - ndotv, 5.0f);
        }

        NodeSystem::PinValue compute(int, NodeSystem::EvaluationContext& ctx) override {
            const float ior = getFloatIn(0, ctx, 1.45f);
            const MaterialEvalContext* mc = getMaterialContext(ctx);
            // Unconnected Normal = the shading normal, not a hard (0,0,1): the compiler emits
            // GeoNormal for it, and the editor preview has to agree or the on-node thumbnail
            // would show a different curve than the render.
            const std::array<float, 3> nFallback = mc ? std::array<float, 3>{ mc->nx, mc->ny, mc->nz }
                                                      : std::array<float, 3>{ 0.0f, 0.0f, 1.0f };
            const std::array<float, 3> n = getVec3In(1, ctx, nFallback);
            return schlick(ior, n, mc);
        }
    };

    /**
     * @brief Layer Weight — Blender's companion to Fresnel: the two outputs people
     * actually reach for when mixing a coat over a base.
     *
     *   Fresnel: Schlick with IOR derived from Blend (edge bias), the standard
     *            "more reflective at grazing angles" ramp.
     *   Facing:  1 at the silhouette, 0 head-on — the rim mask. Wire it into Emission
     *            for a rim glow, or into a Mix factor for velvet/fresnel dust.
     *
     * Both need the REAL view vector; before MatOp::GeoIncoming existed neither could be
     * expressed (Fresnel was reading the normal's world-Z tilt and calling it N.V).
     */
    class LayerWeightNode : public MaterialNodeBase {
    public:
        float blend = 0.5f;   ///< Authority when the Blend pin is unconnected (same
                              ///< pattern as FresnelNode::ior).

        LayerWeightNode() {
            name = "Layer Weight";
            materialNodeType = NodeType::LayerWeight;
            {
                NodeSystem::Pin b = NodeSystem::Pin::createInput("Blend", NodeSystem::DataType::Float,
                                                                 NodeSystem::ImageSemantic::Generic, true);
                b.defaultValue = 0.5f;
                inputs.push_back(b);
                NodeSystem::Pin nrm = NodeSystem::Pin::createInput("Normal", NodeSystem::DataType::Vector3,
                                                                   NodeSystem::ImageSemantic::Generic, true);
                inputs.push_back(nrm);
            }
            outputs.push_back(NodeSystem::Pin::createOutput("Fresnel", NodeSystem::DataType::Float));
            outputs.push_back(NodeSystem::Pin::createOutput("Facing", NodeSystem::DataType::Float));
            metadata.displayName = "Layer Weight";
            metadata.category = "Input";
            metadata.description = "Fresnel / Facing from the real viewing angle. Facing = 1 at the silhouette.";
            metadata.headerColor = HeaderColors::input();
        }
        std::string getTypeId() const override { return "MatV2.LayerWeight"; }

        void serializeParams(nlohmann::json& j) const override { j["blend"] = blend; }
        void deserializeParams(const nlohmann::json& j) override {
            blend = std::clamp(j.value("blend", 0.5f), 0.0f, 1.0f);
            inputs[0].defaultValue = blend;   // keep compute()'s pin fallback in sync
        }

        void drawContent() override {
            ImGui::SetNextItemWidth(110);
            if (ImGui::SliderFloat("Blend", &blend, 0.0f, 1.0f, "%.3f")) {
                blend = std::clamp(blend, 0.0f, 1.0f);
                inputs[0].defaultValue = blend;
                dirty = true;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("0.5 = linear Facing ramp; higher widens the rim.\n"
                                  "A connected Blend pin overrides this.");
            }
        }

        /// Blender's mapping: IOR = 1 / (1 - blend), so the default blend 0.5 lands on IOR 2.
        /// Clamped away from the pole at blend = 1 (IOR -> inf) and pushed just above 1 at
        /// blend = 0, where r0 would otherwise be exactly 0 (no reflection at any angle).
        static float blendToIor(float blend) {
            const float b = std::clamp(blend, 0.0f, 1.0f - 1e-5f);
            return std::max(1.0f + 1e-5f, 1.0f / (1.0f - b));
        }

        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override {
            const float blend = std::clamp(getFloatIn(0, ctx, 0.5f), 0.0f, 1.0f);
            const MaterialEvalContext* mc = getMaterialContext(ctx);
            const std::array<float, 3> nFallback = mc ? std::array<float, 3>{ mc->nx, mc->ny, mc->nz }
                                                      : std::array<float, 3>{ 0.0f, 0.0f, 1.0f };
            const std::array<float, 3> n = getVec3In(1, ctx, nFallback);
            if (outputIndex == 1) {                       // Facing
                const std::array<float, 3> v = mc ? std::array<float, 3>{ mc->vx, mc->vy, mc->vz }
                                                  : std::array<float, 3>{ 0.0f, 0.0f, 1.0f };
                const float ln = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
                const float lv = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
                float ndotv = 1.0f;
                if (ln > 1e-8f && lv > 1e-8f) {
                    ndotv = std::min(1.0f, std::fabs((n[0] * v[0] + n[1] * v[1] + n[2] * v[2]) / (ln * lv)));
                }
                const float facing = std::clamp(1.0f - ndotv, 0.0f, 1.0f);
                // Blend biases the ramp: 0.5 = linear, ->1 widens the rim to the whole surface,
                // ->0 tightens it to the silhouette. THIS EXPRESSION IS THE ONE THE COMPILER
                // LOWERS (see NodeType::LayerWeight) — keep the two literally identical or the
                // on-node preview and the render will show different rims.
                const float e = (1.0f - blend) / std::max(blend, 1e-4f);
                return std::pow(facing, e);
            }
            return FresnelNode::schlick(blendToIor(blend), n, mc);
        }
    };

    /**
     * @brief Ambient Occlusion — the last node on the roadmap, and the only one that
     * traces rays.
     *
     * Pointiness gives LOCAL curvature (cheap, per-vertex, no rays). AO gives REAL
     * closure: a mask that darkens where geometry actually crowds the shading point, so
     * dust settles in corners a curvature mask cannot see (under a chair, inside a pipe).
     *
     * COST, stated plainly: `samples` extra shadow rays on EVERY shading call that runs
     * this chain — the only node in the graph that multiplies the ray count. 4 is usually
     * enough behind a Color Ramp; the noise averages out over accumulation.
     *
     * BACKENDS: Vulkan RT traces it in closesthit (mp_traceAO, reusing the NEE shadow
     * payload/miss). CPU traces it against the same BVH the pass traverses. OptiX is
     * frozen and has no hook, so it keeps the Faz-1 folded value there.
     */
    class AmbientOcclusionNode : public MaterialNodeBase {
    public:
        int   samples = 4;
        bool  inside = false;      ///< occlusion of the cavity BEHIND the surface
        float distance = 1.0f;     ///< WORLD units. Authority when the Distance pin is unconnected.

        AmbientOcclusionNode() {
            name = "Ambient Occlusion";
            materialNodeType = NodeType::AmbientOcclusion;
            {
                NodeSystem::Pin c = NodeSystem::Pin::createInput("Color", NodeSystem::DataType::Vector3,
                                                                 NodeSystem::ImageSemantic::Generic, true);
                c.defaultValue = std::array<float, 3>{ 1.0f, 1.0f, 1.0f };
                inputs.push_back(c);
                NodeSystem::Pin d = NodeSystem::Pin::createInput("Distance", NodeSystem::DataType::Float,
                                                                 NodeSystem::ImageSemantic::Generic, true);
                d.defaultValue = 1.0f;
                inputs.push_back(d);
            }
            outputs.push_back(NodeSystem::Pin::createOutput("Color", NodeSystem::DataType::Vector3));
            outputs.push_back(NodeSystem::Pin::createOutput("AO", NodeSystem::DataType::Float));
            metadata.displayName = "Ambient Occlusion";
            metadata.category = "Input";
            metadata.description = "Real occlusion (traced). 1 = open, 0 = fully enclosed. Costs `samples` rays per shading call.";
            metadata.headerColor = HeaderColors::input();
        }
        std::string getTypeId() const override { return "MatV2.AmbientOcclusion"; }
        float getCustomWidth() const override { return 190.0f; }

        void serializeParams(nlohmann::json& j) const override {
            j["samples"] = samples;
            j["inside"] = inside;
            j["distance"] = distance;
        }
        void deserializeParams(const nlohmann::json& j) override {
            samples = std::clamp(j.value("samples", 4), 1, 64);
            inside = j.value("inside", false);
            distance = std::clamp(j.value("distance", 1.0f), 0.01f, 1000.0f);
            inputs[1].defaultValue = distance;   // keep compute()'s pin fallback in sync
        }

        void drawContent() override {
            ImGui::SetNextItemWidth(110);
            if (ImGui::DragFloat("Distance", &distance, 0.05f, 0.01f, 1000.0f, "%.2f")) {
                distance = std::clamp(distance, 0.01f, 1000.0f);
                inputs[1].defaultValue = distance;
                dirty = true;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("WORLD units — how far occluders are searched.\n"
                                  "A connected Distance pin overrides this.");
            }
            ImGui::SetNextItemWidth(110);
            if (ImGui::SliderInt("Samples", &samples, 1, 32)) { samples = std::clamp(samples, 1, 64); dirty = true; }
            if (ImGui::Checkbox("Inside", &inside)) dirty = true;
            // The cost is the whole story with this node — say it on the node, not in a manual.
            ImGui::TextDisabled("%d ray%s / shading call", samples, samples == 1 ? "" : "s");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("The only node that traces. Vulkan RT + CPU trace it;\n"
                                  "OptiX (frozen) keeps the folded value.\n"
                                  "The editor preview cannot trace: it shows 1.0 (open).");
            }
        }

        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override {
            const MaterialEvalContext* mc = getMaterialContext(ctx);
            const float ao = mc ? mc->ao : 1.0f;   // editor/fold: unoccluded, never a fake pattern
            if (outputIndex == 1) return ao;
            const std::array<float, 3> col = getVec3In(0, ctx, { 1.0f, 1.0f, 1.0f });
            return std::array<float, 3>{ col[0] * ao, col[1] * ao, col[2] * ao };
        }
    };

    /**
     * @brief Bevel — rounded-edge SHADING without touching the geometry.
     *
     * A modeled hard edge is mathematically sharp, and a sharp edge catches no light: no
     * thin highlight runs along it, which is the single biggest "this is CG" tell. Real
     * chamfers do it with polygons (the Geo-DAG BEVEL node); this node fakes only the
     * SHADING of one: short probe rays find the surfaces crowding the shading point and
     * their normals blend in, so the edge catches highlights like a fillet while the
     * silhouette stays exactly as modeled.
     *
     * Wire Normal DIRECTLY into the Material Output's Normal Map socket — the compiler
     * marks that chain WORLD-space (StoreWorldNormal) so it bypasses the tangent frame.
     * A processed bevel normal (through VectorMath etc.) still compiles but is treated as
     * tangent-space and will shade wrong; the description says so instead of a warning
     * system this graph does not have.
     *
     * COST: like AO, `samples` extra rays per shading call (the only two nodes that trace).
     * BACKENDS: Vulkan RT + CPU trace it; OptiX (frozen) has no hook — identity there.
     * KNOWN LIMIT: probes see ALL geometry, not just this object (Blender restricts to the
     * same object) — an object standing on a floor grows a soft shading fillet at the
     * contact line. Usually reads as cheap ambient contact; documented, not hidden.
     */
    class BevelNode : public MaterialNodeBase {
    public:
        int   samples = 4;
        float radius = 0.05f;   ///< WORLD units. Authority when the Radius pin is unconnected —
                                ///< the pin exists for driving the radius per-pixel (a texture).

        BevelNode() {
            name = "Bevel";
            materialNodeType = NodeType::Bevel;
            {
                NodeSystem::Pin r = NodeSystem::Pin::createInput("Radius", NodeSystem::DataType::Float,
                                                                 NodeSystem::ImageSemantic::Generic, true);
                r.defaultValue = 0.05f;
                inputs.push_back(r);
            }
            outputs.push_back(NodeSystem::Pin::createOutput("Normal", NodeSystem::DataType::Vector3));
            metadata.displayName = "Bevel";
            metadata.category = "Vector";
            metadata.description = "Rounded-edge shading normal (traced). Wire DIRECTLY into Normal Map. Costs `samples` rays per shading call.";
            metadata.headerColor = HeaderColors::vector();
        }
        std::string getTypeId() const override { return "MatV2.Bevel"; }

        void serializeParams(nlohmann::json& j) const override {
            j["samples"] = samples;
            j["radius"] = radius;
        }
        void deserializeParams(const nlohmann::json& j) override {
            samples = std::clamp(j.value("samples", 4), 1, 16);
            radius = std::clamp(j.value("radius", 0.05f), 0.001f, 10.0f);
            inputs[0].defaultValue = radius;   // keep compute()'s pin fallback in sync
        }

        void drawContent() override {
            ImGui::SetNextItemWidth(110);
            if (ImGui::DragFloat("Radius", &radius, 0.005f, 0.001f, 10.0f, "%.3f")) {
                radius = std::clamp(radius, 0.001f, 10.0f);
                inputs[0].defaultValue = radius;
                dirty = true;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("WORLD units — how far the edge rounding reaches.\n"
                                  "On a 2m cube try 0.1-0.3. A connected Radius pin overrides this.");
            }
            ImGui::SetNextItemWidth(110);
            if (ImGui::SliderInt("Samples", &samples, 1, 16)) { samples = std::clamp(samples, 1, 16); dirty = true; }
            ImGui::TextDisabled("%d ray%s / shading call", samples, samples == 1 ? "" : "s");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Traces like AO. Vulkan RT + CPU only (OptiX keeps the flat edge).\n"
                                  "The editor preview cannot trace: it shows the plain normal.");
            }
        }

        NodeSystem::PinValue compute(int, NodeSystem::EvaluationContext& ctx) override {
            // No scene to trace in the editor: identity (the plain shading normal) — the
            // same value the render produces anywhere no edge is within Radius.
            if (const auto* mc = getMaterialContext(ctx)) {
                return std::array<float, 3>{ mc->nx, mc->ny, mc->nz };
            }
            return std::array<float, 3>{ 0.0f, 0.0f, 1.0f };
        }
    };

    /**
     * @brief Represents an EXISTING session material as a node â€” snapshots its
     * full parameter set + texture bindings into a Material pin. This is what
     * lets Mix Material blend two already-authored materials.
     */
    class MaterialRefNode : public MaterialNodeBase {
    public:
        char materialName[96] = "";

        MaterialRefNode() {
            name = "Material";
            materialNodeType = NodeType::MaterialRef;
            outputs.push_back(NodeSystem::Pin::createOutput("Material", NodeSystem::DataType::Material));
            metadata.displayName = "Material";
            metadata.category = "Input";
            metadata.description = "An existing session material as a full parameter-set snapshot.";
            metadata.headerColor = HeaderColors::shader();
        }
        std::string getTypeId() const override { return "MatV2.MaterialRef"; }
        float getCustomWidth() const override { return 180.0f; }  // inline preview/widgets fit
        void serializeParams(nlohmann::json& j) const override { j["material"] = std::string(materialName); }
        void deserializeParams(const nlohmann::json& j) override {
            const std::string n = j.value("material", std::string());
            strncpy(materialName, n.c_str(), sizeof(materialName) - 1);
            materialName[sizeof(materialName) - 1] = '\0';
        }
        void drawContent() override {
            auto& mm = MaterialManager::getInstance();
            const char* preview = materialName[0] ? materialName : "<select material>";
            ImGui::SetNextItemWidth(160);
            if (ImGui::BeginCombo("##matref", preview)) {
                for (const auto& mat : mm.getAllMaterials()) {
                    if (!mat || mat->type() != MaterialType::PrincipledBSDF) continue;
                    const bool selected = (mat->materialName == materialName);
                    if (ImGui::Selectable(mat->materialName.c_str(), selected)) {
                        strncpy(materialName, mat->materialName.c_str(), sizeof(materialName) - 1);
                        materialName[sizeof(materialName) - 1] = '\0';
                        dirty = true;
                    }
                }
                ImGui::EndCombo();
            }
        }
        NodeSystem::PinValue compute(int, NodeSystem::EvaluationContext& ctx) override {
            auto& mm = MaterialManager::getInstance();
            const uint16_t matId = mm.getMaterialID(materialName);
            Material* mat = (matId != MaterialManager::INVALID_MATERIAL_ID) ? mm.getMaterial(matId) : nullptr;
            auto* pbsdf = dynamic_cast<PrincipledBSDF*>(mat);
            if (!pbsdf) {
                ctx.addError(id, std::string("Material Ref: material not found: ") + materialName);
                return NodeSystem::PinValue{};
            }
            return std::make_shared<ShadeState>(makeShadeStateFromMaterial(*pbsdf));
        }
    };

    // ============================================================================
    // TEXTURE NODES
    // ============================================================================

    class ImageTextureNode : public MaterialNodeBase {
    public:
        std::shared_ptr<Texture> texture;
        char texName[260] = "";          ///< Texture::name (path) â€” persisted, resolved lazily
        bool browseRequested = false;    ///< handled by the editor UI (file dialog)
        bool resolveTried = false;

        ImageTextureNode() {
            name = "Image Texture";
            materialNodeType = NodeType::ImageTexture;
            {
                NodeSystem::Pin p = NodeSystem::Pin::createInput("Vector", NodeSystem::DataType::Vector2,
                                                                 NodeSystem::ImageSemantic::Generic, true);
                inputs.push_back(p);
            }
            outputs.push_back(NodeSystem::Pin::createOutput("Color", NodeSystem::DataType::Vector3));
            outputs.push_back(NodeSystem::Pin::createOutput("Alpha", NodeSystem::DataType::Float));
            metadata.displayName = "Image Texture";
            metadata.category = "Texture";
            metadata.description = "A session texture (or file). Wire straight into an Output slot to BIND it (lossless).";
            metadata.headerColor = HeaderColors::texture();
        }
        std::string getTypeId() const override { return "MatV2.ImageTexture"; }
        float getCustomWidth() const override { return 190.0f; }  // inline preview/widgets fit
        void serializeParams(nlohmann::json& j) const override {
            j["texture"] = texture ? texture->name : std::string(texName);
        }
        void deserializeParams(const nlohmann::json& j) override {
            const std::string n = j.value("texture", std::string());
            strncpy(texName, n.c_str(), sizeof(texName) - 1);
            texName[sizeof(texName) - 1] = '\0';
            texture = nullptr;
            resolveTried = false;
        }
        void resolveIfNeeded() {
            if (!texture && texName[0] && !resolveTried) {
                resolveTried = true;
                texture = resolveTextureByName(texName);
            }
        }
        void setTexture(const std::shared_ptr<Texture>& tex) {
            texture = tex;
            if (tex) {
                strncpy(texName, tex->name.c_str(), sizeof(texName) - 1);
                texName[sizeof(texName) - 1] = '\0';
            } else {
                texName[0] = '\0';
            }
            resolveTried = true;
            thumbFor_ = nullptr;   // rebuild the preview against the new image
            dirty = true;
        }

        // ---- on-node thumbnail ------------------------------------------------
        // The procedural nodes all preview themselves; the Image Texture node showed only a
        // file NAME, which is the one node where you cannot tell from the name whether you
        // picked the albedo or the roughness map.
        //
        // Sampled on the CPU through the same get_color_bilinear the render uses (no GPU
        // upload, no SDL_Texture, nothing to free), but CACHED: re-sampling the grid every
        // frame would be ~1k bilinear fetches per visible node per frame. Rebuilt only when
        // the bound texture changes.
        static constexpr int kThumbW = 44;
        static constexpr int kThumbH = 26;
        std::vector<float> thumb_;            ///< kThumbW*kThumbH*3, row-major
        const Texture* thumbFor_ = nullptr;   ///< identity of the image thumb_ was built from

        void ensureThumb() {
            if (!texture) { thumb_.clear(); thumbFor_ = nullptr; return; }
            if (thumbFor_ == texture.get() && !thumb_.empty()) return;
            thumb_.assign(static_cast<size_t>(kThumbW) * kThumbH * 3, 0.0f);
            for (int y = 0; y < kThumbH; ++y) {
                for (int x = 0; x < kThumbW; ++x) {
                    const float u = (x + 0.5f) / static_cast<float>(kThumbW);
                    // The preview samples with V flipped so the thumbnail reads the way the
                    // image file looks, not the way UV space runs.
                    const float v = 1.0f - (y + 0.5f) / static_cast<float>(kThumbH);
                    const Vec3 c = texture->get_color_bilinear(u, v);
                    const size_t i = (static_cast<size_t>(y) * kThumbW + x) * 3;
                    thumb_[i + 0] = static_cast<float>(c.x);
                    thumb_[i + 1] = static_cast<float>(c.y);
                    thumb_[i + 2] = static_cast<float>(c.z);
                }
            }
            thumbFor_ = texture.get();
        }
        void drawContent() override {
            resolveIfNeeded();
            auto baseName = [](const std::string& n) {
                const size_t slash = n.find_last_of("/\\");
                return (slash == std::string::npos) ? n : n.substr(slash + 1);
            };
            // A node that KEPT its reference but could not resolve it has to SAY so. The
            // plain "<select texture>" made a failed resolve look like the graph had lost
            // the texture, when the reference is still sitting right there in texName.
            const bool missing = (!texture && texName[0] != '\0');
            const std::string preview = missing ? ("<missing> " + baseName(texName))
                                               : (texture ? baseName(texture->name)
                                                          : std::string("<select texture>"));
            if (missing) ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.95f, 0.45f, 0.35f, 1.0f));
            ImGui::SetNextItemWidth(160);
            if (ImGui::BeginCombo("##imgtex", preview.c_str())) {
                // Session textures â€” the whole point: everything already loaded is one click away.
                for (const auto& st : collectSessionTextures()) {
                    const bool selected = (texture == st.tex);
                    if (ImGui::Selectable(st.displayName.c_str(), selected)) setTexture(st.tex);
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", st.name.c_str());
                }
                ImGui::EndCombo();
            }
            if (missing) {
                ImGui::PopStyleColor();
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Not found: %s", texName);
            }
            if (ImGui::SmallButton("Browse...")) browseRequested = true;
            if (missing) {
                // resolveIfNeeded latches after one attempt (probing the disk per sample
                // during a render would be fatal), so a retry needs an explicit nudge.
                ImGui::SameLine();
                if (ImGui::SmallButton("Retry")) {
                    resolveTried = false;
                    resolveIfNeeded();
                    dirty = true;
                }
            }
            if (texture) {
                ImGui::SameLine();
                if (ImGui::SmallButton("Clear")) setTexture(nullptr);
            }
            if (texture) {
                ensureThumb();
                if (!thumb_.empty()) {
                    drawPreviewGrid(150.0f, 88.0f, kThumbW, kThumbH,
                        [this](float u, float v, float& r, float& g, float& b) {
                            const int x = std::min(kThumbW - 1, static_cast<int>(u * kThumbW));
                            const int y = std::min(kThumbH - 1, static_cast<int>(v * kThumbH));
                            const size_t i = (static_cast<size_t>(y) * kThumbW + x) * 3;
                            r = thumb_[i]; g = thumb_[i + 1]; b = thumb_[i + 2];
                        });
                }
            }
        }
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override {
            resolveIfNeeded();
            const std::array<float, 2> uv = getUVIn(0, ctx);
            if (!texture) {
                if (outputIndex == 1) return 1.0f;
                return std::array<float, 3>{ 0.8f, 0.0f, 0.8f };  // magenta = missing texture
            }
            if (outputIndex == 1) {
                return texture->has_alpha ? texture->get_alpha_bilinear(uv[0], uv[1]) : 1.0f;
            }
            const Vec3 c = texture->get_color_bilinear(uv[0], uv[1]);
            return std::array<float, 3>{ static_cast<float>(c.x), static_cast<float>(c.y), static_cast<float>(c.z) };
        }
    };

    /**
     * @brief Unified procedural texture: ONE node with a Type combo â€” FBM /
     * Ridge / Billow / Warped / Voronoi / Checker (same vocabulary as the
     * terrain Noise Generator), all built on the shared hash-based helpers
     * above so every variant stays point-samplable and ports 1:1 to the Faz 2
     * GLSL/CUDA interpreter. Replaces the former separate Voronoi/Checker
     * nodes: their typeIds are registered as presets of this node
     * (MaterialNodesV2.cpp) and old Checker OUTPUT links â€” which were
     * [Color, Fac] instead of [Fac, Color] â€” are remapped on load
     * (deserializeMaterialGraph, pin_version 3).
     */
    class NoiseTextureNode : public MaterialNodeBase {
    public:
        enum class Kind { FBM = 0, Ridge, Billow, Warped, Voronoi, Checker };

        Kind  kind = Kind::FBM;
        float scale = 5.0f;
        int   detail = 3;          ///< fbm octaves
        float rough = 0.5f;        ///< fbm gain
        float randomness = 1.0f;   ///< Voronoi jitter
        float distortion = 0.5f;   ///< Warped domain-warp strength
        // Fac 0 / Fac 1 colors used by the Color output when the Color1/Color2
        // pins are unconnected. Black->white keeps the pre-merge Noise node's
        // grayscale Color output bit-identical.
        float color1[3] = { 0.0f, 0.0f, 0.0f };
        float color2[3] = { 1.0f, 1.0f, 1.0f };
        int   seed = 0;
        int   dimensions = 2;   ///< 2 = UV-driven (2D), 3 = position-driven (seamless solid)
        /// 3D mode only: drive the noise from the OBJECT-space shading point instead of the
        /// world one. World is the default so existing scenes are untouched — but a
        /// world-driven solid noise SWIMS: the pattern is nailed to the world, so moving or
        /// rotating the object slides the marble/wood through it. Object space nails the
        /// pattern to the object, which is what you want for a carved material and what
        /// makes a scattered rock keep its own veining.
        bool  objectSpace = false;

        NoiseTextureNode() {
            name = "Noise Texture";
            materialNodeType = NodeType::Noise;
            inputs.push_back(NodeSystem::Pin::createInput("Vector", NodeSystem::DataType::Vector2,
                                                          NodeSystem::ImageSemantic::Generic, true));
            inputs.push_back(NodeSystem::Pin::createInput("Color1", NodeSystem::DataType::Vector3,
                                                          NodeSystem::ImageSemantic::Generic, true));
            inputs.push_back(NodeSystem::Pin::createInput("Color2", NodeSystem::DataType::Vector3,
                                                          NodeSystem::ImageSemantic::Generic, true));
            outputs.push_back(NodeSystem::Pin::createOutput("Fac", NodeSystem::DataType::Float));
            outputs.push_back(NodeSystem::Pin::createOutput("Color", NodeSystem::DataType::Vector3));
            metadata.displayName = "Noise Texture";
            metadata.category = "Texture";
            metadata.description = "Procedural texture â€” FBM/Ridge/Billow/Warped/Voronoi/Checker. Shaded per-pixel on CPU + Vulkan RT (2D UV or 3D world-position / seamless solid).";
            metadata.headerColor = HeaderColors::texture();
        }
        std::string getTypeId() const override { return "MatV2.Noise"; }
        float getCustomWidth() const override { return 190.0f; }  // inline preview/widgets fit

        /// Scalar field shared by compute() and the preview. x/y are UV
        /// (scale is applied here). outCellHash: Voronoi winning-cell hash.
        float sampleFac(float x, float y, uint32_t* outCellHash = nullptr) const {
            // Per-kind seed constants preserved from the pre-merge nodes so
            // existing graphs keep their exact pattern per seed value.
            const uint32_t s = (kind == Kind::Voronoi)
                ? static_cast<uint32_t>(seed) * 0x68bc21ebu + 0x2545F491u
                : static_cast<uint32_t>(seed) * 0x51633e2du + 0x9E3779B9u;
            switch (kind) {
                case Kind::Ridge:  return ridge2D(x * scale, y * scale, detail, rough, s);
                case Kind::Billow: return billow2D(x * scale, y * scale, detail, rough, s);
                case Kind::Warped: {
                    const float wx = fbm2D(x * scale + 17.3f, y * scale + 9.1f, detail, rough, s ^ 0xA511u) - 0.5f;
                    const float wy = fbm2D(x * scale - 11.7f, y * scale + 4.9f, detail, rough, s ^ 0x3C6Fu) - 0.5f;
                    return fbm2D(x * scale + wx * distortion * 8.0f,
                                 y * scale + wy * distortion * 8.0f, detail, rough, s);
                }
                case Kind::Voronoi:
                    return std::clamp(voronoiF1(x * scale, y * scale, randomness, s, outCellHash), 0.0f, 1.0f);
                case Kind::Checker: {
                    const int cx = static_cast<int>(std::floor(x * scale));
                    const int cy = static_cast<int>(std::floor(y * scale));
                    return (((cx + cy) & 1) != 0) ? 1.0f : 0.0f;
                }
                case Kind::FBM:
                default:
                    return fbm2D(x * scale, y * scale, detail, rough, s);
            }
        }

        /// 3D variant (world-position driven) â€” seamless solid texturing.
        float sampleFac3D(float x, float y, float z, uint32_t* outCellHash = nullptr) const {
            const uint32_t s = (kind == Kind::Voronoi)
                ? static_cast<uint32_t>(seed) * 0x68bc21ebu + 0x2545F491u
                : static_cast<uint32_t>(seed) * 0x51633e2du + 0x9E3779B9u;
            const float xs = x * scale, ys = y * scale, zs = z * scale;
            switch (kind) {
                case Kind::Ridge:  return ridge3D(xs, ys, zs, detail, rough, s);
                case Kind::Billow: return billow3D(xs, ys, zs, detail, rough, s);
                case Kind::Warped: {
                    const float wx = fbm3D(xs + 17.3f, ys + 9.1f, zs + 3.7f, detail, rough, s ^ 0xA511u) - 0.5f;
                    const float wy = fbm3D(xs - 11.7f, ys + 4.9f, zs - 8.2f, detail, rough, s ^ 0x3C6Fu) - 0.5f;
                    const float wz = fbm3D(xs + 5.1f, ys - 6.3f, zs + 12.9f, detail, rough, s ^ 0x77A5u) - 0.5f;
                    return fbm3D(xs + wx * distortion * 8.0f, ys + wy * distortion * 8.0f,
                                 zs + wz * distortion * 8.0f, detail, rough, s);
                }
                case Kind::Voronoi:
                    return std::clamp(voronoi3D_F1(xs, ys, zs, randomness, s, outCellHash), 0.0f, 1.0f);
                case Kind::Checker: {
                    const int cx = static_cast<int>(std::floor(xs));
                    const int cy = static_cast<int>(std::floor(ys));
                    const int cz = static_cast<int>(std::floor(zs));
                    return (((cx + cy + cz) & 1) != 0) ? 1.0f : 0.0f;
                }
                case Kind::FBM:
                default:
                    return fbm3D(xs, ys, zs, detail, rough, s);
            }
        }

        void serializeParams(nlohmann::json& j) const override {
            j["ntype"] = static_cast<int>(kind);
            j["scale"] = scale; j["detail"] = detail; j["rough"] = rough;
            j["randomness"] = randomness; j["distortion"] = distortion; j["seed"] = seed;
            j["dims"] = dimensions;
            j["objspace"] = objectSpace;
            j["c1"] = { color1[0], color1[1], color1[2] };
            j["c2"] = { color2[0], color2[1], color2[2] };
        }
        void deserializeParams(const nlohmann::json& j) override {
            // No "ntype" key = pre-merge save: KEEP the factory-preset kind
            // (the "MatV2.Voronoi"/"MatV2.Checker" legacy factories preset it).
            kind = static_cast<Kind>(j.value("ntype", static_cast<int>(kind)));
            scale = j.value("scale", scale);
            detail = j.value("detail", 3);
            rough = j.value("rough", 0.5f);
            randomness = j.value("randomness", 1.0f);
            distortion = j.value("distortion", 0.5f);
            seed = j.value("seed", 0);
            dimensions = j.value("dims", 2);
            objectSpace = j.value("objspace", false);   // absent => world (old saves unchanged)
            if (j.contains("c1") && j["c1"].is_array() && j["c1"].size() >= 3)
                for (int i = 0; i < 3; ++i) color1[i] = j["c1"][i].get<float>();
            if (j.contains("c2") && j["c2"].is_array() && j["c2"].size() >= 3)
                for (int i = 0; i < 3; ++i) color2[i] = j["c2"][i].get<float>();
        }

        void drawContent() override {
            // Preview uses the node's fallback colors (wired inputs override at render).
            drawPreviewGrid(150.0f, 60.0f, 50, 20, [&](float u, float v, float& r, float& g, float& b) {
                uint32_t cellHash = 0;
                // 3D preview shows a Z=0 slice of the solid noise (representative).
                const float f = (dimensions == 3) ? sampleFac3D(u, v * 0.4f, 0.0f, &cellHash)
                                                   : sampleFac(u, v * 0.4f, &cellHash);  // preview is 2.5:1
                if (kind == Kind::Voronoi) {
                    r = static_cast<float>(pcgHash(cellHash ^ 0x11u)) * (1.0f / 4294967295.0f);
                    g = static_cast<float>(pcgHash(cellHash ^ 0x22u)) * (1.0f / 4294967295.0f);
                    b = static_cast<float>(pcgHash(cellHash ^ 0x33u)) * (1.0f / 4294967295.0f);
                } else {
                    r = color1[0] + (color2[0] - color1[0]) * f;
                    g = color1[1] + (color2[1] - color1[1]) * f;
                    b = color1[2] + (color2[2] - color1[2]) * f;
                }
            });
            static const char* kindNames[] = { "FBM Noise", "Ridge", "Billow", "Warped", "Voronoi", "Checker" };
            int ki = static_cast<int>(kind);
            ImGui::SetNextItemWidth(120);
            if (ImGui::Combo("Type", &ki, kindNames, 6)) { kind = static_cast<Kind>(ki); dirty = true; }
            ImGui::SetNextItemWidth(120);
            if (ImGui::DragFloat("Scale", &scale, 0.05f, 0.01f, 1000.0f)) dirty = true;
            {
                const char* dimNames[] = { "2D (UV)", "3D (Position)" };
                int di = (dimensions == 3) ? 1 : 0;
                ImGui::SetNextItemWidth(120);
                if (ImGui::Combo("Space", &di, dimNames, 2)) { dimensions = di ? 3 : 2; dirty = true; }
                if (dimensions == 3) {
                    if (ImGui::Checkbox("Object Space", &objectSpace)) dirty = true;
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip("World-space solid noise is nailed to the WORLD, so the\n"
                                          "pattern swims through the object when it moves.\n"
                                          "Object Space nails it to the object instead.");
                }
            }
            if (kind != Kind::Voronoi && kind != Kind::Checker) {
                ImGui::SetNextItemWidth(120);
                if (ImGui::SliderInt("Detail", &detail, 1, 8)) dirty = true;
                ImGui::SetNextItemWidth(120);
                if (ImGui::SliderFloat("Roughness", &rough, 0.0f, 1.0f)) dirty = true;
            }
            if (kind == Kind::Warped) {
                ImGui::SetNextItemWidth(120);
                if (ImGui::SliderFloat("Distortion", &distortion, 0.0f, 2.0f)) dirty = true;
            }
            if (kind == Kind::Voronoi) {
                ImGui::SetNextItemWidth(120);
                if (ImGui::SliderFloat("Randomness", &randomness, 0.0f, 1.0f)) dirty = true;
            }
            if (kind != Kind::Checker) {
                ImGui::SetNextItemWidth(120);
                if (ImGui::DragInt("Seed", &seed, 1)) dirty = true;
            }
            if (kind != Kind::Voronoi) {  // Voronoi's Color output is per-cell random
                if (ImGui::ColorEdit3("##c1", color1, ImGuiColorEditFlags_NoInputs)) dirty = true;
                ImGui::SameLine();
                if (ImGui::ColorEdit3("##c2", color2, ImGuiColorEditFlags_NoInputs)) dirty = true;
                ImGui::SameLine();
                ImGui::TextUnformatted("Colors");
            }
        }

        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override {
            uint32_t cellHash = 0;
            float fac;
            if (dimensions == 3) {
                // World-position driven (seamless). A wired Vector3 overrides the
                // hit position; otherwise use the shading point from the context.
                const auto* mctx = getMaterialContext(ctx);
                std::array<float, 3> p = getVec3In(0, ctx,
                    mctx ? std::array<float, 3>{ mctx->px, mctx->py, mctx->pz }
                         : std::array<float, 3>{ 0.5f, 0.5f, 0.0f });
                fac = sampleFac3D(p[0], p[1], p[2], &cellHash);
            } else {
                const std::array<float, 2> uv = getUVIn(0, ctx);
                fac = sampleFac(uv[0], uv[1], &cellHash);
            }
            if (outputIndex == 0) return fac;
            if (kind == Kind::Voronoi) {
                // Random per-cell color, kept from the pre-merge Voronoi node.
                return std::array<float, 3>{
                    static_cast<float>(pcgHash(cellHash ^ 0x11u)) * (1.0f / 4294967295.0f),
                    static_cast<float>(pcgHash(cellHash ^ 0x22u)) * (1.0f / 4294967295.0f),
                    static_cast<float>(pcgHash(cellHash ^ 0x33u)) * (1.0f / 4294967295.0f)
                };
            }
            const std::array<float, 3> c1 = getVec3In(1, ctx, { color1[0], color1[1], color1[2] });
            const std::array<float, 3> c2 = getVec3In(2, ctx, { color2[0], color2[1], color2[2] });
            return std::array<float, 3>{
                c1[0] + (c2[0] - c1[0]) * fac,
                c1[1] + (c2[1] - c1[1]) * fac,
                c1[2] + (c2[2] - c1[2]) * fac
            };
        }
    };

    // ============================================================================
    // VECTOR NODES
    // ============================================================================

    class MappingNode : public MaterialNodeBase {
    public:
        float scale[2]  = { 1.0f, 1.0f };
        float offset[2] = { 0.0f, 0.0f };
        float rotationDeg = 0.0f;

        MappingNode() {
            name = "Mapping";
            materialNodeType = NodeType::Mapping;
            inputs.push_back(NodeSystem::Pin::createInput("Vector", NodeSystem::DataType::Vector2,
                                                          NodeSystem::ImageSemantic::Generic, true));
            outputs.push_back(NodeSystem::Pin::createOutput("Vector", NodeSystem::DataType::Vector2));
            metadata.displayName = "Mapping";
            metadata.category = "Vector";
            metadata.description = "UV transform. Feeding an Image Texture binds as the material's TextureTransform on Apply.";
            metadata.headerColor = HeaderColors::vector();
        }
        std::string getTypeId() const override { return "MatV2.Mapping"; }
        void serializeParams(nlohmann::json& j) const override {
            j["scale"]  = { scale[0], scale[1] };
            j["offset"] = { offset[0], offset[1] };
            j["rot"]    = rotationDeg;
        }
        void deserializeParams(const nlohmann::json& j) override {
            if (j.contains("scale") && j["scale"].is_array() && j["scale"].size() >= 2) {
                scale[0] = j["scale"][0].get<float>();
                scale[1] = j["scale"][1].get<float>();
            }
            if (j.contains("offset") && j["offset"].is_array() && j["offset"].size() >= 2) {
                offset[0] = j["offset"][0].get<float>();
                offset[1] = j["offset"][1].get<float>();
            }
            rotationDeg = j.value("rot", 0.0f);
        }
        void drawContent() override {
            ImGui::SetNextItemWidth(120);
            if (ImGui::DragFloat2("Scale", scale, 0.01f)) dirty = true;
            ImGui::SetNextItemWidth(120);
            if (ImGui::DragFloat2("Offset", offset, 0.01f)) dirty = true;
            ImGui::SetNextItemWidth(120);
            if (ImGui::DragFloat("Rotation", &rotationDeg, 0.5f, -360.0f, 360.0f)) dirty = true;
        }
        NodeSystem::PinValue compute(int, NodeSystem::EvaluationContext& ctx) override {
            // Editor-preview approximation of PrincipledBSDF::applyTextureTransform
            // (rotate around UV center, then scale, then offset).
            const std::array<float, 2> uv = getUVIn(0, ctx);
            const float rad = rotationDeg * 3.14159265f / 180.0f;
            const float cr = std::cos(rad), sr = std::sin(rad);
            const float cx = uv[0] - 0.5f, cy = uv[1] - 0.5f;
            const float rx = cx * cr - cy * sr;
            const float ry = cx * sr + cy * cr;
            return std::array<float, 2>{ (rx + 0.5f) * scale[0] + offset[0],
                                         (ry + 0.5f) * scale[1] + offset[1] };
        }
    };

    // ============================================================================
    // COLOR NODES
    // ============================================================================

    class MixColorNode : public MaterialNodeBase {
    public:
        enum class Mode { Mix = 0, Add, Multiply, Subtract, Screen, Overlay };
        int mode = 0;

        MixColorNode() {
            name = "Mix Color";
            materialNodeType = NodeType::MixColor;
            {
                NodeSystem::Pin fac = NodeSystem::Pin::createInput("Fac", NodeSystem::DataType::Float,
                                                                   NodeSystem::ImageSemantic::Generic, true);
                fac.defaultValue = 0.5f;
                inputs.push_back(fac);
            }
            inputs.push_back(NodeSystem::Pin::createInput("Color1", NodeSystem::DataType::Vector3,
                                                          NodeSystem::ImageSemantic::Generic, true));
            inputs.push_back(NodeSystem::Pin::createInput("Color2", NodeSystem::DataType::Vector3,
                                                          NodeSystem::ImageSemantic::Generic, true));
            outputs.push_back(NodeSystem::Pin::createOutput("Color", NodeSystem::DataType::Vector3));
            metadata.displayName = "Mix Color";
            metadata.category = "Color";
            metadata.description = "Blend two colors (Mix/Add/Multiply/Subtract/Screen/Overlay).";
            metadata.headerColor = HeaderColors::color();
        }
        std::string getTypeId() const override { return "MatV2.MixColor"; }
        void serializeParams(nlohmann::json& j) const override { j["mode"] = mode; }
        void deserializeParams(const nlohmann::json& j) override { mode = j.value("mode", 0); }
        void drawContent() override {
            const char* modes[] = { "Mix", "Add", "Multiply", "Subtract", "Screen", "Overlay" };
            ImGui::SetNextItemWidth(120);
            if (ImGui::Combo("Mode", &mode, modes, IM_ARRAYSIZE(modes))) dirty = true;
        }
        NodeSystem::PinValue compute(int, NodeSystem::EvaluationContext& ctx) override {
            const float fac = std::clamp(getFloatIn(0, ctx, 0.5f), 0.0f, 1.0f);
            const std::array<float, 3> a = getVec3In(1, ctx, { 0.5f, 0.5f, 0.5f });
            const std::array<float, 3> b = getVec3In(2, ctx, { 0.5f, 0.5f, 0.5f });
            std::array<float, 3> r{};
            for (int i = 0; i < 3; ++i) {
                float m;
                switch (static_cast<Mode>(mode)) {
                    case Mode::Add:      m = a[i] + b[i]; break;
                    case Mode::Multiply: m = a[i] * b[i]; break;
                    case Mode::Subtract: m = a[i] - b[i]; break;
                    case Mode::Screen:   m = 1.0f - (1.0f - a[i]) * (1.0f - b[i]); break;
                    case Mode::Overlay:  m = (a[i] < 0.5f) ? 2.0f * a[i] * b[i]
                                                           : 1.0f - 2.0f * (1.0f - a[i]) * (1.0f - b[i]); break;
                    case Mode::Mix:
                    default:             m = b[i]; break;
                }
                r[i] = a[i] + (m - a[i]) * fac;
            }
            return r;
        }
    };

    class ColorRampNode : public MaterialNodeBase {
    public:
        struct Stop { float pos; float col[3]; };
        std::vector<Stop> stops = { { 0.0f, { 0.0f, 0.0f, 0.0f } }, { 1.0f, { 1.0f, 1.0f, 1.0f } } };
        int interpolation = 0;  ///< 0 = Linear, 1 = Constant
        static constexpr int kMaxStops = 8;

        ColorRampNode() {
            name = "Color Ramp";
            materialNodeType = NodeType::ColorRamp;
            {
                NodeSystem::Pin fac = NodeSystem::Pin::createInput("Fac", NodeSystem::DataType::Float,
                                                                   NodeSystem::ImageSemantic::Generic, true);
                fac.defaultValue = 0.5f;
                inputs.push_back(fac);
            }
            outputs.push_back(NodeSystem::Pin::createOutput("Color", NodeSystem::DataType::Vector3));
            metadata.displayName = "Color Ramp";
            metadata.category = "Color";
            metadata.description = "Map a factor onto a color gradient.";
            metadata.headerColor = HeaderColors::color();
        }
        std::string getTypeId() const override { return "MatV2.ColorRamp"; }
        float getCustomWidth() const override { return 200.0f; }  // inline preview/widgets fit
        void serializeParams(nlohmann::json& j) const override {
            j["interp"] = interpolation;
            nlohmann::json js = nlohmann::json::array();
            for (const auto& s : stops) js.push_back({ s.pos, s.col[0], s.col[1], s.col[2] });
            j["stops"] = std::move(js);
        }
        void deserializeParams(const nlohmann::json& j) override {
            interpolation = j.value("interp", 0);
            if (j.contains("stops") && j["stops"].is_array() && !j["stops"].empty()) {
                stops.clear();
                for (const auto& js : j["stops"]) {
                    if (!js.is_array() || js.size() < 4) continue;
                    Stop s;
                    s.pos = js[0].get<float>();
                    s.col[0] = js[1].get<float>();
                    s.col[1] = js[2].get<float>();
                    s.col[2] = js[3].get<float>();
                    stops.push_back(s);
                    if (static_cast<int>(stops.size()) >= kMaxStops) break;
                }
                if (stops.empty()) stops = { { 0.0f, { 0, 0, 0 } }, { 1.0f, { 1, 1, 1 } } };
            }
        }
        // Interactive gradient editor state (UI-only, not serialized).
        // dragOwner: drawContent renders BOTH inline (node body) and in the
        // properties panel â€” the drag must follow the bar it started on, or the
        // two instances would fight over the stop with different mouse-t spaces.
        int selectedStop = 0;
        int draggingStop = -1;
        ImGuiID dragOwner = 0;

        /// Sample the ramp at t (handles unsorted stops; used by the gradient bar)
        void sampleStops(float t, float out[3]) const {
            std::vector<Stop> sorted = stops;
            std::sort(sorted.begin(), sorted.end(), [](const Stop& a, const Stop& b) { return a.pos < b.pos; });
            if (sorted.empty()) { out[0] = out[1] = out[2] = 0.0f; return; }
            if (t <= sorted.front().pos) { std::copy(sorted.front().col, sorted.front().col + 3, out); return; }
            if (t >= sorted.back().pos) { std::copy(sorted.back().col, sorted.back().col + 3, out); return; }
            for (size_t i = 0; i + 1 < sorted.size(); ++i) {
                if (t <= sorted[i + 1].pos) {
                    if (interpolation == 1) { std::copy(sorted[i].col, sorted[i].col + 3, out); return; }
                    const float span = sorted[i + 1].pos - sorted[i].pos;
                    const float f = (span > 1e-6f) ? (t - sorted[i].pos) / span : 0.0f;
                    for (int k = 0; k < 3; ++k) out[k] = sorted[i].col[k] + (sorted[i + 1].col[k] - sorted[i].col[k]) * f;
                    return;
                }
            }
            std::copy(sorted.back().col, sorted.back().col + 3, out);
        }

        // VDB-panel-style interactive gradient bar: click empty = add stop,
        // click marker = select, drag marker = move; per-node state so multiple
        // ramp nodes coexist (the VDB original used function statics).
        void drawContent() override {
            ImDrawList* dl = ImGui::GetWindowDrawList();
            const float width = 160.0f;
            const float barH = 16.0f;
            const float markerSize = 6.0f;

            const ImVec2 p = ImGui::GetCursorScreenPos();
            ImGui::InvisibleButton("##rampbar", ImVec2(width, barH + markerSize * 2.0f));
            const ImGuiID barId = ImGui::GetItemID();
            const bool clicked = ImGui::IsItemClicked(ImGuiMouseButton_Left);
            const ImVec2 mouse = ImGui::GetIO().MousePos;
            const float mouseT = std::clamp((mouse.x - p.x) / width, 0.0f, 1.0f);

            if (clicked) {
                bool hit = false;
                for (int i = 0; i < static_cast<int>(stops.size()); ++i) {
                    if (std::fabs(p.x + stops[i].pos * width - mouse.x) < 8.0f) {
                        selectedStop = i;
                        draggingStop = i;
                        dragOwner = barId;
                        hit = true;
                        break;
                    }
                }
                if (!hit && static_cast<int>(stops.size()) < kMaxStops) {
                    Stop s;
                    s.pos = mouseT;
                    sampleStops(mouseT, s.col);
                    stops.push_back(s);
                    std::sort(stops.begin(), stops.end(), [](const Stop& a, const Stop& b) { return a.pos < b.pos; });
                    for (int i = 0; i < static_cast<int>(stops.size()); ++i) {
                        if (stops[i].pos == s.pos) { selectedStop = i; draggingStop = i; dragOwner = barId; break; }
                    }
                    dirty = true;
                }
            }
            if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
                draggingStop = -1;
                dragOwner = 0;
            } else if (draggingStop >= 0 && draggingStop < static_cast<int>(stops.size()) && dragOwner == barId) {
                stops[draggingStop].pos = mouseT;
                std::sort(stops.begin(), stops.end(), [](const Stop& a, const Stop& b) { return a.pos < b.pos; });
                for (int i = 0; i < static_cast<int>(stops.size()); ++i) {
                    if (stops[i].pos == mouseT) { draggingStop = i; selectedStop = i; break; }
                }
                dirty = true;
            }

            // Gradient bar
            const int columns = static_cast<int>(width);
            for (int i = 0; i < columns; ++i) {
                float c[3];
                sampleStops(static_cast<float>(i) / width, c);
                dl->AddRectFilled(ImVec2(p.x + i, p.y), ImVec2(p.x + i + 1, p.y + barH),
                    IM_COL32(static_cast<int>(std::clamp(c[0], 0.0f, 1.0f) * 255.0f),
                             static_cast<int>(std::clamp(c[1], 0.0f, 1.0f) * 255.0f),
                             static_cast<int>(std::clamp(c[2], 0.0f, 1.0f) * 255.0f), 255));
            }
            dl->AddRect(p, ImVec2(p.x + width, p.y + barH), IM_COL32(70, 70, 80, 255));
            // Stop markers
            for (int i = 0; i < static_cast<int>(stops.size()); ++i) {
                const float x = p.x + stops[i].pos * width;
                dl->AddTriangleFilled(
                    ImVec2(x, p.y + barH + markerSize * 2.0f),
                    ImVec2(x - markerSize, p.y + barH),
                    ImVec2(x + markerSize, p.y + barH),
                    (i == selectedStop) ? IM_COL32(255, 220, 60, 255) : IM_COL32(235, 235, 235, 255));
            }

            if (selectedStop >= 0 && selectedStop < static_cast<int>(stops.size())) {
                ImGui::SetNextItemWidth(90);
                if (ImGui::SliderFloat("##stoppos", &stops[selectedStop].pos, 0.0f, 1.0f, "%.3f")) dirty = true;
                ImGui::SameLine();
                if (ImGui::ColorEdit3("##stopcol", stops[selectedStop].col, ImGuiColorEditFlags_NoInputs)) dirty = true;
                if (stops.size() > 2) {
                    ImGui::SameLine();
                    if (ImGui::SmallButton("x")) {
                        stops.erase(stops.begin() + selectedStop);
                        selectedStop = std::min(selectedStop, static_cast<int>(stops.size()) - 1);
                        dirty = true;
                    }
                }
            }
            const char* interps[] = { "Linear", "Constant" };
            ImGui::SetNextItemWidth(100);
            if (ImGui::Combo("##interp", &interpolation, interps, IM_ARRAYSIZE(interps))) dirty = true;
        }
        NodeSystem::PinValue compute(int, NodeSystem::EvaluationContext& ctx) override {
            const float fac = std::clamp(getFloatIn(0, ctx, 0.5f), 0.0f, 1.0f);
            std::vector<Stop> sorted = stops;
            std::sort(sorted.begin(), sorted.end(), [](const Stop& a, const Stop& b) { return a.pos < b.pos; });
            if (sorted.empty()) return std::array<float, 3>{ 0, 0, 0 };
            if (fac <= sorted.front().pos) {
                const auto& c = sorted.front().col;
                return std::array<float, 3>{ c[0], c[1], c[2] };
            }
            if (fac >= sorted.back().pos) {
                const auto& c = sorted.back().col;
                return std::array<float, 3>{ c[0], c[1], c[2] };
            }
            for (size_t i = 0; i + 1 < sorted.size(); ++i) {
                if (fac <= sorted[i + 1].pos) {
                    if (interpolation == 1) {  // Constant
                        const auto& c = sorted[i].col;
                        return std::array<float, 3>{ c[0], c[1], c[2] };
                    }
                    const float span = sorted[i + 1].pos - sorted[i].pos;
                    const float t = (span > 1e-6f) ? (fac - sorted[i].pos) / span : 0.0f;
                    std::array<float, 3> r{};
                    for (int k = 0; k < 3; ++k) {
                        r[k] = sorted[i].col[k] + (sorted[i + 1].col[k] - sorted[i].col[k]) * t;
                    }
                    return r;
                }
            }
            const auto& c = sorted.back().col;
            return std::array<float, 3>{ c[0], c[1], c[2] };
        }
    };

    class InvertNode : public MaterialNodeBase {
    public:
        InvertNode() {
            name = "Invert";
            materialNodeType = NodeType::Invert;
            {
                NodeSystem::Pin fac = NodeSystem::Pin::createInput("Fac", NodeSystem::DataType::Float,
                                                                   NodeSystem::ImageSemantic::Generic, true);
                fac.defaultValue = 1.0f;
                inputs.push_back(fac);
            }
            inputs.push_back(NodeSystem::Pin::createInput("Color", NodeSystem::DataType::Vector3,
                                                          NodeSystem::ImageSemantic::Generic, true));
            outputs.push_back(NodeSystem::Pin::createOutput("Color", NodeSystem::DataType::Vector3));
            metadata.displayName = "Invert";
            metadata.category = "Color";
            metadata.description = "1 - color, blended by Fac.";
            metadata.headerColor = HeaderColors::color();
        }
        std::string getTypeId() const override { return "MatV2.Invert"; }
        NodeSystem::PinValue compute(int, NodeSystem::EvaluationContext& ctx) override {
            const float fac = std::clamp(getFloatIn(0, ctx, 1.0f), 0.0f, 1.0f);
            const std::array<float, 3> c = getVec3In(1, ctx, { 0.5f, 0.5f, 0.5f });
            return std::array<float, 3>{
                c[0] + ((1.0f - c[0]) - c[0]) * fac,
                c[1] + ((1.0f - c[1]) - c[1]) * fac,
                c[2] + ((1.0f - c[2]) - c[2]) * fac
            };
        }
    };

    class GammaNode : public MaterialNodeBase {
    public:
        float gamma = 1.0f;

        GammaNode() {
            name = "Gamma";
            materialNodeType = NodeType::Gamma;
            inputs.push_back(NodeSystem::Pin::createInput("Color", NodeSystem::DataType::Vector3,
                                                          NodeSystem::ImageSemantic::Generic, true));
            outputs.push_back(NodeSystem::Pin::createOutput("Color", NodeSystem::DataType::Vector3));
            metadata.displayName = "Gamma";
            metadata.category = "Color";
            metadata.description = "color ^ (1/gamma).";
            metadata.headerColor = HeaderColors::color();
        }
        std::string getTypeId() const override { return "MatV2.Gamma"; }
        void serializeParams(nlohmann::json& j) const override { j["gamma"] = gamma; }
        void deserializeParams(const nlohmann::json& j) override { gamma = j.value("gamma", 1.0f); }
        void drawContent() override {
            ImGui::SetNextItemWidth(120);
            if (ImGui::DragFloat("Gamma", &gamma, 0.01f, 0.01f, 10.0f)) dirty = true;
        }
        NodeSystem::PinValue compute(int, NodeSystem::EvaluationContext& ctx) override {
            const std::array<float, 3> c = getVec3In(0, ctx, { 0.5f, 0.5f, 0.5f });
            const float inv = 1.0f / std::max(gamma, 0.01f);
            return std::array<float, 3>{
                std::pow(std::max(c[0], 0.0f), inv),
                std::pow(std::max(c[1], 0.0f), inv),
                std::pow(std::max(c[2], 0.0f), inv)
            };
        }
    };

    // ============================================================================
    // CONVERT NODES
    // ============================================================================

    class MathNode : public MaterialNodeBase {
    public:
        // Serialized as an int, so NEW OPS ONLY EVER GET APPENDED — inserting one
        // would silently re-point every saved graph's Math node at a different op.
        // The VM (MaterialProgram.h) and the GLSL mirror switch on these same ids.
        enum class Op {
            Add = 0, Subtract, Multiply, Divide, Power, SquareRoot, Absolute, Minimum, Maximum, Clamp01,
            Sine, Cosine, Fraction, Floor, Ceil, Modulo, SmoothStep, GreaterThan, LessThan
        };
        int op = 0;
        float defaultA = 0.5f;
        float defaultB = 0.5f;

        MathNode() {
            name = "Math";
            materialNodeType = NodeType::Math;
            inputs.push_back(NodeSystem::Pin::createInput("A", NodeSystem::DataType::Float,
                                                          NodeSystem::ImageSemantic::Generic, true));
            inputs.push_back(NodeSystem::Pin::createInput("B", NodeSystem::DataType::Float,
                                                          NodeSystem::ImageSemantic::Generic, true));
            outputs.push_back(NodeSystem::Pin::createOutput("Value", NodeSystem::DataType::Float));
            metadata.displayName = "Math";
            metadata.category = "Convert";
            metadata.description = "Scalar math.";
            metadata.headerColor = HeaderColors::convert();
        }
        std::string getTypeId() const override { return "MatV2.Math"; }
        void serializeParams(nlohmann::json& j) const override {
            j["op"] = op; j["a"] = defaultA; j["b"] = defaultB;
        }
        void deserializeParams(const nlohmann::json& j) override {
            op = j.value("op", 0);
            defaultA = j.value("a", 0.5f);
            defaultB = j.value("b", 0.5f);
        }
        void drawContent() override {
            const char* ops[] = { "Add", "Subtract", "Multiply", "Divide", "Power",
                                  "Sqrt", "Abs", "Min", "Max", "Clamp 0-1",
                                  "Sine", "Cosine", "Fraction", "Floor", "Ceil",
                                  "Modulo", "Smooth Step", "Greater Than", "Less Than" };
            ImGui::SetNextItemWidth(120);
            if (ImGui::Combo("Op", &op, ops, IM_ARRAYSIZE(ops))) dirty = true;
            ImGui::SetNextItemWidth(120);
            if (ImGui::DragFloat("A", &defaultA, 0.01f)) dirty = true;
            ImGui::SetNextItemWidth(120);
            if (ImGui::DragFloat("B", &defaultB, 0.01f)) dirty = true;
        }
        NodeSystem::PinValue compute(int, NodeSystem::EvaluationContext& ctx) override {
            const float a = getFloatIn(0, ctx, defaultA);
            const float b = getFloatIn(1, ctx, defaultB);
            switch (static_cast<Op>(op)) {
                case Op::Add:        return a + b;
                case Op::Subtract:   return a - b;
                case Op::Multiply:   return a * b;
                case Op::Divide:     return (std::fabs(b) > 1e-8f) ? a / b : 0.0f;
                case Op::Power:      return std::pow(std::max(a, 0.0f), b);
                case Op::SquareRoot: return std::sqrt(std::max(a, 0.0f));
                case Op::Absolute:   return std::fabs(a);
                case Op::Minimum:    return std::min(a, b);
                case Op::Maximum:    return std::max(a, b);
                case Op::Clamp01:    return std::clamp(a, 0.0f, 1.0f);
                case Op::Sine:       return std::sin(a);
                case Op::Cosine:     return std::cos(a);
                case Op::Fraction:   return a - std::floor(a);
                case Op::Floor:      return std::floor(a);
                case Op::Ceil:       return std::ceil(a);
                case Op::Modulo:     return (std::fabs(b) > 1e-8f) ? a - b * std::floor(a / b) : 0.0f;
                case Op::SmoothStep: { const float t = std::clamp(a, 0.0f, 1.0f); return t * t * (3.0f - 2.0f * t); }
                case Op::GreaterThan: return (a > b) ? 1.0f : 0.0f;
                case Op::LessThan:    return (a < b) ? 1.0f : 0.0f;
            }
            return a;
        }
    };

    class SeparateColorNode : public MaterialNodeBase {
    public:
        SeparateColorNode() {
            name = "Separate RGB";
            materialNodeType = NodeType::SeparateColor;
            inputs.push_back(NodeSystem::Pin::createInput("Color", NodeSystem::DataType::Vector3,
                                                          NodeSystem::ImageSemantic::Generic, true));
            outputs.push_back(NodeSystem::Pin::createOutput("R", NodeSystem::DataType::Float));
            outputs.push_back(NodeSystem::Pin::createOutput("G", NodeSystem::DataType::Float));
            outputs.push_back(NodeSystem::Pin::createOutput("B", NodeSystem::DataType::Float));
            metadata.displayName = "Separate RGB";
            metadata.category = "Convert";
            metadata.description = "Split a color into channels (e.g. read one channel of a packed map).";
            metadata.headerColor = HeaderColors::convert();
        }
        std::string getTypeId() const override { return "MatV2.SeparateColor"; }
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override {
            const std::array<float, 3> c = getVec3In(0, ctx, { 0.0f, 0.0f, 0.0f });
            return c[std::clamp(outputIndex, 0, 2)];
        }
    };

    class CombineColorNode : public MaterialNodeBase {
    public:
        CombineColorNode() {
            name = "Combine RGB";
            materialNodeType = NodeType::CombineColor;
            inputs.push_back(NodeSystem::Pin::createInput("R", NodeSystem::DataType::Float,
                                                          NodeSystem::ImageSemantic::Generic, true));
            inputs.push_back(NodeSystem::Pin::createInput("G", NodeSystem::DataType::Float,
                                                          NodeSystem::ImageSemantic::Generic, true));
            inputs.push_back(NodeSystem::Pin::createInput("B", NodeSystem::DataType::Float,
                                                          NodeSystem::ImageSemantic::Generic, true));
            outputs.push_back(NodeSystem::Pin::createOutput("Color", NodeSystem::DataType::Vector3));
            metadata.displayName = "Combine RGB";
            metadata.category = "Convert";
            metadata.description = "Build a color from channels.";
            metadata.headerColor = HeaderColors::convert();
        }
        std::string getTypeId() const override { return "MatV2.CombineColor"; }
        NodeSystem::PinValue compute(int, NodeSystem::EvaluationContext& ctx) override {
            return std::array<float, 3>{
                getFloatIn(0, ctx, 0.0f),
                getFloatIn(1, ctx, 0.0f),
                getFloatIn(2, ctx, 0.0f)
            };
        }
    };

    /**
     * @brief Clamp a value between Min and Max. Compiles to two existing Math ops
     * (max then min) so the GPU VM runs it with no new opcode / shader recompile.
     */
    class ClampNode : public MaterialNodeBase {
    public:
        float minVal = 0.0f;
        float maxVal = 1.0f;

        ClampNode() {
            name = "Clamp";
            materialNodeType = NodeType::Clamp;
            inputs.push_back(NodeSystem::Pin::createInput("Value", NodeSystem::DataType::Float,
                                                          NodeSystem::ImageSemantic::Generic, true));
            inputs.push_back(NodeSystem::Pin::createInput("Min", NodeSystem::DataType::Float,
                                                          NodeSystem::ImageSemantic::Generic, true));
            inputs.push_back(NodeSystem::Pin::createInput("Max", NodeSystem::DataType::Float,
                                                          NodeSystem::ImageSemantic::Generic, true));
            outputs.push_back(NodeSystem::Pin::createOutput("Value", NodeSystem::DataType::Float));
            metadata.displayName = "Clamp";
            metadata.category = "Convert";
            metadata.description = "Constrain a value to the [Min, Max] range.";
            metadata.headerColor = HeaderColors::convert();
        }
        std::string getTypeId() const override { return "MatV2.Clamp"; }
        void serializeParams(nlohmann::json& j) const override { j["min"] = minVal; j["max"] = maxVal; }
        void deserializeParams(const nlohmann::json& j) override {
            minVal = j.value("min", 0.0f); maxVal = j.value("max", 1.0f);
        }
        void drawContent() override {
            ImGui::SetNextItemWidth(120);
            if (ImGui::DragFloat("Min", &minVal, 0.01f)) dirty = true;
            ImGui::SetNextItemWidth(120);
            if (ImGui::DragFloat("Max", &maxVal, 0.01f)) dirty = true;
        }
        NodeSystem::PinValue compute(int, NodeSystem::EvaluationContext& ctx) override {
            const float v  = getFloatIn(0, ctx, 0.0f);
            const float lo = getFloatIn(1, ctx, minVal);
            const float hi = getFloatIn(2, ctx, maxVal);
            return std::clamp(v, std::min(lo, hi), std::max(lo, hi));
        }
    };

    /**
     * @brief Remap a value from [From Min, From Max] to [To Min, To Max] (Blender's
     * Map Range, Linear). Optional clamp keeps the result inside the target range.
     * Compiles to a chain of existing Math ops â€” no new VM opcode / recompile.
     */
    class FloatCurveNode : public MaterialNodeBase {
    public:
        using Point = CurvePoint;   // shared with RGB Curves — one curve implementation
        std::vector<Point> points = { {0.0f, 0.0f}, {1.0f, 1.0f} };
        int interpolation = 0;  // 0 = Linear, 1 = Constant, 2 = Smooth (monotone cubic)

        /// Number of uniformly-spaced samples the compiler bakes the curve into.
        static constexpr int kLutSize = 32;

        /// Delegates to the shared implementation (evalCurvePoints) — kept as a static here
        /// because the compiler's LUT bake already calls it by this name.
        static float evalCurveOn(const std::vector<Point>& points, int interpolation, float x) {
            return evalCurvePoints(points, interpolation, x);
        }

        float evalCurve(float x) const { return evalCurveOn(points, interpolation, x); }

        FloatCurveNode() {
            name = "Float Curve";
            materialNodeType = NodeType::FloatCurve;
            {
                NodeSystem::Pin fac = NodeSystem::Pin::createInput("Fac", NodeSystem::DataType::Float,
                                                                   NodeSystem::ImageSemantic::Generic, true);
                fac.defaultValue = 0.5f;
                inputs.push_back(fac);
            }
            outputs.push_back(NodeSystem::Pin::createOutput("Value", NodeSystem::DataType::Float));
            metadata.displayName = "Float Curve";
            metadata.category = "Converter";
            metadata.description = "Map a float value to another float value via a curve.";
            metadata.headerColor = HeaderColors::convert();
        }

        std::string getTypeId() const override { return "MatV2.FloatCurve"; }
        float getCustomWidth() const override { return 180.0f; }

        void serializeParams(nlohmann::json& j) const override {
            j["interp"] = interpolation;
            nlohmann::json js = nlohmann::json::array();
            for (const auto& p : points) js.push_back({ p.x, p.y });
            j["points"] = std::move(js);
        }

        void deserializeParams(const nlohmann::json& j) override {
            interpolation = j.value("interp", 0);
            if (j.contains("points") && j["points"].is_array()) {
                points.clear();
                for (const auto& jp : j["points"]) {
                    if (jp.is_array() && jp.size() >= 2) {
                        points.push_back({ jp[0].get<float>(), jp[1].get<float>() });
                    }
                }
                if (points.empty()) points = { {0.0f, 0.0f}, {1.0f, 1.0f} };
            }
        }

        int selectedPoint = -1;
        int draggingPoint = -1;
        ImGuiID dragOwner = 0;   ///< see drawCurveWidget: the node body and the properties
                                 ///< panel both draw this, only one of them may drag

        void drawContent() override {
            // This node used to carry its OWN copy of the curve widget — same look, subtly
            // different behaviour, and without the drag-owner guard (hence points jumping to
            // the bottom when dragged from the properties panel). One widget now.
            const char* interps[] = { "Linear", "Constant", "Smooth" };
            ImGui::SetNextItemWidth(145);
            if (ImGui::Combo("##interp", &interpolation, interps, IM_ARRAYSIZE(interps))) dirty = true;
            if (drawCurveWidget(points, interpolation, 160.0f, 120.0f,
                                selectedPoint, draggingPoint, dragOwner)) {
                dirty = true;
            }
            if (drawCurvePointFields(points, selectedPoint, 70.0f)) dirty = true;
        }

        NodeSystem::PinValue compute(int, NodeSystem::EvaluationContext& ctx) override {
            const float fac = std::clamp(getFloatIn(0, ctx, 0.5f), 0.0f, 1.0f);
            return evalCurve(fac);
        }
    };


    class MapRangeNode : public MaterialNodeBase {
    public:
        float fromMin = 0.0f, fromMax = 1.0f;
        float toMin = 0.0f,   toMax = 1.0f;
        bool  clampResult = true;

        MapRangeNode() {
            name = "Map Range";
            materialNodeType = NodeType::MapRange;
            auto in = [this](const char* n) {
                inputs.push_back(NodeSystem::Pin::createInput(n, NodeSystem::DataType::Float,
                                                              NodeSystem::ImageSemantic::Generic, true));
            };
            in("Value"); in("From Min"); in("From Max"); in("To Min"); in("To Max");
            outputs.push_back(NodeSystem::Pin::createOutput("Value", NodeSystem::DataType::Float));
            metadata.displayName = "Map Range";
            metadata.category = "Convert";
            metadata.description = "Remap a value from one range to another (Linear).";
            metadata.headerColor = HeaderColors::convert();
        }
        std::string getTypeId() const override { return "MatV2.MapRange"; }
        void serializeParams(nlohmann::json& j) const override {
            j["from_min"] = fromMin; j["from_max"] = fromMax;
            j["to_min"] = toMin;     j["to_max"] = toMax;
            j["clamp"] = clampResult;
        }
        void deserializeParams(const nlohmann::json& j) override {
            fromMin = j.value("from_min", 0.0f); fromMax = j.value("from_max", 1.0f);
            toMin   = j.value("to_min", 0.0f);   toMax   = j.value("to_max", 1.0f);
            clampResult = j.value("clamp", true);
        }
        void drawContent() override {
            auto drag = [this](const char* label, float& v) {
                ImGui::SetNextItemWidth(120);
                if (ImGui::DragFloat(label, &v, 0.01f)) dirty = true;
            };
            drag("From Min", fromMin); drag("From Max", fromMax);
            drag("To Min", toMin);     drag("To Max", toMax);
            if (ImGui::Checkbox("Clamp", &clampResult)) dirty = true;
        }
        NodeSystem::PinValue compute(int, NodeSystem::EvaluationContext& ctx) override {
            const float v  = getFloatIn(0, ctx, 0.0f);
            const float fl = getFloatIn(1, ctx, fromMin);
            const float fh = getFloatIn(2, ctx, fromMax);
            const float tl = getFloatIn(3, ctx, toMin);
            const float th = getFloatIn(4, ctx, toMax);
            const float span = fh - fl;
            float t = (std::fabs(span) > 1e-8f) ? (v - fl) / span : 0.0f;
            float r = tl + t * (th - tl);
            if (clampResult) r = std::clamp(r, std::min(tl, th), std::max(tl, th));
            return r;
        }
    };

    /**
     * @brief Brightness / Contrast (Blender's formula): out = a*color + b, with
     * a = 1+contrast, b = brightness - contrast*0.5. Compiles to two MixColor ops
     * (scale then offset, fac=1) â€” no new VM opcode / shader recompile.
     */
    class BrightContrastNode : public MaterialNodeBase {
    public:
        float brightness = 0.0f;
        float contrast = 0.0f;

        BrightContrastNode() {
            name = "Bright/Contrast";
            materialNodeType = NodeType::BrightContrast;
            inputs.push_back(NodeSystem::Pin::createInput("Color", NodeSystem::DataType::Vector3,
                                                          NodeSystem::ImageSemantic::Generic, true));
            outputs.push_back(NodeSystem::Pin::createOutput("Color", NodeSystem::DataType::Vector3));
            metadata.displayName = "Bright/Contrast";
            metadata.category = "Color";
            metadata.description = "Adjust brightness and contrast of a color.";
            metadata.headerColor = HeaderColors::color();
        }
        std::string getTypeId() const override { return "MatV2.BrightContrast"; }
        void serializeParams(nlohmann::json& j) const override {
            j["brightness"] = brightness; j["contrast"] = contrast;
        }
        void deserializeParams(const nlohmann::json& j) override {
            brightness = j.value("brightness", 0.0f); contrast = j.value("contrast", 0.0f);
        }
        void drawContent() override {
            ImGui::SetNextItemWidth(120);
            if (ImGui::DragFloat("Bright", &brightness, 0.01f)) dirty = true;
            ImGui::SetNextItemWidth(120);
            if (ImGui::DragFloat("Contrast", &contrast, 0.01f)) dirty = true;
        }
        NodeSystem::PinValue compute(int, NodeSystem::EvaluationContext& ctx) override {
            const std::array<float, 3> c = getVec3In(0, ctx, { 0.0f, 0.0f, 0.0f });
            const float a = 1.0f + contrast;
            const float b = brightness - contrast * 0.5f;
            return std::array<float, 3>{
                std::max(a * c[0] + b, 0.0f),
                std::max(a * c[1] + b, 0.0f),
                std::max(a * c[2] + b, 0.0f)
            };
        }
    };

    /**
     * @brief Bump â€” turns a scalar HEIGHT field (a procedural chain: Noise, etc.)
     * into a perturbed shading normal, so procedural detail bumps the surface with
     * NO texture. Wire its Normal output into the Material Output's Normal Map slot.
     *
     * Per-pixel only: the compiler finite-differences the Height chain at UV and
     * UV+(dist,0)/UV+(0,dist), builds a TANGENT-space normal (-dh/du, -dh/dv, 1)*k
     * (k = strength/dist), and Stores it to MatSlot::Normal. apply_normal_map (CPU)
     * / closesthit (GPU) transform it through the mesh TBN. Strength & Distance are
     * node params (not pins) because the compiler bakes the UV offset as a constant.
     * Limitation: UV-space only â€” a purely position-driven (3D) height won't bump
     * yet (needs runtime world tangents; a later slice).
     */
    class BumpNode : public MaterialNodeBase {
    public:
        float strength = 1.0f;
        float distance = 0.01f;   ///< finite-difference UV step (and derivative scale)
        bool  invert = false;

        BumpNode() {
            name = "Bump";
            materialNodeType = NodeType::Bump;
            inputs.push_back(NodeSystem::Pin::createInput("Height", NodeSystem::DataType::Float,
                                                          NodeSystem::ImageSemantic::Generic, true));
            outputs.push_back(NodeSystem::Pin::createOutput("Normal", NodeSystem::DataType::Vector3));
            metadata.displayName = "Bump";
            metadata.category = "Vector";
            metadata.description = "Procedural height -> perturbed normal. Wire into the Output's Normal Map slot.";
            metadata.headerColor = HeaderColors::shader();
        }
        std::string getTypeId() const override { return "MatV2.Bump"; }
        void serializeParams(nlohmann::json& j) const override {
            j["strength"] = strength; j["distance"] = distance; j["invert"] = invert;
        }
        void deserializeParams(const nlohmann::json& j) override {
            strength = j.value("strength", 1.0f);
            distance = j.value("distance", 0.01f);
            invert   = j.value("invert", false);
        }
        void drawContent() override {
            ImGui::SetNextItemWidth(120);
            if (ImGui::DragFloat("Strength", &strength, 0.01f, 0.0f, 10.0f)) dirty = true;
            ImGui::SetNextItemWidth(120);
            if (ImGui::DragFloat("Distance", &distance, 0.001f, 0.0005f, 1.0f, "%.4f")) dirty = true;
            if (ImGui::Checkbox("Invert", &invert)) dirty = true;
        }
        // Per-pixel only â€” the fold/preview value is unused (the Normal slot is
        // consumed by apply_normal_map, never folded into a material param).
        NodeSystem::PinValue compute(int, NodeSystem::EvaluationContext&) override {
            return std::array<float, 3>{ 0.0f, 0.0f, 1.0f };  // flat tangent normal
        }
    };

    // ============================================================================
    // SHADER NODES
    // ============================================================================

    /**
     * @brief The multi-material bridge: blends TWO full materials by Fac.
     *
     * The engine has one uber-BSDF, so mixing materials means mixing PARAMETERS, not
     * closures — which is what Blender's Principled does internally anyway (a Fac that
     * drives Transmission from 0 to 1 does give you a diffuse-to-glass blend).
     *
     * Base Color / Roughness / Metallic / Emission blend per-pixel INCLUDING their
     * textures: the compiler lowers each side to a real texture fetch through that
     * material's own UV transform (MatOp::MatMapping) and lerps the two. Drive Fac with
     * a Noise, a mask texture, or Geometry > Pointiness and you get a genuine two-material
     * blend on CPU and Vulkan RT alike.
     *
     * Still on the fold, and therefore still a hard switch at Fac 0.5: the Opacity and
     * Normal Map textures, and the extended slots (clearcoat, subsurface, resin, ...)
     * which have no MatSlot and so cannot be written per-pixel.
     */
    class MixMaterialNode : public MaterialNodeBase {
    public:
        MixMaterialNode() {
            name = "Mix Material";
            materialNodeType = NodeType::MixMaterial;
            {
                NodeSystem::Pin fac = NodeSystem::Pin::createInput("Fac", NodeSystem::DataType::Float,
                                                                   NodeSystem::ImageSemantic::Generic, true);
                fac.defaultValue = 0.5f;
                inputs.push_back(fac);
            }
            inputs.push_back(NodeSystem::Pin::createInput("Material A", NodeSystem::DataType::Material,
                                                          NodeSystem::ImageSemantic::Generic, true));
            inputs.push_back(NodeSystem::Pin::createInput("Material B", NodeSystem::DataType::Material,
                                                          NodeSystem::ImageSemantic::Generic, true));
            outputs.push_back(NodeSystem::Pin::createOutput("Material", NodeSystem::DataType::Material));
            metadata.displayName = "Mix Material";
            metadata.category = "Shader";
            metadata.description = "Blend two full materials by Fac, per-pixel on CPU + Vulkan RT — drive Fac with a Noise/Pointiness mask. Base Color, Roughness, Metallic and Emission blend WITH their textures. Opacity/Normal textures and the extended slots (clearcoat, subsurface, resin) still switch at Fac 0.5.";
            metadata.headerColor = HeaderColors::shader();
        }
        std::string getTypeId() const override { return "MatV2.MixMaterial"; }

        /// set by the evaluator when A and B bind different textures on a slot
        bool lastMixHadTextureConflict = false;

        NodeSystem::PinValue compute(int, NodeSystem::EvaluationContext& ctx) override {
            const float fac = std::clamp(getFloatIn(0, ctx, 0.5f), 0.0f, 1.0f);
            NodeSystem::MaterialValue a, b;
            NodeSystem::tryGetMaterial(getInputValue(1, ctx), a);
            NodeSystem::tryGetMaterial(getInputValue(2, ctx), b);
            if (!a && !b) {
                ctx.addError(id, "Mix Material: neither input is connected");
                return NodeSystem::PinValue{};
            }
            if (!a) return b;
            if (!b) return a;

            auto out = std::make_shared<ShadeState>();
            auto lerpF = [fac](float x, float y) { return x + (y - x) * fac; };
            auto lerpC = [fac](const Vec3& x, const Vec3& y) { return x + (y - x) * fac; };

            out->baseColor          = lerpC(a->baseColor, b->baseColor);
            out->metallic           = lerpF(a->metallic, b->metallic);
            out->roughness          = lerpF(a->roughness, b->roughness);
            out->specular           = lerpF(a->specular, b->specular);
            out->emissionColor      = lerpC(a->emissionColor, b->emissionColor);
            out->emissionStrength   = lerpF(a->emissionStrength, b->emissionStrength);
            out->transmission       = lerpF(a->transmission, b->transmission);
            out->ior                = lerpF(a->ior, b->ior);
            out->opacity            = lerpF(a->opacity, b->opacity);
            out->translucent        = lerpF(a->translucent, b->translucent);
            out->clearcoat          = lerpF(a->clearcoat, b->clearcoat);
            out->clearcoatRoughness = lerpF(a->clearcoatRoughness, b->clearcoatRoughness);
            out->subsurface         = lerpF(a->subsurface, b->subsurface);
            out->subsurfaceColor    = lerpC(a->subsurfaceColor, b->subsurfaceColor);
            out->normalStrength     = lerpF(a->normalStrength, b->normalStrength);
            out->anisotropic        = lerpF(a->anisotropic, b->anisotropic);
            out->dispersion         = lerpF(a->dispersion, b->dispersion);
            out->clearcoatIridescence   = lerpF(a->clearcoatIridescence, b->clearcoatIridescence);
            out->clearcoatFilmThickness = lerpF(a->clearcoatFilmThickness, b->clearcoatFilmThickness);
            out->subsurfaceRadius       = lerpC(a->subsurfaceRadius, b->subsurfaceRadius);
            out->subsurfaceScale        = lerpF(a->subsurfaceScale, b->subsurfaceScale);
            out->subsurfaceAnisotropy   = lerpF(a->subsurfaceAnisotropy, b->subsurfaceAnisotropy);
            out->subsurfaceIOR          = lerpF(a->subsurfaceIOR, b->subsurfaceIOR);
            out->tileBreakStrength      = lerpF(a->tileBreakStrength, b->tileBreakStrength);
            out->transmissionDensity    = lerpF(a->transmissionDensity, b->transmissionDensity);
            out->resinColor             = lerpC(a->resinColor, b->resinColor);
            out->resinRoughness         = lerpF(a->resinRoughness, b->resinRoughness);
            out->resinInclusion         = lerpF(a->resinInclusion, b->resinInclusion);
            out->resinDirt              = lerpF(a->resinDirt, b->resinDirt);
            out->resinInclusionScale    = lerpF(a->resinInclusionScale, b->resinInclusionScale);
            out->resinDirtColor         = lerpC(a->resinDirtColor, b->resinDirtColor);
            out->resinShard             = lerpF(a->resinShard, b->resinShard);
            out->resinShardHue          = lerpF(a->resinShardHue, b->resinShardHue);
            out->dustColorA             = lerpC(a->dustColorA, b->dustColorA);
            out->dustColorB             = lerpC(a->dustColorB, b->dustColorB);
            out->bubbleIor              = lerpF(a->bubbleIor, b->bubbleIor);
            out->bubbleFilm             = lerpF(a->bubbleFilm, b->bubbleFilm);

            const ShadeState* texSrc = (fac < 0.5f) ? a.get() : b.get();
            // Discrete fields can't lerp â€” switch with the texture side at Fac 0.5.
            out->resinObjectSpace  = texSrc->resinObjectSpace;
            out->dustStyle         = texSrc->dustStyle;
            out->shardShape        = texSrc->shardShape;
            out->glassMarbleVolume = texSrc->glassMarbleVolume;
            out->isBubble          = texSrc->isBubble;
            out->baseColorTex = texSrc->baseColorTex;
            out->roughnessTex = texSrc->roughnessTex;
            out->metallicTex  = texSrc->metallicTex;
            out->normalTex    = texSrc->normalTex;
            out->opacityTex   = texSrc->opacityTex;
            out->emissionTex  = texSrc->emissionTex;
            out->uvScale = texSrc->uvScale;
            out->uvOffset = texSrc->uvOffset;
            out->uvRotationDeg = texSrc->uvRotationDeg;
            out->hasUvTransform = texSrc->hasUvTransform;

            lastMixHadTextureConflict =
                (a->baseColorTex != b->baseColorTex && a->baseColorTex && b->baseColorTex) ||
                (a->roughnessTex != b->roughnessTex && a->roughnessTex && b->roughnessTex) ||
                (a->metallicTex  != b->metallicTex  && a->metallicTex  && b->metallicTex)  ||
                (a->normalTex    != b->normalTex    && a->normalTex    && b->normalTex);
            return out;
        }
    };

    /**
     * @brief Material Output â€” the graph's single sink. Unconnected param pins
     * fall back to the `defaults` ShadeState (seeded from the bound material, so
     * an empty graph == the material as it is today). Connected pins override.
     *
     * Pin order is part of the serialized format â€” append only.
     */
    class OutputNode : public MaterialNodeBase {
    public:
        ShadeState defaults;

        // Set by deserializeParams when a saved graph predates a field group;
        // consumed by seedMissingGroupsFromMaterial() at bind time.
        bool pendingSeedExtended = false;
        bool pendingSeedResin = false;
        bool pendingSeedBubble = false;

        // Last sync's per-pin connection state â€” detects "just unplugged" so
        // syncPinVisibility can keep that socket group expanded.
        std::vector<bool> lastConnected_;

        /// Fill only the field groups a pre-upgrade save didn't carry, from the
        /// bound material's live values. No-op for fresh graphs and current saves.
        void seedMissingGroupsFromMaterial(const PrincipledBSDF& m) {
            if (!pendingSeedExtended && !pendingSeedResin && !pendingSeedBubble) return;
            const ShadeState s = makeShadeStateFromMaterial(m);
            if (pendingSeedExtended) {
                defaults.anisotropic = s.anisotropic;
                defaults.dispersion = s.dispersion;
                defaults.clearcoatIridescence = s.clearcoatIridescence;
                defaults.clearcoatFilmThickness = s.clearcoatFilmThickness;
                defaults.subsurfaceRadius = s.subsurfaceRadius;
                defaults.subsurfaceScale = s.subsurfaceScale;
                defaults.subsurfaceAnisotropy = s.subsurfaceAnisotropy;
                defaults.subsurfaceIOR = s.subsurfaceIOR;
                defaults.tileBreakStrength = s.tileBreakStrength;
                pendingSeedExtended = false;
            }
            if (pendingSeedResin) {
                defaults.transmissionDensity = s.transmissionDensity;
                defaults.resinColor = s.resinColor;
                defaults.resinRoughness = s.resinRoughness;
                defaults.resinInclusion = s.resinInclusion;
                defaults.resinDirt = s.resinDirt;
                defaults.resinInclusionScale = s.resinInclusionScale;
                defaults.resinDirtColor = s.resinDirtColor;
                defaults.resinShard = s.resinShard;
                defaults.resinShardHue = s.resinShardHue;
                defaults.resinObjectSpace = s.resinObjectSpace;
                defaults.dustStyle = s.dustStyle;
                defaults.dustColorA = s.dustColorA;
                defaults.dustColorB = s.dustColorB;
                defaults.shardShape = s.shardShape;
                defaults.glassMarbleVolume = s.glassMarbleVolume;
                pendingSeedResin = false;
            }
            if (pendingSeedBubble) {
                defaults.isBubble = s.isBubble;
                defaults.bubbleIor = s.bubbleIor;
                defaults.bubbleFilm = s.bubbleFilm;
                pendingSeedBubble = false;
            }
        }

        // v2 GROUPED pin order â€” contiguous per socket group so collapsed groups
        // read cleanly on the node. Saved v1 graphs' link indices are remapped on
        // load (see deserializeMaterialGraph). Append-only WITHIN this scheme.
        enum PinIdx {
            InSurface = 0,
            // Base
            InBaseColor, InMetallic, InRoughness, InSpecular, InAnisotropic,
            // Emission
            InEmissionColor, InEmissionStrength,
            // Glass / Transmission
            InTransmission, InIOR, InDispersion, InOpacity, InTranslucent,
            // Subsurface
            InSubsurface, InSubsurfaceColor, InSubsurfaceRadius, InSubsurfaceScale,
            // Clearcoat
            InClearcoat, InClearcoatRoughness, InClearcoatIridescence, InClearcoatFilm,
            // Surface detail
            InNormalMap, InNormalStrength, InTileBreak,
            // Interior (resin)
            InInteriorDensity, InInteriorColor, InInclusion, InDirt, InShard,
            // Bubble
            InBubbleIor, InBubbleFilm,
            PinCount
        };

        // Socket-group visibility on the NODE. Collapsed groups' unconnected pins are hidden;
        // connected pins always stay visible.
        //
        // The common groups now default OPEN and each one draws a clickable heading on the
        // node (inputSectionLabel / toggleInputSection), with its value editor sitting on the
        // socket's own row. Before this the node showed Base only and every other parameter
        // had to be opted into from the properties panel one group at a time, so the material
        // node looked like a list of names with no numbers on it. Saved graphs keep whatever
        // flags they stored — these defaults only apply to newly created nodes.
        //
        // Open by default: the four groups people actually reach for. The other four keep
        // their HEADING on the node (one click to open) but start closed — with all eight
        // expanded the node is ~1000px tall, which is not an improvement over hiding the
        // values, it is just a different way to make them unreadable.
        bool grpBase = true;
        bool grpEmission = true;
        bool grpGlass = true;
        bool grpDetail = true;
        bool grpSubsurface = false;
        bool grpClearcoat = false;
        bool grpInterior = false;
        bool grpBubble = false;

        OutputNode() {
            name = "Material Output";
            materialNodeType = NodeType::Output;
            auto in = [this](const char* n, NodeSystem::DataType t) {
                inputs.push_back(NodeSystem::Pin::createInput(n, t, NodeSystem::ImageSemantic::Generic, true));
            };
            in("Surface", NodeSystem::DataType::Material);
            // Base
            in("Base Color", NodeSystem::DataType::Vector3);
            in("Metallic", NodeSystem::DataType::Float);
            in("Roughness", NodeSystem::DataType::Float);
            in("Specular", NodeSystem::DataType::Float);
            in("Anisotropic", NodeSystem::DataType::Float);
            // Emission
            in("Emission Color", NodeSystem::DataType::Vector3);
            in("Emission Strength", NodeSystem::DataType::Float);
            // Glass / Transmission
            in("Transmission", NodeSystem::DataType::Float);
            in("IOR", NodeSystem::DataType::Float);
            in("Dispersion", NodeSystem::DataType::Float);
            in("Opacity", NodeSystem::DataType::Float);
            in("Translucent", NodeSystem::DataType::Float);
            // Subsurface
            in("Subsurface", NodeSystem::DataType::Float);
            in("Subsurface Color", NodeSystem::DataType::Vector3);
            in("Subsurface Radius", NodeSystem::DataType::Vector3);
            in("Subsurface Scale", NodeSystem::DataType::Float);
            // Clearcoat
            in("Clearcoat", NodeSystem::DataType::Float);
            in("Clearcoat Roughness", NodeSystem::DataType::Float);
            in("Clearcoat Iridescence", NodeSystem::DataType::Float);
            in("Clearcoat Film", NodeSystem::DataType::Float);
            // Surface detail
            in("Normal Map", NodeSystem::DataType::Vector3);   // binding-only slot (Image Texture)
            in("Normal Strength", NodeSystem::DataType::Float);
            in("Tile Break", NodeSystem::DataType::Float);
            // Interior (resin)
            in("Interior Density", NodeSystem::DataType::Float);
            in("Interior Color", NodeSystem::DataType::Vector3);
            in("Inclusion", NodeSystem::DataType::Float);
            in("Dirt", NodeSystem::DataType::Float);
            in("Shard", NodeSystem::DataType::Float);
            // Bubble
            in("Bubble IOR", NodeSystem::DataType::Float);
            in("Bubble Film", NodeSystem::DataType::Float);
            outputs.push_back(NodeSystem::Pin::createOutput("Material", NodeSystem::DataType::Material));
            metadata.displayName = "Material Output";
            metadata.category = "Shader";
            metadata.description = "The graph result. Unconnected inputs keep the node's own values (seeded from the bound material).";
            metadata.headerColor = HeaderColors::shader();
        }

        /// Apply group visibility to the pins (connected pins never hide).
        /// Cheap; the editor panel calls it every frame for the bound graph.
        ///
        /// When a pin of a COLLAPSED group loses its connection, its group is
        /// auto-expanded: otherwise the socket would vanish the moment the user
        /// unplugs it, and the re-connect could only land on some other visible
        /// pin (the "reconnect binds the wrong slot" trap).
        void syncPinVisibility(const NodeSystem::GraphBase& graph) {
            bool* groupOf[PinCount] = {};
            auto mapRange = [&](int lo, int hi, bool* flag) {
                for (int i = lo; i <= hi && i < PinCount; ++i) groupOf[i] = flag;
            };
            mapRange(InBaseColor, InAnisotropic, &grpBase);
            mapRange(InEmissionColor, InEmissionStrength, &grpEmission);
            mapRange(InTransmission, InTranslucent, &grpGlass);
            mapRange(InSubsurface, InSubsurfaceScale, &grpSubsurface);
            mapRange(InClearcoat, InClearcoatFilm, &grpClearcoat);
            mapRange(InNormalMap, InTileBreak, &grpDetail);
            mapRange(InInteriorDensity, InShard, &grpInterior);
            mapRange(InBubbleIor, InBubbleFilm, &grpBubble);

            const int n = static_cast<int>(inputs.size());
            if (static_cast<int>(lastConnected_.size()) != n) lastConnected_.assign(n, false);

            for (int i = 0; i < n; ++i) {
                const bool connected = graph.getInputSource(inputs[i].id) != nullptr;
                if (lastConnected_[i] && !connected && i < PinCount && groupOf[i]) {
                    *groupOf[i] = true;  // just unplugged â€” keep the socket group open
                }
                lastConnected_[i] = connected;
                const bool groupVisible = (i < PinCount && groupOf[i]) ? *groupOf[i] : true;  // Surface & unknown: always
                inputs[i].hidden = !(groupVisible || connected);
            }
            if (!inputs.empty()) inputs[InSurface].hidden = false;
        }
        std::string getTypeId() const override { return "MatV2.Output"; }

        // ---- socket sections + per-pin value editors (drawn ON the node) --------
        // The properties panel keeps its full grouped editor (it can show what a 26px pin row
        // cannot: the SSS radius vector, the resin dust styles). These hooks are about the
        // NODE no longer being a column of socket names with no numbers on it.
        bool wantsInlinePinWidgets() const override { return true; }
        float pinRowHeight() const override { return 26.0f; }    // a DragFloat frame + breathing room
        float getCustomWidth() const override { return 268.0f; } // label + value widget on one row

        /// A group's FIRST pin carries its heading.
        const char* inputSectionLabel(int index) const override {
            switch (index) {
                case InBaseColor:       return "Base";
                case InEmissionColor:   return "Emission";
                case InTransmission:    return "Glass";
                case InSubsurface:      return "Subsurface";
                case InClearcoat:       return "Clearcoat";
                case InNormalMap:       return "Surface Detail";
                case InInteriorDensity: return "Interior";
                case InBubbleIor:       return "Bubble";
                default:                return nullptr;
            }
        }
        bool isInputSectionOpen(int index) const override {
            switch (index) {
                case InBaseColor:       return grpBase;
                case InEmissionColor:   return grpEmission;
                case InTransmission:    return grpGlass;
                case InSubsurface:      return grpSubsurface;
                case InClearcoat:       return grpClearcoat;
                case InNormalMap:       return grpDetail;
                case InInteriorDensity: return grpInterior;
                case InBubbleIor:       return grpBubble;
                default:                return true;
            }
        }
        void toggleInputSection(int index) override {
            switch (index) {
                case InBaseColor:       grpBase = !grpBase; break;
                case InEmissionColor:   grpEmission = !grpEmission; break;
                case InTransmission:    grpGlass = !grpGlass; break;
                case InSubsurface:      grpSubsurface = !grpSubsurface; break;
                case InClearcoat:       grpClearcoat = !grpClearcoat; break;
                case InNormalMap:       grpDetail = !grpDetail; break;
                case InInteriorDensity: grpInterior = !grpInterior; break;
                case InBubbleIor:       grpBubble = !grpBubble; break;
                default: return;
            }
            dirty = true;
        }

        /// Bubble is a FEATURE with two parameters, not two parameters that happen to sit
        /// together — its on/off switch lives on the section header itself. (It used to be a
        /// "Thin-Shell Bubble" checkbox inside a second "Bubble" block under Advanced, so the
        /// node said Bubble twice and the two rows meant different things.)
        bool* inputSectionToggle(int index) override {
            return (index == InBubbleIor) ? &defaults.isBubble : nullptr;
        }

        /// The value a pin uses when nothing is wired into it. This is the SAME `defaults`
        /// the properties panel edits and the compiler seeds the fold from, so editing it
        /// here, there, or in the material panel is one edit, not three.
        bool drawInputInlineWidget(int index, float width) override {
            ShadeState& d = defaults;
            ImGui::SetNextItemWidth(width);
            auto col = [](Vec3& c) {
                float f[3] = { c.x, c.y, c.z };
                if (ImGui::ColorEdit3("##v", f, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel)) {
                    c = Vec3(f[0], f[1], f[2]);
                    return true;
                }
                return false;
            };
            auto sld = [](float& v, float lo, float hi) { return ImGui::SliderFloat("##v", &v, lo, hi, "%.3f"); };
            auto drg = [](float& v, float speed, float lo, float hi) {
                return ImGui::DragFloat("##v", &v, speed, lo, hi, "%.3f");
            };

            switch (index) {
                case InBaseColor:            return col(d.baseColor);
                case InMetallic:             return sld(d.metallic, 0.0f, 1.0f);
                case InRoughness:            return sld(d.roughness, 0.0f, 1.0f);
                case InSpecular:             return sld(d.specular, 0.0f, 1.0f);
                case InAnisotropic:          return sld(d.anisotropic, 0.0f, 1.0f);
                case InEmissionColor:        return col(d.emissionColor);
                case InEmissionStrength:     return drg(d.emissionStrength, 0.1f, 0.0f, 10000.0f);
                case InTransmission:         return sld(d.transmission, 0.0f, 1.0f);
                case InIOR:                  return drg(d.ior, 0.005f, 1.0f, 3.0f);
                case InDispersion:           return sld(d.dispersion, 0.0f, 1.0f);
                case InOpacity:              return sld(d.opacity, 0.0f, 1.0f);
                case InTranslucent:          return sld(d.translucent, 0.0f, 1.0f);
                case InSubsurface:           return sld(d.subsurface, 0.0f, 1.0f);
                case InSubsurfaceColor:      return col(d.subsurfaceColor);
                case InSubsurfaceRadius: {
                    // Per-channel scatter DISTANCE (components exceed 1) — a compact 3-drag,
                    // NOT a color swatch. Cramped at ~40px per component, but editable in
                    // place beats living under a second "Subsurface" heading somewhere else.
                    float r[3] = { d.subsurfaceRadius.x, d.subsurfaceRadius.y, d.subsurfaceRadius.z };
                    if (ImGui::DragFloat3("##v", r, 0.01f, 0.0f, 10.0f, "%.2f")) {
                        d.subsurfaceRadius = Vec3(r[0], r[1], r[2]);
                        return true;
                    }
                    return false;
                }
                case InSubsurfaceScale:      return drg(d.subsurfaceScale, 0.01f, 0.0f, 100.0f);
                case InClearcoat:            return sld(d.clearcoat, 0.0f, 1.0f);
                case InClearcoatRoughness:   return sld(d.clearcoatRoughness, 0.0f, 1.0f);
                case InClearcoatIridescence: return sld(d.clearcoatIridescence, 0.0f, 1.0f);
                case InClearcoatFilm:        return drg(d.clearcoatFilmThickness, 1.0f, 0.0f, 2000.0f);
                case InNormalStrength:       return sld(d.normalStrength, 0.0f, 4.0f);
                case InTileBreak:            return sld(d.tileBreakStrength, 0.0f, 1.0f);
                case InInteriorDensity:      return drg(d.transmissionDensity, 0.01f, 0.0f, 100.0f);
                case InInteriorColor:        return col(d.resinColor);
                case InInclusion:            return sld(d.resinInclusion, 0.0f, 1.0f);
                case InDirt:                 return sld(d.resinDirt, 0.0f, 1.0f);
                case InShard:                return sld(d.resinShard, 0.0f, 1.0f);
                case InBubbleIor:            return drg(d.bubbleIor, 0.005f, 1.0f, 3.0f);
                case InBubbleFilm:           return drg(d.bubbleFilm, 1.0f, 0.0f, 2000.0f);
                // Surface / Normal Map deliberately have no one-row editor: Surface takes a
                // Material link and Normal Map is a texture BINDING slot (you wire an Image
                // Texture into it). A fake scalar for either would be a lie.
                default: return false;
            }
        }

        /// No body block: the pin rows carry the values, and each group's overflow sits in the
        /// "..." popup on that group's own header (drawInputSectionExtra). The node is nothing
        /// but its grouped cards.
        bool wantsInlineContent() const override { return false; }

        void initDefaultsFromMaterial(const PrincipledBSDF& m) { defaults = makeShadeStateFromMaterial(m); }

        void serializeParams(nlohmann::json& j) const override {
            const ShadeState& d = defaults;
            j["base_color"] = { d.baseColor.x, d.baseColor.y, d.baseColor.z };
            j["metallic"] = d.metallic;
            j["roughness"] = d.roughness;
            j["specular"] = d.specular;
            j["emission_color"] = { d.emissionColor.x, d.emissionColor.y, d.emissionColor.z };
            j["emission_strength"] = d.emissionStrength;
            j["transmission"] = d.transmission;
            j["ior"] = d.ior;
            j["opacity"] = d.opacity;
            j["translucent"] = d.translucent;
            j["clearcoat"] = d.clearcoat;
            j["clearcoat_roughness"] = d.clearcoatRoughness;
            j["subsurface"] = d.subsurface;
            j["subsurface_color"] = { d.subsurfaceColor.x, d.subsurfaceColor.y, d.subsurfaceColor.z };
            j["normal_strength"] = d.normalStrength;
            j["anisotropic"] = d.anisotropic;
            j["dispersion"] = d.dispersion;
            j["cc_iridescence"] = d.clearcoatIridescence;
            j["cc_film_thickness"] = d.clearcoatFilmThickness;
            j["sss_radius"] = { d.subsurfaceRadius.x, d.subsurfaceRadius.y, d.subsurfaceRadius.z };
            j["sss_scale"] = d.subsurfaceScale;
            j["sss_anisotropy"] = d.subsurfaceAnisotropy;
            j["sss_ior"] = d.subsurfaceIOR;
            j["tile_break"] = d.tileBreakStrength;
            j["resin_density"] = d.transmissionDensity;
            j["resin_color"] = { d.resinColor.x, d.resinColor.y, d.resinColor.z };
            j["resin_roughness"] = d.resinRoughness;
            j["resin_inclusion"] = d.resinInclusion;
            j["resin_dirt"] = d.resinDirt;
            j["resin_inclusion_scale"] = d.resinInclusionScale;
            j["resin_dirt_color"] = { d.resinDirtColor.x, d.resinDirtColor.y, d.resinDirtColor.z };
            j["resin_shard"] = d.resinShard;
            j["resin_shard_hue"] = d.resinShardHue;
            j["resin_object_space"] = d.resinObjectSpace;
            j["dust_style"] = d.dustStyle;
            j["dust_color_a"] = { d.dustColorA.x, d.dustColorA.y, d.dustColorA.z };
            j["dust_color_b"] = { d.dustColorB.x, d.dustColorB.y, d.dustColorB.z };
            j["shard_shape"] = d.shardShape;
            j["glass_marble_volume"] = d.glassMarbleVolume;
            j["is_bubble"] = d.isBubble;
            j["bubble_ior"] = d.bubbleIor;
            j["bubble_film"] = d.bubbleFilm;
            j["grp_base"] = grpBase;
            j["grp_emission"] = grpEmission;
            j["grp_glass"] = grpGlass;
            j["grp_sss"] = grpSubsurface;
            j["grp_clearcoat"] = grpClearcoat;
            j["grp_detail"] = grpDetail;
            j["grp_interior"] = grpInterior;
            j["grp_bubble"] = grpBubble;
            j["base_color_tex"] = d.baseColorTex ? d.baseColorTex->name : "";
            j["roughness_tex"] = d.roughnessTex ? d.roughnessTex->name : "";
            j["metallic_tex"] = d.metallicTex ? d.metallicTex->name : "";
            j["normal_tex"] = d.normalTex ? d.normalTex->name : "";
            j["opacity_tex"] = d.opacityTex ? d.opacityTex->name : "";
            j["emission_tex"] = d.emissionTex ? d.emissionTex->name : "";
        }
        void deserializeParams(const nlohmann::json& j) override {
            auto readV3 = [&j](const char* key, Vec3 def) -> Vec3 {
                if (!j.contains(key) || !j[key].is_array() || j[key].size() < 3) return def;
                return Vec3(j[key][0].get<float>(), j[key][1].get<float>(), j[key][2].get<float>());
            };
            ShadeState& d = defaults;
            d.baseColor = readV3("base_color", d.baseColor);
            d.metallic = j.value("metallic", d.metallic);
            d.roughness = j.value("roughness", d.roughness);
            d.specular = j.value("specular", d.specular);
            d.emissionColor = readV3("emission_color", d.emissionColor);
            d.emissionStrength = j.value("emission_strength", d.emissionStrength);
            d.transmission = j.value("transmission", d.transmission);
            d.ior = j.value("ior", d.ior);
            d.opacity = j.value("opacity", d.opacity);
            d.translucent = j.value("translucent", d.translucent);
            d.clearcoat = j.value("clearcoat", d.clearcoat);
            d.clearcoatRoughness = j.value("clearcoat_roughness", d.clearcoatRoughness);
            d.subsurface = j.value("subsurface", d.subsurface);
            d.subsurfaceColor = readV3("subsurface_color", d.subsurfaceColor);
            d.normalStrength = j.value("normal_strength", d.normalStrength);
            d.anisotropic = j.value("anisotropic", d.anisotropic);
            d.dispersion = j.value("dispersion", d.dispersion);
            d.clearcoatIridescence = j.value("cc_iridescence", d.clearcoatIridescence);
            d.clearcoatFilmThickness = j.value("cc_film_thickness", d.clearcoatFilmThickness);
            d.subsurfaceRadius = readV3("sss_radius", d.subsurfaceRadius);
            d.subsurfaceScale = j.value("sss_scale", d.subsurfaceScale);
            d.subsurfaceAnisotropy = j.value("sss_anisotropy", d.subsurfaceAnisotropy);
            d.subsurfaceIOR = j.value("sss_ior", d.subsurfaceIOR);
            d.tileBreakStrength = j.value("tile_break", d.tileBreakStrength);
            d.transmissionDensity = j.value("resin_density", d.transmissionDensity);
            d.resinColor = readV3("resin_color", d.resinColor);
            d.resinRoughness = j.value("resin_roughness", d.resinRoughness);
            d.resinInclusion = j.value("resin_inclusion", d.resinInclusion);
            d.resinDirt = j.value("resin_dirt", d.resinDirt);
            d.resinInclusionScale = j.value("resin_inclusion_scale", d.resinInclusionScale);
            d.resinDirtColor = readV3("resin_dirt_color", d.resinDirtColor);
            d.resinShard = j.value("resin_shard", d.resinShard);
            d.resinShardHue = j.value("resin_shard_hue", d.resinShardHue);
            d.resinObjectSpace = j.value("resin_object_space", d.resinObjectSpace);
            d.dustStyle = j.value("dust_style", d.dustStyle);
            d.dustColorA = readV3("dust_color_a", d.dustColorA);
            d.dustColorB = readV3("dust_color_b", d.dustColorB);
            d.shardShape = j.value("shard_shape", d.shardShape);
            d.glassMarbleVolume = j.value("glass_marble_volume", d.glassMarbleVolume);
            d.isBubble = j.value("is_bubble", d.isBubble);
            d.bubbleIor = j.value("bubble_ior", d.bubbleIor);
            d.bubbleFilm = j.value("bubble_film", d.bubbleFilm);

            // Older saves predate some field groups: remember which groups were
            // absent so the panel can seed them from the BOUND MATERIAL's real
            // values at bind time (instead of silently applying constructor
            // defaults and resetting e.g. a material's resin setup on Apply).
            pendingSeedExtended = !j.contains("anisotropic");
            pendingSeedResin = !j.contains("resin_density");
            pendingSeedBubble = !j.contains("is_bubble");

            grpBase = j.value("grp_base", grpBase);
            grpEmission = j.value("grp_emission", grpEmission);
            grpGlass = j.value("grp_glass", grpGlass);
            grpSubsurface = j.value("grp_sss", grpSubsurface);
            grpClearcoat = j.value("grp_clearcoat", grpClearcoat);
            grpDetail = j.value("grp_detail", grpDetail);
            grpInterior = j.value("grp_interior", grpInterior);
            grpBubble = j.value("grp_bubble", grpBubble);
            d.baseColorTex = resolveTextureByName(j.value("base_color_tex", std::string()));
            d.roughnessTex = resolveTextureByName(j.value("roughness_tex", std::string()));
            d.metallicTex = resolveTextureByName(j.value("metallic_tex", std::string()));
            d.normalTex = resolveTextureByName(j.value("normal_tex", std::string()));
            d.opacityTex = resolveTextureByName(j.value("opacity_tex", std::string()));
            d.emissionTex = resolveTextureByName(j.value("emission_tex", std::string()));
        }

        // Overflow parameters live in a popup behind the "..." button on their OWN group's
        // header — there is no shared "Advanced" block under the sockets any more. That block
        // was the last place a group's parameters could end up split in two: the node said
        // "Subsurface" twice (the socket card and an Advanced tree of the same name), and
        // which of the two was authoritative was anyone's guess. Now a group is ONE card:
        // pins with their values on the rows, and everything that cannot fit on a row one
        // click away on that same card's header.
        bool inputSectionHasExtra(int index) const override {
            return index == InSubsurface || index == InInteriorDensity;
        }
        void drawInputSectionExtra(int index) override {
            ShadeState& d = defaults;

            auto colorEdit = [this](const char* label, Vec3& c) {
                float f[3] = { c.x, c.y, c.z };
                if (ImGui::ColorEdit3(label, f, ImGuiColorEditFlags_NoInputs)) {
                    c = Vec3(f[0], f[1], f[2]);
                    dirty = true;
                }
            };
            auto slider = [this](const char* label, float& v, float lo, float hi) {
                ImGui::SetNextItemWidth(120);
                if (ImGui::SliderFloat(label, &v, lo, hi)) dirty = true;
            };
            auto drag = [this](const char* label, float& v, float speed, float lo, float hi) {
                ImGui::SetNextItemWidth(120);
                if (ImGui::DragFloat(label, &v, speed, lo, hi)) dirty = true;
            };

            if (index == InSubsurface) {
                ImGui::TextDisabled("Subsurface");
                ImGui::Separator();
                slider("Anisotropy", d.subsurfaceAnisotropy, -1.0f, 1.0f);
                drag("IOR##sss", d.subsurfaceIOR, 0.005f, 1.0f, 2.5f);
            } else if (index == InInteriorDensity) {
                ImGui::TextDisabled("Interior Volume");
                ImGui::Separator();
                slider("Coat Roughness", d.resinRoughness, 0.0f, 1.0f);
                drag("Feature Scale", d.resinInclusionScale, 0.05f, 0.5f, 64.0f);
                colorEdit("Dirt Color", d.resinDirtColor);
                drag("Shard Hue", d.resinShardHue, 0.005f, -1.0f, 1.0f);
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("0..1 = base hue, negative = rainbow palette");
                {
                    const char* dustStyles[] = { "Nebula", "Billow", "Wispy", "Paint Swirl" };
                    ImGui::SetNextItemWidth(120);
                    if (ImGui::Combo("Dust Style", &d.dustStyle, dustStyles, IM_ARRAYSIZE(dustStyles))) dirty = true;
                }
                colorEdit("Dust Color A", d.dustColorA);
                colorEdit("Dust Color B", d.dustColorB);
                {
                    const char* shardShapes[] = { "Chips", "Crystals" };
                    ImGui::SetNextItemWidth(120);
                    if (ImGui::Combo("Shard Shape", &d.shardShape, shardShapes, IM_ARRAYSIZE(shardShapes))) dirty = true;
                }
                if (ImGui::Checkbox("Object Space", &d.resinObjectSpace)) dirty = true;
                if (ImGui::Checkbox("Glass Marble Volume", &d.glassMarbleVolume)) dirty = true;
            }
        }

        NodeSystem::PinValue compute(int, NodeSystem::EvaluationContext& ctx) override {
            auto state = std::make_shared<ShadeState>(defaults);

            NodeSystem::MaterialValue surf;
            if (NodeSystem::tryGetMaterial(getInputValue(InSurface, ctx), surf)) {
                *state = *surf;
            }

            auto v3 = [](const std::array<float, 3>& a) { return Vec3(a[0], a[1], a[2]); };
            if (isInputConnected(InBaseColor, ctx))      state->baseColor = v3(getVec3In(InBaseColor, ctx, {}));
            if (isInputConnected(InMetallic, ctx))       state->metallic = getFloatIn(InMetallic, ctx, state->metallic);
            if (isInputConnected(InRoughness, ctx))      state->roughness = getFloatIn(InRoughness, ctx, state->roughness);
            if (isInputConnected(InSpecular, ctx))       state->specular = getFloatIn(InSpecular, ctx, state->specular);
            if (isInputConnected(InEmissionColor, ctx))  state->emissionColor = v3(getVec3In(InEmissionColor, ctx, {}));
            if (isInputConnected(InEmissionStrength, ctx)) state->emissionStrength = getFloatIn(InEmissionStrength, ctx, state->emissionStrength);
            if (isInputConnected(InTransmission, ctx))   state->transmission = getFloatIn(InTransmission, ctx, state->transmission);
            if (isInputConnected(InIOR, ctx))            state->ior = getFloatIn(InIOR, ctx, state->ior);
            if (isInputConnected(InOpacity, ctx))        state->opacity = getFloatIn(InOpacity, ctx, state->opacity);
            if (isInputConnected(InTranslucent, ctx))    state->translucent = getFloatIn(InTranslucent, ctx, state->translucent);
            if (isInputConnected(InClearcoat, ctx))      state->clearcoat = getFloatIn(InClearcoat, ctx, state->clearcoat);
            if (isInputConnected(InClearcoatRoughness, ctx)) state->clearcoatRoughness = getFloatIn(InClearcoatRoughness, ctx, state->clearcoatRoughness);
            if (isInputConnected(InSubsurface, ctx))     state->subsurface = getFloatIn(InSubsurface, ctx, state->subsurface);
            if (isInputConnected(InSubsurfaceColor, ctx)) state->subsurfaceColor = v3(getVec3In(InSubsurfaceColor, ctx, {}));
            if (isInputConnected(InNormalStrength, ctx)) state->normalStrength = getFloatIn(InNormalStrength, ctx, state->normalStrength);
            if (isInputConnected(InAnisotropic, ctx))    state->anisotropic = getFloatIn(InAnisotropic, ctx, state->anisotropic);
            if (isInputConnected(InDispersion, ctx))     state->dispersion = getFloatIn(InDispersion, ctx, state->dispersion);
            if (isInputConnected(InSubsurfaceRadius, ctx)) state->subsurfaceRadius = v3(getVec3In(InSubsurfaceRadius, ctx, {}));
            if (isInputConnected(InSubsurfaceScale, ctx)) state->subsurfaceScale = getFloatIn(InSubsurfaceScale, ctx, state->subsurfaceScale);
            if (isInputConnected(InClearcoatIridescence, ctx)) state->clearcoatIridescence = getFloatIn(InClearcoatIridescence, ctx, state->clearcoatIridescence);
            if (isInputConnected(InClearcoatFilm, ctx))  state->clearcoatFilmThickness = getFloatIn(InClearcoatFilm, ctx, state->clearcoatFilmThickness);
            if (isInputConnected(InTileBreak, ctx))      state->tileBreakStrength = getFloatIn(InTileBreak, ctx, state->tileBreakStrength);
            if (isInputConnected(InInteriorDensity, ctx)) state->transmissionDensity = getFloatIn(InInteriorDensity, ctx, state->transmissionDensity);
            if (isInputConnected(InInteriorColor, ctx))  state->resinColor = v3(getVec3In(InInteriorColor, ctx, {}));
            if (isInputConnected(InInclusion, ctx))      state->resinInclusion = getFloatIn(InInclusion, ctx, state->resinInclusion);
            if (isInputConnected(InDirt, ctx))           state->resinDirt = getFloatIn(InDirt, ctx, state->resinDirt);
            if (isInputConnected(InShard, ctx))          state->resinShard = getFloatIn(InShard, ctx, state->resinShard);
            if (isInputConnected(InBubbleIor, ctx))      state->bubbleIor = getFloatIn(InBubbleIor, ctx, state->bubbleIor);
            if (isInputConnected(InBubbleFilm, ctx))     state->bubbleFilm = getFloatIn(InBubbleFilm, ctx, state->bubbleFilm);
            // InNormalMap is binding-only: consumed by the texture-binding pass in
            // evaluateMaterialGraph, not sampled numerically.
            return state;
        }
    };

    // ============================================================================
    // GRAPH
    // ============================================================================

    class MaterialNodeGraphV2 : public NodeSystem::GraphBase {
    public:
        /// This graph has not been applied yet in this session, so the backend may not hold
        /// its per-pixel program AND every node still carries the dirty flag it was BORN with
        /// (NodeBase::dirty defaults to true). That second half is the one that bites: live
        /// apply detects an edit as a false->true dirty TRANSITION, and a node that is already
        /// dirty can never transition — so a freshly LOADED graph is deaf to every edit until
        /// something applies it once and clears the flags. (Toggling Live off/on used to be
        /// the only thing that did, which is exactly what "it wakes up when I re-tick Live"
        /// meant.) applyGraph clears this; nothing else may.
        ///
        /// UI-only, deliberately NOT serialized: a graph read back from disk has by definition
        /// not been applied in this session.
        bool needsInitialApply = true;

        NodeSystem::NodeBase* addMaterialNode(NodeType type, float x = 0, float y = 0) {
            NodeSystem::NodeBase* node = nullptr;
            switch (type) {
                case NodeType::Output:            node = addNode<OutputNode>(); break;
                case NodeType::MaterialRef:       node = addNode<MaterialRefNode>(); break;
                case NodeType::MixMaterial:       node = addNode<MixMaterialNode>(); break;
                case NodeType::Value:             node = addNode<ValueNode>(); break;
                case NodeType::Color:             node = addNode<ColorNode>(); break;
                case NodeType::TextureCoordinate: node = addNode<TextureCoordinateNode>(); break;
                case NodeType::ImageTexture:      node = addNode<ImageTextureNode>(); break;
                case NodeType::Mapping:           node = addNode<MappingNode>(); break;
                case NodeType::Noise:             node = addNode<NoiseTextureNode>(); break;
                // Voronoi/Checker were merged into the unified Noise Texture
                // node â€” the enum values remain as kind presets.
                case NodeType::Voronoi: {
                    auto* nn = addNode<NoiseTextureNode>();
                    nn->kind = NoiseTextureNode::Kind::Voronoi;
                    node = nn;
                    break;
                }
                case NodeType::Checker: {
                    auto* nn = addNode<NoiseTextureNode>();
                    nn->kind = NoiseTextureNode::Kind::Checker;
                    node = nn;
                    break;
                }
                case NodeType::ColorRamp:         node = addNode<ColorRampNode>(); break;
                case NodeType::MixColor:          node = addNode<MixColorNode>(); break;
                case NodeType::Invert:            node = addNode<InvertNode>(); break;
                case NodeType::Gamma:             node = addNode<GammaNode>(); break;
                case NodeType::Math:              node = addNode<MathNode>(); break;
                case NodeType::SeparateColor:     node = addNode<SeparateColorNode>(); break;
                case NodeType::CombineColor:      node = addNode<CombineColorNode>(); break;
                case NodeType::Geometry:          node = addNode<GeometryNode>(); break;
                case NodeType::ObjectInfo:        node = addNode<ObjectInfoNode>(); break;
                case NodeType::Attribute:         node = addNode<AttributeNode>(); break;
                case NodeType::Wave:              node = addNode<WaveTextureNode>(); break;
                case NodeType::Gradient:          node = addNode<GradientTextureNode>(); break;
                case NodeType::VectorMath:        node = addNode<VectorMathNode>(); break;
                case NodeType::HueSaturation:     node = addNode<HueSaturationNode>(); break;
                case NodeType::RGBCurves:         node = addNode<RGBCurvesNode>(); break;
                case NodeType::Clamp:             node = addNode<ClampNode>(); break;
                case NodeType::FloatCurve:        node = addNode<FloatCurveNode>(); break;
                case NodeType::MapRange:          node = addNode<MapRangeNode>(); break;
                case NodeType::BrightContrast:    node = addNode<BrightContrastNode>(); break;
                case NodeType::Bump:              node = addNode<BumpNode>(); break;
                case NodeType::Fresnel:           node = addNode<FresnelNode>(); break;
                case NodeType::LayerWeight:       node = addNode<LayerWeightNode>(); break;
                case NodeType::AmbientOcclusion:  node = addNode<AmbientOcclusionNode>(); break;
                case NodeType::Bevel:             node = addNode<BevelNode>(); break;
            }
            if (node) {
                node->x = x;
                node->y = y;
            }
            return node;
        }

        OutputNode* findOutputNode() const {
            for (const auto& n : nodes) {
                if (auto* o = dynamic_cast<OutputNode*>(n.get())) return o;
            }
            return nullptr;
        }
    };

    /**
     * @brief Every texture an Image Texture node in this graph references.
     *
     * The project's texture serializer walks MATERIAL SLOTS. A texture that reaches a slot
     * through a manipulation node (Bright/Contrast, ColorRamp, Mix, ...) is bound to no
     * slot — evaluate's bind pass only binds a DIRECT Image Texture — so the serializer
     * never sees it and the project is saved without it. The graph then reloads holding a
     * name nothing in the project can resolve. Save has to ask the graphs too.
     */
    inline std::vector<std::shared_ptr<Texture>> collectGraphTextures(const MaterialNodeGraphV2& graph) {
        std::vector<std::shared_ptr<Texture>> out;
        for (const auto& n : graph.nodes) {
            auto* img = dynamic_cast<ImageTextureNode*>(n.get());
            if (!img) continue;
            img->resolveIfNeeded();
            if (img->texture && img->texture->is_loaded()) out.push_back(img->texture);
        }
        return out;
    }

    // ============================================================================
    // MATERIALIZE: material -> visible node graph
    // ============================================================================

    /**
     * @brief Build a graph that VISIBLY represents the material's current state:
     * every bound texture becomes an Image Texture node wired into its Output
     * slot, and the albedo color becomes a Color node when no albedo texture is
     * bound. Non-texture parameters live in the Output node's seeded defaults.
     * Replaces the graph's existing content.
     */
    inline void materializeGraphFromMaterial(MaterialNodeGraphV2& graph, const PrincipledBSDF& m) {
        graph.clear();
        auto* out = static_cast<OutputNode*>(graph.addMaterialNode(NodeType::Output, 460, 60));
        if (!out) return;
        out->initDefaultsFromMaterial(m);

        struct SlotBinding {
            std::shared_ptr<Texture> tex;
            int pinIdx;
        };
        const SlotBinding slots[] = {
            { m.albedoProperty.texture,    OutputNode::InBaseColor },
            { m.roughnessProperty.texture, OutputNode::InRoughness },
            { m.metallicProperty.texture,  OutputNode::InMetallic },
            { m.opacityProperty.texture,   OutputNode::InOpacity },
            { m.emissionProperty.texture,  OutputNode::InEmissionColor },
            { m.normalProperty.texture,    OutputNode::InNormalMap },
        };
        float y = 40.0f;
        for (const auto& slot : slots) {
            if (!slot.tex) continue;
            auto* img = static_cast<ImageTextureNode*>(graph.addMaterialNode(NodeType::ImageTexture, 150, y));
            if (!img) continue;
            img->setTexture(slot.tex);
            if (!img->outputs.empty() && slot.pinIdx < static_cast<int>(out->inputs.size())) {
                graph.addLink(img->outputs[0].id, out->inputs[slot.pinIdx].id);
            }
            y += 100.0f;
        }
        if (!m.albedoProperty.texture) {
            auto* col = static_cast<ColorNode*>(graph.addMaterialNode(NodeType::Color, 150, y));
            if (col) {
                col->color[0] = m.albedoProperty.color.x;
                col->color[1] = m.albedoProperty.color.y;
                col->color[2] = m.albedoProperty.color.z;
                if (!col->outputs.empty()) {
                    graph.addLink(col->outputs[0].id, out->inputs[OutputNode::InBaseColor].id);
                }
            }
        }
        // Connected pins stay visible regardless of group flags (syncPinVisibility).
    }

    /**
     * @brief Move any texture still living in the Output node's hidden defaults
     * into VISIBLE Image Texture nodes, then clear the defaults' texture slots.
     *
     * Textures must have exactly ONE owner â€” the wired node. When they also
     * lived in defaults, deleting the node/link could not unbind the texture:
     * compute() starts from defaults, so the "removed" texture kept flowing
     * into every Apply. After this migration an unconnected slot genuinely
     * means "no texture", and Apply clears it on all backends.
     *
     * Runs on every bind (cheap no-op once defaults are clean); old saves that
     * carried textures only in defaults get upgraded to visible nodes.
     * @returns true if the graph changed (caller marks the project modified).
     */
    inline bool migrateDefaultTexturesToNodes(MaterialNodeGraphV2& graph, OutputNode& out) {
        struct Slot {
            std::shared_ptr<Texture> ShadeState::* tex;
            int pinIdx;
        };
        static const Slot slots[] = {
            { &ShadeState::baseColorTex, OutputNode::InBaseColor },
            { &ShadeState::roughnessTex, OutputNode::InRoughness },
            { &ShadeState::metallicTex,  OutputNode::InMetallic },
            { &ShadeState::opacityTex,   OutputNode::InOpacity },
            { &ShadeState::emissionTex,  OutputNode::InEmissionColor },
            { &ShadeState::normalTex,    OutputNode::InNormalMap },
        };
        bool changed = false;
        float y = out.y + 20.0f;
        for (const auto& slot : slots) {
            std::shared_ptr<Texture>& tex = out.defaults.*(slot.tex);
            if (!tex) continue;
            if (slot.pinIdx < static_cast<int>(out.inputs.size()) &&
                graph.getInputSource(out.inputs[slot.pinIdx].id) == nullptr) {
                auto* img = static_cast<ImageTextureNode*>(
                    graph.addMaterialNode(NodeType::ImageTexture, out.x - 300.0f, y));
                if (img) {
                    img->setTexture(tex);
                    if (!img->outputs.empty()) {
                        graph.addLink(img->outputs[0].id, out.inputs[slot.pinIdx].id);
                    }
                }
                y += 100.0f;
            }
            // Already-connected slots just drop the stale default binding.
            tex = nullptr;
            changed = true;
        }
        return changed;
    }

    /**
     * @brief Non-destructive auto-sync FROM the live material INTO the graph, so
     * edits made in the regular Properties panel (or a texture swap) are NOT
     * reverted by the graph's next Apply. Replaces the manual "Pull" button for
     * the live workflow.
     *
     * Why it's safe to run every frame: with live-apply on, any graph-side edit
     * is written straight back to the material, so pulling the material into the
     * Output defaults is idempotent for graph edits â€” the only NEW information a
     * pull brings is an external (Properties-panel) change. Wired slots are
     * overridden at eval, so their default value is irrelevant; textures stay
     * single-owner in Image Texture nodes (defaults kept texture-free), and a
     * main-panel texture swap is forwarded into the wired Image Texture node.
     */
    inline void pullMaterialStateIntoGraph(MaterialNodeGraphV2& graph, OutputNode& out, const PrincipledBSDF& m) {
        // 1) numeric/params: un-wired Output slots mirror the live material.
        ShadeState live = makeShadeStateFromMaterial(m);
        live.baseColorTex.reset(); live.roughnessTex.reset(); live.metallicTex.reset();
        live.normalTex.reset();    live.opacityTex.reset();   live.emissionTex.reset();
        out.defaults = live;

        // 2) a texture swapped in the main panel must follow into its wired Image
        //    Texture node, or Apply would re-bind the graph's stale texture.
        struct TexBind { int pin; std::shared_ptr<Texture> t; };
        const TexBind binds[] = {
            { OutputNode::InBaseColor,     m.albedoProperty.texture },
            { OutputNode::InRoughness,     m.roughnessProperty.texture },
            { OutputNode::InMetallic,      m.metallicProperty.texture },
            { OutputNode::InOpacity,       m.opacityProperty.texture },
            { OutputNode::InEmissionColor, m.emissionProperty.texture },
            { OutputNode::InNormalMap,     m.normalProperty.texture },
        };
        for (const auto& b : binds) {
            if (b.pin >= static_cast<int>(out.inputs.size())) continue;
            auto* img = dynamic_cast<ImageTextureNode*>(graph.getInputSourceNode(out.inputs[b.pin].id));
            if (!img) continue;
            img->resolveIfNeeded();
            if (img->texture != b.t) img->setTexture(b.t);  // setTexture no-ops when equal
        }
    }

    // ============================================================================
    // EVALUATE + FOLD
    // ============================================================================

    struct MaterialGraphResult {
        ShadeState state;
        std::vector<std::string> warnings;
        std::vector<std::string> errors;
        bool ok = false;
    };

    /**
     * @brief Evaluate the graph against the bound material and fold to a single
     * ShadeState. Spatial variation is detected by sampling several UVs and
     * averaged (with a warning) â€” see file header for the Faz 1 / Faz 2 split.
     * Texture bindings are recovered structurally: an Image Texture wired
     * directly (or with a Mapping on its Vector input) into an Output slot
     * binds losslessly to that material slot.
     */
    inline MaterialGraphResult evaluateMaterialGraph(MaterialNodeGraphV2& graph, PrincipledBSDF* bound) {
        MaterialGraphResult res;
        OutputNode* out = graph.findOutputNode();
        if (!out) {
            res.errors.push_back("Graph has no Material Output node");
            return res;
        }

        MaterialEvalContext mctx;
        mctx.boundMaterial = bound;
        NodeSystem::EvaluationContext ctx(&graph);
        ctx.setDomainContext(&mctx);

        // --- multi-UV sampling: average + detect spatial variation ---
        static const float kSamples[5][2] = {
            { 0.5f, 0.5f }, { 0.25f, 0.25f }, { 0.75f, 0.25f }, { 0.25f, 0.75f }, { 0.75f, 0.75f }
        };
        std::vector<ShadeState> states;
        states.reserve(5);
        for (const auto& s : kSamples) {
            mctx.u = s[0];
            mctx.v = s[1];
            // No real geometry during the fold â€” use the sample UV as a stand-in
            // position so 3D-noise chains still vary across samples (representative
            // average + variation detection). The real per-pixel value comes from
            // the compiled program at render time.
            mctx.px = s[0]; mctx.py = s[1]; mctx.pz = 0.0f;
            ctx.clearCache();
            graph.markAllDirty();
            NodeSystem::MaterialValue mv;
            if (!NodeSystem::tryGetMaterial(out->requestOutput(0, ctx), mv)) {
                for (const auto& e : ctx.getErrors()) res.errors.push_back(e.message);
                if (res.errors.empty()) res.errors.push_back("Material Output produced no value");
                return res;
            }
            states.push_back(*mv);
        }
        for (const auto& e : ctx.getErrors()) res.errors.push_back(e.message);

        // Average numeric fields; textures / uv-transform from the center sample.
        ShadeState avg = states[0];
        auto avgF = [&states](float ShadeState::* f) {
            float sum = 0.0f;
            for (const auto& s : states) sum += s.*f;
            return sum / static_cast<float>(states.size());
        };
        auto avgC = [&states](Vec3 ShadeState::* f) {
            Vec3 sum(0.0f, 0.0f, 0.0f);
            for (const auto& s : states) sum = sum + s.*f;
            return sum / static_cast<float>(states.size());
        };
        auto varyF = [&states](float ShadeState::* f) {
            float mn = 1e30f, mx = -1e30f;
            for (const auto& s : states) { mn = std::min(mn, s.*f); mx = std::max(mx, s.*f); }
            return (mx - mn) > 1e-3f;
        };
        auto varyC = [&states](Vec3 ShadeState::* f) {
            Vec3 mn(1e30f, 1e30f, 1e30f), mx(-1e30f, -1e30f, -1e30f);
            for (const auto& s : states) {
                const Vec3& v = s.*f;
                mn = Vec3(std::min(mn.x, v.x), std::min(mn.y, v.y), std::min(mn.z, v.z));
                mx = Vec3(std::max(mx.x, v.x), std::max(mx.y, v.y), std::max(mx.z, v.z));
            }
            return std::max({ mx.x - mn.x, mx.y - mn.y, mx.z - mn.z }) > 1e-3f;
        };
        struct FieldF { const char* name; float ShadeState::* ptr; };
        struct FieldC { const char* name; Vec3 ShadeState::* ptr; };
        static const FieldF floatFields[] = {
            { "Metallic", &ShadeState::metallic }, { "Roughness", &ShadeState::roughness },
            { "Specular", &ShadeState::specular }, { "Emission Strength", &ShadeState::emissionStrength },
            { "Transmission", &ShadeState::transmission }, { "IOR", &ShadeState::ior },
            { "Opacity", &ShadeState::opacity }, { "Translucent", &ShadeState::translucent },
            { "Clearcoat", &ShadeState::clearcoat }, { "Clearcoat Roughness", &ShadeState::clearcoatRoughness },
            { "Subsurface", &ShadeState::subsurface }, { "Normal Strength", &ShadeState::normalStrength },
            { "Anisotropic", &ShadeState::anisotropic }, { "Dispersion", &ShadeState::dispersion },
            { "Clearcoat Iridescence", &ShadeState::clearcoatIridescence },
            { "Clearcoat Film Thickness", &ShadeState::clearcoatFilmThickness },
            { "Subsurface Scale", &ShadeState::subsurfaceScale },
            { "Subsurface Anisotropy", &ShadeState::subsurfaceAnisotropy },
            { "Subsurface IOR", &ShadeState::subsurfaceIOR },
            { "Tile Break", &ShadeState::tileBreakStrength },
            { "Interior Density", &ShadeState::transmissionDensity },
            { "Resin Roughness", &ShadeState::resinRoughness },
            { "Resin Inclusion", &ShadeState::resinInclusion },
            { "Resin Dirt", &ShadeState::resinDirt },
            { "Resin Inclusion Scale", &ShadeState::resinInclusionScale },
            { "Resin Shard", &ShadeState::resinShard },
            { "Bubble IOR", &ShadeState::bubbleIor },
            { "Bubble Film", &ShadeState::bubbleFilm }
        };
        static const FieldC colorFields[] = {
            { "Base Color", &ShadeState::baseColor }, { "Emission Color", &ShadeState::emissionColor },
            { "Subsurface Color", &ShadeState::subsurfaceColor },
            { "Subsurface Radius", &ShadeState::subsurfaceRadius },
            { "Resin Color", &ShadeState::resinColor },
            { "Resin Dirt Color", &ShadeState::resinDirtColor },
            { "Dust Color A", &ShadeState::dustColorA },
            { "Dust Color B", &ShadeState::dustColorB }
        };
        // Note: the per-pixel program (compileMaterialProgram) covers base color/
        // roughness/metallic/specular/transmission/emission/opacity/IOR â€” those
        // shade PER-PIXEL on BOTH the CPU render and Vulkan RT (Faz 2b is done).
        // The 5-sample average below is what the FROZEN OptiX backend uses, plus
        // any slot outside that covered set (subsurface/clearcoat/resin/â€¦) and any
        // chain the compiler can't lower (still folds on every backend).
        for (const auto& f : floatFields) {
            if (varyF(f.ptr)) {
                avg.*(f.ptr) = avgF(f.ptr);
                res.warnings.push_back(std::string(f.name) +
                    ": spatially varying â€” shaded per-pixel on CPU + Vulkan RT where the chain compiles; the frozen OptiX backend (and uncompilable chains) use this 5-sample average");
            }
        }
        for (const auto& f : colorFields) {
            if (varyC(f.ptr)) {
                avg.*(f.ptr) = avgC(f.ptr);
                res.warnings.push_back(std::string(f.name) +
                    ": spatially varying â€” shaded per-pixel on CPU + Vulkan RT where the chain compiles; the frozen OptiX backend (and uncompilable chains) use this 5-sample average");
            }
        }

        // --- structural texture-binding pass ---
        // Direct ImageTexture upstream of an Output slot binds losslessly; a
        // Mapping on that texture's Vector input becomes the material-wide
        // TextureTransform (one per material â€” conflicting mappings warn).
        bool mappingCaptured = avg.hasUvTransform;
        auto bindSlot = [&](int pinIdx, std::shared_ptr<Texture> ShadeState::* slot, const char* slotName) {
            if (pinIdx >= static_cast<int>(out->inputs.size())) return;
            NodeSystem::NodeBase* src = graph.getInputSourceNode(out->inputs[pinIdx].id);
            auto* img = dynamic_cast<ImageTextureNode*>(src);
            if (!img) {
                if (src && pinIdx == OutputNode::InNormalMap) {
                    res.warnings.push_back("Normal Map slot: only an Image Texture can bind here (other chains are ignored)");
                }
                return;
            }
            img->resolveIfNeeded();
            if (!img->texture) {
                res.warnings.push_back(std::string(slotName) + ": Image Texture node has no texture selected");
                return;
            }
            avg.*slot = img->texture;
            if (!img->inputs.empty()) {
                if (auto* map = dynamic_cast<MappingNode*>(graph.getInputSourceNode(img->inputs[0].id))) {
                    if (mappingCaptured &&
                        (avg.uvScale.u != map->scale[0] || avg.uvScale.v != map->scale[1] ||
                         avg.uvOffset.u != map->offset[0] || avg.uvOffset.v != map->offset[1] ||
                         avg.uvRotationDeg != map->rotationDeg)) {
                        res.warnings.push_back("Multiple different Mapping nodes: the material has ONE TextureTransform, last one wins");
                    }
                    avg.uvScale = Vec2(map->scale[0], map->scale[1]);
                    avg.uvOffset = Vec2(map->offset[0], map->offset[1]);
                    avg.uvRotationDeg = map->rotationDeg;
                    avg.hasUvTransform = true;
                    mappingCaptured = true;
                }
            }
        };
        bindSlot(OutputNode::InBaseColor, &ShadeState::baseColorTex, "Base Color");
        bindSlot(OutputNode::InRoughness, &ShadeState::roughnessTex, "Roughness");
        bindSlot(OutputNode::InMetallic, &ShadeState::metallicTex, "Metallic");
        bindSlot(OutputNode::InOpacity, &ShadeState::opacityTex, "Opacity");
        bindSlot(OutputNode::InEmissionColor, &ShadeState::emissionTex, "Emission");
        bindSlot(OutputNode::InNormalMap, &ShadeState::normalTex, "Normal Map");

        // Mix Material texture-switch warning
        // Mix Material: what the FOLD can't do. Base Color / Roughness / Metallic /
        // Emission now blend per-pixel including their textures (the compiled program
        // samples both materials' textures and lerps). What is left on the fold — and
        // therefore still hard-switches at Fac 0.5 — is Opacity's texture and every
        // slot past MatSlot::Count (clearcoat, subsurface, resin, ...).
        for (const auto& n : graph.nodes) {
            if (auto* mix = dynamic_cast<MixMaterialNode*>(n.get())) {
                if (mix->lastMixHadTextureConflict && (avg.opacityTex || avg.normalTex)) {
                    res.warnings.push_back("Mix Material: Opacity / Normal Map textures still switch at Fac 0.5 (Base Color, Roughness, Metallic and Emission do blend per-pixel)");
                }
            }
        }

        res.state = avg;
        res.ok = res.errors.empty();
        return res;
    }

    // ============================================================================
    // FAZ 2a: COMPILE GRAPH -> per-pixel MaterialProgram (SVM-lite)
    // ============================================================================
    // Companion to evaluateMaterialGraph. Where evaluate FOLDS the graph to
    // constants+bound-textures (Faz 1), this compiles the SPATIALLY-VARYING
    // chains into an instruction stream (MaterialProgram.h) that the CPU render
    // runs per shading point â€” so Noise/Voronoi/Checker/ColorRamp finally wrap
    // the surface per pixel instead of showing their 5-sample average.
    //
    // A slot is compiled only when its chain (a) is fully supported, (b) is
    // spatial (varies with UV), and (c) is NOT a direct texture bind (those are
    // already handled losslessly by evaluate's binding pass; the GPU samples the
    // bound texture). Everything else stays on the Faz-1 constant/texture path,
    // so a pure constant material still costs zero per-pixel work (active=false).
    //
    // Scope note: the Output's Surface (Mix Material / Material Ref) IS compiled
    // now - a per-pixel Mix lowers each covered slot to lerp(A.slot, B.slot, fac)
    // via the existing MixColor op (see compileMatSlot below), and a Mix side that
    // is a TEXTURED material lowers to a real per-pixel fetch of that material's
    // texture through its own UV transform. Slots beyond MatSlot::Count
    // (subsurface/clearcoat/resin/...) and the Opacity/Normal textures still keep
    // their folded constant / Fac-0.5 texture switch.

    inline MaterialProgram compileMaterialProgram(MaterialNodeGraphV2& graph, PrincipledBSDF* /*bound*/) {
        MaterialProgram prog;
        OutputNode* out = graph.findOutputNode();
        if (!out) return prog;

        struct SrcRef { NodeSystem::NodeBase* node; int outIndex; };
        auto sourceOf = [&](uint32_t inputPinId) -> SrcRef {
            NodeSystem::Pin* p = graph.getInputSource(inputPinId);
            NodeSystem::NodeBase* n = graph.getInputSourceNode(inputPinId);
            if (!p || !n) return { nullptr, -1 };
            int oi = 0;
            for (size_t i = 0; i < n->outputs.size(); ++i) {
                if (n->outputs[i].id == p->id) { oi = static_cast<int>(i); break; }
            }
            return { n, oi };
        };
        auto nodeType = [](NodeSystem::NodeBase* n) -> NodeType {
            auto* mn = dynamic_cast<MaterialNodeBase*>(n);
            return mn ? mn->materialNodeType : NodeType::Value;
        };
        auto isSupported = [](NodeType t) {
            switch (t) {
                case NodeType::Value: case NodeType::Color: case NodeType::TextureCoordinate:
                case NodeType::ImageTexture: case NodeType::Mapping: case NodeType::Noise:
                case NodeType::Voronoi: case NodeType::Checker: case NodeType::ColorRamp:
                case NodeType::MixColor: case NodeType::Invert: case NodeType::Gamma:
                case NodeType::Math: case NodeType::SeparateColor: case NodeType::CombineColor:
                case NodeType::Geometry: case NodeType::Clamp: case NodeType::FloatCurve: case NodeType::MapRange:
                case NodeType::BrightContrast: case NodeType::Bump:
                case NodeType::Fresnel: case NodeType::ObjectInfo: case NodeType::Attribute:
                case NodeType::Wave: case NodeType::Gradient: case NodeType::VectorMath:
                case NodeType::HueSaturation: case NodeType::RGBCurves:
                case NodeType::LayerWeight: case NodeType::AmbientOcclusion:
                case NodeType::Bevel:
                    return true;
                default: return false;  // MaterialRef / MixMaterial / Output
            }
        };

        // --- structural predicates over the source subtree ------------------
        // Every walk below is guarded by a RECURSION-STACK set. Graph::addLink refuses to
        // close a cycle, but a project saved before that check existed can carry one on
        // disk (deserialization rebuilds the link list directly), and an unguarded walk
        // over a cyclic graph does not produce a wrong picture — it recurses until the
        // stack is gone and takes the process with it. A cycle here simply reports
        // "not supported / not spatial", which drops the material back to the Faz-1 fold.
        std::unordered_set<NodeSystem::NodeBase*> supGuard, spatGuard, matGuard, nodeGuard;

        std::function<bool(NodeSystem::NodeBase*)> subtreeSupported =
            [&](NodeSystem::NodeBase* n) -> bool {
                if (!n || !isSupported(nodeType(n))) return false;
                if (!supGuard.insert(n).second) return false;   // cycle
                bool ok = true;
                for (const auto& in : n->inputs) {
                    SrcRef s = sourceOf(in.id);
                    if (s.node && !subtreeSupported(s.node)) { ok = false; break; }
                }
                supGuard.erase(n);
                return ok;
            };
        std::function<bool(NodeSystem::NodeBase*)> subtreeSpatial =
            [&](NodeSystem::NodeBase* n) -> bool {
                if (!n) return false;
                switch (nodeType(n)) {
                    case NodeType::TextureCoordinate: case NodeType::ImageTexture:
                    case NodeType::Noise: case NodeType::Voronoi: case NodeType::Checker:
                    case NodeType::Geometry: case NodeType::Fresnel:
                    case NodeType::Wave: case NodeType::Gradient:
                    case NodeType::LayerWeight:
                        return true;   // vary per surface point by construction
                    case NodeType::AmbientOcclusion:
                        // Folding this would be worse than useless: the fold's 5 sample points
                        // have no scene to trace against, so it would bake "1.0, everywhere
                        // open" into the slot and the dirt would silently not exist.
                        return true;
                    case NodeType::Bevel:
                        return true;   // same: only real at a real hit, with a real scene
                    case NodeType::ObjectInfo:
                        // Constant WITHIN one object, but the whole reason the node exists is
                        // that a single material is shared by many objects. Folding it would
                        // bake one object's value (really: the 5-sample average of it) into all
                        // of them and the scatter would come out uniform — so it has to compile
                        // to the program like anything else that cannot be reduced to a constant.
                        return true;
                    case NodeType::Attribute:
                        return true;   // per-vertex channel: varies across the surface by definition
                    default: break;
                }
                if (!spatGuard.insert(n).second) return false;   // cycle
                bool spatial = false;
                for (const auto& in : n->inputs) {
                    SrcRef s = sourceOf(in.id);
                    if (s.node && subtreeSpatial(s.node)) { spatial = true; break; }
                }
                spatGuard.erase(n);
                return spatial;
            };
        auto isDirectTextureBind = [&](int pin) -> bool {
            SrcRef s = sourceOf(out->inputs[pin].id);
            if (!s.node) return false;
            if (nodeType(s.node) == NodeType::ImageTexture) return true;
            if (nodeType(s.node) == NodeType::Mapping) {
                SrcRef inner = sourceOf(s.node->inputs[0].id);
                return inner.node && nodeType(inner.node) == NodeType::ImageTexture;
            }
            return false;
        };

        // --- codegen state ---------------------------------------------------
        int nextReg = 0;
        // UV bias for Bump finite-differencing: when >=0, every emitted UV is
        // shifted by this (du,dv,0) const register. biasEpoch namespaces the memo
        // so the biased height copies don't collide with the unbiased one.
        int uvBiasReg_ = -1;
        int biasEpoch_ = 0;
        std::unordered_map<uint64_t, int> memo;  // (nodeId<<16 | epoch<<8 | outIndex) -> reg
        auto emitConst3 = [&](float a, float b, float c) -> int {
            const int off = static_cast<int>(prog.constPool.size());
            prog.constPool.push_back(a); prog.constPool.push_back(b); prog.constPool.push_back(c);
            const int reg = nextReg++;
            MatInstr in; in.op = static_cast<uint16_t>(MatOp::Const); in.outReg = static_cast<int16_t>(reg); in.constOff = off;
            prog.instrs.push_back(in);
            return reg;
        };
        auto emitUV = [&]() -> int {
            const int reg = nextReg++;
            MatInstr in; in.op = static_cast<uint16_t>(MatOp::UV); in.outReg = static_cast<int16_t>(reg);
            prog.instrs.push_back(in);
            if (uvBiasReg_ < 0) return reg;
            // biased UV = UV + (du,dv,0)  via MixColor Add (fac=1)
            const int foff = static_cast<int>(prog.constPool.size());
            prog.constPool.push_back(1.0f); prog.constPool.push_back(1.0f); prog.constPool.push_back(1.0f);
            const int one = nextReg++;
            { MatInstr c; c.op = static_cast<uint16_t>(MatOp::Const); c.outReg = static_cast<int16_t>(one); c.constOff = foff; prog.instrs.push_back(c); }
            const int bReg = nextReg++;
            MatInstr add; add.op = static_cast<uint16_t>(MatOp::MixColor);
            add.outReg = static_cast<int16_t>(bReg);
            add.inReg[0] = static_cast<int16_t>(one);
            add.inReg[1] = static_cast<int16_t>(reg);
            add.inReg[2] = static_cast<int16_t>(uvBiasReg_);
            add.iparam = 1;  // MixColor Add
            prog.instrs.push_back(add);
            return bReg;
        };
        auto emitGeo = [&](MatOp op) -> int {   // GeoPosition / GeoNormal / GeoPointiness
            const int reg = nextReg++;
            MatInstr in; in.op = static_cast<uint16_t>(op); in.outReg = static_cast<int16_t>(reg);
            prog.instrs.push_back(in);
            // Pointiness is the one geometry channel that is not free at the shading
            // point: it needs a per-vertex precompute + an extra GPU vertex buffer.
            // Flag it so the geometry side only pays for it when a graph reads it.
            if (op == MatOp::GeoPointiness) prog.usesPointiness = true;
            return reg;
        };
        // Attribute node: intern the NAME into a scene-wide slot here, at compile time —
        // the runtime (and the shader especially) only ever sees the integer. Returns -1
        // when the name is empty or the 4 slots are exhausted; the caller then treats the
        // node as unsupported and the chain falls back to the fold, which is honest: a
        // silently-wrong slot would read some OTHER mask.
        auto emitAttribute = [&](const std::string& attrName) -> int {
            const int slot = materialAttributeSlot(attrName);
            if (slot < 0) return -1;
            const int reg = nextReg++;
            MatInstr in; in.op = static_cast<uint16_t>(MatOp::Attribute);
            in.outReg = static_cast<int16_t>(reg);
            in.aux = slot;
            prog.instrs.push_back(in);
            prog.usesAttributes = true;   // gates the per-vertex block (mesh cache + GPU upload)
            return reg;
        };
        // Scalar Math op (mode = MathNode::Op value): Clamp / Map Range lower to
        // chains of these, so the GPU VM needs no new opcode.
        auto emitMath = [&](int aReg, int bReg, int mode) -> int {
            const int reg = nextReg++;
            MatInstr in; in.op = static_cast<uint16_t>(MatOp::Math);
            in.outReg = static_cast<int16_t>(reg);
            in.inReg[0] = static_cast<int16_t>(aReg); in.inReg[1] = static_cast<int16_t>(bReg);
            in.iparam = mode;
            prog.instrs.push_back(in);
            return reg;
        };
        // Per-channel MixColor op (mode = MixColorNode mode: 0 Mix..5 Overlay).
        // With fac=1 and Add/Multiply this is a vec3 add/scale â€” Brightness/Contrast
        // lowers to two of these, so the GPU VM needs no new opcode.
        auto emitMix = [&](int facReg, int aReg, int bReg, int mode) -> int {
            const int reg = nextReg++;
            MatInstr in; in.op = static_cast<uint16_t>(MatOp::MixColor);
            in.outReg = static_cast<int16_t>(reg);
            in.inReg[0] = static_cast<int16_t>(facReg);
            in.inReg[1] = static_cast<int16_t>(aReg); in.inReg[2] = static_cast<int16_t>(bReg);
            in.iparam = mode;
            prog.instrs.push_back(in);
            return reg;
        };

        std::function<int(NodeSystem::NodeBase*, int)> compileNode =
            [&](NodeSystem::NodeBase* n, int outIndex) -> int {
            const uint64_t key = (static_cast<uint64_t>(n->id) << 16) |
                                 (static_cast<uint64_t>(biasEpoch_ & 0xFF) << 8) |
                                 static_cast<uint32_t>(outIndex & 0xFF);
            auto it = memo.find(key);
            if (it != memo.end()) return it->second;
            // The memo is only written AFTER the inputs compile, so it is no defence
            // against a cycle — this is. Break the loop with a black constant.
            if (!nodeGuard.insert(n).second) return emitConst3(0.0f, 0.0f, 0.0f);
            struct GuardPop {
                std::unordered_set<NodeSystem::NodeBase*>& s; NodeSystem::NodeBase* n;
                ~GuardPop() { s.erase(n); }
            } guardPop{ nodeGuard, n };

            // Compile input `idx`; if unconnected, bake its per-node fallback.
            auto inOr = [&](int idx, float fx, float fy, float fz) -> int {
                SrcRef s = sourceOf(n->inputs[idx].id);
                if (s.node) return compileNode(s.node, s.outIndex);
                return emitConst3(fx, fy, fz);
            };
            auto inUV = [&](int idx) -> int {
                SrcRef s = sourceOf(n->inputs[idx].id);
                return s.node ? compileNode(s.node, s.outIndex) : emitUV();
            };

            int resultReg = -1;
            switch (nodeType(n)) {
                case NodeType::Value: {
                    auto* v = static_cast<ValueNode*>(n);
                    resultReg = emitConst3(v->value, v->value, v->value);
                    break;
                }
                case NodeType::Color: {
                    auto* c = static_cast<ColorNode*>(n);
                    resultReg = emitConst3(c->color[0], c->color[1], c->color[2]);
                    break;
                }
                case NodeType::TextureCoordinate: {
                    resultReg = emitUV();
                    break;
                }
                case NodeType::Geometry: {
                    resultReg = emitGeo(outIndex == 1 ? MatOp::GeoNormal
                                      : outIndex == 2 ? MatOp::GeoPointiness
                                      : outIndex == 3 ? MatOp::GeoPositionObj
                                      : outIndex == 4 ? MatOp::GeoIncoming
                                                      : MatOp::GeoPosition);
                    break;
                }
                case NodeType::ObjectInfo: {
                    // Same shape as Geometry: a nullary op the runtime answers from the hit.
                    resultReg = emitGeo(outIndex == 1 ? MatOp::ObjRandom : MatOp::ObjLocation);
                    break;
                }
                case NodeType::Attribute: {
                    auto* an = static_cast<AttributeNode*>(n);
                    const int reg = emitAttribute(an->attributeName);
                    // Both outputs (Fac / Color) are the same broadcast value — the VM stores
                    // scalars splatted, so a Color read is a passthrough, no extra op.
                    resultReg = (reg >= 0) ? reg : emitConst3(0.0f, 0.0f, 0.0f);
                    break;
                }
                case NodeType::Wave: {
                    auto* wv = static_cast<WaveTextureNode*>(n);
                    const bool is3D = (wv->dimensions == 3);
                    int vecReg;
                    if (is3D) {
                        SrcRef s = sourceOf(n->inputs[0].id);
                        vecReg = s.node ? compileNode(s.node, s.outIndex)
                                        : emitGeo(wv->objectSpace ? MatOp::GeoPositionObj : MatOp::GeoPosition);
                    } else {
                        vecReg = inUV(0);
                    }
                    const int off = static_cast<int>(prog.constPool.size());
                    prog.constPool.insert(prog.constPool.end(),
                        { static_cast<float>(wv->waveType), static_cast<float>(wv->direction),
                          static_cast<float>(wv->profile), wv->scale, wv->distortion,
                          static_cast<float>(wv->detail), wv->detailScale, wv->phase });
                    resultReg = nextReg++;
                    MatInstr in; in.op = static_cast<uint16_t>(MatOp::Wave);
                    in.outReg = static_cast<int16_t>(resultReg);
                    in.inReg[0] = static_cast<int16_t>(vecReg);
                    in.constOff = off;
                    in.aux = is3D ? 3 : 2;
                    prog.instrs.push_back(in);
                    break;   // Fac and Color read the same broadcast register
                }
                case NodeType::Gradient: {
                    auto* gr = static_cast<GradientTextureNode*>(n);
                    const bool is3D = (gr->dimensions == 3);
                    int vecReg;
                    if (is3D) {
                        SrcRef s = sourceOf(n->inputs[0].id);
                        vecReg = s.node ? compileNode(s.node, s.outIndex)
                                        : emitGeo(gr->objectSpace ? MatOp::GeoPositionObj : MatOp::GeoPosition);
                    } else {
                        vecReg = inUV(0);
                    }
                    resultReg = nextReg++;
                    MatInstr in; in.op = static_cast<uint16_t>(MatOp::Gradient);
                    in.outReg = static_cast<int16_t>(resultReg);
                    in.inReg[0] = static_cast<int16_t>(vecReg);
                    in.iparam = gr->gradType;
                    prog.instrs.push_back(in);
                    break;
                }
                case NodeType::VectorMath: {
                    auto* vm = static_cast<VectorMathNode*>(n);
                    const int a = inOr(0, 0.0f, 0.0f, 0.0f);
                    const int b = inOr(1, 0.0f, 0.0f, 0.0f);
                    int off = -1;
                    if (vm->op == 10) {   // Scale is the only op with a constant operand
                        off = static_cast<int>(prog.constPool.size());
                        prog.constPool.push_back(vm->scaleValue);
                    }
                    resultReg = nextReg++;
                    MatInstr in; in.op = static_cast<uint16_t>(MatOp::VectorMath);
                    in.outReg = static_cast<int16_t>(resultReg);
                    in.inReg[0] = static_cast<int16_t>(a);
                    in.inReg[1] = static_cast<int16_t>(b);
                    in.constOff = off;
                    in.iparam = vm->op;
                    prog.instrs.push_back(in);
                    // Value output: Dot/Length/Distance already store a broadcast scalar, and
                    // for the vector ops a scalar read is the channel average (same as compute()).
                    break;
                }
                case NodeType::HueSaturation: {
                    const int hue = inOr(0, 0.5f, 0.5f, 0.5f);
                    const int sat = inOr(1, 1.0f, 1.0f, 1.0f);
                    const int val = inOr(2, 1.0f, 1.0f, 1.0f);
                    const int fac = inOr(3, 1.0f, 1.0f, 1.0f);
                    const int col = inOr(4, 0.8f, 0.8f, 0.8f);
                    // The op wants (h,s,v) in ONE register: build it from the three scalars with
                    // the existing Combine op rather than growing the instruction to four inputs.
                    const int hsv = nextReg++;
                    MatInstr cb; cb.op = static_cast<uint16_t>(MatOp::Combine);
                    cb.outReg = static_cast<int16_t>(hsv);
                    cb.inReg[0] = static_cast<int16_t>(hue);
                    cb.inReg[1] = static_cast<int16_t>(sat);
                    cb.inReg[2] = static_cast<int16_t>(val);
                    prog.instrs.push_back(cb);

                    resultReg = nextReg++;
                    MatInstr in; in.op = static_cast<uint16_t>(MatOp::HSV);
                    in.outReg = static_cast<int16_t>(resultReg);
                    in.inReg[0] = static_cast<int16_t>(col);
                    in.inReg[1] = static_cast<int16_t>(hsv);
                    in.inReg[2] = static_cast<int16_t>(fac);
                    prog.instrs.push_back(in);
                    break;
                }
                case NodeType::RGBCurves: {
                    // No new opcode: Separate -> (C curve -> channel curve) -> Combine, out of ops
                    // the VM already has. The curves are baked to uniform LUTs here, so the VM
                    // never learns about splines and the backends cannot interpolate differently.
                    auto* rc = static_cast<RGBCurvesNode*>(n);
                    const int fac = inOr(0, 1.0f, 1.0f, 1.0f);
                    const int col = inOr(1, 0.8f, 0.8f, 0.8f);

                    auto emitLut = [&](const RGBCurvesNode::Curve& cv, int srcReg) -> int {
                        const int off = static_cast<int>(prog.constPool.size());
                        for (int i = 0; i < FloatCurveNode::kLutSize; ++i) {
                            const float x = static_cast<float>(i) / static_cast<float>(FloatCurveNode::kLutSize - 1);
                            prog.constPool.push_back(evalCurvePoints(cv.points, cv.interpolation, x));
                        }
                        const int reg = nextReg++;
                        MatInstr in; in.op = static_cast<uint16_t>(MatOp::CurveLUT);
                        in.outReg = static_cast<int16_t>(reg);
                        in.inReg[0] = static_cast<int16_t>(srcReg);
                        in.constOff = off;
                        in.aux = FloatCurveNode::kLutSize;
                        prog.instrs.push_back(in);
                        return reg;
                    };

                    int graded[3];
                    for (int ch = 0; ch < 3; ++ch) {
                        const int sw = nextReg++;
                        MatInstr sz; sz.op = static_cast<uint16_t>(MatOp::Swizzle);
                        sz.outReg = static_cast<int16_t>(sw);
                        sz.inReg[0] = static_cast<int16_t>(col);
                        sz.iparam = ch;
                        prog.instrs.push_back(sz);
                        graded[ch] = emitLut(rc->curves[ch + 1], emitLut(rc->curves[0], sw));
                    }
                    const int mixed = nextReg++;
                    MatInstr cb; cb.op = static_cast<uint16_t>(MatOp::Combine);
                    cb.outReg = static_cast<int16_t>(mixed);
                    cb.inReg[0] = static_cast<int16_t>(graded[0]);
                    cb.inReg[1] = static_cast<int16_t>(graded[1]);
                    cb.inReg[2] = static_cast<int16_t>(graded[2]);
                    prog.instrs.push_back(cb);

                    resultReg = emitMix(fac, col, mixed, 0);   // Fac blends against the original
                    break;
                }
                case NodeType::Fresnel: {
                    // Unconnected IOR = the node's drag value (see the Bevel case below).
                    auto* fr = static_cast<FresnelNode*>(n);
                    const int ior = inOr(0, fr->ior, fr->ior, fr->ior);
                    int nrmReg = 0;
                    SrcRef ns = sourceOf(n->inputs[1].id);
                    if (ns.node) nrmReg = compileNode(ns.node, ns.outIndex);
                    else nrmReg = emitGeo(MatOp::GeoNormal);
                    
                    resultReg = nextReg++;
                    MatInstr in; in.op = static_cast<uint16_t>(MatOp::Fresnel);
                    in.outReg = static_cast<int16_t>(resultReg);
                    in.inReg[0] = static_cast<int16_t>(ior); in.inReg[1] = static_cast<int16_t>(nrmReg);
                    prog.instrs.push_back(in);
                    break;
                }
                case NodeType::LayerWeight: {
                    // Lowered onto existing ops — no new opcode. The formulas here ARE the
                    // node's compute() (LayerWeightNode), so the on-node preview, the fold and
                    // the render cannot drift apart.
                    // Unconnected Blend = the node's slider value (see the Bevel case below).
                    auto* lw = static_cast<LayerWeightNode*>(n);
                    const int blendReg = inOr(0, lw->blend, lw->blend, lw->blend);
                    SrcRef ns = sourceOf(n->inputs[1].id);
                    const int nrmRaw = ns.node ? compileNode(ns.node, ns.outIndex) : emitGeo(MatOp::GeoNormal);
                    const int one = emitConst3(1.0f, 1.0f, 1.0f);

                    // normalize(N) — a Normal socket can be fed any vector, and an
                    // unnormalized one would push |N.V| past 1 and invert the ramp.
                    const int nrmReg = [&] {
                        const int reg = nextReg++;
                        MatInstr v; v.op = static_cast<uint16_t>(MatOp::VectorMath);
                        v.outReg = static_cast<int16_t>(reg);
                        v.inReg[0] = static_cast<int16_t>(nrmRaw);
                        v.inReg[1] = static_cast<int16_t>(nrmRaw);
                        v.iparam = 6;   // Normalize
                        prog.instrs.push_back(v);
                        return reg;
                    }();

                    if (outIndex == 1) {
                        // Facing = pow(1 - |dot(N, V)|, (1 - blend) / max(blend, 1e-4))
                        const int viewReg = emitGeo(MatOp::GeoIncoming);
                        const int dotReg = nextReg++;
                        MatInstr d; d.op = static_cast<uint16_t>(MatOp::VectorMath);
                        d.outReg = static_cast<int16_t>(dotReg);
                        d.inReg[0] = static_cast<int16_t>(nrmReg);
                        d.inReg[1] = static_cast<int16_t>(viewReg);
                        d.iparam = 5;   // Dot (broadcasts the scalar)
                        prog.instrs.push_back(d);

                        const int absDot = emitMath(dotReg, dotReg, static_cast<int>(MathNode::Op::Absolute));
                        const int facing = emitMath(one, absDot, static_cast<int>(MathNode::Op::Subtract));
                        const int eps    = emitConst3(1e-4f, 1e-4f, 1e-4f);
                        const int bSafe  = emitMath(blendReg, eps, static_cast<int>(MathNode::Op::Maximum));
                        const int numer  = emitMath(one, blendReg, static_cast<int>(MathNode::Op::Subtract));
                        const int expo   = emitMath(numer, bSafe, static_cast<int>(MathNode::Op::Divide));
                        resultReg = emitMath(facing, expo, static_cast<int>(MathNode::Op::Power));
                    } else {
                        // Fresnel with IOR = 1 / (1 - blend)  (blend 0.5 -> IOR 2, Blender's default)
                        const int epsD  = emitConst3(1e-5f, 1e-5f, 1e-5f);
                        const int denom = emitMath(one, blendReg, static_cast<int>(MathNode::Op::Subtract));
                        const int dSafe = emitMath(denom, epsD, static_cast<int>(MathNode::Op::Maximum));
                        const int iorReg = emitMath(one, dSafe, static_cast<int>(MathNode::Op::Divide));

                        resultReg = nextReg++;
                        MatInstr in; in.op = static_cast<uint16_t>(MatOp::Fresnel);
                        in.outReg = static_cast<int16_t>(resultReg);
                        in.inReg[0] = static_cast<int16_t>(iorReg);
                        in.inReg[1] = static_cast<int16_t>(nrmReg);
                        prog.instrs.push_back(in);
                    }
                    break;
                }
                case NodeType::Bevel: {
                    auto* bn = static_cast<BevelNode*>(n);
                    // Unconnected Radius = the node's own drag value, NOT a hardcoded
                    // constant. The first Bevel shipped with inOr(0, 0.05f, ...) and no
                    // radius widget at all: the only way to change the radius was wiring
                    // a Value node, and the on-node slider people actually reached for
                    // (Samples) changed nothing visible — 0.05 world units on a 2m cube
                    // is a 2.5% band. Same trap fixed on the AO node's Distance below.
                    const int radReg = inOr(0, bn->radius, bn->radius, bn->radius);
                    const int off = static_cast<int>(prog.constPool.size());
                    prog.constPool.push_back(static_cast<float>(std::clamp(bn->samples, 1, 16)));

                    resultReg = nextReg++;
                    MatInstr in; in.op = static_cast<uint16_t>(MatOp::Bevel);
                    in.outReg = static_cast<int16_t>(resultReg);
                    in.inReg[0] = static_cast<int16_t>(radReg);
                    in.constOff = off;
                    prog.instrs.push_back(in);
                    prog.usesBevel = true;
                    // Register value is a WORLD-space normal. Wired straight into the
                    // Output's Normal Map pin, the slot pass below emits StoreWorldNormal
                    // for it; used mid-chain it is just a vector (dot it with Geometry >
                    // Normal for a free edge mask).
                    break;
                }
                case NodeType::AmbientOcclusion: {
                    auto* aon = static_cast<AmbientOcclusionNode*>(n);
                    // Unconnected Distance = the node's drag value (see the Bevel case above).
                    const int distReg = inOr(1, aon->distance, aon->distance, aon->distance);
                    const int off = static_cast<int>(prog.constPool.size());
                    prog.constPool.insert(prog.constPool.end(),
                        { static_cast<float>(std::clamp(aon->samples, 1, 64)), aon->inside ? 1.0f : 0.0f });

                    const int aoReg = nextReg++;
                    MatInstr in; in.op = static_cast<uint16_t>(MatOp::AmbientOcclusion);
                    in.outReg = static_cast<int16_t>(aoReg);
                    in.inReg[0] = static_cast<int16_t>(distReg);
                    in.constOff = off;
                    prog.instrs.push_back(in);
                    prog.usesAO = true;

                    if (outIndex == 1) {          // AO (fac)
                        resultReg = aoReg;
                    } else {                      // Color = input color * AO — no new opcode
                        const int colReg = inOr(0, 1.0f, 1.0f, 1.0f);
                        resultReg = emitMath(colReg, aoReg, static_cast<int>(MathNode::Op::Multiply));
                    }
                    break;
                }
                case NodeType::Mapping: {
                    auto* m = static_cast<MappingNode*>(n);
                    const int uvReg = inUV(0);
                    const int off = static_cast<int>(prog.constPool.size());
                    prog.constPool.insert(prog.constPool.end(),
                        { m->scale[0], m->scale[1], m->offset[0], m->offset[1], m->rotationDeg });
                    resultReg = nextReg++;
                    MatInstr in; in.op = static_cast<uint16_t>(MatOp::Mapping);
                    in.outReg = static_cast<int16_t>(resultReg); in.inReg[0] = static_cast<int16_t>(uvReg); in.constOff = off;
                    prog.instrs.push_back(in);
                    break;
                }
                case NodeType::ImageTexture: {
                    auto* img = static_cast<ImageTextureNode*>(n);
                    img->resolveIfNeeded();
                    const int texIdx = static_cast<int>(prog.textures.size());
                    prog.textures.push_back(img->texture);
                    const int uvReg = inUV(0);
                    resultReg = nextReg++;
                    MatInstr in;
                    in.op = static_cast<uint16_t>(outIndex == 1 ? MatOp::TexAlpha : MatOp::TexColor);
                    in.outReg = static_cast<int16_t>(resultReg); in.inReg[0] = static_cast<int16_t>(uvReg); in.aux = texIdx;
                    prog.instrs.push_back(in);
                    break;
                }
                case NodeType::Noise: {
                    // Unified procedural texture (kind = FBM/Ridge/Billow/Warped/
                    // Voronoi/Checker). NoiseTextureNode::Kind values line up 1:1
                    // with the VM Noise op's kind constants. Pins: 0=Fac, 1=Color.
                    auto* nz = static_cast<NoiseTextureNode*>(n);
                    const bool wantColor = (outIndex == 1);
                    const bool voronoi = (nz->kind == NoiseTextureNode::Kind::Voronoi);
                    const bool is3D = (nz->dimensions == 3);
                    // 3D noise is driven by the shading-point position (seamless
                    // solid texturing); a wired Vector overrides it. 2D uses UV.
                    int uvReg;
                    if (is3D) {
                        SrcRef s = sourceOf(n->inputs[0].id);
                        uvReg = s.node ? compileNode(s.node, s.outIndex)
                                       : emitGeo(nz->objectSpace ? MatOp::GeoPositionObj : MatOp::GeoPosition);
                    } else {
                        uvReg = inUV(0);
                    }
                    int c1 = -1, c2 = -1;
                    if (wantColor && !voronoi) {   // Color = lerp(Color1, Color2, Fac)
                        c1 = inOr(1, nz->color1[0], nz->color1[1], nz->color1[2]);
                        c2 = inOr(2, nz->color2[0], nz->color2[1], nz->color2[2]);
                    }
                    const int off = static_cast<int>(prog.constPool.size());
                    // kind,scale,detail,rough,rand,dist,seed
                    prog.constPool.insert(prog.constPool.end(),
                        { static_cast<float>(static_cast<int>(nz->kind)), nz->scale,
                          static_cast<float>(nz->detail), nz->rough, nz->randomness, nz->distortion,
                          static_cast<float>(nz->seed) });
                    resultReg = nextReg++;
                    MatInstr in; in.op = static_cast<uint16_t>(MatOp::Noise);
                    in.outReg = static_cast<int16_t>(resultReg); in.inReg[0] = static_cast<int16_t>(uvReg);
                    in.constOff = off;
                    in.aux = is3D ? 3 : 2;             // 2D / 3D noise dimensions
                    if (!wantColor)   in.iparam = 0;   // Fac (splat)
                    else if (voronoi) in.iparam = 1;   // per-cell random color
                    else { in.inReg[1] = static_cast<int16_t>(c1); in.inReg[2] = static_cast<int16_t>(c2); in.iparam = 2; }
                    prog.instrs.push_back(in);
                    break;
                }
                case NodeType::ColorRamp: {
                    auto* rp = static_cast<ColorRampNode*>(n);
                    const int facReg = inOr(0, 0.5f, 0.5f, 0.5f);
                    std::vector<ColorRampNode::Stop> sorted = rp->stops;
                    std::sort(sorted.begin(), sorted.end(),
                              [](const ColorRampNode::Stop& a, const ColorRampNode::Stop& b) { return a.pos < b.pos; });
                    const int off = static_cast<int>(prog.constPool.size());
                    for (const auto& s : sorted) {
                        prog.constPool.push_back(s.pos);
                        prog.constPool.push_back(s.col[0]); prog.constPool.push_back(s.col[1]); prog.constPool.push_back(s.col[2]);
                    }
                    resultReg = nextReg++;
                    MatInstr in; in.op = static_cast<uint16_t>(MatOp::ColorRamp);
                    in.outReg = static_cast<int16_t>(resultReg); in.inReg[0] = static_cast<int16_t>(facReg);
                    in.constOff = off; in.aux = static_cast<int>(sorted.size()); in.iparam = rp->interpolation;
                    prog.instrs.push_back(in);
                    break;
                }
                case NodeType::MixColor: {
                    auto* mx = static_cast<MixColorNode*>(n);
                    const int fac = inOr(0, 0.5f, 0.5f, 0.5f);
                    const int a = inOr(1, 0.5f, 0.5f, 0.5f);
                    const int b = inOr(2, 0.5f, 0.5f, 0.5f);
                    resultReg = nextReg++;
                    MatInstr in; in.op = static_cast<uint16_t>(MatOp::MixColor);
                    in.outReg = static_cast<int16_t>(resultReg);
                    in.inReg[0] = static_cast<int16_t>(fac); in.inReg[1] = static_cast<int16_t>(a); in.inReg[2] = static_cast<int16_t>(b);
                    in.iparam = mx->mode;
                    prog.instrs.push_back(in);
                    break;
                }
                case NodeType::Invert: {
                    const int fac = inOr(0, 1.0f, 1.0f, 1.0f);
                    const int col = inOr(1, 0.5f, 0.5f, 0.5f);
                    resultReg = nextReg++;
                    MatInstr in; in.op = static_cast<uint16_t>(MatOp::Invert);
                    in.outReg = static_cast<int16_t>(resultReg);
                    in.inReg[0] = static_cast<int16_t>(fac); in.inReg[1] = static_cast<int16_t>(col);
                    prog.instrs.push_back(in);
                    break;
                }
                case NodeType::Gamma: {
                    auto* gm = static_cast<GammaNode*>(n);
                    const int col = inOr(0, 0.5f, 0.5f, 0.5f);
                    const int off = static_cast<int>(prog.constPool.size());
                    prog.constPool.push_back(gm->gamma);
                    resultReg = nextReg++;
                    MatInstr in; in.op = static_cast<uint16_t>(MatOp::Gamma);
                    in.outReg = static_cast<int16_t>(resultReg); in.inReg[0] = static_cast<int16_t>(col); in.constOff = off;
                    prog.instrs.push_back(in);
                    break;
                }
                case NodeType::Math: {
                    auto* mt = static_cast<MathNode*>(n);
                    const int a = inOr(0, mt->defaultA, mt->defaultA, mt->defaultA);
                    const int b = inOr(1, mt->defaultB, mt->defaultB, mt->defaultB);
                    resultReg = nextReg++;
                    MatInstr in; in.op = static_cast<uint16_t>(MatOp::Math);
                    in.outReg = static_cast<int16_t>(resultReg);
                    in.inReg[0] = static_cast<int16_t>(a); in.inReg[1] = static_cast<int16_t>(b); in.iparam = mt->op;
                    prog.instrs.push_back(in);
                    break;
                }
                case NodeType::Clamp: {
                    // clamp(v, lo, hi) = min(max(v, lo), hi). Uses Min=7, Max=8.
                    auto* cn = static_cast<ClampNode*>(n);
                    const int v  = inOr(0, 0.0f, 0.0f, 0.0f);
                    const int lo = inOr(1, cn->minVal, cn->minVal, cn->minVal);
                    const int hi = inOr(2, cn->maxVal, cn->maxVal, cn->maxVal);
                    const int t  = emitMath(v, lo, 8);   // max(v, lo)
                    resultReg    = emitMath(t, hi, 7);   // min(.., hi)
                    break;
                }
                case NodeType::FloatCurve: {
                    // Bake the curve to a uniform LUT instead of shipping the control
                    // points to the VM. Three things fall out of that:
                    //   - Smooth (monotone-cubic) costs the renderer nothing: the spline
                    //     is evaluated here, at compile time, never per pixel.
                    //   - CPU and GPU cannot disagree — neither one knows what a spline is.
                    //   - It ends a real bug: this used to ride on the ColorRamp op with
                    //     `min(points, 8)`, so a curve with 9..16 points (the editor lets
                    //     you place 16) rendered with points 9+ silently dropped, while
                    //     the node's own preview drew all of them.
                    // O(1) at runtime — cheaper than the O(n) stop search it replaces.
                    auto* fc = static_cast<FloatCurveNode*>(n);
                    const int facReg = inOr(0, 0.5f, 0.5f, 0.5f);

                    std::vector<FloatCurveNode::Point> sorted = fc->points;
                    std::sort(sorted.begin(), sorted.end(),
                              [](const FloatCurveNode::Point& a, const FloatCurveNode::Point& b) { return a.x < b.x; });

                    const int N = FloatCurveNode::kLutSize;
                    const int off = static_cast<int>(prog.constPool.size());
                    for (int i = 0; i < N; ++i) {
                        const float x = static_cast<float>(i) / static_cast<float>(N - 1);
                        prog.constPool.push_back(FloatCurveNode::evalCurveOn(sorted, fc->interpolation, x));
                    }
                    resultReg = nextReg++;
                    MatInstr in; in.op = static_cast<uint16_t>(MatOp::CurveLUT);
                    in.outReg = static_cast<int16_t>(resultReg);
                    in.inReg[0] = static_cast<int16_t>(facReg);
                    in.constOff = off;
                    in.aux = N;
                    prog.instrs.push_back(in);
                    break;
                }
                case NodeType::MapRange: {
                    // tl + clamp01?(v-fl)/(fh-fl) * (th-tl). Ops: Sub=1,Mul=2,Div=3,Add=0.
                    auto* mr = static_cast<MapRangeNode*>(n);
                    const int v  = inOr(0, 0.0f, 0.0f, 0.0f);
                    const int fl = inOr(1, mr->fromMin, mr->fromMin, mr->fromMin);
                    const int fh = inOr(2, mr->fromMax, mr->fromMax, mr->fromMax);
                    const int tl = inOr(3, mr->toMin,   mr->toMin,   mr->toMin);
                    const int th = inOr(4, mr->toMax,   mr->toMax,   mr->toMax);
                    const int num   = emitMath(v, fl, 1);      // v - fl
                    const int span  = emitMath(fh, fl, 1);     // fh - fl
                    const int t     = emitMath(num, span, 3);  // / (Divide guards /0)
                    const int range = emitMath(th, tl, 1);     // th - tl
                    const int scl   = emitMath(t, range, 2);   // t * range
                    int res         = emitMath(scl, tl, 0);    // + tl
                    if (mr->clampResult) {
                        // clamp to [min(tl,th), max(tl,th)]
                        const int mn = emitMath(tl, th, 7);    // min(tl, th)
                        const int mx = emitMath(tl, th, 8);    // max(tl, th)
                        const int c1 = emitMath(res, mn, 8);   // max(res, mn)
                        res          = emitMath(c1, mx, 7);    // min(.., mx)
                    }
                    resultReg = res;
                    break;
                }
                case NodeType::BrightContrast: {
                    // out = a*color + b (a=1+contrast, b=bright-contrast*0.5).
                    // fac=1 MixColor: Multiply then Add. Params are compile-time consts.
                    auto* bc = static_cast<BrightContrastNode*>(n);
                    const int col = inOr(0, 0.0f, 0.0f, 0.0f);
                    const float a = 1.0f + bc->contrast;
                    const float b = bc->brightness - bc->contrast * 0.5f;
                    const int one  = emitConst3(1.0f, 1.0f, 1.0f);
                    const int aReg = emitConst3(a, a, a);
                    const int bReg = emitConst3(b, b, b);
                    const int mul  = emitMix(one, col, aReg, 2);   // color * a (MixColor Multiply)
                    resultReg      = emitMix(one, mul, bReg, 1);   // + b       (MixColor Add)
                    break;
                }
                case NodeType::Bump: {
                    // Finite-difference the Height chain at UV, UV+(d,0), UV+(0,d);
                    // tangent normal = ((h0-hu), (h0-hv), 1)*k  (k = strength/d).
                    // Compiled 3x with UV bias (const) â€” no runtime tangents.
                    auto* bp = static_cast<BumpNode*>(n);
                    SrcRef hs = sourceOf(n->inputs[0].id);
                    if (!hs.node) { resultReg = emitConst3(0.0f, 0.0f, 1.0f); break; }  // flat
                    const float d = std::max(bp->distance, 5e-4f);
                    float k = bp->strength / d;
                    if (bp->invert) k = -k;
                    // h0 (unbiased)
                    biasEpoch_ = 0; uvBiasReg_ = -1;
                    const int h0 = compileNode(hs.node, hs.outIndex);
                    // hu (UV + (d,0,0))
                    const int biasU = emitConst3(d, 0.0f, 0.0f);
                    uvBiasReg_ = biasU; biasEpoch_ = 1;
                    const int hu = compileNode(hs.node, hs.outIndex);
                    // hv (UV + (0,d,0))
                    const int biasV = emitConst3(0.0f, d, 0.0f);
                    uvBiasReg_ = biasV; biasEpoch_ = 2;
                    const int hv = compileNode(hs.node, hs.outIndex);
                    uvBiasReg_ = -1; biasEpoch_ = 0;
                    const int kReg = emitConst3(k, k, k);
                    const int dx = emitMath(emitMath(h0, hu, 1), kReg, 2);  // (h0-hu)*k
                    const int dy = emitMath(emitMath(h0, hv, 1), kReg, 2);  // (h0-hv)*k
                    const int one = emitConst3(1.0f, 1.0f, 1.0f);
                    resultReg = nextReg++;
                    MatInstr cmb; cmb.op = static_cast<uint16_t>(MatOp::Combine);
                    cmb.outReg = static_cast<int16_t>(resultReg);
                    cmb.inReg[0] = static_cast<int16_t>(dx);
                    cmb.inReg[1] = static_cast<int16_t>(dy);
                    cmb.inReg[2] = static_cast<int16_t>(one);
                    prog.instrs.push_back(cmb);
                    break;
                }
                case NodeType::SeparateColor: {
                    const int col = inOr(0, 0.0f, 0.0f, 0.0f);
                    resultReg = nextReg++;
                    MatInstr in; in.op = static_cast<uint16_t>(MatOp::Swizzle);
                    in.outReg = static_cast<int16_t>(resultReg); in.inReg[0] = static_cast<int16_t>(col);
                    in.iparam = std::clamp(outIndex, 0, 2);
                    prog.instrs.push_back(in);
                    break;
                }
                case NodeType::CombineColor: {
                    const int r = inOr(0, 0.0f, 0.0f, 0.0f);
                    const int g = inOr(1, 0.0f, 0.0f, 0.0f);
                    const int b = inOr(2, 0.0f, 0.0f, 0.0f);
                    resultReg = nextReg++;
                    MatInstr in; in.op = static_cast<uint16_t>(MatOp::Combine);
                    in.outReg = static_cast<int16_t>(resultReg);
                    in.inReg[0] = static_cast<int16_t>(r); in.inReg[1] = static_cast<int16_t>(g); in.inReg[2] = static_cast<int16_t>(b);
                    prog.instrs.push_back(in);
                    break;
                }
                default:
                    resultReg = emitConst3(0.0f, 0.0f, 0.0f);  // unreachable (subtreeSupported gated)
                    break;
            }
            memo[key] = resultReg;
            return resultReg;
        };

        // --- per-pixel Mix Material (Output.Surface) -------------------------
        // The Surface input takes a MATERIAL (Mix Material / Material Ref). Since
        // the VM drives uber-material INPUTS rather than whole BSDFs, a per-pixel
        // Mix lowers each slot to  lerp(A.slot, B.slot, facPerPixel)  using the
        // EXISTING MixColor op â€” no new opcode, so the already-compiled closesthit
        // VM runs it unchanged. Material Ref slots are constants; the per-pixel
        // variation comes from the Mix's Fac chain (e.g. a Noise mask). Texture-
        // bearing slots can't blend in the VM yet (Faz 2c) â€” those keep the fold's
        // Fac-0.5 texture switch and are left undriven here.
        auto slotVec = [](const ShadeState& s, MatSlot slot) -> std::array<float, 3> {
            switch (slot) {
                case MatSlot::BaseColor:     return { (float)s.baseColor.x, (float)s.baseColor.y, (float)s.baseColor.z };
                case MatSlot::EmissionColor: return { (float)s.emissionColor.x, (float)s.emissionColor.y, (float)s.emissionColor.z };
                case MatSlot::Metallic:         return { s.metallic, s.metallic, s.metallic };
                case MatSlot::Roughness:        return { s.roughness, s.roughness, s.roughness };
                case MatSlot::Specular:         return { s.specular, s.specular, s.specular };
                case MatSlot::Transmission:     return { s.transmission, s.transmission, s.transmission };
                case MatSlot::EmissionStrength: return { s.emissionStrength, s.emissionStrength, s.emissionStrength };
                case MatSlot::Opacity:          return { s.opacity, s.opacity, s.opacity };
                case MatSlot::IOR:              return { s.ior, s.ior, s.ior };
                default:                        return { 0.0f, 0.0f, 0.0f };
            }
        };
        auto slotHasTex = [](const ShadeState& s, MatSlot slot) -> bool {
            switch (slot) {
                case MatSlot::BaseColor:     return (bool)s.baseColorTex;
                case MatSlot::Roughness:     return (bool)s.roughnessTex;
                case MatSlot::Metallic:      return (bool)s.metallicTex;
                case MatSlot::EmissionColor: return (bool)s.emissionTex;
                case MatSlot::Opacity:       return (bool)s.opacityTex;
                default:                     return false;
            }
        };
        // The referenced material's OWN texture for this slot, or null. Only the slots
        // the VM can reproduce faithfully are listed: Opacity is deliberately absent
        // because get_opacity() reads a texture through a different path (alpha, not
        // rgb), and guessing wrong there would look like a fixed bug rather than an
        // unimplemented one. An unlisted textured slot falls back to the Faz-1 fold.
        auto slotTexture = [](const PrincipledBSDF& m, MatSlot slot) -> std::shared_ptr<Texture> {
            switch (slot) {
                case MatSlot::BaseColor:     return m.albedoProperty.texture;
                case MatSlot::Roughness:     return m.roughnessProperty.texture;
                case MatSlot::Metallic:      return m.metallicProperty.texture;
                case MatSlot::EmissionColor: return m.emissionProperty.texture;
                default:                     return nullptr;
            }
        };
        // Sample a material's texture the way the material itself would: through ITS
        // texture transform (scale/rotate/translate/tiling from the material panel),
        // which is a different chain than the Mapping node's — hence MatOp::MatMapping.
        auto emitMaterialTex = [&](const PrincipledBSDF& m, const std::shared_ptr<Texture>& tex) -> int {
            const int uvReg = emitUV();
            const auto& xf = m.textureTransform;
            const int off = static_cast<int>(prog.constPool.size());
            prog.constPool.insert(prog.constPool.end(), {
                static_cast<float>(xf.scale.u),       static_cast<float>(xf.scale.v),
                static_cast<float>(xf.translation.u), static_cast<float>(xf.translation.v),
                xf.rotation_degrees,
                static_cast<float>(xf.tilingFactor.u), static_cast<float>(xf.tilingFactor.v)
            });
            const int mapReg = nextReg++;
            { MatInstr in; in.op = static_cast<uint16_t>(MatOp::MatMapping);
              in.outReg = static_cast<int16_t>(mapReg); in.inReg[0] = static_cast<int16_t>(uvReg);
              in.constOff = off; prog.instrs.push_back(in); }

            const int texIdx = static_cast<int>(prog.textures.size());
            prog.textures.push_back(tex);
            const int reg = nextReg++;
            { MatInstr in; in.op = static_cast<uint16_t>(MatOp::TexColor);
              in.outReg = static_cast<int16_t>(reg); in.inReg[0] = static_cast<int16_t>(mapReg);
              in.aux = texIdx; prog.instrs.push_back(in); }
            return reg;
        };

        struct SlotRes { int reg; bool spatial; bool hasTex; bool ok; };
        // `texAllowed`: emit the referenced material's texture as real per-pixel
        // instructions rather than reporting hasTex and falling back to the fold.
        //
        // Only true UNDER a Mix Material. A lone Material Ref already has its texture
        // bound losslessly to the slot by evaluate's bind pass, so compiling it again
        // here would just be a second, redundant sampler doing the same work. Under a
        // Mix there is no such option: one material slot can hold one texture, so the
        // fold had to pick A's or B's at Fac 0.5 and the blend was a hard switch. This
        // is what "no real multi-material mixing" was.
        std::function<SlotRes(NodeSystem::NodeBase*, MatSlot, bool)> compileMatSlot =
            [&](NodeSystem::NodeBase* n, MatSlot slot, bool texAllowed) -> SlotRes {
            if (!n) return { -1, false, false, false };
            if (!matGuard.insert(n).second) return { -1, false, false, false };   // cycle
            struct GuardPop {
                std::unordered_set<NodeSystem::NodeBase*>& s; NodeSystem::NodeBase* n;
                ~GuardPop() { s.erase(n); }
            } guardPop{ matGuard, n };
            switch (nodeType(n)) {
                case NodeType::MaterialRef: {
                    auto* ref = static_cast<MaterialRefNode*>(n);
                    auto& mm = MaterialManager::getInstance();
                    const uint16_t id = mm.getMaterialID(ref->materialName);
                    auto* pb = dynamic_cast<PrincipledBSDF*>(
                        id != MaterialManager::INVALID_MATERIAL_ID ? mm.getMaterial(id) : nullptr);
                    if (!pb) return { -1, false, false, false };

                    if (texAllowed) {
                        auto tex = slotTexture(*pb, slot);
                        if (tex && tex->is_loaded()) {
                            // Spatial by construction: it varies per shading point.
                            return { emitMaterialTex(*pb, tex), true, false, true };
                        }
                    }
                    const ShadeState s = makeShadeStateFromMaterial(*pb);
                    const auto v = slotVec(s, slot);
                    return { emitConst3(v[0], v[1], v[2]), false, slotHasTex(s, slot), true };
                }
                case NodeType::MixMaterial: {
                    auto* mix = static_cast<MixMaterialNode*>(n);
                    const SlotRes a = compileMatSlot(sourceOf(mix->inputs[1].id).node, slot, true);
                    const SlotRes b = compileMatSlot(sourceOf(mix->inputs[2].id).node, slot, true);
                    if (!a.ok && !b.ok) return { -1, false, false, false };
                    if (!a.ok) return b;   // only Material B connected
                    if (!b.ok) return a;   // only Material A connected
                    SrcRef fs = sourceOf(mix->inputs[0].id);
                    int facReg; bool facSpatial = false;
                    if (fs.node && subtreeSupported(fs.node)) {
                        facReg = compileNode(fs.node, fs.outIndex);
                        facSpatial = subtreeSpatial(fs.node);
                    } else {
                        facReg = emitConst3(0.5f, 0.5f, 0.5f);  // unconnected Fac = pin default
                    }
                    const int reg = nextReg++;
                    MatInstr in; in.op = static_cast<uint16_t>(MatOp::MixColor);
                    in.outReg = static_cast<int16_t>(reg);
                    in.inReg[0] = static_cast<int16_t>(facReg);
                    in.inReg[1] = static_cast<int16_t>(a.reg);
                    in.inReg[2] = static_cast<int16_t>(b.reg);
                    in.iparam = 0;  // Mix (lerp), matches MixColorNode mode 0
                    prog.instrs.push_back(in);
                    return { reg, facSpatial || a.spatial || b.spatial, a.hasTex || b.hasTex, true };
                }
                default:
                    return { -1, false, false, false };  // not a Material producer
            }
        };
        NodeSystem::NodeBase* surfaceNode = sourceOf(out->inputs[OutputNode::InSurface].id).node;

        struct SlotMap { int pin; MatSlot slot; };
        static const SlotMap kSlots[] = {
            { OutputNode::InBaseColor,        MatSlot::BaseColor },
            { OutputNode::InMetallic,         MatSlot::Metallic },
            { OutputNode::InRoughness,        MatSlot::Roughness },
            { OutputNode::InSpecular,         MatSlot::Specular },
            { OutputNode::InTransmission,     MatSlot::Transmission },
            { OutputNode::InEmissionColor,    MatSlot::EmissionColor },
            { OutputNode::InEmissionStrength, MatSlot::EmissionStrength },
            { OutputNode::InOpacity,          MatSlot::Opacity },
            { OutputNode::InIOR,              MatSlot::IOR },
        };
        auto storeSlot = [&](int reg, MatSlot slot) {
            MatInstr st; st.op = static_cast<uint16_t>(MatOp::Store);
            st.inReg[0] = static_cast<int16_t>(reg); st.aux = static_cast<int>(slot);
            prog.instrs.push_back(st);
            prog.drivenSlots |= (1u << static_cast<uint32_t>(slot));
        };
        for (const auto& sm : kSlots) {
            SrcRef s = sourceOf(out->inputs[sm.pin].id);
            if (s.node) {
                // A per-slot pin drives this slot (overrides Surface, like Output::compute).
                if (isDirectTextureBind(sm.pin)) continue;   // handled losslessly by evaluate's bind pass
                if (!subtreeSupported(s.node)) continue;     // contains Mix/Ref/etc. -> keep fold
                if (!subtreeSpatial(s.node)) continue;       // pure constant -> Faz-1 fold already exact
                storeSlot(compileNode(s.node, s.outIndex), sm.slot);
                continue;
            }
            // No per-slot pin: if Surface is a Mix Material / Material Ref, lower it to
            // a per-pixel lerp for this slot. `hasTex` now only survives for slots the
            // VM can't reproduce (Opacity, and anything past MatSlot::Count) — those
            // still fall back to the fold's Fac-0.5 texture switch.
            if (surfaceNode) {
                const SlotRes r = compileMatSlot(surfaceNode, sm.slot, false);
                if (r.ok && r.spatial && !r.hasTex) {
                    storeSlot(r.reg, sm.slot);
                }
            }
        }

        // Normal Map slot: a Bump node here drives a per-pixel tangent-space normal
        // (MatSlot::Normal), consumed by apply_normal_map / closesthit. A direct
        // Image Texture stays a losslessly-bound normal map (evaluate's bind pass).
        // A direct BEVEL node stores a WORLD-space normal instead (StoreWorldNormal) —
        // its output already lives in world space and the TBN must not touch it.
        {
            SrcRef ns = sourceOf(out->inputs[OutputNode::InNormalMap].id);
            if (ns.node && nodeType(ns.node) == NodeType::Bump) {
                SrcRef hs = sourceOf(ns.node->inputs[0].id);
                if (hs.node && subtreeSupported(hs.node) && subtreeSpatial(hs.node)) {
                    storeSlot(compileNode(ns.node, ns.outIndex), MatSlot::Normal);
                }
            } else if (ns.node && nodeType(ns.node) == NodeType::Bevel) {
                const int reg = compileNode(ns.node, ns.outIndex);
                MatInstr st; st.op = static_cast<uint16_t>(MatOp::StoreWorldNormal);
                st.inReg[0] = static_cast<int16_t>(reg);
                prog.instrs.push_back(st);
                prog.drivenSlots |= (1u << static_cast<uint32_t>(MatSlot::Normal));
            }
        }

        // ── Register live-range compaction (post-pass) ──────────────────────
        // The emitter above allocates one fresh register per produced value
        // (single assignment, defs strictly before uses). Remap to a compact
        // set by releasing a register after its last reader: regCount drops
        // from "number of instructions" to the peak number of simultaneously
        // live values (≈ expression depth). This is what lets both VMs keep a
        // small fixed register file — the GPU interpreter's regs[] is a
        // dynamically indexed local array, i.e. per-thread scratch memory, and
        // its size is paid by every closesthit invocation whether or not the
        // material even has a program (see material_program.glsl MP_MAX_REGS).
        // Both VMs read all inputs before writing the output, so an output may
        // safely reuse a register freed by this same instruction's input.
        {
            const int n = static_cast<int>(prog.instrs.size());
            std::vector<int> lastUse(static_cast<size_t>(nextReg), -1);
            std::vector<char> defined(static_cast<size_t>(nextReg), 0);
            bool ssaOk = true;
            for (int i = 0; i < n && ssaOk; ++i) {
                const MatInstr& ins = prog.instrs[i];
                for (int s = 0; s < 3; ++s) {
                    const int r = ins.inReg[s];
                    if (r < 0) continue;
                    if (r >= nextReg || !defined[r]) { ssaOk = false; break; }
                    lastUse[r] = i;
                }
                if (ins.outReg >= 0) {
                    if (ins.outReg >= nextReg || defined[ins.outReg]) ssaOk = false;
                    else defined[ins.outReg] = 1;
                }
            }
            if (ssaOk) {
                std::vector<int16_t> remap(static_cast<size_t>(nextReg), -1);
                std::vector<int16_t> freeRegs;
                int16_t peak = 0;
                for (int i = 0; i < n; ++i) {
                    MatInstr& ins = prog.instrs[i];
                    const int orig[3] = { ins.inReg[0], ins.inReg[1], ins.inReg[2] };
                    for (int s = 0; s < 3; ++s) {
                        if (orig[s] >= 0) ins.inReg[s] = remap[orig[s]];
                    }
                    for (int s = 0; s < 3; ++s) {
                        const int r = orig[s];
                        if (r < 0 || lastUse[r] != i) continue;
                        bool dup = false;   // same reg feeding two input slots frees once
                        for (int q = 0; q < s; ++q) if (orig[q] == r) dup = true;
                        if (!dup) freeRegs.push_back(remap[r]);
                    }
                    if (ins.outReg >= 0) {
                        int16_t nr;
                        if (!freeRegs.empty()) { nr = freeRegs.back(); freeRegs.pop_back(); }
                        else { nr = peak; ++peak; }
                        remap[ins.outReg] = nr;
                        ins.outReg = nr;
                    }
                }
                prog.regCount = peak;
            } else {
                // Anomalous stream (use before def / double def) — keep the raw
                // allocation; the cap below then decides activation as before.
                prog.regCount = nextReg;
            }
        }
        prog.active = (prog.drivenSlots != 0) && (prog.regCount <= kMatMaxRegs);
        return prog;
    }

    /**
     * @brief Compile every loaded graph's per-pixel program and attach it to its material.
     *
     * Call once after a project/scene load. Loading restores the FOLDED material — which is
     * correct as far as it goes — but `proceduralProgram` is a compile-time artifact that
     * lives only in RAM, and nothing rebuilt it. So every per-pixel chain (noise, a texture
     * behind a manipulation node, pointiness) stayed dark until the user happened to open the
     * node editor on that material and toggle Live, which re-ran applyGraph. The graph's whole
     * effect hung on a UI panel being visited.
     *
     * Deliberately does NOT fold/apply: the material on disk already IS the last applied fold.
     * This only rebuilds the compiled artifact. Run it BEFORE the geometry rebuild so a graph
     * that reads Pointiness is visible to MeshAttr::anyMaterialUsesPointiness().
     *
     * @returns how many materials ended up with an active program.
     */
    inline size_t compileGraphProgramsForScene(
        const std::unordered_map<std::string, std::shared_ptr<MaterialNodeGraphV2>>& graphs) {
        if (graphs.empty()) return 0;

        // Whole-scene recompile: this is the ONE place allowed to re-intern the Attribute
        // slots from scratch. Every program below is rebuilt against the fresh table, and
        // the per-vertex mesh blocks are rebuilt afterwards (Renderer::rebuildBVH) — so the
        // slot numbering stays consistent end to end. Without the reset, loading project
        // after project in one session would keep piling names into the 4-slot budget until
        // a legitimate attribute could no longer be interned.
        resetMaterialAttributeSlots();

        auto& mgr = MaterialManager::getInstance();
        std::unordered_map<std::string, PrincipledBSDF*> byName;
        for (uint16_t id = 0; id < static_cast<uint16_t>(mgr.getMaterialCount()); ++id) {
            Material* m = mgr.getMaterial(id);
            if (!m || m->type() != MaterialType::PrincipledBSDF) continue;
            byName[mgr.getMaterialName(id)] = static_cast<PrincipledBSDF*>(m);
        }

        size_t active = 0;
        for (const auto& [matName, graphPtr] : graphs) {
            if (!graphPtr) continue;
            auto it = byName.find(matName);
            if (it == byName.end()) continue;   // graph orphaned by a material rename/delete
            PrincipledBSDF* pbsdf = it->second;

            MaterialProgram prog = compileMaterialProgram(*graphPtr, pbsdf);
            if (prog.active) {
                pbsdf->proceduralProgram = std::make_shared<MaterialProgram>(std::move(prog));
                ++active;
            } else {
                pbsdf->proceduralProgram.reset();
            }
        }
        return active;
    }

    // ============================================================================
    // GRAPH SERIALIZATION (same typeId + pin-index format as GeometryNodesV2)
    // ============================================================================

    inline void serializeMaterialGraph(const MaterialNodeGraphV2& graph, nlohmann::json& j) {
        // Bumped when a node type's PIN ORDER changes (links persist pin
        // indices). v2: Material Output moved to grouped/contiguous pin order.
        // v3: Voronoi/Checker merged into Noise Texture â€” the old Checker's
        // outputs were [Color, Fac], the unified node's are [Fac, Color].
        j["pin_version"] = 3;
        nlohmann::json jNodes = nlohmann::json::array();
        for (const auto& n : graph.nodes) {
            nlohmann::json jn;
            jn["type_id"] = n->getTypeId();
            jn["id"] = n->id;
            jn["x"] = n->x;
            jn["y"] = n->y;
            if (const auto* mn = dynamic_cast<const MaterialNodeBase*>(n.get())) {
                nlohmann::json params = nlohmann::json::object();
                mn->serializeParams(params);
                if (!params.empty()) jn["params"] = params;
            }
            jNodes.push_back(std::move(jn));
        }
        j["nodes"] = std::move(jNodes);

        auto pinRef = [&graph](uint32_t pinId, bool output) -> std::pair<uint32_t, int> {
            for (const auto& n : graph.nodes) {
                const auto& pins = output ? n->outputs : n->inputs;
                for (size_t i = 0; i < pins.size(); ++i) {
                    if (pins[i].id == pinId) return { n->id, static_cast<int>(i) };
                }
            }
            return { 0u, -1 };
        };
        nlohmann::json jLinks = nlohmann::json::array();
        for (const auto& l : graph.links) {
            const auto [fromNode, fromIdx] = pinRef(l.startPinId, true);
            const auto [toNode, toIdx] = pinRef(l.endPinId, false);
            if (fromIdx < 0 || toIdx < 0) continue;
            nlohmann::json jl;
            jl["from_node"] = fromNode;
            jl["from_out"] = fromIdx;
            jl["to_node"] = toNode;
            jl["to_in"] = toIdx;
            jLinks.push_back(std::move(jl));
        }
        j["links"] = std::move(jLinks);
    }

    inline std::shared_ptr<MaterialNodeGraphV2> deserializeMaterialGraph(const nlohmann::json& j) {
        auto graph = std::make_shared<MaterialNodeGraphV2>();
        std::unordered_map<uint32_t, NodeSystem::NodeBase*> byOldId;
        // Old ids of nodes SAVED as the standalone Checker type â€” their output
        // links need the v3 swap below. Must key off the serialized type_id:
        // after the merge these load as NoiseTextureNode (getTypeId "MatV2.Noise").
        std::unordered_set<uint32_t> legacyCheckerIds;

        if (j.contains("nodes") && j["nodes"].is_array()) {
            for (const auto& jn : j["nodes"]) {
                const std::string typeId = jn.value("type_id", std::string());
                auto node = NodeSystem::NodeRegistry::instance().create(typeId);
                if (!node) continue;  // unknown node type â€” keep loading the rest
                if (typeId == "MatV2.Checker") legacyCheckerIds.insert(jn.value("id", 0u));
                NodeSystem::NodeBase* raw = graph->registerNode(std::move(node));
                raw->x = jn.value("x", 0.0f);
                raw->y = jn.value("y", 0.0f);
                if (jn.contains("params")) {
                    if (auto* mn = dynamic_cast<MaterialNodeBase*>(raw)) {
                        mn->deserializeParams(jn["params"]);
                    }
                }
                byOldId[jn.value("id", 0u)] = raw;
            }
        }

        const int pinVersion = j.value("pin_version", 1);

        if (j.contains("links") && j["links"].is_array()) {
            for (const auto& jl : j["links"]) {
                auto fromIt = byOldId.find(jl.value("from_node", 0u));
                auto toIt = byOldId.find(jl.value("to_node", 0u));
                if (fromIt == byOldId.end() || toIt == byOldId.end()) continue;
                int oi = jl.value("from_out", -1);
                int ii = jl.value("to_in", -1);
                // v2 and earlier: standalone Checker outputs were [Color, Fac];
                // the unified Noise Texture's are [Fac, Color]. Swap so old
                // links land on the same-named socket.
                if (pinVersion < 3 && oi >= 0 && oi <= 1 &&
                    legacyCheckerIds.count(jl.value("from_node", 0u))) {
                    oi = 1 - oi;
                }
                // v1 saves used the Material Output's original append-order pin
                // indices; v2 regrouped them. Remap so old links land on the
                // same-named socket.
                if (pinVersion < 2 && toIt->second->getTypeId() == "MatV2.Output") {
                    static const int kOutputPinRemapV1[17] = {
                        OutputNode::InSurface, OutputNode::InBaseColor, OutputNode::InMetallic,
                        OutputNode::InRoughness, OutputNode::InSpecular, OutputNode::InEmissionColor,
                        OutputNode::InEmissionStrength, OutputNode::InTransmission, OutputNode::InIOR,
                        OutputNode::InOpacity, OutputNode::InTranslucent, OutputNode::InClearcoat,
                        OutputNode::InClearcoatRoughness, OutputNode::InSubsurface,
                        OutputNode::InSubsurfaceColor, OutputNode::InNormalMap, OutputNode::InNormalStrength
                    };
                    if (ii >= 0 && ii < 17) ii = kOutputPinRemapV1[ii];
                }
                if (oi < 0 || oi >= static_cast<int>(fromIt->second->outputs.size())) continue;
                if (ii < 0 || ii >= static_cast<int>(toIt->second->inputs.size())) continue;
                graph->addLink(fromIt->second->outputs[oi].id, toIt->second->inputs[ii].id);
            }
        }

        return graph->nodes.empty() ? nullptr : graph;
    }

} // namespace MaterialNodesV2
