/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          MaterialGraphApply.cpp
* Author:        Kemal Demirtas
* Date:          July 2026
* License:       MIT
* =========================================================================
*
* See MaterialGraphApply.h for what "apply" means and why it is more than a
* backend material push. Extracted verbatim out of the node editor's
* applyGraph() so the editor and the scripting facade share one implementation.
*/

#include "MaterialGraphApply.h"

#include <memory>
#include <string>

#include "scene_ui.h"            // UIContext
#include "MaterialNodesV2.h"
#include "PrincipledBSDF.h"
#include "PBRMaterialSnapshot.h"
#include "VolumeShader.h"
#include "VolumeMaterialProgram.h"
#include "ProjectManager.h"

namespace MaterialNodesV2 {

bool publishVolumeProgram(UIContext& ctx, const std::string& material_name,
                          const VolumeMaterialProgram& program) {
    bool changed = false;
    bool surfaceChanged = false;
    const auto programJson = program.toJson();
    const auto publish = [&](const std::shared_ptr<VolumeShader>& shader) {
        if (!shader || shader->material_graph != material_name) return;
        if (shader->material_program.toJson() == programJson) return;
        shader->material_program = program;
        changed = true;
    };

    for (const auto& vdb : ctx.scene.vdb_volumes) {
        if (!vdb) continue;
        const auto shader = vdb->getShader();
        publish(shader);
        if (vdb->render_as_isosurface && shader &&
            shader->material_graph == material_name) {
            // The same asset's Surface output owns this SDF boundary.
            surfaceChanged = true;
        }
    }
    for (const auto& gas : ctx.scene.gas_volumes) {
        if (gas) publish(gas->getShader());
    }

    if (changed || surfaceChanged) {
        // Graph publication belongs to the material lifecycle, not to the
        // VDB/Gas panel. Consumers therefore stay live even while their
        // properties panel is closed.
        ctx.renderer.updateBackendGasVolumes(ctx.scene);
    }
    return changed || surfaceChanged;
}

GraphApplyReport applyMaterialGraph(UIContext& ctx, MaterialNodeGraphV2& graph,
                                    PrincipledBSDF* material, unsigned short material_id,
                                    bool mark_project_modified) {
    GraphApplyReport report;

    const MaterialGraphResult res = evaluateMaterialGraph(graph, material);
    report.warnings = res.warnings;
    report.errors = res.errors;
    report.ok = res.ok;
    if (!res.ok || !material) return report;

    report.texture_changed = applyShadeStateToMaterial(res.state, *material);

    const VolumeGraphResult volumeResult = evaluateVolumeMaterialGraph(graph);
    if (volumeResult.ok) {
        publishVolumeProgram(ctx, material->materialName, volumeResult.program);
        if (volumeResult.program.density_noise_enabled && ctx.render_settings.use_optix) {
            report.warnings.push_back(
                "Density field noise currently executes on Vulkan RT; OptiX parity is scheduled after the field contract stabilizes");
        }
    } else {
        report.warnings.push_back(
            "Volume output could not be published; open diagnostics for the volume branch");
        report.errors.insert(report.errors.end(), volumeResult.errors.begin(), volumeResult.errors.end());
        report.ok = false;
    }

    // CPU -> GPU struct (the canonical sync, see MaterialManager::syncAllGpuMaterials)
    if (!material->gpuMaterial) material->gpuMaterial = std::make_shared<GpuMaterial>();
    const PBRMaterialSnapshot snapshot = capturePBRMaterialSnapshot(*material);
    applyPBRMaterialSnapshotToGpuMaterial(snapshot, *material->gpuMaterial);

    // Compile the spatially-varying chains into a per-pixel program. Both the
    // CPU render (Faz 2a) and Vulkan RT (Faz 2b, closesthit VM) consume it, so
    // these slots shade per-pixel on both. active=false (all-constant / direct
    // texture bind) clears it -> zero per-pixel cost, unchanged behaviour.
    const bool hadPointiness = material->proceduralProgram && material->proceduralProgram->usesPointiness;
    const bool hadAttributes = material->proceduralProgram && material->proceduralProgram->usesAttributes;
    const size_t attrSlotsBefore = MaterialNodesV2::materialAttributeSlots().size();
    {
        MaterialProgram prog = compileMaterialProgram(graph, material);
        int drivenCount = 0;
        if (prog.active) {
            for (uint32_t s = 0; s < static_cast<uint32_t>(MatSlot::Count); ++s)
                if (prog.drivenSlots & (1u << s)) ++drivenCount;
        }
        if (prog.active) material->proceduralProgram = std::make_shared<MaterialProgram>(std::move(prog));
        else material->proceduralProgram.reset();
        if (drivenCount > 0) {
            report.warnings.push_back(std::to_string(drivenCount) +
                " slot(s) shade PER-PIXEL on CPU + Vulkan RT; the frozen OptiX backend uses the folded average");
        }
    }

    // Pointiness is the one Geometry output that isn't free at the shading point: it
    // needs a per-vertex precompute (CPU caches, built in rebuildBVH) and a per-vertex
    // GPU block (uploaded with the BLAS). Both are lazy, so the FIRST graph to read it
    // has to force one geometry pass. Strictly on the transition (or when the geometry
    // on the GPU predates it) — otherwise every slider tweak would rebuild the scene.
    // The project-load sync (mark_project_modified == false) rebuilds geometry itself.
    const bool usesPointiness = material->proceduralProgram && material->proceduralProgram->usesPointiness;
    // The Attribute node needs the exact same one-time geometry pass, plus one extra
    // trigger the pointiness gate has no equivalent of: picking a DIFFERENT attribute
    // name interns a NEW slot, and the per-vertex blocks already on the GPU carry only
    // the old slots. Without the slot-count check the newly chosen channel would read 0
    // everywhere until something else happened to rebuild the geometry.
    const bool usesAttributes = material->proceduralProgram && material->proceduralProgram->usesAttributes;
    const bool attrSlotsGrew  = MaterialNodesV2::materialAttributeSlots().size() != attrSlotsBefore;
    const bool needAttribPass = usesAttributes &&
        (!hadAttributes || attrSlotsGrew ||
         (ctx.backend_ptr && !ctx.backend_ptr->geometryHasAttributes()));
    const bool needPointinessPass = usesPointiness &&
        (!hadPointiness || (ctx.backend_ptr && !ctx.backend_ptr->geometryHasPointiness()));

    if (mark_project_modified && (needPointinessPass || needAttribPass)) {
        ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
        if (ctx.backend_ptr) ctx.renderer.rebuildBackendGeometry(ctx.scene);
    }

    if (usesAttributes) {
        // An Attribute node whose name is not interned compiled to a constant 0 — say so,
        // or the user just sees a black material and has no idea the budget ran out.
        for (const auto& n : graph.nodes) {
            auto* an = dynamic_cast<MaterialNodesV2::AttributeNode*>(n.get());
            if (!an) continue;
            if (an->attributeName.empty()) {
                report.warnings.push_back("Attribute node has no channel selected - reads 0");
            } else if (MaterialNodesV2::findMaterialAttributeSlot(an->attributeName) < 0) {
                report.warnings.push_back("Attribute '" + an->attributeName + "': all " +
                    std::to_string(MaterialNodesV2::kMatAttribSlots) +
                    " attribute slots are in use - reads 0");
            }
        }
    }

    ctx.renderer.resetCPUAccumulation();
    if (ctx.backend_ptr) {
        ctx.renderer.updateBackendMaterial(ctx.scene, material_id);
        // Faz 2b: re-upload the per-pixel program buffer so Vulkan RT reflects
        // Noise/ColorRamp edits live (updateBackendMaterial only refreshes the
        // folded VkGpuMaterial, not the program stream). No-op off Vulkan.
        ctx.renderer.syncMaterialProgramsToBackend(ctx.backend_ptr);
        ctx.backend_ptr->resetAccumulation();
    }
    // First-open sync (mark_project_modified == false) must NOT dirty the project:
    // it only pushes the freshly-materialized graph's folded material + per-pixel
    // program to the backend, which is semantically identical to what's on disk.
    if (mark_project_modified) ProjectManager::getInstance().markModified();

    // Clear every node's dirty flag so next frame's edit detection (a false->true
    // transition) fires reliably. evaluate only clears nodes it actually FOLDS;
    // nodes on binding-only chains (e.g. a Bump on the Normal Map slot) are never
    // evaluated, so markAllDirty leaves them stuck dirty — which would swallow all
    // their later param edits (strength/distance wouldn't re-apply live).
    //
    // This is ALSO what arms live apply for a graph loaded from disk: its nodes are born
    // dirty (NodeBase::dirty = true), and until they are cleared once no edit can produce
    // the false->true transition live apply looks for. See MaterialNodeGraphV2::
    // needsInitialApply — this function is the only place that clears it.
    for (auto& n : graph.nodes) n->dirty = false;
    graph.needsInitialApply = false;

    return report;
}

} // namespace MaterialNodesV2
