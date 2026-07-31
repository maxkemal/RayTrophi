/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          MaterialGraphApply.h
* Author:        Kemal Demirtas
* Date:          July 2026
* License:       MIT
* =========================================================================
*
* Applying a material node graph to its material — extracted out of the node
* editor UI so the scripting facade can run the exact same path.
*
* "Apply" is NOT the terrain sense of evaluate (a long async bake with progress
* and cancel). It is a fold + compile + publish:
*
*   1. evaluateMaterialGraph folds the constant part of the graph and
*      applyShadeStateToMaterial writes those values INTO the PrincipledBSDF.
*      Without this the graph is just data sitting NEXT TO the material.
*   2. The Volume branch is published to every VDB/Gas shader bound to it.
*   3. The CPU material is snapshotted into its GpuMaterial struct.
*   4. compileMaterialProgram turns the spatially-varying chains into the
*      per-pixel VM program (CPU Faz 2a + Vulkan RT Faz 2b consume it). This is
*      a COMPILE step — pushing the material to a device cannot substitute.
*   5. Pointiness / Attribute reads force a one-time geometry pass, because
*      they need per-vertex data uploaded with the BLAS.
*   6. Backend material + program re-upload, accumulation reset.
*
* The OptiX per-triangle texture-bundle refresh stays in the editor: it needs
* CUDA types, and on a flat scene it is a no-op anyway (it walks legacy
* Triangle soup). `texture_changed` in the report is what drives it.
*/
#pragma once

#include <string>
#include <vector>

struct UIContext;
class PrincipledBSDF;
struct VolumeMaterialProgram;

namespace MaterialNodesV2 {

class MaterialNodeGraphV2;

struct GraphApplyReport {
    bool ok = false;
    bool texture_changed = false;   ///< drives the caller's OptiX bundle refresh
    std::vector<std::string> warnings;
    std::vector<std::string> errors;
};

// `mark_project_modified` false is the first-open / project-load sync: it pushes
// the freshly materialized graph to the backend without dirtying the project and
// without forcing the geometry pass (the load path rebuilds geometry itself).
GraphApplyReport applyMaterialGraph(UIContext& ctx, MaterialNodeGraphV2& graph,
                                    PrincipledBSDF* material, unsigned short material_id,
                                    bool mark_project_modified = true);

// Publishes a compiled volume program to every VDB/Gas shader bound to this
// material name. Returns true when something actually changed. Graph publication
// belongs to the material lifecycle, not to the VDB/Gas panel, so consumers stay
// live even while their properties panel is closed.
bool publishVolumeProgram(UIContext& ctx, const std::string& material_name,
                          const VolumeMaterialProgram& program);

} // namespace MaterialNodesV2
