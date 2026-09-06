/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          Import/ImportedModel.h
* Author:        Kemal Demirtas
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*
* WHAT AN IMPORTER PRODUCES — one shape, every format.
*
* WHY THIS EXISTS
* ---------------
* Three readers are planned (cgltf for glTF/GLB, ufbx for FBX, a small one for
* OBJ) and Assimp is on its way out. Without a shared result type each reader
* grows its own wiring into Renderer::create_scene — three integration surfaces
* for one job, and the repo's most expensive recurring failure is exactly that:
* two code paths kept alive side by side until they quietly disagree.
*
* ★ DELIBERATELY A STRUCT, NOT A VIRTUAL INTERFACE.
*   - Format selection is one dispatch on the file extension, not a hot
*     polymorphic call. A vtable would buy nothing.
*   - A plain value is what the acceptance test compares: the parity probe
*     diffs two FILLS OF THIS STRUCT for the same file. With an interface you
*     would still need the struct, so the interface is the part that is optional.
*   - No lifetime coupling. ImportedModelContext used to hold the AssimpLoader
*     alive because the animation runtime walked its live aiScene every frame.
*     Breaking that was the whole point of Faz 0.5; handing callers an importer
*     OBJECT to keep would invite exactly that mistake back.
*
* ★★ THE FIELDS ARE DERIVED FROM WHAT create_scene CONSUMES, NOT FROM WHAT
* AssimpLoader HAPPENS TO EXPOSE. Designing a "neutral" type around the one
* implementation you can see is how Assimp became a data type in this codebase
* in the first place (see docs/dev/ASSIMP_IMPORT_REPLACEMENT_BRIEF.md). Every
* member below answers a real call site in Renderer::create_scene.
* =========================================================================
*/
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "Animation/NodeHierarchy.h"

class Triangle;
class TriangleMesh;
class Light;
class Camera;
struct AnimationData;
struct BoneData;

namespace rtimport {

// Measured, not estimated — filled from what was actually read. Mirrors
// rtgltf::WriteStats on the export side so import cost finally becomes a number
// (today scene.import_model returns nothing at all).
struct ImportStats {
    uint64_t mesh_count = 0;
    uint64_t vertex_count = 0;
    uint64_t triangle_count = 0;
    uint64_t material_count = 0;
    uint64_t image_count = 0;
    uint64_t node_count = 0;
    uint64_t animation_count = 0;
    uint64_t animation_channel_count = 0;
    uint64_t bone_count = 0;
    uint64_t skinned_mesh_count = 0;
    uint64_t camera_count = 0;
    uint64_t light_count = 0;
    uint64_t instance_group_count = 0;
    uint64_t instance_count = 0;
    double   seconds_total = 0.0;
    double   seconds_parse = 0.0;      // container/JSON parse only
    double   seconds_geometry = 0.0;
    double   seconds_materials = 0.0;  // includes texture decode — usually the big one
    double   seconds_animation = 0.0;
    // Which reader actually ran. ★ A fallback to Assimp must never be silent:
    // "the new reader works" and "the new reader failed and Assimp covered for
    // it" look identical from the outside otherwise.
    std::string reader;                // "cgltf" | "assimp" | ...
    std::string fallback_reason;       // empty unless a fallback happened
};

// One EXT_mesh_gpu_instancing node: a prototype mesh plus its placements.
//
// ★ THIS CLOSES A ONE-WAY DOOR. GltfDirectWriter already EMITS this extension
// for every scatter group, so before it was read back, a scatter scene exported
// from RayTrophi and reopened came back with the prototype and ZERO placements
// — a valid file, no error, just an empty forest. "We support the format" and
// "we can reopen our own output" are different claims.
//
// ★ Placements are MATRICES, deliberately. InstanceGroup stores a decomposed
// triple (position, EULER DEGREES, scale) in a Y*X*Z basis with row scaling — a
// convention that belongs to InstanceTransform, not to a file format. The
// importer therefore hands over the matrix and InstanceTransform::fromMatrix()
// (which lives next to its own inverse, toMatrix) does the conversion. The
// writer round-trips at the matrix level too, so this is the same currency on
// both sides.
struct ImportedInstancePlacement {
    Matrix4x4 transform = Matrix4x4::identity();
};

struct ImportedInstanceGroup {
    std::string name;              // the instancing node's name
    std::string sourceNodeName;    // prototype node name, for InstanceGroup::source_node_name

    // ★★★ FLAT MESHES, NOT FACADES, AND THEY STAY IN `objects`.
    //
    // ScatterSource has two shapes and they are not interchangeable:
    //   - flat_meshes            → Vulkan emits ONE BLAS entry PER MESH, each with
    //                              its own material, and reuses the BLAS the base
    //                              world-object pass already built
    //                              (m_meshRegistry["[DirectMesh]-..."]).
    //   - centered_triangles_ptr → one self-contained BLAS built from facades,
    //                              and its material is taken from triangle 0 ONLY.
    //
    // A scattered tree is multi-material (bark, needles, cards). The facade shape
    // would render the whole tree in whichever material happened to be first —
    // plausible-looking and wrong — so the flat shape is the correct one, and it
    // costs no facade soup.
    //
    // ★ The consequence is that the prototype must BE a world object, because
    // that is where its BLAS comes from. That also matches how RayTrophi scatter
    // is authored: the source is a real mesh in the scene, which is why the
    // writer bakes `translation(-mesh_center) * sourceWorld` into the placements.
    std::vector<std::shared_ptr<TriangleMesh>> sourceMeshes;

    std::vector<ImportedInstancePlacement> placements;
};

// ---------------------------------------------------------------------------
// WHAT A CALLER ASKS FOR — one shape, every format.
//
// ★ This was `GltfReadOptions` and lived in GltfDirectReader.h. Nothing in it
// was ever glTF-specific; it only lived there because glTF was the first reader
// written against ImportedModel. Now that a single entry point dispatches over
// formats (Import/ModelImport.h) that name would be a lie — and per CLAUDE.md
// rule 5 a field whose MEANING widens gets renamed, so no call site can keep
// reading it as something narrower.
// ---------------------------------------------------------------------------
struct ImportOptions {
    // Prefix applied to every node/material/animation name, so two imports of
    // the same file cannot collide. Mirrors AssimpLoader::currentImportName.
    std::string importPrefix;

    bool loadGeometry   = true;
    bool loadMaterials  = true;
    bool loadAnimations = true;
    bool loadSkinning   = true;
    bool loadCameras    = true;
    bool loadLights     = true;

    // ★★ ONE representative facade per non-skinned mesh, instead of one per face.
    //
    // This is NOT a performance dial — it changes what the result MEANS, and the
    // two consumers genuinely differ:
    //   true  — create_scene: downstream bookkeeping is O(meshes) and reads
    //           `parentMesh` for the real geometry.
    //   false — ScatterSource's facade path: InstanceManager walks
    //           `source.triangles` ONE TRIANGLE AT A TIME to bake centred
    //           copies, so a representative facade would collapse a whole tree
    //           into a single triangle. Plausible-looking and wrong.
    bool emitSingleFacadePerMesh = true;
};

struct ImportedModel {
    // Scene objects. ★ These are REPRESENTATIVE facades, one per mesh, each
    // carrying `parentMesh` = the canonical flat SoA TriangleMesh. This is the
    // shape create_scene's downstream bookkeeping already consumes (bone-index
    // dedupe, pivot recentre, import-flat collapse, nodeName grouping), and it
    // is O(meshes), NOT the O(faces) facade soup.
    //
    // ★★ Skinned meshes are the exception and still carry one facade per face:
    // Renderer.cpp's import-flat collapse excludes them ("SoA skinning is a
    // later increment"). A reader must match that, not quietly change it —
    // changing which meshes go flat is a separate, testable increment.
    std::vector<std::shared_ptr<Triangle>> objects;

    std::vector<std::shared_ptr<AnimationData>> animations;
    std::shared_ptr<BoneData> bones;   // shared_ptr so this header need not see BoneData

    // The node tree the animation runtime walks every frame. Copied into
    // ImportedModelContext; nothing downstream may need the importer to survive.
    RayTrophi::NodeHierarchy hierarchy;

    std::vector<ImportedInstanceGroup> instanceGroups;

    std::vector<std::shared_ptr<Light>>  lights;
    std::vector<std::shared_ptr<Camera>> cameras;
    std::shared_ptr<Camera> fallbackCamera; // legacy reader's default, only used for a new scene

    std::string importName;   // prefix applied to every node/material name
    ImportStats stats;
};

} // namespace rtimport
