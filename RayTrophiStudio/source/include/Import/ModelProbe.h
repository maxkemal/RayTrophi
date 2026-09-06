/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          Import/ModelProbe.h
* Author:        Kemal Demirtas
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*
* COUNTS AND A BOUNDING BOX, WITHOUT BUILDING A SCENE.
*
* ★ This was `GltfProbe` in GltfDirectReader.h. Nothing in it was ever
* glTF-specific — it is counts and a box — and it only lived there because glTF
* was the first format that could be probed without Assimp. Now that FBX and
* OBJ have direct readers too, keeping the glTF name would be a lie, so per
* CLAUDE.md rule 5 the type is RENAMED as its meaning widens rather than
* quietly gaining non-glTF callers.
*
* ★★ WHY A PROBE EXISTS AT ALL. The asset browser wants a triangle count and a
* size for every file in a library. Doing that with a full import meant Assimp
* decoding every buffer and running aiProcess_ImproveCacheLocality — a Tipsify
* reorder — to produce A NUMBER AND A BOX. On a vegetation library that is the
* difference between scanning and waiting.
*
* ★★★ A PROBE THAT FAILS MUST SAY SO, NOT FALL BACK. If the probe cannot read
* a file, the IMPORTER cannot read it either. Showing plausible numbers from
* some other code path for a file that will fail to open is exactly the
* "panel lies" failure this repo keeps paying for.
* =========================================================================
*/
#pragma once

#include <cstdint>
#include <string>

namespace rtimport {

struct ModelProbe {
    uint64_t mesh_count = 0;
    uint64_t material_count = 0;
    uint64_t animation_count = 0;
    uint64_t node_count = 0;
    uint64_t skin_count = 0;
    uint64_t triangle_count = 0;
    uint64_t vertex_count = 0;
    uint64_t texture_reference_count = 0;

    bool  has_bounds = false;
    float bounds_min[3] = { 0.0f, 0.0f, 0.0f };
    float bounds_max[3] = { 0.0f, 0.0f, 0.0f };
};

// `applyNodeTransforms` picks the bounding box's SPACE, and the two callers
// genuinely want different ones, so it is a parameter rather than a default:
//   false — union of raw mesh-local bounds over EVERY mesh, including ones no
//           node references (what AssetRegistry has always reported)
//   true  — scene-graph placed bounds, for the preview camera framing
//
// Dispatches on the extension exactly like loadModel(). Returns false if the
// format is unsupported or the file cannot be read; `out` is then left empty.
bool probeModel(const std::string& filepath, bool applyNodeTransforms, ModelProbe& out);

// Per-format implementations, next to the reader that shares their parser.
bool probeGltf(const std::string& filepath, bool applyNodeTransforms, ModelProbe& out);
bool probeFbx(const std::string& filepath, bool applyNodeTransforms, ModelProbe& out);
// OBJ has no node transforms, so the flag is accepted and ignored — the two
// spaces are identical for this format.
bool probeObj(const std::string& filepath, bool applyNodeTransforms, ModelProbe& out);

} // namespace rtimport
