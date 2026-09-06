/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          Import/ModelImport.h
* Author:        Kemal Demirtas
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*
* ONE ENTRY POINT FOR "LOAD A MODEL FILE".
*
* ★★★ WHY THIS EXISTS — A MEASUREMENT, NOT A TIDY-UP (2026-09-05)
* ---------------------------------------------------------------------------
* Faz 2 closed the Assimp fallback in Renderer::create_scene, and the batch was
* recorded as "glTF/GLB never reaches Assimp any more". That was FALSE, and the
* reason it read as true is the classic one in this repo: the claim was checked
* at ONE call site.
*
* `loadModelToTriangles` had THREE callers:
*     Renderer.cpp                — had the glTF branch          OK
*     FoliageAssetLibrary.cpp     — did NOT                      BROKEN
*     scene_ui.cpp (anim clips)   — did NOT                      BROKEN
*
* And the ENTIRE vegetation library is `.glb` (assets/vegetation/**: every tree,
* grass and flower). So after Faz 2 the same file produced two different results
* depending on how it entered the scene:
*   - dragged in as a model  -> direct reader: UV V-flipped, photometric lights,
*                               spec-correct skinning
*   - planted as foliage     -> Assimp: raw glTF UVs (there is no
*                               aiProcess_FlipUVs anywhere in AssimpLoader), so
*                               every scattered tree sampled its atlas UPSIDE
*                               DOWN
* Two readers, two conventions, one scene, no error message. Exactly the failure
* CLAUDE.md rule 5 is written about.
*
* ★ So the dispatch lives HERE, once, and callers ask for a FILE — not for a
* reader. Adding ufbx in Faz 3 is then one line in this file instead of a hunt
* through call sites.
*
* Faz 3 increment 1: create_scene now uses loadSceneModel() below. The shared
* result carries lights, cameras, instances and timings; no caller needs its
* own reader dispatch or a live loader object.
* =========================================================================
*/
#pragma once

#include <string>

#include "Import/ImportedModel.h"

namespace rtimport {

// Dispatches on the file extension:
//   .gltf / .glb -> GltfDirectReader  (cgltf)
//   .fbx         -> UfbxReader        (ufbx)
//   .obj         -> ObjReader         (own parser)
// Other extensions fail explicitly. Reader failures never fall back — there is
// nothing left to fall back TO, which is the point: Assimp is gone.
//
// Returns false and fills `error` on failure. `out` is left empty on failure —
// there is no silent partial result, because "loaded, but only half of it" is
// the shape that gets shipped by accident.
bool loadModel(const std::string& filepath,
               const ImportOptions& options,
               ImportedModel& out,
               std::string& error);

// Scene adapter: supplies the prefix/facade contract, logs measured import cost,
// and throws on failure so ProjectManager cannot register a failed import.
ImportedModel loadSceneModel(const std::string& filepath, const std::string& prefix,
                             bool singleFacade);

} // namespace rtimport
