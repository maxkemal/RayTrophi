/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          Import/ObjReader.h
* Author:        Kemal Demirtas
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*
* WAVEFRONT OBJ / MTL, READ DIRECTLY. Faz 3 increment 3.
*
* ★ NO THIRD-PARTY PARSER, AND THAT IS THE CHEAP OPTION HERE. glTF got cgltf
* and FBX got ufbx because those are binary container formats with extensions,
* compression and deformer graphs — formats where a hand-written parser is a
* liability. OBJ is a line-oriented text format with about a dozen keywords.
* Vendoring tinyobjloader would add a dependency, its own conventions and its
* own conversion layer, to replace roughly 300 lines of parsing.
*
* WHAT THIS READER DELIBERATELY GETS RIGHT (each one is a known trap)
* ---------------------------------------------------------------------------
*   1. Numbers are parsed with std::from_chars, NOT strtof/atof/sscanf.
*      Main.cpp calls setlocale(LC_ALL, "Turkish") at startup, so the C locale's
*      decimal separator is a COMMA. `strtof("1.5")` would return 1.0 and stop
*      at the dot — every fractional coordinate in the file silently truncated,
*      producing a model that still loads and still looks like a model.
*      from_chars is locale-independent by specification. The repo has no
*      atof/strtod anywhere today; this reader does not become the first.
*
*   2. The V coordinate is NOT flipped. OBJ places the texture origin at the
*      BOTTOM-LEFT, same as FBX and same as this engine; glTF is the odd one
*      out (V-down) and that is why GltfDirectReader flips. Flipping here
*      "for consistency" would mirror every OBJ texture vertically.
*
*   3. Negative face indices are relative to the CURRENT vertex count, per the
*      OBJ spec. Treating them as absolute yields scrambled geometry on exports
*      from Blender/Houdini that use the relative form.
*
*   4. All material sub-meshes of one `o`/`g` object share ONE nodeName.
*      nodeName is a GROUPING KEY, not an identity (see the multi-material
*      lesson in docs/dev/FAZ3_DEVIR_NOTU.md §3.1): giving each material its
*      own node splits one object into N in the outliner and breaks scatter
*      and selection.
*
*   5. Missing normals are GENERATED, honouring `s` smoothing groups. A file
*      with no `vn` is common; without generation every surface renders with
*      whatever the attribute was zero-initialised to.
*
*   6. Textures are decoded on a bounded worker pool before materials are
*      built. Assimp's importer did this and the first direct reader dropped
*      it, which reported as nothing at all except a slower import.
* =========================================================================
*/
#pragma once

#include <string>

#include "Import/ImportedModel.h"
#include "Import/ModelProbe.h"

namespace rtimport {

// Returns false and fills `error` on failure; `out` is left empty. There is no
// fallback to another reader — "the new reader worked" and "the new reader
// failed and something covered for it" must not look alike.
bool readObj(const std::string& path,
             const ImportOptions& options,
             ImportedModel& out,
             std::string& error);

// Counts and a bounding box only — a text scan, no geometry built and no
// material registered. `applyNodeTransforms` is accepted and ignored: OBJ has
// no node transforms, so both spaces are the same one.
bool probeObj(const std::string& path, bool applyNodeTransforms, ModelProbe& out);

} // namespace rtimport
