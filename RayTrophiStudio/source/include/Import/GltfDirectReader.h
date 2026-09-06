/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          Import/GltfDirectReader.h
* Author:        Kemal Demirtas
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*
* Direct glTF 2.0 / GLB reader — the mirror of GltfDirectWriter.
*
* WHY THIS EXISTS
* ---------------
* The writer already proved the shape: RayTrophi's in-memory geometry IS glTF's
* on-disk layout, so the conversion both ways should be a copy, not a rebuild.
* The Assimp import path instead materialises the scene twice — once as an
* aiScene (its own full object graph, one heap allocation per face for
* aiFace::mIndices) and once as RayTrophi geometry — before anything is usable.
*
* This reader parses the JSON into DESCRIPTORS (cgltf) and streams the buffer
* bytes straight into flat SoA. That is why cgltf and not tiny_gltf: tiny_gltf
* decodes every buffer into std::vectors and builds its own mesh graph, which is
* precisely the third copy the export side spent this whole effort escaping.
*
* WHAT IT PRODUCES
* ----------------
* An rtimport::ImportedModel — the same struct any other importer fills, so the
* acceptance test is a diff of two fills for the same file (see
* scripts/probe_import_export_parity.py).
*
* ★ CONVENTIONS THAT MUST NOT DRIFT FROM THE ASSIMP PATH
*   - glTF is Y-up, right-handed, metres, and Assimp applies no conversion for
*     it either. So this reader applies NONE. (FBX is the format that needs
*     aiProcess_GlobalScale; that is Faz 3's problem, not this file's.)
*   - Node matrices are COLUMN-major in glTF; Matrix4x4 here is row-major
*     m[row][col]. Getting this backwards produces a transposed pose that still
*     looks like a pose.
*   - Skinned meshes keep the per-face facade path, exactly as the Assimp path
*     does, because Renderer's import-flat collapse excludes them. Changing
*     which meshes go flat is a separate, testable increment.
* =========================================================================
*/
#pragma once

#include <string>

#include "Import/ImportedModel.h"
#include "Import/ModelProbe.h"

namespace rtimport {

// True for paths this reader claims (.gltf / .glb, case-insensitive).
bool isGltfPath(const std::string& filepath);

// ---------------------------------------------------------------------------
// METADATA WITHOUT LOADING.
//
// The asset browser and the preview-bounds helper each ran their OWN
// Assimp::Importer to answer "how many meshes, how big is it" — a full parse,
// buffer decode and (in AssetRegistry) an aiProcess_ImproveCacheLocality
// Tipsify pass, just to read counts and a box. Two more Assimp dependencies
// that had nothing to do with importing.
//
// ★ glTF makes this nearly free: the spec REQUIRES min/max on every POSITION
// accessor, so the bounding box is in the JSON. This probe therefore parses the
// container only — no cgltf_load_buffers, no image decode — and touches buffer
// bytes solely for the malformed case where a POSITION accessor omits its
// bounds.
// ---------------------------------------------------------------------------
// ★ The probe RESULT type moved to Import/ModelProbe.h and is now
// `rtimport::ModelProbe`: it was never glTF-specific, and FBX/OBJ probe into
// the same shape. probeGltf() is declared there alongside its siblings.

// Returns false and fills `error` on failure; never throws out of here.
// On failure `out` is left in a well-defined empty state so the caller can fall
// back — LOUDLY; see ImportStats::fallback_reason.
bool readGltf(const std::string& filepath,
              const ImportOptions& options,
              ImportedModel& out,
              std::string& error);

} // namespace rtimport
