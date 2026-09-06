/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          GltfDirectWriter.h
* Author:        Kemal Demirtas
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*
* Direct glTF 2.0 / GLB writer - replaces the Assimp export round trip.
*
* WHY THIS EXISTS
* ---------------
* RayTrophi's in-memory geometry is ALREADY glTF's on-disk layout, byte for byte:
*
*     Vec3                                 -> accessor VEC3 / FLOAT   (5126), 12 B stride
*     Vec2                                 -> accessor VEC2 / FLOAT   (5126),  8 B stride
*     DNA::GeometryDetail::indices         -> accessor SCALAR / UNSIGNED_INT (5125)
*
* The Assimp path spent minutes and ~15 GB converting those bytes into themselves:
* it exploded a flat SoA mesh into one facade object per triangle, re-deduplicated
* the corners through a hash map, copied the result into aiMesh (one heap
* allocation PER FACE for aiFace::mIndices), and then let Assimp's glTF2 exporter
* build its own third copy before writing. Every one of those stages was serial
* for a single-mesh scene, because the parallelism was split by OBJECT count.
*
* This writer streams the source pointers straight into the file instead.
*
* DESIGN: PLAN, THEN WRITE (single pass, no temp file, no concatenation)
* ---------------------------------------------------------------------
* A GLB must state its JSON chunk before its BIN chunk, but the JSON needs every
* byte offset inside the BIN chunk. Rather than buffer the BIN (RAM) or write it
* twice (I/O), the writer plans the complete binary layout first - every length is
* known from a cheap counting pass - emits the finished JSON, and only then
* streams the payload pieces in the planned order.
*
* Peak RAM is therefore O(JSON + encoded textures + small generated arrays), NOT
* O(scene geometry): a 36M-triangle mesh contributes zero heap allocation on the
* position/normal/uv path (the source arrays are written where they already live).
*/
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

struct SceneData;
struct ExportSettings;
class Hittable;
class Material;

namespace rtgltf {

// Measured, not estimated: every field is filled from what was actually written.
// Reported over IPC (scene.export_gltf) so an agent can regress export cost.
struct WriteStats {
    uint64_t mesh_count = 0;
    uint64_t primitive_count = 0;      // one per (mesh, material) pair
    uint64_t vertex_count = 0;
    uint64_t triangle_count = 0;
    uint64_t node_count = 0;
    uint64_t instanced_group_count = 0; // EXT_mesh_gpu_instancing nodes emitted
    uint64_t instance_count = 0;        // transforms folded into those nodes
    uint64_t material_count = 0;
    uint64_t image_count = 0;
    uint64_t bin_bytes = 0;
    uint64_t json_bytes = 0;
    uint64_t file_bytes = 0;
    // Heap the writer itself asked for at its high-water mark, in MB. This is the
    // number the Assimp path blew up on; it is deliberately OUR allocation, not
    // process RSS, so it stays comparable across runs and machines.
    double   peak_writer_mb = 0.0;
    double   seconds_total = 0.0;
    double   seconds_collect = 0.0;
    double   seconds_materials = 0.0;
    double   seconds_plan = 0.0;
    double   seconds_write = 0.0;
};

// Writes `filepath` (.glb -> single binary file; .gltf -> JSON + sidecar .bin).
// `materialOverrides` lets the caller substitute baked materials (terrain) by id.
// Returns false and fills `error` on failure; never throws out of here.
//
// ★ LIFETIME: because flat SoA attribute arrays are streamed straight from where
// they live, `scene`'s geometry must not be mutated, resized or re-evaluated for
// the duration of this call. Callers run it either on the main thread (the IPC
// path) or with rendering stopped (the UI path); do not call it from a worker
// while the frame loop is still touching the same meshes.
bool writeScene(const std::string& filepath,
                SceneData& scene,
                const ExportSettings& settings,
                const std::vector<std::shared_ptr<Hittable>>& selected_objects,
                const std::map<uint16_t, std::shared_ptr<Material>>& materialOverrides,
                WriteStats& stats,
                std::string& error);

} // namespace rtgltf
