/*
 * MeshEdit/FlatMeshPublisher.h
 *
 * Publishes validated half-edge topology into the canonical TriangleMesh/DNA
 * SoA representation.  This is the only bridge needed by the future edit
 * service; UI facades are not part of the publish path.
 */
#pragma once

#include "MeshEdit/HalfEdgeMesh.h"
#include "MeshEdit/MeshTool.h"

class TriangleMesh;

namespace DNA { class GeometryDetail; }

namespace MeshEdit {

// Replaces mesh.geometry only after the source topology validates. Existing
// vertex attributes are copied by stable vertex id where possible; newly
// created topology receives deterministic defaults and a diagnostic warning.
MeshOperationReport publishHalfEdgeMeshToFlat(TriangleMesh& mesh,
                                              const HalfEdgeMesh& topology,
                                              const DNA::GeometryDetail* source = nullptr);

bool runFlatMeshPublisherSelfTest(std::string& report);

} // namespace MeshEdit
