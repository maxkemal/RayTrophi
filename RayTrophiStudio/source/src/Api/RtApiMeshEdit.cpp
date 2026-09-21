/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          Api/RtApiMeshEdit.cpp
 * Author:        Kemal Demirtas
 * License:       MIT
 * =========================================================================
 * Polygon mesh editing over the core API.
 *
 * ★★★ The operators here are NOT new. extrudeSelectedMeshFaces,
 * insetSelectedMeshFaces, bevelSelectedEdges, loopCutSelectedEdges,
 * dissolveSelectedEdges, dissolveSelectedVertices, mergeSelectedVerticesToCenter
 * and weldSelectedVerticesByDistance have lived on SceneUI for a long time and
 * mesh.tools.list has advertised all of them as "scriptable": true,
 * "ipc_exposed": true. Nothing could call them: the only entry point was a
 * button in the edit-mode overlay. This file is the missing reach, not a
 * missing implementation.
 *
 * ★★ Why this is a rule violation and not merely a gap: the tool catalogue
 * measured its own INTENT. An agent asking "what can this application do"
 * received eight scriptable operators and could drive exactly zero of them,
 * with no error to hint at it -- the operators simply had no address. That is
 * the same failure class as a counter that reports fullness.
 *
 * ★ The editable mesh cache (topology + selection) lives on SceneUI. That is
 * core state sitting in the UI, and the rule says UI holds no state of its
 * own. Moving the cache is a larger job than opening the operators; what this
 * file guarantees meanwhile is that the state is READABLE and WRITABLE from
 * outside, so core and panel can no longer diverge unobserved.
 */

#include "Api/RtApi.h"
#include "Api/RtApiInternal.h"
#include "MeshEdit/FlatMeshValidator.h"
#include "MeshEdit/MeshOperation.h"
#include "MeshEdit/MeshTool.h"
#include "TriangleMesh.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace rtapi {


Result validateMesh(const std::string& name, MeshValidationInfo& out) {
    if (!g_ctx) return notBound();

    TriangleMesh* mesh = nullptr;
    for (const auto& object : g_ctx->scene.world.objects) {
        auto candidate = std::dynamic_pointer_cast<TriangleMesh>(object);
        if (candidate && candidate->nodeName == name) {
            mesh = candidate.get();
            break;
        }
    }
    if (!mesh) return Result::fail("object not found: " + name);

    const MeshEdit::FlatMeshValidation report = MeshEdit::validateFlatMesh(*mesh);
    out.valid = report.valid;
    out.vertex_count = static_cast<size_t>(report.vertex_count);
    out.triangle_count = static_cast<size_t>(report.triangle_count);
    out.non_finite_vertices = static_cast<size_t>(report.non_finite_vertices);
    out.out_of_range_indices = static_cast<size_t>(report.out_of_range_indices);
    out.degenerate_triangles = static_cast<size_t>(report.degenerate_triangles);
    out.non_finite_normals = static_cast<size_t>(report.non_finite_normals);
    return Result::success();
}

Result planMeshOperation(const std::string& object_name,
                         const std::string& operation_id,
                         const std::string& backend,
                         bool preview,
                         bool commit,
                         MeshOperationPlanInfo& out) {
    if (!g_ctx) return notBound();
    TriangleMesh* mesh = nullptr;
    for (const auto& object : g_ctx->scene.world.objects) {
        auto candidate = std::dynamic_pointer_cast<TriangleMesh>(object);
        if (candidate && candidate->nodeName == object_name) { mesh = candidate.get(); break; }
    }
    if (!mesh || !mesh->geometry) return Result::fail("object not found or has no flat geometry: " + object_name);

    MeshEdit::MeshOperationBackend requested = MeshEdit::MeshOperationBackend::Auto;
    if (backend == "cpu") requested = MeshEdit::MeshOperationBackend::CPU;
    else if (backend == "gpu") requested = MeshEdit::MeshOperationBackend::GPU;
    else if (!backend.empty() && backend != "auto") return Result::fail("unknown mesh operation backend: " + backend);

    const auto* tool = MeshEdit::MeshToolRegistry::instance().find(operation_id);
    MeshEdit::MeshOperationRequest request;
    request.operation_id = operation_id;
    request.object_name = object_name;
    request.selection_domain = tool ? tool->selection_domain : MeshEdit::MeshSelectionDomain::Object;
    request.backend = requested;
    request.preview = preview;
    request.commit = commit;
    const auto plan = MeshEdit::planMeshOperation(request, tool,
        static_cast<uint64_t>(mesh->geometry->get_vertex_count()),
        0, static_cast<uint64_t>(mesh->geometry->indices.size() / 3));

    out = {};
    out.ok = plan.ok;
    out.operation_id = plan.operation_id;
    out.object_name = plan.object_name;
    out.backend = plan.backend;
    out.preview = plan.preview;
    out.commit = plan.commit;
    out.undoable = plan.undoable;
    out.requires_cpu_fallback = plan.requires_cpu_fallback;
    out.expected_revision = plan.expected_revision;
    for (const auto& diagnostic : plan.diagnostics) {
        out.diagnostic_codes.push_back(diagnostic.code);
        out.diagnostic_messages.push_back(diagnostic.message);
        out.diagnostic_warnings.push_back(diagnostic.warning);
    }
    return Result::success();
}


namespace {

// ★★★★ The selection CANNOT live in the editable cache between IPC calls.
//
// Measured on a live build: mesh.edit.select reported 12 edges selected and
// the very next call read 0, with the edit object name back to "". The UI
// clears active_mesh_edit_object_name and lets rebuildMeshCache wipe the
// editable cache whenever it is not in edit mode -- which it never is while a
// script drives the application. Producer and consumer sit in different
// loops, the recurring failure in this repo.
//
// So rtapi OWNS the selection and re-applies it to the cache immediately
// before each operator runs. The cache stays the UI's; what an automated
// caller selected is no longer at the mercy of a frame it cannot see.
//
// ★ A committed operator CLEARS this, because it rewrites topology and every
// stored id then refers to elements that may no longer exist. Silently
// reusing them would edit the wrong faces -- a plausible-looking result, the
// worst kind.
struct MeshEditSession {
    std::string object;
    std::vector<int> vertex_ids;
    std::vector<int> edge_ids;
    std::vector<int> face_ids;

    void clear() { object.clear(); vertex_ids.clear(); edge_ids.clear(); face_ids.clear(); }
};

MeshEditSession g_mesh_edit;

// Resolves the object an operator should act on and leaves it active.
//
// ★ The active object is NOT restored afterwards, and that is deliberate: the
// selection lives inside that object's editable cache, so restoring a previous
// active object would silently strand every id the caller just selected.
// Selection and edit target travel together or neither is usable.
Result activateEditObject(const std::string& object, std::string& out_name) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");

    std::string name = object;
    if (name.empty()) name = ui.active_mesh_edit_object_name;
    if (name.empty()) {
        return Result::fail(
            "no mesh edit object: pass 'object', or call mesh.edit.begin first");
    }
    if (!objectExists(name)) return Result::fail("object not found: " + name);

    // ★★★ The active object MUST be set BEFORE the cache is built, not after.
    // ensureEditableMeshCache calls rebuildMeshCache when the mesh cache has
    // been invalidated, and rebuildMeshCache preserves the editable cache for
    // exactly ONE object: the active edit object. Setting the name afterwards
    // meant every call arrived with the previous call's selection already
    // wiped -- select reported 12 edges selected, and the operator that
    // followed it read zero. The symptom is indistinguishable from a broken
    // operator, because the selection is gone by the time anyone can look.
    ui.active_mesh_edit_object_name = name;
    if (!ui.ensureEditableMeshCache(*g_ctx, name)) {
        ui.active_mesh_edit_object_name.clear();
        return Result::fail("could not build an editable mesh cache for: " + name);
    }

    // ★★★ The half-edge topology is built LAZILY, not when the cache is built.
    // Without this call half_edge_valid reads false on a perfectly good cube --
    // and it reads false with an EMPTY build report (manifold true, zero
    // non-manifold edges, no message), because the builder never ran. That is
    // a defaulted measurement wearing the clothes of a real one: "not built
    // yet" is indistinguishable from "this mesh cannot be built", and the
    // second reading is the one every caller would draw.
    //
    // The four edit-mode operators that need it call it themselves; extrude
    // does not, which is why it silently takes the legacy triangle path.
    ui.ensureEditableHalfEdge();

    // Re-apply what this caller selected. The cache may have been rebuilt (and
    // its selection lost) any number of times since the select call.
    if (g_mesh_edit.object == name) {
        SceneUI::EditableMeshSelection& selection = ui.editable_mesh_cache.selection;
        selection.vertex_ids = g_mesh_edit.vertex_ids;
        selection.edge_ids = g_mesh_edit.edge_ids;
        selection.face_ids = g_mesh_edit.face_ids;
        selection.active_vertex_id = selection.vertex_ids.empty() ? -1 : selection.vertex_ids.back();
        selection.active_edge_id = selection.edge_ids.empty() ? -1 : selection.edge_ids.back();
        selection.active_face_id = selection.face_ids.empty() ? -1 : selection.face_ids.back();
    } else {
        g_mesh_edit.clear();
        g_mesh_edit.object = name;
    }

    out_name = name;
    return Result::success();
}

std::vector<int>& selectionList(SceneUI::EditableMeshSelection& selection,
                                MeshElementDomain domain) {
    switch (domain) {
    case MeshElementDomain::Vertex: return selection.vertex_ids;
    case MeshElementDomain::Edge:   return selection.edge_ids;
    case MeshElementDomain::Face:   break;
    }
    return selection.face_ids;
}

const std::vector<int>& selectionListConst(const MeshEditSession& session,
                                          MeshElementDomain domain) {
    switch (domain) {
    case MeshElementDomain::Vertex: return session.vertex_ids;
    case MeshElementDomain::Edge:   return session.edge_ids;
    case MeshElementDomain::Face:   break;
    }
    return session.face_ids;
}

size_t domainElementCount(const SceneUI::EditableMeshCache& cache,
                          MeshElementDomain domain) {
    switch (domain) {
    case MeshElementDomain::Vertex: return cache.vertices.size();
    case MeshElementDomain::Edge:   return cache.polygon_edges.size();
    case MeshElementDomain::Face:   break;
    }
    return cache.polygon_faces.size();
}

const char* domainLabel(MeshElementDomain domain) {
    switch (domain) {
    case MeshElementDomain::Vertex: return "vertex";
    case MeshElementDomain::Edge:   return "edge";
    case MeshElementDomain::Face:   break;
    }
    return "face";
}

void setActiveId(SceneUI::EditableMeshSelection& selection, MeshElementDomain domain) {
    const std::vector<int>& ids = selectionList(selection, domain);
    const int active = ids.empty() ? -1 : ids.back();
    switch (domain) {
    case MeshElementDomain::Vertex: selection.active_vertex_id = active; return;
    case MeshElementDomain::Edge:   selection.active_edge_id = active; return;
    case MeshElementDomain::Face:   selection.active_face_id = active; return;
    }
}

// Copies the cache's selection into the session so it survives the next cache
// wipe. Called after every selection write.
void rememberSelection(const std::string& object) {
    const SceneUI::EditableMeshSelection& selection = ui.editable_mesh_cache.selection;
    g_mesh_edit.object = object;
    g_mesh_edit.vertex_ids = selection.vertex_ids;
    g_mesh_edit.edge_ids = selection.edge_ids;
    g_mesh_edit.face_ids = selection.face_ids;
}

void applySelection(SceneUI::EditableMeshSelection& selection, MeshElementDomain domain,
                    MeshSelectMode mode, std::vector<int> ids) {
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());

    std::vector<int>& target = selectionList(selection, domain);
    switch (mode) {
    case MeshSelectMode::Set:
        target = std::move(ids);
        break;
    case MeshSelectMode::Add: {
        target.insert(target.end(), ids.begin(), ids.end());
        std::sort(target.begin(), target.end());
        target.erase(std::unique(target.begin(), target.end()), target.end());
        break;
    }
    case MeshSelectMode::Remove: {
        std::vector<int> kept;
        kept.reserve(target.size());
        for (int id : target) {
            if (!std::binary_search(ids.begin(), ids.end(), id)) kept.push_back(id);
        }
        target = std::move(kept);
        break;
    }
    }
    setActiveId(selection, domain);
}

// Area-weighted polygon normal from the cache's flat position buffer. Newell's
// method so a non-planar ngon still yields a meaningful direction instead of
// depending on which triangle happened to be sampled.
bool polygonNormal(const SceneUI::EditableMeshCache& cache, size_t face_id, Vec3& out) {
    if (face_id >= cache.polygon_faces.size()) return false;
    const std::vector<int>& verts = cache.polygon_faces[face_id].vertex_ids;
    if (verts.size() < 3) return false;

    Vec3 normal(0.0f, 0.0f, 0.0f);
    const size_t count = verts.size();
    for (size_t i = 0; i < count; ++i) {
        const int a = verts[i];
        const int b = verts[(i + 1) % count];
        if (a < 0 || b < 0 ||
            a >= static_cast<int>(cache.vertex_positions.size()) ||
            b >= static_cast<int>(cache.vertex_positions.size())) {
            return false;
        }
        const Vec3& p = cache.vertex_positions[static_cast<size_t>(a)];
        const Vec3& q = cache.vertex_positions[static_cast<size_t>(b)];
        normal.x += (p.y - q.y) * (p.z + q.z);
        normal.y += (p.z - q.z) * (p.x + q.x);
        normal.z += (p.x - q.x) * (p.y + q.y);
    }
    const float length = std::sqrt(normal.x * normal.x + normal.y * normal.y +
                                   normal.z * normal.z);
    if (length <= 1e-6f) return false;   // degenerate face, not a zero normal
    out = normal * (1.0f / length);
    return true;
}

// ★ Operators report failure as a bare bool. Turning that into "nothing
// happened" would be a defaulted measurement, so the state is read back and
// the most likely cause is named -- an empty selection is by far the most
// common one and looks identical to a broken operator from outside.
Result operatorResult(bool ok, const std::string& object, const char* operation,
                      MeshElementDomain domain) {
    if (ok) {
        // Topology changed, so every remembered id may now name a different
        // element -- or none. Dropping them turns a stale reuse into a clear
        // "nothing is selected" instead of an edit applied to the wrong faces.
        g_mesh_edit.clear();
        g_mesh_edit.object = object;
        return Result::success();
    }

    // Read the session, not the cache: activateEditObject just re-applied the
    // session into the cache, but reporting the cache here would blame an
    // empty selection for a failure that was really the operator refusing.
    const size_t selected = (g_mesh_edit.object == object)
                                ? selectionListConst(g_mesh_edit, domain).size()
                                : selectionList(ui.editable_mesh_cache.selection, domain).size();
    if (selected == 0) {
        return Result::fail(std::string(operation) + " needs a " + domainLabel(domain) +
                            " selection on '" + object + "'; none is selected");
    }
    return Result::fail(std::string(operation) + " rejected the " +
                        std::to_string(selected) + " selected " + domainLabel(domain) +
                        " element(s) on '" + object +
                        "' (unsupported topology for this operation)");
}

} // namespace

Result meshEditBegin(const std::string& object) {
    std::string name;
    return activateEditObject(object, name);
}

Result meshEditGetState(const std::string& object, MeshEditState& out) {
    if (!g_ctx) return notBound();

    std::string name = object;
    if (name.empty()) name = ui.active_mesh_edit_object_name;
    if (name.empty()) {
        out = {};
        return Result::success();   // no active edit object is a valid answer
    }
    if (!objectExists(name)) return Result::fail("object not found: " + name);

    // ★★ A read must not destroy what it measures. Building the cache for
    // another object used to wipe the selection, which would have made
    // get_state the instrument that breaks its own subject. It is safe now
    // ONLY because the selection lives in the rtapi session rather than in the
    // cache -- activateEditObject re-applies it. Keep that ordering in mind
    // before moving the session back into the cache.
    std::string resolved;
    if (Result r = activateEditObject(name, resolved); !r) return r;

    const SceneUI::EditableMeshCache& cache = ui.editable_mesh_cache;
    out = {};
    out.object = name;
    out.cache_valid = true;
    out.half_edge_valid = cache.half_edge_valid;
    out.vertex_count = cache.vertices.size();
    out.edge_count = cache.polygon_edges.size();
    out.face_count = cache.polygon_faces.size();
    out.triangle_count = cache.faces.size();
    // Report the SESSION's selection, not the cache's: the cache copy is what
    // the frame loop wipes, so reading it would report 0 for a selection that
    // the next operator will happily use.
    const bool mine = (g_mesh_edit.object == name);
    out.selected_vertices = mine ? g_mesh_edit.vertex_ids.size() : cache.selection.vertex_ids.size();
    out.selected_edges = mine ? g_mesh_edit.edge_ids.size() : cache.selection.edge_ids.size();
    out.selected_faces = mine ? g_mesh_edit.face_ids.size() : cache.selection.face_ids.size();
    // ★ A bare half_edge_valid=false is a defaulted measurement: the caller
    // cannot tell "this mesh is non-manifold" from "the build was never run".
    // Carry the builder's own reason across.
    out.half_edge_message = cache.half_edge_build.message;
    out.half_edge_manifold = cache.half_edge_build.manifold;
    out.non_manifold_edges = cache.half_edge_build.non_manifold_edges;
    out.skipped_polygons = cache.half_edge_build.skipped_polygons;
    return Result::success();
}

Result meshEditSelect(const std::string& object, MeshElementDomain domain,
                      MeshSelectMode mode, const std::vector<int>& ids) {
    std::string name;
    if (Result r = activateEditObject(object, name); !r) return r;

    const size_t limit = domainElementCount(ui.editable_mesh_cache, domain);
    for (int id : ids) {
        if (id < 0 || static_cast<size_t>(id) >= limit) {
            return Result::fail(std::string(domainLabel(domain)) + " id " +
                                std::to_string(id) + " is out of range on '" + name +
                                "' (0.." + std::to_string(limit ? limit - 1 : 0) + ")");
        }
    }

    applySelection(ui.editable_mesh_cache.selection, domain, mode, ids);
    rememberSelection(name);
    return Result::success();
}

Result meshEditSelectAll(const std::string& object, MeshElementDomain domain) {
    std::string name;
    if (Result r = activateEditObject(object, name); !r) return r;

    const size_t count = domainElementCount(ui.editable_mesh_cache, domain);
    std::vector<int> ids;
    ids.reserve(count);
    for (size_t i = 0; i < count; ++i) ids.push_back(static_cast<int>(i));

    applySelection(ui.editable_mesh_cache.selection, domain, MeshSelectMode::Set,
                   std::move(ids));
    rememberSelection(name);
    return Result::success();
}

Result meshEditClearSelection(const std::string& object) {
    std::string name;
    if (Result r = activateEditObject(object, name); !r) return r;

    ui.editable_mesh_cache.selection = SceneUI::EditableMeshSelection{};
    rememberSelection(name);
    return Result::success();
}

Result meshEditGetSelection(const std::string& object, MeshElementDomain domain,
                            std::vector<int>& out) {
    std::string name;
    if (Result r = activateEditObject(object, name); !r) return r;

    out = (g_mesh_edit.object == name)
              ? selectionListConst(g_mesh_edit, domain)
              : selectionList(ui.editable_mesh_cache.selection, domain);
    return Result::success();
}

Result meshEditSelectByNormal(const std::string& object, const Vec3& direction,
                              float max_angle_degrees, MeshSelectMode mode,
                              size_t* out_count) {
    std::string name;
    if (Result r = activateEditObject(object, name); !r) return r;

    const float length = std::sqrt(direction.x * direction.x + direction.y * direction.y +
                                    direction.z * direction.z);
    if (length <= 1e-6f) return Result::fail("direction must be a non-zero vector");
    const Vec3 axis = direction * (1.0f / length);

    const float angle = std::clamp(max_angle_degrees, 0.0f, 180.0f);
    const float min_dot = std::cos(angle * 3.14159265358979323846f / 180.0f);

    const SceneUI::EditableMeshCache& cache = ui.editable_mesh_cache;
    std::vector<int> hits;
    for (size_t face = 0; face < cache.polygon_faces.size(); ++face) {
        Vec3 normal;
        if (!polygonNormal(cache, face, normal)) continue;
        const float dot = normal.x * axis.x + normal.y * axis.y + normal.z * axis.z;
        if (dot >= min_dot) hits.push_back(static_cast<int>(face));
    }

    if (out_count) *out_count = hits.size();
    applySelection(ui.editable_mesh_cache.selection, MeshElementDomain::Face, mode,
                   std::move(hits));
    rememberSelection(name);
    return Result::success();
}

Result meshEditSelectByBox(const std::string& object, const Vec3& box_min,
                           const Vec3& box_max, MeshElementDomain domain,
                           MeshSelectMode mode, bool world_space, size_t* out_count) {
    std::string name;
    if (Result r = activateEditObject(object, name); !r) return r;

    const Vec3 lo(std::min(box_min.x, box_max.x), std::min(box_min.y, box_max.y),
                  std::min(box_min.z, box_max.z));
    const Vec3 hi(std::max(box_min.x, box_max.x), std::max(box_min.y, box_max.y),
                  std::max(box_min.z, box_max.z));

    // The cache stores object-space positions. A world-space box is compared
    // by pushing each vertex out through the object's world transform rather
    // than inverting the box, so a rotated object still selects correctly.
    Matrix4x4 world;
    bool have_world = false;
    if (world_space) {
        have_world = getObjectWorldTransform(name, world, nullptr).ok;
        if (!have_world) {
            return Result::fail("world-space selection needs a world transform for: " + name);
        }
    }

    const SceneUI::EditableMeshCache& cache = ui.editable_mesh_cache;
    auto positionOf = [&](int vertex) -> Vec3 {
        const Vec3& local = cache.vertex_positions[static_cast<size_t>(vertex)];
        return have_world ? world.transform_point(local) : local;
    };
    auto inside = [&](const Vec3& p) {
        return p.x >= lo.x && p.x <= hi.x && p.y >= lo.y && p.y <= hi.y &&
               p.z >= lo.z && p.z <= hi.z;
    };
    auto vertexValid = [&](int v) {
        return v >= 0 && static_cast<size_t>(v) < cache.vertex_positions.size();
    };

    std::vector<int> hits;
    switch (domain) {
    case MeshElementDomain::Vertex:
        for (size_t v = 0; v < cache.vertex_positions.size(); ++v) {
            if (inside(positionOf(static_cast<int>(v)))) hits.push_back(static_cast<int>(v));
        }
        break;
    case MeshElementDomain::Edge:
        // Both endpoints must be inside: a partially covered edge cannot be
        // dissolved or cut meaningfully, so including it would hand the
        // operator elements it will reject.
        for (size_t e = 0; e < cache.polygon_edges.size(); ++e) {
            const auto& edge = cache.polygon_edges[e];
            if (!vertexValid(edge.v0) || !vertexValid(edge.v1)) continue;
            if (inside(positionOf(edge.v0)) && inside(positionOf(edge.v1))) {
                hits.push_back(static_cast<int>(e));
            }
        }
        break;
    case MeshElementDomain::Face:
        for (size_t f = 0; f < cache.polygon_faces.size(); ++f) {
            const std::vector<int>& verts = cache.polygon_faces[f].vertex_ids;
            if (verts.size() < 3) continue;
            bool all_inside = true;
            for (int v : verts) {
                if (!vertexValid(v) || !inside(positionOf(v))) { all_inside = false; break; }
            }
            if (all_inside) hits.push_back(static_cast<int>(f));
        }
        break;
    }

    if (out_count) *out_count = hits.size();
    applySelection(ui.editable_mesh_cache.selection, domain, mode, std::move(hits));
    rememberSelection(name);
    return Result::success();
}

// --- Operators -------------------------------------------------------------

Result meshExtrudeFaces(const std::string& object, float distance) {
    std::string name;
    if (Result r = activateEditObject(object, name); !r) return r;
    if (std::fabs(distance) <= 1e-6f) {
        return Result::fail("extrude distance must be non-zero");
    }
    return operatorResult(ui.extrudeSelectedMeshFaces(*g_ctx, distance), name,
                          "extrude", MeshElementDomain::Face);
}

Result meshInsetFaces(const std::string& object, float amount) {
    std::string name;
    if (Result r = activateEditObject(object, name); !r) return r;
    if (amount <= 0.0f) return Result::fail("inset amount must be positive");
    return operatorResult(ui.insetSelectedMeshFaces(*g_ctx, amount), name,
                          "inset", MeshElementDomain::Face);
}

Result meshBevelEdges(const std::string& object, float width, int segments,
                      bool round_profile) {
    std::string name;
    if (Result r = activateEditObject(object, name); !r) return r;
    if (width <= 0.0f) return Result::fail("bevel width must be positive");
    if (segments < 1) return Result::fail("bevel segments must be at least 1");
    return operatorResult(ui.bevelSelectedEdges(*g_ctx, width, segments, round_profile),
                          name, "bevel", MeshElementDomain::Edge);
}

Result meshLoopCut(const std::string& object, float t) {
    std::string name;
    if (Result r = activateEditObject(object, name); !r) return r;
    if (!(t > 0.0f && t < 1.0f)) {
        return Result::fail("loop cut t must lie strictly between 0 and 1");
    }
    return operatorResult(ui.loopCutSelectedEdges(*g_ctx, t), name,
                          "loop cut", MeshElementDomain::Edge);
}

Result meshDissolveEdges(const std::string& object) {
    std::string name;
    if (Result r = activateEditObject(object, name); !r) return r;
    return operatorResult(ui.dissolveSelectedEdges(*g_ctx), name,
                          "dissolve edges", MeshElementDomain::Edge);
}

Result meshDissolveVertices(const std::string& object) {
    std::string name;
    if (Result r = activateEditObject(object, name); !r) return r;
    return operatorResult(ui.dissolveSelectedVertices(*g_ctx), name,
                          "dissolve vertices", MeshElementDomain::Vertex);
}

Result meshMergeVertices(const std::string& object) {
    std::string name;
    if (Result r = activateEditObject(object, name); !r) return r;
    return operatorResult(ui.mergeSelectedVerticesToCenter(*g_ctx), name,
                          "merge vertices", MeshElementDomain::Vertex);
}

Result meshWeldVertices(const std::string& object, float distance) {
    std::string name;
    if (Result r = activateEditObject(object, name); !r) return r;
    if (distance <= 0.0f) return Result::fail("weld distance must be positive");
    return operatorResult(ui.weldSelectedVerticesByDistance(*g_ctx, distance), name,
                          "weld vertices", MeshElementDomain::Vertex);
}

} // namespace rtapi
