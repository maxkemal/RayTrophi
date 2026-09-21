/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          Api/RtIpcMeshEdit.cpp
 * Author:        Kemal Demirtas
 * License:       MIT
 * =========================================================================
 * IPC adapter for polygon mesh editing.
 *
 * ★★★ mesh.tools.list has advertised extrude, inset, loop cut and dissolve as
 * "scriptable": true, "ipc_exposed": true for a long time while NO method
 * existed to call them. This adapter is the address those tools claimed to
 * have. bevel, dissolve vertices, merge and weld were never in the catalogue
 * at all and are opened here too.
 *
 * ★★ Everything runs through the caller's enqueue, never by calling rtapi
 * inline: these operators rebuild topology, publish a flat mesh and record an
 * undo command, so running them on the IPC thread would race the frame loop
 * that reads the very same mesh.
 */

#include "Api/RtIpcMeshEdit.h"

#include "Api/RtApi.h"

#include <string>
#include <vector>

namespace {

using nlohmann::json;

json errorResult(std::string message) {
    return json{{"__error", std::move(message)}};
}

bool parseDomain(const std::string& name, rtapi::MeshElementDomain& out) {
    if (name == "vertex") { out = rtapi::MeshElementDomain::Vertex; return true; }
    if (name == "edge")   { out = rtapi::MeshElementDomain::Edge;   return true; }
    if (name == "face")   { out = rtapi::MeshElementDomain::Face;   return true; }
    return false;
}

bool parseMode(const std::string& name, rtapi::MeshSelectMode& out) {
    if (name == "set")    { out = rtapi::MeshSelectMode::Set;    return true; }
    if (name == "add")    { out = rtapi::MeshSelectMode::Add;    return true; }
    if (name == "remove") { out = rtapi::MeshSelectMode::Remove; return true; }
    return false;
}

bool parseVec3(const json& params, const char* key, Vec3& out) {
    if (!params.contains(key)) return false;
    const json& value = params.at(key);
    if (!value.is_array() || value.size() != 3) return false;
    for (const auto& component : value) {
        if (!component.is_number()) return false;
    }
    out = Vec3(value[0].get<float>(), value[1].get<float>(), value[2].get<float>());
    return true;
}

json stateToJson(const rtapi::MeshEditState& state) {
    return json{
        {"object", state.object},
        {"cache_valid", state.cache_valid},
        // ★ half_edge_valid false means the operators fall back to the legacy
        // triangle-soup path, where several of them simply refuse. Without it
        // a caller reads "operation rejected" and cannot tell an unsupported
        // selection from a mesh whose topology never built.
        {"half_edge_valid", state.half_edge_valid},
        {"vertices", state.vertex_count},
        {"edges", state.edge_count},
        {"faces", state.face_count},
        {"triangles", state.triangle_count},
        {"selected_vertices", state.selected_vertices},
        {"selected_edges", state.selected_edges},
        {"selected_faces", state.selected_faces},
        {"half_edge_manifold", state.half_edge_manifold},
        {"non_manifold_edges", state.non_manifold_edges},
        {"skipped_polygons", state.skipped_polygons},
        {"half_edge_message", state.half_edge_message}
    };
}

// Every operator answers with the resulting state, so a caller sees what the
// edit produced without a second round trip -- and, because ids are
// invalidated by a topology change, immediately learns the new element counts.
json runOperator(const std::string& object, const rtapi::Result& result) {
    if (!result.ok) return errorResult(result.error);
    rtapi::MeshEditState state;
    if (rtapi::Result r = rtapi::meshEditGetState(object, state); !r) {
        return errorResult(r.error);
    }
    json out = stateToJson(state);
    out["ok"] = true;
    return out;
}

} // namespace

bool dispatchMeshEditIpc(const std::string& method,
                         const nlohmann::json& params,
                         const RtIpcTemplateEnqueue& enqueue_query,
                         nlohmann::json& out_result) {
    // ── State and selection ────────────────────────────────────────────
    if (method == "mesh.edit.begin") {
        const std::string object = params.value("object", "");
        if (object.empty()) { out_result = errorResult("missing required parameter: object"); return true; }
        out_result = enqueue_query([object](UIContext&) {
            if (rtapi::Result r = rtapi::meshEditBegin(object); !r) return errorResult(r.error);
            rtapi::MeshEditState state;
            if (rtapi::Result r = rtapi::meshEditGetState(object, state); !r) return errorResult(r.error);
            return stateToJson(state);
        });
        return true;
    }

    if (method == "mesh.edit.get_state") {
        const std::string object = params.value("object", "");
        out_result = enqueue_query([object](UIContext&) {
            rtapi::MeshEditState state;
            if (rtapi::Result r = rtapi::meshEditGetState(object, state); !r) return errorResult(r.error);
            return stateToJson(state);
        });
        return true;
    }

    if (method == "mesh.edit.select") {
        const std::string object = params.value("object", "");
        const std::string domain_name = params.value("domain", "");
        const std::string mode_name = params.value("mode", "set");

        rtapi::MeshElementDomain domain{};
        if (!parseDomain(domain_name, domain)) {
            out_result = errorResult("domain must be one of vertex|edge|face");
            return true;
        }
        rtapi::MeshSelectMode mode{};
        if (!parseMode(mode_name, mode)) {
            out_result = errorResult("mode must be one of set|add|remove");
            return true;
        }

        // `all: true` replaces an id list the caller cannot realistically
        // produce by hand for a dense mesh.
        const bool select_all = params.value("all", false);
        std::vector<int> ids;
        if (!select_all) {
            if (!params.contains("ids") || !params.at("ids").is_array()) {
                out_result = errorResult("mesh.edit.select needs 'ids' (an array) or 'all': true");
                return true;
            }
            for (const auto& id : params.at("ids")) {
                if (!id.is_number_integer()) {
                    out_result = errorResult("ids must be integers");
                    return true;
                }
                ids.push_back(id.get<int>());
            }
        }

        out_result = enqueue_query([object, domain, mode, ids, select_all](UIContext&) {
            const rtapi::Result r = select_all
                ? rtapi::meshEditSelectAll(object, domain)
                : rtapi::meshEditSelect(object, domain, mode, ids);
            if (!r) return errorResult(r.error);
            rtapi::MeshEditState state;
            if (rtapi::Result s = rtapi::meshEditGetState(object, state); !s) return errorResult(s.error);
            return stateToJson(state);
        });
        return true;
    }

    if (method == "mesh.edit.clear_selection") {
        const std::string object = params.value("object", "");
        out_result = enqueue_query([object](UIContext&) {
            if (rtapi::Result r = rtapi::meshEditClearSelection(object); !r) return errorResult(r.error);
            rtapi::MeshEditState state;
            if (rtapi::Result s = rtapi::meshEditGetState(object, state); !s) return errorResult(s.error);
            return stateToJson(state);
        });
        return true;
    }

    if (method == "mesh.edit.get_selection") {
        const std::string object = params.value("object", "");
        const std::string domain_name = params.value("domain", "");
        rtapi::MeshElementDomain domain{};
        if (!parseDomain(domain_name, domain)) {
            out_result = errorResult("domain must be one of vertex|edge|face");
            return true;
        }
        out_result = enqueue_query([object, domain, domain_name](UIContext&) {
            std::vector<int> ids;
            if (rtapi::Result r = rtapi::meshEditGetSelection(object, domain, ids); !r) {
                return errorResult(r.error);
            }
            return json{{"domain", domain_name}, {"count", ids.size()}, {"ids", ids}};
        });
        return true;
    }

    if (method == "mesh.edit.select_by_normal") {
        const std::string object = params.value("object", "");
        Vec3 direction;
        if (!parseVec3(params, "direction", direction)) {
            out_result = errorResult("direction must be an array of three numbers");
            return true;
        }
        const float max_angle = params.value("max_angle", 30.0f);
        rtapi::MeshSelectMode mode{};
        if (!parseMode(params.value("mode", "set"), mode)) {
            out_result = errorResult("mode must be one of set|add|remove");
            return true;
        }
        out_result = enqueue_query([object, direction, max_angle, mode](UIContext&) {
            size_t matched = 0;
            if (rtapi::Result r = rtapi::meshEditSelectByNormal(object, direction, max_angle,
                                                                mode, &matched); !r) {
                return errorResult(r.error);
            }
            rtapi::MeshEditState state;
            if (rtapi::Result s = rtapi::meshEditGetState(object, state); !s) return errorResult(s.error);
            json out = stateToJson(state);
            out["matched"] = matched;
            return out;
        });
        return true;
    }

    if (method == "mesh.edit.select_by_box") {
        const std::string object = params.value("object", "");
        Vec3 box_min, box_max;
        if (!parseVec3(params, "min", box_min) || !parseVec3(params, "max", box_max)) {
            out_result = errorResult("min and max must each be an array of three numbers");
            return true;
        }
        rtapi::MeshElementDomain domain{};
        if (!parseDomain(params.value("domain", "face"), domain)) {
            out_result = errorResult("domain must be one of vertex|edge|face");
            return true;
        }
        rtapi::MeshSelectMode mode{};
        if (!parseMode(params.value("mode", "set"), mode)) {
            out_result = errorResult("mode must be one of set|add|remove");
            return true;
        }
        const bool world_space = params.value("world_space", true);
        out_result = enqueue_query([object, box_min, box_max, domain, mode, world_space](UIContext&) {
            size_t matched = 0;
            if (rtapi::Result r = rtapi::meshEditSelectByBox(object, box_min, box_max, domain,
                                                              mode, world_space, &matched); !r) {
                return errorResult(r.error);
            }
            rtapi::MeshEditState state;
            if (rtapi::Result s = rtapi::meshEditGetState(object, state); !s) return errorResult(s.error);
            json out = stateToJson(state);
            out["matched"] = matched;
            return out;
        });
        return true;
    }

    // ── Operators ──────────────────────────────────────────────────────
    if (method == "mesh.extrude") {
        const std::string object = params.value("object", "");
        const float distance = params.value("distance", 0.0f);
        out_result = enqueue_query([object, distance](UIContext&) {
            return runOperator(object, rtapi::meshExtrudeFaces(object, distance));
        });
        return true;
    }

    if (method == "mesh.inset") {
        const std::string object = params.value("object", "");
        const float amount = params.value("amount", 0.0f);
        out_result = enqueue_query([object, amount](UIContext&) {
            return runOperator(object, rtapi::meshInsetFaces(object, amount));
        });
        return true;
    }

    if (method == "mesh.bevel") {
        const std::string object = params.value("object", "");
        const float width = params.value("width", 0.0f);
        const int segments = params.value("segments", 1);
        const bool round_profile = params.value("round", false);
        out_result = enqueue_query([object, width, segments, round_profile](UIContext&) {
            return runOperator(object, rtapi::meshBevelEdges(object, width, segments, round_profile));
        });
        return true;
    }

    if (method == "mesh.loop_cut") {
        const std::string object = params.value("object", "");
        const float t = params.value("t", 0.5f);
        out_result = enqueue_query([object, t](UIContext&) {
            return runOperator(object, rtapi::meshLoopCut(object, t));
        });
        return true;
    }

    if (method == "mesh.dissolve_edges") {
        const std::string object = params.value("object", "");
        out_result = enqueue_query([object](UIContext&) {
            return runOperator(object, rtapi::meshDissolveEdges(object));
        });
        return true;
    }

    if (method == "mesh.dissolve_vertices") {
        const std::string object = params.value("object", "");
        out_result = enqueue_query([object](UIContext&) {
            return runOperator(object, rtapi::meshDissolveVertices(object));
        });
        return true;
    }

    if (method == "mesh.merge_vertices") {
        const std::string object = params.value("object", "");
        out_result = enqueue_query([object](UIContext&) {
            return runOperator(object, rtapi::meshMergeVertices(object));
        });
        return true;
    }

    if (method == "mesh.weld_vertices") {
        const std::string object = params.value("object", "");
        const float distance = params.value("distance", 0.0f);
        out_result = enqueue_query([object, distance](UIContext&) {
            return runOperator(object, rtapi::meshWeldVertices(object, distance));
        });
        return true;
    }

    return false;
}
