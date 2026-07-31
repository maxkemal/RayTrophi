/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          Api/RtApiNodes.cpp
* Author:        Kemal Demirtas
* Date:          July 2026
* License:       MIT
* =========================================================================
*
* Node graph facade: construction (Faz 3d), pin parameters (Faz 5.1b) and the
* serialized property reflection (Faz 5.5b).
*
* The reflection layer is the important part. Instead of hand-writing a binding
* per parameter, a node is serialized to JSON, the tree is walked into dotted
* paths with inferred types, and a write deserializes back onto the node. Every
* node family that implements its serialization hook is scriptable for free.
*
* Moved out of RtApi.cpp (past its size budget) when the reflection gate was
* widened past terrain nodes.
*/

#include "RtApiInternal.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "json.hpp"
#include "GeometryNodesV2.h"
#include "MaterialNodesV2.h"
#include "MaterialGraphApply.h"
#include "MaterialManager.h"
#include "PrincipledBSDF.h"
#include "TerrainManager.h"
#include "TerrainNodesV2.h"
#include "NodeSystem/Graph.h"
#include "NodeSystem/NodeRegistry.h"

namespace rtapi {

// ---------------------------------------------------------------------------
// Node graphs (Faz 3d).
// ---------------------------------------------------------------------------
namespace {

NodeSystem::GraphBase* findNodeGraph(UIContext& ctx, const std::string& graph_type,
                                     const std::string& graph_name, Result& err) {
    if (graph_type == "material") {
        auto it = ctx.scene.material_node_graphs.find(graph_name);
        if (it == ctx.scene.material_node_graphs.end() || !it->second) {
            err = Result::fail("material node graph not found: " + graph_name);
            return nullptr;
        }
        return it->second.get();
    }
    if (graph_type == "geometry") {
        auto it = ctx.scene.geometry_node_graphs.find(graph_name);
        if (it == ctx.scene.geometry_node_graphs.end() || !it->second) {
            err = Result::fail("geometry node graph not found: " + graph_name);
            return nullptr;
        }
        return it->second.get();
    }
    if (graph_type == "terrain") {
        TerrainObject* terrain = TerrainManager::getInstance().getTerrainByName(graph_name);
        if (!terrain || !terrain->nodeGraph) {
            err = Result::fail("terrain node graph not found: " + graph_name);
            return nullptr;
        }
        return terrain->nodeGraph.get();
    }
    err = Result::fail("unknown graph_type '" + graph_type + "' (expected material|geometry|terrain)");
    return nullptr;
}

} // namespace

std::vector<std::string> listNodeGraphs(const std::string& graph_type) {
    std::vector<std::string> out;
    if (!g_ctx) return out;
    if (graph_type == "material") {
        for (const auto& [name, graph] : g_ctx->scene.material_node_graphs)
            if (graph) out.push_back(name);
    } else if (graph_type == "geometry") {
        for (const auto& [name, graph] : g_ctx->scene.geometry_node_graphs)
            if (graph) out.push_back(name);
    } else if (graph_type == "terrain") {
        for (const TerrainObject& terrain : TerrainManager::getInstance().getTerrains())
            if (terrain.nodeGraph) out.push_back(terrain.name);
    }
    std::sort(out.begin(), out.end());
    return out;
}

Result createNodeGraph(const std::string& graph_type, const std::string& graph_name) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (graph_name.empty()) return Result::fail("graph name must not be empty");

    if (graph_type == "material") {
        // Material graphs are keyed by material name, so the asset must exist.
        auto& manager = MaterialManager::getInstance();
        const uint16_t id = manager.getMaterialID(graph_name);
        if (id == MaterialManager::INVALID_MATERIAL_ID)
            return Result::fail("material not found: " + graph_name);

        auto& graph = g_ctx->scene.material_node_graphs[graph_name];
        if (!graph) graph = std::make_shared<MaterialNodesV2::MaterialNodeGraphV2>();
        // Seed it from the material exactly like the node editor does, so a
        // scripted graph starts from the surface the artist already sees rather
        // than an empty canvas. Volumetrics stay empty here: their materialize
        // helper lives in a UI header (scene_ui_materialnodes.hpp) and pulling
        // ImGui into the facade to reach it is not worth it — open the node
        // editor once for those, or build the graph node by node.
        if (graph->nodes.empty()) {
            if (auto* surface = dynamic_cast<PrincipledBSDF*>(manager.getMaterial(id)))
                MaterialNodesV2::materializeGraphFromMaterial(*graph, *surface);
        }
        return Result::success();
    }

    if (graph_type == "geometry") {
        // Geometry graphs are keyed by object nodeName.
        if (!objectExists(graph_name)) return Result::fail("object not found: " + graph_name);
        auto& graph = g_ctx->scene.geometry_node_graphs[graph_name];
        if (!graph) graph = std::make_shared<GeometryNodesV2::GeometryNodeGraphV2>();
        return Result::success();
    }

    if (graph_type == "terrain") {
        // A terrain's graph is owned by its TerrainObject and built by the
        // preset system; creating a bare one here would bypass that ownership.
        return Result::fail("terrain graphs are created by rt.terrain.apply_preset");
    }

    return Result::fail("unknown graph_type '" + graph_type + "' (expected material|geometry|terrain)");
}

Result removeNodeGraph(const std::string& graph_type, const std::string& graph_name) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");

    if (graph_type == "material") {
        if (g_ctx->scene.material_node_graphs.erase(graph_name) == 0)
            return Result::fail("material node graph not found: " + graph_name);
        return Result::success();
    }
    if (graph_type == "geometry") {
        if (g_ctx->scene.geometry_node_graphs.erase(graph_name) == 0)
            return Result::fail("geometry node graph not found: " + graph_name);
        return Result::success();
    }
    if (graph_type == "terrain")
        return Result::fail("terrain graphs are owned by the terrain object");
    return Result::fail("unknown graph_type '" + graph_type + "' (expected material|geometry|terrain)");
}

Result applyNodeGraph(const std::string& graph_type, const std::string& graph_name,
                      NodeGraphApplyInfo& out) {
    out = {};
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");

    Result err;
    NodeSystem::GraphBase* graph = findNodeGraph(*g_ctx, graph_type, graph_name, err);
    if (!graph) return err;

    if (graph_type == "material") {
        auto* material_graph = dynamic_cast<MaterialNodesV2::MaterialNodeGraphV2*>(graph);
        if (!material_graph) return Result::fail("not a material graph: " + graph_name);

        auto& manager = MaterialManager::getInstance();
        const uint16_t id = manager.getMaterialID(graph_name);
        if (id == MaterialManager::INVALID_MATERIAL_ID)
            return Result::fail("material not found: " + graph_name);
        auto* surface = dynamic_cast<PrincipledBSDF*>(manager.getMaterial(id));
        if (!surface)
            return Result::fail("apply currently supports Principled BSDF materials: " + graph_name);

        // Exactly the path the node editor's Apply takes — same fold, same
        // compile, same publish. The OptiX per-triangle bundle refresh is the
        // one editor-side step not run here (it needs CUDA types, and on a flat
        // scene it walks legacy Triangle soup and does nothing).
        const MaterialNodesV2::GraphApplyReport report =
            MaterialNodesV2::applyMaterialGraph(*g_ctx, *material_graph, surface, id, true);
        out.ok = report.ok;
        out.warnings = report.warnings;
        out.errors = report.errors;
        g_ctx->start_render = true;
        if (!report.ok) {
            std::string message = "material graph apply failed";
            if (!report.errors.empty()) message += ": " + report.errors.front();
            return Result::fail(message);
        }
        return Result::success();
    }

    if (graph_type == "geometry") {
        auto* geometry_graph = dynamic_cast<GeometryNodesV2::GeometryNodeGraphV2*>(graph);
        if (!geometry_graph) return Result::fail("not a geometry graph: " + graph_name);
        // Geo-DAG apply runs the graph and swaps the object's TriangleMesh in
        // world.objects; SceneUI owns that flow, so reuse it rather than fork it.
        out.ok = ui.evaluateGeometryGraph(*g_ctx, graph_name, *geometry_graph);
        g_ctx->start_render = true;
        if (!out.ok) {
            out.errors.push_back("geometry graph evaluation failed");
            return Result::fail("geometry graph apply failed: " + graph_name);
        }
        return Result::success();
    }

    if (graph_type == "terrain") {
        // Terrain evaluation is a long async bake with progress and cancel —
        // a different contract from this synchronous apply.
        return Result::fail("use rt.terrain.evaluate for terrain graphs");
    }

    return Result::fail("unknown graph_type '" + graph_type + "' (expected material|geometry|terrain)");
}

std::vector<NodeTypeDesc> listNodeTypes() {
    std::vector<NodeTypeDesc> out;
    for (const auto& info : NodeSystem::NodeRegistry::instance().getAllTypes()) {
        NodeTypeDesc d;
        d.type_id = info.typeId;
        d.category = info.category;
        d.display_name = info.displayName;
        d.description = info.description;
        out.push_back(std::move(d));
    }
    return out;
}

Result addNode(const std::string& graph_type, const std::string& graph_name,
               const std::string& type_id, unsigned int& out_node_id) {
    if (!g_ctx) return notBound();
    Result err;
    NodeSystem::GraphBase* graph = findNodeGraph(*g_ctx, graph_type, graph_name, err);
    if (!graph) return err;

    std::shared_ptr<NodeSystem::NodeBase> node =
        NodeSystem::NodeRegistry::instance().create(type_id);
    if (!node) return Result::fail("unknown node type: " + type_id);

    NodeSystem::NodeBase* added = graph->registerNode(std::move(node));
    out_node_id = added->id;
    graph->markAllDirty();
    return Result::success();
}

Result removeNode(const std::string& graph_type, const std::string& graph_name,
                  unsigned int node_id) {
    if (!g_ctx) return notBound();
    Result err;
    NodeSystem::GraphBase* graph = findNodeGraph(*g_ctx, graph_type, graph_name, err);
    if (!graph) return err;
    if (!graph->getNode(node_id)) return Result::fail("node id not found: " + std::to_string(node_id));
    graph->removeNode(node_id);
    graph->markAllDirty();
    return Result::success();
}

Result linkNodes(const std::string& graph_type, const std::string& graph_name,
                 unsigned int from_node, int from_output, unsigned int to_node,
                 int to_input, unsigned int& out_link_id) {
    if (!g_ctx) return notBound();
    Result err;
    NodeSystem::GraphBase* graph = findNodeGraph(*g_ctx, graph_type, graph_name, err);
    if (!graph) return err;

    NodeSystem::NodeBase* src = graph->getNode(from_node);
    NodeSystem::NodeBase* dst = graph->getNode(to_node);
    if (!src) return Result::fail("from_node not found: " + std::to_string(from_node));
    if (!dst) return Result::fail("to_node not found: " + std::to_string(to_node));
    if (from_output < 0 || from_output >= static_cast<int>(src->outputs.size()))
        return Result::fail("from_output index out of range");
    if (to_input < 0 || to_input >= static_cast<int>(dst->inputs.size()))
        return Result::fail("to_input index out of range");

    const uint32_t start_pin = src->outputs[from_output].id;
    const uint32_t end_pin = dst->inputs[to_input].id;
    const uint32_t link_id = graph->addLink(start_pin, end_pin);
    if (link_id == 0) {
        return Result::fail("link rejected (type/semantic mismatch, would create a cycle, "
                            "or invalid pins)");
    }
    out_link_id = link_id;
    return Result::success();
}

Result listNodes(const std::string& graph_type, const std::string& graph_name,
                 std::vector<NodeDesc>& out) {
    out.clear();
    if (!g_ctx) return notBound();
    Result err;
    NodeSystem::GraphBase* graph = findNodeGraph(*g_ctx, graph_type, graph_name, err);
    if (!graph) return err;

    for (const auto& n : graph->nodes) {
        if (!n) continue;
        NodeDesc d;
        d.id = n->id;
        d.type_id = n->metadata.typeId.empty() ? n->getTypeId() : n->metadata.typeId;
        d.display_name = n->metadata.displayName.empty() ? n->metadata.typeId : n->metadata.displayName;
        d.input_count = static_cast<int>(n->inputs.size());
        d.output_count = static_cast<int>(n->outputs.size());
        out.push_back(std::move(d));
    }
    return Result::success();
}

// ---------------------------------------------------------------------------
// Node parameters (Faz 5.1b). A node's input-pin default values are its
// scriptable parameters. findNodeGraph (above) locates the graph.
// ---------------------------------------------------------------------------
namespace {

const char* nodeDataTypeName(NodeSystem::DataType t) {
    using DT = NodeSystem::DataType;
    switch (t) {
        case DT::Float:    return "float";
        case DT::Int:      return "int";
        case DT::Bool:     return "bool";
        case DT::Vector2:  return "vector2";
        case DT::Vector3:  return "vector3";
        case DT::Vector4:  return "vector4";
        case DT::Color:    return "color";
        case DT::String:   return "string";
        case DT::Image2D:  return "image2d";
        case DT::Geometry: return "geometry";
        case DT::Material: return "material";
        default:           return "none";
    }
}

// PinValue (variant) -> NodeParamValue. Reports whichever alternative is stored,
// regardless of the pin's declared type (an unset default reads back as None).
NodeParamValue pinValueToParam(const NodeSystem::PinValue& v) {
    NodeParamValue out;
    using K = NodeParamValue::Kind;
    if (auto* f = std::get_if<float>(&v)) {
        out.kind = K::Float; out.floats[0] = *f;
    } else if (auto* i = std::get_if<int>(&v)) {
        out.kind = K::Int; out.int_value = *i; out.floats[0] = static_cast<float>(*i);
    } else if (auto* b = std::get_if<bool>(&v)) {
        out.kind = K::Bool; out.bool_value = *b; out.floats[0] = *b ? 1.0f : 0.0f;
    } else if (auto* v2 = std::get_if<std::array<float, 2>>(&v)) {
        out.kind = K::Vector2; out.floats[0] = (*v2)[0]; out.floats[1] = (*v2)[1];
    } else if (auto* v3 = std::get_if<std::array<float, 3>>(&v)) {
        out.kind = K::Vector3; out.floats[0] = (*v3)[0]; out.floats[1] = (*v3)[1]; out.floats[2] = (*v3)[2];
    } else if (auto* v4 = std::get_if<std::array<float, 4>>(&v)) {
        out.kind = K::Vector4;
        for (int k = 0; k < 4; ++k) out.floats[k] = (*v4)[k];
    } else if (auto* s = std::get_if<std::string>(&v)) {
        out.kind = K::String; out.string_value = *s;
    }
    return out;
}

// Write a NodeParamValue into a pin's defaultValue, COERCED to the pin's declared
// data type. The graph evaluator reads defaultValue expecting a specific variant
// alternative (tryGetFloat etc.), so we must store the alternative that matches
// the pin — not whatever the caller happened to pass.
Result setPinDefault(NodeSystem::Pin& pin, const NodeParamValue& val) {
    using DT = NodeSystem::DataType;
    using K = NodeParamValue::Kind;
    auto scalar = [&]() -> float {
        switch (val.kind) {
            case K::Int:  return static_cast<float>(val.int_value);
            case K::Bool: return val.bool_value ? 1.0f : 0.0f;
            default:      return val.floats[0];  // Float / Vector* use component 0
        }
    };
    switch (pin.dataType) {
        case DT::Float:
            pin.defaultValue = scalar();
            break;
        case DT::Int:
            pin.defaultValue = (val.kind == K::Int) ? val.int_value
                                                    : static_cast<int>(scalar());
            break;
        case DT::Bool:
            pin.defaultValue = (val.kind == K::Bool) ? val.bool_value : (scalar() != 0.0f);
            break;
        case DT::Vector2:
            pin.defaultValue = std::array<float, 2>{ val.floats[0], val.floats[1] };
            break;
        case DT::Vector3:
            pin.defaultValue = std::array<float, 3>{ val.floats[0], val.floats[1], val.floats[2] };
            break;
        case DT::Vector4:
        case DT::Color:
            pin.defaultValue = std::array<float, 4>{ val.floats[0], val.floats[1],
                                                     val.floats[2], val.floats[3] };
            break;
        case DT::String:
            if (val.kind != K::String)
                return Result::fail("pin '" + pin.name + "' is a string; provide a string value");
            pin.defaultValue = val.string_value;
            break;
        default:
            return Result::fail(std::string("pin '") + pin.name + "' (type " +
                                nodeDataTypeName(pin.dataType) + ") has no scriptable default value");
    }
    return Result::success();
}

} // namespace

Result listNodeParams(const std::string& graph_type, const std::string& graph_name,
                      unsigned int node_id, std::vector<NodeParamInfo>& out) {
    out.clear();
    if (!g_ctx) return notBound();
    Result err;
    NodeSystem::GraphBase* graph = findNodeGraph(*g_ctx, graph_type, graph_name, err);
    if (!graph) return err;
    NodeSystem::NodeBase* node = graph->getNode(node_id);
    if (!node) return Result::fail("node id not found: " + std::to_string(node_id));

    for (size_t i = 0; i < node->inputs.size(); ++i) {
        const NodeSystem::Pin& pin = node->inputs[i];
        NodeParamInfo info;
        info.index = static_cast<int>(i);
        info.name = pin.name;
        info.data_type = nodeDataTypeName(pin.dataType);
        info.connected = graph->getInputSource(pin.id) != nullptr;
        info.value = pinValueToParam(pin.defaultValue);
        out.push_back(std::move(info));
    }
    return Result::success();
}

Result getNodeParam(const std::string& graph_type, const std::string& graph_name,
                    unsigned int node_id, int pin_index, NodeParamValue& out) {
    if (!g_ctx) return notBound();
    Result err;
    NodeSystem::GraphBase* graph = findNodeGraph(*g_ctx, graph_type, graph_name, err);
    if (!graph) return err;
    NodeSystem::NodeBase* node = graph->getNode(node_id);
    if (!node) return Result::fail("node id not found: " + std::to_string(node_id));
    if (pin_index < 0 || pin_index >= static_cast<int>(node->inputs.size()))
        return Result::fail("pin_index out of range (node has " +
                            std::to_string(node->inputs.size()) + " inputs)");
    out = pinValueToParam(node->inputs[pin_index].defaultValue);
    return Result::success();
}

Result setNodeParam(const std::string& graph_type, const std::string& graph_name,
                    unsigned int node_id, int pin_index, const NodeParamValue& value) {
    if (!g_ctx) return notBound();
    Result err;
    NodeSystem::GraphBase* graph = findNodeGraph(*g_ctx, graph_type, graph_name, err);
    if (!graph) return err;
    NodeSystem::NodeBase* node = graph->getNode(node_id);
    if (!node) return Result::fail("node id not found: " + std::to_string(node_id));
    if (pin_index < 0 || pin_index >= static_cast<int>(node->inputs.size()))
        return Result::fail("pin_index out of range (node has " +
                            std::to_string(node->inputs.size()) + " inputs)");
    Result r = setPinDefault(node->inputs[pin_index], value);
    if (!r) return r;
    // Mark this node and everything downstream dirty so the next evaluation picks
    // up the new default (graph-construction scope, same as Faz 3d — the editor's
    // Live path or a future rt.nodes.evaluate performs the actual apply).
    node->dirty = true;
    graph->markDirtyDownstream(node_id);
    return Result::success();
}

namespace {

bool isStructuralNodeProperty(const std::string& name) {
    return name == "id" || name == "typeId" || name == "name" || name == "position" ||
           name == "size" || name == "inputs" || name == "outputs" || name == "metadata";
}

// `filter_structural` must be set ONLY for serializers that emit the whole node
// (terrain). Material and geometry nodes serialize just their parameter block,
// where a top-level key like "position" is a genuine parameter — several nodes
// use exactly that name, and filtering it there would silently hide it from
// listing and reading while writes still worked.
void collectNodeProperties(const nlohmann::json& value, const std::string& prefix,
                           std::vector<NodePropertyInfo>& out, bool filter_structural) {
    if (value.is_object()) {
        for (auto it = value.begin(); it != value.end(); ++it) {
            if (filter_structural && prefix.empty() && isStructuralNodeProperty(it.key())) continue;
            const std::string path = prefix.empty() ? it.key() : prefix + "." + it.key();
            collectNodeProperties(it.value(), path, out, filter_structural);
        }
        return;
    }
    if (prefix.empty() || value.is_array() || value.is_null()) return;
    NodePropertyInfo info;
    info.name = prefix;
    if (value.is_boolean()) {
        info.data_type = "bool";
        info.value.kind = NodeParamValue::Kind::Bool;
        info.value.bool_value = value.get<bool>();
    } else if (value.is_number_integer() || value.is_number_unsigned()) {
        info.data_type = "int";
        info.value.kind = NodeParamValue::Kind::Int;
        info.value.int_value = value.get<int>();
    } else if (value.is_number_float()) {
        info.data_type = "float";
        info.value.kind = NodeParamValue::Kind::Float;
        info.value.floats[0] = value.get<float>();
    } else if (value.is_string()) {
        info.data_type = "string";
        info.value.kind = NodeParamValue::Kind::String;
        info.value.string_value = value.get<std::string>();
    } else {
        return;
    }
    out.push_back(std::move(info));
}

nlohmann::json* findNodePropertyJson(nlohmann::json& root, const std::string& path) {
    if (path.empty()) return nullptr;
    nlohmann::json* current = &root;
    size_t start = 0;
    while (start < path.size()) {
        const size_t dot = path.find('.', start);
        const std::string key = path.substr(start, dot == std::string::npos ? std::string::npos : dot - start);
        if (!current->is_object() || !current->contains(key)) return nullptr;
        current = &(*current)[key];
        if (dot == std::string::npos) break;
        start = dot + 1;
    }
    return current;
}

// Every node family exposes a JSON persistence hook, but under two different
// names: terrain nodes serialize the WHOLE node (serializeToJson — structural
// keys are filtered out by collectNodeProperties), while material and geometry
// nodes serialize only their parameter block (serializeParams). Dispatching on
// both is what turns "terrain nodes only" into "every node that can be saved".
enum class NodeSerializerKind { None, Terrain, MaterialParams, GeometryParams };

NodeSerializerKind nodeSerializerKind(NodeSystem::NodeBase* node) {
    if (dynamic_cast<TerrainNodesV2::TerrainNodeBase*>(node))    return NodeSerializerKind::Terrain;
    if (dynamic_cast<MaterialNodesV2::MaterialNodeBase*>(node))  return NodeSerializerKind::MaterialParams;
    if (dynamic_cast<GeometryNodesV2::GeometryNodeBase*>(node))  return NodeSerializerKind::GeometryParams;
    return NodeSerializerKind::None;
}

const char* kUnsupportedNodeSerializer =
    "node family does not expose a serialized property block";

Result serializedNode(UIContext& ctx, const std::string& graph_type, const std::string& graph_name,
                      unsigned int node_id, NodeSystem::GraphBase*& graph,
                      NodeSystem::NodeBase*& node, nlohmann::json& serialized,
                      NodeSerializerKind& out_kind) {
    Result err;
    graph = findNodeGraph(ctx, graph_type, graph_name, err);
    if (!graph) return err;
    node = graph->getNode(node_id);
    if (!node) return Result::fail("node not found: " + std::to_string(node_id));

    out_kind = nodeSerializerKind(node);
    switch (out_kind) {
        case NodeSerializerKind::Terrain:
            static_cast<TerrainNodesV2::TerrainNodeBase*>(node)->serializeToJson(serialized);
            return Result::success();
        case NodeSerializerKind::MaterialParams:
            static_cast<MaterialNodesV2::MaterialNodeBase*>(node)->serializeParams(serialized);
            return Result::success();
        case NodeSerializerKind::GeometryParams:
            static_cast<GeometryNodesV2::GeometryNodeBase*>(node)->serializeParams(serialized);
            return Result::success();
        default:
            return Result::fail(kUnsupportedNodeSerializer);
    }
}

// A node whose serializer writes nothing (the base-class default) is not an
// error — it simply has no scriptable properties. Distinguish that from a
// family we cannot serialize at all, which is a real failure.
Result applySerializedNode(NodeSystem::NodeBase* node, NodeSerializerKind kind,
                           const nlohmann::json& serialized) {
    switch (kind) {
        case NodeSerializerKind::Terrain:
            static_cast<TerrainNodesV2::TerrainNodeBase*>(node)->deserializeFromJson(serialized);
            return Result::success();
        case NodeSerializerKind::MaterialParams:
            static_cast<MaterialNodesV2::MaterialNodeBase*>(node)->deserializeParams(serialized);
            return Result::success();
        case NodeSerializerKind::GeometryParams:
            static_cast<GeometryNodesV2::GeometryNodeBase*>(node)->deserializeParams(serialized);
            return Result::success();
        default:
            return Result::fail(kUnsupportedNodeSerializer);
    }
}

} // namespace

Result listNodeProperties(const std::string& graph_type, const std::string& graph_name,
                          unsigned int node_id, std::vector<NodePropertyInfo>& out) {
    out.clear();
    if (!g_ctx) return notBound();
    NodeSystem::GraphBase* graph = nullptr;
    NodeSystem::NodeBase* node = nullptr;
    nlohmann::json serialized;
    NodeSerializerKind kind = NodeSerializerKind::None;
    Result r = serializedNode(*g_ctx, graph_type, graph_name, node_id, graph, node, serialized, kind);
    if (!r.ok) return r;
    collectNodeProperties(serialized, "", out, kind == NodeSerializerKind::Terrain);
    std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) { return a.name < b.name; });
    return Result::success();
}

Result getNodeProperty(const std::string& graph_type, const std::string& graph_name,
                       unsigned int node_id, const std::string& property, NodeParamValue& out) {
    std::vector<NodePropertyInfo> properties;
    Result r = listNodeProperties(graph_type, graph_name, node_id, properties);
    if (!r.ok) return r;
    for (const auto& item : properties) {
        if (item.name == property) { out = item.value; return Result::success(); }
    }
    return Result::fail("node property not found: " + property);
}

Result setNodeProperty(const std::string& graph_type, const std::string& graph_name,
                       unsigned int node_id, const std::string& property,
                       const NodeParamValue& value) {
    if (!g_ctx) return notBound();
    NodeSystem::GraphBase* graph = nullptr;
    NodeSystem::NodeBase* node = nullptr;
    nlohmann::json serialized;
    NodeSerializerKind kind = NodeSerializerKind::None;
    Result r = serializedNode(*g_ctx, graph_type, graph_name, node_id, graph, node, serialized, kind);
    if (!r.ok) return r;
    nlohmann::json* target = findNodePropertyJson(serialized, property);
    if (!target || target->is_array() || target->is_object() || target->is_null())
        return Result::fail("node property not found or not scalar: " + property);
    if (target->is_boolean() && value.kind == NodeParamValue::Kind::Bool) *target = value.bool_value;
    else if ((target->is_number_integer() || target->is_number_unsigned()) && value.kind == NodeParamValue::Kind::Int) *target = value.int_value;
    else if (target->is_number() && value.kind == NodeParamValue::Kind::Float) *target = value.floats[0];
    else if (target->is_string() && value.kind == NodeParamValue::Kind::String) *target = value.string_value;
    else return Result::fail("node property type mismatch: " + property);

    if (Result applied = applySerializedNode(node, kind, serialized); !applied) return applied;
    graph->markDirtyDownstream(node_id);
    return Result::success();
}


} // namespace rtapi
