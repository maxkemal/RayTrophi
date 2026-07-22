/*
* =========================================================================
* Project:       RayTrophi Studio
* File:          Api/RtPython.cpp
* Date:          July 2026
* License:       MIT
* =========================================================================
*/
#include "Api/RtPython.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "json.hpp"

#ifdef _WIN32
#include <cstdlib>
#endif

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "imgui.h"      // Faz 4b: rt.ui immediate-mode widgets + addon panels
#include "Api/RtApi.h"

namespace py = pybind11;

namespace {

std::unique_ptr<py::scoped_interpreter> g_interpreter;
std::mutex g_output_mutex;
std::string g_output;

void appendConsoleOutput(const std::string& text) {
    if (text.empty()) return;
    std::lock_guard<std::mutex> lock(g_output_mutex);
    g_output += text;
    constexpr std::size_t kMaxConsoleBytes = 4u * 1024u * 1024u;
    if (g_output.size() > kMaxConsoleBytes) {
        const std::size_t trim = g_output.size() - kMaxConsoleBytes;
        const std::size_t line = g_output.find('\n', trim);
        g_output.erase(0, line == std::string::npos ? trim : line + 1);
    }
}

void appendConsoleLine(const std::string& text) {
    appendConsoleOutput(text);
    if (text.empty() || text.back() != '\n') appendConsoleOutput("\n");
}

void requireResult(const rtapi::Result& result) {
    if (!result.ok) throw std::runtime_error(result.error);
}

// rt.ui panels (Faz 4b). Defined here — before PYBIND11_EMBEDDED_MODULE — so the
// module's rt.ui bindings can see them (module code is global scope; an anonymous
// namespace at file scope is visible there, an inner one under `namespace rtpython`
// further down is not). Each panel holds a Python draw callback invoked once per
// frame by drawAddonPanels(); g_addon_ui_drawing gates the immediate-mode widget
// calls so rt.ui.button()/text()/... only run while a panel is being drawn.
struct AddonPanel {
    std::string title;
    py::function draw;
};
std::map<int, AddonPanel> g_addon_panels;
int  g_next_panel_id = 1;
bool g_addon_ui_drawing = false;

void requireAddonUiContext() {
    if (!g_addon_ui_drawing)
        throw std::runtime_error("rt.ui.* widgets are only valid inside a panel draw callback");
}

py::list matrixToPython(const Matrix4x4& matrix) {
    py::list rows;
    for (int row = 0; row < 4; ++row) {
        py::list values;
        for (int col = 0; col < 4; ++col) values.append(matrix.m[row][col]);
        rows.append(std::move(values));
    }
    return rows;
}

Matrix4x4 matrixFromPython(const py::handle& value) {
    py::sequence rows = py::reinterpret_borrow<py::sequence>(value);
    if (py::len(rows) != 4) throw py::value_error("transform must contain four rows");

    Matrix4x4 matrix = Matrix4x4::identity();
    for (int row = 0; row < 4; ++row) {
        py::sequence columns = py::reinterpret_borrow<py::sequence>(rows[row]);
        if (py::len(columns) != 4) throw py::value_error("each transform row must contain four values");
        for (int col = 0; col < 4; ++col) matrix.m[row][col] = py::cast<float>(columns[col]);
    }
    return matrix;
}

py::tuple vec3ToPython(const Vec3& value) {
    return py::make_tuple(value.x, value.y, value.z);
}

Vec3 vec3FromPython(const py::handle& value) {
    py::sequence values = py::reinterpret_borrow<py::sequence>(value);
    if (py::len(values) != 3) throw py::value_error("position must contain three values");
    return Vec3(py::cast<float>(values[0]), py::cast<float>(values[1]), py::cast<float>(values[2]));
}

struct TransformProxy {
    py::list get(const std::string& name) const {
        Matrix4x4 matrix;
        requireResult(rtapi::getObjectTransform(name, matrix));
        return matrixToPython(matrix);
    }

    void set(const std::string& name, const py::handle& value) const {
        requireResult(rtapi::setObjectTransform(name, matrixFromPython(value)));
    }
};

// Wraps engine-owned memory in a zero-copy NumPy view. The capsule's no-op
// deleter exists only so pybind11 treats `view.data` as borrowed rather than
// copying it; the array is valid only for the current script call — see
// rtapi::MeshBufferView's lifetime note in RtApi.h.
py::array_t<float> meshBufferToArray(const rtapi::MeshBufferView& view) {
    const auto rows = static_cast<py::ssize_t>(view.vertex_count);
    const auto cols = static_cast<py::ssize_t>(view.components);
    if (!view.data || rows == 0) return py::array_t<float>(std::vector<py::ssize_t>{rows, cols});
    py::capsule owner(view.data, [](void*) {});
    return py::array_t<float>({ rows, cols },
                               { static_cast<py::ssize_t>(cols * sizeof(float)), static_cast<py::ssize_t>(sizeof(float)) },
                               view.data, owner);
}

using MeshWriteArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

void requireMeshShape(const MeshWriteArray& array, int expected_components, const char* what) {
    if (array.ndim() != 2 || array.shape(1) != expected_components) {
        throw py::value_error(std::string(what) + " array must have shape (N, " +
                               std::to_string(expected_components) + ")");
    }
}

struct ConsoleStream {
    std::size_t write(const std::string& text) {
        appendConsoleOutput(text);
        return text.size();
    }
    void flush() {}
    bool isatty() const { return false; }
};

// ── Node parameter marshaling (Faz 5.1b) ────────────────────────────────────
// NodeParamValue <-> native Python: scalars round-trip as float/int/bool/str,
// vectors as tuples. See rtapi::NodeParamValue in RtApi.h.
py::object nodeParamToPython(const rtapi::NodeParamValue& v) {
    using K = rtapi::NodeParamValue::Kind;
    switch (v.kind) {
        case K::Float:   return py::float_(v.floats[0]);
        case K::Int:     return py::int_(v.int_value);
        case K::Bool:    return py::bool_(v.bool_value);
        case K::Vector2: return py::make_tuple(v.floats[0], v.floats[1]);
        case K::Vector3: return py::make_tuple(v.floats[0], v.floats[1], v.floats[2]);
        case K::Vector4: return py::make_tuple(v.floats[0], v.floats[1], v.floats[2], v.floats[3]);
        case K::String:  return py::str(v.string_value);
        case K::None:    default: return py::none();
    }
}

rtapi::NodeParamValue nodeParamFromPython(const py::handle& value) {
    using K = rtapi::NodeParamValue::Kind;
    rtapi::NodeParamValue out;
    // bool must be checked before int — Python bool is a subclass of int.
    if (py::isinstance<py::bool_>(value)) {
        out.kind = K::Bool; out.bool_value = py::cast<bool>(value);
    } else if (py::isinstance<py::int_>(value)) {
        out.kind = K::Int; out.int_value = py::cast<int>(value); out.floats[0] = static_cast<float>(out.int_value);
    } else if (py::isinstance<py::float_>(value)) {
        out.kind = K::Float; out.floats[0] = py::cast<float>(value);
    } else if (py::isinstance<py::str>(value)) {
        out.kind = K::String; out.string_value = py::cast<std::string>(value);
    } else if (py::isinstance<py::tuple>(value) || py::isinstance<py::list>(value)) {
        auto seq = py::cast<std::vector<float>>(value);
        if (seq.size() < 2 || seq.size() > 4)
            throw py::value_error("vector parameter must have 2, 3 or 4 components");
        out.kind = (seq.size() == 2) ? K::Vector2 : (seq.size() == 3) ? K::Vector3 : K::Vector4;
        for (size_t i = 0; i < seq.size(); ++i) out.floats[i] = seq[i];
    } else {
        throw py::type_error("unsupported node parameter value (expected float/int/bool/str/tuple)");
    }
    return out;
}

} // namespace

PYBIND11_EMBEDDED_MODULE(rt, module) {
    module.doc() = "RayTrophi Studio embedded scripting API";

    module.def("version", [] {
        const rtapi::Version v = rtapi::version();
        return std::to_string(v.major) + "." + std::to_string(v.minor) + "." + std::to_string(v.patch);
    });

    py::class_<ConsoleStream>(module, "_ConsoleStream")
        .def(py::init<>())
        .def("write", &ConsoleStream::write)
        .def("flush", &ConsoleStream::flush)
        .def("isatty", &ConsoleStream::isatty);

    py::class_<TransformProxy>(module, "_TransformProxy")
        .def(py::init<>())
        .def("__getitem__", &TransformProxy::get)
        .def("__setitem__", &TransformProxy::set);

    py::module_ scene = module.def_submodule("scene", "Scene queries and undoable mutations");
    scene.def("objects", [] {
        py::list result;
        for (const std::string& name : rtapi::listObjects()) {
            rtapi::ObjectInfo info;
            requireResult(rtapi::getObjectInfo(name, info));
            py::dict item;
            item["name"] = info.name;
            item["triangles"] = info.triangle_count;
            item["vertices"] = info.vertex_count;
            result.append(std::move(item));
        }
        return result;
    });
    scene.def("exists", &rtapi::objectExists, py::arg("name"));
    scene.def("delete", [](const std::string& name) { requireResult(rtapi::deleteObject(name)); });
    scene.def("duplicate", [](const std::string& name) {
        std::string newName;
        requireResult(rtapi::duplicateObject(name, newName));
        return newName;
    });
    scene.def("import_model", [](const std::string& path) { requireResult(rtapi::importModel(path)); });
    scene.def("add_primitive", [](const std::string& type, const std::string& name, float size) {
        std::string newName;
        requireResult(rtapi::addPrimitive(type, name, size, newName));
        return newName;
    }, py::arg("type"), py::arg("name") = std::string(), py::arg("size") = 1.0f);
    scene.def("get_transform", [](const std::string& name) -> py::dict {
        Matrix4x4 m;
        requireResult(rtapi::getObjectTransform(name, m));
        Vec3 t = m.getTranslation();
        py::dict d;
        d["matrix"] = matrixToPython(m);
        d["translation"] = vec3ToPython(t);
        return d;
    }, py::arg("name"));

    scene.def("set_transform", [](const std::string& name, const py::handle& translation, const py::handle& rotation, const py::handle& scale, const py::handle& matrix) {
        if (!matrix.is_none()) {
            requireResult(rtapi::setObjectTransform(name, matrixFromPython(matrix)));
            return;
        }
        Matrix4x4 m = Matrix4x4::identity();
        requireResult(rtapi::getObjectTransform(name, m));
        if (!translation.is_none()) {
            Vec3 t = vec3FromPython(translation);
            m.m[0][3] = t.x; m.m[1][3] = t.y; m.m[2][3] = t.z;
        }
        requireResult(rtapi::setObjectTransform(name, m));
    }, py::arg("name"), py::arg("translation") = py::none(), py::arg("rotation") = py::none(), py::arg("scale") = py::none(), py::arg("matrix") = py::none());

    scene.attr("transform") = TransformProxy{};

    py::module_ mesh = module.def_submodule("mesh",
        "Flat SoA mesh vertex data (local/bind space, zero-copy reads)");
    mesh.def("positions", [](const std::string& name) {
        rtapi::MeshBufferView view;
        requireResult(rtapi::getMeshPositions(name, view));
        return meshBufferToArray(view);
    }, py::arg("name"));
    mesh.def("normals", [](const std::string& name) {
        rtapi::MeshBufferView view;
        requireResult(rtapi::getMeshNormals(name, view));
        return meshBufferToArray(view);
    }, py::arg("name"));
    mesh.def("uvs", [](const std::string& name) {
        rtapi::MeshBufferView view;
        requireResult(rtapi::getMeshUVs(name, view));
        return meshBufferToArray(view);
    }, py::arg("name"));
    mesh.def("set_positions", [](const std::string& name, const MeshWriteArray& positions) {
        requireMeshShape(positions, 3, "positions");
        requireResult(rtapi::setMeshPositions(name, positions.data(), static_cast<size_t>(positions.shape(0))));
    }, py::arg("name"), py::arg("positions"));
    mesh.def("set_normals", [](const std::string& name, const MeshWriteArray& normals) {
        requireMeshShape(normals, 3, "normals");
        requireResult(rtapi::setMeshNormals(name, normals.data(), static_cast<size_t>(normals.shape(0))));
    }, py::arg("name"), py::arg("normals"));
    mesh.def("set_uvs", [](const std::string& name, const MeshWriteArray& uvs) {
        requireMeshShape(uvs, 2, "uvs");
        requireResult(rtapi::setMeshUVs(name, uvs.data(), static_cast<size_t>(uvs.shape(0))));
    }, py::arg("name"), py::arg("uvs"));
    mesh.def("recompute_normals", [](const std::string& name) {
        requireResult(rtapi::recomputeMeshNormals(name));
    }, py::arg("name"));

    // ── Mesh Modifiers (Faz 5.2b) ───────────────────────────────────────
    py::module_ modifiers = module.def_submodule("modifiers", "Mesh modifier stack: subdivision, simple, smooth");
    modifiers.def("get_stack", [](const std::string& object_name) -> py::list {
        std::vector<rtapi::ModifierInfo> stack;
        requireResult(rtapi::getModifierStack(object_name, stack));
        py::list result;
        for (const auto& mod : stack) {
            py::dict d;
            d["index"] = mod.index;
            d["name"] = mod.name;
            d["type"] = mod.type;
            d["enabled"] = mod.enabled;
            d["levels"] = mod.levels;
            d["render_levels"] = mod.render_levels;
            d["smooth_angle"] = mod.smooth_angle;
            result.append(d);
        }
        return result;
    }, py::arg("object_name"));

    modifiers.def("add", [](const std::string& object_name, const std::string& type, const std::string& name,
                            int levels, int render_levels) -> py::dict {
        rtapi::ModifierInfo mod;
        requireResult(rtapi::addModifier(object_name, type, name, levels, render_levels, mod));
        py::dict d;
        d["index"] = mod.index;
        d["name"] = mod.name;
        d["type"] = mod.type;
        d["enabled"] = mod.enabled;
        d["levels"] = mod.levels;
        d["render_levels"] = mod.render_levels;
        d["smooth_angle"] = mod.smooth_angle;
        return d;
    }, py::arg("object_name"), py::arg("type") = "catmull_clark", py::arg("name") = "",
       py::arg("levels") = 1, py::arg("render_levels") = 2);

    modifiers.def("remove", [](const std::string& object_name, int index) {
        requireResult(rtapi::removeModifier(object_name, index));
    }, py::arg("object_name"), py::arg("index") = 0);

    modifiers.def("set_param", [](const std::string& object_name, int index, const py::kwargs& kwargs) {
        std::string name_val;
        const std::string* p_name = nullptr;
        if (kwargs.contains("name")) { name_val = py::cast<std::string>(kwargs["name"]); p_name = &name_val; }

        bool enabled_val = false;
        const bool* p_enabled = nullptr;
        if (kwargs.contains("enabled")) { enabled_val = py::cast<bool>(kwargs["enabled"]); p_enabled = &enabled_val; }

        int levels_val = 0;
        const int* p_levels = nullptr;
        if (kwargs.contains("levels")) { levels_val = py::cast<int>(kwargs["levels"]); p_levels = &levels_val; }

        int rlevels_val = 0;
        const int* p_rlevels = nullptr;
        if (kwargs.contains("render_levels")) { rlevels_val = py::cast<int>(kwargs["render_levels"]); p_rlevels = &rlevels_val; }

        float smooth_val = 0.0f;
        const float* p_smooth = nullptr;
        if (kwargs.contains("smooth_angle")) { smooth_val = py::cast<float>(kwargs["smooth_angle"]); p_smooth = &smooth_val; }

        requireResult(rtapi::updateModifier(object_name, index, p_name, p_enabled, p_levels, p_rlevels, p_smooth));
    }, py::arg("object_name"), py::arg("index") = 0);

    modifiers.def("apply", [](const std::string& object_name, int index) {
        requireResult(rtapi::applyModifier(object_name, index));
    }, py::arg("object_name"), py::arg("index") = 0);

    // ── Scatter & Foliage System (Faz 5.2c) ─────────────────────────────
    py::module_ scatter = module.def_submodule("scatter", "Scatter and foliage layers");
    scatter.def("list_groups", []() -> py::list {
        std::vector<rtapi::ScatterGroupInfo> groups;
        requireResult(rtapi::listScatterGroups(groups));
        py::list result;
        for (const auto& g : groups) {
            py::dict d;
            d["id"] = g.id;
            d["name"] = g.name;
            d["target_type"] = g.target_type;
            d["target_node_name"] = g.target_node_name;
            d["instance_count"] = g.instance_count;
            d["triangle_count"] = g.triangle_count;

            py::list sources;
            for (const auto& s : g.sources) {
                py::dict sd;
                sd["name"] = s.name;
                sd["weight"] = s.weight;
                sd["scale_min"] = s.scale_min;
                sd["scale_max"] = s.scale_max;
                sd["rotation_random_y"] = s.rotation_random_y;
                sd["align_to_normal"] = s.align_to_normal;
                sources.append(sd);
            }
            d["sources"] = sources;
            result.append(d);
        }
        return result;
    });

    scatter.def("create_group", [](const std::string& name, const std::string& target_node, const std::string& target_type) -> py::dict {
        rtapi::ScatterGroupInfo info;
        requireResult(rtapi::createScatterGroup(name, target_node, target_type, info));
        py::dict d;
        d["id"] = info.id;
        d["name"] = info.name;
        d["target_type"] = info.target_type;
        d["target_node"] = info.target_node_name;
        d["target_node_name"] = info.target_node_name;
        d["instance_count"] = info.instance_count;
        d["triangle_count"] = info.triangle_count;
        return d;
    }, py::arg("name"), py::arg("target_node") = "", py::arg("target_type") = "mesh");

    scatter.def("delete_group", [](const std::string& group) {
        requireResult(rtapi::deleteScatterGroup(group));
    }, py::arg("group"));

    scatter.def("clear", [](const std::string& group) {
        requireResult(rtapi::clearScatterGroup(group));
    }, py::arg("group"));

    scatter.def("add_source", [](const std::string& group, const std::string& mesh, float weight,
                                float scale_min, float scale_max, float rotation_y, bool align_to_normal) {
        requireResult(rtapi::addScatterSource(group, mesh, weight, scale_min, scale_max, rotation_y, align_to_normal));
    }, py::arg("group"), py::arg("mesh"), py::arg("weight") = 1.0f,
       py::arg("scale_min") = 0.8f, py::arg("scale_max") = 1.2f,
       py::arg("rotation_y") = 360.0f, py::arg("align_to_normal") = true);

    scatter.def("remove_source", [](const std::string& group, int source_index) {
        requireResult(rtapi::removeScatterSource(group, source_index));
    }, py::arg("group"), py::arg("source_index") = 0);

    scatter.def("set_settings", [](const std::string& group, const py::kwargs& kwargs) {
        int tcount_val = 0; const int* p_tcount = nullptr;
        if (kwargs.contains("target_count")) { tcount_val = py::cast<int>(kwargs["target_count"]); p_tcount = &tcount_val; }

        int seed_val = 0; const int* p_seed = nullptr;
        if (kwargs.contains("seed")) { seed_val = py::cast<int>(kwargs["seed"]); p_seed = &seed_val; }

        float min_dist_val = 0.0f; const float* p_min_dist = nullptr;
        if (kwargs.contains("min_distance")) { min_dist_val = py::cast<float>(kwargs["min_distance"]); p_min_dist = &min_dist_val; }

        float slope_max_val = 0.0f; const float* p_slope_max = nullptr;
        if (kwargs.contains("slope_max")) { slope_max_val = py::cast<float>(kwargs["slope_max"]); p_slope_max = &slope_max_val; }

        float hmin_val = 0.0f; const float* p_hmin = nullptr;
        if (kwargs.contains("height_min")) { hmin_val = py::cast<float>(kwargs["height_min"]); p_hmin = &hmin_val; }

        float hmax_val = 0.0f; const float* p_hmax = nullptr;
        if (kwargs.contains("height_max")) { hmax_val = py::cast<float>(kwargs["height_max"]); p_hmax = &hmax_val; }

        std::string dmask_val; const std::string* p_dmask = nullptr;
        if (kwargs.contains("density_mask")) { dmask_val = py::cast<std::string>(kwargs["density_mask"]); p_dmask = &dmask_val; }

        std::string smask_val; const std::string* p_smask = nullptr;
        if (kwargs.contains("scale_mask")) { smask_val = py::cast<std::string>(kwargs["scale_mask"]); p_smask = &smask_val; }

        requireResult(rtapi::setScatterGroupSettings(group, p_tcount, p_seed, p_min_dist, p_slope_max, p_hmin, p_hmax, p_dmask, p_smask));
    }, py::arg("group"));

    scatter.def("fill", [](const std::string& group) -> int {
        int spawned = 0;
        requireResult(rtapi::fillScatterGroup(group, spawned));
        return spawned;
    }, py::arg("group"));

    scatter.def("add_instance", [](const std::string& group, const py::tuple& position, const py::tuple& rotation, const py::tuple& scale, int source_index) {
        Vec3 pos = vec3FromPython(position);
        Vec3 rot = vec3FromPython(rotation);
        Vec3 scl = vec3FromPython(scale);
        requireResult(rtapi::addScatterInstance(group, pos, rot, scl, source_index));
    }, py::arg("group"), py::arg("position") = py::make_tuple(0,0,0), py::arg("rotation") = py::make_tuple(0,0,0), py::arg("scale") = py::make_tuple(1,1,1), py::arg("source_index") = 0);

    // ── Physics Engine (Faz 5.3a) ────────────────────────────────────────
    py::module_ physics = module.def_submodule("physics", "Rigid body, soft body, and cloth physics");
    physics.def("get", [](const std::string& object_name) -> py::dict {
        rtapi::PhysicsBodyInfo info;
        requireResult(rtapi::getPhysicsBody(object_name, info));
        py::dict d;
        d["object_name"] = info.object_name;
        d["kind"] = info.kind;
        d["motion_type"] = info.motion_type;
        d["shape"] = info.shape;
        d["enabled"] = info.enabled;
        d["mass"] = info.mass;
        d["friction"] = info.friction;
        d["restitution"] = info.restitution;
        d["linear_damping"] = info.linear_damping;
        d["angular_damping"] = info.angular_damping;
        d["gravity_scale"] = info.gravity_scale;
        d["soft_stiffness"] = info.soft_stiffness;
        d["soft_pressure"] = info.soft_pressure;
        d["soft_damping"] = info.soft_damping;
        return d;
    }, py::arg("object_name"));

    physics.def("add_body", [](const std::string& object_name, const std::string& kind, const std::string& motion_type,
                               const std::string& shape, float mass) -> py::dict {
        rtapi::PhysicsBodyInfo info;
        requireResult(rtapi::addPhysicsBody(object_name, kind, motion_type, shape, mass, info));
        py::dict d;
        d["object_name"] = info.object_name;
        d["kind"] = info.kind;
        d["motion_type"] = info.motion_type;
        d["shape"] = info.shape;
        d["enabled"] = info.enabled;
        d["mass"] = info.mass;
        d["friction"] = info.friction;
        d["restitution"] = info.restitution;
        d["linear_damping"] = info.linear_damping;
        d["angular_damping"] = info.angular_damping;
        d["gravity_scale"] = info.gravity_scale;
        return d;
    }, py::arg("object_name"), py::arg("kind") = "rigid", py::arg("motion_type") = "dynamic",
       py::arg("shape") = "box", py::arg("mass") = 1.0f);

    physics.def("remove_body", [](const std::string& object_name) {
        requireResult(rtapi::removePhysicsBody(object_name));
    }, py::arg("object_name"));

    physics.def("set_param", [](const std::string& object_name, const py::kwargs& kwargs) {
        std::string kind_val; const std::string* p_kind = nullptr;
        if (kwargs.contains("kind")) { kind_val = py::cast<std::string>(kwargs["kind"]); p_kind = &kind_val; }

        std::string motion_val; const std::string* p_motion = nullptr;
        if (kwargs.contains("motion_type")) { motion_val = py::cast<std::string>(kwargs["motion_type"]); p_motion = &motion_val; }

        std::string shape_val; const std::string* p_shape = nullptr;
        if (kwargs.contains("shape")) { shape_val = py::cast<std::string>(kwargs["shape"]); p_shape = &shape_val; }

        bool enabled_val = false; const bool* p_enabled = nullptr;
        if (kwargs.contains("enabled")) { enabled_val = py::cast<bool>(kwargs["enabled"]); p_enabled = &enabled_val; }

        float mass_val = 0.0f; const float* p_mass = nullptr;
        if (kwargs.contains("mass")) { mass_val = py::cast<float>(kwargs["mass"]); p_mass = &mass_val; }

        float fric_val = 0.0f; const float* p_fric = nullptr;
        if (kwargs.contains("friction")) { fric_val = py::cast<float>(kwargs["friction"]); p_fric = &fric_val; }

        float rest_val = 0.0f; const float* p_rest = nullptr;
        if (kwargs.contains("restitution")) { rest_val = py::cast<float>(kwargs["restitution"]); p_rest = &rest_val; }

        float ldamp_val = 0.0f; const float* p_ldamp = nullptr;
        if (kwargs.contains("linear_damping")) { ldamp_val = py::cast<float>(kwargs["linear_damping"]); p_ldamp = &ldamp_val; }

        float adamp_val = 0.0f; const float* p_adamp = nullptr;
        if (kwargs.contains("angular_damping")) { adamp_val = py::cast<float>(kwargs["angular_damping"]); p_adamp = &adamp_val; }

        float gscale_val = 0.0f; const float* p_gscale = nullptr;
        if (kwargs.contains("gravity_scale")) { gscale_val = py::cast<float>(kwargs["gravity_scale"]); p_gscale = &gscale_val; }

        float sstiff_val = 0.0f; const float* p_sstiff = nullptr;
        if (kwargs.contains("soft_stiffness")) { sstiff_val = py::cast<float>(kwargs["soft_stiffness"]); p_sstiff = &sstiff_val; }

        float spress_val = 0.0f; const float* p_spress = nullptr;
        if (kwargs.contains("soft_pressure")) { spress_val = py::cast<float>(kwargs["soft_pressure"]); p_spress = &spress_val; }

        float sdamp_val = 0.0f; const float* p_sdamp = nullptr;
        if (kwargs.contains("soft_damping")) { sdamp_val = py::cast<float>(kwargs["soft_damping"]); p_sdamp = &sdamp_val; }

        requireResult(rtapi::updatePhysicsBody(object_name, p_kind, p_motion, p_shape, p_enabled,
                                                 p_mass, p_fric, p_rest, p_ldamp, p_adamp,
                                                 p_gscale, p_sstiff, p_spress, p_sdamp));
    }, py::arg("object_name"));

    physics.def("reset", []() {
        requireResult(rtapi::resetPhysicsSimulation());
    });

    physics.def("step", [](float dt) {
        requireResult(rtapi::stepPhysicsSimulation(dt));
    }, py::arg("dt") = 0.0166667f);

    physics.def("set_gravity", [](const py::handle& gravity) {
        requireResult(rtapi::setPhysicsGravity(vec3FromPython(gravity)));
    }, py::arg("gravity") = py::make_tuple(0.0f, -9.81f, 0.0f));

    physics.def("get_gravity", []() -> py::tuple {
        Vec3 g(0, -9.81f, 0);
        requireResult(rtapi::getPhysicsGravity(g));
        return vec3ToPython(g);
    });

    // ── Fluid & Gas Simulation Engine (Faz 5.3b) ────────────────────────
    py::module_ fluid = module.def_submodule("fluid", "APIC liquid & fluid domain simulation");
    auto py_create_domain = [](const std::string& name, const py::tuple& domain_min, const py::tuple& domain_max, float voxel_size, const std::string& type) -> py::dict {
        Vec3 dmin = vec3FromPython(domain_min);
        Vec3 dmax = vec3FromPython(domain_max);
        rtapi::FluidDomainInfo info;
        requireResult(rtapi::createFluidDomain(name, dmin, dmax, voxel_size, type, info));
        py::dict d;
        d["id"] = info.id;
        d["name"] = info.name;
        d["type"] = info.type;
        d["domain_min"] = vec3ToPython(info.domain_min);
        d["domain_max"] = vec3ToPython(info.domain_max);
        d["voxel_size"] = info.voxel_size;
        d["particle_count"] = info.particle_count;
        d["render_mode"] = info.render_mode;
        d["backend"] = info.backend;
        d["boundary"] = info.boundary;
        d["preset"] = info.preset;
        d["viscosity"] = info.viscosity;
        d["enabled"] = info.enabled;
        d["visible"] = info.visible;
        return d;
    };

    fluid.def("create_domain", py_create_domain, py::arg("name") = "Fluid", py::arg("domain_min") = py::make_tuple(-1.0f, 0.0f, -1.0f),
       py::arg("domain_max") = py::make_tuple(1.0f, 2.0f, 1.0f), py::arg("voxel_size") = 0.05f, py::arg("type") = "fluid");

    py::module_ gas = module.def_submodule("gas", "Smoke & gas grid domain simulation");
    gas.def("create_domain", [py_create_domain](const std::string& name, const py::tuple& domain_min, const py::tuple& domain_max, float voxel_size) {
        return py_create_domain(name, domain_min, domain_max, voxel_size, "gas");
    }, py::arg("name") = "GasDomain", py::arg("domain_min") = py::make_tuple(-1.0f, 0.0f, -1.0f),
       py::arg("domain_max") = py::make_tuple(1.0f, 2.0f, 1.0f), py::arg("voxel_size") = 0.05f);

    fluid.def("remove_domain", [](const std::string& domain) {
        requireResult(rtapi::removeFluidDomain(domain));
    }, py::arg("domain"));

    fluid.def("get", [](const std::string& domain) -> py::dict {
        rtapi::FluidDomainInfo info;
        requireResult(rtapi::getFluidDomain(domain, info));
        py::dict d;
        d["id"] = info.id;
        d["name"] = info.name;
        d["domain_min"] = vec3ToPython(info.domain_min);
        d["domain_max"] = vec3ToPython(info.domain_max);
        d["voxel_size"] = info.voxel_size;
        d["particle_count"] = info.particle_count;
        d["render_mode"] = info.render_mode;
        d["backend"] = info.backend;
        d["boundary"] = info.boundary;
        d["preset"] = info.preset;
        d["viscosity"] = info.viscosity;
        d["enabled"] = info.enabled;
        d["visible"] = info.visible;
        return d;
    }, py::arg("domain"));

    fluid.def("seed", [](const std::string& domain, const py::tuple& seed_min, const py::tuple& seed_max, int particles_per_cell, bool replace) {
        Vec3 smin = vec3FromPython(seed_min);
        Vec3 smax = vec3FromPython(seed_max);
        requireResult(rtapi::seedFluidParticles(domain, smin, smax, particles_per_cell, replace));
    }, py::arg("domain"), py::arg("seed_min") = py::make_tuple(-0.5f, 1.0f, -0.5f),
       py::arg("seed_max") = py::make_tuple(0.5f, 1.5f, 0.5f), py::arg("particles_per_cell") = 4, py::arg("replace") = true);

    fluid.def("clear", [](const std::string& domain) {
        requireResult(rtapi::clearFluidParticles(domain));
    }, py::arg("domain"));

    fluid.def("set_param", [](const std::string& domain, const py::kwargs& kwargs) {
        Vec3 dmin_val; const Vec3* p_dmin = nullptr;
        if (kwargs.contains("domain_min")) { dmin_val = vec3FromPython(kwargs["domain_min"]); p_dmin = &dmin_val; }

        Vec3 dmax_val; const Vec3* p_dmax = nullptr;
        if (kwargs.contains("domain_max")) { dmax_val = vec3FromPython(kwargs["domain_max"]); p_dmax = &dmax_val; }

        float vs_val = 0.0f; const float* p_vs = nullptr;
        if (kwargs.contains("voxel_size")) { vs_val = py::cast<float>(kwargs["voxel_size"]); p_vs = &vs_val; }

        std::string rm_val; const std::string* p_rm = nullptr;
        if (kwargs.contains("render_mode")) { rm_val = py::cast<std::string>(kwargs["render_mode"]); p_rm = &rm_val; }

        std::string dev_val; const std::string* p_dev = nullptr;
        if (kwargs.contains("backend") || kwargs.contains("device")) {
            dev_val = kwargs.contains("backend") ? py::cast<std::string>(kwargs["backend"]) : py::cast<std::string>(kwargs["device"]);
            p_dev = &dev_val;
        }

        std::string bound_val; const std::string* p_bound = nullptr;
        if (kwargs.contains("boundary")) { bound_val = py::cast<std::string>(kwargs["boundary"]); p_bound = &bound_val; }

        std::string preset_val; const std::string* p_preset = nullptr;
        if (kwargs.contains("preset")) { preset_val = py::cast<std::string>(kwargs["preset"]); p_preset = &preset_val; }

        float visc_val = 0.0f; const float* p_visc = nullptr;
        if (kwargs.contains("viscosity")) { visc_val = py::cast<float>(kwargs["viscosity"]); p_visc = &visc_val; }

        bool enabled_val = false; const bool* p_enabled = nullptr;
        if (kwargs.contains("enabled")) { enabled_val = py::cast<bool>(kwargs["enabled"]); p_enabled = &enabled_val; }

        bool visible_val = false; const bool* p_visible = nullptr;
        if (kwargs.contains("visible")) { visible_val = py::cast<bool>(kwargs["visible"]); p_visible = &visible_val; }

        requireResult(rtapi::updateFluidDomain(domain, p_dmin, p_dmax, p_vs, p_rm, p_dev, p_bound, p_preset, p_visc, p_enabled, p_visible));
    }, py::arg("domain"));

    fluid.def("reset", []() {
        requireResult(rtapi::resetFluidSimulation());
    });

    fluid.def("step", [](float dt) {
        requireResult(rtapi::stepFluidSimulation(dt));
    }, py::arg("dt") = 0.0166667f);

    py::module_ terrain = module.def_submodule("terrain", "Terrain creation, queries, and procedural operations");
    auto terrain_info_to_dict = [](const rtapi::TerrainInfo& info) -> py::dict {
        py::dict d;
        d["id"] = info.id;
        d["name"] = info.name;
        d["resolution"] = py::make_tuple(info.width, info.height);
        d["size"] = info.size;
        d["height_scale"] = info.height_scale;
        d["has_node_graph"] = info.has_node_graph;
        d["dirty"] = info.dirty;
        return d;
    };
    terrain.def("list", [terrain_info_to_dict]() -> py::list {
        std::vector<rtapi::TerrainInfo> terrains;
        requireResult(rtapi::listTerrains(terrains));
        py::list out;
        for (const auto& info : terrains) out.append(terrain_info_to_dict(info));
        return out;
    });
    terrain.def("get", [terrain_info_to_dict](const std::string& name) -> py::dict {
        rtapi::TerrainInfo info;
        requireResult(rtapi::getTerrain(name, info));
        return terrain_info_to_dict(info);
    }, py::arg("name"));
    terrain.def("create", [terrain_info_to_dict](const std::string& name, int resolution,
                                                   float size, float height_scale) -> py::dict {
        rtapi::TerrainInfo info;
        requireResult(rtapi::createTerrain(name, resolution, size, height_scale, info));
        return terrain_info_to_dict(info);
    }, py::arg("name") = "Terrain", py::arg("resolution") = 1024,
       py::arg("size") = 1000.0f, py::arg("height_scale") = 100.0f);
    terrain.def("import_heightmap", [terrain_info_to_dict](const std::string& filepath,
                         const std::string& name, float size, float height_scale,
                         int max_resolution) -> py::dict {
        rtapi::TerrainInfo info;
        requireResult(rtapi::importTerrainHeightmap(filepath, name, size, height_scale,
                                                     max_resolution, info));
        return terrain_info_to_dict(info);
    }, py::arg("filepath"), py::arg("name") = "TerrainImported",
       py::arg("size") = 1000.0f, py::arg("height_scale") = 100.0f,
       py::arg("max_resolution") = 2048);
    terrain.def("remove", [](const std::string& name) {
        requireResult(rtapi::removeTerrain(name));
    }, py::arg("name"));
    terrain.def("export_heightmap", [](const std::string& name, const std::string& filepath) {
        requireResult(rtapi::exportTerrainHeightmap(name, filepath));
    }, py::arg("name"), py::arg("filepath"));
    auto terrain_eval_to_dict = [](const rtapi::TerrainEvaluationInfo& info) -> py::dict {
        py::dict d;
        d["terrain"] = info.terrain_name;
        d["state"] = info.state;
        d["progress"] = info.progress;
        d["current_node_id"] = info.current_node_id;
        d["error"] = info.error;
        return d;
    };
    terrain.def("evaluate", [terrain_eval_to_dict](const std::string& name) -> py::dict {
        rtapi::TerrainEvaluationInfo info;
        requireResult(rtapi::evaluateTerrain(name, info));
        return terrain_eval_to_dict(info);
    }, py::arg("name"));
    terrain.def("evaluation_status", [terrain_eval_to_dict](const std::string& name) -> py::dict {
        rtapi::TerrainEvaluationInfo info;
        requireResult(rtapi::getTerrainEvaluationStatus(name, info));
        return terrain_eval_to_dict(info);
    }, py::arg("name"));
    terrain.def("cancel_evaluation", [](const std::string& name) {
        requireResult(rtapi::cancelTerrainEvaluation(name));
    }, py::arg("name"));
    terrain.def("erode", [](const std::string& name, const std::string& type,
                             const std::string& backend, const py::kwargs& kwargs) {
        rtapi::TerrainErosionSettings settings;
        settings.type = type;
        settings.backend = backend;
        if (kwargs.contains("iterations")) settings.iterations = py::cast<int>(kwargs["iterations"]);
        if (kwargs.contains("seed")) settings.seed = py::cast<unsigned int>(kwargs["seed"]);
        if (kwargs.contains("strength")) settings.strength = py::cast<float>(kwargs["strength"]);
        if (kwargs.contains("direction")) settings.direction = py::cast<float>(kwargs["direction"]);
        if (kwargs.contains("talus_angle")) settings.talus_angle = py::cast<float>(kwargs["talus_angle"]);
        if (kwargs.contains("amount")) settings.amount = py::cast<float>(kwargs["amount"]);
        if (kwargs.contains("undo")) settings.undo = py::cast<bool>(kwargs["undo"]);
        requireResult(rtapi::erodeTerrain(name, settings));
    }, py::arg("name"), py::arg("type") = "hydraulic", py::arg("backend") = "auto");
    terrain.def("apply_preset", [](const std::string& name, const std::string& preset,
                                    bool replace_graph) {
        requireResult(rtapi::applyTerrainPreset(name, preset, replace_graph));
    }, py::arg("name"), py::arg("preset"), py::arg("replace_graph") = false);
    terrain.def("calculate_flow", [](const std::string& name) {
        requireResult(rtapi::calculateTerrainFlow(name));
    }, py::arg("name"));
    terrain.def("sample_height", [](const std::string& name, float world_x, float world_z) {
        float height = 0.0f;
        requireResult(rtapi::sampleTerrainHeight(name, world_x, world_z, height));
        return height;
    }, py::arg("name"), py::arg("world_x"), py::arg("world_z"));
    terrain.def("carve_river", [](const std::string& name, const std::string& river,
                                   const py::kwargs& kwargs) {
        rtapi::TerrainRiverCarveSettings settings;
        if (kwargs.contains("mode")) settings.mode = py::cast<std::string>(kwargs["mode"]);
        if (kwargs.contains("depth_multiplier")) settings.depth_multiplier = py::cast<float>(kwargs["depth_multiplier"]);
        if (kwargs.contains("smoothness")) settings.smoothness = py::cast<float>(kwargs["smoothness"]);
        if (kwargs.contains("post_erosion")) settings.post_erosion = py::cast<bool>(kwargs["post_erosion"]);
        if (kwargs.contains("post_erosion_iterations")) settings.post_erosion_iterations = py::cast<int>(kwargs["post_erosion_iterations"]);
        if (kwargs.contains("noise_strength")) settings.noise_strength = py::cast<float>(kwargs["noise_strength"]);
        if (kwargs.contains("deep_pools")) settings.deep_pools = py::cast<bool>(kwargs["deep_pools"]);
        if (kwargs.contains("riffles")) settings.riffles = py::cast<bool>(kwargs["riffles"]);
        if (kwargs.contains("asymmetric_banks")) settings.asymmetric_banks = py::cast<bool>(kwargs["asymmetric_banks"]);
        if (kwargs.contains("point_bars")) settings.point_bars = py::cast<bool>(kwargs["point_bars"]);
        if (kwargs.contains("undo")) settings.undo = py::cast<bool>(kwargs["undo"]);
        requireResult(rtapi::carveTerrainRiver(name, river, settings));
    }, py::arg("name"), py::arg("river"));
    terrain.def("list_rivers", []() {
        std::vector<rtapi::TerrainRiverInfo> rivers;
        requireResult(rtapi::listTerrainRivers(rivers));
        py::list result;
        for (const auto& river : rivers) {
            py::dict item;
            item["id"] = river.id;
            item["name"] = river.name;
            item["control_points"] = river.control_point_count;
            item["follow_terrain"] = river.follow_terrain;
            result.append(std::move(item));
        }
        return result;
    });

    py::module_ hair = module.def_submodule("hair", "Deterministic hair groom generation and styling");
    auto hair_settings_to_dict = [](const rtapi::HairSettings& s) {
        py::dict d;
#define RT_HAIR_PY_FIELD(name) d[#name] = s.name
        RT_HAIR_PY_FIELD(guide_count); RT_HAIR_PY_FIELD(children_per_guide);
        RT_HAIR_PY_FIELD(points_per_strand); RT_HAIR_PY_FIELD(length);
        RT_HAIR_PY_FIELD(length_variation); RT_HAIR_PY_FIELD(root_radius);
        RT_HAIR_PY_FIELD(tip_radius); RT_HAIR_PY_FIELD(clumpiness);
        RT_HAIR_PY_FIELD(child_radius); RT_HAIR_PY_FIELD(curl_frequency);
        RT_HAIR_PY_FIELD(curl_radius); RT_HAIR_PY_FIELD(wave_frequency);
        RT_HAIR_PY_FIELD(wave_amplitude); RT_HAIR_PY_FIELD(frizz);
        RT_HAIR_PY_FIELD(roughness); RT_HAIR_PY_FIELD(gravity);
        RT_HAIR_PY_FIELD(force_influence); RT_HAIR_PY_FIELD(use_dynamics);
        RT_HAIR_PY_FIELD(physics_damping); RT_HAIR_PY_FIELD(physics_stiffness);
        RT_HAIR_PY_FIELD(physics_mass); RT_HAIR_PY_FIELD(use_tangent_shading);
        RT_HAIR_PY_FIELD(use_bspline); RT_HAIR_PY_FIELD(subdivisions);
#undef RT_HAIR_PY_FIELD
        return d;
    };
    auto hair_info_to_dict = [hair_settings_to_dict](const rtapi::HairGroomInfo& info) {
        py::dict d;
        d["name"] = info.name; d["bound_mesh"] = info.bound_mesh;
        d["guide_count"] = info.guide_count; d["child_count"] = info.child_count;
        d["point_count"] = info.point_count; d["material"] = info.material;
        d["visible"] = info.visible; d["dirty"] = info.dirty;
        d["settings"] = hair_settings_to_dict(info.settings);
        return d;
    };
    auto apply_hair_kwargs = [](rtapi::HairSettings& s, const py::kwargs& k) {
#define RT_HAIR_KW(name, type) if (k.contains(#name)) s.name = py::cast<type>(k[#name])
        RT_HAIR_KW(guide_count, uint32_t); RT_HAIR_KW(children_per_guide, uint32_t);
        RT_HAIR_KW(points_per_strand, uint32_t); RT_HAIR_KW(length, float);
        RT_HAIR_KW(length_variation, float); RT_HAIR_KW(root_radius, float);
        RT_HAIR_KW(tip_radius, float); RT_HAIR_KW(clumpiness, float);
        RT_HAIR_KW(child_radius, float); RT_HAIR_KW(curl_frequency, float);
        RT_HAIR_KW(curl_radius, float); RT_HAIR_KW(wave_frequency, float);
        RT_HAIR_KW(wave_amplitude, float); RT_HAIR_KW(frizz, float);
        RT_HAIR_KW(roughness, float); RT_HAIR_KW(gravity, float);
        RT_HAIR_KW(force_influence, float); RT_HAIR_KW(use_dynamics, bool);
        RT_HAIR_KW(physics_damping, float); RT_HAIR_KW(physics_stiffness, float);
        RT_HAIR_KW(physics_mass, float); RT_HAIR_KW(use_tangent_shading, bool);
        RT_HAIR_KW(use_bspline, bool); RT_HAIR_KW(subdivisions, uint32_t);
#undef RT_HAIR_KW
    };
    hair.def("list", [hair_info_to_dict]() {
        std::vector<rtapi::HairGroomInfo> grooms;
        requireResult(rtapi::listHairGrooms(grooms));
        py::list out; for (const auto& groom : grooms) out.append(hair_info_to_dict(groom));
        return out;
    });
    hair.def("get", [hair_info_to_dict](const std::string& name) {
        rtapi::HairGroomInfo info; requireResult(rtapi::getHairGroom(name, info));
        return hair_info_to_dict(info);
    }, py::arg("name"));
    hair.def("create", [hair_info_to_dict, apply_hair_kwargs](const std::string& mesh,
             const std::string& name, const py::kwargs& kwargs) {
        rtapi::HairSettings settings; apply_hair_kwargs(settings, kwargs);
        rtapi::HairGroomInfo info;
        requireResult(rtapi::createHairGroom(mesh, name, settings, info));
        return hair_info_to_dict(info);
    }, py::arg("mesh"), py::arg("name") = "HairGroom");
    hair.def("update", [apply_hair_kwargs](const std::string& name, const py::kwargs& kwargs) {
        rtapi::HairGroomInfo info; requireResult(rtapi::getHairGroom(name, info));
        apply_hair_kwargs(info.settings, kwargs);
        bool visible = info.visible;
        const bool* visible_ptr = nullptr;
        if (kwargs.contains("visible")) { visible = py::cast<bool>(kwargs["visible"]); visible_ptr = &visible; }
        requireResult(rtapi::updateHairGroom(name, info.settings, visible_ptr));
    }, py::arg("name"));
    hair.def("rename", [hair_info_to_dict](const std::string& name, const std::string& new_name) {
        rtapi::HairGroomInfo info; requireResult(rtapi::renameHairGroom(name, new_name, info));
        return hair_info_to_dict(info);
    }, py::arg("name"), py::arg("new_name"));
    hair.def("remove", [](const std::string& name) { requireResult(rtapi::removeHairGroom(name)); }, py::arg("name"));
    hair.def("restyle", [](const std::string& name) { requireResult(rtapi::restyleHairGroom(name)); }, py::arg("name"));
    hair.def("list_presets", []() {
        std::vector<std::string> presets; requireResult(rtapi::listHairPresets(presets));
        return presets;
    });
    hair.def("apply_preset", [](const std::string& name, const std::string& preset) {
        requireResult(rtapi::applyHairPreset(name, preset));
    }, py::arg("name"), py::arg("preset"));
    hair.def("trim", [](const std::string& name, float length_factor) {
        requireResult(rtapi::trimHairGroom(name, length_factor));
    }, py::arg("name"), py::arg("length_factor"));
    hair.def("grow", [](const std::string& name, float length_factor) {
        requireResult(rtapi::growHairGroom(name, length_factor));
    }, py::arg("name"), py::arg("length_factor"));
    hair.def("comb", [](const std::string& name, const py::handle& direction,
                          float strength, float root_stiffness) {
        requireResult(rtapi::combHairGroom(name, vec3FromPython(direction), strength, root_stiffness));
    }, py::arg("name"), py::arg("direction"), py::arg("strength") = 0.5f,
       py::arg("root_stiffness") = 0.75f);
    hair.def("smooth", [](const std::string& name, float strength, int iterations) {
        requireResult(rtapi::smoothHairGroom(name, strength, iterations));
    }, py::arg("name"), py::arg("strength") = 0.5f, py::arg("iterations") = 2);
    hair.def("reset_simulation", [](const std::string& name) {
        requireResult(rtapi::resetHairSimulation(name));
    }, py::arg("name"));
    hair.def("bake", [](const std::string& name) {
        requireResult(rtapi::bakeHairGroom(name));
    }, py::arg("name"));

    py::module_ paint = module.def_submodule("paint", "Deterministic mesh paint layer automation");
    auto paint_layer_to_dict = [](const rtapi::PaintLayerInfo& layer) {
        py::dict d;
        d["index"] = layer.index; d["id"] = layer.id; d["name"] = layer.name;
        d["visible"] = layer.visible; d["locked"] = layer.locked;
        d["opacity"] = layer.opacity; d["blend_mode"] = layer.blend_mode;
        d["channels"] = layer.channels;
        return d;
    };
    auto paint_target_to_dict = [paint_layer_to_dict](const rtapi::PaintTargetInfo& info) {
        py::dict d;
        d["object"] = info.object_name; d["material_id"] = info.material_id;
        d["resolution"] = info.resolution; d["channels"] = info.channels;
        py::list layers; for (const auto& layer : info.layers) layers.append(paint_layer_to_dict(layer));
        d["layers"] = std::move(layers);
        return d;
    };
    paint.def("get", [paint_target_to_dict](const std::string& object, int material_id) {
        rtapi::PaintTargetInfo info; requireResult(rtapi::getPaintTarget(object, material_id, info));
        return paint_target_to_dict(info);
    }, py::arg("object"), py::arg("material_id") = -1);
    paint.def("ensure", [paint_target_to_dict](const std::string& object, int material_id, int resolution) {
        rtapi::PaintTargetInfo info;
        requireResult(rtapi::ensurePaintTarget(object, material_id, resolution, info));
        return paint_target_to_dict(info);
    }, py::arg("object"), py::arg("material_id") = -1, py::arg("resolution") = 1024);
    paint.def("add_layer", [paint_layer_to_dict](const std::string& object, const std::string& name,
                                                   int material_id, int insert_at) {
        rtapi::PaintLayerInfo info;
        requireResult(rtapi::addPaintLayer(object, material_id, name, insert_at, info));
        return paint_layer_to_dict(info);
    }, py::arg("object"), py::arg("name") = "Paint Layer", py::arg("material_id") = -1,
       py::arg("insert_at") = -1);
    paint.def("remove_layer", [](const std::string& object, int layer_index, int material_id) {
        requireResult(rtapi::removePaintLayer(object, material_id, layer_index));
    }, py::arg("object"), py::arg("layer_index"), py::arg("material_id") = -1);
    paint.def("update_layer", [](const std::string& object, int layer_index, int material_id,
                                   const py::kwargs& kwargs) {
        std::string name, blend; bool visible = true, locked = false; float opacity = 1.0f;
        const std::string* p_name = nullptr; const std::string* p_blend = nullptr;
        const bool* p_visible = nullptr; const bool* p_locked = nullptr; const float* p_opacity = nullptr;
        if (kwargs.contains("name")) { name = py::cast<std::string>(kwargs["name"]); p_name = &name; }
        if (kwargs.contains("visible")) { visible = py::cast<bool>(kwargs["visible"]); p_visible = &visible; }
        if (kwargs.contains("locked")) { locked = py::cast<bool>(kwargs["locked"]); p_locked = &locked; }
        if (kwargs.contains("opacity")) { opacity = py::cast<float>(kwargs["opacity"]); p_opacity = &opacity; }
        if (kwargs.contains("blend_mode")) { blend = py::cast<std::string>(kwargs["blend_mode"]); p_blend = &blend; }
        requireResult(rtapi::updatePaintLayer(object, material_id, layer_index, p_name, p_visible,
                                               p_locked, p_opacity, p_blend));
    }, py::arg("object"), py::arg("layer_index"), py::arg("material_id") = -1);
    paint.def("fill", [](const std::string& object, int layer_index, const std::string& channel,
                           const py::handle& color, int material_id) {
        requireResult(rtapi::fillPaintLayer(object, material_id, layer_index, channel,
                                             vec3FromPython(color)));
    }, py::arg("object"), py::arg("layer_index"), py::arg("channel"),
       py::arg("color"), py::arg("material_id") = -1);
    paint.def("clear_channel", [](const std::string& object, int layer_index,
                                    const std::string& channel, int material_id) {
        requireResult(rtapi::clearPaintLayerChannel(object, material_id, layer_index, channel));
    }, py::arg("object"), py::arg("layer_index"), py::arg("channel"),
       py::arg("material_id") = -1);
    paint.def("duplicate_layer", [paint_layer_to_dict](const std::string& object,
              int layer_index, int material_id) {
        rtapi::PaintLayerInfo info;
        requireResult(rtapi::duplicatePaintLayer(object, material_id, layer_index, info));
        return paint_layer_to_dict(info);
    }, py::arg("object"), py::arg("layer_index"), py::arg("material_id") = -1);
    paint.def("move_layer", [](const std::string& object, int from_index, int to_index,
                                 int material_id) {
        requireResult(rtapi::movePaintLayer(object, material_id, from_index, to_index));
    }, py::arg("object"), py::arg("from_index"), py::arg("to_index"),
       py::arg("material_id") = -1);
    paint.def("merge_down", [](const std::string& object, int layer_index, int material_id) {
        requireResult(rtapi::mergePaintLayerDown(object, material_id, layer_index));
    }, py::arg("object"), py::arg("layer_index"), py::arg("material_id") = -1);
    paint.def("flatten", [](const std::string& object, int material_id) {
        requireResult(rtapi::flattenPaintLayers(object, material_id));
    }, py::arg("object"), py::arg("material_id") = -1);
    paint.def("bake_height_to_normal", [](const std::string& object, float strength,
                                            bool clear_height, int material_id) {
        requireResult(rtapi::bakePaintHeightToNormal(object, material_id, strength, clear_height));
    }, py::arg("object"), py::arg("strength") = 4.0f,
       py::arg("clear_height") = false, py::arg("material_id") = -1);
    paint.def("import_channel", [](const std::string& object, int layer_index,
                                     const std::string& channel, const std::string& filepath,
                                     int material_id) {
        requireResult(rtapi::importPaintChannel(object, material_id, layer_index, channel, filepath));
    }, py::arg("object"), py::arg("layer_index"), py::arg("channel"),
       py::arg("filepath"), py::arg("material_id") = -1);
    paint.def("export_channel", [](const std::string& object, const std::string& channel,
                                     const std::string& filepath, int layer_index, int material_id) {
        requireResult(rtapi::exportPaintChannel(object, material_id, layer_index, channel, filepath));
    }, py::arg("object"), py::arg("channel"), py::arg("filepath"),
       py::arg("layer_index") = -1, py::arg("material_id") = -1);
    paint.def("list_mask_presets", []() {
        std::vector<std::string> presets; requireResult(rtapi::listPaintMaskPresets(presets));
        return presets;
    });
    paint.def("apply_mask", [](const std::string& object, int layer_index,
                                 const std::string& preset, float strength,
                                 unsigned int seed, int material_id) {
        requireResult(rtapi::applyPaintMaskPreset(object, material_id, layer_index,
                                                   preset, strength, seed));
    }, py::arg("object"), py::arg("layer_index"), py::arg("preset"),
       py::arg("strength") = 1.0f, py::arg("seed") = 1337,
       py::arg("material_id") = -1);

    py::module_ sculpt = module.def_submodule("sculpt", "Deterministic flat-mesh sculpt automation");
    auto sculpt_points = [](const py::iterable& values) {
        std::vector<Vec3> result;
        for (const py::handle value : values) result.push_back(vec3FromPython(value));
        return result;
    };
    sculpt.def("get", [](const std::string& object) {
        rtapi::SculptInfo info; requireResult(rtapi::getSculptInfo(object, info));
        py::dict d; d["object"] = info.object_name; d["vertex_count"] = info.vertex_count;
        d["has_mask"] = info.has_mask; d["mask_min"] = info.mask_min; d["mask_max"] = info.mask_max;
        return d;
    }, py::arg("object"));
    sculpt.def("stroke", [sculpt_points](const std::string& object, const std::string& tool,
                 const py::iterable& points, float radius, float strength, float falloff,
                 const py::handle& direction, unsigned int seed, bool use_mask, bool undo) {
        rtapi::SculptStrokeSettings s; s.tool = tool; s.points = sculpt_points(points);
        s.radius = radius; s.strength = strength; s.falloff = falloff;
        s.direction = vec3FromPython(direction); s.seed = seed; s.use_mask = use_mask; s.undo = undo;
        requireResult(rtapi::applySculptStroke(object, s));
    }, py::arg("object"), py::arg("tool"), py::arg("points"), py::arg("radius") = 0.25f,
       py::arg("strength") = 0.05f, py::arg("falloff") = 0.75f,
       py::arg("direction") = py::make_tuple(0.0f, 1.0f, 0.0f), py::arg("seed") = 1337,
       py::arg("use_mask") = true, py::arg("undo") = true);
    sculpt.def("paint_mask", [sculpt_points](const std::string& object, const py::iterable& points,
                 float radius, float value, float strength, bool undo) {
        requireResult(rtapi::paintSculptMask(object, sculpt_points(points), radius, value, strength, undo));
    }, py::arg("object"), py::arg("points"), py::arg("radius"), py::arg("value"),
       py::arg("strength") = 1.0f, py::arg("undo") = true);
    sculpt.def("mask_operation", [](const std::string& object, const std::string& operation,
                                      unsigned int seed, bool undo) {
        requireResult(rtapi::applySculptMaskOperation(object, operation, seed, undo));
    }, py::arg("object"), py::arg("operation"), py::arg("seed") = 1337, py::arg("undo") = true);

    py::module_ materials = module.def_submodule("material", "Object material parameters");
    materials.def("get", [](const std::string& object_name, const std::string& param) -> py::object {
        rtapi::MaterialParamValue value;
        requireResult(rtapi::getMaterialParam(object_name, param, value));
        if (value.is_color) return vec3ToPython(value.color);
        return py::float_(value.scalar);
    }, py::arg("object_name"), py::arg("param"));
    materials.def("set", [](const std::string& object_name, const std::string& param,
                             const py::handle& value) {
        if (PyFloat_Check(value.ptr()) || PyLong_Check(value.ptr())) {
            requireResult(rtapi::setMaterialParam(object_name, param, py::cast<float>(value)));
        } else {
            requireResult(rtapi::setMaterialParam(object_name, param, vec3FromPython(value)));
        }
    }, py::arg("object_name"), py::arg("param"), py::arg("value"));

    py::module_ lights = module.def_submodule("lights", "Scene light operations");
    lights.def("list", [] {
        py::list result;
        for (const rtapi::LightInfo& info : rtapi::listLights()) {
            py::dict item;
            item["index"] = info.index;
            item["name"] = info.name;
            item["type"] = info.type;
            item["position"] = vec3ToPython(info.position);
            result.append(std::move(item));
        }
        return result;
    });
    lights.def("add", [](const std::string& type, const py::handle& position) {
        std::string name;
        requireResult(rtapi::addLight(type, vec3FromPython(position), name));
        return name;
    }, py::arg("type"), py::arg("position"));
    lights.def("delete", [](int index) { requireResult(rtapi::deleteLight(index)); });
    lights.def("set_position", [](int index, const py::handle& position) {
        requireResult(rtapi::setLightPosition(index, vec3FromPython(position)));
    });

    py::module_ anim = module.def_submodule("anim", "Keyframe animation (transform tracks)");
    anim.def("insert_key", [](const std::string& object_name, const std::string& channel,
                              int frame, const py::handle& value) {
        requireResult(rtapi::insertKeyframe(object_name, channel, frame, vec3FromPython(value)));
    }, py::arg("object_name"), py::arg("channel"), py::arg("frame"), py::arg("value"));
    anim.def("remove_key", [](const std::string& object_name, int frame) {
        requireResult(rtapi::removeKeyframe(object_name, frame));
    }, py::arg("object_name"), py::arg("frame"));
    anim.def("list_keys", [](const std::string& object_name) {
        return rtapi::listKeyframes(object_name);
    }, py::arg("object_name"));

    py::module_ nodes = module.def_submodule("nodes",
        "Node graph construction (material / geometry / terrain graphs via NodeRegistry)");
    nodes.def("types", [] {
        py::list result;
        for (const rtapi::NodeTypeDesc& t : rtapi::listNodeTypes()) {
            py::dict item;
            item["type_id"] = t.type_id;
            item["category"] = t.category;
            item["display_name"] = t.display_name;
            item["description"] = t.description;
            result.append(std::move(item));
        }
        return result;
    });
    nodes.def("add", [](const std::string& graph_type, const std::string& graph_name,
                        const std::string& type_id) {
        unsigned int id = 0;
        requireResult(rtapi::addNode(graph_type, graph_name, type_id, id));
        return id;
    }, py::arg("graph_type"), py::arg("graph_name"), py::arg("type_id"));
    nodes.def("remove", [](const std::string& graph_type, const std::string& graph_name,
                           unsigned int node_id) {
        requireResult(rtapi::removeNode(graph_type, graph_name, node_id));
    }, py::arg("graph_type"), py::arg("graph_name"), py::arg("node_id"));
    nodes.def("link", [](const std::string& graph_type, const std::string& graph_name,
                         unsigned int from_node, int from_output,
                         unsigned int to_node, int to_input) {
        unsigned int link_id = 0;
        requireResult(rtapi::linkNodes(graph_type, graph_name, from_node, from_output,
                                       to_node, to_input, link_id));
        return link_id;
    }, py::arg("graph_type"), py::arg("graph_name"), py::arg("from_node"),
       py::arg("from_output"), py::arg("to_node"), py::arg("to_input"));
    nodes.def("list", [](const std::string& graph_type, const std::string& graph_name) {
        std::vector<rtapi::NodeDesc> descs;
        requireResult(rtapi::listNodes(graph_type, graph_name, descs));
        py::list result;
        for (const rtapi::NodeDesc& d : descs) {
            py::dict item;
            item["id"] = d.id;
            item["type_id"] = d.type_id;
            item["display_name"] = d.display_name;
            item["inputs"] = d.input_count;
            item["outputs"] = d.output_count;
            result.append(std::move(item));
        }
        return result;
    }, py::arg("graph_type"), py::arg("graph_name"));

    // Node parameters (Faz 5.1b): a node's input-pin default values. list_params
    // enumerates them (index/name/type/connected/value); get/set_param address a
    // pin by input-slot index. A connected input keeps its link and ignores its
    // default until unlinked.
    nodes.def("list_params", [](const std::string& graph_type, const std::string& graph_name,
                                unsigned int node_id) {
        std::vector<rtapi::NodeParamInfo> params;
        requireResult(rtapi::listNodeParams(graph_type, graph_name, node_id, params));
        py::list result;
        for (const rtapi::NodeParamInfo& p : params) {
            py::dict item;
            item["index"] = p.index;
            item["name"] = p.name;
            item["type"] = p.data_type;
            item["connected"] = p.connected;
            item["value"] = nodeParamToPython(p.value);
            result.append(std::move(item));
        }
        return result;
    }, py::arg("graph_type"), py::arg("graph_name"), py::arg("node_id"));
    nodes.def("get_param", [](const std::string& graph_type, const std::string& graph_name,
                              unsigned int node_id, int pin_index) {
        rtapi::NodeParamValue value;
        requireResult(rtapi::getNodeParam(graph_type, graph_name, node_id, pin_index, value));
        return nodeParamToPython(value);
    }, py::arg("graph_type"), py::arg("graph_name"), py::arg("node_id"), py::arg("pin_index"));
    nodes.def("set_param", [](const std::string& graph_type, const std::string& graph_name,
                              unsigned int node_id, int pin_index, const py::handle& value) {
        requireResult(rtapi::setNodeParam(graph_type, graph_name, node_id, pin_index,
                                          nodeParamFromPython(value)));
    }, py::arg("graph_type"), py::arg("graph_name"), py::arg("node_id"),
       py::arg("pin_index"), py::arg("value"));
    nodes.def("list_properties", [](const std::string& graph_type, const std::string& graph_name,
                                     unsigned int node_id) {
        std::vector<rtapi::NodePropertyInfo> properties;
        requireResult(rtapi::listNodeProperties(graph_type, graph_name, node_id, properties));
        py::list result;
        for (const auto& p : properties) {
            py::dict item;
            item["name"] = p.name;
            item["type"] = p.data_type;
            item["value"] = nodeParamToPython(p.value);
            result.append(std::move(item));
        }
        return result;
    }, py::arg("graph_type"), py::arg("graph_name"), py::arg("node_id"));
    nodes.def("get_property", [](const std::string& graph_type, const std::string& graph_name,
                                  unsigned int node_id, const std::string& property) {
        rtapi::NodeParamValue value;
        requireResult(rtapi::getNodeProperty(graph_type, graph_name, node_id, property, value));
        return nodeParamToPython(value);
    }, py::arg("graph_type"), py::arg("graph_name"), py::arg("node_id"), py::arg("property"));
    nodes.def("set_property", [](const std::string& graph_type, const std::string& graph_name,
                                  unsigned int node_id, const std::string& property,
                                  const py::handle& value) {
        requireResult(rtapi::setNodeProperty(graph_type, graph_name, node_id, property,
                                              nodeParamFromPython(value)));
    }, py::arg("graph_type"), py::arg("graph_name"), py::arg("node_id"),
       py::arg("property"), py::arg("value"));

    // Events (Faz 3b). Callbacks fire on the main thread; re-acquire the GIL
    // there and never let a Python exception cross back into C++.
    module.def("on_frame_change", [](py::function fn) {
        return rtapi::addFrameChangeCallback([fn](int frame) {
            py::gil_scoped_acquire gil;
            try { fn(frame); }
            catch (const py::error_already_set& e) { appendConsoleLine(e.what()); }
        });
    }, py::arg("callback"));
    module.def("on_scene_load", [](py::function fn) {
        return rtapi::addSceneLoadCallback([fn]() {
            py::gil_scoped_acquire gil;
            try { fn(); }
            catch (const py::error_already_set& e) { appendConsoleLine(e.what()); }
        });
    }, py::arg("callback"));
    module.def("remove_frame_change_callback", &rtapi::removeFrameChangeCallback, py::arg("id"));
    module.def("remove_scene_load_callback", &rtapi::removeSceneLoadCallback, py::arg("id"));

    py::module_ camera = module.def_submodule("camera", "Active camera get/set (Faz 5.1a)");
    camera.def("get", [] {
        rtapi::CameraState s;
        requireResult(rtapi::getCamera(s));
        py::dict d;
        d["position"]       = vec3ToPython(s.position);
        d["target"]         = vec3ToPython(s.target);
        d["up"]             = vec3ToPython(s.up);
        d["fov"]            = s.fov;
        d["focus_distance"] = s.focus_distance;
        d["aperture"]       = s.aperture;
        return d;
    });
    camera.def("set", [](const py::kwargs& kwargs) {
        // Only the provided fields change (rt.camera.set(position=(..), fov=60)).
        if (kwargs.contains("position"))
            requireResult(rtapi::setCameraPosition(vec3FromPython(kwargs["position"])));
        if (kwargs.contains("target"))
            requireResult(rtapi::setCameraTarget(vec3FromPython(kwargs["target"])));
        if (kwargs.contains("fov"))
            requireResult(rtapi::setCameraFov(py::cast<float>(kwargs["fov"])));
        if (kwargs.contains("focus_distance"))
            requireResult(rtapi::setCameraFocusDistance(py::cast<float>(kwargs["focus_distance"])));
        if (kwargs.contains("aperture"))
            requireResult(rtapi::setCameraAperture(py::cast<float>(kwargs["aperture"])));
    });

    py::module_ world = module.def_submodule("world", "World/environment: background + Nishita sun (Faz 5.1c)");
    world.def("get", [] {
        rtapi::WorldState s;
        requireResult(rtapi::getWorld(s));
        py::dict d;
        d["mode"]                  = s.mode;
        d["background_color"]      = vec3ToPython(s.background_color);
        d["sun_elevation"]         = s.sun_elevation;
        d["sun_azimuth"]           = s.sun_azimuth;
        d["sun_intensity"]         = s.sun_intensity;
        d["atmosphere_intensity"]  = s.atmosphere_intensity;
        d["sun_size"]              = s.sun_size;
        return d;
    });
    world.def("set", [](const py::kwargs& kwargs) {
        // Apply mode first so a same-call background_color lands in solid mode.
        if (kwargs.contains("mode"))
            requireResult(rtapi::setWorldMode(py::cast<std::string>(kwargs["mode"])));
        if (kwargs.contains("background_color"))
            requireResult(rtapi::setWorldBackgroundColor(vec3FromPython(kwargs["background_color"])));
        if (kwargs.contains("sun_elevation"))
            requireResult(rtapi::setWorldSunElevation(py::cast<float>(kwargs["sun_elevation"])));
        if (kwargs.contains("sun_azimuth"))
            requireResult(rtapi::setWorldSunAzimuth(py::cast<float>(kwargs["sun_azimuth"])));
        if (kwargs.contains("sun_intensity"))
            requireResult(rtapi::setWorldSunIntensity(py::cast<float>(kwargs["sun_intensity"])));
        if (kwargs.contains("atmosphere_intensity"))
            requireResult(rtapi::setWorldAtmosphereIntensity(py::cast<float>(kwargs["atmosphere_intensity"])));
        if (kwargs.contains("sun_size"))
            requireResult(rtapi::setWorldSunSize(py::cast<float>(kwargs["sun_size"])));
    });

    py::module_ post = module.def_submodule("post", "Post-processing: exposure, tonemap, vignette, stylize (Faz 5.1d)");
    post.def("get", [] {
        rtapi::PostState s;
        requireResult(rtapi::getPost(s));
        py::dict d;
        d["exposure"]          = s.exposure;
        d["gamma"]             = s.gamma;
        d["saturation"]        = s.saturation;
        d["color_temperature"] = s.color_temperature;
        d["tone_mapping"]      = s.tone_mapping;
        d["vignette_enabled"]  = s.vignette_enabled;
        d["vignette_strength"] = s.vignette_strength;
        d["stylize_enabled"]   = s.stylize_enabled;
        d["stylize_strength"]  = s.stylize_strength;
        return d;
    });
    post.def("set", [](const py::kwargs& kwargs) {
        if (kwargs.contains("exposure"))
            requireResult(rtapi::setPostExposure(py::cast<float>(kwargs["exposure"])));
        if (kwargs.contains("gamma"))
            requireResult(rtapi::setPostGamma(py::cast<float>(kwargs["gamma"])));
        if (kwargs.contains("saturation"))
            requireResult(rtapi::setPostSaturation(py::cast<float>(kwargs["saturation"])));
        if (kwargs.contains("color_temperature"))
            requireResult(rtapi::setPostColorTemperature(py::cast<float>(kwargs["color_temperature"])));
        if (kwargs.contains("tone_mapping"))
            requireResult(rtapi::setPostToneMapping(py::cast<std::string>(kwargs["tone_mapping"])));
        if (kwargs.contains("vignette_enabled"))
            requireResult(rtapi::setPostVignetteEnabled(py::cast<bool>(kwargs["vignette_enabled"])));
        if (kwargs.contains("vignette_strength"))
            requireResult(rtapi::setPostVignetteStrength(py::cast<float>(kwargs["vignette_strength"])));
        if (kwargs.contains("stylize_enabled"))
            requireResult(rtapi::setPostStylizeEnabled(py::cast<bool>(kwargs["stylize_enabled"])));
        if (kwargs.contains("stylize_strength"))
            requireResult(rtapi::setPostStylizeStrength(py::cast<float>(kwargs["stylize_strength"])));
    });

    py::module_ project = module.def_submodule("project", "Project file operations");
    project.def("path", &rtapi::currentProjectPath);
    project.def("save", [](const std::string& path) {
        requireResult(rtapi::saveProject(path));
    }, py::arg("path") = std::string());
    project.def("open", [](const std::string& path) {
        requireResult(rtapi::openProject(path));
    }, py::arg("path"));

    py::module_ timeline = module.def_submodule("timeline", "Timeline controls");
    timeline.def("get_frame", &rtapi::currentFrame);
    timeline.def("set_frame", [](int frame) { requireResult(rtapi::setFrame(frame)); }, py::arg("frame"));

    py::module_ render = module.def_submodule("render", "Asynchronous final-render jobs");
    render.def("start", [](const std::string& output_path, int spp) {
        requireResult(rtapi::renderFrame(output_path, spp));
    }, py::arg("output_path"), py::arg("spp"));
    render.def("status", [] {
        const rtapi::RenderJobInfo info = rtapi::renderStatus();
        const char* state = "idle";
        switch (info.state) {
            case rtapi::RenderJobState::Rendering: state = "rendering"; break;
            case rtapi::RenderJobState::Completed: state = "completed"; break;
            case rtapi::RenderJobState::Failed:    state = "failed"; break;
            case rtapi::RenderJobState::Cancelled: state = "cancelled"; break;
            case rtapi::RenderJobState::Idle:
            default: break;
        }
        py::dict result;
        result["state"] = state;
        result["output_path"] = info.output_path;
        result["error"] = info.error;
        result["current_samples"] = info.current_samples;
        result["target_samples"] = info.target_samples;
        result["progress"] = info.progress;
        return result;
    });
    render.def("cancel", [] { requireResult(rtapi::cancelRender()); });

    // Multi-frame sequence render (maps to the g_seq_save_active state machine).
    render.def("start_sequence", [](const std::string& output_dir, int spp,
                                    int start_frame, int end_frame) {
        requireResult(rtapi::renderSequence(output_dir, spp, start_frame, end_frame));
    }, py::arg("output_dir"), py::arg("spp"), py::arg("start_frame"), py::arg("end_frame"));
    render.def("sequence_status", [] {
        const rtapi::SequenceJobInfo info = rtapi::sequenceStatus();
        py::dict result;
        result["active"]          = info.active;
        result["current_frame"]   = info.current_frame;
        result["start_frame"]     = info.start_frame;
        result["end_frame"]       = info.end_frame;
        result["frame_progress"]  = info.frame_progress;
        result["total_progress"]  = info.total_progress;
        result["output_dir"]      = info.output_dir;
        result["error"]           = info.error;
        return result;
    });
    render.def("cancel_sequence", [] { requireResult(rtapi::cancelSequence()); });


    py::module_ addons = module.def_submodule("addons", "Addon management (Faz 4a)");
    addons.def("list", [] {
        py::list result;
        for (const rtpython::AddonInfo& a : rtpython::listAddons()) {
            py::dict item;
            item["module_name"]  = a.module_name;
            item["display_name"] = a.display_name;
            item["description"]  = a.description;
            item["version"]      = a.version;
            item["enabled"]      = a.enabled;
            item["loaded"]       = a.loaded;
            result.append(std::move(item));
        }
        return result;
    });
    addons.def("enable", [](const std::string& module_name) {
        std::string err;
        if (!rtpython::enableAddon(module_name, err)) throw std::runtime_error(err);
    }, py::arg("module_name"));
    addons.def("disable", [](const std::string& module_name) {
        std::string err;
        if (!rtpython::disableAddon(module_name, err)) throw std::runtime_error(err);
    }, py::arg("module_name"));
    addons.def("reload", [](const std::string& module_name) {
        std::string err;
        if (!rtpython::reloadAddon(module_name, err)) throw std::runtime_error(err);
    }, py::arg("module_name"));

    // rt.ui (Faz 4b): addons register an immediate-mode panel; its draw callback
    // runs each frame from drawAddonPanels(). The widget calls below are only valid
    // inside such a callback (guarded by requireAddonUiContext()).
    py::module_ ui = module.def_submodule("ui", "Addon panels & immediate-mode widgets (Faz 4b)");
    ui.def("register_panel", [](const std::string& title, py::function draw) {
        const int id = g_next_panel_id++;
        g_addon_panels[id] = AddonPanel{ title, std::move(draw) };
        return id;
    }, py::arg("title"), py::arg("draw_callback"));
    ui.def("unregister_panel", [](int panel_id) {
        g_addon_panels.erase(panel_id);
    }, py::arg("panel_id"));

    ui.def("text", [](const std::string& s) {
        requireAddonUiContext();
        ImGui::TextUnformatted(s.c_str());
    }, py::arg("text"));
    ui.def("button", [](const std::string& label) {
        requireAddonUiContext();
        return ImGui::Button(label.c_str());
    }, py::arg("label"));
    ui.def("checkbox", [](const std::string& label, bool value) {
        requireAddonUiContext();
        ImGui::Checkbox(label.c_str(), &value);
        return value;
    }, py::arg("label"), py::arg("value"));
    ui.def("slider_float", [](const std::string& label, float value, float v_min, float v_max) {
        requireAddonUiContext();
        ImGui::SliderFloat(label.c_str(), &value, v_min, v_max);
        return value;
    }, py::arg("label"), py::arg("value"), py::arg("v_min"), py::arg("v_max"));
    ui.def("input_text", [](const std::string& label, const std::string& value) {
        requireAddonUiContext();
        char buf[1024];
        std::snprintf(buf, sizeof(buf), "%s", value.c_str());
        ImGui::InputText(label.c_str(), buf, sizeof(buf));
        return std::string(buf);
    }, py::arg("label"), py::arg("value"));
    ui.def("separator", [] { requireAddonUiContext(); ImGui::Separator(); });
    ui.def("same_line", [] { requireAddonUiContext(); ImGui::SameLine(); });

    module.def("undo", [] { requireResult(rtapi::undo()); });
    module.def("redo", [] { requireResult(rtapi::redo()); });
    module.def("request_render", [] { requireResult(rtapi::requestRender()); });
    module.def("reset_accumulation", [] { requireResult(rtapi::resetAccumulation()); });
    module.def("run_script", [](const std::string& path) { requireResult(rtapi::runScriptFile(path)); });
}

namespace rtpython {

bool initialize(std::string& error) {
    error.clear();
    if (g_interpreter) return true;
    if (!rtapi::isBound()) {
        error = "rtapi must be bound before Python initialization";
        return false;
    }

    try {
        const std::filesystem::path pythonHome = std::filesystem::current_path() / "python";
#ifdef _WIN32
        if (std::filesystem::exists(pythonHome)) {
            _putenv_s("PYTHONHOME", pythonHome.string().c_str());
        }
#endif
        g_interpreter = std::make_unique<py::scoped_interpreter>();

        py::module_ sys = py::module_::import("sys");
        py::module_ rt = py::module_::import("rt");
        py::object stream = rt.attr("_ConsoleStream")();
        sys.attr("stdout") = stream;
        sys.attr("stderr") = stream;

        py::list path = sys.attr("path");
        const std::filesystem::path scripts = std::filesystem::current_path() / "scripts";
        if (std::filesystem::exists(scripts)) path.insert(0, scripts.string());

        appendConsoleLine("RayTrophi Python ready (rt " + py::cast<std::string>(rt.attr("version")()) + ")");

        // Faz 4a: auto-load persisted-enabled addons. Failures are logged per-addon
        // and never abort Python init (one bad addon must not break scripting).
        loadEnabledAddons();
        return true;
    } catch (const py::error_already_set& e) {
        error = e.what();
    } catch (const std::exception& e) {
        error = e.what();
    }

    g_interpreter.reset();
    appendConsoleLine("Python initialization failed: " + error);
    return false;
}

void shutdown() noexcept {
    if (!g_interpreter) return;
    try {
        unloadAllAddons();  // run unregister() while the interpreter is still alive
        // Destroy any py::function objects (event callbacks + rt.ui panels) NOW,
        // while the interpreter and GIL are still alive. rtapi::unbind() (called
        // after this in Main.cpp) also clears the event callbacks, but a py::function
        // destructor needs the GIL — running it after the interpreter is gone would
        // fault. This makes unbind()'s clear a no-op.
        g_addon_panels.clear();
        rtapi::clearEventCallbacks();
        g_interpreter.reset();
    } catch (...) {
        // Shutdown must never escape into the renderer teardown path.
    }
}

bool isInitialized() {
    return static_cast<bool>(g_interpreter);
}

ExecutionResult execute(const std::string& source, const std::string& filename) {
    if (!g_interpreter) return { false, "Python runtime is not initialized" };
    if (source.empty()) return { true, {} };

    try {
        py::dict globals = py::module_::import("__main__").attr("__dict__");
        globals["__file__"] = filename;
        if (filename == "<console>") {
            // Py_single_input invokes sys.displayhook, giving the console true
            // REPL behaviour for expressions while still accepting statements.
            py::eval<py::eval_single_statement>(source, globals, globals);
        } else {
            py::exec(source, globals, globals);
        }
        return { true, {} };
    } catch (const py::error_already_set& e) {
        const std::string traceback = e.what();
        appendConsoleLine(traceback);
        return { false, traceback };
    } catch (const std::exception& e) {
        appendConsoleLine(e.what());
        return { false, e.what() };
    }
}

ExecutionResult executeFile(const std::string& filepath) {
    if (!g_interpreter) return { false, "Python runtime is not initialized" };
    if (!std::filesystem::is_regular_file(filepath)) {
        return { false, "script file not found: " + filepath };
    }
    try {
        py::dict globals = py::module_::import("__main__").attr("__dict__");
        globals["__file__"] = filepath;
        py::eval_file(filepath, globals, globals);
        return { true, {} };
    } catch (const py::error_already_set& e) {
        const std::string traceback = e.what();
        appendConsoleLine(traceback);
        return { false, traceback };
    } catch (const std::exception& e) {
        appendConsoleLine(e.what());
        return { false, e.what() };
    }
}

std::string consoleOutputSnapshot() {
    std::lock_guard<std::mutex> lock(g_output_mutex);
    return g_output;
}

void appendConsoleText(const std::string& text) {
    appendConsoleOutput(text);
}

void clearConsoleOutput() {
    std::lock_guard<std::mutex> lock(g_output_mutex);
    g_output.clear();
}

std::vector<std::string> getSubmodulePaths() {
    std::vector<std::string> result;
    result.push_back("rt");  // root is always present
    if (!g_interpreter) return result;
    try {
        py::module_ rt = py::module_::import("rt");
        py::list dir_list = py::module_::import("builtins").attr("dir")(rt);
        std::vector<std::string> subs;
        for (auto item : dir_list) {
            std::string name = py::cast<std::string>(item);
            if (name.rfind("__", 0) == 0) continue;
            py::object attr = rt.attr(name.c_str());
            if (PyModule_Check(attr.ptr())) {
                subs.push_back("rt." + name);
            }
        }
        std::sort(subs.begin(), subs.end());
        for (auto& s : subs) result.push_back(std::move(s));
    } catch (...) {
        // Reflection failure: fall back to just the root module.
    }
    return result;
}

// ---------------------------------------------------------------------------
// Addons (Faz 4a). State lives in two in-memory sets mirrored to addon_state.json;
// all functions run on the main thread with the GIL held (same as execute()).
// ---------------------------------------------------------------------------
namespace {

std::set<std::string> g_enabled_addons;   // persisted enable set
std::set<std::string> g_loaded_addons;    // register() called this session

std::filesystem::path addonsDir() {
    return std::filesystem::current_path() / "scripts" / "addons";
}
std::filesystem::path addonStatePath() {
    return std::filesystem::current_path() / "addon_state.json";
}

void loadAddonState() {
    g_enabled_addons.clear();
    std::error_code ec;
    const auto path = addonStatePath();
    if (!std::filesystem::is_regular_file(path, ec)) return;
    try {
        std::ifstream in(path);
        nlohmann::json j;
        in >> j;
        if (j.contains("enabled") && j["enabled"].is_array()) {
            for (const auto& e : j["enabled"]) g_enabled_addons.insert(e.get<std::string>());
        }
    } catch (...) {
        // Corrupt state file: start from an empty enabled set rather than crash.
    }
}

void saveAddonState() {
    try {
        nlohmann::json j;
        j["enabled"] = nlohmann::json::array();
        for (const auto& n : g_enabled_addons) j["enabled"].push_back(n);
        std::ofstream out(addonStatePath());
        out << j.dump(2);
    } catch (...) {
    }
}

bool isAddonFolder(const std::filesystem::path& dir) {
    std::error_code ec;
    return std::filesystem::is_directory(dir, ec) &&
           std::filesystem::is_regular_file(dir / "__init__.py", ec);
}

void ensureAddonsOnPath() {
    py::module_ sys = py::module_::import("sys");
    py::list path = sys.attr("path");
    const std::string dir = addonsDir().string();
    for (auto item : path) {
        if (py::cast<std::string>(item) == dir) return;
    }
    path.insert(0, dir);
}

// import + register(); does not touch persisted state (callers own that).
bool registerAddon(const std::string& name, std::string& error) {
    try {
        ensureAddonsOnPath();
        py::module_ mod = py::module_::import(name.c_str());
        if (py::hasattr(mod, "register")) mod.attr("register")();
        g_loaded_addons.insert(name);
        return true;
    } catch (const py::error_already_set& e) {
        error = e.what();
        return false;
    } catch (const std::exception& e) {
        error = e.what();
        return false;
    }
}

bool unregisterAddon(const std::string& name, std::string& error) {
    try {
        if (g_loaded_addons.count(name)) {
            py::module_ mod = py::module_::import(name.c_str());  // sys.modules cache hit
            if (py::hasattr(mod, "unregister")) mod.attr("unregister")();
            g_loaded_addons.erase(name);
        }
        return true;
    } catch (const py::error_already_set& e) {
        error = e.what();
        return false;
    } catch (const std::exception& e) {
        error = e.what();
        return false;
    }
}

void readBlInfo(py::module_& mod, AddonInfo& info) {
    if (!py::hasattr(mod, "bl_info")) return;
    try {
        py::dict bl = mod.attr("bl_info");
        if (bl.contains("name"))        info.display_name = py::cast<std::string>(bl["name"]);
        if (bl.contains("description")) info.description  = py::cast<std::string>(bl["description"]);
        if (bl.contains("version"))     info.version      = py::cast<std::string>(py::str(bl["version"]));
    } catch (...) {
    }
}

} // namespace

std::vector<AddonInfo> listAddons() {
    std::vector<AddonInfo> result;
    if (!g_interpreter) return result;
    std::error_code ec;
    const auto dir = addonsDir();
    if (!std::filesystem::is_directory(dir, ec)) return result;

    for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
        if (!isAddonFolder(entry.path())) continue;
        const std::string name = entry.path().filename().string();
        AddonInfo info;
        info.module_name = name;
        info.display_name = name;
        info.enabled = g_enabled_addons.count(name) != 0;
        info.loaded  = g_loaded_addons.count(name) != 0;
        // Only read bl_info from already-loaded modules: importing an unloaded one
        // would run its top-level code as a side effect. Its display name shows as
        // the folder name until enabled.
        if (info.loaded) {
            try {
                py::module_ mod = py::module_::import(name.c_str());
                readBlInfo(mod, info);
            } catch (...) {
            }
        }
        result.push_back(std::move(info));
    }
    std::sort(result.begin(), result.end(),
              [](const AddonInfo& a, const AddonInfo& b) { return a.module_name < b.module_name; });
    return result;
}

bool enableAddon(const std::string& module_name, std::string& error) {
    if (!g_interpreter) { error = "Python runtime is not initialized"; return false; }
    if (!registerAddon(module_name, error)) return false;
    g_enabled_addons.insert(module_name);
    saveAddonState();
    appendConsoleLine("Addon enabled: " + module_name);
    return true;
}

bool disableAddon(const std::string& module_name, std::string& error) {
    if (!g_interpreter) { error = "Python runtime is not initialized"; return false; }
    if (!unregisterAddon(module_name, error)) return false;
    g_enabled_addons.erase(module_name);
    saveAddonState();
    appendConsoleLine("Addon disabled: " + module_name);
    return true;
}

bool reloadAddon(const std::string& module_name, std::string& error) {
    if (!g_interpreter) { error = "Python runtime is not initialized"; return false; }
    try {
        std::string ignore;
        unregisterAddon(module_name, ignore);  // best-effort; a fresh addon may not be loaded yet
        ensureAddonsOnPath();
        py::module_ importlib = py::module_::import("importlib");
        py::module_ mod = py::module_::import(module_name.c_str());
        importlib.attr("reload")(mod);
        if (py::hasattr(mod, "register")) mod.attr("register")();
        g_loaded_addons.insert(module_name);
        appendConsoleLine("Addon reloaded: " + module_name);
        return true;
    } catch (const py::error_already_set& e) {
        error = e.what();
        return false;
    }
}

void loadEnabledAddons() {
    if (!g_interpreter) return;
    loadAddonState();
    if (g_enabled_addons.empty()) return;
    ensureAddonsOnPath();
    // Iterate a copy: registerAddon mutates g_loaded_addons (not g_enabled_addons,
    // but copy anyway for clarity/safety against future changes).
    const std::vector<std::string> names(g_enabled_addons.begin(), g_enabled_addons.end());
    for (const auto& name : names) {
        std::string err;
        if (registerAddon(name, err)) {
            appendConsoleLine("Addon loaded: " + name);
        } else {
            appendConsoleLine("Addon '" + name + "' failed to load: " + err);
        }
    }
}

void unloadAllAddons() noexcept {
    if (!g_interpreter) return;
    try {
        const std::vector<std::string> names(g_loaded_addons.begin(), g_loaded_addons.end());
        for (const auto& name : names) {
            std::string err;
            unregisterAddon(name, err);
        }
    } catch (...) {
        // Teardown must never throw into renderer shutdown.
    }
}

void drawAddonPanels() {
    if (!g_interpreter || g_addon_panels.empty()) return;
    py::gil_scoped_acquire gil;

    // Snapshot ids: a draw callback may register/unregister panels, which would
    // otherwise invalidate the map iterator mid-loop.
    std::vector<int> ids;
    ids.reserve(g_addon_panels.size());
    for (const auto& [id, panel] : g_addon_panels) ids.push_back(id);

    g_addon_ui_drawing = true;
    std::vector<int> closed;
    for (int id : ids) {
        auto it = g_addon_panels.find(id);
        if (it == g_addon_panels.end()) continue;  // unregistered by an earlier callback
        bool open = true;
        if (ImGui::Begin(it->second.title.c_str(), &open)) {
            try {
                it->second.draw();
            } catch (const py::error_already_set& e) {
                ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "panel draw error (see console)");
                appendConsoleLine(e.what());
            }
        }
        ImGui::End();
        if (!open) closed.push_back(id);
    }
    g_addon_ui_drawing = false;

    // Window close button removes the panel; the addon re-registers to show it again.
    for (int id : closed) g_addon_panels.erase(id);
}

std::vector<std::string> getModuleAttributes(const std::string& module_path) {
    std::vector<std::string> attrs;
    if (!g_interpreter) return attrs;
    try {
        py::object obj;
        if (module_path.empty() || module_path == "rt") {
            obj = py::module_::import("rt");
        } else {
            std::string sub = module_path;
            if (sub.rfind("rt.", 0) == 0) {
                sub = sub.substr(3);
            }
            obj = py::module_::import("rt");
            size_t pos = 0;
            std::string token;
            std::string s = sub;
            while ((pos = s.find('.')) != std::string::npos) {
                token = s.substr(0, pos);
                if (!token.empty()) obj = obj.attr(token.c_str());
                s.erase(0, pos + 1);
            }
            if (!s.empty()) obj = obj.attr(s.c_str());
        }

        py::list dir_list = py::module_::import("builtins").attr("dir")(obj);
        for (auto item : dir_list) {
            std::string s = py::cast<std::string>(item);
            if (s.rfind("__", 0) != 0) {
                attrs.push_back(s);
            }
        }
    } catch (...) {
        // Safe fallback in case of module resolve error
    }
    return attrs;
}

std::string getAttributeDocstring(const std::string& module_path, const std::string& attr) {
    if (!g_interpreter) return "";
    try {
        py::object obj;
        if (module_path.empty() || module_path == "rt") {
            obj = py::module_::import("rt");
        } else {
            std::string sub = module_path;
            if (sub.rfind("rt.", 0) == 0) {
                sub = sub.substr(3);
            }
            obj = py::module_::import("rt");
            size_t pos = 0;
            std::string token;
            std::string s = sub;
            while ((pos = s.find('.')) != std::string::npos) {
                token = s.substr(0, pos);
                if (!token.empty()) obj = obj.attr(token.c_str());
                s.erase(0, pos + 1);
            }
            if (!s.empty()) obj = obj.attr(s.c_str());
        }

        py::object target = obj.attr(attr.c_str());
        if (py::hasattr(target, "__doc__") && !target.attr("__doc__").is_none()) {
            return py::cast<std::string>(target.attr("__doc__"));
        }
    } catch (...) {
        // Safe fallback in case of doc resolution error
    }
    return "";
}

std::string getAttributeType(const std::string& module_path, const std::string& attr) {
    if (!g_interpreter) return "unknown";
    try {
        py::object obj;
        if (module_path.empty() || module_path == "rt") {
            obj = py::module_::import("rt");
        } else {
            std::string sub = module_path;
            if (sub.rfind("rt.", 0) == 0) {
                sub = sub.substr(3);
            }
            obj = py::module_::import("rt");
            size_t pos = 0;
            std::string token;
            std::string s = sub;
            while ((pos = s.find('.')) != std::string::npos) {
                token = s.substr(0, pos);
                if (!token.empty()) obj = obj.attr(token.c_str());
                s.erase(0, pos + 1);
            }
            if (!s.empty()) obj = obj.attr(s.c_str());
        }

        py::object target = obj.attr(attr.c_str());
        py::object builtins = py::module_::import("builtins");

        bool is_callable = py::cast<bool>(builtins.attr("callable")(target));
        if (is_callable) {
            bool is_class = py::cast<bool>(builtins.attr("isinstance")(target, builtins.attr("type")));
            if (is_class) {
                return "class";
            }
            return "method";
        }

        std::string type_str = py::cast<std::string>(builtins.attr("type")(target).attr("__name__"));
        if (type_str == "module") {
            return "module";
        }

        return "property";
    } catch (...) {
    }
    return "unknown";
}

} // namespace rtpython
