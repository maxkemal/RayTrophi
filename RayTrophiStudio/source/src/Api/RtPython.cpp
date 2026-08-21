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
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <vector>

#ifdef _WIN32
#include <cstdlib>
#endif

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "Api/RtApi.h"
#include "Api/RtUi.h"   // Faz 4b: addon panels + regions (implementation in RtUi.cpp)
#include "RtUiBindings.h"
#include "RtPyCommon.h" // shared binding helpers + rtpy::registerSceneBindings
#include "RtPythonFracture.h"
#include "RtPythonPressure.h"
#include "RtPythonDebris.h"
#include "RtPythonMassTransfer.h"
#include "RtPythonTemplates.h"
#include "RtPythonAgent.h"

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

// requireResult / vec3ToPython / vec3FromPython now live in RtPyCommon.h so
// RtPyScene.cpp can share them; they stay at global scope, so the unqualified
// call sites below are unchanged.

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

    rtpy::registerTemplateBindings(module);
    rtpy::registerAgentBindings(module);

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
    // ★★★ The pose a SOLVER produced, not the one an author typed. The rigid
    // solver bakes its motion into the MESH VERTICES and never touches the
    // transform handle, so get_transform keeps reporting the spawn pose for a
    // body that has been falling for a thousand steps - successfully, with
    // plausible numbers. Any physics check (free fall, buoyancy draft,
    // restitution) has to read this one, AND check its 'simulated' flag.
    scene.def("get_world_transform", [](const std::string& name) -> py::dict {
        Matrix4x4 m;
        bool simulated = false;
        requireResult(rtapi::getObjectWorldTransform(name, m, &simulated));
        Vec3 t(0.0f), r(0.0f), s(1.0f);
        m.decompose(t, r, s);
        py::dict d;
        d["matrix"] = matrixToPython(m);
        d["translation"] = vec3ToPython(t);
        d["rotation"] = vec3ToPython(r);
        d["scale"] = vec3ToPython(s);
        d["simulated"] = simulated;
        return d;
    }, py::arg("name"),
       "The simulated world pose: the rigid solver's motion composed onto the "
       "authored transform. Use this to measure physics; get_transform answers "
       "what was authored and does not move when a solver runs. The returned "
       "dict carries 'simulated': False means no solver has contributed, so "
       "these are merely the authored numbers - a check must not read that as "
       "'the body did not move'.");

    scene.def("get_transform", [](const std::string& name) -> py::dict {
        Matrix4x4 m;
        requireResult(rtapi::getObjectTransform(name, m));
        Vec3 t(0.0f), r(0.0f), s(1.0f);
        m.decompose(t, r, s);
        py::dict d;
        d["matrix"] = matrixToPython(m);
        d["translation"] = vec3ToPython(t);
        // Reported alongside the matrix so a script can ask which way an object
        // actually points instead of assuming the axis it was authored on. A
        // test that hardcodes "the beam runs along X" is measuring its own
        // assumption the moment anyone rescales the scene.
        d["rotation"] = vec3ToPython(r);
        d["scale"] = vec3ToPython(s);
        return d;
    }, py::arg("name"));

    scene.def("set_transform", [](const std::string& name, const py::handle& translation, const py::handle& rotation, const py::handle& scale, const py::handle& matrix) {
        if (!matrix.is_none()) {
            requireResult(rtapi::setObjectTransform(name, matrixFromPython(matrix)));
            return;
        }
        Matrix4x4 m = Matrix4x4::identity();
        requireResult(rtapi::getObjectTransform(name, m));
        // ★ `rotation` and `scale` used to be ACCEPTED AND DROPPED: only the
        // translation column was written, so `set_transform(o, scale=(3,.3,.3))`
        // returned success and left a unit cube. Silent no-op parameters are
        // worse than missing ones — every scene built by a script that passed
        // them was quietly the wrong shape, and the tests written against those
        // scenes measured geometry that was never there.
        if (rotation.is_none() && scale.is_none()) {
            // Translation only: patch the column in place rather than round-trip
            // through Euler, so a matrix carrying shear (or any basis this
            // decomposition cannot name) survives a move untouched.
            if (!translation.is_none()) {
                Vec3 t = vec3FromPython(translation);
                m.m[0][3] = t.x; m.m[1][3] = t.y; m.m[2][3] = t.z;
            }
            requireResult(rtapi::setObjectTransform(name, m));
            return;
        }
        Vec3 t(0.0f), r(0.0f), s(1.0f);
        m.decompose(t, r, s);
        if (!translation.is_none()) t = vec3FromPython(translation);
        if (!rotation.is_none())    r = vec3FromPython(rotation);
        if (!scale.is_none())       s = vec3FromPython(scale);
        requireResult(rtapi::setObjectTransform(
            name, Matrix4x4::composeTRS(t, r, s)));
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

    scatter.def("list_assets", []() -> py::list {
        std::vector<rtapi::FoliageAssetInfo> assets;
        requireResult(rtapi::listFoliageAssets(assets));
        py::list result;
        for (const auto& a : assets) {
            py::dict d;
            d["name"] = a.name;
            d["category"] = a.category;
            d["relative_path"] = a.relative_path;
            result.append(d);
        }
        return result;
    });

    scatter.def("add_library_source", [](const std::string& group, const std::string& relative_path) {
        requireResult(rtapi::addLibraryScatterSource(group, relative_path));
    }, py::arg("group"), py::arg("relative_path"));

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
    rtpy::registerFractureBindings(physics);
    py::module_ debris = module.def_submodule("debris", "Budgeted ash and small debris LOD");
    rtpy::registerDebrisBindings(debris);
    rtpy::registerMassTransferBindings(module);
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

    // ★★★★ Advances the solver AND the playhead. It used to move only the
    // solver, so the frame loop found a rigid state that disagreed with the
    // displayed frame and reset it to rest - erasing every step while this
    // call returned success.
    physics.def("step", [](float dt) {
        requireResult(rtapi::stepPhysicsSimulation(dt));
    }, py::arg("dt") = 0.0166667f,
       "Advance the physics solver by dt and move the playhead with it. Claims "
       "the timeline until the user scrubs, plays or stops; check "
       "rt.sim.control_state()['epoch'] around a measurement to find out "
       "whether that happened.");

    // rt.sim - control state shared by every solver.
    py::module_ sim = module.def_submodule(
        "sim", "Simulation control state: who is driving the solvers.");
    sim.def("control_state", []() -> py::dict {
        rtapi::SimControlStateInfo info;
        requireResult(rtapi::getSimControlState(info));
        py::dict d;
        d["epoch"] = info.epoch;
        d["driver"] = info.driver;
        d["script_driving"] = info.script_driving;
        d["script_sim_seconds"] = info.script_sim_seconds;
        d["frame"] = info.frame;
        d["playing"] = info.playing;
        return d;
    },
    "Who last moved the solvers, and an epoch that changes whenever anything "
    "did. Read it before and after a measurement: if the epoch moved, the "
    "solvers were re-posed under you and the numbers are not a physics "
    "result. Without this a reverted pose and a body that never moved read "
    "exactly the same.");

    physics.def("set_gravity", [](const py::handle& gravity) {
        requireResult(rtapi::setPhysicsGravity(vec3FromPython(gravity)));
    }, py::arg("gravity") = py::make_tuple(0.0f, -9.81f, 0.0f));

    physics.def("get_gravity", []() -> py::tuple {
        Vec3 g(0, -9.81f, 0);
        requireResult(rtapi::getPhysicsGravity(g));
        return vec3ToPython(g);
    });

    // ── Force fields (Faz 5.6a) ─────────────────────────────────────────
    // One field drives gas + particles + cloth + rigid bodies + APIC liquid.
    // The facade takes a whole struct; set_param turns kwargs into a partial
    // edit by reading the field first and writing back only what changed.
    py::module_ forcefield = module.def_submodule(
        "forcefield", "Universal force fields shared by every simulation family");

    auto forceFieldToDict = [](const rtapi::ForceFieldInfo& info) -> py::dict {
        py::dict d;
        d["id"] = info.id;
        d["name"] = info.name;
        d["type"] = info.type;
        d["shape"] = info.shape;
        d["falloff"] = info.falloff;
        d["enabled"] = info.enabled;
        d["visible"] = info.visible;
        d["position"] = vec3ToPython(info.position);
        d["rotation"] = vec3ToPython(info.rotation);
        d["scale"] = vec3ToPython(info.scale);
        d["direction"] = vec3ToPython(info.direction);
        d["axis"] = vec3ToPython(info.axis);
        d["strength"] = info.strength;
        d["falloff_radius"] = info.falloff_radius;
        d["inner_radius"] = info.inner_radius;
        d["use_noise"] = info.use_noise;
        d["noise_octaves"] = info.noise_octaves;
        d["noise_seed"] = info.noise_seed;
        d["noise_frequency"] = info.noise_frequency;
        d["noise_lacunarity"] = info.noise_lacunarity;
        d["noise_persistence"] = info.noise_persistence;
        d["noise_amplitude"] = info.noise_amplitude;
        d["noise_speed"] = info.noise_speed;
        d["inward_force"] = info.inward_force;
        d["upward_force"] = info.upward_force;
        d["linear_drag"] = info.linear_drag;
        d["quadratic_drag"] = info.quadratic_drag;
        d["thermal_delta_kelvin"] = info.thermal_delta_kelvin;
        d["fluid_surface_drag"] = info.fluid_surface_drag;
        d["fluid_drag_coupling"] = info.fluid_drag_coupling;
        d["fluid_surface_depth"] = info.fluid_surface_depth;
        d["fluid_curl_detail"] = info.fluid_curl_detail;
        d["start_frame"] = info.start_frame;
        d["end_frame"] = info.end_frame;
        d["phase"] = info.phase;
        d["affects_gas"] = info.affects_gas;
        d["affects_particles"] = info.affects_particles;
        d["affects_cloth"] = info.affects_cloth;
        d["affects_rigidbody"] = info.affects_rigidbody;
        d["affects_fluid"] = info.affects_fluid;
        return d;
    };

    forcefield.def("types", []() { return rtapi::forceFieldTypes(); });

    forcefield.def("list", [forceFieldToDict]() -> py::list {
        py::list out;
        for (const rtapi::ForceFieldInfo& info : rtapi::listForceFields())
            out.append(forceFieldToDict(info));
        return out;
    });

    forcefield.def("get", [forceFieldToDict](const std::string& field) {
        rtapi::ForceFieldInfo info;
        requireResult(rtapi::getForceField(field, info));
        return forceFieldToDict(info);
    }, py::arg("field"));

    forcefield.def("create", [forceFieldToDict](const std::string& type, const std::string& name) {
        rtapi::ForceFieldInfo info;
        requireResult(rtapi::createForceField(type, name, info));
        return forceFieldToDict(info);
    }, py::arg("type") = "wind", py::arg("name") = "");

    forcefield.def("remove", [](const std::string& field) {
        requireResult(rtapi::removeForceField(field));
    }, py::arg("field"));

    forcefield.def("set_param", [](const std::string& field, const py::kwargs& kwargs) {
        rtapi::ForceFieldInfo info;
        requireResult(rtapi::getForceField(field, info));
        auto str = [&](const char* key, std::string& target) {
            if (kwargs.contains(key)) target = py::cast<std::string>(kwargs[key]);
        };
        auto flt = [&](const char* key, float& target) {
            if (kwargs.contains(key)) target = py::cast<float>(kwargs[key]);
        };
        auto integer = [&](const char* key, int& target) {
            if (kwargs.contains(key)) target = py::cast<int>(kwargs[key]);
        };
        auto boolean = [&](const char* key, bool& target) {
            if (kwargs.contains(key)) target = py::cast<bool>(kwargs[key]);
        };
        auto vector = [&](const char* key, Vec3& target) {
            if (kwargs.contains(key)) target = vec3FromPython(kwargs[key]);
        };
        str("name", info.name); str("type", info.type);
        str("shape", info.shape); str("falloff", info.falloff);
        boolean("enabled", info.enabled); boolean("visible", info.visible);
        vector("position", info.position); vector("rotation", info.rotation);
        vector("scale", info.scale); vector("direction", info.direction);
        vector("axis", info.axis);
        flt("strength", info.strength); flt("falloff_radius", info.falloff_radius);
        flt("inner_radius", info.inner_radius);
        boolean("use_noise", info.use_noise);
        integer("noise_octaves", info.noise_octaves); integer("noise_seed", info.noise_seed);
        flt("noise_frequency", info.noise_frequency);
        flt("noise_lacunarity", info.noise_lacunarity);
        flt("noise_persistence", info.noise_persistence);
        flt("noise_amplitude", info.noise_amplitude);
        flt("noise_speed", info.noise_speed);
        flt("inward_force", info.inward_force); flt("upward_force", info.upward_force);
        flt("linear_drag", info.linear_drag); flt("quadratic_drag", info.quadratic_drag);
        flt("thermal_delta_kelvin", info.thermal_delta_kelvin);
        boolean("fluid_surface_drag", info.fluid_surface_drag);
        flt("fluid_drag_coupling", info.fluid_drag_coupling);
        flt("fluid_surface_depth", info.fluid_surface_depth);
        flt("fluid_curl_detail", info.fluid_curl_detail);
        flt("start_frame", info.start_frame); flt("end_frame", info.end_frame);
        flt("phase", info.phase);
        boolean("affects_gas", info.affects_gas);
        boolean("affects_particles", info.affects_particles);
        boolean("affects_cloth", info.affects_cloth);
        boolean("affects_rigidbody", info.affects_rigidbody);
        boolean("affects_fluid", info.affects_fluid);
        requireResult(rtapi::updateForceField(field, info));
    }, py::arg("field"));

    forcefield.def("evaluate", [](const py::handle& position, float time,
                                  const py::handle& velocity) -> py::tuple {
        Vec3 force;
        requireResult(rtapi::evaluateForceFields(vec3FromPython(position), time,
                                                 vec3FromPython(velocity), force));
        return vec3ToPython(force);
    }, py::arg("position"), py::arg("time") = 0.0f,
       py::arg("velocity") = py::make_tuple(0.0f, 0.0f, 0.0f));

    // ── Particle systems (Faz 5.6b) ─────────────────────────────────────
    // Emitters + solver settings + stats. Particle colliders and grid domains
    // live on the SAME runtime and are scripted under rt.fluid, not here.
    py::module_ particle = module.def_submodule(
        "particle", "Particle emitters, solver settings and live statistics");

    auto emitterToDict = [](const rtapi::ParticleEmitterInfo& info) -> py::dict {
        py::dict d;
        d["index"] = info.index;
        d["name"] = info.name;
        d["source_mode"] = info.source_mode;
        d["spawn_mode"] = info.spawn_mode;
        d["source_name"] = info.source_name;
        d["enabled"] = info.enabled;
        d["point"] = vec3ToPython(info.point);
        d["local_offset"] = vec3ToPython(info.local_offset);
        d["direction"] = vec3ToPython(info.direction);
        d["surface_offset"] = info.surface_offset;
        d["rate_per_second"] = info.rate_per_second;
        d["burst_count"] = info.burst_count;
        d["speed"] = info.speed;
        d["spread"] = info.spread;
        d["lifetime_seconds"] = info.lifetime_seconds;
        d["mass"] = info.mass;
        d["start_size"] = info.start_size;
        d["end_size"] = info.end_size;
        d["size_jitter"] = info.size_jitter;
        d["start_opacity"] = info.start_opacity;
        d["end_opacity"] = info.end_opacity;
        d["start_color"] = vec3ToPython(info.start_color);
        d["end_color"] = vec3ToPython(info.end_color);
        d["angular_velocity"] = info.angular_velocity;
        d["angular_jitter"] = info.angular_jitter;
        d["seed"] = info.seed;
        d["parent_object"] = info.parent_object;
        d["velocity_space"] = info.velocity_space;
        d["inherit_velocity"] = info.inherit_velocity;
        d["override_grid_deposit"] = info.override_grid_deposit;
        d["grid_density_deposit"] = info.grid_density_deposit;
        d["grid_temperature_deposit"] = info.grid_temperature_deposit;
        d["grid_fuel_deposit"] = info.grid_fuel_deposit;
        return d;
    };

    // Shared by add_emitter (patches a default desc) and set_emitter (patches
    // the live one), so both accept exactly the same keyword set.
    auto patchEmitter = [](const py::kwargs& kwargs, rtapi::ParticleEmitterInfo& info) {
        auto str = [&](const char* key, std::string& target) {
            if (kwargs.contains(key)) target = py::cast<std::string>(kwargs[key]);
        };
        auto flt = [&](const char* key, float& target) {
            if (kwargs.contains(key)) target = py::cast<float>(kwargs[key]);
        };
        auto boolean = [&](const char* key, bool& target) {
            if (kwargs.contains(key)) target = py::cast<bool>(kwargs[key]);
        };
        auto vector = [&](const char* key, Vec3& target) {
            if (kwargs.contains(key)) target = vec3FromPython(kwargs[key]);
        };
        str("name", info.name); str("source_mode", info.source_mode);
        str("spawn_mode", info.spawn_mode); str("source_name", info.source_name);
        boolean("enabled", info.enabled);
        vector("point", info.point); vector("local_offset", info.local_offset);
        vector("direction", info.direction);
        flt("surface_offset", info.surface_offset);
        flt("rate_per_second", info.rate_per_second);
        if (kwargs.contains("burst_count")) info.burst_count = py::cast<int>(kwargs["burst_count"]);
        flt("speed", info.speed); flt("spread", info.spread);
        flt("lifetime_seconds", info.lifetime_seconds); flt("mass", info.mass);
        flt("start_size", info.start_size); flt("end_size", info.end_size);
        flt("size_jitter", info.size_jitter);
        flt("start_opacity", info.start_opacity); flt("end_opacity", info.end_opacity);
        vector("start_color", info.start_color); vector("end_color", info.end_color);
        flt("angular_velocity", info.angular_velocity);
        flt("angular_jitter", info.angular_jitter);
        if (kwargs.contains("seed"))
            info.seed = py::cast<unsigned int>(kwargs["seed"]);
        str("parent_object", info.parent_object);
        str("velocity_space", info.velocity_space);
        flt("inherit_velocity", info.inherit_velocity);
        boolean("override_grid_deposit", info.override_grid_deposit);
        flt("grid_density_deposit", info.grid_density_deposit);
        flt("grid_temperature_deposit", info.grid_temperature_deposit);
        flt("grid_fuel_deposit", info.grid_fuel_deposit);
    };

    particle.def("emitters", [emitterToDict]() -> py::list {
        py::list out;
        for (const rtapi::ParticleEmitterInfo& info : rtapi::listParticleEmitters())
            out.append(emitterToDict(info));
        return out;
    });

    particle.def("get_emitter", [emitterToDict](const std::string& emitter) {
        rtapi::ParticleEmitterInfo info;
        requireResult(rtapi::getParticleEmitter(emitter, info));
        return emitterToDict(info);
    }, py::arg("emitter"));

    particle.def("add_emitter", [emitterToDict, patchEmitter](const py::kwargs& kwargs) {
        rtapi::ParticleEmitterInfo info;   // facade defaults
        patchEmitter(kwargs, info);
        rtapi::ParticleEmitterInfo created;
        requireResult(rtapi::addParticleEmitter(info, created));
        return emitterToDict(created);
    });

    particle.def("set_emitter", [patchEmitter](const std::string& emitter, const py::kwargs& kwargs) {
        rtapi::ParticleEmitterInfo info;
        requireResult(rtapi::getParticleEmitter(emitter, info));
        patchEmitter(kwargs, info);
        requireResult(rtapi::updateParticleEmitter(emitter, info));
    }, py::arg("emitter"));

    // Timeline keys. Only the channels actually passed are keyed, so
    //   rt.particle.key_emitter("Embers", 120, enabled=False)
    // keys ONLY the switch and leaves rate/speed free to be keyed separately.
    particle.def("key_emitter", [](const std::string& emitter, int frame,
                                   const py::kwargs& kw) {
        rtapi::ParticleEmitterKey key;
        key.frame = frame;
        if (kw.contains("enabled")) { key.has_enabled = true; key.enabled = py::cast<bool>(kw["enabled"]); }
        if (kw.contains("rate_per_second")) { key.has_rate = true; key.rate_per_second = py::cast<float>(kw["rate_per_second"]); }
        if (kw.contains("speed")) { key.has_speed = true; key.speed = py::cast<float>(kw["speed"]); }
        if (kw.contains("spread")) { key.has_spread = true; key.spread = py::cast<float>(kw["spread"]); }
        if (kw.contains("point")) { key.has_point = true; key.point = vec3FromPython(kw["point"]); }
        if (kw.contains("direction")) { key.has_direction = true; key.direction = vec3FromPython(kw["direction"]); }
        requireResult(rtapi::keyParticleEmitter(emitter, key));
    }, py::arg("emitter"), py::arg("frame"));
    particle.def("clear_emitter_key", [](const std::string& emitter, int frame) {
        requireResult(rtapi::clearParticleEmitterKey(emitter, frame));
    }, py::arg("emitter"), py::arg("frame"));
    particle.def("remove_emitter", [](const std::string& emitter) {
        requireResult(rtapi::removeParticleEmitter(emitter));
    }, py::arg("emitter"));

    particle.def("clear_emitters", []() { requireResult(rtapi::clearParticleEmitters()); });

    // Systems, not particles. clear_emitters/flow_source.remove/collider.remove
    // only ever touch the ACTIVE system; these reach the rest.
    particle.def("list_systems", []() -> py::list {
        std::vector<rtapi::ParticleSystemInfo> systems;
        requireResult(rtapi::listParticleSystems(systems));
        py::list out;
        for (const auto& s : systems) {
            py::dict d;
            d["index"] = s.index;
            d["id"] = s.id;
            d["name"] = s.name;
            d["active"] = s.active;
            d["domain_count"] = s.domain_count;
            d["flow_source_count"] = s.flow_source_count;
            d["emitter_count"] = s.emitter_count;
            d["collider_count"] = s.collider_count;
            out.append(d);
        }
        return out;
    });

    particle.def("clear_systems", []() { requireResult(rtapi::clearParticleSystems()); });

    particle.def("get_physics", []() -> py::dict {
        rtapi::ParticlePhysicsInfo info;
        requireResult(rtapi::getParticlePhysics(info));
        py::dict d;
        d["mode"] = info.mode;
        d["quality"] = info.quality;
        d["particle_radius"] = info.particle_radius;
        d["self_collision_enabled"] = info.self_collision_enabled;
        d["solver_iterations"] = info.solver_iterations;
        d["max_neighbors_per_particle"] = info.max_neighbors_per_particle;
        d["viscosity"] = info.viscosity;
        d["cohesion"] = info.cohesion;
        d["pressure_stiffness"] = info.pressure_stiffness;
        d["rest_density"] = info.rest_density;
        d["buoyancy"] = info.buoyancy;
        d["gravity_scale"] = info.gravity_scale;
        d["vorticity"] = info.vorticity;
        d["grid_density_deposit"] = info.grid_density_deposit;
        d["grid_temperature_deposit"] = info.grid_temperature_deposit;
        d["grid_fuel_deposit"] = info.grid_fuel_deposit;
        d["grid_deposit_fade_with_age"] = info.grid_deposit_fade_with_age;
        return d;
    });

    particle.def("set_physics", [](const py::kwargs& kwargs) {
        rtapi::ParticlePhysicsInfo info;
        requireResult(rtapi::getParticlePhysics(info));
        auto str = [&](const char* key, std::string& target) {
            if (kwargs.contains(key)) target = py::cast<std::string>(kwargs[key]);
        };
        auto flt = [&](const char* key, float& target) {
            if (kwargs.contains(key)) target = py::cast<float>(kwargs[key]);
        };
        auto integer = [&](const char* key, int& target) {
            if (kwargs.contains(key)) target = py::cast<int>(kwargs[key]);
        };
        auto boolean = [&](const char* key, bool& target) {
            if (kwargs.contains(key)) target = py::cast<bool>(kwargs[key]);
        };
        str("mode", info.mode); str("quality", info.quality);
        flt("particle_radius", info.particle_radius);
        boolean("self_collision_enabled", info.self_collision_enabled);
        integer("solver_iterations", info.solver_iterations);
        integer("max_neighbors_per_particle", info.max_neighbors_per_particle);
        flt("viscosity", info.viscosity); flt("cohesion", info.cohesion);
        flt("pressure_stiffness", info.pressure_stiffness);
        flt("rest_density", info.rest_density); flt("buoyancy", info.buoyancy);
        flt("gravity_scale", info.gravity_scale); flt("vorticity", info.vorticity);
        flt("grid_density_deposit", info.grid_density_deposit);
        flt("grid_temperature_deposit", info.grid_temperature_deposit);
        flt("grid_fuel_deposit", info.grid_fuel_deposit);
        boolean("grid_deposit_fade_with_age", info.grid_deposit_fade_with_age);
        requireResult(rtapi::updateParticlePhysics(info));
    });

    particle.def("stats", []() -> py::dict {
        rtapi::ParticleStatsInfo info;
        requireResult(rtapi::getParticleStats(info));
        py::dict d;
        d["alive_count"] = info.alive_count;
        d["capacity"] = info.capacity;
        d["emitter_count"] = info.emitter_count;
        d["collider_count"] = info.collider_count;
        d["domain_count"] = info.domain_count;
        d["total_ms"] = info.total_ms;
        d["emit_ms"] = info.emit_ms;
        d["integrate_ms"] = info.integrate_ms;
        d["self_collision_ms"] = info.self_collision_ms;
        d["grid_domain_ms"] = info.grid_domain_ms;
        return d;
    });

    particle.def("spawn", [](const py::handle& position, const py::handle& velocity,
                             float lifetime_seconds, float mass, float size) {
        int index = -1;
        requireResult(rtapi::spawnParticle(vec3FromPython(position), vec3FromPython(velocity),
                                           lifetime_seconds, mass, size, index));
        return index;
    }, py::arg("position") = py::make_tuple(0.0f, 1.0f, 0.0f),
       py::arg("velocity") = py::make_tuple(0.0f, 0.0f, 0.0f),
       py::arg("lifetime_seconds") = 5.0f, py::arg("mass") = 1.0f, py::arg("size") = 0.05f);

    particle.def("clear", []() { requireResult(rtapi::clearParticles()); });

    particle.def("step", [](float dt) {
        requireResult(rtapi::stepParticleSimulation(dt));
    }, py::arg("dt") = 0.0166667f);

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
        d["live_state"] = info.live_state;
        d["render_mode"] = info.render_mode;
        d["backend"] = info.backend;
        d["boundary"] = info.boundary;
        d["preset"] = info.preset;
        d["kinematic_viscosity"] = info.kinematic_viscosity;
        d["viscosity_sweeps"] = info.viscosity_sweeps;
        d["viscosity_wall_slip"] = info.viscosity_wall_slip;
        d["granular_enabled"] = info.granular_enabled;
        d["granular_friction_angle"] = info.granular_friction_angle_degrees;
        d["granular_cohesion"] = info.granular_cohesion;
        d["granular_dilatancy"] = info.granular_dilatancy_degrees;
        d["granular_young_modulus"] = info.granular_young_modulus;
        d["granular_poisson_ratio"] = info.granular_poisson_ratio;
        d["granular_tensile_cutoff"] = info.granular_tensile_cutoff;
        d["granular_hardening"] = info.granular_hardening;
        d["granular_fracture_strain"] = info.granular_fracture_strain;
        d["granular_damage_rate"] = info.granular_damage_rate;
        d["granular_healing_rate"] = info.granular_healing_rate;
        d["granular_rebonding"] = info.granular_rebonding;
        d["granular_max_solver_substeps"] = info.granular_max_solver_substeps;
        d["granular_softening_temperature"] = info.granular_softening_temperature;
        d["granular_softening_range"] = info.granular_softening_range;
        d["granular_residual_strength"] = info.granular_residual_strength;
        d["granular_tack_peak"] = info.granular_tack_peak;
        d["granular_thermal_conductivity"] = info.granular_thermal_conductivity;
        d["granular_yielded"] = info.granular_yielded_particles;
        d["granular_detached"] = info.granular_detached_particles;
        d["granular_invalid"] = info.granular_invalid_particles;
        d["granular_sleeping"] = info.granular_sleeping_particles;
        d["granular_damaged"] = info.granular_damaged_particles;
        d["granular_damage_over_10"] = info.granular_damage_over_10_particles;
        d["granular_damage_over_50"] = info.granular_damage_over_50_particles;
        d["granular_damage_over_90"] = info.granular_damage_over_90_particles;
        d["granular_max_yield"] = info.granular_max_yield;
        d["granular_max_plastic"] = info.granular_max_plastic_increment;
        d["granular_max_accumulated_plastic"] = info.granular_max_accumulated_plastic;
        d["granular_mean_accumulated_plastic"] = info.granular_mean_accumulated_plastic;
        d["granular_max_fracture_history"] = info.granular_max_fracture_history;
        d["granular_mean_fracture_history"] = info.granular_mean_fracture_history;
        d["granular_max_damage"] = info.granular_max_damage;
        d["granular_mean_damage"] = info.granular_mean_damage;
        d["granular_requested_young_modulus"] = info.granular_requested_young_modulus;
        d["granular_effective_young_modulus"] = info.granular_effective_young_modulus;
        d["granular_required_substeps"] = info.granular_required_substeps;
        d["granular_solver_substeps"] = info.granular_solver_substeps;
        d["granular_stiffness_capped"] = info.granular_stiffness_capped;
        d["granular_wave_substeps"] = info.granular_wave_substeps;
        d["granular_strain_substeps"] = info.granular_strain_substeps;
        d["granular_strain_rate"] = info.granular_strain_rate;
        d["granular_strain_limited"] = info.granular_strain_limited_particles;
        d["granular_compaction_capped"] = info.granular_compaction_capped_particles;
        d["granular_min_softening"] = info.granular_min_softening;
        d["granular_softened"] = info.granular_softened_particles;
        d["granular_overburden_pressure"] = info.granular_overburden_pressure;
        d["granular_young_modulus_for_load"] = info.granular_young_modulus_for_load;
        d["granular_stiffness_below_load"] = info.granular_stiffness_below_load;
        d["surface_material"] = info.surface_material;
        d["splat_material"] = info.splat_material;
        d["surface_offset_voxels"] = info.surface_offset_voxels;
        // Porosity was readable over IPC but not from Python — a script could
        // set it and never read it back. Closed here rather than in a separate
        // pass: the file was already open for the coordinate fields below.
        d["pore_amount"] = info.pore_amount;
        d["pore_scale"] = info.pore_scale;
        d["pore_detail"] = info.pore_detail;
        d["coord_space"] = info.coord_space == 1 ? "domain"
                         : info.coord_space == 2 ? "world" : "material";
        d["uvw_available"] = info.uvw_available;
        d["uvw_dim"] = py::make_tuple(info.uvw_dim[0], info.uvw_dim[1], info.uvw_dim[2]);
        d["uvw_origin"] = py::make_tuple(info.uvw_origin[0], info.uvw_origin[1],
                                         info.uvw_origin[2]);
        d["uvw_voxel"] = info.uvw_voxel;
        d["uvw_refresh_period"] = info.uvw_refresh_period;
        {
            py::list subs;
            for (const auto& s : info.substances) {
                py::dict e;
                e["name"] = s.name; e["tag"] = s.tag; e["particles"] = s.particles;
                subs.append(e);
            }
            d["substances"] = subs;
            py::list binds;
            for (const auto& b : info.substance_materials) {
                py::dict e;
                e["substance"] = b.substance; e["material_id"] = b.material_id;
                e["material"] = b.material; e["representation"] = b.representation;
                // "inherit" already resolved against the domain mode: what is
                // actually DRAWN. Assert on this, not on the authored value.
                e["effective_representation"] = b.effective_representation;
                // As authored: a negative viscosity is the inherit sentinel.
                e["kinematic_viscosity"] = b.kinematic_viscosity;
                e["miscibility"] = b.miscibility;
                // State of matter, a separate axis from representation.
                e["phase"] = b.phase;
                binds.append(e);
            }
            d["substance_materials"] = binds;
        }
        d["uvw_drift"] = info.uvw_drift;
        // Measured on the last simulated step (see FluidDomainInfo).
        d["solid_phase_particles"] = info.solid_phase_particles;
        d["solid_phase_cells"] = info.solid_phase_cells;
        d["sealed_pockets"] = info.sealed_pockets;
        d["sealed_pocket_cells"] = info.sealed_pocket_cells;
        d["sealed_pockets_measured"] = info.sealed_pockets_measured;
        d["interior_fluid_cells"] = info.interior_fluid_cells;
        d["reseed_added_particles"] = info.reseed_added_particles;
        d["reseed_removed_particles"] = info.reseed_removed_particles;
        d["solid_phase"] = info.solid_phase_enabled;
        d["solid_phase_fill"] = info.solid_phase_fill;
        d["uvw_particles"] = info.uvw_particles;
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

    // Enumerates liquid AND gas domains — a script that only knows the names it
    // authored itself cannot tear down what a UI preset created.
    fluid.def("list_domains", []() -> py::list {
        std::vector<rtapi::FluidDomainInfo> domains;
        requireResult(rtapi::listFluidDomains(domains));
        py::list out;
        for (const auto& info : domains) {
            py::dict d;
            d["id"] = info.id;
            d["name"] = info.name;
            d["type"] = info.type;
            d["domain_min"] = vec3ToPython(info.domain_min);
            d["domain_max"] = vec3ToPython(info.domain_max);
            d["voxel_size"] = info.voxel_size;
            d["particle_count"] = info.particle_count;
            d["live_state"] = info.live_state;
            d["render_mode"] = info.render_mode;
            d["backend"] = info.backend;
            d["boundary"] = info.boundary;
            d["preset"] = info.preset;
            d["kinematic_viscosity"] = info.kinematic_viscosity;
            d["viscosity_sweeps"] = info.viscosity_sweeps;
            d["viscosity_wall_slip"] = info.viscosity_wall_slip;
            d["granular_enabled"] = info.granular_enabled;
            d["granular_friction_angle"] = info.granular_friction_angle_degrees;
            d["granular_cohesion"] = info.granular_cohesion;
            d["granular_dilatancy"] = info.granular_dilatancy_degrees;
            d["granular_young_modulus"] = info.granular_young_modulus;
            d["granular_poisson_ratio"] = info.granular_poisson_ratio;
            d["granular_tensile_cutoff"] = info.granular_tensile_cutoff;
            d["granular_hardening"] = info.granular_hardening;
            d["granular_fracture_strain"] = info.granular_fracture_strain;
            d["granular_damage_rate"] = info.granular_damage_rate;
            d["granular_healing_rate"] = info.granular_healing_rate;
            d["granular_rebonding"] = info.granular_rebonding;
            d["granular_max_solver_substeps"] = info.granular_max_solver_substeps;
        d["granular_softening_temperature"] = info.granular_softening_temperature;
        d["granular_softening_range"] = info.granular_softening_range;
        d["granular_residual_strength"] = info.granular_residual_strength;
        d["granular_tack_peak"] = info.granular_tack_peak;
        d["granular_thermal_conductivity"] = info.granular_thermal_conductivity;
            d["granular_yielded"] = info.granular_yielded_particles;
            d["granular_detached"] = info.granular_detached_particles;
            d["granular_invalid"] = info.granular_invalid_particles;
            d["granular_sleeping"] = info.granular_sleeping_particles;
            d["granular_damaged"] = info.granular_damaged_particles;
            d["granular_damage_over_10"] = info.granular_damage_over_10_particles;
            d["granular_damage_over_50"] = info.granular_damage_over_50_particles;
            d["granular_damage_over_90"] = info.granular_damage_over_90_particles;
            d["granular_max_yield"] = info.granular_max_yield;
            d["granular_max_plastic"] = info.granular_max_plastic_increment;
            d["granular_max_accumulated_plastic"] = info.granular_max_accumulated_plastic;
            d["granular_mean_accumulated_plastic"] = info.granular_mean_accumulated_plastic;
            d["granular_max_fracture_history"] = info.granular_max_fracture_history;
            d["granular_mean_fracture_history"] = info.granular_mean_fracture_history;
            d["granular_max_damage"] = info.granular_max_damage;
            d["granular_mean_damage"] = info.granular_mean_damage;
            d["granular_requested_young_modulus"] = info.granular_requested_young_modulus;
            d["granular_effective_young_modulus"] = info.granular_effective_young_modulus;
            d["granular_required_substeps"] = info.granular_required_substeps;
            d["granular_solver_substeps"] = info.granular_solver_substeps;
            d["granular_stiffness_capped"] = info.granular_stiffness_capped;
            d["granular_wave_substeps"] = info.granular_wave_substeps;
            d["granular_strain_substeps"] = info.granular_strain_substeps;
            d["granular_strain_rate"] = info.granular_strain_rate;
            d["granular_strain_limited"] = info.granular_strain_limited_particles;
        d["granular_compaction_capped"] = info.granular_compaction_capped_particles;
        d["granular_min_softening"] = info.granular_min_softening;
        d["granular_softened"] = info.granular_softened_particles;
            d["granular_overburden_pressure"] = info.granular_overburden_pressure;
            d["granular_young_modulus_for_load"] = info.granular_young_modulus_for_load;
            d["granular_stiffness_below_load"] = info.granular_stiffness_below_load;
            d["surface_material"] = info.surface_material;
            d["splat_material"] = info.splat_material;
            d["surface_offset_voxels"] = info.surface_offset_voxels;
            d["pore_amount"] = info.pore_amount;
            d["pore_scale"] = info.pore_scale;
            d["pore_detail"] = info.pore_detail;
            d["coord_space"] = info.coord_space == 1 ? "domain"
                             : info.coord_space == 2 ? "world" : "material";
            d["uvw_available"] = info.uvw_available;
            d["uvw_dim"] = py::make_tuple(info.uvw_dim[0], info.uvw_dim[1], info.uvw_dim[2]);
            d["uvw_origin"] = py::make_tuple(info.uvw_origin[0], info.uvw_origin[1],
                                             info.uvw_origin[2]);
            d["uvw_voxel"] = info.uvw_voxel;
            d["uvw_refresh_period"] = info.uvw_refresh_period;
            {
                py::list subs;
                for (const auto& s : info.substances) {
                    py::dict e;
                    e["name"] = s.name; e["tag"] = s.tag; e["particles"] = s.particles;
                    subs.append(e);
                }
                d["substances"] = subs;
                py::list binds;
                for (const auto& b : info.substance_materials) {
                    py::dict e;
                    e["substance"] = b.substance; e["material_id"] = b.material_id;
                    e["material"] = b.material; e["representation"] = b.representation;
                    // "inherit" already resolved against the domain mode: what is
                    // actually DRAWN. Assert on this, not on the authored value.
                    e["effective_representation"] = b.effective_representation;
                    // As authored: a negative viscosity is the inherit sentinel.
                    e["kinematic_viscosity"] = b.kinematic_viscosity;
                    e["miscibility"] = b.miscibility;
                    // State of matter, a separate axis from representation.
                    e["phase"] = b.phase;
                    binds.append(e);
                }
                d["substance_materials"] = binds;
            }
            d["uvw_drift"] = info.uvw_drift;
        // Measured on the last simulated step (see FluidDomainInfo).
        d["solid_phase_particles"] = info.solid_phase_particles;
        d["solid_phase_cells"] = info.solid_phase_cells;
        d["sealed_pockets"] = info.sealed_pockets;
        d["sealed_pocket_cells"] = info.sealed_pocket_cells;
        d["sealed_pockets_measured"] = info.sealed_pockets_measured;
        d["interior_fluid_cells"] = info.interior_fluid_cells;
        d["reseed_added_particles"] = info.reseed_added_particles;
        d["reseed_removed_particles"] = info.reseed_removed_particles;
        d["solid_phase"] = info.solid_phase_enabled;
        d["solid_phase_fill"] = info.solid_phase_fill;
            d["uvw_particles"] = info.uvw_particles;
            d["enabled"] = info.enabled;
            d["visible"] = info.visible;
            out.append(d);
        }
        return out;
    });

    fluid.def("get", [](const std::string& domain) -> py::dict {
        rtapi::FluidDomainInfo info;
        requireResult(rtapi::getFluidDomain(domain, info));
        py::dict d;
        d["id"] = info.id;
        d["name"] = info.name;
        d["type"] = info.type;
        d["domain_min"] = vec3ToPython(info.domain_min);
        d["domain_max"] = vec3ToPython(info.domain_max);
        d["voxel_size"] = info.voxel_size;
        d["particle_count"] = info.particle_count;
        d["live_state"] = info.live_state;
        d["render_mode"] = info.render_mode;
        d["backend"] = info.backend;
        d["boundary"] = info.boundary;
        d["preset"] = info.preset;
        d["kinematic_viscosity"] = info.kinematic_viscosity;
        d["viscosity_sweeps"] = info.viscosity_sweeps;
        d["viscosity_wall_slip"] = info.viscosity_wall_slip;
        d["granular_enabled"] = info.granular_enabled;
        d["granular_friction_angle"] = info.granular_friction_angle_degrees;
        d["granular_cohesion"] = info.granular_cohesion;
        d["granular_dilatancy"] = info.granular_dilatancy_degrees;
        d["granular_young_modulus"] = info.granular_young_modulus;
        d["granular_poisson_ratio"] = info.granular_poisson_ratio;
        d["granular_tensile_cutoff"] = info.granular_tensile_cutoff;
        d["granular_hardening"] = info.granular_hardening;
        d["granular_fracture_strain"] = info.granular_fracture_strain;
        d["granular_damage_rate"] = info.granular_damage_rate;
        d["granular_healing_rate"] = info.granular_healing_rate;
        d["granular_rebonding"] = info.granular_rebonding;
        d["granular_max_solver_substeps"] = info.granular_max_solver_substeps;
        d["granular_softening_temperature"] = info.granular_softening_temperature;
        d["granular_softening_range"] = info.granular_softening_range;
        d["granular_residual_strength"] = info.granular_residual_strength;
        d["granular_tack_peak"] = info.granular_tack_peak;
        d["granular_thermal_conductivity"] = info.granular_thermal_conductivity;
        d["granular_yielded"] = info.granular_yielded_particles;
        d["granular_detached"] = info.granular_detached_particles;
        d["granular_invalid"] = info.granular_invalid_particles;
        d["granular_sleeping"] = info.granular_sleeping_particles;
        d["granular_damaged"] = info.granular_damaged_particles;
        d["granular_damage_over_10"] = info.granular_damage_over_10_particles;
        d["granular_damage_over_50"] = info.granular_damage_over_50_particles;
        d["granular_damage_over_90"] = info.granular_damage_over_90_particles;
        d["granular_max_yield"] = info.granular_max_yield;
        d["granular_max_plastic"] = info.granular_max_plastic_increment;
        d["granular_max_accumulated_plastic"] = info.granular_max_accumulated_plastic;
        d["granular_mean_accumulated_plastic"] = info.granular_mean_accumulated_plastic;
        d["granular_max_fracture_history"] = info.granular_max_fracture_history;
        d["granular_mean_fracture_history"] = info.granular_mean_fracture_history;
        d["granular_max_damage"] = info.granular_max_damage;
        d["granular_mean_damage"] = info.granular_mean_damage;
        d["granular_requested_young_modulus"] = info.granular_requested_young_modulus;
        d["granular_effective_young_modulus"] = info.granular_effective_young_modulus;
        d["granular_required_substeps"] = info.granular_required_substeps;
        d["granular_solver_substeps"] = info.granular_solver_substeps;
        d["granular_stiffness_capped"] = info.granular_stiffness_capped;
        d["granular_wave_substeps"] = info.granular_wave_substeps;
        d["granular_strain_substeps"] = info.granular_strain_substeps;
        d["granular_strain_rate"] = info.granular_strain_rate;
        d["granular_strain_limited"] = info.granular_strain_limited_particles;
        d["granular_compaction_capped"] = info.granular_compaction_capped_particles;
        d["granular_min_softening"] = info.granular_min_softening;
        d["granular_softened"] = info.granular_softened_particles;
        d["granular_overburden_pressure"] = info.granular_overburden_pressure;
        d["granular_young_modulus_for_load"] = info.granular_young_modulus_for_load;
        d["granular_stiffness_below_load"] = info.granular_stiffness_below_load;
        d["surface_material"] = info.surface_material;
        d["splat_material"] = info.splat_material;
        d["surface_offset_voxels"] = info.surface_offset_voxels;
        // Porosity was readable over IPC but not from Python — a script could
        // set it and never read it back. Closed here rather than in a separate
        // pass: the file was already open for the coordinate fields below.
        d["pore_amount"] = info.pore_amount;
        d["pore_scale"] = info.pore_scale;
        d["pore_detail"] = info.pore_detail;
        d["coord_space"] = info.coord_space == 1 ? "domain"
                         : info.coord_space == 2 ? "world" : "material";
        d["uvw_available"] = info.uvw_available;
        d["uvw_dim"] = py::make_tuple(info.uvw_dim[0], info.uvw_dim[1], info.uvw_dim[2]);
        d["uvw_origin"] = py::make_tuple(info.uvw_origin[0], info.uvw_origin[1],
                                         info.uvw_origin[2]);
        d["uvw_voxel"] = info.uvw_voxel;
        d["uvw_refresh_period"] = info.uvw_refresh_period;
        {
            py::list subs;
            for (const auto& s : info.substances) {
                py::dict e;
                e["name"] = s.name; e["tag"] = s.tag; e["particles"] = s.particles;
                subs.append(e);
            }
            d["substances"] = subs;
            py::list binds;
            for (const auto& b : info.substance_materials) {
                py::dict e;
                e["substance"] = b.substance; e["material_id"] = b.material_id;
                e["material"] = b.material; e["representation"] = b.representation;
                // "inherit" already resolved against the domain mode: what is
                // actually DRAWN. Assert on this, not on the authored value.
                e["effective_representation"] = b.effective_representation;
                // As authored: a negative viscosity is the inherit sentinel.
                e["kinematic_viscosity"] = b.kinematic_viscosity;
                e["miscibility"] = b.miscibility;
                // State of matter, a separate axis from representation.
                e["phase"] = b.phase;
                binds.append(e);
            }
            d["substance_materials"] = binds;
        }
        d["uvw_drift"] = info.uvw_drift;
        // Measured on the last simulated step (see FluidDomainInfo).
        d["solid_phase_particles"] = info.solid_phase_particles;
        d["solid_phase_cells"] = info.solid_phase_cells;
        d["sealed_pockets"] = info.sealed_pockets;
        d["sealed_pocket_cells"] = info.sealed_pocket_cells;
        d["sealed_pockets_measured"] = info.sealed_pockets_measured;
        d["interior_fluid_cells"] = info.interior_fluid_cells;
        d["reseed_added_particles"] = info.reseed_added_particles;
        d["reseed_removed_particles"] = info.reseed_removed_particles;
        d["solid_phase"] = info.solid_phase_enabled;
        d["solid_phase_fill"] = info.solid_phase_fill;
        d["uvw_particles"] = info.uvw_particles;
        d["enabled"] = info.enabled;
        d["visible"] = info.visible;
        return d;
    }, py::arg("domain"));

    // ★ seed_min/seed_max default to None, NOT to a box. The old defaults were
    // a fixed region at y 1.0-1.5 derived from nothing, so any domain that did
    // not contain it seeded zero particles and said so to nobody.
    fluid.def("seed", [](const std::string& domain, const py::object& seed_min,
                          const py::object& seed_max, int particles_per_cell,
                          bool replace, bool persistent) {
        const bool has_region = !seed_min.is_none() && !seed_max.is_none();
        Vec3 smin, smax;
        if (has_region) {
            smin = vec3FromPython(seed_min.cast<py::tuple>());
            smax = vec3FromPython(seed_max.cast<py::tuple>());
        }
        requireResult(rtapi::seedFluidParticles(domain,
                                                 has_region ? &smin : nullptr,
                                                 has_region ? &smax : nullptr,
                                                 particles_per_cell, replace,
                                                 persistent));
    }, py::arg("domain"), py::arg("seed_min") = py::none(),
       py::arg("seed_max") = py::none(),
       py::arg("particles_per_cell") = 4, py::arg("replace") = true,
       py::arg("persistent") = false,
       "Fill a fluid domain with particles. Omit seed_min/seed_max to fill the "
       "bottom half of the domain; a region that does not overlap the domain is "
       "refused rather than silently creating nothing.");

    fluid.def("clear", [](const std::string& domain, bool clear_seed) {
        requireResult(rtapi::clearFluidParticles(domain, clear_seed));
    }, py::arg("domain"), py::arg("clear_seed") = false);

    // Bind a material to a SUBSTANCE inside one domain. "" clears the binding;
    // "dielectric" selects the built-in refractive liquid for that substance,
    // which is what lets one substance be a Principled BSDF and another stay
    // plain water in the same body.
    // Also carries the substance's PHYSICS: a substance is one authored thing,
    // and separating how it looks from how it flows is what made the old panel
    // imply that mixing was a shading choice.
    //   kinematic_viscosity — ABSOLUTE m^2/s; NEGATIVE = inherit the domain's.
    //   miscibility         — 1 soft gradient, 0 sharp front. The pair takes the
    //                         minimum of its two: refusing to mix is unilateral.
    //   phase               — "liquid" (default) or "solid". Solid parcels are
    //                         rasterized into the grid's solid mask each step,
    //                         so the liquid flows around and clings to them.
    // ★ None means "leave alone", not "reset". For viscosity a sentinel could
    // not have carried that, because a negative value is itself meaningful.
    // ★★ `phase` is NOT `representation`: one says what the matter IS, the
    // other how it is DRAWN. A solid can be splat spheres or an isosurface.
    fluid.def("set_splat_material",
              [](const std::string& domain, const std::string& material) {
        requireResult(rtapi::setFluidSplatMaterial(domain, material));
    }, py::arg("domain"), py::arg("material") = std::string());

    fluid.def("set_substance_material",
              [](const std::string& domain, const std::string& substance,
                 const std::string& material, const py::object& representation,
                 const py::object& kinematic_viscosity, const py::object& miscibility,
                 const py::object& phase) {
        std::string rep;
        const std::string* p_rep = nullptr;
        if (!representation.is_none()) { rep = py::cast<std::string>(representation); p_rep = &rep; }
        float visc_val = 0.0f; const float* p_visc = nullptr;
        if (!kinematic_viscosity.is_none()) {
            visc_val = py::cast<float>(kinematic_viscosity); p_visc = &visc_val;
        }
        float misc_val = 0.0f; const float* p_misc = nullptr;
        if (!miscibility.is_none()) {
            misc_val = py::cast<float>(miscibility); p_misc = &misc_val;
        }
        std::string phase_val; const std::string* p_phase = nullptr;
        if (!phase.is_none()) { phase_val = py::cast<std::string>(phase); p_phase = &phase_val; }
        requireResult(rtapi::setFluidSubstanceMaterial(domain, substance, material,
                                                       p_rep, p_visc, p_misc, p_phase));
    }, py::arg("domain"), py::arg("substance"), py::arg("material") = std::string(),
       py::arg("representation") = py::none(),
       py::arg("kinematic_viscosity") = py::none(),
       py::arg("miscibility") = py::none(),
       py::arg("phase") = py::none());

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

        // No `viscosity` alias on purpose: it used to be a unitless 0..200 dial
        // and is now m^2/s, so silently accepting the old keyword would turn a
        // script's honey into lava. Reject it with the conversion spelled out.
        if (kwargs.contains("viscosity")) {
            throw std::runtime_error(
                "'viscosity' was replaced by 'kinematic_viscosity' in m^2/s "
                "(water 1e-6, oil 1e-4, chocolate 4e-3, honey 7e-3, lava 0.5). "
                "The old value was unitless and is not convertible.");
        }
        float visc_val = 0.0f; const float* p_visc = nullptr;
        if (kwargs.contains("kinematic_viscosity")) { visc_val = py::cast<float>(kwargs["kinematic_viscosity"]); p_visc = &visc_val; }

        int sweeps_val = 0; const int* p_sweeps = nullptr;
        if (kwargs.contains("viscosity_sweeps")) { sweeps_val = py::cast<int>(kwargs["viscosity_sweeps"]); p_sweeps = &sweeps_val; }

        float slip_val = 0.0f; const float* p_slip = nullptr;
        if (kwargs.contains("viscosity_wall_slip")) { slip_val = py::cast<float>(kwargs["viscosity_wall_slip"]); p_slip = &slip_val; }

        // Material name shading the SurfaceSDF isosurface; "" = built-in dielectric.
        std::string surf_mat_val; const std::string* p_surf_mat = nullptr;
        if (kwargs.contains("surface_material")) { surf_mat_val = py::cast<std::string>(kwargs["surface_material"]); p_surf_mat = &surf_mat_val; }

        float surface_offset_val = 0.0f; const float* p_surface_offset = nullptr;
        if (kwargs.contains("surface_offset_voxels")) {
            surface_offset_val = py::cast<float>(kwargs["surface_offset_voxels"]);
            p_surface_offset = &surface_offset_val;
        }

        // Procedural porosity on the SDF isosurface (crumb / aeration).
        float pore_amt_val = 0.0f; const float* p_pore_amt = nullptr;
        if (kwargs.contains("pore_amount")) { pore_amt_val = py::cast<float>(kwargs["pore_amount"]); p_pore_amt = &pore_amt_val; }
        float pore_scl_val = 0.0f; const float* p_pore_scl = nullptr;
        if (kwargs.contains("pore_scale")) { pore_scl_val = py::cast<float>(kwargs["pore_scale"]); p_pore_scl = &pore_scl_val; }
        float pore_det_val = 0.0f; const float* p_pore_det = nullptr;
        if (kwargs.contains("pore_detail")) { pore_det_val = py::cast<float>(kwargs["pore_detail"]); p_pore_det = &pore_det_val; }

        bool enabled_val = false; const bool* p_enabled = nullptr;
        if (kwargs.contains("enabled")) { enabled_val = py::cast<bool>(kwargs["enabled"]); p_enabled = &enabled_val; }

        bool visible_val = false; const bool* p_visible = nullptr;
        if (kwargs.contains("visible")) { visible_val = py::cast<bool>(kwargs["visible"]); p_visible = &visible_val; }

        // Coordinate space for every isosurface pattern. By NAME, matching the
        // IPC vocabulary and the read-back, so a script never keeps an enum
        // table of its own; an int is still accepted for wire symmetry.
        int coord_val = 0; const int* p_coord = nullptr;
        if (kwargs.contains("coord_space")) {
            const py::object cs = kwargs["coord_space"];
            if (py::isinstance<py::str>(cs)) {
                const std::string name = py::cast<std::string>(cs);
                coord_val = (name == "domain") ? 1 : (name == "world") ? 2 : 0;
            } else {
                coord_val = py::cast<int>(cs);
            }
            p_coord = &coord_val;
        }
        int uvw_period = 0; const int* p_uvw_period = nullptr;
        if (kwargs.contains("uvw_refresh_period")) {
            uvw_period = py::cast<int>(kwargs["uvw_refresh_period"]);
            p_uvw_period = &uvw_period;
        }

        // Solid-phase coupling. The master switch is the control that answers
        // "is the solid causing this?" without destroying the authoring under
        // test, so it has to be reachable from a script like everything else.
        bool solid_phase_val = false; const bool* p_solid_phase = nullptr;
        if (kwargs.contains("solid_phase")) {
            solid_phase_val = py::cast<bool>(kwargs["solid_phase"]);
            p_solid_phase = &solid_phase_val;
        }
        float solid_fill_val = 0.0f; const float* p_solid_fill = nullptr;
        if (kwargs.contains("solid_phase_fill")) {
            solid_fill_val = py::cast<float>(kwargs["solid_phase_fill"]);
            p_solid_fill = &solid_fill_val;
        }

        bool granular_enabled_val=false; const bool* p_granular_enabled=nullptr;
        float granular_friction=0, granular_cohesion=0, granular_dilatancy=0;
        float granular_young=0, granular_poisson=0, granular_tensile=0, granular_hardening=0;
        float granular_fracture=0, granular_damage=0, granular_healing=0;
        bool granular_rebonding=false; const bool* p_granular_rebonding=nullptr;
        int granular_max_solver_substeps=0; const int* p_granular_max_solver_substeps=nullptr;
        const float *p_granular_friction=nullptr,*p_granular_cohesion=nullptr,*p_granular_dilatancy=nullptr;
        const float *p_granular_young=nullptr,*p_granular_poisson=nullptr,*p_granular_tensile=nullptr,*p_granular_hardening=nullptr;
        const float *p_granular_fracture=nullptr,*p_granular_damage=nullptr,*p_granular_healing=nullptr;
        if(kwargs.contains("granular_enabled")){granular_enabled_val=py::cast<bool>(kwargs["granular_enabled"]);p_granular_enabled=&granular_enabled_val;}
        if(kwargs.contains("granular_friction_angle")){granular_friction=py::cast<float>(kwargs["granular_friction_angle"]);p_granular_friction=&granular_friction;}
        if(kwargs.contains("granular_cohesion")){granular_cohesion=py::cast<float>(kwargs["granular_cohesion"]);p_granular_cohesion=&granular_cohesion;}
        if(kwargs.contains("granular_dilatancy")){granular_dilatancy=py::cast<float>(kwargs["granular_dilatancy"]);p_granular_dilatancy=&granular_dilatancy;}
        if(kwargs.contains("granular_young_modulus")){granular_young=py::cast<float>(kwargs["granular_young_modulus"]);p_granular_young=&granular_young;}
        if(kwargs.contains("granular_poisson_ratio")){granular_poisson=py::cast<float>(kwargs["granular_poisson_ratio"]);p_granular_poisson=&granular_poisson;}
        if(kwargs.contains("granular_tensile_cutoff")){granular_tensile=py::cast<float>(kwargs["granular_tensile_cutoff"]);p_granular_tensile=&granular_tensile;}
        if(kwargs.contains("granular_hardening")){granular_hardening=py::cast<float>(kwargs["granular_hardening"]);p_granular_hardening=&granular_hardening;}
        if(kwargs.contains("granular_fracture_strain")){granular_fracture=py::cast<float>(kwargs["granular_fracture_strain"]);p_granular_fracture=&granular_fracture;}
        if(kwargs.contains("granular_damage_rate")){granular_damage=py::cast<float>(kwargs["granular_damage_rate"]);p_granular_damage=&granular_damage;}
        if(kwargs.contains("granular_healing_rate")){granular_healing=py::cast<float>(kwargs["granular_healing_rate"]);p_granular_healing=&granular_healing;}
        if(kwargs.contains("granular_rebonding")){granular_rebonding=py::cast<bool>(kwargs["granular_rebonding"]);p_granular_rebonding=&granular_rebonding;}
        if(kwargs.contains("granular_max_solver_substeps")){granular_max_solver_substeps=py::cast<int>(kwargs["granular_max_solver_substeps"]);p_granular_max_solver_substeps=&granular_max_solver_substeps;}
        float granular_soft_temp=0,granular_soft_range=0,granular_residual=0;
        const float *p_granular_soft_temp=nullptr,*p_granular_soft_range=nullptr,*p_granular_residual=nullptr;
        if(kwargs.contains("granular_softening_temperature")){granular_soft_temp=py::cast<float>(kwargs["granular_softening_temperature"]);p_granular_soft_temp=&granular_soft_temp;}
        if(kwargs.contains("granular_softening_range")){granular_soft_range=py::cast<float>(kwargs["granular_softening_range"]);p_granular_soft_range=&granular_soft_range;}
        if(kwargs.contains("granular_residual_strength")){granular_residual=py::cast<float>(kwargs["granular_residual_strength"]);p_granular_residual=&granular_residual;}
        float granular_tack=0.0f;const float* p_granular_tack=nullptr;
        if(kwargs.contains("granular_tack_peak")){granular_tack=py::cast<float>(kwargs["granular_tack_peak"]);p_granular_tack=&granular_tack;}
        float granular_cond=0.0f;const float* p_granular_cond=nullptr;
        if(kwargs.contains("granular_thermal_conductivity")){granular_cond=py::cast<float>(kwargs["granular_thermal_conductivity"]);p_granular_cond=&granular_cond;}

        requireResult(rtapi::updateFluidDomain(domain, p_dmin, p_dmax, p_vs, p_rm, p_dev, p_bound, p_preset, p_visc, p_sweeps, p_slip, p_surf_mat,
                                               p_surface_offset,
                                               p_pore_amt, p_pore_scl, p_pore_det, p_coord,
                                               p_uvw_period, p_solid_phase, p_solid_fill,
                                               p_enabled, p_visible,
                                               p_granular_enabled,p_granular_friction,p_granular_cohesion,
                                               p_granular_dilatancy,p_granular_young,p_granular_poisson,
                                               p_granular_tensile,p_granular_hardening,
                                               p_granular_fracture,p_granular_damage,
                                               p_granular_healing,p_granular_rebonding,
                                               p_granular_max_solver_substeps,
                                               p_granular_soft_temp,p_granular_soft_range,
                                               p_granular_residual,
                                               p_granular_tack, p_granular_cond));
    }, py::arg("domain"));

    fluid.def("reset", []() {
        requireResult(rtapi::resetFluidSimulation());
    });

    fluid.def("step", [](float dt) {
        requireResult(rtapi::stepFluidSimulation(dt));
    }, py::arg("dt") = 0.0166667f);

    // Gas domains share the production grid-domain implementation with
    // rt.fluid, so expose the complete lifecycle/control surface under the
    // semantically correct rt.gas namespace as well.
    gas.attr("get") = fluid.attr("get");
    gas.attr("remove_domain") = fluid.attr("remove_domain");
    gas.attr("clear") = fluid.attr("clear");
    gas.attr("set_param") = fluid.attr("set_param");
    gas.attr("reset") = fluid.attr("reset");
    gas.attr("step") = fluid.attr("step");
    rtpy::registerPressureBindings(gas);

    auto gas_settings_to_dict = [](const rtapi::GasDomainSettings& s) {
        py::dict d;
        d["quality_profile"] = s.quality_profile;
        d["resource_budget_mb"] = s.resource_budget_mb;
        d["enforce_resource_budget"] = s.enforce_resource_budget;
        d["use_sparse_tiles"] = s.use_sparse_tiles;
        d["render_to_nanovdb"] = s.render_to_nanovdb;
        d["fire_enabled"] = s.fire_enabled;
        d["ignition_temperature"] = s.ignition_temperature;
        d["burn_rate"] = s.burn_rate;
        d["heat_release"] = s.heat_release;
        d["smoke_generation"] = s.smoke_generation;
        d["flame_dissipation"] = s.flame_dissipation;
        d["fire_max_temperature"] = s.fire_max_temperature;
        d["structural_coupling_enabled"] = s.structural_coupling_enabled;
        d["structural_pressure_scale"] = s.structural_pressure_scale;
        d["structural_min_intensity"] = s.structural_min_intensity;
        d["structural_event_interval"] = s.structural_event_interval;
        d["buoyancy_heat"] = s.buoyancy_heat;
        d["buoyancy_density"] = s.buoyancy_density;
        d["vorticity"] = s.vorticity;
        d["fire_expansion"] = s.fire_expansion;
        d["turbulence_strength"] = s.turbulence_strength;
        d["turbulence_scale"] = s.turbulence_scale;
        d["turbulence_octaves"] = s.turbulence_octaves;
        d["turbulence_lacunarity"] = s.turbulence_lacunarity;
        d["turbulence_persistence"] = s.turbulence_persistence;
        d["turbulence_speed"] = s.turbulence_speed;
        return d;
    };
    gas.def("get_settings", [gas_settings_to_dict](const std::string& domain) {
        rtapi::GasDomainSettings settings;
        requireResult(rtapi::getGasDomainSettings(domain, settings));
        return gas_settings_to_dict(settings);
    }, py::arg("domain"));
    gas.def("set_settings", [](const std::string& domain, const py::kwargs& kwargs) {
        rtapi::GasDomainSettings s;
        requireResult(rtapi::getGasDomainSettings(domain, s));
#define RT_GAS_KW(name, type) if (kwargs.contains(#name)) s.name = py::cast<type>(kwargs[#name])
        RT_GAS_KW(quality_profile, std::string);
        RT_GAS_KW(resource_budget_mb, uint32_t);
        RT_GAS_KW(enforce_resource_budget, bool);
        RT_GAS_KW(use_sparse_tiles, bool);
        RT_GAS_KW(render_to_nanovdb, bool);
        RT_GAS_KW(fire_enabled, bool);
        RT_GAS_KW(ignition_temperature, float);
        RT_GAS_KW(burn_rate, float);
        RT_GAS_KW(heat_release, float);
        RT_GAS_KW(smoke_generation, float);
        RT_GAS_KW(flame_dissipation, float);
        RT_GAS_KW(fire_max_temperature, float);
        RT_GAS_KW(structural_coupling_enabled, bool);
        RT_GAS_KW(structural_pressure_scale, float);
        RT_GAS_KW(structural_min_intensity, float);
        RT_GAS_KW(structural_event_interval, float);
        RT_GAS_KW(buoyancy_heat, float);
        RT_GAS_KW(buoyancy_density, float);
        RT_GAS_KW(vorticity, float);
        RT_GAS_KW(fire_expansion, float);
        RT_GAS_KW(turbulence_strength, float);
        RT_GAS_KW(turbulence_scale, float);
        RT_GAS_KW(turbulence_octaves, int);
        RT_GAS_KW(turbulence_lacunarity, float);
        RT_GAS_KW(turbulence_persistence, float);
        RT_GAS_KW(turbulence_speed, float);
#undef RT_GAS_KW
        requireResult(rtapi::updateGasDomainSettings(domain, s));
    }, py::arg("domain"));

    // Volume appearance. Without this a script could enable fire and still get
    // flat grey smoke, because the default shader has no blackbody emission.
    auto gas_shader_to_dict = [](const rtapi::GasShaderSettings& s) {
        py::dict d;
        d["preset"] = s.preset;
        d["density_multiplier"] = s.density_multiplier;
        d["density_cutoff"] = s.density_cutoff;
        d["blackbody_intensity"] = s.blackbody_intensity;
        d["temperature_min"] = s.temperature_min;
        d["temperature_max"] = s.temperature_max;
        d["scattering_coefficient"] = s.scattering_coefficient;
        d["absorption_coefficient"] = s.absorption_coefficient;
        return d;
    };
    gas.def("get_shader", [gas_shader_to_dict](const std::string& domain) {
        rtapi::GasShaderSettings s;
        requireResult(rtapi::getGasShaderSettings(domain, s));
        return gas_shader_to_dict(s);
    }, py::arg("domain"));
    gas.def("set_shader", [](const std::string& domain, const py::kwargs& kwargs) {
        rtapi::GasShaderSettings s;
        requireResult(rtapi::getGasShaderSettings(domain, s));
        // An omitted preset keeps the current look and only applies the
        // individual overrides below; passing one re-bases from that preset.
        if (!kwargs.contains("preset")) s.preset.clear();
#define RT_GASSHADER_KW(name, type) if (kwargs.contains(#name)) s.name = py::cast<type>(kwargs[#name])
        RT_GASSHADER_KW(preset, std::string);
        RT_GASSHADER_KW(density_multiplier, float);
        RT_GASSHADER_KW(density_cutoff, float);
        RT_GASSHADER_KW(blackbody_intensity, float);
        RT_GASSHADER_KW(temperature_min, float);
        RT_GASSHADER_KW(temperature_max, float);
        RT_GASSHADER_KW(scattering_coefficient, float);
        RT_GASSHADER_KW(absorption_coefficient, float);
#undef RT_GASSHADER_KW
        requireResult(rtapi::updateGasShaderSettings(domain, s));
    }, py::arg("domain"));

    fluid.def("get_combustion", [](const std::string& domain) {
        rtapi::CombustibleFluidSettings s;
        requireResult(rtapi::getCombustibleFluidSettings(domain,s));
        py::dict d;
        d["chemistry_preset"] = s.chemistry_preset;
        d["enabled"]=s.enabled;
        d["auto_ignite"]=s.auto_ignite;
        d["ignition_temperature"]=s.ignition_temperature;
        d["evaporation_rate"]=s.evaporation_rate;
        d["surface_fuel_capacity"]=s.surface_fuel_capacity;
        d["heat_release"]=s.heat_release;
        d["smoke_yield"]=s.smoke_yield;
        d["surface_cooling"]=s.surface_cooling;
        return d;
    },py::arg("domain"));
    fluid.def("set_combustion", [](const std::string& domain,
                                   const py::kwargs& kwargs) {
        rtapi::CombustibleFluidSettings s;
        requireResult(rtapi::getCombustibleFluidSettings(domain,s));
#define RT_FLUID_FIRE_KW(name,type) \
        if(kwargs.contains(#name)) s.name=py::cast<type>(kwargs[#name])
        RT_FLUID_FIRE_KW(enabled,bool);
        if(kwargs.contains("chemistry_preset")) s.chemistry_preset=py::cast<std::string>(kwargs["chemistry_preset"]);
        RT_FLUID_FIRE_KW(auto_ignite,bool);
        RT_FLUID_FIRE_KW(ignition_temperature,float);
        RT_FLUID_FIRE_KW(evaporation_rate,float);
        RT_FLUID_FIRE_KW(surface_fuel_capacity,float);
        RT_FLUID_FIRE_KW(heat_release,float);
        RT_FLUID_FIRE_KW(smoke_yield,float);
        RT_FLUID_FIRE_KW(surface_cooling,float);
#undef RT_FLUID_FIRE_KW
        requireResult(rtapi::updateCombustibleFluidSettings(domain,s));
    },py::arg("domain"));

    auto flow_to_dict = [](const rtapi::SimulationFlowSourceInfo& s) {
        py::dict d;
        d["name"] = s.name; d["domain"] = s.domain;
        d["source_mode"] = s.source_mode; d["source_object"] = s.source_object;
        d["enabled"] = s.enabled; d["parent_object"] = s.parent_object;
        d["velocity_space"] = s.velocity_space;
        d["inherit_velocity"] = s.inherit_velocity;
        d["position"] = vec3ToPython(s.position);
        d["velocity"] = vec3ToPython(s.velocity); d["radius"] = s.radius;
        d["velocity_coupling"] = s.velocity_coupling;
        d["density"] = s.density; d["temperature"] = s.temperature;
        d["fuel"] = s.fuel; d["falloff"] = s.falloff;
        d["fluid_particles_per_second"] = s.fluid_particles_per_second;
        d["fluid_velocity_spread"] = s.fluid_velocity_spread;
        d["fluid_emit_along_normal"] = s.fluid_emit_along_normal;
        d["fluid_substance"] = s.fluid_substance;
        d["use_time_limit"] = s.use_time_limit;
        d["start_time"] = s.start_time; d["end_time"] = s.end_time;
        d["use_particle_limit"] = s.use_particle_limit;
        d["max_emitted_particles"] = s.max_emitted_particles;
        return d;
    };
    auto apply_flow_kwargs = [](rtapi::SimulationFlowSourceInfo& s,
                                const py::kwargs& kw) {
#define RT_FLOW_KW(name,type) if (kw.contains(#name)) s.name = py::cast<type>(kw[#name])
        RT_FLOW_KW(name, std::string); RT_FLOW_KW(domain, std::string);
        RT_FLOW_KW(source_mode, std::string); RT_FLOW_KW(source_object, std::string);
        RT_FLOW_KW(enabled, bool); RT_FLOW_KW(radius, float);
        RT_FLOW_KW(parent_object, std::string); RT_FLOW_KW(velocity_space, std::string);
        RT_FLOW_KW(inherit_velocity, float);
        RT_FLOW_KW(velocity_coupling, float); RT_FLOW_KW(density, float);
        RT_FLOW_KW(temperature, float); RT_FLOW_KW(fuel, float);
        RT_FLOW_KW(falloff, float); RT_FLOW_KW(fluid_particles_per_second, float);
        RT_FLOW_KW(fluid_velocity_spread, float);
        RT_FLOW_KW(fluid_emit_along_normal, bool);
        RT_FLOW_KW(fluid_substance, std::string);
        RT_FLOW_KW(use_time_limit, bool);
        RT_FLOW_KW(start_time, float); RT_FLOW_KW(end_time, float);
        RT_FLOW_KW(use_particle_limit, bool); RT_FLOW_KW(max_emitted_particles, int);
#undef RT_FLOW_KW
        if (kw.contains("position")) s.position = vec3FromPython(kw["position"]);
        if (kw.contains("velocity")) s.velocity = vec3FromPython(kw["velocity"]);
    };
    py::module_ flow_source = module.def_submodule(
        "flow_source", "Gas scalar injection and APIC liquid flow emitters");
    flow_source.def("list", [flow_to_dict] {
        std::vector<rtapi::SimulationFlowSourceInfo> sources;
        requireResult(rtapi::listSimulationFlowSources(sources));
        py::list out; for (const auto& s : sources) out.append(flow_to_dict(s));
        return out;
    });
    flow_source.def("get", [flow_to_dict](const std::string& name) {
        rtapi::SimulationFlowSourceInfo s;
        requireResult(rtapi::getSimulationFlowSource(name, s));
        return flow_to_dict(s);
    }, py::arg("name"));
    flow_source.def("create", [flow_to_dict, apply_flow_kwargs](
        const std::string& name, const std::string& domain, const py::kwargs& kw) {
        rtapi::SimulationFlowSourceInfo s; s.name = name; s.domain = domain;
        apply_flow_kwargs(s, kw);
        rtapi::SimulationFlowSourceInfo created;
        requireResult(rtapi::createSimulationFlowSource(s, created));
        return flow_to_dict(created);
    }, py::arg("name"), py::arg("domain"));
    flow_source.def("update", [apply_flow_kwargs](
        const std::string& name, const py::kwargs& kw) {
        rtapi::SimulationFlowSourceInfo s;
        requireResult(rtapi::getSimulationFlowSource(name, s));
        apply_flow_kwargs(s, kw);
        requireResult(rtapi::updateSimulationFlowSource(name, s));
    }, py::arg("name"));
    flow_source.def("remove", [](const std::string& name) {
        requireResult(rtapi::removeSimulationFlowSource(name));
    }, py::arg("name"));
    // Timeline keys — only the channels actually passed are keyed. `flow_rate`
    // is the liquid emission rate, i.e. the hose valve.
    flow_source.def("key", [](const std::string& name, int frame,
                              const py::kwargs& kw) {
        rtapi::SimulationFlowSourceKey key;
        key.frame = frame;
        if (kw.contains("enabled")) { key.has_enabled = true; key.enabled = py::cast<bool>(kw["enabled"]); }
        if (kw.contains("position")) { key.has_position = true; key.position = vec3FromPython(kw["position"]); }
        if (kw.contains("velocity")) { key.has_velocity = true; key.velocity = vec3FromPython(kw["velocity"]); }
        if (kw.contains("radius")) { key.has_radius = true; key.radius = py::cast<float>(kw["radius"]); }
        if (kw.contains("density")) { key.has_density = true; key.density = py::cast<float>(kw["density"]); }
        if (kw.contains("temperature")) { key.has_temperature = true; key.temperature = py::cast<float>(kw["temperature"]); }
        if (kw.contains("fuel")) { key.has_fuel = true; key.fuel = py::cast<float>(kw["fuel"]); }
        if (kw.contains("falloff")) { key.has_falloff = true; key.falloff = py::cast<float>(kw["falloff"]); }
        if (kw.contains("velocity_coupling")) { key.has_velocity_coupling = true; key.velocity_coupling = py::cast<float>(kw["velocity_coupling"]); }
        if (kw.contains("flow_rate")) { key.has_flow_rate = true; key.flow_rate = py::cast<float>(kw["flow_rate"]); }
        requireResult(rtapi::keySimulationFlowSource(name, key));
    }, py::arg("name"), py::arg("frame"));
    flow_source.def("clear_key", [](const std::string& name, int frame) {
        requireResult(rtapi::clearSimulationFlowSourceKey(name, frame));
    }, py::arg("name"), py::arg("frame"));

    auto collider_to_dict = [](const rtapi::SimulationColliderInfo& c) {
        py::dict d;
        d["name"] = c.name; d["source_mode"] = c.source_mode;
        d["source_object"] = c.source_object; d["enabled"] = c.enabled;
        d["fluid_collision_enabled"] = c.fluid_collision_enabled;
        d["plane_y"] = c.plane_y; d["sphere_center"] = vec3ToPython(c.sphere_center);
        d["sphere_radius"] = c.sphere_radius;
        d["capsule_start"] = vec3ToPython(c.capsule_start);
        d["capsule_end"] = vec3ToPython(c.capsule_end);
        d["capsule_radius"] = c.capsule_radius;
        d["bounds_min"] = vec3ToPython(c.bounds_min);
        d["bounds_max"] = vec3ToPython(c.bounds_max);
        d["friction"] = c.friction; d["restitution"] = c.restitution;
        d["thickness"] = c.thickness;
        d["sdf_resolution_mode"] = c.sdf_resolution_mode;
        d["sdf_ready"] = c.sdf_ready;
        d["sdf_resolution"] = c.sdf_resolution;
        d["gas_interaction_enabled"] = c.gas_interaction_enabled;
        d["gas_density_rate"] = c.gas_density_rate;
        d["gas_temperature_rate"] = c.gas_temperature_rate;
        d["gas_fuel_rate"] = c.gas_fuel_rate;
        d["gas_flame_rate"] = c.gas_flame_rate;
        d["gas_ignite_on_contact"] = c.gas_ignite_on_contact;
        d["msf_substance"] = c.msf_substance;
        d["msf_override_ignition"] = c.msf_override_ignition;
        d["msf_ignition_kelvin"] = c.msf_ignition_kelvin;
        d["msf_burn_rate_scale"] = c.msf_burn_rate_scale;
        d["msf_fuel_capacity_scale"] = c.msf_fuel_capacity_scale;
        d["msf_mask_resolution"] = c.msf_mask_resolution;
        d["msf_generate_char_mask"] = c.msf_generate_char_mask;
        d["msf_auto_transfer"] = c.msf_auto_transfer;
        d["msf_transfer_domain"] = c.msf_transfer_domain;
        d["msf_transfer_rate_kg_s"] = c.msf_transfer_rate_kg_s;
        d["msf_transfer_min_mass_kg"] = c.msf_transfer_min_mass_kg;
        d["msf_transfer_particles_per_kg"] = c.msf_transfer_particles_per_kg;
        d["msf_transfer_max_batch_particles"] = c.msf_transfer_max_batch_particles;
        d["msf_transfer_velocity"] = vec3ToPython(c.msf_transfer_velocity);
        d["msf_melt_flow_enabled"] = c.msf_melt_flow_enabled;
        d["msf_melt_height_loss"] = c.msf_melt_height_loss;
        d["msf_melt_spread"] = c.msf_melt_spread;
        d["msf_melt_sdf_refresh"] = c.msf_melt_sdf_refresh;
        d["msf_melt_sdf_revision_interval"] = c.msf_melt_sdf_revision_interval;
        d["msf_melt_sdf_change_threshold"] = c.msf_melt_sdf_change_threshold;
        return d;
    };
    auto apply_collider_kwargs = [](rtapi::SimulationColliderInfo& c,
                                    const py::kwargs& kw) {
#define RT_COLLIDER_KW(name,type) if (kw.contains(#name)) c.name = py::cast<type>(kw[#name])
        RT_COLLIDER_KW(name, std::string); RT_COLLIDER_KW(source_mode, std::string);
        RT_COLLIDER_KW(source_object, std::string); RT_COLLIDER_KW(enabled, bool);
        RT_COLLIDER_KW(fluid_collision_enabled, bool);
        RT_COLLIDER_KW(plane_y, float); RT_COLLIDER_KW(sphere_radius, float);
        RT_COLLIDER_KW(capsule_radius, float); RT_COLLIDER_KW(friction, float);
        RT_COLLIDER_KW(restitution, float); RT_COLLIDER_KW(thickness, float);
        RT_COLLIDER_KW(gas_interaction_enabled, bool);
        RT_COLLIDER_KW(gas_density_rate, float); RT_COLLIDER_KW(gas_temperature_rate, float);
        RT_COLLIDER_KW(gas_fuel_rate, float); RT_COLLIDER_KW(gas_flame_rate, float);
        RT_COLLIDER_KW(gas_ignite_on_contact, bool);
        RT_COLLIDER_KW(msf_substance, std::string);
        RT_COLLIDER_KW(msf_override_ignition, bool);
        RT_COLLIDER_KW(msf_ignition_kelvin, float);
        RT_COLLIDER_KW(msf_burn_rate_scale, float);
        RT_COLLIDER_KW(msf_fuel_capacity_scale, float);
        RT_COLLIDER_KW(msf_mask_resolution, int);
        RT_COLLIDER_KW(msf_generate_char_mask, bool);
        RT_COLLIDER_KW(msf_auto_transfer, bool);
        RT_COLLIDER_KW(msf_transfer_domain, std::string);
        RT_COLLIDER_KW(msf_transfer_rate_kg_s, float);
        RT_COLLIDER_KW(msf_transfer_min_mass_kg, float);
        RT_COLLIDER_KW(msf_transfer_particles_per_kg, float);
        RT_COLLIDER_KW(msf_transfer_max_batch_particles, uint32_t);
        RT_COLLIDER_KW(msf_melt_flow_enabled, bool);
        RT_COLLIDER_KW(msf_melt_height_loss, float);
        RT_COLLIDER_KW(msf_melt_spread, float);
        RT_COLLIDER_KW(msf_melt_sdf_refresh, bool);
        RT_COLLIDER_KW(msf_melt_sdf_revision_interval, uint32_t);
        RT_COLLIDER_KW(msf_melt_sdf_change_threshold, float);
        if (kw.contains("msf_transfer_velocity"))
            c.msf_transfer_velocity = vec3FromPython(kw["msf_transfer_velocity"]);
        RT_COLLIDER_KW(sdf_resolution_mode, int);
#undef RT_COLLIDER_KW
        if (kw.contains("sphere_center")) c.sphere_center = vec3FromPython(kw["sphere_center"]);
        if (kw.contains("capsule_start")) c.capsule_start = vec3FromPython(kw["capsule_start"]);
        if (kw.contains("capsule_end")) c.capsule_end = vec3FromPython(kw["capsule_end"]);
        if (kw.contains("bounds_min")) c.bounds_min = vec3FromPython(kw["bounds_min"]);
        if (kw.contains("bounds_max")) c.bounds_max = vec3FromPython(kw["bounds_max"]);
    };
    py::module_ collider = module.def_submodule(
        "collider", "Shared particle, APIC liquid and gas simulation colliders");
    collider.def("list", [collider_to_dict] {
        std::vector<rtapi::SimulationColliderInfo> colliders;
        requireResult(rtapi::listSimulationColliders(colliders));
        py::list out; for (const auto& c : colliders) out.append(collider_to_dict(c));
        return out;
    });
    collider.def("get", [collider_to_dict](const std::string& name) {
        rtapi::SimulationColliderInfo c;
        requireResult(rtapi::getSimulationCollider(name, c));
        return collider_to_dict(c);
    }, py::arg("name"));
    collider.def("create", [collider_to_dict, apply_collider_kwargs](
        const std::string& name, const py::kwargs& kw) {
        rtapi::SimulationColliderInfo c; c.name = name;
        apply_collider_kwargs(c, kw);
        rtapi::SimulationColliderInfo created;
        requireResult(rtapi::createSimulationCollider(c, created));
        return collider_to_dict(created);
    }, py::arg("name"));
    collider.def("update", [apply_collider_kwargs](
        const std::string& name, const py::kwargs& kw) {
        rtapi::SimulationColliderInfo c;
        requireResult(rtapi::getSimulationCollider(name, c));
        apply_collider_kwargs(c, kw);
        requireResult(rtapi::updateSimulationCollider(name, c));
    }, py::arg("name"));
    collider.def("remove", [](const std::string& name) {
        requireResult(rtapi::removeSimulationCollider(name));
    }, py::arg("name"));
    collider.def("rebuild_sdf", [](const std::string& name) {
        requireResult(rtapi::rebuildSimulationColliderSDF(name));
    }, py::arg("name"), "Force an asynchronous rebuild of a mesh_sdf collider");

    // Material State Field: what scene objects are MADE of. Substance names are
    // assigned on the collider that represents the object, because MSF is keyed
    // by the collider's source object.
    py::module_ msf = module.def_submodule(
        "msf", "Material State Field: substances, ignition and charring");
    msf.def("substances", [] {
        std::vector<std::string> names;
        requireResult(rtapi::listMaterialSubstances(names));
        py::list out; for (const auto& n : names) out.append(n);
        return out;
    }, "Names accepted by collider(msf_substance=...)");
    msf.def("fields", [] {
        std::vector<rtapi::MaterialFieldInfo> fields;
        requireResult(rtapi::listMaterialFields(fields));
        py::list out;
        for (const auto& f : fields) {
            py::dict d;
            d["object_key"] = f.object_key;
            d["substance"] = f.substance;
            d["topology_generation"] = f.topology_generation;
            d["content_generation"] = f.content_generation;
            d["element_count"] = f.element_count;
            d["mask_resolution"] = f.mask_resolution;
            d["centers_dirty"] = f.centers_dirty;
            d["mean_integrity"] = f.mean_integrity;
            d["minimum_integrity"] = f.minimum_integrity;
            d["mass_loss"] = f.mass_loss;
            d["initial_mass"] = f.initial_mass;
            d["solid_mass"] = f.solid_mass;
            d["pyrolyzed_mass"] = f.pyrolyzed_mass;
            d["molten_reservoir_mass"] = f.molten_reservoir_mass;
            d["transferred_mass"] = f.transferred_mass;
            d["mass_conservation_error"] = f.mass_conservation_error;
            d["mass_budget_overflow"] = f.mass_budget_overflow;
            d["mass_negative"] = f.mass_negative;
            d["mass_invalid_elements"] = f.mass_invalid_elements;
            d["domain"] = "surface_uv";
            py::list semantics;
            for (const auto& semantic : f.semantics) semantics.append(semantic);
            d["semantics"] = semantics;
            out.append(std::move(d));
        }
        return out;
    }, "Read-only metadata for live Material State Fields");

    py::module_ terrain = module.def_submodule("terrain", "Terrain creation, queries, and procedural operations");
    auto terrain_info_to_dict = [](const rtapi::TerrainInfo& info) -> py::dict {
        py::dict d;
        d["id"] = info.id;
        d["name"] = info.name;
        d["resolution"] = py::make_tuple(info.width, info.height);   // FIELD grid
        d["mesh_resolution"] = info.mesh_resolution;                 // 0 = follow the field
        d["mesh_grid"] = py::make_tuple(info.mesh_width, info.mesh_height);
        d["paint_resolution"] = info.paint_resolution;               // 0 = follow the field
        d["paint_grid"] = py::make_tuple(info.paint_width, info.paint_height);
        d["has_surface_semantic"] = info.has_surface_semantic;
        d["surface_semantic_channels"] = py::make_tuple(
            "flow", "wetness", "ice", "hardness");
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
                                                   float size, float height_scale,
                                                   int mesh_resolution) -> py::dict {
        rtapi::TerrainInfo info;
        requireResult(rtapi::createTerrain(name, resolution, size, height_scale,
                                           mesh_resolution, info));
        return terrain_info_to_dict(info);
    }, py::arg("name") = "Terrain", py::arg("resolution") = 1024,
       py::arg("size") = 1000.0f, py::arg("height_scale") = 100.0f,
       py::arg("mesh_resolution") = 0,
       "resolution is the FIELD (analysis) grid; mesh_resolution is the vertex "
       "grid, 0 to follow the field. Both are creation parameters because "
       "decimating afterwards still pays one full-resolution build.");
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
    terrain.def("set_mesh_resolution", [terrain_info_to_dict](const std::string& name,
                                                              int mesh_resolution) -> py::dict {
        rtapi::TerrainInfo info;
        requireResult(rtapi::setTerrainMeshResolution(name, mesh_resolution, info));
        return terrain_info_to_dict(info);
    }, py::arg("name"), py::arg("mesh_resolution"),
       "Vertex-grid resolution, independent of the field (analysis) resolution. "
       "0 follows the field, which is the historical behaviour. Asking for more "
       "than the field is refused, not clamped -- a mesh denser than its data "
       "would interpolate detail that does not exist.");
    terrain.def("set_paint_resolution", [terrain_info_to_dict](const std::string& name,
                                                               int paint_resolution) -> py::dict {
        rtapi::TerrainInfo info;
        requireResult(rtapi::setTerrainPaintResolution(name, paint_resolution, info));
        return terrain_info_to_dict(info);
    }, py::arg("name"), py::arg("paint_resolution"),
       "Set the independent splat/SatMap evaluation grid. 0 follows the field; "
       "higher values let procedural paint nodes generate real high-frequency detail.");
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
                                    bool replace_graph, bool add_satmap) {
        requireResult(rtapi::applyTerrainPreset(name, preset, replace_graph, add_satmap));
    }, py::arg("name"), py::arg("preset"), py::arg("replace_graph") = false,
       py::arg("add_satmap") = false);
    terrain.def("list_satmap_presets", []() {
        std::vector<rtapi::TerrainSatMapPresetInfo> presets;
        requireResult(rtapi::listTerrainSatMapPresets(presets));
        py::list result;
        for (const auto& preset : presets) {
            py::dict item;
            item["id"] = preset.id; item["label"] = preset.label;
            item["category"] = preset.category; item["description"] = preset.description;
            item["version"] = preset.version; item["layer_count"] = preset.layer_count;
            result.append(item);
        }
        return result;
    });
    terrain.def("apply_satmap_preset", [](const std::string& name, const std::string& preset) {
        std::vector<std::string> warnings;
        requireResult(rtapi::applyTerrainSatMapPreset(name, preset, warnings));
        return warnings;
    }, py::arg("name"), py::arg("preset"));
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

    // rt.select + rt.material + rt.lights live in RtPyScene.cpp (this TU was
    // already at its size budget when the scene-core surface was expanded).
    rtpy::registerSceneBindings(module);

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

    // ── Skeletal playback (Faz 5.6c) ────────────────────────────────────
    // Transport + graph parameters only; node-graph topology is deliberately
    // not exposed while the animation runtime is still being decided.
    auto characterToDict = [](const rtapi::AnimCharacterInfo& info) -> py::dict {
        py::dict d;
        d["name"] = info.name;
        d["has_animation"] = info.has_animation;
        d["clip_count"] = info.clip_count;
        d["bone_count"] = info.bone_count;
        d["uses_graph"] = info.uses_graph;
        d["graph_asset_key"] = info.graph_asset_key;
        d["graph_follows_timeline"] = info.graph_follows_timeline;
        d["root_motion"] = info.root_motion;
        d["root_motion_bone"] = info.root_motion_bone;
        d["visible"] = info.visible;
        return d;
    };
    auto playbackToDict = [](const rtapi::AnimPlaybackInfo& info) -> py::dict {
        py::dict d;
        d["clip"] = info.clip;
        d["playing"] = info.playing;
        d["paused"] = info.paused;
        d["blending"] = info.blending;
        d["time"] = info.time;
        d["normalized_time"] = info.normalized_time;
        d["layer"] = info.layer;
        return d;
    };

    anim.def("characters", [characterToDict]() -> py::list {
        py::list out;
        for (const rtapi::AnimCharacterInfo& info : rtapi::listAnimCharacters())
            out.append(characterToDict(info));
        return out;
    });
    anim.def("character", [characterToDict](const std::string& character) {
        rtapi::AnimCharacterInfo info;
        requireResult(rtapi::getAnimCharacter(character, info));
        return characterToDict(info);
    }, py::arg("character"));
    anim.def("clips", [](const std::string& character) -> py::list {
        std::vector<rtapi::AnimClipInfo> clips;
        requireResult(rtapi::listAnimClips(character, clips));
        py::list out;
        for (const rtapi::AnimClipInfo& clip : clips) {
            py::dict d;
            d["name"] = clip.name;
            d["duration_seconds"] = clip.duration_seconds;
            d["ticks_per_second"] = clip.ticks_per_second;
            d["loop"] = clip.loop;
            d["start_frame"] = clip.start_frame;
            d["end_frame"] = clip.end_frame;
            out.append(d);
        }
        return out;
    }, py::arg("character"));
    anim.def("play", [](const std::string& character, const std::string& clip,
                        float blend, int layer) {
        requireResult(rtapi::playAnimClip(character, clip, blend, layer));
    }, py::arg("character"), py::arg("clip"), py::arg("blend") = 0.3f, py::arg("layer") = 0);
    anim.def("stop", [](const std::string& character, float blend_out, int layer) {
        requireResult(rtapi::stopAnimation(character, blend_out, layer));
    }, py::arg("character"), py::arg("blend_out") = 0.3f, py::arg("layer") = 0);
    anim.def("set_paused", [](const std::string& character, bool paused) {
        requireResult(rtapi::setAnimPaused(character, paused));
    }, py::arg("character"), py::arg("paused") = true);
    anim.def("set_time", [](const std::string& character, float seconds, int layer) {
        requireResult(rtapi::setAnimTime(character, seconds, layer));
    }, py::arg("character"), py::arg("seconds"), py::arg("layer") = 0);
    anim.def("set_speed", [](const std::string& character, float speed, int layer) {
        requireResult(rtapi::setAnimSpeed(character, speed, layer));
    }, py::arg("character"), py::arg("speed"), py::arg("layer") = 0);
    anim.def("set_loop", [](const std::string& character, bool loop, int layer) {
        requireResult(rtapi::setAnimLoop(character, loop, layer));
    }, py::arg("character"), py::arg("loop"), py::arg("layer") = 0);
    anim.def("status", [playbackToDict](const std::string& character, int layer) {
        rtapi::AnimPlaybackInfo info;
        requireResult(rtapi::getAnimPlayback(character, layer, info));
        return playbackToDict(info);
    }, py::arg("character"), py::arg("layer") = 0);
    anim.def("set_graph_param", [](const std::string& character, const std::string& name,
                                   const py::handle& value) {
        if (py::isinstance<py::bool_>(value))
            requireResult(rtapi::setAnimGraphBool(character, name, py::cast<bool>(value)));
        else
            requireResult(rtapi::setAnimGraphFloat(character, name, py::cast<float>(value)));
    }, py::arg("character"), py::arg("name"), py::arg("value"));
    anim.def("trigger_graph_param", [](const std::string& character, const std::string& name) {
        requireResult(rtapi::triggerAnimGraphParam(character, name));
    }, py::arg("character"), py::arg("name"));
    anim.def("graph_status", [playbackToDict](const std::string& character) {
        rtapi::AnimPlaybackInfo info;
        requireResult(rtapi::getAnimGraphPlayback(character, info));
        return playbackToDict(info);
    }, py::arg("character"));

    py::module_ nodes = module.def_submodule("nodes",
        "Node graph construction (material / geometry / terrain graphs via NodeRegistry)");
    nodes.def("graphs", [](const std::string& graph_type) {
        return rtapi::listNodeGraphs(graph_type);
    }, py::arg("graph_type"), "Existing graph names for material|geometry|terrain.");
    nodes.def("create_graph", [](const std::string& graph_type, const std::string& graph_name) {
        requireResult(rtapi::createNodeGraph(graph_type, graph_name));
    }, py::arg("graph_type"), py::arg("graph_name"),
       "Material graphs are keyed by material name and seeded from it; geometry "
       "graphs are keyed by object name and start empty. Terrain graphs come from "
       "rt.terrain.apply_preset.");
    nodes.def("remove_graph", [](const std::string& graph_type, const std::string& graph_name) {
        requireResult(rtapi::removeNodeGraph(graph_type, graph_name));
    }, py::arg("graph_type"), py::arg("graph_name"));
    nodes.def("apply", [](const std::string& graph_type, const std::string& graph_name) {
        rtapi::NodeGraphApplyInfo info;
        requireResult(rtapi::applyNodeGraph(graph_type, graph_name, info));
        py::dict out;
        out["ok"] = info.ok;
        out["warnings"] = info.warnings;
        out["errors"] = info.errors;
        return out;
    }, py::arg("graph_type"), py::arg("graph_name"),
       "Applies a graph to what it drives (material: the editor's own Apply — fold, "
       "publish, compile the per-pixel program, re-upload; geometry: Geo-DAG apply). "
       "Terrain keeps its async contract: use rt.terrain.evaluate. Returns "
       "{ok, warnings, errors} — warnings mirror the editor's diagnostics.");
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

    // Ambient thermal condition every uncoupled substance relaxes toward.
    // Distinct from world.get/set above (render sky) -- see WorldThermalInfo.
    world.def("get_thermal", [] {
        rtapi::WorldThermalInfo t;
        requireResult(rtapi::getWorldThermal(t));
        py::dict d;
        d["ambient_kelvin"]          = t.ambient_kelvin;
        d["kelvin_per_unit"]         = t.kelvin_per_unit;
        d["convection_coefficient"]  = t.convection_coefficient;
        d["oxygen_availability"]     = t.oxygen_availability;
        return d;
    });
    world.def("set_thermal", [](const py::kwargs& kwargs) {
        float ambient_val = 0.0f;      const float* p_ambient = nullptr;
        float kelvin_val = 0.0f;       const float* p_kelvin = nullptr;
        float convection_val = 0.0f;   const float* p_convection = nullptr;
        float oxygen_val = 0.0f;       const float* p_oxygen = nullptr;
        if (kwargs.contains("ambient_kelvin")) { ambient_val = py::cast<float>(kwargs["ambient_kelvin"]); p_ambient = &ambient_val; }
        if (kwargs.contains("kelvin_per_unit")) { kelvin_val = py::cast<float>(kwargs["kelvin_per_unit"]); p_kelvin = &kelvin_val; }
        if (kwargs.contains("convection_coefficient")) { convection_val = py::cast<float>(kwargs["convection_coefficient"]); p_convection = &convection_val; }
        if (kwargs.contains("oxygen_availability")) { oxygen_val = py::cast<float>(kwargs["oxygen_availability"]); p_oxygen = &oxygen_val; }
        requireResult(rtapi::setWorldThermal(p_ambient, p_kelvin, p_convection, p_oxygen));
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
    render.def("volume_counters", [](bool enabled) {
        requireResult(rtapi::setVolumeInstrumentation(enabled));
    }, py::arg("enabled"),
       "Enable/disable the Vulkan volume GPU counters and zero them. They are "
       "opt-in: the atomics cost a little, so leave them off while measuring "
       "frame time.");
    render.def("volume_stats", [] {
        const rtapi::VolumeInstrumentationInfo s = rtapi::volumeStats();
        py::dict d;
        d["available"] = s.available;
        d["enabled"] = s.enabled;
        d["volume_rays"] = s.volume_rays;
        d["density_samples"] = s.density_samples;
        d["shadow_density_samples"] = s.shadow_density_samples;
        d["empty_segments_skipped"] = s.empty_segments_skipped;
        d["topology_segments_skipped"] = s.topology_segments_skipped;
        d["majorant_segments_skipped"] = s.majorant_segments_skipped;
        d["majorant_queries"] = s.majorant_queries;
        d["majorant_available_queries"] = s.majorant_available_queries;
        d["extinction_terminations"] = s.extinction_terminations;
        d["step_budget_exhausted"] = s.step_budget_exhausted;
        d["completed_intervals"] = s.completed_intervals;
        d["temporal_accepted"] = s.temporal_accepted;
        d["temporal_rejected"] = s.temporal_rejected;
        d["solid_probe_runs"] = s.solid_probe_runs;
        d["solid_probe_hits"] = s.solid_probe_hits;
        d["gas_handoffs"] = s.gas_handoffs;
        d["layered_handoffs"] = s.layered_handoffs;
        d["arbiter_rejects"] = s.arbiter_rejects;
        d["teleports"] = s.teleports;
        d["arbiter_candidates"] = s.arbiter_candidates;
        d["arbiter_gate_open"] = s.arbiter_gate_open;
        d["arbiter_no_box"] = s.arbiter_no_box;
        d["arbiter_empty_range"] = s.arbiter_empty_range;
        d["arbiter_no_crossing"] = s.arbiter_no_crossing;
        return d;
    }, "Synchronized snapshot of the Vulkan volume counters. Check 'available' "
       "(Vulkan backend present) and 'enabled' before reading any number — an "
       "all-zero result means one of those is false far more often than it "
       "means the volume was free.");

    // ── Viewport measurement ────────────────────────────────────────────────
    // ★ Reading a render's DATA instead of saving it and looking. See
    // docs/dev/AGENT_VIEWPORT_MEASUREMENT_PLAN.md for what this cost to learn.
    render.def("probe", [](int x, int y, int width, int height, float threshold) {
        rtapi::ViewportProbeRegion region;
        region.x = x; region.y = y; region.width = width; region.height = height;
        const rtapi::ViewportProbeInfo p = rtapi::probeViewportFrame(region, threshold);
        py::dict d;
        d["available"] = p.available;
        d["width"] = p.width;
        d["height"] = p.height;
        d["pixels"] = p.pixels;
        d["mean_luminance"] = p.mean_luminance;
        d["min_luminance"] = p.min_luminance;
        d["max_luminance"] = p.max_luminance;
        d["black_fraction"] = p.black_fraction;
        d["nan_fraction"] = p.nan_fraction;
        py::list hist;
        for (uint32_t bucket : p.histogram) hist.append(bucket);
        d["histogram"] = hist;
        return d;
    }, py::arg("x") = 0, py::arg("y") = 0, py::arg("width") = 0,
       py::arg("height") = 0, py::arg("threshold") = 0.001f,
       "Numeric statistics of the captured viewport frame. Requires "
       "rt.viewport.capture(True); 'available' is false until a frame has been "
       "captured, and that is NOT the same as a black frame.");

    // ── rt.perf: where the time went ────────────────────────────────────────
    //
    // ★★★ The predecessor of this surface wrote its numbers to the Scene Log
    // and then had its macro compiled out. Both halves defeat the way this
    // project is actually tested: a script cannot read the Scene Log, and a
    // disabled macro measures nothing. Sections are values.
    py::module_ perf = module.def_submodule("perf",
        "Build/render phase timings as values: name, last/total/max ms, call "
        "count and the working-set delta across the scope.");
    perf.def("list", [] {
        py::list out;
        for (const rtapi::PerfSection& s : rtapi::perfSections()) {
            py::dict d;
            d["name"] = s.name;
            d["last_ms"] = s.last_ms;
            d["total_ms"] = s.total_ms;
            d["max_ms"] = s.max_ms;
            d["count"] = s.count;
            d["last_rss_delta_mb"] = s.last_rss_delta_mb;
            d["rss_after_mb"] = s.rss_after_mb;
            d["seq"] = s.seq;
            out.append(std::move(d));
        }
        return out;
    }, "Every recorded section, newest write first.");
    perf.def("get", [](const std::string& name) -> py::object {
        rtapi::PerfSection s;
        // ★ None, not a zeroed record. A zero reads as "measured, and it cost
        // nothing", which is how a missing measurement becomes a false one.
        if (!rtapi::perfSection(name, s)) return py::none();
        py::dict d;
        d["name"] = s.name;
        d["last_ms"] = s.last_ms;
        d["total_ms"] = s.total_ms;
        d["max_ms"] = s.max_ms;
        d["count"] = s.count;
        d["last_rss_delta_mb"] = s.last_rss_delta_mb;
        d["rss_after_mb"] = s.rss_after_mb;
        d["seq"] = s.seq;
        return d;
    }, py::arg("name"),
       "One section by name, or None when it has never been recorded.");
    perf.def("reset", [] {
        requireResult(rtapi::perfReset());
    }, "Clear the counters. Write ordering (seq) stays monotonic across it.");
    perf.def("set_logging", [](bool enabled) {
        requireResult(rtapi::perfSetLogging(enabled));
    }, py::arg("enabled"),
       "Also mirror completed sections into the Scene Log. Off by default.");
    perf.def("logging", [] { return rtapi::perfLogging(); },
       "Whether Scene Log mirroring is on.");

    // ── rt.editor: the application's own editor view state ──────────────────
    //
    // ★★★ Distinct from `rt.ui` on purpose. `rt.ui` means "draw your own panel"
    // and can only work in-process, because a widget call is meaningless outside
    // a frame's draw context. `rt.editor` means "which editor is showing", which
    // is a value and crosses IPC like any other. Keeping them apart is what
    // makes the drawing exemption honest instead of a blanket blind spot — see
    // the EditorState note in RtApi.h.
    py::module_ editor = module.def_submodule("editor",
        "Editor view state: which bottom editor is open and which node graph "
        "domain it shows. Values only -- widgets are rt.ui and stay in-process.");
    editor.def("get_state", [] {
        const rtapi::EditorState s = rtapi::editorState();
        py::dict d;
        d["bottom_editor"] = s.bottom_editor;
        d["node_editor_domain"] = s.node_editor_domain;
        d["node_editor_open"] = s.node_editor_open;
        d["open_editors"] = s.open_editors;
        d["sim_graph_scope"] = s.sim_graph_scope;
        d["sim_graph_owner"] = s.sim_graph_owner;
        return d;
    }, "Which bottom editor is on screen right now.");
    editor.def("set_bottom_editor", [](const std::string& name) {
        requireResult(rtapi::setBottomEditor(name));
    }, py::arg("name"),
       "none|dope_sheet|graph_editor|console|assets|simulation|geometry|"
       "material|terrain|anim_graph");
    editor.def("set_node_domain", [](const std::string& name) {
        requireResult(rtapi::setNodeEditorDomain(name));
    }, py::arg("name"),
       "simulation|geometry|material|terrain|animation");
    editor.def("set_sim_graph_scope", [](const std::string& scope,
                                         const std::string& owner) {
        requireResult(rtapi::setSimGraphScope(scope, owner));
    }, py::arg("scope"), py::arg("owner") = std::string(),
       "Which scoped simulation graph the Nodes canvas shows: object|domain|"
       "world plus the owner name. Selecting a scope with no graph is allowed "
       "and draws the empty state -- that is how creation is reached.");

    // ── Simulation node graph (Faz N0/N1/N2) ────────────────────────────────
    // ★ Evaluating produces COMMANDS, not effects. A script can read what a
    // graph MEANS without running a simulation, which is what makes the layer
    // testable at all. See NODE_SIMULATION_ARCHITECTURE_PLAN.md, BOLUM D.
    py::module_ sim_graph = module.def_submodule("sim_graph",
        "Simulation node graph: build it, read its intent, apply it as an "
        "override layer. Nothing here owns simulation state.");
    // *** scope and owner are POSITIONAL AND REQUIRED on every call below.
    // Giving `scope` a default would restore the "active domain" assumption the
    // scoped model exists to remove: the call would succeed against whichever
    // entity happened to be current and never report that it guessed.
    sim_graph.def("list", [] {
        py::list arr;
        for (const auto& g : rtapi::simGraphList()) {
            py::dict d;
            d["scope"] = g.scope; d["owner"] = g.owner;
            d["nodes"] = g.node_count; d["owner_node"] = g.owner_node;
            // True = the entity this graph names no longer exists. The graph
            // still draws and still accepts edits, and drives nothing.
            d["owner_missing"] = g.owner_missing;
            arr.append(d);
        }
        return arr;
    }, "Every simulation graph in the scene, with its scope and owner.");
    sim_graph.def("create", [](const std::string& scope, const std::string& owner) {
        requireResult(rtapi::simGraphCreate(scope, owner));
    }, py::arg("scope"), py::arg("owner") = std::string(),
       "Create the graph for a scene entity, seeded with its owner node. "
       "Idempotent. Fails if the named entity does not exist, so a typo cannot "
       "produce a live graph that drives nothing.");
    sim_graph.def("delete", [](const std::string& scope, const std::string& owner) {
        requireResult(rtapi::simGraphDelete(scope, owner));
    }, py::arg("scope"), py::arg("owner") = std::string());
    sim_graph.def("clear", [](const std::string& scope, const std::string& owner) {
        requireResult(rtapi::simGraphClear(scope, owner));
    }, py::arg("scope"), py::arg("owner") = std::string(),
       "Empty the canvas. The owner node is re-seeded: a scoped graph is never "
       "ownerless.");
    sim_graph.def("add_node", [](const std::string& scope, const std::string& owner,
                                 const std::string& type_id) {
        uint32_t id = 0;
        requireResult(rtapi::simGraphAddNode(scope, owner, type_id, id));
        return id;
    }, py::arg("scope"), py::arg("owner"), py::arg("type"));
    sim_graph.def("set_node", [](const std::string& scope, const std::string& owner,
                                 uint32_t node, const std::string& key,
                                 const std::string& value) {
        requireResult(rtapi::simGraphSetNodeText(scope, owner, node, key, value));
    }, py::arg("scope"), py::arg("owner"), py::arg("node"), py::arg("key"), py::arg("value"));
    sim_graph.def("set_node_value", [](const std::string& scope, const std::string& owner,
                                       uint32_t node, const std::string& key, float value) {
        requireResult(rtapi::simGraphSetNodeValue(scope, owner, node, key, value));
    }, py::arg("scope"), py::arg("owner"), py::arg("node"), py::arg("key"), py::arg("value"));
    sim_graph.def("apply", [](const std::string& scope, const std::string& owner,
                              bool allow_restart) {
        const rtapi::SimApplyResult r = rtapi::simGraphApply(scope, owner, allow_restart);
        py::dict d;
        d["ok"] = r.ok; d["applied"] = r.applied;
        d["overrides_held"] = r.overrides_held;
        d["refused"] = r.refused; d["failed"] = r.failed;
        return d;
    }, py::arg("scope"), py::arg("owner"), py::arg("allow_restart") = false,
       "Apply the graph as an OVERRIDE layer. Authored values are captured "
       "before the first write and restored by clear_overrides(). Parameters "
       "needing a restart are refused and reported unless allow_restart=True.");
    sim_graph.def("clear_overrides", [] {
        requireResult(rtapi::simGraphClearOverrides());
    }, "Restore every authored value ANY graph overrode. Deliberately not "
       "scoped: two graphs that wrote the same key share one authored value, so "
       "a per-graph restore could not put it back.");
    sim_graph.def("override_count", [] { return rtapi::simGraphOverrideCount(); });
    sim_graph.def("connect", [](const std::string& scope, const std::string& owner,
                                uint32_t from_node, uint32_t to_node,
                                int from_pin, int to_pin) {
        requireResult(rtapi::simGraphConnect(scope, owner, from_node, from_pin,
                                             to_node, to_pin));
    }, py::arg("scope"), py::arg("owner"), py::arg("from_node"), py::arg("to_node"),
       py::arg("from_pin") = 0, py::arg("to_pin") = 0);
    sim_graph.def("evaluate", [](const std::string& scope, const std::string& owner) {
        const rtapi::SimGraphEvaluation e = rtapi::simGraphEvaluate(scope, owner);
        py::list commands;
        for (const auto& c : e.commands) {
            py::dict d;
            d["kind"] = c.kind; d["target"] = c.target; d["key"] = c.key;
            d["value"] = c.value; d["text"] = c.text; d["source_node"] = c.source_node;
            commands.append(d);
        }
        py::list restarts;
        for (const auto& r : e.restart_requests) {
            py::dict d; d["node"] = r.node_id; d["reason"] = r.reason;
            restarts.append(d);
        }
        py::dict out;
        out["evaluated"] = e.evaluated;
        // * Says WHY nothing was evaluated. Without it, a graph that was not
        // found is indistinguishable from one that declares nothing.
        out["error"] = e.error;
        out["commands"] = commands;
        out["restart_requests"] = restarts;
        return out;
    }, py::arg("scope"), py::arg("owner"),
       "Evaluate the graph and return its intent. Nothing is applied, and a "
       "restart request is REPORTED, never acted on.");
    sim_graph.def("nodes", [](const std::string& scope, const std::string& owner) {
        std::vector<rtapi::SimNodeInfo> nodes;
        // ★ Raises when the graph does not exist. An empty list would say
        // "this graph has no nodes" about a graph that was never found.
        requireResult(rtapi::simGraphNodes(scope, owner, nodes));
        py::list arr;
        for (const auto& n : nodes) {
            py::dict d;
            d["id"] = n.id; d["owner_node"] = n.is_owner_node;
            d["type"] = n.type_id; d["name"] = n.display_name;
            d["enabled"] = n.enabled; d["inputs"] = n.input_count;
            d["outputs"] = n.output_count; d["domain"] = n.domain;
            d["channel"] = n.channel; d["source"] = n.source;
            if (n.has_stats) {
                d["stats_available"] = n.stats_available;
                d["particle_count"] = n.particle_count;
                d["array_size"] = n.array_size;
                d["array_in_sync"] = n.array_in_sync;
                d["host_fresh"] = n.host_fresh;
                d["min_value"] = n.min_value;
                d["max_value"] = n.max_value;
                d["mean_value"] = n.mean_value;
            }
            if (n.has_cache_status) {
                d["cache_valid"] = n.cache_valid;
                d["cache_baking"] = n.cache_baking;
                d["cache_stale"] = n.cache_stale;
                d["cache_ram_frames"] = n.cache_ram_frames;
            }
            if (!n.fields.empty()) {
                py::list fields;
                for (const auto& f : n.fields) {
                    py::dict fd;
                    fd["key"] = f.key;
                    // ★ False means this graph has no opinion about the
                    // parameter -- NOT that its value is zero.
                    fd["in_use"] = f.in_use;
                    fd["value"] = f.value;
                    fd["requires_restart"] = f.requires_restart;
                    fields.append(fd);
                }
                d["fields"] = fields;
            }
            arr.append(d);
        }
        return arr;
    }, py::arg("scope"), py::arg("owner"),
       "Nodes with their parameters. On a Field Inspect node, 'stats_available' "
       "false means the value could NOT be measured -- not that it measured zero. "
       "'array_in_sync' false means the attribute array is longer than the live "
       "particle count; the statistics still cover the live particles only.");
    sim_graph.def("couplings", [] {
        const rtapi::SimCouplingReport r = rtapi::simGraphCouplings();
        auto pack = [](const std::vector<rtapi::SimCouplingEntry>& v) {
            py::list arr;
            for (const auto& e : v) {
                py::dict d;
                d["coupling"] = e.coupling;
                d["producer"] = e.producer;
                d["consumer"] = e.consumer;
                d["source_domain"] = e.source_domain;
                d["target_domain"] = e.target_domain;
                d["active"] = e.active;
                d["source_node"] = e.source_node;
                d["scope"] = e.scope;
                d["owner"] = e.owner;
                arr.append(d);
            }
            return arr;
        };
        py::dict out;
        out["declared"] = pack(r.declared);
        out["actual"] = pack(r.actual);
        out["traced"] = r.traced;
        out["order_matches"] = r.order_matches;
        out["declared_not_running"] = r.declared_not_running;
        out["running_not_declared"] = r.running_not_declared;
        return out;
    }, "Couplings the graph DECLARES and the ones the solver actually RAN last "
       "step, in execution order. The graph does not schedule couplings -- "
       "comparing the two is the point. 'traced' false means the solver was "
       "never asked; an empty 'actual' with traced true is a measurement.");
    // N6 — the bake, and the state a script needs to reason about it.
    py::module_ sim_cache = module.def_submodule(
        "sim_cache", "Simulation bake and timeline cache state");
    sim_cache.def("status", [] {
        rtapi::SimCacheStatus s;
        requireResult(rtapi::simCacheStatus(s));
        py::dict d;
        d["valid"] = s.valid;
        d["baking"] = s.baking;
        d["cache_dir"] = s.cache_dir;
        d["ram_frames"] = s.ram_frames;
        d["has_range"] = s.has_range;
        d["first_frame"] = s.first_frame;
        d["last_frame"] = s.last_frame;
        d["config_signature"] = s.config_signature;
        return d;
    }, "Bake state. 'valid' (disk bake bound), 'baking' (running) and "
       "'ram_frames' (timeline scrub cache) are SEPARATE: all three can read "
       "'nothing usable' and they mean different things. 'config_signature' is "
       "the authored config the cache was built from -- when it changes, a bake "
       "that still exists no longer describes this scene.");
    sim_cache.def("bake", [](const std::string& cache_dir, int end_frame,
                             int start_frame, float fps) {
        requireResult(rtapi::simBake(cache_dir, start_frame, end_frame, fps));
    }, py::arg("cache_dir"), py::arg("end_frame"), py::arg("start_frame") = 0,
       py::arg("fps") = 24.0f,
       "Deterministically bake [start_frame, end_frame] to a disk point cache. "
       "BLOCKING: it walks the whole simulation on the calling thread.");
    sim_cache.def("clear", [] {
        requireResult(rtapi::simClearCache());
    }, "Drop the timeline cache and unbind the disk bake (free-run preview).");

    sim_graph.def("surface_attributes", [](const std::string& object) {
        return rtapi::simListSurfaceAttributes(object);
    }, py::arg("object"),
       "Per-element MSF attributes on one object: temperature, char, melt, "
       "moisture, fuel_remaining, mass_loss. Empty when the object carries no "
       "Material State Field -- which means UNMEASURED, not zero.");
    sim_graph.def("attributes", [](const std::string& domain) {
        return rtapi::simListAttributes(domain);
    }, py::arg("domain"),
       "Per-particle attributes that actually exist on this domain. This is the "
       "naming layer: until now the only way to learn an attribute existed was "
       "to read the solver source.");

    py::module_ viewport = module.def_submodule("viewport",
        "Viewport state and per-frame capture for render.probe().");
    viewport.def("capture", [](bool enabled) {
        requireResult(rtapi::setViewportCapture(enabled));
    }, py::arg("enabled"),
       "Enable per-frame capture of the displayed frame so render.probe can "
       "measure it. Off by default: it costs one copy per frame.");
    viewport.def("status", [] {
        const rtapi::ViewportStatusInfo s = rtapi::viewportStatus();
        py::dict d;
        d["available"] = s.available;
        d["backend"] = s.backend;
        d["width"] = s.width;
        d["height"] = s.height;
        d["samples"] = s.samples;
        d["accumulation_complete"] = s.accumulation_complete;
        d["ms_per_sample"] = s.ms_per_sample;
        d["rendering_active"] = s.rendering_active;
        d["capture_enabled"] = s.capture_enabled;
        d["frame_available"] = s.frame_available;
        d["shading"] = s.shading;
        return d;
    }, "Viewport state. Check 'rendering_active' and 'samples' before trusting "
       "any counter: an idle viewport reports zeros that look exactly like a "
       "cheap scene.");
    // ★ Script/IPC parity: viewport.render_frames existed over IPC with no
    // binding, so an in-app test could enable capture and probe but never make
    // the viewport actually produce the frame it was probing.
    viewport.def("render_frames", [](int count) {
        const rtapi::ViewportRenderResult r = rtapi::renderViewportFrames(count);
        py::dict d;
        d["samples_rendered"] = r.samples_rendered;
        d["converged"] = r.converged;
        d["ms_per_frame"] = r.ms_per_frame;
        return d;
    }, py::arg("count"),
       "Render this many viewport frames and report what was accumulated. "
       
       "IMPORTANT: this accumulates but does not PUBLISH a frame for capture - "
       "the display loop does that, and it cannot run while your script holds "
       "the main thread. Inside one script, frame_available stays false no "
       "matter how many frames you render here, so screenshot_base64 and "
       "render.probe have nothing fresh to read. Split the sequence across "
       "separate script runs, or drive it over IPC where each call returns to "
       "the loop.");

    // ★ Was panel-only until 2026-08-19. Every chain that needed a specific
    // viewport ("switch to Rendered, then measure") had to stop and ask a human
    // to click, which is the manual step CLAUDE.md rule 1 exists to prevent.
    viewport.def("shading", [] {
        const rtapi::ViewportShadingInfo s = rtapi::viewportShading();
        py::dict d;
        d["mode"] = s.mode;
        d["matcap_preset"] = s.matcap_preset;
        d["interactive_available"] = s.interactive_available;
        return d;
    }, "Which viewport is on screen: solid | material | rendered | matcap. "
       "'interactive_available' false means this machine has no raster viewport "
       "and only 'rendered' can be selected.");
    // Parity with viewport.get_screenshot over IPC. A script test that wants to
    // check what a vision model would be handed must be able to ask for the same
    // bytes; otherwise the only way to test the vision path is to run a model.
    viewport.def("screenshot_base64", [] {
        const std::string img = rtapi::getViewportScreenshotAsBase64();
        if (img.empty())
            throw std::runtime_error(
                "No frame to encode. Enable rt.viewport.capture(True) and render "
                "at least one frame first - an empty string here means 'not "
                "measured', not 'the screen is empty'.");
        return img;
    }, "The captured viewport frame as a base64 JPEG (quality 80). This is what "
       "a vision-capable model is shown. Requires rt.viewport.capture(True).");
    viewport.def("set_shading", [](const std::string& mode, int matcap_preset) {
        requireResult(rtapi::setViewportShading(mode, matcap_preset));
    }, py::arg("mode"), py::arg("matcap_preset") = -1,
       "Switch the viewport: solid | material | rendered | matcap ('preview' is "
       "accepted for material). Resets accumulation, so probe again after "
       "switching rather than reading the frame from the mode you left. Raises "
       "when the mode is unavailable instead of silently falling back.");

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

    // rt.ui (Faz 4b): panels, regions and the immediate-mode widget surface.
    // Implemented in RtUi.cpp — it is the only part of the Python layer that
    // links against ImGui, and RtPython.cpp had outgrown its size budget.
    rtui::registerBindings(module);

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
        rtui::clearPanels();
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

// Addon loader (Faz 4a) moved to Api/RtAddons.cpp — discovery, enable/disable,
// reload and addon_state.json persistence. Declarations stay in Api/RtPython.h.

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
