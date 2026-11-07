"""Non-build checks for the UV module's build, API, security and dependency boundaries."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import re
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "RayTrophiStudio"
API = PROJECT / "source/src/Api"
METHODS = {"capabilities", "targets", "info", "layout", "transform", "normalize", "pack", "unwrap", "open",
           "flip_u", "flip_v", "swap_uv", "project_view", "set_corners"}


class UvContracts(unittest.TestCase):
    def test_attribute_only_update_boundary(self):
        service = (PROJECT / "source/src/UV/UvService.cpp").read_text()
        edit = service[service.index('class EditCommand'):service.index('json failure')]
        self.assertIn('publishAttributeUpdate(ctx, mesh_.get())', edit)
        self.assertNotIn('scheduleSceneMutationRebuilds', edit)
        self.assertIn('attribute_revision.fetch_add', edit)
        topology = (PROJECT / "source/src/UV/UvGeometryTransaction.cpp").read_text()
        self.assertIn('scheduleSceneMutationRebuilds(ctx, true)', topology)
        for source in ('Backend/VulkanBackend_UvUpdate.cpp', 'Render/OptixWrapper_UvUpdate.cpp'):
            code = (PROJECT / 'source/src' / source).read_text()
            self.assertIn('updateFlatMeshUVs', code)
            for operation in ('updateFlatMeshBLAS(', 'updateTLAS(', 'rebuildAccelerationStructure(',
                              'optixAccelBuild(', 'vkCmdBuildAccelerationStructuresKHR('):
                self.assertNotIn(operation, code)
        ast.parse((ROOT / 'scripts/test/rt_test_uv_attribute_updates_ipc.py').read_text())

    def test_vertex_scope_and_native_preview(self):
        service = (PROJECT / "source/src/UV/UvService.cpp").read_text()
        self.assertIn('params.contains("vertices") ? &vertices : nullptr', service)
        self.assertIn('"vertex_selection", true', service)
        binding = (API / "RtUvBindings.cpp").read_text()
        self.assertIn('{"vertices", "array", false', binding)
        self.assertIn('count = 12;', binding)
        view = (PROJECT / "source/src/UI/UvEditorUI.cpp").read_text()
        self.assertIn('params["vertices"] = state.gesture_vertices', view)
        self.assertIn('params["vertices"] = selectedVertices()', view)
        self.assertNotIn('const int samples = 128', view)
        preview = (PROJECT / "source/src/UI/UvTexturePreview.cpp").read_text()
        self.assertIn('SDL_UpdateTexture', preview)
        self.assertIn('->AddImage', preview)
        self.assertNotIn('scene_ui.h', preview)

    def test_uv_shortcut_ownership(self):
        view = (PROJECT / "source/src/UI/UvEditorUI.cpp").read_text()
        self.assertIn('bool capturesShortcuts()', view)
        self.assertIn('return state.visible;', view)
        for source in ("UI/scene_ui_gizmos.cpp", "UI/RigViewportUI.cpp", "Core/Main.cpp"):
            code = (PROJECT / "source/src" / source).read_bytes().decode("utf-8", errors="replace")
            self.assertIn('#include "UI/UvEditorUI.h"', code)
            self.assertIn('!UvEditorUI::capturesShortcuts()', code)
        service = (PROJECT / "source/src/UV/UvService.cpp").read_text()
        self.assertIn('summary["textures"]', service)
        self.assertIn('previewTextures(object, mesh_index)', service)

    def test_project_sources_and_headers(self):
        namespace = {"m": "http://schemas.microsoft.com/developer/msbuild/2003"}
        project = ET.parse(PROJECT / "RayTrophiStudio.vcxproj")
        filters = ET.parse(PROJECT / "RayTrophiStudio.vcxproj.filters")
        sources = list((PROJECT / "source/src/UV").glob("*.cpp"))
        sources += [PROJECT / "source/src/UI/UvEditorUI.cpp", API / "RtUvBindings.cpp", API / "RtIpcUv.cpp",
                    PROJECT / "source/src/UI/UvTexturePreview.cpp",
                    PROJECT / "source/src/Backend/VulkanBackend_UvUpdate.cpp",
                    PROJECT / "source/src/Render/OptixWrapper_UvUpdate.cpp",
                    PROJECT / "source/src/UI/MaterialUvSamplingUI.cpp",
                    PROJECT / "source/src/MeshEdit/GeometryRemap.cpp"]
        compiled = [e.attrib["Include"] for e in project.findall(".//m:ClCompile", namespace)
                    if "Include" in e.attrib]
        visible = {e.attrib["Include"] for e in filters.findall(".//m:ClCompile", namespace)}
        for source in sources:
            relative = str(source.relative_to(PROJECT)).replace("/", "\\")
            self.assertEqual(compiled.count(relative), 1, relative)
            self.assertIn(relative, visible)
        self.assertIn('"source/src/UV/*.cpp"', (PROJECT / "CMakeLists.txt").read_text())

    def test_surfaces_and_security(self):
        ipc = (API / "RtIpcUv.cpp").read_text()
        binding = (API / "RtUvBindings.cpp").read_text()
        exposed = set(re.findall(r'method == "uv\.([^"]+)"', ipc))
        self.assertEqual(exposed, METHODS)
        for method in METHODS:
            self.assertIn('"uv.' + method + '"', binding)
        spec = importlib.util.spec_from_file_location("uv_audit", ROOT / "scripts/audit_ipc_capabilities.py")
        audit = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(audit)
        security = (API / "RtIpcSecurity.cpp").read_text()
        namespaces = audit.namespace_table(security)
        uv_read_rule = security[security.index('if (method == "uv.capabilities"'):]
        uv_read_rule = uv_read_rule[:uv_read_rule.index("return Read;")]
        for method in METHODS:
            name = "uv." + method
            expected = "Read" if method in {"capabilities", "targets", "info", "layout"} else "SceneWrite"
            self.assertEqual(audit.required(name, namespaces), expected)
            if expected == "Read":
                self.assertIn('"' + name + '"', uv_read_rule)
            else:
                self.assertIn("uv.", namespaces)
        self.assertIn("enqueue(", ipc)

    def test_flat_module_boundary(self):
        for source in (PROJECT / "source/src/UV").glob("*.cpp"):
            code = source.read_text()
            self.assertNotIn("shared_ptr<Triangle>", code)
            self.assertNotIn("make_shared<Triangle>", code)
            self.assertLess(len(code.splitlines()), 2000)
        service = (PROJECT / "source/src/UV/UvService.cpp").read_text()
        for guard in ("uv_stale_revision", "uv_render_locked", "uv_history_unavailable",
                      "uv_texture_change_ack_required", "uv_async_required"):
            self.assertIn(guard, service)

    def test_vendor_provenance_and_probe_syntax(self):
        vendor = PROJECT / "external/xatlas"
        for file, digest in {
            "xatlas.h": "e7675335ad8ab1c1cc9060ad153cf6b8ba2ee914282044eb5f02c49590218fbd",
            "LICENSE": "2c16d5b1c2808277fe975b4f70f0fd9afc9b3bcf04e6c676665a82cb2d5579e3",
        }.items():
            self.assertEqual(hashlib.sha256((vendor / file).read_bytes()).hexdigest(), digest)
        self.assertIn("MIT License", (vendor / "xatlas.cpp").read_text())
        self.assertIn("f700c7790aaa030e794b52ba7791a05c085faf0c", (vendor / "SOURCE.txt").read_text())
        ast.parse((ROOT / "scripts/test/rt_test_uv_authoring_ipc.py").read_text())
        ast.parse((ROOT / "scripts/test/rt_test_uv_python.py").read_text())
        ast.parse((ROOT / "scripts/test/rt_test_uv_corner_ipc.py").read_text())
        ast.parse((ROOT / "scripts/test/rt_test_uv_corner_python.py").read_text())

    def test_corner_remap_integration(self):
        service = (PROJECT / "source/src/UV/UvService.cpp").read_text()
        self.assertIn('"corner_remap", true', service)
        self.assertIn('DNA::remapGeometry', service)
        self.assertIn('commitGeometry', service)
        self.assertIn('"vertex_indices"', service)
        backend = (PROJECT / "source/src/UV/UvXatlasBackend.cpp").read_text()
        self.assertIn('applyCornerCoordinates', backend)
        self.assertNotIn('uv_vertex_split_required', backend)
        cmake = (ROOT / "tests/uv/CMakeLists.txt").read_text()
        for source in ("UvCornerRemapTests.cpp", "GeometryRemap.cpp", "UvCornerRemap.cpp"):
            self.assertIn(source, cmake)

    def test_material_panel_has_only_sampling_and_editor_entry(self):
        panel = (PROJECT / "source/src/UI/scene_ui_materials.cpp").read_text()
        self.assertNotIn('BeginSection("UV Workflow"', panel)
        self.assertNotIn('uv_workflow_preview_entries.push_back', panel)
        self.assertIn('MaterialUvSamplingUI::draw', panel)
        sampling = (PROJECT / "source/src/UI/MaterialUvSamplingUI.cpp").read_text()
        for removed in ('apply("flip_u")', 'apply("normalize")', 'Project UVs from View'):
            self.assertNotIn(removed, sampling)
        self.assertIn('textureTransform', sampling)
        self.assertIn('UvEditorUI::open', sampling)

    def test_uv_view_does_not_depend_on_private_scene_ui(self):
        view = (PROJECT / "source/src/UI/UvEditorUI.cpp").read_text()
        self.assertNotIn('#include "scene_ui.h"', view)
        self.assertNotIn("->dockToBottom", view)
        self.assertIn("dock_to_bottom(\"UV Editor\")", view)


if __name__ == "__main__":
    unittest.main()
