"""Non-build regression contracts for opening projects from Template Hub."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "RayTrophiStudio/source"


def read(name):
    data = (SOURCE / name).read_bytes()
    bom = bytes.fromhex("efbbbf")
    assert bom not in (data[3:] if data.startswith(bom) else data), name
    return data.decode("utf-8-sig")


def main():
    hub = read("src/UI/TemplateHubUI.cpp")
    # The loader updates recent_projects_; it must never run within its UI walk.
    assert "g_ProjectManager.openProject(" not in hub
    assert hub.count("ui.performOpenProject(") == 1
    dispatch = hub[hub.index("void TemplateHubUI::dispatchPendingProjectOpen("):
                   hub.index("void TemplateHubUI::performAutosaveRecovery(")]
    assert "ui.performOpenProject(context, path);" in dispatch
    assert dispatch.index("project_open_pending_ = false;") < dispatch.index(
        "ui.performOpenProject(context, path);")
    assert "pending_project_path_ = path.string();" in hub
    assert "queueProjectOpen(recent_projects_[0].path);" in hub
    assert hub.count("queueProjectOpen(recent.path);") == 2
    assert "queueProjectOpen(auto_path);" in hub
    assert hub.count("queueProjectOpen({});") == 2
    menu = read("src/UI/scene_ui_menu.hpp")
    dispatch_at = menu.index(".dispatchPendingProjectOpen(ctx, *this);")
    assert menu.index(".render(ctx, *this, &history);") < dispatch_at
    # Dispatch is the final operation: no more panels may inspect the old scene.
    end = menu.index("// Helper Functions for Menu Actions", dispatch_at)
    assert menu[dispatch_at:end].split(";", 1)[1].strip().startswith("}")
    ui = read("src/UI/scene_ui.cpp")
    draw_at = ui.index("    drawMainMenuBar(ctx);")
    guard_at = ui.index("if (scene_loading.load() || g_scene_loading_in_progress.load())", draw_at)
    assert draw_at < guard_at < ui.index("bool gizmo_hit = drawOverlays(ctx);", draw_at)
    open_at = ui.index("void SceneUI::performOpenProject(")
    open_body = ui[open_at:ui.index("loader_thread.detach();", open_at)]
    assert "requested_path.empty()" in open_body
    worker = open_body.index("std::thread loader_thread")
    for setup in ("ctx.selection.clearSelection();", "history.clear();",
                  "g_scene_loading_in_progress = true;", "scene_loading = true;"):
        assert open_body.index(setup) < worker
    gizmos = read("src/UI/scene_ui_gizmos.cpp")
    overlay = gizmos[gizmos.index("void SceneUI::drawParticleDebugOverlay("):]
    assert overlay.index("g_scene_loading_in_progress.load()") < overlay.index(
        "ctx.scene.getParticleSimulationSystem()")
    print("PASS Hub deferred project-open ownership/order, shared normal-open cleanup, overlay load guard")


if __name__ == "__main__":
    main()
