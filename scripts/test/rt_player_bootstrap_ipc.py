"""External Player bootstrap acceptance. Never launches either application.

After building, open Studio and Player, then run:
    python scripts/test/rt_player_bootstrap_ipc.py
Close Player normally and run again with --studio-only to check Studio survived.
Use an outer process timeout if diagnosing a server that does not answer.
"""
import argparse

from rt_ipc import PLAYER_PIPE_NAME, RtIpc, RtIpcError


CATALOG = {
    "agent.discover", "agent.describe", "agent.list_methods",
    "agent.search_capabilities", "agent.get_state_summary",
}
METHODS = CATALOG | {
    "project.open", "project.path", "camera.get", "camera.orbit", "camera.dolly",
    "viewport.capture", "viewport.get_screenshot", "viewport.frame_telemetry", "viewport.status",
    # Read-only simulated pose (PLAYER_GAME_PAINT_ROOM step 1).
    "scene.list_objects", "scene.get_world_transform",
    # Material values through the shared Renderer::updateBackendMaterial path.
    "material.list", "material.get", "material.of_object", "material.set",
}


def check_player():
    ipc = RtIpc(pipe_name=PLAYER_PIPE_NAME)
    try:
        discovery = ipc.call("agent.discover")
        assert discovery["app"] == "RayTrophi Player", discovery
        assert discovery["runtime_stage"] == "static_scene_viewer", discovery
        assert discovery["scene_loading"] and discovery["rendering"], discovery
        assert set(discovery["agent_methods"]) == CATALOG, discovery
        listing = ipc.call("agent.list_methods")
        assert {item["method"] for item in listing["methods"]} == METHODS, listing
        for method in sorted(METHODS):
            ipc.call("agent.describe", method=method)
        search = ipc.call("agent.search_capabilities", query="scatter fracture mesh")
        assert not search["relevant_workflows"], search
        assert all(item["method"] in METHODS for item in search["relevant_methods"]), search
        # No scene or arguments are needed: the product gate must run first.
        for method in ("mesh.bevel", "scatter.fill", "physics.fracture_object",
                       "script.run_file", "batch"):
            try:
                ipc.call(method)
            except RtIpcError as error:
                assert "unsupported_in_player" in str(error), (method, str(error))
            else:
                raise AssertionError(f"Player falsely accepted {method}")
        summary = ipc.call("agent.get_state_summary", include_probe=False)
        assert summary["runtime"]["host_active"], summary
        assert summary["runtime"]["host"] == "player", summary
        # Malformed supported calls must still use normal parameter validation.
        try:
            ipc.call("agent.describe", method=12)
        except RtIpcError:
            pass
        else:
            raise AssertionError("agent.describe accepted a numeric method")
    finally:
        ipc.close()


def check_studio():
    ipc = RtIpc()
    try:
        discovery = ipc.call("agent.discover")
        assert discovery["app"] == "RayTrophi Studio", discovery
        assert discovery["registered_methods"] > len(CATALOG), discovery
    finally:
        ipc.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studio-only", action="store_true")
    args = parser.parse_args()
    check_studio()
    if not args.studio_only:
        check_player()
    print("PASS: " + ("Studio still responds" if args.studio_only else
                      "separate Studio/Player pipes, Player discovery and rejection semantics"))


if __name__ == "__main__":
    main()
