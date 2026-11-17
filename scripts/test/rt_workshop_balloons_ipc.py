"""Post-build live acceptance of the already-open isolated pastel workshop.

No launch/open/save or camera changes. Leaves test paint visible and the demo
stopped, with all mesh balls respawned. Restores display/capture flags.
"""
import argparse
import copy
import json
from pathlib import Path
import time

from rt_ipc import PLAYER_PIPE_NAME, RtIpc, RtIpcError

ROOT = Path(__file__).resolve().parents[2]
ASSET = ROOT / "RayTrophiStudio/assets/scenes/pastel-paint-workshop/workshop.rtp"
REPORT = ROOT / "tools/perception_bridge/out/workshop-mesh-acceptance.json"
COLORS = [[0.95,0.08,0.06],[0.06,0.20,0.95],[0.95,0.85,0.05],[0.10,0.85,0.15]]


def close(a, b):
    return len(a) == len(b) and all(abs(x-y) < 1e-4 for x,y in zip(a,b))


def inspect_meshes(ipc, state):
    assert state["visual_mode"] == "flat_mesh"
    for ball in state["balloons"]:
        assert ball["visual_resolved"] and ball["visual_visible"] == ball["inflated"]
        name = ball["visual_node"]
        transform = ipc.call("scene.get_world_transform", name=name)
        assert close(transform["translation"],ball["position"]), (ball,transform)
        color = ipc.call("material.get", object_name=name, param="base_color")
        # The palette is configurable (default 8 colours); read it from the demo.
        assert close(color,state["palette"][ball["color"]]), (name,color)
        roughness = ipc.call("material.get", object_name=name, param="roughness")
        assert abs(roughness-0.18) < 1e-4


def main():
    ipc = RtIpc(PLAYER_PIPE_NAME)
    report = {}
    original = None
    try:
        current = ipc.call("project.path")
        assert Path(current).resolve() == ASSET.resolve(), "open the isolated workshop first"
        state = ipc.call("paint.wet_demo")
        assert state["prepared"] and not state["running"], "stop the automatic demo first"
        assert len(state["balloons"]) == 12 and all(b["inflated"] for b in state["balloons"])
        inspect_meshes(ipc,state)
        report["initial"] = state
        # Bad references must be refused before changing the live demo/meshes.
        for bad in (state["config"]["visual_nodes"][:-1],
                    [state["config"]["visual_nodes"][0]]*12,
                    ["MissingWorkshopBall"]+state["config"]["visual_nodes"][1:]):
            config = copy.deepcopy(state["config"])
            config["visual_nodes"] = bad
            try:
                ipc.call("paint.wet_demo", operation="configure", config=config)
            except RtIpcError:
                pass
            else:
                raise AssertionError("invalid visual_nodes accepted")
            unchanged = ipc.call("paint.wet_demo")
            assert unchanged["config"] == state["config"]
            assert unchanged["balloons"] == state["balloons"]
        original = {"display":ipc.call("player.display"),
                    "capture":ipc.call("viewport.status")["capture_enabled"]}
        ipc.call("viewport.capture", enabled=False)
        ipc.call("player.set_display", continuous=True, pacing_hz=60)
        report["trials"] = []
        for count in (1,3):
            before = ipc.call("paint.wet_coverage")
            burst = ipc.call("paint.wet_demo", operation="burst", config={"count":count})
            assert burst["events"][-1]["spawned"] == count*1800
            assert sum(not b["inflated"] for b in burst["balloons"]) == count
            inspect_meshes(ipc,burst)
            deadline = time.monotonic()+30
            while True:
                time.sleep(0.05)
                settled = ipc.call("paint.wet_demo")
                assert not settled["error"], settled
                if all(b["inflated"] for b in settled["balloons"]):
                    break
                assert time.monotonic() < deadline, "mesh respawn timed out"
            inspect_meshes(ipc,settled)
            coverage = ipc.call("paint.wet_coverage")
            assert coverage["covered_texels"] > before["covered_texels"]
            wet = ipc.call("paint.wet_status")
            assert wet["publish_host_downloads"] == 0 and wet["publish_bytes_downloaded"] == 0
            telemetry = ipc.call("viewport.frame_telemetry")
            assert not telemetry["device_lost"] and telemetry["frames_consumed"] > 0
            report["trials"].append(dict(count=count,burst=burst,settled=settled,
                                         coverage=coverage,wet=wet,telemetry=telemetry))
        report["pipeline_pass"] = True
        report["scope"] = "mesh pose/color/hide/respawn + real droplets/paint; not visual/performance acceptance"
    finally:
        if original:
            ipc.call("viewport.capture", enabled=original["capture"])
            ipc.call("player.set_display", continuous=original["display"]["continuous"],
                     pacing_hz=float(original["display"]["pacing_hz"]))
        ipc.close()
        REPORT.parent.mkdir(parents=True,exist_ok=True)
        REPORT.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({"pipeline_pass":True,"report":str(REPORT)}))


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
