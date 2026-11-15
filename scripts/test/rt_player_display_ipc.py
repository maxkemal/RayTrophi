"""Player display surface + frame timing acceptance, and a first frame-cost table.

Never launches an application. Player open (Studio optional: if it is open, the
test also checks that player.* fails there honestly instead of pretending).

    python scripts/test/rt_player_display_ipc.py [--project scene.rtp] [--seconds 3]

Measures uncapped, continuous rendering at render_scale 1.0 / 0.75 / 0.5 in the
current window, then fullscreen at 1.0, and prints one row per configuration:
fps, frame ms, and the render (GPU + readback) / upload / simulation / present
split. Restores the original display settings at the end.
"""
import argparse
import json
from pathlib import Path
import sys
import time

from rt_ipc import PLAYER_PIPE_NAME, RtIpc, RtIpcError
from rt_player_materials_ipc import FIXTURE as MATERIAL_FIXTURE


def runtime(ipc):
    return ipc.call("agent.get_state_summary", include_probe=False)["runtime"]


def measure(ipc, seconds, **settings):
    status = ipc.call("player.set_display", **settings)
    time.sleep(1.2)  # the first one-second window may straddle the switch
    deadline = time.monotonic() + seconds
    last = None
    while time.monotonic() < deadline:
        time.sleep(0.5)
        last = ipc.call("player.display")
    frame = last["frame"]
    return {"settings": settings, "render": last["render"], "output": last["output"],
            "fullscreen": last["fullscreen"],
            "idle": frame.get("idle"),
            **{k: round(float(frame.get(k, 0.0)), 2) for k in (
                "rendered_fps", "presented_fps", "frame_cost_ms", "capacity_fps",
                "backend_frame_ms_avg", "cpu_record_ms_avg", "slot_wait_ms_avg",
                "submit_ms_avg", "upload_ms_avg", "advance_ms_avg", "present_ms_avg",
                "present_interval_ms_max")}}


def check_studio_refuses():
    try:
        studio = RtIpc()
    except RtIpcError:
        return "studio not running (skipped)"
    try:
        studio.call("player.display")
    except RtIpcError as error:
        assert "requires the RayTrophi Player runtime" in str(error), str(error)
        return "refused honestly"
    finally:
        studio.close()
    raise AssertionError("Studio answered player.display without a Player display host")


def run(project, seconds):
    studio_check = check_studio_refuses()
    ipc = RtIpc(pipe_name=PLAYER_PIPE_NAME)
    original = ipc.call("player.display")
    capture_before = ipc.call("viewport.status")["capture_enabled"]
    rows = []
    try:
        revision = runtime(ipc)["scene_revision"]
        ipc.call("project.open", path=str(project))
        deadline = time.monotonic() + 90
        while not (runtime(ipc)["scene_ready"] and runtime(ipc)["scene_revision"] == revision + 1):
            assert time.monotonic() < deadline, runtime(ipc)
            time.sleep(0.1)
        # No capture here: capture forces synchronous present (a GPU wait in every
        # frame) and would measure a different path. Continuous mode renders an
        # unfocused Player on its own; capture is restored to its old state below.
        ipc.call("viewport.capture", enabled=False)

        # Wrong input is refused and changes nothing.
        for bad in ({"render_scale": 2.0}, {"pacing_hz": 3.0}, {"render_scale": "half"},
                    {"fullscreen": True, "width": 800, "height": 600}):
            try:
                ipc.call("player.set_display", **bad)
            except RtIpcError:
                pass
            else:
                raise AssertionError(f"player.set_display accepted {bad}")
        assert ipc.call("player.display")["render_scale"] == original["render_scale"]

        for scale in (1.0, 0.75, 0.5):
            row = measure(ipc, seconds, fullscreen=False, render_scale=scale,
                          pacing_hz=0.0, continuous=True)
            out_w, out_h = row["output"]
            ren_w, ren_h = row["render"]
            assert abs(ren_w - out_w * scale) <= 1 and abs(ren_h - out_h * scale) <= 1, row
            # Cost is measured from DRAWN frames only (never sleep or idle);
            # presented_fps needs a foreground window, rendered_fps does not.
            assert not row["idle"] and row["rendered_fps"] > 0, row
            assert row["frame_cost_ms"] > 0 and row["capacity_fps"] > 0, row
            rows.append(row)
        row = measure(ipc, seconds, fullscreen=True, render_scale=1.0, pacing_hz=0.0,
                      continuous=True)
        assert row["fullscreen"], row
        rows.append(row)
        row = measure(ipc, seconds, fullscreen=False, render_scale=1.0, pacing_hz=60.0,
                      continuous=True)
        assert row["rendered_fps"] <= 63, ("60 Hz pacing not honoured", row)
        # Pacing changes the rate, not the cost of a frame.
        row["cost_vs_uncapped"] = round(row["frame_cost_ms"] / max(rows[0]["frame_cost_ms"], 1e-6), 2)
        rows.append(row)
        state = runtime(ipc)
        assert not state["device_lost"] and not state["error"], state
        assert "loop" in state and state["loop"].get("frame"), "runtime.loop not published"
        print(json.dumps({"result": "PASS", "studio": studio_check, "rows": rows}, indent=2))
        print("\n scale fullscr render      drawn  cost   capac  backend cpu_rec slot_w upload  pacing")
        for r in rows:
            print(f" {r['settings'].get('render_scale', 1.0):4.2f} {str(r['fullscreen']):6}"
                  f" {r['render'][0]}x{r['render'][1]:<5} {r['rendered_fps']:6.1f}"
                  f" {r['frame_cost_ms']:6.2f} {r['capacity_fps']:6.1f}"
                  f" {r['backend_frame_ms_avg']:7.2f} {r['cpu_record_ms_avg']:7.2f}"
                  f" {r['slot_wait_ms_avg']:6.2f} {r['upload_ms_avg']:6.2f}"
                  f"  {r['settings'].get('pacing_hz')}")
    finally:
        restore = {k: original[k] for k in ("fullscreen", "render_scale", "pacing_hz",
                                            "continuous", "hud")}
        try:
            ipc.call("player.set_display", **restore)
            ipc.call("viewport.capture", enabled=capture_before)
        finally:
            ipc.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", type=Path, default=MATERIAL_FIXTURE)
    parser.add_argument("--seconds", type=float, default=3.0)
    args = parser.parse_args()
    run(args.project, args.seconds)


if __name__ == "__main__":
    sys.exit(main())
