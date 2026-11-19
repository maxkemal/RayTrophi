"""Run externally after rebuilding/opening Player; never launches the application.

python scripts/test/rt_player_resolution_ipc.py
Checks supported choices, atomic rejection and presentation/render independence.
"""
import time

from rt_ipc import PLAYER_PIPE_NAME, RtIpc, RtIpcError


def wait_render(ipc, expected):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        status = ipc.call("player.display")
        if status["render"] == expected:
            return status
        time.sleep(0.1)
    raise AssertionError(f"render did not become {expected}: {status}")


def run():
    ipc = RtIpc(pipe_name=PLAYER_PIPE_NAME)
    original = ipc.call("player.display")
    modes = original["resolutions"]
    assert modes, "OS display-mode query failed"
    assert len(modes) == len({tuple(mode) for mode in modes})
    recommended = original["recommended_resolution"]
    assert recommended in modes
    if [1920, 1080] in modes:
        assert recommended == [1920, 1080]
    try:
        ipc.call("player.set_display", render_width=recommended[0],
                 render_height=recommended[1])
        selected = wait_render(ipc, recommended)
        assert selected["render_scale"] == 1.0
        for bad in ({"render_width": 99999, "render_height": 99999},
                    {"render_width": recommended[0]},
                    {"render_width": recommended[0] + 0.5, "render_height": recommended[1]},
                    {"render_width": "1920", "render_height": 1080},
                    {"render_width": None, "render_height": None}):
            try:
                ipc.call("player.set_display", hud=not selected["hud"], **bad)
            except RtIpcError:
                pass
            else:
                raise AssertionError(f"accepted invalid resolution: {bad}")
            after = ipc.call("player.display")
            assert after["hud"] == selected["hud"]
            assert after["render_base"] == recommended
        for settings in ({"fullscreen": False, "width": 800, "height": 600},
                         {"fullscreen": True}, {"fullscreen": False}):
            ipc.call("player.set_display", **settings)
            time.sleep(0.3)
            after = ipc.call("player.display")
            assert after["render_base"] == recommended
            assert after["render"] == recommended
        # Also exercise a second supported mode without resizing presentation.
        mode = modes[0]
        before = ipc.call("player.display")
        ipc.call("player.set_display", render_width=mode[0], render_height=mode[1])
        after = wait_render(ipc, mode)
        assert after["window"] == before["window"]
        print("PASS: resolution choices, validation, resize/fullscreen independence")
    finally:
        # Restore window size before fullscreen; width/height are windowed-only.
        ipc.call("player.set_display", fullscreen=False,
                 width=original["window"][0], height=original["window"][1])
        base = original["render_base"]
        ipc.call("player.set_display", render_width=base[0], render_height=base[1],
                 render_scale=original["render_scale"], fullscreen=original["fullscreen"],
                 hud=original["hud"])
        ipc.close()


if __name__ == "__main__":
    run()
