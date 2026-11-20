"""Post-build visual acceptance on paint-reveal-rabbit.rtp; clears session paint.

Run externally: python scripts/test/rt_paint_reveal_ipc.py [--player]
Requires Pillow. Captures stay in memory, no screenshot files are created.
"""
import argparse
import base64
import io
import time

from PIL import Image, ImageChops, ImageStat
from rt_ipc import PLAYER_PIPE_NAME, PIPE_NAME, RtIpc, RtIpcError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--player", action="store_true")
    args = parser.parse_args()
    ipc = RtIpc(PLAYER_PIPE_NAME if args.player else PIPE_NAME)
    saved_visibility = {}
    camera = None
    status = None
    shading = None
    rabbit = None
    restore_reveal = False

    def frame():
        time.sleep(1.0)
        response = ipc.call("viewport.get_screenshot")
        return Image.open(io.BytesIO(base64.b64decode(response["image_base64"]))).convert("RGB")

    def difference(a, b):
        return sum(ImageStat.Stat(ImageChops.difference(a, b)).mean) / (3 * 255)

    try:
        names = ipc.call("scene.list_objects")
        rabbits = [name for name in names if "PaintRevealRabbit" in name]
        assert len(rabbits) == 1, "Open paint-reveal-rabbit.rtp first"
        rabbit = rabbits[0]
        settings = ipc.call("paint.wet_get_settings", object=rabbit)
        assert settings.get("reveal_opacity") is True, "Build/reopen the updated reveal fixture"
        restore_reveal = True
        for patch in ({"reveal_opacity": 1}, {"reveal_opacity": "yes"},
                      {"reveal_opacity": False, "unknown_reveal_key": True}):
            try:
                ipc.call("paint.wet_set_settings", object=rabbit, **patch)
            except RtIpcError:
                pass
            else:
                raise AssertionError(f"Invalid setting accepted: {patch}")
            assert ipc.call("paint.wet_get_settings", object=rabbit) == settings
        print("PASS strict bool / unknown fields / rejected edits do not mutate settings")
        status = ipc.call("viewport.status")
        shading = status["shading"]
        camera = ipc.call("camera.get")
        for name in names:
            visible = ipc.call("scene.get_visible", name=name)["visible"]
            saved_visibility[name] = visible
            ipc.call("scene.set_visible", name=name, visible=name in (rabbit, "PaintBack"))
        ipc.call("viewport.set_shading", mode="material")
        ipc.call("camera.set_position", position=[0, 1.75, -.5])
        ipc.call("camera.set_target", target=[0, 1.75, -3.5])
        ipc.call("viewport.capture", enabled=True)
        ipc.call("paint.wet_reset")
        clean = frame()
        ipc.call("scene.set_visible", name=rabbit, visible=False)
        absent = frame()
        invisible_error = difference(clean, absent)
        assert invisible_error < .01, f"Clean rabbit remains visible: {invisible_error}"
        ipc.call("scene.set_visible", name=rabbit, visible=True)
        ipc.call("paint.wet_dab", position=[0, 1.98, -3.24], radius=.42,
                 color=[.02, .45, .95], amount=1.0, normal=[0, 0, 1], facing=True)
        painted = frame()
        paint_difference = difference(painted, absent)
        assert paint_difference > .001, f"Paint did not reveal the rabbit: {paint_difference}"
        print("PASS empty opacity is invisible; real wet dab reveals geometry",
              {"clean_error": invisible_error, "paint_difference": paint_difference})
        ipc.call("paint.wet_reset")
        reset = frame()
        assert difference(reset, absent) < .01, "Reset did not hide the rabbit"
        ipc.call("paint.wet_set_settings", object=rabbit, reveal_opacity=False)
        solid = frame()
        assert difference(solid, absent) > .005, "Disabling reveal did not restore solid opacity"
        ipc.call("paint.wet_set_settings", object=rabbit, reveal_opacity=True)
        assert difference(frame(), absent) < .01, "Re-enabling reveal did not hide clean geometry"
        print("PASS reset hides / disable is solid / re-enable hides; no screenshots saved")
    finally:
        if restore_reveal:
            ipc.call("paint.wet_set_settings", object=rabbit, reveal_opacity=True)
        for name, visible in saved_visibility.items():
            ipc.call("scene.set_visible", name=name, visible=visible)
        if camera:
            ipc.call("camera.set_position", position=camera["position"])
            ipc.call("camera.set_target", target=camera["target"])
        if shading:
            ipc.call("viewport.set_shading", mode=shading)
        if status:
            ipc.call("viewport.capture", enabled=status["capture_enabled"])
        ipc.close()


if __name__ == "__main__":
    main()
