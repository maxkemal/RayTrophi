"""Wet paint canvas over a tiled base (docs/dev/WET_PAINT_CANVAS_UV.md), over IPC.

Run on a DISPOSABLE copy of a game scene (Studio or Player) whose wall has a base
colour texture. Tiling the wall's material must switch that surface to
canvas_mode 'overlay' and KEEP the tiling on the material; tiling 1 must switch it
back. A surface without a base texture must get its tiling reset to 1.

  python scripts/test/rt_wet_paint_tiled_canvas_ipc.py --object PaintBack [--player]
      [--tile 6] [--roundtrip E:/tmp/tiled.rtp]

Pixels are not compared here: look at the wall while the script holds the tiled
state (--hold seconds): the wallpaper repeats, the hidden picture/paint do not.
"""
import argparse
import time
from pathlib import Path

from rt_ipc import RtIpc, PLAYER_PIPE_NAME, PIPE_NAME


def surface(ipc, obj):
    for s in ipc.call("paint.wet_status")["surfaces"]:
        if s["object"] == obj:
            return s
    raise AssertionError(f"{obj} is not a wet paint surface")


def set_tile(ipc, obj, value):
    for axis in ("uv_scale_x", "uv_scale_y"):
        ipc.call("material.set", object_name=obj, param=axis, value=float(value))


def wait_mode(ipc, obj, mode, timeout=5.0):
    deadline = time.monotonic() + timeout
    while True:
        s = surface(ipc, obj)
        if s.get("canvas_mode") == mode:
            return s
        assert time.monotonic() < deadline, f"{obj}: canvas_mode {s.get('canvas_mode')!r}, expected {mode!r}"
        time.sleep(0.1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", required=True)
    parser.add_argument("--player", action="store_true")
    parser.add_argument("--tile", type=float, default=6.0)
    parser.add_argument("--hold", type=float, default=0.0, help="seconds to keep the tiled state for a look")
    parser.add_argument("--roundtrip", help="New disposable .rtp path: save tiled, reopen, expect overlay")
    args = parser.parse_args()
    ipc = RtIpc(PLAYER_PIPE_NAME if args.player else PIPE_NAME)

    start = surface(ipc, args.object)
    assert "canvas_mode" in start, "old executable: paint.wet_status has no canvas_mode"
    print(f"start: {start['canvas_mode']} base={start.get('base_texture')!r}")

    set_tile(ipc, args.object, args.tile)
    time.sleep(0.5)
    tiled = surface(ipc, args.object)
    scale = ipc.call("material.get", object_name=args.object, param="uv_scale_x")
    if tiled["canvas_mode"] != "overlay":
        # No base texture under the canvas: the tiling may only tile the paint
        # itself, so it is undone.
        tiled = wait_mode(ipc, args.object, "composite")
        scale = ipc.call("material.get", object_name=args.object, param="uv_scale_x")
        assert abs(scale - 1.0) < 1e-4, f"untextured canvas kept uv_scale {scale}"
        print("PASS (no base texture): tiling reset to 1, composite kept")
        return
    assert tiled["base_texture"], "overlay without a base_texture name"
    assert abs(scale - args.tile) < 1e-4, f"overlay must keep the tiling, uv_scale_x={scale}"
    print(f"overlay: base={tiled['base_texture']!r} uv_scale_x={scale}")
    if args.hold > 0:
        print(f"holding {args.hold:.0f} s: wallpaper repeats, paint and picture cover the wall once")
        time.sleep(args.hold)

    if args.roundtrip:
        path = Path(args.roundtrip).resolve()
        assert not path.exists(), "Use a new test project path"
        ipc.call("project.save", path=str(path))
        ipc.call("project.open", path=str(path))
        deadline = time.monotonic() + 90
        while ipc.call("paint.wet_design")["pending"]:
            assert time.monotonic() < deadline, "paint restoration timed out"
            time.sleep(0.1)
        wait_mode(ipc, args.object, "overlay", timeout=10.0)
        scale = ipc.call("material.get", object_name=args.object, param="uv_scale_x")
        assert abs(scale - args.tile) < 1e-4, f"reload lost the tiling, uv_scale_x={scale}"
        print("roundtrip: overlay and tiling survive save/open")

    set_tile(ipc, args.object, 1.0)
    back = wait_mode(ipc, args.object, "composite")
    assert back["base_texture"] == "", back
    print("PASS: tiling -> overlay (tiling kept), tiling 1 -> composite")


if __name__ == "__main__":
    main()
