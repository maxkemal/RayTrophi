"""Capture an open Windows Player or compare two lossless captures."""

import argparse
import base64
from pathlib import Path
import time

CAPTURE_DIR = Path("docs/dev/.captures")


def capture_viewport():
    from rt_ipc import PLAYER_PIPE_NAME, RtIpc

    ipc = RtIpc(PLAYER_PIPE_NAME)
    enabled = ipc.call("viewport.status")["capture_enabled"]
    try:
        ipc.call("viewport.capture", enabled=True)
        time.sleep(0.3)
        result = ipc.call("viewport.get_screenshot")
        return base64.b64decode(result["image_base64"].split(",")[-1])
    finally:
        ipc.call("viewport.capture", enabled=enabled)


def compare(before, after):
    from PIL import Image, ImageChops

    with Image.open(before) as source, Image.open(after) as target:
        if source.size != target.size:
            print(f"Capture sizes differ: {source.size} vs {target.size}")
            return 1
        diff = ImageChops.difference(source.convert("RGB"), target.convert("RGB"))
        changed_pixels = sum(pixel != (0, 0, 0) for pixel in diff.getdata())
        max_channel_delta = max(high for _, high in diff.getextrema())
        print(f"changed_pixels={changed_pixels} max_channel_delta={max_channel_delta}")
        return int(changed_pixels != 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="player-ghost.png")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"), type=Path)
    mode.add_argument(
        "--screen-box", nargs=4, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"), type=int,
        help="Capture visible desktop pixels in physical coordinates (exclude dynamic UI).",
    )
    args = parser.parse_args()
    if args.compare:
        return compare(*args.compare)
    if Path(args.name).name != args.name or not args.name.lower().endswith(".png"):
        parser.error("--name must be a PNG filename inside docs/dev/.captures")

    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    path = CAPTURE_DIR / args.name
    if args.screen_box:
        from PIL import ImageGrab

        left, top, right, bottom = args.screen_box
        if right <= left or bottom <= top:
            parser.error("--screen-box must have a positive width and height")
        ImageGrab.grab(bbox=tuple(args.screen_box), all_screens=True).save(path)
    else:
        path.write_bytes(capture_viewport())
    print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
