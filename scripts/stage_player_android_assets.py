#!/usr/bin/env python3
"""Stage the Android Player APK assets from the ONE Player package (dist/RayTrophiPlayer).

Packaging rule (docs/dev/AJAN_KOORDINASYON.md rule 7): there is a single Player package,
built by scripts/package_player.ps1. Android does not get a parallel packaging flow; it takes a
filtered copy of that package:

  shaders/**           compiled SPIR-V, verbatim
  assets/**            AssetRegistry data (i18n refreshed from source)
  scenes/<scene>/**    the chosen scene with its .rtp.bin / sidecars / audio / environment
  assets/player_icon.png  the app icon (largest .ico frame), shown in the game menu header
  player_scenes.json   generated: ONE scene (default; the Player opens it at launch) or several
                       (--scenes/--all; the Player then starts on the game library)

Never staged: executables/DLLs, perception runtime, redist, logs/dumps, imgui.ini, and local
configuration that can hold secrets (ipc_tokens.json, player_operator.json, agents_config.json).

Output: android/app/build/rt_assets/ (git-ignored) + rt_asset_manifest.txt whose first line is
"stamp <sha256 of paths+contents>". Platform::extractBundledAssets (PlatformOs.cpp) copies the
listed files into the app's internal storage when the stamp changes.

Demo scenes stay local (never committed) and are read from the local dist folder.
Only staged embedded JPEG/PNG material textures are resized, to 512 pixels by default.

Usage:
  python scripts/stage_player_android_assets.py                     # scene = first in player_scenes.json
  python scripts/stage_player_android_assets.py --scene pastel-paint-workshop
  python scripts/stage_player_android_assets.py --scenes pastel-paint-workshop,kids-play-room
  python scripts/stage_player_android_assets.py --all --max-texture 512
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST = os.path.join(ROOT, "dist", "RayTrophiPlayer")
OUT = os.path.join(ROOT, "android", "app", "build", "rt_assets")
MANIFEST = "rt_asset_manifest.txt"

# Inside a staged scene folder: documentation/test notes are not runtime data.
SCENE_SKIP_SUFFIXES = (".md", ".TEST.md")
DEFAULT_MAX_TEXTURE = 512
# Environment maps on Android: a 4096x2048 HDRI is 64 MB of RGBA16F GPU memory and
# is sampled unmipped (sky, transmission cone taps), which thrashes a mobile texture
# cache. 2048 is 16 MB; the PC package keeps the original (quality floor rule: the
# device class is raised, the authored asset is not degraded on hardware that can
# carry it).
DEFAULT_MAX_HDRI = 2048
ENVIRONMENT_SUFFIXES = (".exr", ".hdr")


def shrink_environment_maps(rels, max_width):
    """Downscale staged environment maps (in place, under OUT) to max_width.

    INTER_AREA = box average, so the integrated radiance (what lights the scene)
    is preserved; only angular detail is lost. Same file name and format, so the
    project's path keeps resolving. Values stay float; EXR is written as half,
    like the source assets.
    """
    targets = [rel for rel in rels if rel.lower().endswith(ENVIRONMENT_SUFFIXES)]
    if not targets:
        return
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    import cv2  # imported lazily: only scenes with an HDRI need it
    for rel in targets:
        path = os.path.join(OUT, rel)
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError("cannot read environment map %s (OpenCV EXR/HDR support)" % rel)
        height, width = image.shape[:2]
        if width <= max_width:
            print("hdri: %s %dx%d kept" % (rel, width, height))
            continue
        new_w = max_width
        new_h = max(1, round(height * max_width / width))
        small = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        params = []
        if rel.lower().endswith(".exr"):
            params = [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF]
        if not cv2.imwrite(path, small, params):
            raise ValueError("cannot write environment map %s" % rel)
        print("hdri: %s %dx%d -> %dx%d (%d KB)" % (rel, width, height, new_w, new_h,
                                                   os.path.getsize(path) // 1024))
EMBEDDED_IMAGE_FORMATS = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG"}


def fail(message: str) -> None:
    print("FAIL: " + message, file=sys.stderr)
    sys.exit(1)


def collect(src_dir: str, rel_prefix: str, skip_suffixes=()) -> list:
    files = []
    for base, dirs, names in os.walk(src_dir):
        dirs.sort()
        for name in sorted(names):
            if name.endswith(skip_suffixes):
                continue
            full = os.path.join(base, name)
            rel = os.path.relpath(full, src_dir).replace("\\", "/")
            files.append((full, rel_prefix + rel))
    return files


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive texture size")
    return number


def normalize_normal_map(image) -> None:
    normalized = []
    for pixel in image.getdata():
        x, y, z = (component / 127.5 - 1.0 for component in pixel[:3])
        length = math.sqrt(x * x + y * y + z * z)
        if length < 0.01:
            x, y, z, length = 0.0, 0.0, 1.0, 1.0
        rgb = tuple(round(127.5 * (component / length + 1.0)) for component in (x, y, z))
        normalized.append(rgb + pixel[3:])
    image.putdata(normalized)


def resize_embedded_texture(payload: bytes, entry: dict, limit: int) -> bytes:
    from PIL import Image

    codec = EMBEDDED_IMAGE_FORMATS.get(entry.get("format", "").lower())
    if codec is None:
        return payload
    with Image.open(io.BytesIO(payload), formats=[codec]) as source:
        width, height = source.size
        if max(width, height) <= limit:
            return payload
        scale = limit / max(width, height)
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        normal = entry.get("usage") == "normal"
        has_alpha = "A" in source.getbands() or "transparency" in source.info
        if normal or source.mode in ("P", "1"):
            source = source.convert("RGBA" if has_alpha else "RGB")
        if normal and has_alpha:
            # Resize encoded vector components independently of opacity.
            image = Image.merge("RGBA", tuple(
                channel.resize(size, Image.Resampling.LANCZOS) for channel in source.split()
            ))
        else:
            image = source.resize(size, Image.Resampling.LANCZOS)
        if normal:
            normalize_normal_map(image)
        output = io.BytesIO()
        options = {"quality": 97, "subsampling": 0} if codec == "JPEG" else {}
        if "icc_profile" in source.info:
            options["icc_profile"] = source.info["icc_profile"]
        image.save(output, format=codec, **options)
    entry["width"], entry["height"] = size
    return output.getvalue()


def stage_scene_textures(project: Path, limit: int) -> None:
    shared = project.with_suffix(".rtp.shared")
    binary = project.with_suffix(".rtp.bin")
    if not shared.is_file():
        print(f"{project.stem}: no shared texture manifest")
        return
    manifest = json.loads(shared.read_bytes())
    textures = manifest.get("textures", [])
    embedded = sorted(
        (entry for entry in textures if entry["mode"] == "embed"),
        key=lambda entry: entry["offset"],
    )
    before = sum(entry["size"] for entry in embedded)
    after = before
    resized = 0
    if embedded:
        # ProjectManager writes textures as the final contiguous section.
        # Leave all earlier geometry and embedded non-texture offsets intact.
        start = embedded[0]["offset"]
        binary_size = binary.stat().st_size
        output_offset = start
        with tempfile.TemporaryFile(dir=binary.parent) as tail, binary.open("r+b") as handle:
            handle.seek(start)
            for entry in embedded:
                offset, size = entry["offset"], entry["size"]
                if offset != handle.tell() or size <= 0 or offset + size > binary_size:
                    raise ValueError(
                        f"{project.name}: invalid embedded texture range (ID {entry['id']})"
                    )
                original = handle.read(size)
                payload = resize_embedded_texture(original, entry, limit)
                resized += payload is not original
                tail.write(payload)
                entry["offset"], entry["size"] = output_offset, len(payload)
                output_offset += len(payload)
            if handle.tell() != binary_size:
                raise ValueError(f"{project.name}: embedded textures must end at the binary tail")
            if resized:
                handle.seek(start)
                tail.seek(0)
                shutil.copyfileobj(tail, handle)
                handle.truncate()
        if resized:
            shared.write_text(json.dumps(manifest, ensure_ascii=False, separators=(",", ":")),
                              encoding="utf-8")
            after = output_offset - start
    print(f"{project.stem}: textures {len(textures)} -> {len(textures)}, "
          f"embedded {len(embedded)} -> {len(embedded)}, "
          f"bytes {before} -> {after}, resized {resized}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene", help="scene folder name under dist/RayTrophiPlayer/scenes")
    parser.add_argument("--scenes", help="comma-separated scene folder names (library shows a card each)")
    parser.add_argument("--all", action="store_true", help="stage every scene listed in dist player_scenes.json")
    parser.add_argument("--max-texture", type=positive_int, default=DEFAULT_MAX_TEXTURE,
                        help="maximum embedded JPEG/PNG edge in pixels (Android default: 512)")
    parser.add_argument("--max-hdri", type=int, default=DEFAULT_MAX_HDRI,
                        help="maximum environment map (.exr/.hdr) width in pixels; 0 keeps the "
                             "original (Android default: %d)" % DEFAULT_MAX_HDRI)
    args = parser.parse_args()

    if not os.path.isdir(DIST):
        fail("dist/RayTrophiPlayer not found - run scripts/package_player.ps1 first")
    with open(os.path.join(DIST, "player_scenes.json"), encoding="utf-8") as handle:
        packaged = json.load(handle).get("scenes", [])
    if not packaged:
        fail("dist player_scenes.json lists no scenes")

    by_folder = {e["path"].replace("\\", "/").split("/")[1]: e for e in packaged}
    if args.all:
        wanted = list(by_folder)
    elif args.scenes:
        wanted = [name.strip() for name in args.scenes.split(",") if name.strip()]
    elif args.scene:
        wanted = [args.scene]
    else:
        wanted = [next(iter(by_folder))]
    for name in wanted:
        if name not in by_folder:
            fail("scene '%s' not in dist player_scenes.json" % name)
    chosen = [(by_folder[name], name) for name in wanted]

    files = []
    # Shader SET from the package (it is the Player's list), CONTENTS from x64/Release/shaders:
    # the APK's libmain.so is built from current source, and an older .spv with a newer binary
    # is a known silent ABI mismatch (binding counts / push sizes).
    fresh_dir = os.path.join(ROOT, "x64", "Release", "shaders")
    # Android consumes the Vulkan 1.1 (SPIR-V 1.3) build of the same GLSL on every device:
    # a 1.1 driver rejects SPIR-V 1.5. scripts/compile_player_android_vk11.py writes it;
    # ray tracing stages it cannot lower keep the Release copy (1.1 devices never load them).
    vk11_dir = os.path.join(ROOT, "x64", "Release", "shaders_vk11")
    vk11_staged = 0
    refreshed = 0
    for full, rel in collect(os.path.join(DIST, "shaders"), "shaders/"):
        fresh = os.path.join(fresh_dir, rel[len("shaders/"):])
        vk11 = os.path.join(vk11_dir, rel[len("shaders/"):])
        if os.path.isfile(vk11) and (not os.path.isfile(fresh) or
                                     os.path.getmtime(vk11) >= os.path.getmtime(fresh)):
            files.append((vk11, rel))
            vk11_staged += 1
            continue
        if os.path.isfile(vk11):
            fail("%s is older than x64/Release/shaders: run scripts/compile_player_android_vk11.py"
                 % os.path.basename(vk11))
        if os.path.isfile(fresh):
            with open(full, "rb") as a, open(fresh, "rb") as b:
                if a.read() != b.read():
                    refreshed += 1
            full = fresh
        files.append((full, rel))
    print("shaders: %d from shaders_vk11 (Vulkan 1.1 / SPIR-V 1.3)" % vk11_staged)
    if not vk11_staged:
        print("note: no Vulkan 1.1 shader set; Vulkan 1.1 devices will reject SPIR-V 1.5. "
              "Run scripts/compile_player_android_vk11.py")
    if refreshed:
        print("note: %d packaged shaders differ from x64/Release/shaders; staged the Release copies "
              "(re-run scripts/package_player.ps1 to refresh the Windows package too)" % refreshed)
    files += [
        (full, rel) for full, rel in collect(os.path.join(DIST, "assets"), "assets/")
        if not rel.startswith("assets/i18n/")
    ]
    files += collect(os.path.join(ROOT, "RayTrophiStudio", "assets", "i18n"), "assets/i18n/")
    icon = os.path.join(ROOT, "RayTrophiStudio", "source", "resources", "RayTrophi_icon.ico")
    if os.path.isfile(icon):
        # SDL_image cannot decode the PNG-compressed frames of a modern .ico: ship the largest
        # frame as PNG instead (written under OUT, which is cleared and re-staged every run).
        from PIL import Image
        with Image.open(icon) as image:
            sizes = sorted(image.info.get("sizes", [image.size]), key=lambda size: size[0] * size[1])
            image.size = sizes[-1]
            png = os.path.join(OUT, "..", "rt_icon_player_icon.png")
            image.convert("RGBA").save(png, "PNG")
        files.append((os.path.abspath(png), "assets/player_icon.png"))
    for entry, folder in chosen:
        files += collect(os.path.join(DIST, "scenes", folder), "scenes/%s/" % folder, SCENE_SKIP_SUFFIXES)
        if not any(rel == entry["path"].replace("\\", "/") for _, rel in files):
            fail("scene file %s missing from the package" % entry["path"])
    if not any(rel.endswith(".spv") for _, rel in files):
        fail("no .spv shaders in the package")

    scene_list = json.dumps(
        {"scenes": [{"name": e.get("name", f), "path": e["path"]} for e, f in chosen]},
        ensure_ascii=False, indent=2).encode("utf-8")

    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    digest = hashlib.sha256()
    total = 0
    for full, rel in files:
        target = os.path.join(OUT, rel)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copyfile(full, target)
    try:
        for entry, _ in chosen:
            stage_scene_textures(Path(OUT) / entry["path"].replace("\\", "/"), args.max_texture)
        if args.max_hdri:
            shrink_environment_maps([rel for _, rel in files if rel.startswith("scenes/")],
                                    args.max_hdri)
    except (ImportError, OSError, ValueError) as error:
        fail(str(error))
    for _, rel in files:
        target = os.path.join(OUT, rel)
        digest.update(rel.encode("utf-8") + b"\0")
        with open(target, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        total += os.path.getsize(target)
    with open(os.path.join(OUT, "player_scenes.json"), "wb") as handle:
        handle.write(scene_list)
    digest.update(b"player_scenes.json\0" + scene_list)
    rels = [rel for _, rel in files] + ["player_scenes.json"]

    with open(os.path.join(OUT, MANIFEST), "w", encoding="utf-8", newline="\n") as handle:
        handle.write("stamp %s\n" % digest.hexdigest())
        for rel in rels:
            handle.write(rel + "\n")
    print("staged %d files, %.1f MiB, scene(s) %s -> %s" % (
        len(rels), total / (1024.0 * 1024.0), ", ".join(f for _, f in chosen), os.path.relpath(OUT, ROOT)))


if __name__ == "__main__":
    main()
