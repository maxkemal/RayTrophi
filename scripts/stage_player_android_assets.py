#!/usr/bin/env python3
"""Stage the Android Player APK assets from the ONE Player package (dist/RayTrophiPlayer).

Packaging rule (docs/dev/AJAN_KOORDINASYON.md rule 7): there is a single Player package,
built by scripts/package_player.ps1. Android does not get a parallel packaging flow; it takes a
filtered copy of that package:

  shaders/**           compiled SPIR-V, verbatim
  assets/**            AssetRegistry data (i18n, ...)
  scenes/<scene>/**    the chosen scene with its .rtp.bin / sidecars / audio / environment
  player_scenes.json   generated: exactly ONE scene, so the Player opens it at launch
                       (the scene library UI is part of the Windows operator layer)

Never staged: executables/DLLs, perception runtime, redist, logs/dumps, imgui.ini, and local
configuration that can hold secrets (ipc_tokens.json, player_operator.json, agents_config.json).

Output: android/app/build/rt_assets/ (git-ignored) + rt_asset_manifest.txt whose first line is
"stamp <sha256 of paths+contents>". Platform::extractBundledAssets (PlatformOs.cpp) copies the
listed files into the app's internal storage when the stamp changes.

Demo scenes stay local (never committed): this script only reads the local dist folder.

Usage:
  python scripts/stage_player_android_assets.py                     # scene = first in player_scenes.json
  python scripts/stage_player_android_assets.py --scene pastel-paint-workshop
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST = os.path.join(ROOT, "dist", "RayTrophiPlayer")
OUT = os.path.join(ROOT, "android", "app", "build", "rt_assets")
MANIFEST = "rt_asset_manifest.txt"

# Inside a staged scene folder: documentation/test notes are not runtime data.
SCENE_SKIP_SUFFIXES = (".md", ".TEST.md")


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene", help="scene folder name under dist/RayTrophiPlayer/scenes")
    args = parser.parse_args()

    if not os.path.isdir(DIST):
        fail("dist/RayTrophiPlayer not found - run scripts/package_player.ps1 first")
    with open(os.path.join(DIST, "player_scenes.json"), encoding="utf-8") as handle:
        packaged = json.load(handle).get("scenes", [])
    if not packaged:
        fail("dist player_scenes.json lists no scenes")

    chosen = None
    for entry in packaged:
        folder = entry["path"].replace("\\", "/").split("/")[1]
        if args.scene is None or folder == args.scene:
            chosen = (entry, folder)
            break
    if chosen is None:
        fail("scene '%s' not in dist player_scenes.json" % args.scene)
    entry, folder = chosen

    files = []
    # Shader SET from the package (it is the Player's list), CONTENTS from x64/Release/shaders:
    # the APK's libmain.so is built from current source, and an older .spv with a newer binary
    # is a known silent ABI mismatch (binding counts / push sizes).
    fresh_dir = os.path.join(ROOT, "x64", "Release", "shaders")
    refreshed = 0
    for full, rel in collect(os.path.join(DIST, "shaders"), "shaders/"):
        fresh = os.path.join(fresh_dir, rel[len("shaders/"):])
        if os.path.isfile(fresh):
            with open(full, "rb") as a, open(fresh, "rb") as b:
                if a.read() != b.read():
                    refreshed += 1
            full = fresh
        files.append((full, rel))
    if refreshed:
        print("note: %d packaged shaders differ from x64/Release/shaders; staged the Release copies "
              "(re-run scripts/package_player.ps1 to refresh the Windows package too)" % refreshed)
    files += collect(os.path.join(DIST, "assets"), "assets/")
    files += collect(os.path.join(DIST, "scenes", folder), "scenes/%s/" % folder, SCENE_SKIP_SUFFIXES)
    if not any(rel.endswith(".spv") for _, rel in files):
        fail("no .spv shaders in the package")
    if not any(rel == entry["path"].replace("\\", "/") for _, rel in files):
        fail("scene file %s missing from the package" % entry["path"])

    scene_list = json.dumps({"scenes": [{"name": entry.get("name", folder), "path": entry["path"]}]},
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
        digest.update(rel.encode("utf-8") + b"\0")
        with open(full, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        total += os.path.getsize(full)
    with open(os.path.join(OUT, "player_scenes.json"), "wb") as handle:
        handle.write(scene_list)
    digest.update(b"player_scenes.json\0" + scene_list)
    rels = [rel for _, rel in files] + ["player_scenes.json"]

    with open(os.path.join(OUT, MANIFEST), "w", encoding="utf-8", newline="\n") as handle:
        handle.write("stamp %s\n" % digest.hexdigest())
        for rel in rels:
            handle.write(rel + "\n")
    print("staged %d files, %.1f MiB, scene '%s' -> %s" % (
        len(rels), total / (1024.0 * 1024.0), entry.get("name", folder), os.path.relpath(OUT, ROOT)))


if __name__ == "__main__":
    main()
