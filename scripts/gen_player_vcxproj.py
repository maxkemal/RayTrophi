#!/usr/bin/env python3
"""Generate RayTrophiPlayer.vcxproj (+ .filters) from the Studio project.

  python scripts/gen_player_vcxproj.py           # write the Player project
  python scripts/gen_player_vcxproj.py --check   # FAIL if the checked-in copy is stale

The Player project is a GENERATED file: do not edit it by hand. It is derived
from two sources of truth so the two executables cannot drift apart silently:

  * RayTrophiStudio/RayTrophiStudio.vcxproj - build settings (configurations,
    compiler flags, per-file metadata such as ExceptionHandling or Jolt's
    ObjectFileName) and the list of compiled sources.
  * scripts/player_include_manifest.json - which first-party sources belong to
    the Player (include/exclude globs, same rule as audit_player_includes.py)
    and `player_link_alternatives`: sources Studio lists as None that the
    Player compiles instead of an offline/Studio implementation (Host
    adapters, PlayerMain.cpp).

Player = third-party sources compiled by Studio
       + Studio first-party ClCompile sources inside the manifest
       + link alternatives.

Settings transform: CUDA/OptiX, OIDN, Embree and Python include paths,
libraries, delay-loads, the CUDA build customization and the embedded Python
copy step are removed; RT_NO_CUDA is defined; intermediates go to a separate
IntDir so the two projects never share object files. Output directory is the
same as Studio (x64/<Config>/) so shaders, PTX-free assets and runtime DLLs are
shared. Plan: docs/dev/RAYTROPHI_PLAYER_PLAN.md
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.join(REPO, "RayTrophiStudio")
STUDIO = os.path.join(PROJECT_DIR, "RayTrophiStudio.vcxproj")
PLAYER = os.path.join(PROJECT_DIR, "RayTrophiPlayer.vcxproj")
PLAYER_FILTERS = PLAYER + ".filters"
MANIFEST = os.path.join(REPO, "scripts", "player_include_manifest.json")

PLAYER_GUID = "{7E2C4B1A-3D5F-4A8E-9C61-2B7F0D4E9A53}"
SRC_PREFIX = "source\\src\\"

# Exact substrings removed from the copied settings. Each entry must match at
# least once, so a renamed macro in the Studio project fails loudly here.
REMOVALS = [
    r"$(ProjectDir)..\vcpkg\installed\x64-windows\include\python$(RayTrophiPythonVersion);",
    r"$(OIDN_ROOT)\include;",
    r"$(OPTIX_ROOT)\include;",
    r"$(CUDA_PATH)\include;",
    r"$(EMBREE_ROOT)\include;",
    r"$(OIDN_ROOT)\lib;",
    r"$(CUDA_PATH)\lib\x64;",
    r"$(EMBREE_ROOT)\lib;",
    r"$(RayTrophiPythonLib);",
    r"OpenImageDenoise_core.lib;OpenImageDenoise.lib;embree4.lib;",
    r"cuda.lib;cudart.lib;cufft.lib;",
    r"nvcuda.dll;cudart64_12.dll;cufft64_11.dll;",
    '    <Import Project="$(VCTargetsPath)\\BuildCustomizations\\CUDA 12.6.props" />\r\n',
]

# Anything left that matches these means the transform missed a dependency.
FORBIDDEN_AFTER = re.compile(
    r"OIDN_ROOT|OPTIX_ROOT|CUDA_PATH|EMBREE_ROOT|RayTrophiPython|OpenImageDenoise|"
    r"embree4|cudart|cufft|cuda\.lib|nvcuda|CUDA 12", re.I)


def fail(msg: str) -> None:
    raise SystemExit("gen_player_vcxproj: " + msg)


def load_manifest() -> dict:
    with open(MANIFEST, "r", encoding="utf-8") as fh:
        return json.load(fh)


def in_player_set(relpath: str, manifest: dict) -> bool:
    if not any(fnmatch.fnmatch(relpath, g) for g in manifest["include"]):
        return False
    return not any(fnmatch.fnmatch(relpath, g) for g in manifest["exclude"])


def manifest_rel(item_path: str) -> str:
    # "source\src\Core\Foo.cpp" -> "src/Core/Foo.cpp" (manifest is relative to source/)
    return item_path[len("source\\"):].replace("\\", "/")


ITEM_RE = re.compile(
    r'^[ \t]*<ClCompile Include="(?P<path>[^"]+)"(?P<attrs>[^>]*?)(?:/>|>(?P<body>.*?)</ClCompile>)[ \t]*\r?\n',
    re.S | re.M)
NONE_RE = re.compile(r'<None Include="([^"]+)"')


def build_prolog(studio: str) -> str:
    last_def = studio.rfind("</ItemDefinitionGroup>")
    if last_def < 0:
        fail("no ItemDefinitionGroup in the Studio project")
    end = studio.find("\r\n", last_def) + 2
    prolog = studio[:end]

    for token in REMOVALS:
        if token not in prolog:
            fail("expected Studio setting not found (project changed?): " + token.strip())
        prolog = prolog.replace(token, "")

    # CUDA item definitions are meaningless without the CUDA customization.
    prolog = re.sub(r"[ \t]*<CudaCompile>.*?</CudaCompile>\r\n", "", prolog, flags=re.S)
    # Embedded Python macros: nothing in the Player references them.
    prolog = re.sub(r"[ \t]*<RayTrophiPython\w+[^>]*>[^<]*</RayTrophiPython\w+>\r\n", "", prolog)

    replacements = [
        ("<ProjectGuid>{5bac7720-07df-4c9c-b8ac-aaf8fe410cd8}</ProjectGuid>",
         "<ProjectGuid>" + PLAYER_GUID.lower() + "</ProjectGuid>"),
        ("<RootNamespace>RayTrophiStudio</RootNamespace>", "<RootNamespace>RayTrophiPlayer</RootNamespace>"),
        ("<ProjectName>RayTrophi Studio</ProjectName>", "<ProjectName>RayTrophi Player</ProjectName>"),
        ("<TargetName>RayTrophiStudio</TargetName>",
         "<TargetName>RayTrophiPlayer</TargetName>\r\n"
         "    <IntDir>$(Platform)\\$(Configuration)\\Player\\</IntDir>"),
    ]
    for old, new in replacements:
        if prolog.count(old) != 1:
            fail("expected exactly one occurrence of " + old)
        prolog = prolog.replace(old, new)

    # SDL2main supplies WinMain -> SDL_main for the GUI subsystem. Release
    # builds must not allocate a console window; Debug retains console output.
    def release_subsystem(match):
        block = match.group(0)
        condition = block.split(">", 1)[0]
        if "Release" in condition:
            block = block.replace("<SubSystem>Console</SubSystem>",
                                  "<SubSystem>Windows</SubSystem>")
        return block

    prolog = re.sub(r"<ItemDefinitionGroup\b[^>]*>.*?</ItemDefinitionGroup>",
                    release_subsystem, prolog, flags=re.S)

    # Target identity is independent of the CUDA SDK boundary.
    prolog, n = re.subn(r"<PreprocessorDefinitions>(?!RT_NO_CUDA;)",
                        "<PreprocessorDefinitions>RT_PLAYER;RT_NO_CUDA;", prolog)
    if n == 0:
        fail("no PreprocessorDefinitions to extend")

    leftover = FORBIDDEN_AFTER.search(prolog)
    if leftover:
        fail("forbidden dependency left in Player settings: " + leftover.group(0))

    header = ("<!-- GENERATED by scripts/gen_player_vcxproj.py from RayTrophiStudio.vcxproj and\r\n"
              "     scripts/player_include_manifest.json. Do not edit; rerun the generator. -->\r\n")
    first_nl = prolog.find("\r\n") + 2  # after the <?xml ...?> declaration
    return prolog[:first_nl] + header + prolog[first_nl:]


def collect_items(studio: str, manifest: dict):
    items = []          # (path, raw element text)
    excluded = []
    for m in ITEM_RE.finditer(studio):
        path = m.group("path")
        raw = m.group(0)
        if not raw.endswith("\r\n"):
            raw = raw.rstrip("\n") + "\r\n"
        if not path.startswith(SRC_PREFIX):
            items.append((path, raw))            # third-party: Jolt, ImGui, ozz, miniz, xatlas
        elif in_player_set(manifest_rel(path), manifest):
            items.append((path, raw))
        else:
            excluded.append(path)

    studio_none = set(NONE_RE.findall(studio))
    alternatives = manifest.get("player_link_alternatives", [])
    for rel in alternatives:
        path = "source\\" + rel.replace("/", "\\")
        if not os.path.isfile(os.path.join(PROJECT_DIR, path)):
            fail("link alternative does not exist: " + rel)
        if path not in studio_none:
            fail("link alternative must be listed as None in the Studio project: " + rel)
        if not in_player_set(rel, manifest):
            fail("link alternative is outside the manifest Player set: " + rel)
        items.append((path, '    <ClCompile Include="%s" />\r\n' % path))

    paths = [p for p, _ in items]
    dups = sorted({p for p in paths if paths.count(p) > 1})
    if dups:
        fail("duplicate sources: " + ", ".join(dups))
    items.sort(key=lambda it: it[0].lower())
    return items, excluded


def build_project(studio: str, manifest: dict):
    prolog = build_prolog(studio)
    items, excluded = collect_items(studio, manifest)
    body = ["  <ItemGroup>\r\n"] + [raw for _, raw in items] + ["  </ItemGroup>\r\n"]
    rc = re.findall(r'[ \t]*<ResourceCompile Include="[^"]+" />\r\n', studio)
    if rc:
        body += ["  <ItemGroup>\r\n"] + rc + ["  </ItemGroup>\r\n"]
    epilog = (
        '  <Import Project="$(VCTargetsPath)\\Microsoft.Cpp.targets" />\r\n'
        '  <Import Project="$(ProjectDir)third_party\\FidelityFX\\1.1.4\\FidelityFX.targets" />\r\n'
        "</Project>\r\n")
    return prolog + "".join(body) + epilog, items, excluded


def build_filters(items) -> str:
    folders = set()
    lines = []
    for path, _ in items:
        folder = os.path.dirname(path)
        f = folder
        while f:
            folders.add(f)
            f = os.path.dirname(f)
        lines.append('    <ClCompile Include="%s">\r\n      <Filter>%s</Filter>\r\n    </ClCompile>\r\n'
                     % (path, folder))
    out = ['<?xml version="1.0" encoding="utf-8"?>\r\n',
           '<Project ToolsVersion="4.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">\r\n',
           "  <ItemGroup>\r\n"]
    for f in sorted(folders, key=str.lower):
        out.append('    <Filter Include="%s" />\r\n' % f)
    out.append("  </ItemGroup>\r\n  <ItemGroup>\r\n")
    out += lines
    out.append("  </ItemGroup>\r\n</Project>\r\n")
    return "".join(out)


def read_bytes(path: str) -> bytes:
    try:
        with open(path, "rb") as fh:
            return fh.read()
    except FileNotFoundError:
        return b""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="fail if the generated files are stale")
    args = ap.parse_args()

    with open(STUDIO, "rb") as fh:
        studio = fh.read().decode("utf-8-sig")
    if "\r\n" not in studio:
        fail("Studio project is expected to use CRLF line endings")
    manifest = load_manifest()

    project, items, excluded = build_project(studio, manifest)
    filters = build_filters(items)
    outputs = {PLAYER: ("﻿" + project).encode("utf-8"),
               PLAYER_FILTERS: ("﻿" + filters).encode("utf-8")}

    first_party = sum(1 for p, _ in items if p.startswith(SRC_PREFIX))
    print("Player sources: %d (first-party %d, third-party %d); Studio-only first-party: %d"
          % (len(items), first_party, len(items) - first_party, len(excluded)))

    if args.check:
        stale = [os.path.basename(p) for p, data in outputs.items() if read_bytes(p) != data]
        if stale:
            print("FAIL: stale generated file(s): " + ", ".join(stale) +
                  " - run python scripts/gen_player_vcxproj.py")
            return 1
        print("PASS: Player project is up to date")
        return 0

    for path, data in outputs.items():
        with open(path, "wb") as fh:
            fh.write(data)
        print("wrote " + os.path.relpath(path, REPO))
    return 0


if __name__ == "__main__":
    sys.exit(main())
