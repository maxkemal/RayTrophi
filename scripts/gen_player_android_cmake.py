#!/usr/bin/env python3
"""Generate android/app/src/main/cpp/player_sources.cmake for the Android Player.

  python scripts/gen_player_android_cmake.py           # write the CMake fragment
  python scripts/gen_player_android_cmake.py --check   # FAIL if the checked-in copy is stale

Same inputs and same selection logic as scripts/gen_player_vcxproj.py (the functions are
imported, not mirrored): RayTrophiStudio.vcxproj + scripts/player_include_manifest.json decide
the Player source set, the Release|x64 settings supply include dirs, defines and C++ standard.
On top of that, scripts/player_android_manifest.json (DATA, no lists in this script) removes
Windows-only files (android_exclude) and adds their stubs (android_stubs).

The fragment is included by the hand-written android/app/src/main/cpp/CMakeLists.txt.
There is no hand-maintained source list anywhere (see CLAUDE.md section 1).
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_player_vcxproj as g  # noqa: E402

REPO = g.REPO
ANDROID_MANIFEST = os.path.join(REPO, "scripts", "player_android_manifest.json")
OUT_DIR = os.path.join(REPO, "android", "app", "src", "main", "cpp")
OUT = os.path.join(OUT_DIR, "player_sources.cmake")
PROJ = "${RT_PROJECT_DIR}"


def fail(msg: str) -> None:
    raise SystemExit("gen_player_android_cmake: " + msg)


def release_group(prolog: str) -> str:
    for m in re.finditer(r"<ItemDefinitionGroup\b[^>]*>.*?</ItemDefinitionGroup>", prolog, re.S):
        if "=='Release|x64'" in m.group(0).split(">", 1)[0]:
            return m.group(0)
    fail("no Release|x64 ItemDefinitionGroup in the Studio project")
    return ""


def tag(block: str, name: str) -> str:
    m = re.search(r"<%s>(.*?)</%s>" % (name, name), block, re.S)
    if not m:
        fail("missing <%s> in the Release|x64 settings" % name)
    return m.group(1)


def xatlas_present() -> bool:
    # Mirrors the RayTrophiXatlasRoot condition in the vcxproj.
    ext = os.path.join(g.PROJECT_DIR, "external", "xatlas")
    return (os.path.isfile(os.path.join(ext, "xatlas.cpp")) and os.path.isfile(os.path.join(ext, "xatlas.h"))) or \
        (os.path.isfile(os.path.join(ext, "source", "xatlas", "xatlas.cpp")) and
         os.path.isfile(os.path.join(ext, "source", "xatlas", "xatlas.h")))


def xatlas_root() -> str:
    ext = os.path.join(g.PROJECT_DIR, "external", "xatlas")
    if os.path.isfile(os.path.join(ext, "xatlas.cpp")) and os.path.isfile(os.path.join(ext, "xatlas.h")):
        return "external/xatlas"
    return "external/xatlas/source/xatlas"


def cmake_path(p: str) -> str:
    p = p.replace("$(ProjectDir)", "").replace("\\", "/")
    return PROJ + "/" + p.lstrip("/")


def include_dirs(settings: str, android: dict) -> list:
    raw = tag(settings, "AdditionalIncludeDirectories")
    has_xatlas = xatlas_present()
    raw = raw.replace("$(RayTrophiXatlasInclude)", (";$(ProjectDir)" + xatlas_root()) if has_xatlas else "")
    out = []
    for tok in raw.split(";"):
        tok = tok.strip()
        if not tok or tok.startswith("%("):
            continue
        if "$(" in tok.replace("$(ProjectDir)", ""):
            fail("unhandled macro in include dir: " + tok)
        path = cmake_path(tok)
        if path not in out:       # vcxproj lists several dirs twice (absolute + relative form)
            out.append(path)
    for directory in android.get("include_dirs", []):
        path = cmake_path(directory)
        if path not in out:
            out.append(path)
    return out


def defines(settings: str, android: dict) -> list:
    raw = tag(settings, "PreprocessorDefinitions")
    raw = raw.replace("$(RayTrophiXatlasDefine)", ";RAYTROPHI_HAS_XATLAS=1" if xatlas_present() else "")
    drop = set(android.get("drop_defines", []))
    out = []
    for tok in raw.split(";"):
        tok = tok.strip()
        if not tok or tok.startswith("%(") or tok in drop:
            continue
        if "$(" in tok:
            fail("unhandled macro in define: " + tok)
        out.append(tok)
    for tok in android.get("add_defines", []):
        if tok not in out:
            out.append(tok)
    return out


def cxx_standard(settings: str) -> str:
    m = re.fullmatch(r"stdcpp(\d+)", tag(settings, "LanguageStandard"))
    if not m:
        fail("unsupported LanguageStandard in the Studio project")
    return m.group(1)


def source_set(studio: str, manifest: dict, android: dict):
    items, _ = g.collect_items(studio, manifest)
    excludes = android.get("android_exclude", [])
    kept = []
    removed = []
    for path, raw in items:
        rel = g.manifest_rel(path) if path.startswith(g.SRC_PREFIX) else path.replace("\\", "/")
        if any(fnmatch.fnmatch(rel, pat) for pat in excludes):
            removed.append(rel)
            continue
        kept.append((path, raw))

    for pat in excludes:
        if not any(fnmatch.fnmatch(r, pat) for r in removed):
            print("note: android_exclude entry matches nothing in the Player set: " + pat)

    stubs = []
    for entry in android.get("android_stubs", []):
        stub = entry["stub"]
        if not os.path.isfile(os.path.join(g.PROJECT_DIR, "source", stub.replace("/", os.sep))):
            fail("stub does not exist: " + stub)
        for pat in entry["replaces"]:
            if pat not in excludes:
                fail("stub '%s' replaces '%s' which is not listed in android_exclude" % (stub, pat))
        stubs.append(stub)
    if len(set(stubs)) != len(stubs):
        fail("duplicate stub entries")

    # Sources that exist only conditionally (xatlas has two possible layouts).
    final = []
    for path, raw in kept:
        if not os.path.isfile(os.path.join(g.PROJECT_DIR, path)):
            if "xatlas" in path:
                continue
            fail("source listed by the project does not exist: " + path)
        sync = "<ExceptionHandling>Sync</ExceptionHandling>" in raw
        final.append((path.replace("\\", "/"), sync))
    for stub in stubs:
        final.append(("source/" + stub, False))
    final.sort(key=lambda it: it[0].lower())
    return final, removed, stubs


def build(studio: str, manifest: dict, android: dict):
    prolog = g.build_prolog(studio)
    settings = release_group(prolog)
    files, removed, stubs = source_set(studio, manifest, android)
    inc = include_dirs(settings, android)
    defs = defines(settings, android)
    std = cxx_standard(settings)
    exception_sources = {"source/" + path for path in android.get("exception_sources", [])}
    missing = exception_sources - {path for path, _ in files}
    if missing:
        fail("exception source not in the Android source set: " + ", ".join(sorted(missing)))
    sync = [path for path, enabled in files if enabled or path in exception_sources]

    L = []
    w = L.append
    w("# GENERATED by scripts/gen_player_android_cmake.py from RayTrophiStudio.vcxproj,")
    w("# scripts/player_include_manifest.json and scripts/player_android_manifest.json.")
    w("# Do not edit; rerun the generator. Included by CMakeLists.txt (RT_PROJECT_DIR must be set).")
    w("")
    w("set(RT_CXX_STANDARD %s)" % std)
    w("set(RT_ABIS %s)" % " ".join(android["abis"]))
    w("")
    w("set(RT_PLAYER_DEFINES")
    for d in defs:
        w("    " + d)
    w(")")
    w("")
    w("set(RT_PLAYER_INCLUDE_DIRS")
    for d in inc:
        w('    "%s"' % d)
    w(")")
    w("")
    w("set(RT_CXX_FLAGS_COMMON %s)" % " ".join(android.get("cxx_flags_common", [])))
    for abi, flags in sorted(android.get("cxx_flags_per_abi", {}).items()):
        w("set(RT_CXX_FLAGS_%s %s)" % (abi.replace("-", "_"), " ".join(flags)))
    w("set(RT_SYSTEM_LIBS %s)" % " ".join(android["link_system_libs"]))
    w("")
    w("# Dependencies the integrator must provide (see android/README.md).")
    for dep in android["deps"]:
        w("set(RT_DEP_%s_KIND %s)" % (dep["var"], dep["kind"]))
    w("set(RT_DEP_LIST %s)" % " ".join(d["var"] for d in android["deps"]))
    for dep in android["deps"]:
        if dep["kind"] == "subproject":
            w("set(RT_DEP_%s_TARGET %s)" % (dep["var"], dep["target"]))
        else:
            w("set(RT_DEP_%s_LIBS %s)" % (dep["var"], " ".join(dep["libs"])))
    w("")
    w("# Player sources (%d, of which %d are Android stubs)." % (len(files), len(stubs)))
    w("set(RT_PLAYER_SOURCES")
    for p, _ in files:
        w('    "%s/%s"' % (PROJ, p))
    w(")")
    w("")
    w("# Files with ExceptionHandling=Sync or an Android manifest exception override.")
    w("set(RT_PLAYER_EXCEPTION_SOURCES")
    for p in sync:
        w('    "%s/%s"' % (PROJ, p))
    w(")")
    w("")
    w("# Excluded for Android (data: android_exclude in player_android_manifest.json):")
    for r in sorted(removed):
        w("#   " + r)
    return "\n".join(L) + "\n", len(files), removed, stubs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="fail if the generated file is stale")
    args = ap.parse_args()

    with open(g.STUDIO, "rb") as fh:
        studio = fh.read().decode("utf-8-sig")
    manifest = g.load_manifest()
    with open(ANDROID_MANIFEST, "r", encoding="utf-8") as fh:
        android = json.load(fh)

    text, count, removed, stubs = build(studio, manifest, android)
    data = text.encode("utf-8")
    print("Android sources: %d (excluded %d Windows-only, %d stub files)" % (count, len(removed), len(stubs)))

    if args.check:
        try:
            with open(OUT, "rb") as fh:
                current = fh.read()
        except FileNotFoundError:
            current = b""
        if current != data:
            print("FAIL: stale generated file: %s - run python scripts/gen_player_android_cmake.py"
                  % os.path.relpath(OUT, REPO))
            return 1
        print("PASS: Android CMake fragment is up to date")
        return 0

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT, "wb") as fh:
        fh.write(data)
    print("wrote " + os.path.relpath(OUT, REPO))
    return 0


if __name__ == "__main__":
    sys.exit(main())
