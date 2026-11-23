"""Compile the Android Player's shader set for a Vulkan 1.1 target (SPIR-V 1.3).

Why: the Windows SPIR-V is built for vulkan1.2/1.3 (SPIR-V 1.5/1.6). A Vulkan 1.1
device (Mali-G76 Omix X700, Adreno 610) may only consume SPIR-V 1.3. Android uses this
set on EVERY device (one path); the code it runs is the same GLSL, only the SPIR-V
version differs. Windows keeps x64/Release/shaders untouched.

Set   : the shaders packaged in dist/RayTrophiPlayer/shaders (the Player's list).
Flags : RayTrophiStudio/compile_shaders.ps1 -ListPlan (the single per-shader flag table),
        with --target-env replaced by vulkan1.1.
Output: x64/Release/shaders_vk11/<name>.spv. stage_player_android_assets.py prefers it.

Shaders that cannot target 1.1 (ray tracing stages, ray queries need SPIR-V 1.4) are
reported and keep their Release copy: a 1.1 device has no ray tracing / ray query and
never loads them. Any OTHER failure is an error, because a 1.1 device would load it.

Usage: python scripts/compile_player_android_vk11.py [--force]
"""
import concurrent.futures
import json
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STUDIO = os.path.join(ROOT, "RayTrophiStudio")
DIST_SHADERS = os.path.join(ROOT, "dist", "RayTrophiPlayer", "shaders")
RELEASE = os.path.join(ROOT, "x64", "Release", "shaders")
OUT = os.path.join(ROOT, "x64", "Release", "shaders_vk11")
RT_STAGES = (".rgen", ".rmiss", ".rchit", ".rahit", ".rint", ".rcall")


def glslc_path():
    sdk = os.environ.get("VULKAN_SDK")
    candidates = []
    if sdk:
        candidates.append(os.path.join(sdk, "Bin", "glslc.exe"))
    candidates.append(os.path.join(ROOT, "..", "external_dependencies", "1.3.275.0", "Bin", "glslc.exe"))
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)
    sys.exit("glslc not found (set VULKAN_SDK)")


def load_plan():
    out = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
         os.path.join(STUDIO, "compile_shaders.ps1"), "-ListPlan"],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    if out.returncode != 0:
        sys.exit("compile_shaders.ps1 -ListPlan failed:\n" + out.stdout + out.stderr)
    text = out.stdout.strip().lstrip("﻿")
    plan = json.loads(text)
    return {os.path.basename(e["Target"]).lower(): e for e in plan}


def needs_spirv14(source):
    if source.lower().endswith(RT_STAGES):
        return True
    try:
        with open(source, encoding="utf-8", errors="replace") as f:
            return "GL_EXT_ray_query" in f.read()
    except OSError:
        return False


def main():
    force = "--force" in sys.argv
    glslc = glslc_path()
    plan = load_plan()
    os.makedirs(OUT, exist_ok=True)
    names = sorted(n for n in os.listdir(DIST_SHADERS) if n.lower().endswith(".spv"))
    jobs, missing = [], []
    for name in names:
        entry = plan.get(name.lower())
        if not entry:
            missing.append(name)
            continue
        flags = [f for f in entry["Flags"] if not f.startswith("--target-env=")]
        target = os.path.join(OUT, name)
        release = os.path.join(RELEASE, name)
        fresh = (not force and os.path.isfile(target) and os.path.isfile(release) and
                 os.path.getmtime(target) >= os.path.getmtime(release) and
                 os.path.getmtime(target) >= os.path.getmtime(entry["Source"]))
        jobs.append((name, entry["Source"], flags, target, fresh))

    def compile_one(job):
        name, source, flags, target, fresh = job
        if fresh:
            return name, "fresh", ""
        tmp = target + ".tmp"
        r = subprocess.run([glslc, source, "-o", tmp, "--target-env=vulkan1.1"] + flags,
                           capture_output=True, text=True, errors="replace")
        if r.returncode != 0:
            if os.path.exists(tmp):
                os.remove(tmp)
            return name, "fail", (r.stdout + r.stderr).strip().splitlines()[:3]
        os.replace(tmp, target)
        return name, "ok", ""

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        results = list(pool.map(compile_one, jobs))

    sources = {j[0]: j[1] for j in jobs}
    ok = sum(1 for _, s, _ in results if s == "ok")
    fresh = sum(1 for _, s, _ in results if s == "fresh")
    expected, errors = [], []
    for name, status, detail in results:
        if status != "fail":
            continue
        (expected if needs_spirv14(sources[name]) else errors).append((name, detail))
    print("vk11: %d compiled, %d up to date, %d need SPIR-V 1.4 (kept 1.2 copy), %d errors"
          % (ok, fresh, len(expected), len(errors)))
    for name, _ in expected:
        print("  1.4-only (ray tracing / ray query): " + name)
    if missing:
        print("  not in compile_shaders plan (kept Release copy): " + ", ".join(missing))
    for name, detail in errors:
        print("ERROR " + name)
        for line in detail:
            print("    " + line)
    # Stale outputs for shaders that left the set would be staged by name; remove them.
    keep = {j[0].lower() for j in jobs}
    for n in os.listdir(OUT):
        if n.lower().endswith(".spv") and n.lower() not in keep:
            os.remove(os.path.join(OUT, n))
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
