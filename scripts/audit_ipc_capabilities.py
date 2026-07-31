#!/usr/bin/env python3
"""Audit the remote-IPC capability table against the dispatch chain.

RtIpc.cpp::dispatchMethod is a hand-written `if (method == "...")` chain and
RtIpcSecurity.cpp::requiredCapabilities classifies methods with a namespace
table. Nothing links the two, so they drift: a method missing from the table
gets 0 required capabilities and authorize() rejects it fail-closed, which
looks exactly like an unrelated permission problem from the client side.
(That is how `lights.` / `nodes.` / `modifiers.` / `anim.` were lost behind
singular `light.` / `node.` / `modifier.` prefixes for 14 write methods.)

The namespace table is parsed out of the source rather than duplicated here,
so this script keeps working when namespaces are added. The handful of
special cases above it are mirrored — keep them in sync if they change.

Usage:  python scripts/audit_ipc_capabilities.py
Exit code 1 if anything is unreachable or the table has dead entries.
"""

import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API = os.path.join(ROOT, "RayTrophiStudio", "source", "src", "Api")
IPC = os.path.join(API, "RtIpc.cpp")
SECURITY = os.path.join(API, "RtIpcSecurity.cpp")

# `batch` is authorized per child call instead of at the top level
# (RtIpc.cpp, "batch" branch), so a 0 here is correct, not a finding.
EXEMPT = {"batch"}


def read(path):
    return io.open(path, encoding="utf-8", errors="replace").read()


def dispatched_methods(source):
    return sorted(set(re.findall(r'method\s*==\s*"([^"]+)"', source)))


def namespace_table(source):
    """Pull the `namespaces[]` array out of requiredCapabilities()."""
    match = re.search(r"namespaces\[\]\s*=\s*\{(.*?)\};", source, re.S)
    if not match:
        sys.exit("could not find namespaces[] in " + SECURITY)
    return re.findall(r'"([^"]+)"', match.group(1))


def required(method, namespaces):
    """Mirror of RtIpcSecurity.cpp::requiredCapabilities (name-level only)."""
    if method == "ipc.admin.audit.export":
        return "Admin|FilesWrite"
    if method.startswith("ipc.admin."):
        return "Admin"
    if method == "script.run_file":
        return "Scripts|FilesRead"
    if method in ("addons.enable", "addons.disable", "addons.reload"):
        return "Addons"
    if method in ("project.open", "scene.import_model",
                  "terrain.import_heightmap", "paint.import_channel"):
        return "FilesRead|SceneWrite"
    if method in ("project.save", "terrain.export_heightmap",
                  "paint.export_channel"):
        return "FilesWrite"
    if method.startswith("render.") or method in ("request_render",
                                                  "reset_accumulation"):
        return "Render"
    if method in ("material.info", "material.of_object", "material.textures",
                  "nodes.graphs", "forcefield.evaluate", "particle.stats",
                  "particle.emitters", "anim.characters", "anim.character",
                  "anim.clips", "anim.graph_status"):
        return "Read"
    if (method in ("version", "project.path", "undo_description",
                   "redo_description")
            or ".get" in method or ".list" in method or ".status" in method
            or ".types" in method or ".object_exists" in method
            or ".sample_height" in method):
        return "Read"
    if any(method.startswith(prefix) for prefix in namespaces):
        return "SceneWrite"
    return None


def main():
    methods = dispatched_methods(read(IPC))
    namespaces = namespace_table(read(SECURITY))

    unreachable = [m for m in methods
                   if required(m, namespaces) is None and m not in EXEMPT]
    dead = [p for p in namespaces
            if not any(m.startswith(p) for m in methods)]

    print("%d dispatched methods, %d namespace prefixes" %
          (len(methods), len(namespaces)))
    if unreachable:
        print("\nUNREACHABLE over remote IPC (requiredCapabilities == 0):")
        for m in unreachable:
            print("  " + m)
    if dead:
        print("\nDEAD namespace prefixes (no method uses them):")
        for p in dead:
            print("  " + p)
    if not unreachable and not dead:
        print("OK - every dispatched method is classified, no dead prefixes.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
