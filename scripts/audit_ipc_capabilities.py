#!/usr/bin/env python3
"""Audit the remote-IPC capability table against every dispatch module.

RtIpc.cpp and focused RtIpc*.cpp modules use hand-written
`if (method == "...")` chains, while RtIpcSecurity.cpp::requiredCapabilities
classifies methods with a namespace table. Nothing links them, so they drift:
a method missing from the table
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
IPC_FILES = [
    os.path.join(API, name)
    for name in os.listdir(API)
    if name.startswith("RtIpc") and name.endswith(".cpp")
    and name not in {"RtIpcSecurity.cpp", "RtIpcAudit.cpp",
                     "RtIpcSession.cpp", "RtIpcTransportLocal.cpp",
                     "RtIpcTransportTls.cpp", "RtIpcPanel.cpp"}
]
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
    # viewport.* drives and measures the engine (not the rt.ui panel-drawing
    # exception). Must come BEFORE the ".status" read heuristic below, or only
    # the query half of the namespace is classified and the command half falls
    # through to None — which is how this mirror first went stale.
    if method.startswith("viewport."):
        return "Render"
    # sim_graph.*: queries are Read, everything that builds the graph is
    # SceneWrite. Same ordering lesson as viewport.* — the read heuristics below
    # would otherwise classify only part of the namespace.
    if method in ("sim_graph.nodes", "sim_graph.evaluate", "sim_graph.attributes"):
        return "Read"
    if method.startswith("sim_graph."):
        return "SceneWrite"
    if method in ("material.info", "material.of_object", "material.textures",
                  "nodes.graphs", "forcefield.evaluate", "particle.stats",
                  "particle.emitters", "anim.characters", "anim.character",
                  "anim.clips", "anim.graph_status", "msf.substances",
                  "msf.fields", "templates.refresh", "templates.validate",
                  "templates.prepare"):
        return "Read"
    # agent.chat_send posts into the user's chat panel under a sender name the
    # caller chooses, so it is not part of the read-only agent.* surface.
    if method == "agent.chat_send":
        return "AgentChat"
    if (method in ("version", "project.path", "undo_description",
                   "redo_description")
            or ".get" in method or ".list" in method or ".status" in method
            or ".types" in method or ".object_exists" in method
            or ".sample_height" in method or method.startswith("agent.")):
        return "Read"
    if any(method.startswith(prefix) for prefix in namespaces):
        return "SceneWrite"
    return None


def main():
    methods = dispatched_methods("\n".join(read(path) for path in IPC_FILES))
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
    # Second pass: the descriptor table that answers agent.describe is generated
    # from these same dispatch sources. If it is stale, a method exists over IPC
    # but has no schema - which reads to an agent as "this capability does not
    # exist" and is exactly the drift this script was written to catch.
    import gen_ipc_descriptors
    descriptors_stale = gen_ipc_descriptors.main(["--check"]) != 0

    if not unreachable and not dead and not descriptors_stale:
        print("OK - every dispatched method is classified, no dead prefixes, "
              "descriptors current.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
