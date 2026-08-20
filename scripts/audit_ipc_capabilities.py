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
    if method.startswith("render."):
        # These two write an image file; the C++ says Render|FilesWrite and a
        # token without FilesWrite is refused at call time, so reporting plain
        # Render would promise a call the engine will reject.
        if method in ("render.start", "render.start_sequence"):
            return "Render|FilesWrite"
        return "Render"
    if method in ("request_render", "reset_accumulation"):
        return "Render"
    # viewport.* drives and measures the engine (not the rt.ui panel-drawing
    # exception). Must come BEFORE the ".status" read heuristic below, or only
    # the query half of the namespace is classified and the command half falls
    # through to None — which is how this mirror first went stale.
    if method.startswith("viewport."):
        return "Render"
    # sim.control_state reports who is driving the solvers and changes
    # nothing. An agent must be able to read it to know whether its own
    # measurement was invalidated, so gating it above Read would make the
    # honest path the privileged one.
    if method == "sim.control_state":
        return "Read"
    # sim_graph.*: queries are Read, everything that builds the graph is
    # SceneWrite. Same ordering lesson as viewport.* — the read heuristics below
    # would otherwise classify only part of the namespace.
    if method in ("sim_graph.nodes", "sim_graph.evaluate", "sim_graph.attributes",
                  "sim_graph.couplings", "sim_graph.surface_attributes",
                  "sim_graph.list"):
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
    # caller chooses, and agent.send_prompt queues work for another agent, so
    # neither is part of the read-only agent.* surface.
    if method in ("agent.chat_send", "agent.send_prompt"):
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
    drift = mirror_drift(namespaces)

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
    if drift:
        print("\nMIRROR DRIFT - required() disagrees with RtIpcSecurity.cpp, so "
              "the generated descriptors state the wrong capability:")
        for method, cpp, mirrored in drift:
            print("  %-32s C++=%-20s required()=%s" % (method, cpp, mirrored))
    # Second pass: the descriptor table that answers agent.describe is generated
    # from these same dispatch sources. If it is stale, a method exists over IPC
    # but has no schema - which reads to an agent as "this capability does not
    # exist" and is exactly the drift this script was written to catch.
    import gen_ipc_descriptors
    descriptors_stale = gen_ipc_descriptors.main(["--check"]) != 0

    if not unreachable and not dead and not drift and not descriptors_stale:
        print("OK - every dispatched method is classified, mirror agrees with "
              "RtIpcSecurity.cpp, no dead prefixes, descriptors current.")
        return 0
    return 1



# ---------------------------------------------------------------------------
# Mirror drift
# ---------------------------------------------------------------------------
# ★★★ `required()` above is a HAND-WRITTEN copy of RtIpcSecurity.cpp's
# requiredCapabilities(). Two sources of truth, and on 2026-08-19 they had
# already drifted: the C++ listed six sim_graph read methods, the mirror listed
# three, so the generated descriptor table told agents `sim_graph.couplings`
# needs SceneWrite while the engine accepted it with Read. The descriptor lied
# about permissions, and nothing noticed — the audit only compared dispatch
# against the mirror, never the mirror against the code it mirrors.
#
# This does not replace the mirror; it makes the drift LOUD. Every method the
# C++ names explicitly is extracted with the capability it returns, and the
# mirror has to agree.

EXPLICIT_RULE_RE = re.compile(
    r'((?:\s*(?:\|\||if)\s*\(?\s*method\s*==\s*"[^"]+"[^;{]*?)+)'
    r'\breturn\s+([A-Za-z]+(?:\s*\|\s*[A-Za-z]+)*)\s*;',
    re.S)


def explicit_cpp_rules(source):
    """(method, capability) pairs the C++ states by name, in source order."""
    body = source[source.index("uint32_t requiredCapabilities"):]
    body = body[:body.index("\n}\n")]
    rules = []
    for match in EXPLICIT_RULE_RE.finditer(body):
        names = re.findall(r'method\s*==\s*"([^"]+)"', match.group(1))
        capability = "|".join(part.strip() for part in match.group(2).split("|"))
        for name in names:
            rules.append((name, capability))
    return rules


def mirror_drift(namespaces):
    source = read(SECURITY)
    disagreements = []
    seen = set()
    for method, cpp_capability in explicit_cpp_rules(source):
        if method in seen:      # first rule wins in C++ too
            continue
        seen.add(method)
        mirrored = required(method, namespaces)
        if mirrored is None:
            disagreements.append((method, cpp_capability, "not classified"))
        elif set(mirrored.split("|")) != set(cpp_capability.split("|")):
            disagreements.append((method, cpp_capability, mirrored))
    return disagreements

if __name__ == "__main__":
    sys.exit(main())
