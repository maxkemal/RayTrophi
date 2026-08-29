#!/usr/bin/env python3
"""Fail when a terrain node class exists but cannot be reached.

A node needs FOUR things to be real, and three of them are silent when absent:

  1. a class                       -- otherwise it does not compile
  2. a factory case                -- otherwise addTerrainNode() returns null
  3. a typeId deserialize entry    -- otherwise a saved project loses the node
  4. a menu entry                  -- otherwise nobody can place one by hand

Alluvial Fan, Sediment Deposition and Delta Formation had (1) with working
compute() bodies and NONE of the other three. They were unreachable from the
UI, from script, and from their own saved files - dead code with a solver
inside it - and nothing said so. Two of them were edited in a work batch
before anyone noticed the edits could never run.

This is the same shape as the IPC capability audit: the four touches must
agree, and only a script checks that they do.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HEADERS = [
    ROOT / "RayTrophiStudio/source/include/TerrainNodesV2.h",
    ROOT / "RayTrophiStudio/source/include/TerrainSurfaceNodes.h",
]
IMPLS = [
    ROOT / "RayTrophiStudio/source/src/Physics/TerrainNodesV2.cpp",
    ROOT / "RayTrophiStudio/source/src/Physics/TerrainSurfaceNodes.cpp",
]
MENU = ROOT / "RayTrophiStudio/source/src/UI/scene_ui_nodeeditor.hpp"

# Types the graph creates for itself and never offers as a placeable node.
# Keep this list SHORT and justified: every entry is a node the audit can no
# longer protect.
MENU_EXEMPT = {
    # Output sinks and internal plumbing are placed by setups and by the
    # node-editor's own output section rather than the add menu.
    "HeightmapInput",
    # Disabled legacy passthrough. It must stay creatable so old projects
    # load, and must stay OUT of the menu so no new one is ever placed.
    "ErosionWizard",
    # Composite convenience node placed by the river setup.
    "RiverLakeEasy",
}

# Types deliberately loadable only for backward compatibility. Registering
# them would expose disabled legacy behavior through generic nodes.add.
REGISTRY_EXEMPT = {"ErosionWizard"}


def read(path):
    if not path.exists():
        sys.exit(f"FAIL: missing source file {path}")
    return path.read_text(encoding="utf-8", errors="replace")


def enum_values(header_text):
    match = re.search(r"enum class NodeType\s*\{(.*?)\n\s*\};", header_text, re.S)
    if not match:
        sys.exit("FAIL: could not find 'enum class NodeType' in TerrainNodesV2.h")
    body = match.group(1)
    body = re.sub(r"//[^\n]*", "", body)
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    values = []
    for raw in body.split(","):
        name = raw.strip()
        if not name:
            continue
        name = name.split("=")[0].strip()
        if re.fullmatch(r"[A-Za-z_]\w*", name):
            values.append(name)
    return values


def main():
    header = "\n".join(read(path) for path in HEADERS)
    impl = "\n".join(read(path) for path in IMPLS)
    menu = read(MENU)

    values = enum_values(header)
    if not values:
        sys.exit("FAIL: NodeType enum parsed as empty - the parser is broken, not the code")

    classes = dict(re.findall(r"class\s+(\w+)\s*:\s*public\s+TerrainNodeBase", header) and
                   [(m, m) for m in re.findall(r"class\s+(\w+)\s*:\s*public\s+TerrainNodeBase", header)])
    # NodeType assigned inside each class body, so a class is tied to its type.
    type_of_class = dict(re.findall(
        r"class\s+(\w+)\s*:\s*public\s+TerrainNodeBase.*?terrainNodeType\s*=\s*NodeType::(\w+)",
        header, re.S | re.M))
    # The regex above is greedy across classes; redo it per class block instead.
    type_of_class = {}
    for cls_match in re.finditer(r"class\s+(\w+)\s*:\s*public\s+TerrainNodeBase", header):
        cls = cls_match.group(1)
        window = header[cls_match.end(): cls_match.end() + 6000]
        assigned = re.search(r"terrainNodeType\s*=\s*NodeType::(\w+)", window)
        if assigned:
            type_of_class[cls] = assigned.group(1)
    # Focused modules keep constructors out of the already oversized umbrella
    # header. Read their explicit constructor assignment too.
    for cls in classes:
        assigned = re.search(
            rf"{re.escape(cls)}::{re.escape(cls)}\s*\(\s*\)\s*\{{.*?"
            r"terrainNodeType\s*=\s*NodeType::(\w+)", impl, re.S)
        if assigned:
            type_of_class[cls] = assigned.group(1)

    factory = set(re.findall(r"case\s+NodeType::(\w+)\s*:", impl))
    type_ids = set(re.findall(r'typeId\s*==\s*"TerrainV2\.(\w+)"', impl))
    menu_types = set(re.findall(r"\{\s*NodeType::(\w+)\s*,", menu))
    declared_type_ids = set(re.findall(
        r'getTypeId\(\).*?return\s+"TerrainV2\.(\w+)"', header, re.S))
    registered_type_ids = set(re.findall(
        r'AutoRegisterNode<\w+>.*?\("TerrainV2\.(\w+)"\)', impl, re.S))

    failures = []
    warnings = []

    for cls, node_type in sorted(type_of_class.items()):
        if node_type not in values:
            failures.append(f"{cls}: NodeType::{node_type} is not in the enum")
            continue
        if node_type not in factory:
            failures.append(
                f"{cls}: no 'case NodeType::{node_type}' in the factory - "
                f"addTerrainNode() returns null, the node cannot be created by ANY path")

    for type_id in sorted(declared_type_ids):
        if type_id not in type_ids:
            failures.append(
                f"TerrainV2.{type_id}: getTypeId() declares it but no deserialize branch "
                f"reads it - a saved project silently loses this node on load")
        if type_id not in registered_type_ids and type_id not in REGISTRY_EXEMPT:
            failures.append(
                f"TerrainV2.{type_id}: no NodeRegistry registration - generic Python/IPC "
                f"nodes.add cannot create it")

    for node_type in sorted(set(type_of_class.values())):
        if node_type in MENU_EXEMPT:
            continue
        if node_type not in menu_types:
            warnings.append(
                f"NodeType::{node_type} is not in the add menu - reachable from script "
                f"and setups only, so a human cannot place one")

    print(f"{len(values)} NodeType values, {len(type_of_class)} node classes, "
          f"{len(factory)} factory cases, {len(type_ids)} typeId branches, "
          f"{len(menu_types)} menu entries")

    for line in warnings:
        print(f"  warn  {line}")

    if failures:
        print()
        print(f"UNREACHABLE ({len(failures)}):")
        for line in failures:
            print(f"  {line}")
        return 1

    print("OK - every terrain node class can be created, saved and reloaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
