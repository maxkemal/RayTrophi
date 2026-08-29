#!/usr/bin/env python3
"""Static contract audit for structural hardness and surface relief."""

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
UMBRELLA = ROOT / "RayTrophiStudio/source/include/TerrainNodesV2.h"
NODE_HEADER = ROOT / "RayTrophiStudio/source/include/TerrainSurfaceNodes.h"
NODE_SOURCE = ROOT / "RayTrophiStudio/source/src/Physics/TerrainSurfaceNodes.cpp"
GRAPH_SOURCE = ROOT / "RayTrophiStudio/source/src/Physics/TerrainNodesV2.cpp"
PROJECT = ROOT / "RayTrophiStudio/RayTrophiStudio.vcxproj"

# createInput(name, type, semantic, optional, channels, unit)
# createOutput(name, type, semantic, channels, unit)
PIN_RE = re.compile(
    r'create(Input|Output)\(\s*\n?\s*"([^"]+)"[^;]*?ImageSemantic::(\w+)\s*,\s*'
    r'([^,)]+?)\s*(?:,\s*([^,)]+?)\s*)?[,)]',
    re.S)


def read(path):
    if not path.exists():
        raise AssertionError(f"missing {path.relative_to(ROOT)}")
    return path.read_text(encoding="utf-8", errors="replace")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    umbrella = read(UMBRELLA)
    header = read(NODE_HEADER)
    source = read(NODE_SOURCE)
    graph = read(GRAPH_SOURCE)
    project = read(PROJECT)

    require("groundDetailMeters" not in umbrella and
            "groundDetailMeters" not in graph,
            "uniform Noise Generator Ground Detail is still active")

    for node_type in ("StructuralHardness", "SurfaceRelief"):
        require(f"NodeType::{node_type}" in graph,
                f"{node_type} has no graph factory/deserialization wiring")
        require(f'TerrainV2.{node_type}' in header + source + graph,
                f"{node_type} has no public type id")

    # This block used to require Surface Relief to declare a TWO-CHANNEL flow
    # direction input. The requirement outlived its supply: Hydraulic Erosion's
    # slot 5 was the only pin in the terrain set that published a 2-channel
    # vector direction, the compact port contract removed it, and the single
    # Direction OUTPUT left in the whole node set is Watershed Analysis's, at
    # 1 channel. So the audit was guarding a socket nothing could fill while
    # Surface Relief silently fell back to the downhill gradient on every
    # evaluation. The pin is gone; the gradient is computed from the height the
    # node already reads, so no measurement was lost.
    #
    # What replaces it is the invariant that makes that state unreachable: no
    # input may demand a channel count that no output of the same semantic can
    # supply. A pin nothing can feed is indistinguishable from a working one
    # until somebody tries to wire it.
    produced, demanded = {}, []
    for blob in (umbrella, header, source, graph):
        for kind, pin_name, semantic, arg_a, arg_b in PIN_RE.findall(blob):
            raw = (arg_b if kind == "Input" else arg_a).strip()
            if not raw.isdigit():
                continue
            channels = int(raw)
            if kind == "Output":
                produced.setdefault(semantic, set()).add(channels)
            elif channels != 0:
                demanded.append((semantic, channels, pin_name))
    for semantic, channels, pin_name in demanded:
        supply = produced.get(semantic, set())
        require(any(c in (0, channels) for c in supply),
                f"input '{pin_name}' demands {channels}-channel "
                f"ImageSemantic::{semantic}, and no output publishes that "
                f"(outputs offer {sorted(supply) if supply else 'nothing'}) "
                f"- a pin nothing can feed")

    # This used to require the literal link `erosion->outputs[5].id ->
    # relief->inputs[6].id`. That assertion outlived the pin it guarded too:
    # slot 5 became a read PAST THE END of a four-element vector, and this
    # audit stayed green the whole time because it checked that a line of text
    # exists, not that a link is possible. Presence is not truth.
    #
    # What replaces it is the invariant that would have caught the breakage:
    # no setup may index a Hydraulic output or input beyond what the node
    # actually declares.
    hydraulic_ctor = re.search(
        r'HydraulicErosionNode\(\)\s*\{([\s\S]*?)metadata\.displayName',
        umbrella)
    require(hydraulic_ctor is not None,
            "cannot find the HydraulicErosionNode constructor to count its pins")
    declared = {
        "outputs": len(re.findall(r'outputs\.push_back', hydraulic_ctor.group(1))),
        "inputs": len(re.findall(r'inputs\.push_back', hydraulic_ctor.group(1))),
    }
    require(declared["outputs"] > 0 and declared["inputs"] > 0,
            "HydraulicErosionNode declares no pins - the constructor scan broke")
    for alias in ("erosion", "hydraulic"):
        for direction, count in declared.items():
            for match in re.finditer(
                    rf'\b{alias}->{direction}\[\s*(\d+)\s*\]', graph):
                index = int(match.group(1))
                require(index < count,
                        f"setup indexes {alias}->{direction}[{index}] but Hydraulic "
                        f"Erosion declares only {count} {direction} "
                        f"- a pruned pin left a caller behind")

    # Every terrain node must receive an exposure profile, not only the ones
    # with a hand-written branch. 62 of 79 type ids had none, which is why the
    # whole SatMap family published every socket as an equally important
    # authoring choice.
    presentation = read(ROOT / "RayTrophiStudio/source/src/Physics/TerrainNodePortPresentation.cpp")
    require("applyDefaultProfile(node)" in presentation,
            "configureTerrainNodePorts no longer applies a default exposure profile, "
            "so any node without a hand-written branch publishes every pin as Primary")

    require(re.search(
        r'structural->outputs\[0\]\.id\s*,\s*erosion->inputs\[2\]\.id', graph),
        "ready terrain does not route Structural Hardness into Hydraulic Erosion")
    require(re.search(
        r'relief->outputs\[0\]\.id\s*,\s*snow->inputs\[0\]\.id', graph),
        "Surface Relief is not inserted before snow/Height Output")
    require("meshGridWidth" in source,
            "surface band limit ignores the independent mesh resolution")

    for path in ("source\\include\\TerrainSurfaceNodes.h",
                 "source\\src\\Physics\\TerrainSurfaceNodes.cpp"):
        require(project.count(path) == 1,
                f"Visual Studio project must contain exactly one {path} entry")

    print("OK - terrain flow direction, structural hardness and surface relief contracts are wired.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as error:
        print(f"FAIL: {error}")
        sys.exit(1)
