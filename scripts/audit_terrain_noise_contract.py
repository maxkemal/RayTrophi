#!/usr/bin/env python3
"""Static contract audit for the terrain Noise Generator and the terrain
measurement surface.

Four rules, each written because the corresponding failure is SILENT:

1. Every authored field of NoiseGeneratorNode is drawn, serialized AND
   deserialized. A field with a panel widget and no serializer is a dial that
   forgets; a field with a serializer and no widget is script-only. Both were
   already paid for in this repo.

2. No setup may assign featureSizeMeters on a Noise Generator without also
   settling autoFeatureSize. When Auto is on the metre value is ignored, so the
   assignment reads like authority and is dead code.

3. Every dispatched terrain.* method has a Python binding of the same name.
   The IPC dispatch and the Python module are separate touches, and skipping
   one leaves the method reachable from a pipe and invisible from a script.

4. No single-letter operand pin (A, B) is declared optional. `optional` means
   two different things in this node set - "a modifier you may leave out" and
   "an operand with a scalar fallback" - and the default exposure profile
   treats both as hideable. Math shipped with one visible socket because of it.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HEADER = ROOT / "RayTrophiStudio/source/include/TerrainNodesV2.h"
GRAPH = ROOT / "RayTrophiStudio/source/src/Physics/TerrainNodesV2.cpp"
IPC = ROOT / "RayTrophiStudio/source/src/Api/RtIpc.cpp"
PY = ROOT / "RayTrophiStudio/source/src/Api/RtPython.cpp"


def read(path):
    if not path.exists():
        raise AssertionError(f"missing {path.relative_to(ROOT)}")
    return path.read_text(encoding="utf-8", errors="replace")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def section(blob, opener, closer):
    """Body of the first block that starts at `opener`, by brace depth."""
    start = blob.index(opener)
    depth, i = 0, blob.index(closer, start)
    begin = i
    while i < len(blob):
        if blob[i] == "{":
            depth += 1
        elif blob[i] == "}":
            depth -= 1
            if depth == 0:
                return blob[begin:i]
        i += 1
    raise AssertionError(f"unterminated block after {opener}")


def main():
    header = read(HEADER)
    graph = read(GRAPH)

    # ---- Rule 1: authored fields reach the panel and both serializers -------
    node = section(header, "class NoiseGeneratorNode", "class NoiseGeneratorNode")
    draw = section(node, "void drawContent()", "void drawContent()")
    ser = section(node, "void serializeToJson", "void serializeToJson")
    deser = section(node, "void deserializeFromJson", "void deserializeFromJson")

    declarations = re.findall(
        r"^\s{8}(?:float|int|bool|NoiseType|TerrainNoiseModel)\s+(\w+)\s*=",
        node, re.M)
    authored = [name for name in declarations if not name.startswith("last")]
    require(len(authored) >= 15,
            f"only {len(authored)} authored fields found - the declaration scan broke")
    for name in authored:
        require(name in draw, f"NoiseGeneratorNode.{name} has no panel widget "
                              "- it is script-only, and a value only scripts can "
                              "reach is a value the panel silently disagrees with")
        require(name in ser, f"NoiseGeneratorNode.{name} is never serialized "
                             "- the dial forgets on save, and nodes.set_property "
                             "cannot see it either")
        require(name in deser, f"NoiseGeneratorNode.{name} is never deserialized "
                               "- it saves and then loads back as the default")

    # ---- Rule 2: no dead metre assignment behind Auto -----------------------
    noise_vars = set(re.findall(
        r"auto\*\s+(\w+)\s*=\s*dynamic_cast<NoiseGeneratorNode\*>", graph))
    require(noise_vars, "no NoiseGeneratorNode is built by any setup - scan broke")
    for var in sorted(noise_vars):
        assigns_metres = re.search(rf"\b{var}->featureSizeMeters\s*=", graph)
        settles_auto = re.search(rf"\b{var}->autoFeatureSize\s*=", graph)
        require(not assigns_metres or settles_auto,
                f"setup assigns {var}->featureSizeMeters without settling "
                f"{var}->autoFeatureSize - with Auto on the metre value is "
                "ignored, so the line looks like authority and does nothing")

    # ---- Rule 3: terrain.* reaches Python too -------------------------------
    ipc = read(IPC)
    py = read(PY)
    dispatched = {name for ns, name in
                  re.findall(r'method == "([a-z_]+)\.([a-z_0-9]+)"', ipc)
                  if ns == "terrain"}
    bound = {name for _, name in
             re.findall(r'(\w+)\.def\(\s*"([a-z_0-9]+)"', py)}
    require(len(dispatched) >= 20,
            f"only {len(dispatched)} terrain methods dispatched - scan broke")
    for name in sorted(dispatched):
        require(name in bound,
                f"terrain.{name} is dispatched over IPC but has no Python "
                "binding - reachable from a pipe, invisible from a script")

    # ---- Rule 4: an OPERAND may not be declared optional --------------------
    # `optional` on a terrain pin carries two meanings that look identical in
    # the declaration: "a modifier you may leave out" and "an operand with a
    # scalar fallback". The blanket exposure profile treats both the same and
    # draws the socket only while connected - so Math shipped with ONE visible
    # input and the second operand could not be wired at all. Single-letter
    # operand pins are the case where that reading is never right.
    operand_pins = 0
    for blob in (header, read(ROOT / "RayTrophiStudio/source/include/TerrainSurfaceNodes.h")):
        for name, tail in re.findall(
                r'createInput\(\s*"([AB])"(.*?)\)\s*\)?;', blob, re.S):
            operand_pins += 1
            require(not re.search(r",\s*true\s*(?:[,)]|$)", tail),
                    f"input '{name}' is declared optional, but a single-letter "
                    "operand is what the node combines - declared optional it is "
                    "drawn only while connected, so it cannot be connected")
    require(operand_pins >= 2,
            f"only {operand_pins} operand pins found - the scan broke")

    print(f"OK - {len(authored)} authored Noise Generator fields drawn/saved/loaded, "
          f"{len(dispatched)} terrain methods bound in Python, "
          f"{operand_pins} operand pins required.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as error:
        print(f"FAIL: {error}")
        sys.exit(1)
