"""Read an existing SPIR-V module; never invoke a compiler or modify artifacts.

Reports static instruction/call counts, not GPU cost or compiler crash attribution.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import struct


def inspect(path):
    data = path.read_bytes()
    if len(data) < 20 or len(data) % 4:
        raise ValueError("Invalid SPIR-V byte length")
    words = struct.unpack(f"<{len(data) // 4}I", data)
    if words[0] != 0x07230203:
        raise ValueError("Invalid SPIR-V magic")
    names = {}
    functions = {}
    controls = {}
    calls = defaultdict(Counter)
    current = None
    cursor = 5
    while cursor < len(words):
        count, opcode = words[cursor] >> 16, words[cursor] & 0xffff
        if count == 0 or cursor + count > len(words):
            raise ValueError(f"Invalid instruction at word {cursor}")
        args = words[cursor + 1:cursor + count]
        if opcode == 5:  # OpName
            raw = struct.pack(f"<{len(args) - 1}I", *args[1:])
            names[args[0]] = raw.split(b"\0", 1)[0].decode("utf-8", errors="replace")
        if opcode == 54:  # OpFunction
            current = args[1]
            functions[current] = Counter()
            controls[current] = args[2]
        if current is not None:
            functions[current][opcode] += 1
            if opcode == 57:  # OpFunctionCall: result type, result id, callee
                calls[current][args[2]] += 1
        if opcode == 56:  # OpFunctionEnd
            current = None
        cursor += count

    def label(identifier):
        return names.get(identifier, f"id:{identifier}").split("(", 1)[0]

    cycles = []
    visited = set()
    active = []

    def visit(node):
        if node in active:
            cycles.append([label(n) for n in active[active.index(node):] + [node]])
            return
        if node in visited:
            return
        active.append(node)
        for child in calls[node]:
            visit(child)
        active.pop()
        visited.add(node)

    for node in list(functions):
        visit(node)
    incoming = Counter()
    for edges in list(calls.values()):
        incoming.update(edges)
    largest = sorted(functions, key=lambda n: sum(functions[n].values()), reverse=True)
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "spirv_version": f"{(words[1] >> 16) & 255}.{(words[1] >> 8) & 255}",
        "generator_word": words[2],
        "id_bound": words[3],
        "function_count": len(functions),
        "dont_inline_function_count": sum(bool(control & 2) for control in controls.values()),
        "function_instructions": sum(sum(counts.values()) for counts in functions.values()),
        "function_call_sites": sum(incoming.values()),
        "call_cycles": cycles,
        "largest_functions": [
            {"name": label(n), "instructions": sum(functions[n].values()),
             "local_variables": functions[n][59], "loads": functions[n][61],
             "stores": functions[n][62], "access_chains": functions[n][65],
             "call_sites": sum(calls[n].values())}
            for n in largest[:12]
        ],
        "most_called": [
            {"name": label(n), "sites": count} for n, count in incoming.most_common(12)
        ],
        "limitations": "Existing artifact only; source/flags provenance, compiler passes, "
                       "dynamic call counts, registers and GPU stack use are not established.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module", type=Path, nargs="?", default=Path(__file__).resolve().parents[2]
                        / "RayTrophiStudio/source/shaders/volume_closesthit.spv")
    args = parser.parse_args()
    print(json.dumps(inspect(args.module), indent=2))
