"""Check every Vulkan simulation kernel's table binding count against its shader.

SimulationComputeVulkan.cpp builds each pipeline layout from the kernel
table's buffer count. A shader that declares more bindings than that gets a
pipeline layout without them (validation VUID-VkComputePipelineCreateInfo-
layout-07988); the dispatcher now refuses such a dispatch at runtime. This
script finds the mismatch before the build does.

Usage: python scripts/audit_sim_kernel_bindings.py   (exit code 1 on mismatch)
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "RayTrophiStudio", "source")
TABLE = os.path.join(ROOT, "src", "Device", "SimulationComputeVulkan.cpp")
SHADERS = os.path.join(ROOT, "shaders")


def shader_text(path: str) -> str:
    text = open(path, encoding="utf-8", errors="replace").read()
    for include in re.findall(r'#include\s+"([^"]+)"', text):
        inc = os.path.join(SHADERS, include)
        if os.path.exists(inc):
            text += open(inc, encoding="utf-8", errors="replace").read()
    return text


def main() -> int:
    table = open(TABLE, encoding="utf-8", errors="replace").read()
    rows = re.findall(r'\{\s*"([a-z0-9_]+)"\s*,\s*"([a-z0-9_]+)\.spv"\s*,\s*(\d+)\s*,\s*(\d+)\s*\}', table)
    failures = []
    for name, spv, count, _ in rows:
        comp = os.path.join(SHADERS, spv + ".comp")
        if not os.path.exists(comp):
            failures.append(f"{name}: no shader source {spv}.comp")
            continue
        bindings = [int(b) for b in re.findall(r"binding\s*=\s*(\d+)", shader_text(comp))]
        needed = max(bindings) + 1 if bindings else 0
        if needed != int(count):
            failures.append(f"{name}: table declares {count}, shader uses {needed}")
    for line in failures:
        print("FAIL", line)
    print(f"{len(rows)} kernels, {len(failures)} mismatches")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
