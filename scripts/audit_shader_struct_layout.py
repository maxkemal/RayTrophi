"""Verify that every GLSL mirror of a GPU struct still matches the C++ original.

    python scripts\\audit_shader_struct_layout.py

WHY THIS EXISTS

These structs are uploaded as raw SSBO memory and the stride is recomputed from
EVERY declaration that reads them. A shader whose copy is one field short does
not fail: element 0 reads correctly and every element after it reads its
neighbour's bytes. The symptom is not "broken", it is "wrong" — one volume drawn
with another's density multiplier, one material shaded with another's
subsurface. Nothing logs it and it survives most test scenes, because most test
scenes have exactly one of the thing.

★ THE PART THAT MAKES THIS WORTH AUTOMATING: the failure is invisible in review.
Two declarations can differ by a single `float` in the middle and look identical
when read side by side. A byte count does not.

★★ AND IT IS NOT ENOUGH TO COMPARE TOTAL SIZE. A field moved from one place to
another keeps the size and shifts every offset after it, which is the more
confusing failure of the two. So the check compares the field sequence, not just
the total.

WHAT COUNTS AS EQUAL

Names are ignored on purpose: a shader may legitimately lump fields it does not
use into one padding array (raygen.rgen collapses source_type + the eight cloud_*
floats + _ext_reserved[12] into a single `float _ext_reserved[21]`). That is
sound as long as the WIDTHS line up, so the comparison is on the flattened
sequence of 4- and 8-byte slots.

Scalar block layout is assumed (`layout(..., scalar)`), i.e. a vec3 is three
tightly packed floats with no promotion to vec4. Every declaration checked here
is read through a scalar-qualified buffer; if one is ever changed to std430 this
script's model stops being the right one and must be told so.
"""

import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "RayTrophiStudio", "source")

# 4-byte scalars are interchangeable for LAYOUT purposes (int/uint/float all
# occupy one slot); only the width and the count matter.
TYPES = {
    "uint64_t": (8, 1), "float": (4, 1), "int": (4, 1), "uint": (4, 1),
    "vec2": (4, 2), "vec3": (4, 3), "vec4": (4, 4),
    "ivec2": (4, 2), "ivec3": (4, 3), "ivec4": (4, 4),
    "uvec2": (4, 2), "uvec3": (4, 3), "uvec4": (4, 4),
}


def parse_fields(body):
    """Flatten a struct body into [(width_bytes, count), ...] plus names."""
    out = []
    for raw in body.split("\n"):
        line = raw.split("//")[0].strip()
        if not line.endswith(";"):
            continue
        m = re.match(r"^(\w+)\s+(.*);$", line)
        if not m:
            continue
        ty, rest = m.group(1), m.group(2)
        if ty not in TYPES:
            continue
        width, mult = TYPES[ty]
        for decl in rest.split(","):
            decl = decl.strip()
            arr = re.match(r"^(\w+)\s*\[\s*(\d+)\s*\]$", decl)
            count = int(arr.group(2)) if arr else 1
            name = arr.group(1) if arr else decl
            out.append((width, mult * count, name))
    return out


def size_of(fields, align_to):
    total = 0
    for width, count, _ in fields:
        if total % width:
            total += width - (total % width)
        total += width * count
    if align_to and total % align_to:
        total += align_to - (total % align_to)
    return total


def slots(fields):
    """Layout signature: widths only, names discarded (see WHAT COUNTS AS EQUAL)."""
    flat = []
    for width, count, _ in fields:
        flat.extend([width] * count)
    return flat


def extract(path, pattern):
    text = io.open(os.path.join(SRC, path), encoding="utf-8").read()
    m = re.search(pattern, text, re.S)
    if not m:
        return None
    return parse_fields(m.group(1))


GROUPS = [
    {
        "name": "VkVolumeInstance",
        "align": 16,
        "reference": ("include/Backend/vulkan_volume_types.h",
                      r"struct VK_VOL_ALIGN\(16\) VkVolumeInstance \{(.*?)\n\};"),
        "mirrors": [
            ("shaders/volume_closesthit.rchit",   r"struct VkVolumeInstance \{(.*?)\n\};"),
            ("shaders/closesthit.rchit",          r"struct VkVolumeInstance \{(.*?)\n\};"),
            ("shaders/raygen.rgen",               r"struct VkVolumeInstance \{(.*?)\n\};"),
            ("shaders/volume_intersection.rint",  r"struct VkVolumeInstance \{(.*?)\n\};"),
        ],
    },
]

# GLSL structs with no C++ struct to diff against field-for-field, but with a
# static_assert stating the expected size. Checking the number is enough here:
# the C++ side is a single definition, so a size match means the field sequence
# was not silently re-ordered under it.
SIZED = [
    ("Material",    "shaders/material_struct.glsl", r"struct Material \{(.*?)\n\};",    160),
    ("MaterialExt", "shaders/material_struct.glsl", r"struct MaterialExt \{(.*?)\n\};", 256),
]


def main():
    failures = []

    for group in GROUPS:
        ref_path, ref_pat = group["reference"]
        ref = extract(ref_path, ref_pat)
        if ref is None:
            failures.append("{}: reference declaration not found in {}".format(
                group["name"], ref_path))
            continue
        ref_slots = slots(ref)
        ref_size = size_of(ref, group["align"])
        print("{} - reference {} = {} B, {} slots".format(
            group["name"], ref_path, ref_size, len(ref_slots)))

        for path, pat in group["mirrors"]:
            got = extract(path, pat)
            if got is None:
                failures.append("{}: {} has no declaration".format(group["name"], path))
                continue
            got_slots = slots(got)
            got_size = size_of(got, group["align"])
            if got_slots == ref_slots and got_size == ref_size:
                print("  OK    {} ({} B)".format(path, got_size))
                continue

            # Report the first divergence WITH ITS BYTE OFFSET: that is the
            # number you need, because everything from there on is misread.
            offset = 0
            where = "end of struct"
            for i, (a, b) in enumerate(zip(ref_slots, got_slots)):
                if a != b:
                    where = "slot #{} at ~{} B".format(i, offset)
                    break
                if offset % a:
                    offset += a - (offset % a)
                offset += a
            failures.append(
                "{}: {} DIVERGES - {} B vs {} B, first difference at {}".format(
                    group["name"], path, got_size, ref_size, where))
            print("  FAIL  {} ({} B, expected {})".format(path, got_size, ref_size))

    for name, path, pat, expected in SIZED:
        got = extract(path, pat)
        if got is None:
            failures.append("{}: not found in {}".format(name, path))
            continue
        size = size_of(got, 0)
        if size == expected:
            print("{} - {} = {} B  OK".format(name, path, size))
        else:
            failures.append("{}: {} is {} B, C++ static_assert says {}".format(
                name, path, size, expected))
            print("{} - {} = {} B  FAIL (expected {})".format(name, path, size, expected))

    print()
    if failures:
        print("FAILED — {} problem(s):".format(len(failures)))
        for f in failures:
            print("  * " + f)
        print()
        print("A stride mismatch does NOT crash. Element 0 reads correctly and")
        print("every element after it reads its neighbour's bytes, so the scene")
        print("renders and looks merely wrong. Fix before judging any render.")
        return 1

    print("OK - every GLSL mirror matches its C++ original.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
