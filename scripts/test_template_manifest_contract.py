#!/usr/bin/env python3
"""Offline contract checks for Template Hub manifest v1.

This does not build or launch RayTrophi Studio. It validates that the canonical
schema and documentation examples remain valid JSON, English-only runtime
contracts with unique IDs and safe relative paths. If `jsonschema` is installed,
Draft 2020-12 validation is also performed.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "docs" / "template_hub"
SCHEMA_PATH = CONTRACT / "template_manifest.schema.json"
EXAMPLES = CONTRACT / "examples"
RUNTIME_PACKAGES = ROOT / "RayTrophiStudio" / "assets" / "templates"
TURKISH = re.compile(r"[çğıöşüÇĞİÖŞÜ]")


def fail(message: str) -> None:
    raise AssertionError(message)


def safe_relative(value: str) -> bool:
    path = Path(value)
    return bool(value) and not path.is_absolute() and not re.match(r"^[A-Za-z]:", value) and ".." not in path.parts


def main() -> int:
    schema_text = SCHEMA_PATH.read_text(encoding="utf-8")
    if TURKISH.search(schema_text):
        fail("schema contains Turkish runtime text")
    schema = json.loads(schema_text)

    try:
        import jsonschema  # type: ignore
    except ImportError:
        jsonschema = None

    ids: set[str] = set()
    files = sorted(EXAMPLES.glob("*.json"))
    if len(files) != 6:
        fail(f"expected six canonical Start examples, found {len(files)}")

    for path in files:
        text = path.read_text(encoding="utf-8")
        if TURKISH.search(text):
            fail(f"{path.name} contains Turkish runtime text")
        data = json.loads(text)
        if data["schema_version"] != "1.0":
            fail(f"{path.name} does not use manifest v1")
        if data["kind"] != "start":
            fail(f"{path.name} is not a Start template")
        if data["id"] in ids:
            fail(f"duplicate template id: {data['id']}")
        ids.add(data["id"])

        paths = [data["preview"]["image"], data["scene"]["path"]]
        paths.extend(data["assets"]["required"])
        paths.extend(data["assets"]["optional"])
        if "guidance" in data:
            paths.append(data["guidance"]["path"])
        for value in paths:
            if not safe_relative(value):
                fail(f"unsafe relative path in {path.name}: {value}")

        if jsonschema is not None:
            jsonschema.Draft202012Validator(schema).validate(data)
        print(f"CONTRACT OK  {path.name}")

    print(f"OK - {len(files)} manifests, unique IDs, English JSON, safe paths")

    runtime_manifests = sorted(RUNTIME_PACKAGES.glob("*/manifest.json"))
    if not runtime_manifests:
        fail("expected at least one runtime template package")
    for path in runtime_manifests:
        text = path.read_text(encoding="utf-8")
        if TURKISH.search(text):
            fail(f"{path} contains Turkish runtime text")
        data = json.loads(text)
        if jsonschema is not None:
            jsonschema.Draft202012Validator(schema).validate(data)
        package_root = path.parent
        required_paths = [data["preview"]["image"], data["scene"]["path"]]
        required_paths.extend(data["assets"]["required"])
        if "guidance" in data:
            required_paths.append(data["guidance"]["path"])
        for value in required_paths:
            if not safe_relative(value) or not (package_root / value).is_file():
                fail(f"missing or unsafe runtime package file in {path}: {value}")
        print(f"RUNTIME OK   {data['id']}")

    if jsonschema is None:
        print("NOTE - jsonschema is not installed; structural fallback checks were used")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"FAIL - {error}", file=sys.stderr)
        raise SystemExit(1)
