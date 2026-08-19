#!/usr/bin/env python3
"""Generate RtIpcMethodDescriptors.cpp from the IPC dispatch code itself.

Why generated and not hand-written
----------------------------------
The first cut of the discovery layer registered 300 methods by hand, and 299 of
them carried zero parameters and a placeholder summary while `agent.discover`
still reported `coverage ~ 1.0`.  A hand-maintained catalogue of 300 entries
does not stay true; it only stays *present*.

So the machine-checkable half of a descriptor -- which parameters a method
reads, whether each one is required, its type and default, and which security
capability it needs -- is EXTRACTED from the dispatch source.  That half cannot
drift: regenerate and it follows the code.

The half a machine cannot know -- what the method is FOR, what to call before
and after it, units and gotchas -- lives in `ipc_descriptor_overlay.json`,
written by hand, keyed by method name.  Anything absent from the overlay is
emitted with `documented = false` so `agent.discover` can report real
documentation coverage instead of pretending.

    python scripts/gen_ipc_descriptors.py            # rewrite the .cpp
    python scripts/gen_ipc_descriptors.py --check    # exit 1 if out of date

`--check` is what `audit_ipc_capabilities.py` runs, so a new IPC method that
never got a descriptor fails the audit instead of silently answering
`agent.describe` with an empty schema.
"""

import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_ipc_capabilities import (  # noqa: E402
    API, IPC_FILES, SECURITY, namespace_table, read, required)

OVERLAY = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "ipc_descriptor_overlay.json")
OUTPUT = os.path.join(API, "RtIpcMethodDescriptors.cpp")
# Descriptors are generated from the other dispatch modules; the output file
# itself must never be parsed as one.
SOURCES = [p for p in IPC_FILES
           if os.path.basename(p) != "RtIpcMethodDescriptors.cpp"]


# ---------------------------------------------------------------------------
# Dispatch parsing
# ---------------------------------------------------------------------------

# A handler starts at `if (method == "...")`. Matching a bare `method == "..."`
# is not enough: bodies compare the method again inline
# (`std::string t = (method == "gas.create_domain") ? ...`), and treating that
# as the start of the next handler truncates the block to a few characters --
# which is exactly how the first pass produced a parameterless descriptor for
# `fluid.create_domain`.
HEAD_RE = re.compile(r'\bif\s*\(\s*method\s*==\s*"([^"]+)"')
# `if (method == "a" || method == "b")` shares one body: both names get the
# same parameters.
ALSO_RE = re.compile(r'[\s)]*\|\|\s*method\s*==\s*"([^"]+)"')


def brace_block(source, start):
    """Text of the `{...}` that begins at or after `start`, or None."""
    opening = source.find("{", start)
    if opening < 0:
        return None
    depth, index = 0, opening
    while index < len(source):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[opening:index + 1]
        index += 1
    return None


def dispatch_blocks(source):
    """Map every dispatched method name to the source text of its handler.

    The handler body is taken by brace matching, not by "everything up to the
    next handler". The last handler in a file would otherwise swallow the rest
    of the concatenated sources -- which is how `agent.chat_poll` first came
    out owning `site_count`, `break_velocity` and two dozen other fracture
    parameters that belong to a different file entirely.
    """
    blocks = {}
    heads = list(HEAD_RE.finditer(source))
    for index, head in enumerate(heads):
        limit = heads[index + 1].start() if index + 1 < len(heads) else len(source)
        names = [head.group(1)]
        cursor = head.end()
        while True:
            more = ALSO_RE.match(source, cursor)
            if not more:
                break
            names.append(more.group(1))
            cursor = more.end()
        condition_end = source.find(")", cursor)
        body = None
        if 0 <= condition_end < limit:
            candidate = brace_block(source, condition_end)
            if candidate is not None and len(candidate) <= limit - condition_end:
                body = candidate
        if body is None:  # braceless `if (method == "x") return ...;`
            body = source[cursor:limit]
        for name in names:
            blocks.setdefault(name, "")
            blocks[name] += body
    return blocks


REQUIRE_TYPES = {
    "requireString": "string",
    "requireInt": "int",
    "requireFloat": "float",
    "requireBool": "bool",
    "requireVec3": "vec3",
    "requireVec2": "vec2",
    "requireMatrix": "matrix",
    "requireColor": "vec3",
}
OPTIONAL_TYPES = {
    "optionalString": "string",
    "optionalInt": "int",
    "optionalFloat": "float",
    "optionalBool": "bool",
    "optionalVec3": "vec3",
}
C_TYPES = {
    "std::string": "string", "string": "string", "const char*": "string",
    "float": "float", "double": "float",
    "int": "int", "uint32_t": "int", "int32_t": "int", "size_t": "int",
    "unsigned": "int", "bool": "bool",
}


def literal_type(text):
    """Infer a parameter type from the default value written in the code."""
    text = text.strip().rstrip(")").strip()
    if not text:
        return None, None
    if text.startswith('"') and text.endswith('"'):
        return "string", text[1:-1]
    if text.startswith("std::string("):
        return "string", text[len("std::string("):].rstrip(")").strip().strip('"')
    if text in ("true", "false"):
        return "bool", text
    if text.startswith("Vec3"):
        return "vec3", None
    if text.startswith("json::object"):
        return "object", None
    if text.startswith("json::array"):
        return "array", None
    if re.fullmatch(r"-?\d+\.\d*f?|-?\.\d+f?|-?\d+f", text):
        return "float", text.rstrip("f")
    if re.fullmatch(r"-?\d+", text):
        return "int", text
    return None, None


def split_args(text):
    """Split a call argument list on commas that are not nested or quoted."""
    parts, depth, quoted, escaped, current = [], 0, False, False, ""
    for char in text:
        if quoted:
            current += char
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
            current += char
        elif char in "([{":
            depth += 1
            current += char
        elif char in ")]}":
            if depth == 0:
                break
            depth -= 1
            current += char
        elif char == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += char
    parts.append(current)
    return [p.strip() for p in parts]


def macro_table(source):
    """Field macros like RT_GAS_JSON(fire_enabled, bool) hide parameter names.

    A handler that sets thirty fields writes them through a local `#define`, so
    a regex looking for `params.at("...")` finds nothing and the method comes
    out with one visible parameter (`domain`) and thirty invisible ones. This
    walks the `#define` body to learn which macro argument carries the key and
    which carries the type, then reads the call sites.
    """
    table = {}
    # The body runs to the end of the line, or through every line that ends in
    # a backslash. Stopping at the first newline would drop RT_FS_FIELD (whose
    # body is on the continuation line) and take its twenty-five emitter
    # parameters with it.
    for match in re.finditer(
            r'#define\s+(\w+)\(([^)]*)\)((?:[^\n]*\\\n)*[^\n]*)', source):
        name, args, body = match.group(1), match.group(2), match.group(3)
        args = [a.strip() for a in args.split(",") if a.strip()]
        if not args or "contains" not in body:
            continue
        key_index = None
        for index, arg in enumerate(args):
            if re.search(r'contains\(\s*#?' + re.escape(arg) + r'\b', body):
                key_index = index
                break
        if key_index is None:
            continue
        type_index = None
        for index, arg in enumerate(args):
            if index != key_index and re.search(
                    r'get<\s*' + re.escape(arg) + r'\s*>', body):
                type_index = index
                break
        table[name] = (key_index, type_index)
    return table


def helper_table(source):
    """Handlers that hand the whole request to a helper (applyFlowSourceJson).

    Without following the call, `flow_source.create` reports zero parameters
    while actually accepting twenty-five.
    """
    helpers = {}
    for match in re.finditer(r'\b(\w+)\s*\(([^;{}]*)\)\s*\{', source):
        name, args = match.group(1), match.group(2)
        if name in ("if", "for", "while", "switch", "catch", "return"):
            continue
        root = re.search(r'(?:const\s+)?(?:nlohmann::)?json\s*&\s*(\w+)', args)
        if not root:
            continue
        start = match.end() - 1
        depth, index = 0, start
        while index < len(source):
            if source[index] == "{":
                depth += 1
            elif source[index] == "}":
                depth -= 1
                if depth == 0:
                    break
            index += 1
        helpers[name] = (source[start:index], root.group(1))
    return helpers


def lambda_setters(body, root):
    """Local `auto flt = [&](const char* key, float& target) {...}` setters.

    `applyParticleEmitterPatch` assigns forty fields through three of these, so
    a scan for `patch.contains("...")` sees only the handful written inline and
    the emitter comes out with three parameters instead of forty.
    """
    setters = {}
    for match in re.finditer(
            r'auto\s+(\w+)\s*=\s*\[[^\]]*\]\s*\(([^)]*)\)\s*(?:->[^{]*)?\{',
            body):
        name, args = match.group(1), split_args(match.group(2))
        start = match.end() - 1
        depth, index = 0, start
        while index < len(body):
            if body[index] == "{":
                depth += 1
            elif body[index] == "}":
                depth -= 1
                if depth == 0:
                    break
            index += 1
        inner = body[start:index]
        key_index, key_name = None, None
        for position, arg in enumerate(args):
            identifier = arg.split()[-1].lstrip("&*") if arg.split() else ""
            if identifier and (
                    re.search(re.escape(root) + r'[.\[]\w*\(?\s*'
                              + re.escape(identifier) + r'\b', inner)
                    or re.search(r'\(\s*' + re.escape(root) + r'\s*,\s*'
                                 + re.escape(identifier) + r'\b', inner)):
                key_index, key_name = position, identifier
                break
        if key_index is None:
            continue
        ptype = "any"
        for position, arg in enumerate(args):
            if position == key_index:
                continue
            declared = arg.replace("&", " ").replace("*", " ").split()
            if len(declared) >= 2:
                ptype = C_TYPES.get(" ".join(declared[:-1]).strip(),
                                    "vec3" if declared[0] == "Vec3" else "any")
            break
        setters[name] = (key_index, ptype)
    return setters


def extract_params(body, root="params", macros=None, helpers=None, seen=None):
    """Every request key the handler reads, with type/required/default."""
    macros = macros or {}
    helpers = helpers or {}
    seen = seen or set()
    found = {}
    esc = re.escape(root)

    def note(name, ptype, requiredness, default=None):
        entry = found.setdefault(name, {"type": None, "required": False,
                                        "default": None})
        if ptype and (entry["type"] is None or entry["type"] == "any"):
            entry["type"] = ptype
        if requiredness:
            entry["required"] = True
        if default is not None and entry["default"] is None:
            entry["default"] = default

    # A `requireX` inside a `<root>.contains("x")` guard is NOT required -- the
    # handler offers it as one of several forms (`scene.set_transform` takes
    # `matrix` OR translation/rotation/scale). Marking those required would
    # send an agent hunting for parameters it must not send together.
    guarded = set(re.findall(esc + r'\.contains\(\s*"([^"]+)"', body))
    for helper, ptype in REQUIRE_TYPES.items():
        for match in re.finditer(helper + r'\(\s*' + esc + r'\s*,\s*"([^"]+)"',
                                 body):
            key = match.group(1)
            note(key, ptype, key not in guarded)
    for helper, ptype in OPTIONAL_TYPES.items():
        for match in re.finditer(
                helper + r'\(\s*' + esc + r'\s*,\s*"([^"]+)"([^;]*)', body):
            args = split_args(match.group(2).lstrip(","))
            default = literal_type(args[0])[1] if args and args[0] else None
            note(match.group(1), ptype, False, default)
    for match in re.finditer(esc + r'\.value\(\s*"([^"]+)"\s*,([^;]*)', body):
        args = split_args(match.group(2))
        ptype, default = literal_type(args[0]) if args else (None, None)
        note(match.group(1), ptype or "any", False, default)
    for match in re.finditer(esc + r'\.contains\(\s*"([^"]+)"', body):
        note(match.group(1), "any", False)
    for match in re.finditer(esc + r'\[\s*"([^"]+)"\s*\]', body):
        note(match.group(1), "any", False)
    for match in re.finditer(esc + r'\.at\(\s*"([^"]+)"', body):
        note(match.group(1), "any", match.group(1) not in guarded)

    # The `#define` line itself looks exactly like a call site, and would
    # register the macro's own argument name (`key`) as a request parameter.
    call_sites = re.sub(r'#define\s+\w+\(([^)]*)\)((?:[^\n]*\\\n)*[^\n]*)',
                        "", body)
    for macro, (key_index, type_index) in macros.items():
        for match in re.finditer(r'\b' + macro + r'\s*\(([^;]*)\)', call_sites):
            args = split_args(match.group(1))
            if key_index >= len(args):
                continue
            key = args[key_index].strip().strip('"')
            if not re.fullmatch(r'[A-Za-z_]\w*', key):
                continue
            ptype = "any"
            if type_index is not None and type_index < len(args):
                ptype = C_TYPES.get(args[type_index].strip(), "any")
            note(key, ptype, False)

    for setter, (key_index, ptype) in lambda_setters(body, root).items():
        for match in re.finditer(r'\b' + setter + r'\s*\(([^;]*)\)',
                                 call_sites):
            args = split_args(match.group(1))
            if key_index >= len(args):
                continue
            key = args[key_index].strip()
            if not (key.startswith('"') and key.endswith('"')):
                continue
            note(key.strip('"'), ptype, False)

    for name, (helper_body, helper_root) in helpers.items():
        if name in seen:
            continue
        if re.search(r'\b' + name + r'\s*\([^;]*\b' + esc + r'\b', body):
            nested = extract_params(helper_body, helper_root, macros, helpers,
                                    seen | {name})
            for key, info in nested.items():
                note(key, info["type"], info["required"], info["default"])
    return found


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------

def c_string(value):
    if value is None:
        return "nullptr"
    text = (str(value).replace("\\", "\\\\").replace('"', '\\"')
            .replace("\n", "\\n").replace("\r", ""))
    return '"%s"' % text


def symbol(method):
    return re.sub(r"[^a-z0-9]+", "_", method.lower())


ACCESS_BY_CAPABILITY = {
    "Read": "read",
    "Render": "render",
    "Admin": "admin",
    "Admin|FilesWrite": "admin",
    "Scripts|FilesRead": "write",
}


def build(overlay, namespaces):
    source = "\n".join(read(path) for path in SOURCES)
    blocks = dispatch_blocks(source)
    macros = macro_table(source)
    helpers = helper_table(source)
    records = []
    for method in sorted(blocks):
        entry = overlay.get(method, {})
        capability = required(method, namespaces) or "Read"
        access = entry.get("access") or ACCESS_BY_CAPABILITY.get(capability,
                                                                 "write")
        # `json patch = params;` then `applyXPatch(patch, info)` is the usual
        # shape for the "overlay whatever the caller mentioned" handlers, so
        # the request keys are reachable only through the alias.
        params = {}
        roots = ["params"] + re.findall(r'json\s+(\w+)\s*=\s*params\s*;',
                                        blocks[method])
        for root in roots:
            for key, info in extract_params(blocks[method], root, macros,
                                            helpers).items():
                current = params.setdefault(key, dict(info))
                current["required"] = current["required"] or info["required"]
                if current["type"] in (None, "any"):
                    current["type"] = info["type"]
                if current["default"] is None:
                    current["default"] = info["default"]
        described = entry.get("params", {})
        order = list(described)
        rows = []
        for name in sorted(params, key=lambda key: (
                not params[key]["required"],
                order.index(key) if key in order else 999, key)):
            info = params[name]
            hand = described.get(name, {})
            rows.append({
                "name": name,
                "type": hand.get("type") or info["type"] or "any",
                "required": hand.get("required", info["required"]),
                "description": hand.get("description", ""),
                "default": hand.get("default", info["default"]),
                "enum": hand.get("enum"),
            })
        # Tags are what agent.search_capabilities scores against. The domain and
        # the words of the method name are free and always true, so they are
        # added here; the overlay only has to carry the synonyms a caller would
        # actually type ("burn", "pour", "shatter").
        domain = method.split(".")[0] if "." in method else method
        tags = [domain] + [t for t in re.split(r"[^a-z0-9]+", method.lower())
                           if t and t != domain]
        for extra in (entry.get("tags") or "").split("|"):
            if extra and extra not in tags:
                tags.append(extra)
        records.append({
            "method": method,
            "domain": domain,
            "summary": entry.get("summary"),
            "notes": entry.get("notes"),
            "access": access,
            "capability": capability,
            "undoable": entry.get("undoable", False),
            "returns": entry.get("returns", "any"),
            "tags": "|".join(tags),
            "related": entry.get("related"),
            "params": rows,
            "documented": bool(entry.get("summary")),
        })
    return records


HEADER = """/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          Api/RtIpcMethodDescriptors.cpp
 * Date:          August 2026
 * License:       MIT
 * =========================================================================
 * GENERATED FILE - do not edit by hand.
 *
 *   python scripts/gen_ipc_descriptors.py
 *
 * Parameters, types, requiredness, defaults and the security capability are
 * read out of the dispatch sources, so that half cannot drift from the code.
 * Summaries, notes, units, tags and related-method links come from
 * scripts/ipc_descriptor_overlay.json - edit THAT file, then regenerate.
 *
 * A method with no overlay entry is emitted with documented = false. That is
 * deliberate: agent.discover reports documented_coverage from this flag, so an
 * undocumented method shows up as a measured gap instead of as an empty schema
 * that looks complete.
 * =========================================================================
 */

#include "RtIpcMethodRegistry.h"

namespace {

"""


def emit(records):
    out = [HEADER]
    for record in records:
        name = symbol(record["method"])
        if record["params"]:
            out.append("static const MethodParam params_%s[] = {\n" % name)
            for row in record["params"]:
                out.append("    {%s, %s, %s, %s, %s, %s},\n" % (
                    c_string(row["name"]), c_string(row["type"]),
                    "true" if row["required"] else "false",
                    c_string(row["description"]),
                    c_string(row["default"]), c_string(row["enum"])))
            out.append("};\n")
        out.append("static const MethodDescriptor desc_%s = {\n" % name)
        out.append("    %s, %s,\n" % (c_string(record["method"]),
                                      c_string(record["domain"])))
        out.append("    %s,\n" % c_string(record["summary"]))
        out.append("    %s,\n" % c_string(record["notes"]))
        out.append("    %s, %s, %s, %s,\n" % (
            c_string(record["access"]), c_string(record["capability"]),
            "true" if record["undoable"] else "false",
            c_string(record["returns"])))
        out.append("    %s,\n" % c_string(record["tags"]))
        out.append("    %s,\n" % c_string(record["related"]))
        if record["params"]:
            out.append("    params_%s, %d,\n" % (name, len(record["params"])))
        else:
            out.append("    nullptr, 0,\n")
        out.append("    %s\n};\n" % ("true" if record["documented"] else "false"))
        out.append("static const MethodRegistration reg_%s(desc_%s);\n\n"
                   % (name, name))
    out.append("} // namespace\n")
    return "".join(out)


def main(argv):
    overlay = {}
    if os.path.exists(OVERLAY):
        overlay = json.loads(read(OVERLAY))
        overlay.pop("$comment", None)
    namespaces = namespace_table(read(SECURITY))
    records = build(overlay, namespaces)
    text = emit(records)

    stale = [m for m in overlay if m not in {r["method"] for r in records}]
    documented = sum(1 for r in records if r["documented"])
    with_params = sum(1 for r in records if r["params"])

    if "--check" in argv:
        current = read(OUTPUT) if os.path.exists(OUTPUT) else ""
        if current.replace("\r\n", "\n") != text:
            print("STALE: RtIpcMethodDescriptors.cpp does not match the "
                  "dispatch sources. Run: python scripts/gen_ipc_descriptors.py")
            return 1
        if stale:
            print("STALE overlay entries (method no longer dispatched): "
                  + ", ".join(sorted(stale)))
            return 1
        print("descriptors up to date - %d methods, %d documented, "
              "%d carry parameters" % (len(records), documented, with_params))
        return 0

    io.open(OUTPUT, "w", encoding="utf-8", newline="\n").write(text)
    print("wrote %s" % OUTPUT)
    print("  %d methods, %d documented (%.0f%%), %d carry parameters"
          % (len(records), documented,
             100.0 * documented / max(1, len(records)), with_params))
    if stale:
        print("  overlay entries for methods that are no longer dispatched: "
              + ", ".join(sorted(stale)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
