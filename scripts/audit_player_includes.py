#!/usr/bin/env python3
"""Include-graph audit for the RayTrophi Player split.

The Player executable must not depend on OptiX/CUDA, Embree, OIDN or Python.
ImGui, SDL, audio and Jolt are supported runtime infrastructure; Studio editor
modules are separated at the source/link boundary, not by banning ImGui headers. This script measures how far those dependencies reach
through the *transitive* include graph, before any Player target exists.

  python scripts/audit_player_includes.py            # report
  python scripts/audit_player_includes.py --check    # ratchet: FAIL if worse than baseline
  python scripts/audit_player_includes.py --why src/Physics/Foo.cpp
  python scripts/audit_player_includes.py --write-baseline

Method and limits (read before trusting a number):
  * Includes are resolved by path against the project include roots, then by
    the including file's directory. Unresolved names are treated as external
    and matched against the forbidden patterns.
  * The view is the PLAYER build's host compiler: `__CUDACC__` undefined and
    `RT_NO_CUDA` defined (see include/GpuInteropTypes.h). Conditions made only
    of those are evaluated. Every other #if keeps BOTH branches, so the result
    is an upper bound: it can report a dependency a real build would skip,
    never hide one a real build would take.
  * A clean result means "no forbidden header is reachable". It does not prove
    the file links without the library (a forward-declared type can still need
    a symbol). The link boundary is the final proof; this is the early one.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import sys
from collections import defaultdict

def _find_repo() -> str:
    # The script is mirrored into x64/Release/scripts; walk up to the checkout.
    d = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.isdir(os.path.join(d, "RayTrophiStudio", "source")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise SystemExit("audit_player_includes: repository root not found")
        d = parent


REPO = _find_repo()
PROJECT = os.path.join(REPO, "RayTrophiStudio")
SOURCE = os.path.join(PROJECT, "source")
# Always the checked-in manifest, so both copies of the script share one baseline.
MANIFEST = os.path.join(REPO, "scripts", "player_include_manifest.json")

INCLUDE_ROOTS = [
    os.path.join(SOURCE, "include"),
    os.path.join(SOURCE, "header"),
    os.path.join(SOURCE, "src"),
] + [os.path.join(SOURCE, "src", d) for d in (
    "UI", "Render", "Scene", "Physics", "Math", "Core", "Utils",
    "Animation", "Hair", "Device")]

SCAN_DIRS = [os.path.join(SOURCE, d) for d in ("include", "header", "src")]
SOURCE_EXT = (".cpp", ".c", ".cc")
HEADER_EXT = (".h", ".hpp", ".inl", ".cuh", ".hh")

# category -> regexes matched against the include spelling of an EXTERNAL header
FORBIDDEN = {
    "cuda_optix": [r"^optix", r"^cuda", r"^curand", r"^vector_types\.h$",
                   r"^nvrtc", r"^cufft"],
    "embree": [r"^embree\d*/", r"^rtcore"],
    "oidn": [r"^OpenImageDenoise/"],
    "python": [r"^Python\.h$", r"^pybind11/", r"^structmember\.h$"],
}
FORBIDDEN_RE = {k: [re.compile(p, re.I) for p in v] for k, v in FORBIDDEN.items()}

INCLUDE_RE = re.compile(r'^\s*#\s*include\s*([<"])([^>"]+)[>"]')
COND_RE = re.compile(r'^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)')


def rel(path: str) -> str:
    return os.path.relpath(path, SOURCE).replace("\\", "/")


def classify_external(name: str) -> str | None:
    n = name.replace("\\", "/")
    for cat, regs in FORBIDDEN_RE.items():
        if any(r.search(n) for r in regs):
            return cat
    return None


# Macros with a known value in the view being audited: the HOST compiler of the
# PLAYER build. __CUDACC__ is never defined there; RT_NO_CUDA always is.
KNOWN_MACROS = {"__CUDACC__": False, "RT_NO_CUDA": True}
_DEFINED_RE = re.compile(r'defined\s*\(\s*(\w+)\s*\)|defined\s+(\w+)')


def cudacc_state(directive: str, rest: str):
    """Return True/False when the condition is decided by KNOWN_MACROS alone,
    else None (unknown: both branches are kept)."""
    rest = rest.split("//")[0].strip()
    if directive in ("ifdef", "ifndef"):
        if rest not in KNOWN_MACROS:
            return None
        return KNOWN_MACROS[rest] == (directive == "ifdef")
    if directive in ("if", "elif"):
        unknown = False

        def repl(m):
            nonlocal unknown
            name = m.group(1) or m.group(2)
            if name not in KNOWN_MACROS:
                unknown = True
                return "False"
            return str(KNOWN_MACROS[name])

        expr = _DEFINED_RE.sub(repl, rest)
        if unknown:
            return None
        expr = expr.replace("&&", " and ").replace("||", " or ")
        expr = re.sub(r'!(?!=)', ' not ', expr)
        if not re.fullmatch(r'(?:\s|True|False|and|or|not|\(|\))+', expr):
            return None
        try:
            return bool(eval(expr, {"__builtins__": {}}, {}))
        except Exception:
            return None
    return None


def parse_includes(path: str):
    """Yield (spelling, is_angle) for includes live in the host compiler's view."""
    out = []
    # stack entries: None (unknown: keep both branches) or bool (taken?)
    stack: list = []
    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
            lines = fh.readlines()
    except OSError:
        return out
    for line in lines:
        m = COND_RE.match(line)
        if m:
            d, rest = m.group(1), m.group(2)
            if d in ("if", "ifdef", "ifndef"):
                stack.append(cudacc_state(d, rest))
            elif d == "elif":
                if stack:
                    prev = stack[-1]
                    stack[-1] = None if prev is None else (not prev and None)
                    if prev is True:
                        stack[-1] = False
            elif d == "else":
                if stack and stack[-1] is not None:
                    stack[-1] = not stack[-1]
            elif d == "endif":
                if stack:
                    stack.pop()
            continue
        if any(s is False for s in stack):
            continue
        m = INCLUDE_RE.match(line)
        if m:
            out.append((m.group(2).strip(), m.group(1) == "<"))
    return out


class Graph:
    def __init__(self):
        self.files: list[str] = []
        self.by_lower: dict[str, str] = {}
        self.edges: dict[str, list[str]] = {}
        # file -> {category: [external spelling,...]} for DIRECT includes
        self.direct: dict[str, dict[str, list[str]]] = {}
        self._reach: dict[str, dict[str, str]] = {}

    def build(self):
        for root in SCAN_DIRS:
            for dp, dn, fn in os.walk(root):
                dn[:] = [d for d in dn if d != "_Unused"]
                for f in fn:
                    if f.lower().endswith(SOURCE_EXT + HEADER_EXT):
                        p = os.path.normpath(os.path.join(dp, f))
                        self.files.append(p)
                        self.by_lower[p.lower()] = p
        for p in self.files:
            deps, direct = [], defaultdict(list)
            for spelling, _angle in parse_includes(p):
                target = self.resolve(p, spelling)
                if target:
                    deps.append(target)
                else:
                    cat = classify_external(spelling)
                    if cat:
                        direct[cat].append(spelling)
            self.edges[p] = deps
            self.direct[p] = dict(direct)

    def resolve(self, includer: str, spelling: str) -> str | None:
        cands = [os.path.join(os.path.dirname(includer), spelling)]
        cands += [os.path.join(r, spelling) for r in INCLUDE_ROOTS]
        for c in cands:
            hit = self.by_lower.get(os.path.normpath(c).lower())
            if hit:
                return hit
        return None

    def reach(self, start: str) -> dict[str, str]:
        """category -> the first-party file whose DIRECT include introduces it."""
        found: dict[str, str] = {}
        seen = {start}
        stack = [start]
        while stack:
            cur = stack.pop()
            for cat in self.direct.get(cur, {}):
                found.setdefault(cat, cur)
            for nxt in self.edges.get(cur, []):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return found

    def chain(self, start: str, cat: str) -> list[str] | None:
        """Shortest include chain from start to a file that directly pulls cat."""
        prev = {start: None}
        queue = [start]
        while queue:
            cur = queue.pop(0)
            if cat in self.direct.get(cur, {}):
                path = []
                while cur is not None:
                    path.append(cur)
                    cur = prev[cur]
                return path[::-1]
            for nxt in self.edges.get(cur, []):
                if nxt not in prev:
                    prev[nxt] = cur
                    queue.append(nxt)
        return None


def load_manifest() -> dict:
    with open(MANIFEST, "r", encoding="utf-8") as fh:
        return json.load(fh)


def in_player_set(relpath: str, manifest: dict) -> bool:
    if not any(fnmatch.fnmatch(relpath, g) for g in manifest["include"]):
        return False
    return not any(fnmatch.fnmatch(relpath, g) for g in manifest["exclude"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="fail when the Player set is worse than the baseline")
    ap.add_argument("--write-baseline", action="store_true")
    ap.add_argument("--why", metavar="FILE",
                    help="print the include chain that contaminates FILE")
    ap.add_argument("--list", metavar="CATEGORY",
                    help="list contaminated Player-set files for one category")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    g = Graph()
    g.build()
    manifest = load_manifest()
    cats = list(FORBIDDEN)

    if args.why:
        want = args.why.replace("\\", "/").lower()
        hits = [p for p in g.files if rel(p).lower().endswith(want)]
        if not hits:
            print(f"no such file in the scanned tree: {args.why}")
            return 2
        for p in hits:
            print(rel(p))
            for cat in cats:
                ch = g.chain(p, cat)
                if ch:
                    ext = g.direct[ch[-1]][cat][0]
                    print(f"  {cat}: " + " -> ".join(rel(c) for c in ch) + f" -> <{ext}>")
            print()
        return 0

    sources = [p for p in g.files if p.lower().endswith(SOURCE_EXT)]
    reach = {p: g.reach(p) for p in sources}
    player = [p for p in sources if in_player_set(rel(p), manifest)]

    # 1) whole tree, per top-level src directory
    per_dir = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(int)
    for p in sources:
        parts = rel(p).split("/")
        d = "/".join(parts[:2]) if len(parts) > 2 else parts[0]
        totals[d] += 1
        for cat in reach[p]:
            per_dir[d][cat] += 1
    print(f"Translation units scanned: {len(sources)}   (first-party files: {len(g.files)})")
    print()
    print("Transitive reach per directory (TU count that can see the header):")
    head = f"  {'directory':<22}{'TUs':>5}" + "".join(f"{c:>12}" for c in cats)
    print(head)
    for d in sorted(totals, key=lambda k: -totals[k]):
        print(f"  {d:<22}{totals[d]:>5}" + "".join(f"{per_dir[d][c]:>12}" for c in cats))
    print()

    # 2) Player set
    counts = {c: sum(1 for p in player if c in reach[p]) for c in cats}
    clean = sum(1 for p in player if not reach[p])
    print(f"Player set (scripts/player_include_manifest.json): {len(player)} TUs, "
          f"{clean} clean")
    for c in cats:
        print(f"  {c:<12}{counts[c]:>5}")
    print()

    # 3) hubs: which first-party file is the entry point, by Player TUs it taints
    print(f"Entry points ranked by Player TUs they contaminate (top {args.top}):")
    for cat in cats:
        blame = defaultdict(int)
        for p in player:
            ch = g.chain(p, cat) if cat in reach[p] else None
            if ch:
                blame[ch[-1]] += 1
        if not blame:
            continue
        print(f"  [{cat}]")
        for f, n in sorted(blame.items(), key=lambda kv: -kv[1])[:args.top]:
            ext = ", ".join(sorted(set(g.direct[f][cat]))[:3])
            print(f"    {n:>5}  {rel(f)}   <{ext}>")
    print()
    print("Note: each TU is blamed on the NEAREST entry point only. Fixing the top "
          "hub exposes the next one; rerun after every header change.")

    if args.list:
        print()
        for p in sorted(player):
            if args.list in reach[p]:
                print("  " + rel(p))

    if args.write_baseline:
        manifest["baseline"] = counts
        with open(MANIFEST, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(manifest, fh, indent=2)
            fh.write("\n")
        print("\nbaseline written:", counts)
        return 0

    if args.check:
        base = manifest.get("baseline") or {}
        worse = [(c, base.get(c), counts[c]) for c in cats
                 if c in base and counts[c] > base[c]]
        if not base:
            print("\nFAIL: no baseline recorded (run --write-baseline).")
            return 1
        if worse:
            print("\nFAIL: Player include surface grew:")
            for c, b, n in worse:
                print(f"  {c}: {b} -> {n}")
            return 1
        better = [(c, base[c], counts[c]) for c in cats if c in base and counts[c] < base[c]]
        print("\nPASS: no category above baseline.")
        for c, b, n in better:
            print(f"  improved {c}: {b} -> {n}  (run --write-baseline to lock it in)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
