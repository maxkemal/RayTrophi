#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Check whether the descriptor PROSE is telling the truth.

    python scripts/verify_descriptor_claims.py            # static pass
    python scripts/verify_descriptor_claims.py --live     # also query a running app
    python scripts/verify_descriptor_claims.py --accept   # re-baseline (deliberate)

★★★ Why this exists. `agent.discover` reports `documented_coverage: 1.0`, and
that number means "a summary was WRITTEN", not "the summary is TRUE". Every
method has prose; nothing has ever checked it. That is the same instrument that
reported 299 empty records as full coverage, moved up one floor: the metric
measures presence and gets read as correctness.

Prose cannot be verified in general. But most of what these notes actually claim
is mechanical, and mechanical claims can be checked:

  - A note that names `voxel_size` claims that parameter exists.
  - `verify_with` claims a method will CHECK the result. A method that writes
    cannot verify anything, so a write there is wrong by construction.
  - `invalidates` claims a named piece of state goes stale. Free text makes that
    unusable to a caller ("sim_cache" vs "simulation_cache" read differently and
    mean the same), so the vocabulary is closed.
  - A note naming a result field claims the call returns it. With --live that is
    checked against what the engine actually returns.

Anything left over is an UNGROUNDED claim: prose that names something this build
does not have. Those are held in a baseline file, so the count can only go down
without a deliberate --accept. A rename that silently invalidates a note is
exactly what this catches.
"""

from __future__ import division, print_function

import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import gen_ipc_descriptors as gen  # noqa: E402

BASELINE = os.path.join(HERE, "descriptor_claim_baseline.json")

# State names `invalidates` may use. Closed on purpose: a caller has to be able
# to match these against something, and free text cannot be matched.
KNOWN_STATE = {
    "simulation_cache",   # baked sim frames for a domain
    "render_accumulation",  # the progressive sample buffer
    "tlas",               # GPU acceleration structure
    "viewport_frame",     # the captured frame used by render.probe
    "geometry_cache",     # derived meshes (subdiv, scatter, fracture shards)
    "material_cache",     # GPU-side material table
    "node_graph_eval",    # cooked node graph output
    "undo_history",
}

# Prose words that look like identifiers but are English, not API surface.
PROSE_WORDS = {
    "read_only", "fail_closed", "black_band", "did_you_mean", "no_op",
    "world_space", "local_space", "per_frame", "per_object", "per_vertex",
    "up_to", "left_to_right", "front_to_back", "opt_in", "round_trip",
    "case_insensitive", "one_shot", "in_place", "out_of_range", "as_is",
}

TOKEN_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
METHOD_RE = re.compile(r"\b[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*\b")


def load_baseline():
    if not os.path.exists(BASELINE):
        return {}
    with io.open(BASELINE, encoding="utf-8") as handle:
        return json.load(handle).get("ungrounded", {})


def records():
    overlay = json.loads(gen.io.open(gen.OVERLAY, encoding="utf-8").read())
    overlay.pop("$comment", None)
    namespaces = gen.namespace_table(gen.io.open(gen.SECURITY, encoding="utf-8").read())
    return gen.build(overlay, namespaces)


def param_names(record):
    """Parameter names AND their enum values.

    ★ An enum value is schema, not prose: once `param` declares
    solid|material|rendered, a note naming `rendered` is grounded. Counting
    only the parameter name would keep reporting documented behaviour as
    unverified, and a checker that cries wolf gets baselined into silence.
    """
    names = set()
    for row in record["params"]:
        names.add(row["name"])
        for value in (row.get("enum") or "").split("|"):
            if value:
                names.add(value)
    return names


def linked(record, *fields):
    out = []
    for field in fields:
        out.extend(t for t in (record[field] or "").split("|") if t)
    return out


def check(live_keys=None):
    """Return (hard_errors, ungrounded, claim_total, claim_grounded)."""
    live_keys = live_keys or {}
    recs = records()
    by_name = {r["method"]: r for r in recs}
    all_methods = set(by_name)
    domains = {m.split('.')[0] for m in all_methods}

    hard = []
    ungrounded = {}
    total = grounded = 0

    for record in recs:
        method = record["method"]
        mine = param_names(record)

        # -- structural claims ------------------------------------------------
        for target in linked(record, "verify_with"):
            other = by_name.get(target)
            if other and other["access"] not in ("read", "render"):
                hard.append((method, "verify_with -> %s is a %s method; a call "
                                     "that mutates cannot verify anything"
                             % (target, other["access"])))
        for field in ("prerequisites", "next_steps", "verify_with"):
            if method in linked(record, field):
                hard.append((method, "%s names itself" % field))
        for state in linked(record, "invalidates"):
            if state not in KNOWN_STATE:
                hard.append((method, "invalidates '%s' is not in the known state "
                                     "vocabulary (%s)"
                             % (state, ", ".join(sorted(KNOWN_STATE)))))

        # -- prose claims -----------------------------------------------------
        prose = " ".join(filter(None, [record["summary"], record["notes"]]))
        if not prose:
            continue

        # Method names in prose must exist. Same rule as the link check, but
        # prose is where a rename hides longest.
        for named in METHOD_RE.findall(prose):
            if named.split(".")[0] in {m.split(".")[0] for m in all_methods}:
                total += 1
                if named in all_methods:
                    grounded += 1
                else:
                    ungrounded.setdefault(method, []).append(named)

        # ★ Strip the method references first. "gas.set_shader" is one claim,
        # already counted above; leaving it in makes the tokenizer see a second,
        # bogus claim called `set_shader` and blame the wrong thing.
        prose = METHOD_RE.sub(" ", prose)

        # Identifier-shaped tokens claim a parameter or a result field.
        # A note may legitimately name a field of a method it points AT
        # ("check frame_available with viewport.status"), so linked methods
        # contribute both their parameters and their result keys.
        reachable = set(mine) | set(live_keys.get(method, ()))
        for target in linked(record, "related", "prerequisites",
                             "next_steps", "verify_with"):
            other = by_name.get(target)
            if other:
                reachable |= param_names(other)
                reachable |= set(live_keys.get(target, ()))
        # Domain names are API surface too: "flow_source" is a namespace, not a
        # typo, even though it never appears as a parameter.
        reachable |= domains

        for token in set(TOKEN_RE.findall(prose)):
            if token in PROSE_WORDS or token in KNOWN_STATE:
                continue
            total += 1
            if token in reachable:
                grounded += 1
            else:
                ungrounded.setdefault(method, []).append(token)

    for method in ungrounded:
        ungrounded[method] = sorted(set(ungrounded[method]))
    return hard, ungrounded, total, grounded


def harvest_live():
    """Ask a running app what read methods actually return.

    ★ Only methods with no required parameters are called, and only read ones.
    The point is to learn the RESULT KEYS, so prose naming a field that the
    engine does not return stops being unverifiable.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "RayTrophiAgent"))
    from core.ipc_client import IPCClient

    client = IPCClient(agent_id="claim_verifier", label="claims")
    client.connect()
    keys, called, failed = {}, 0, 0
    # ★ Not by access class: viewport.status is classified Render because the
    # whole viewport.* namespace drives the engine, yet it only reports. Calling
    # by NAME shape keeps the harvest read-only without trusting the class.
    def reports_only(name):
        leaf = name.split(".")[-1]
        return (leaf in ("status", "list", "shading", "probe", "discover",
                         "roles", "version", "summary", "capabilities")
                or leaf.startswith("get")
                or leaf.startswith("list_")
                or leaf.endswith("_status"))

    for record in records():
        if not (reports_only(record["method"]) or record["access"] == "read"):
            continue
        if any(row["required"] for row in record["params"]):
            continue
        response = client.call(record["method"], {})
        called += 1
        result = response.get("result")
        if "error" in response:
            failed += 1
            continue
        if isinstance(result, dict):
            found = set(result)
            # One level down: most read methods answer with a list of records,
            # and the prose names the fields of those records.
            for value in result.values():
                if isinstance(value, dict):
                    found |= set(value)
                elif isinstance(value, list) and value and isinstance(value[0], dict):
                    found |= set(value[0])
            keys[record["method"]] = found
    client.close()
    print("live: called %d parameterless read methods, %d returned an error, "
          "%d contributed result keys" % (called, failed, len(keys)))
    return keys


def main(argv):
    live_keys = harvest_live() if "--live" in argv else {}
    hard, ungrounded, total, grounded = check(live_keys)

    print("claims checked: %d, grounded: %d (%.1f%%)"
          % (total, grounded, 100.0 * grounded / max(1, total)))

    if "--accept" in argv:
        with io.open(BASELINE, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(
                {"$comment": "Ungrounded prose claims accepted as of the last "
                             "--accept run. This file exists so the count can "
                             "only go DOWN; a new entry means a note now names "
                             "something the build does not have.",
                 "ungrounded": ungrounded},
                indent=2, ensure_ascii=False, sort_keys=True) + "\n")
        print("baseline written: %d methods carry ungrounded claims"
              % len(ungrounded))
        return 0

    if not live_keys:
        # ★ Without --live, a note naming a RESULT field cannot be grounded -
        # nothing here knows what the engine returns. Reporting those as new
        # failures would train the reader to ignore this script, so the static
        # pass checks only what static analysis can actually decide.
        print("static pass: result-field claims are NOT checked here; "
              "run with --live against a running app for the full number.")
        if hard:
            print("\nWRONG BY CONSTRUCTION (%d):" % len(hard))
            for method, message in hard:
                print("  %-34s %s" % (method, message))
            return 1
        print("OK - no wrong-by-construction claims (structural checks only).")
        return 0

    baseline = load_baseline()
    fresh = {}
    for method, tokens in ungrounded.items():
        new = [t for t in tokens if t not in baseline.get(method, [])]
        if new:
            fresh[method] = new

    status = 0
    if hard:
        print("\nWRONG BY CONSTRUCTION (%d):" % len(hard))
        for method, message in hard:
            print("  %-34s %s" % (method, message))
        status = 1
    if fresh:
        print("\nNEW UNGROUNDED CLAIMS (%d methods) - prose names something this "
              "build does not have:" % len(fresh))
        for method in sorted(fresh):
            print("  %-34s %s" % (method, ", ".join(fresh[method])))
        print("\nEither fix the note, or re-baseline deliberately with --accept.")
        status = 1
    if not status:
        print("OK - no wrong-by-construction claims, no new ungrounded prose.")
    print("(%d methods carry ungrounded claims in total; baseline holds %d)"
          % (len(ungrounded), len(baseline)))
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
