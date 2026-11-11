"""IPC fuzz probe: a null / wrong-typed parameter must get an ERROR REPLY, never kill the app.

This build compiles nlohmann::json with JSON_NOEXCEPTION, so get<T>() on null or
on a wrong type, and .at() on a missing key, call abort(). dispatchMethod() now
checks every request against the generated descriptor table first
(RtIpcParamValidation.cpp). This script drives that check from the LIVE table:

  * every method whose descriptor says access == "read",
  * every parameter typed string / float / int / number / bool / vec3 / array,
  * two probes per parameter: null, and a value of the wrong type,
  * the method's other required parameters are filled with type-correct dummies,
    so a rejection must name the probed key, not a missing neighbour.

Verdicts per probe:
  REJECTED  - the param carries `validated_as` and the reply names the key (expected)
  ACCEPTED  - the param carries `validated_as` and the call SUCCEEDED (FAIL: check is dead)
  HANDLER   - the param is not pre-validated; the handler answered (ok, the app lives)
  CRASH     - the pipe broke: the app died on this probe (FAIL, run stops)

Read-only methods only, so the scene is not changed. Still refuses to run while
the open scene has unsaved changes (project.autosave_status scene_is_modified):
a CRASH would lose them. Refuses an old binary (no `validated_as` anywhere in
agent.describe): without the dispatcher check the first probe kills the app.

Run externally:  python scripts/test/rt_test_ipc_param_validation.py [--json OUT]
"""
import json
import sys

from rt_ipc import RtIpc, RtIpcError

PROBED_TYPES = {"string", "float", "int", "integer", "number", "uint", "bool",
                "boolean", "vec3", "float[3]", "array", "string[]", "int[]"}
# A method whose own job is to describe/drive the probe would only test itself.
SKIP = {"batch"}


def dummy(kind, ptype):
    """A type-correct value for a required neighbour parameter."""
    kind = kind or {"float": "number", "int": "number", "integer": "number",
                    "uint": "number", "boolean": "bool", "float[3]": "vec3",
                    "string[]": "string[]", "int[]": "number[]"}.get(ptype, ptype)
    return {"string": "__ipc_probe__", "number": 0, "bool": False,
            "vec3": [0, 0, 0], "string[]": [], "number[]": []}.get(kind)


def wrong_value(kind, ptype):
    """A value of a type the parameter does not take."""
    kind = kind or ptype
    if kind in ("string",):
        return 12345
    if kind in ("vec3", "float[3]"):
        return [0, "x", 0]
    if kind in ("string[]",):
        return [7]
    if kind in ("number[]", "int[]", "array"):
        return ["x"]
    return "not-a-%s" % kind


def main(argv):
    out_path = argv[argv.index("--json") + 1] if "--json" in argv else None
    client = RtIpc()

    status = client.call("project.autosave_status")
    if status.get("scene_is_modified"):
        print("REFUSED: the open scene has unsaved changes (scene_is_modified). "
              "Save it first - a crashing probe would lose them.")
        return 2

    methods = client.call("agent.list_methods")["methods"]
    described = {}
    for row in methods:
        name = row["method"]
        if row.get("access") != "read" or name in SKIP or not row.get("param_count"):
            continue
        described[name] = client.call("agent.describe", method=name)

    if not any("validated_as" in p for d in described.values()
               for p in (d.get("params") or {}).values()):
        print("REFUSED: no parameter carries `validated_as` - this exe predates the "
              "pre-dispatch check (RtIpcParamValidation.cpp). Build first; on this "
              "binary the first probe would abort the app.")
        return 2

    results, covered, failures = [], [], []
    for name in sorted(described):
        params = described[name].get("params") or {}
        probed_any = False
        for key, spec in params.items():
            ptype = spec.get("type", "any")
            kind = spec.get("validated_as")
            if ptype not in PROBED_TYPES and not kind:
                continue
            base = {}
            for other, ospec in params.items():
                if other == key:
                    continue
                if ospec.get("required") or ospec.get("rejected_if_missing"):
                    value = dummy(ospec.get("validated_as"), ospec.get("type"))
                    if value is not None:
                        base[other] = value
            for label, value in (("null", None), ("wrong", wrong_value(kind, ptype))):
                request = dict(base)
                request[key] = value
                probed_any = True
                try:
                    client.call(name, **request)
                    verdict = "ACCEPTED" if kind else "HANDLER"
                    message = "ok"
                except RtIpcError as exc:
                    message = str(exc)
                    if kind:
                        verdict = "REJECTED" if key in message else "REJECTED_OTHER"
                    else:
                        verdict = "HANDLER"
                except Exception as exc:  # pipe broke: the app is gone
                    verdict, message = "CRASH", "%s: %s" % (type(exc).__name__, exc)
                row = {"method": name, "param": key, "type": ptype, "validated_as": kind,
                       "probe": label, "verdict": verdict, "message": message[:200]}
                results.append(row)
                if verdict in ("ACCEPTED", "CRASH"):
                    failures.append(row)
                if verdict == "CRASH":
                    break
            if failures and failures[-1]["verdict"] == "CRASH":
                break
        if probed_any:
            covered.append(name)
        if failures and failures[-1]["verdict"] == "CRASH":
            break

    counts = {}
    for row in results:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    print("probed %d read-only methods, %d probes: %s"
          % (len(covered), len(results),
             ", ".join("%s=%d" % kv for kv in sorted(counts.items()))))
    print("covered methods:")
    for name in covered:
        print("  " + name)
    for row in failures:
        print("FAIL %(verdict)s %(method)s %(param)s=%(probe)s (%(type)s/%(validated_as)s): "
              "%(message)s" % row)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump({"covered": covered, "counts": counts, "results": results},
                      handle, indent=1)
    if failures:
        return 1
    # The app must still answer after the whole run.
    client.call("project.autosave_status")
    print("PASS - app alive, every pre-validated probe answered with an error")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
