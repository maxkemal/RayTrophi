# -*- coding: utf-8 -*-
"""Measure where a small local model falls apart on a LONG task.

    python RayTrophiAgent/measure_horizon.py      # needs Studio running + Ollama


Not "can it call a tool" - that was measured and it can. The question small
models actually fail is the HORIZON: 15 steps in, do they still remember the
plan, still read results, still stop when done.

Everything is logged per step so the failure MODE is identifiable, not just the
pass/fail: repeats, hallucinated methods, unread errors, and the ending.
"""
import json
import os
import sys
import time
import collections

AGENT = r"e:\RayTrophi_projesi\raytracing_Proje_Moduler\RayTrophiAgent"
sys.path.insert(0, AGENT)
os.chdir(AGENT)

from core.ipc_client import IPCClient          # noqa: E402
from core.registry import CapabilityRegistry   # noqa: E402
from core.tool_executor import ToolExecutor    # noqa: E402
from core.orchestrator import Orchestrator     # noqa: E402
from providers.local_provider import LocalLLMProvider  # noqa: E402

TASK = (
    "Set up a small test scene and verify it, step by step:\n"
    "1. Add a cube named AgentBox at the world origin.\n"
    "2. Move it to position (0, 1, 0).\n"
    "3. Add a directional light.\n"
    "4. Set the world/sky to the nishita atmosphere model.\n"
    "5. Set AgentBox's material base_color to red.\n"
    "6. Enable viewport frame capture, render some frames, and probe the frame.\n"
    "7. Report the mean luminance you measured, and list every object now in "
    "the scene.\n"
    "Verify each step actually landed before moving to the next one. "
    "When all seven are done, stop and give a short report."
)

log = []
steps = []


def report(kind, text):
    log.append((time.time(), kind, text))


class Watched(ToolExecutor):
    """Same executor the panel uses, with a tape recorder attached."""

    def execute(self, name, args):
        started = time.time()
        result = ToolExecutor.execute(self, name, args)
        ok = not (isinstance(result, dict) and result.get("ok") is False)
        steps.append({
            "n": len(steps) + 1,
            "tool": name,
            "args": json.dumps(args, default=str)[:200],
            "ok": ok,
            "error": (result.get("error") if isinstance(result, dict) else None),
            "secs": round(time.time() - started, 1),
        })
        print("  %2d. %-20s %-4s %5.1fs %s"
              % (len(steps), name, "ok" if ok else "FAIL",
                 time.time() - started,
                 json.dumps(args, default=str)[:110]), flush=True)
        return result


def main():
    ipc_tools = IPCClient(agent_id="horizon_probe", label="tools")
    ipc_tools.connect()
    registry = CapabilityRegistry()
    registry.bootstrap(ipc_tools)

    before = ipc_tools.call("scene.list_objects").get("result")
    print("scene before:", before, flush=True)

    provider = LocalLLMProvider(base_url="http://localhost:11434/v1",
                                model="qwen3:8b")
    executor = Watched(ipc_tools, registry, report=report)
    orchestrator = Orchestrator(provider, executor, report=report)

    print("\n--- turn log ---", flush=True)
    started = time.time()
    try:
        answer = orchestrator.handle_user_message(TASK)
    except Exception as exc:                       # noqa: BLE001
        answer = "EXCEPTION: %r" % (exc,)
    elapsed = time.time() - started

    # ★★★ The scene BEFORE and AFTER, not just the call results. On 2026-08-19
    # every call in a run reported success and the object was not in the scene
    # at the end (docs/dev/BUG_DELETED_NAME_REUSE_GHOST.md). A harness that
    # trusts return values would have scored that run 9/9.
    after = ipc_tools.call("scene.list_objects").get("result")

    calls = collections.Counter(s["tool"] for s in steps)
    signatures = collections.Counter((s["tool"], s["args"]) for s in steps)
    repeats = {k: v for k, v in signatures.items() if v > 1}
    failures = [s for s in steps if not s["ok"]]
    unknown = [s for s in failures
               if s["error"] and "no method named" in str(s["error"])]

    out = {
        "task_seconds": round(elapsed, 1),
        "tool_calls": len(steps),
        "by_tool": dict(calls),
        "failures": len(failures),
        "hallucinated_methods": [json.loads(s["args"]).get("method")
                                 for s in unknown],
        "identical_repeats": [{"tool": k[0], "args": k[1], "times": v}
                              for k, v in repeats.items()],
        "scene_before": before,
        "scene_after": after,
        "metrics": orchestrator.metrics,
        "final_answer": answer,
        "steps": steps,
    }
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "horizon_result.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2, ensure_ascii=False)

    print("\n--- summary ---")
    print("elapsed        : %.0f s" % elapsed)
    print("tool calls     : %d  %s" % (len(steps), dict(calls)))
    print("failures       : %d" % len(failures))
    print("hallucinated   : %s" % (out["hallucinated_methods"] or "none"))
    print("exact repeats  : %d" % len(repeats))
    print("scene after    : %s" % after)
    print("metrics        : %s" % orchestrator.metrics)
    print("\nfinal answer:\n%s" % (answer or "(empty)"))
    ipc_tools.close()


if __name__ == "__main__":
    main()
