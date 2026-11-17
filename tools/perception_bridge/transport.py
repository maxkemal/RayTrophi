"""Reuse the repository's external named-pipe client, imported only when live."""
import sys
import time
from pathlib import Path

PIPE = r"\\.\pipe\RayTrophiPlayer"


def connect(pipe=PIPE):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "test"))
    from rt_ipc import RtIpc
    return RtIpc(pipe_name=pipe)


def summary_ms(samples):
    ordered = sorted(samples)
    if not ordered:
        return {"count": 0, "p50_ms": None, "p95_ms": None, "max_ms": None}
    return {"count": len(ordered), "p50_ms": ordered[len(ordered)//2],
            "p95_ms": ordered[min(len(ordered)-1, int(len(ordered)*0.95))],
            "max_ms": ordered[-1]}


def wait_frame(start, frame, hz=30):
    delay = start + frame / hz - time.perf_counter()
    if delay > 0:
        time.sleep(delay)
