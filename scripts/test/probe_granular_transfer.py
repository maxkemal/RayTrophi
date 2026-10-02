"""Sample a fluid domain's per-step host<->device traffic while the sim plays.

Usage: python scripts/test/probe_granular_transfer.py ["Physics Domain 1"] [seconds]

Prints one row per distinct step (upload/download MB, transfer calls, p2g/g2p ms,
granular substeps). Start timeline playback in the app first; the reader does
not drive the frame loop, it only watches it.

What it answers: whether the granular grid velocity stays on the device across
elastic substeps. Before the resident chain, traffic scaled with grid cells x
substeps (a 114^3 domain: ~41 MB up + ~18 MB down PER SUBSTEP). After it,
per-substep traffic should be ~one cell-sized field up (the fluid mask) and
particle-sized buffers, with the 3 face fields coming down once per frame.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rt_ipc import RtIpc, RtIpcError  # noqa: E402

MB = 1024.0 * 1024.0


def main() -> int:
    domain = sys.argv[1] if len(sys.argv) > 1 else "Physics Domain 1"
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    c = RtIpc()
    info = None
    for _ in range(20):  # a long sim frame starves main-thread dispatch
        try:
            info = c.call("fluid.get", domain=domain)
            break
        except RtIpcError as e:
            if "timeout" not in str(e):
                raise
    if info is None:
        print("main thread never answered - a frame is taking longer than the IPC timeout")
        return 2
    dims = info.get("uvw_dim") or [0, 0, 0]
    cells = dims[0] * dims[1] * dims[2]
    field_mb = cells * 4 / MB
    print(f"domain={domain!r} particles={info.get('particle_count')} "
          f"cells={cells} (one cell field = {field_mb:.2f} MB)")
    print(f"{'up MB':>9} {'down MB':>9} {'up#':>5} {'dn#':>5} {'p2g ms':>8} "
          f"{'g2p ms':>8} {'substeps':>8} {'up/substep MB':>14}")
    last = None
    rows = []
    timeouts = 0
    end = time.time() + seconds
    while time.time() < end:
        try:
            s = c.call("fluid.step_stats", domain=domain)
            d = c.call("fluid.get", domain=domain)
        except RtIpcError as e:
            if "timeout" not in str(e):
                raise
            timeouts += 1  # the frame outlived the dispatch timeout: itself a stall sample
            continue
        key = (s.get("upload_bytes"), s.get("download_bytes"), s.get("p2g_ms"))
        if s.get("measured") and key != last:
            last = key
            sub = max(1, int(d.get("granular_solver_substeps") or 1))
            up = s["upload_bytes"] / MB
            dn = s["download_bytes"] / MB
            rows.append((up, dn, sub))
            print(f"{up:9.1f} {dn:9.1f} {s['upload_calls']:5d} {s['download_calls']:5d} "
                  f"{s['p2g_ms']:8.1f} {s['g2p_ms']:8.1f} {sub:8d} {up / sub:14.2f}")
        time.sleep(0.05)
    if not rows:
        print(f"ipc_timeouts={timeouts}")
        print("NO STEPS OBSERVED - is the timeline playing? (a paused sim reports "
              "the last step forever, which this script de-duplicates)")
        return 1
    worst = max(rows, key=lambda r: r[0])
    print(json.dumps({"steps": len(rows), "ipc_timeouts": timeouts, "max_up_mb": round(worst[0], 1),
                      "max_down_mb": round(worst[1], 1),
                      "substeps_at_max": worst[2]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
