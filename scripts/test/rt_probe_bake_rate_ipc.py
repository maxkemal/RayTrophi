"""Read-only: does the sky bake / probe field restart every frame?

Usage: python scripts\test\rt_probe_bake_rate_ipc.py [player|studio] [seconds]
Never changes the scene. Reads rayfusion.bake_counters, probe_field and
viewport.frame_telemetry twice and prints per-frame rates.
"""
import sys, time, json
sys.path.insert(0, __file__.rsplit("\\", 1)[0] if "\\" in __file__ else ".")
import rt_ipc

target = sys.argv[1] if len(sys.argv) > 1 else "player"
secs = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
c = rt_ipc.RtIpc(rt_ipc.PLAYER_PIPE_NAME if target == "player" else rt_ipc.PIPE_NAME)

def snap():
    t = c.call("viewport.frame_telemetry")
    return (c.call("rayfusion.bake_counters"), c.call("rayfusion.probe_field"), t)

a = snap(); time.sleep(secs); b = snap()
fa = a[2].get("frames_consumed", a[2].get("frames", 0)); fb = b[2].get("frames_consumed", b[2].get("frames", 0))
frames = max(1, fb - fa)
print("frames:", frames, "(telemetry keys:", sorted(b[2].keys())[:8], ")")
for k in b[0]:
    d = b[0][k] - a[0][k] if isinstance(b[0][k], int) else b[0][k]
    print(f"  {k:24s} total={b[0][k]:<8} delta={d:<6} per_frame={d/frames:.3f}")
p = b[1]
print("probe: producer=%s traced_requested=%s valid=%s/%s pending=%s bound=%s auto_fit=%s"
      % (p["producer"], p.get("producer_traced_requested"), p["valid"], p["total"],
         p["pending"], p["bound"], p["auto_fit_mode"]))
