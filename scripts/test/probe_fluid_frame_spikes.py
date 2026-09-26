"""Catch slow frames while a fluid sim plays and say where the time went.

Run from a terminal (NOT the app's script workspace) with the app open, then
press play in the app:

    python scripts/test/probe_fluid_frame_spikes.py [seconds] [threshold_ms]

Every poll it diffs perf.list. When loop.frame grew by more than the threshold
since the last poll, it prints each scope's share of that interval plus the
domain's last fluid.step_stats. Time in loop.frame that no child scope claims
is printed as "untimed" - a large untimed share means the stall lives in code
that has no RTPERF scope yet, which is itself the finding.

An idle app parks in SDL's event wait INSIDE loop.frame (~500 ms, no child
scope moves). Those intervals are counted as idle, not reported as spikes.
"""
import sys
import time

import rt_ipc

DURATION_S = float(sys.argv[1]) if len(sys.argv) > 1 else 180.0
THRESHOLD_MS = float(sys.argv[2]) if len(sys.argv) > 2 else 250.0
POLL_S = 0.1

# Scopes nested inside another listed scope; excluded from the untimed sum so
# they are not counted twice.
NESTED_PREFIXES = ("sim.", "ui.")


def snap(c):
    return {e["name"]: (e["count"], e["total_ms"], e["max_ms"])
            for e in c.call("perf.list")}


def main():
    c = rt_ipc.RtIpc()
    domains = c.call("fluid.list_domains")["domains"]
    domain = domains[0]["name"] if domains else None
    prev = snap(c)
    t_end = time.time() + DURATION_S
    spikes = 0
    frames = 0
    idle = 0
    while time.time() < t_end:
        time.sleep(POLL_S)
        cur = snap(c)
        delta = {}
        for k, (n, tot, _mx) in cur.items():
            pn, pt, _ = prev.get(k, (0, 0.0, 0.0))
            if n > pn:
                delta[k] = (n - pn, tot - pt)
        prev = cur
        lf = delta.get("loop.frame")
        if not lf:
            continue
        frames += lf[0]
        if lf[1] < THRESHOLD_MS:
            continue
        if not any(t >= 1.0 for k, (n, t) in delta.items() if k != "loop.frame"):
            idle += 1
            continue
        spikes += 1
        loop_children = sum(t for k, (n, t) in delta.items()
                            if k.startswith("loop.") and k != "loop.frame")
        print("\n=== spike %d: loop.frame %d frame(s) %.0f ms (untimed %.0f ms)"
              % (spikes, lf[0], lf[1], lf[1] - loop_children))
        for k, (n, t) in sorted(delta.items(), key=lambda kv: -kv[1][1]):
            if t >= 1.0:
                print("  %-40s %3dx %8.1f ms" % (k, n, t))
        if domain and any(k.startswith("sim.") for k in delta):
            ok, st = c.try_call("fluid.step_stats", domain=domain)
            if ok:
                keys = ("particle_count", "p2g_ms", "pressure_ms", "g2p_ms",
                        "advect_ms", "density_ms", "synchronize_ms",
                        "upload_bytes", "download_bytes")
                print("  step_stats: " + " ".join(
                    "%s=%s" % (k, st.get(k)) for k in keys))
    print("\n%d frame(s) observed, %d spike interval(s) over %.0f ms, "
          "%d idle-wait interval(s) ignored"
          % (frames, spikes, THRESHOLD_MS, idle))


if __name__ == "__main__":
    main()
